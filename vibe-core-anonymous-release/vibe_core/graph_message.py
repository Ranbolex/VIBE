import torch as th
import torch.nn as nn


class GraphMessage(nn.Module):
    """Aggregate sparse graph payloads without silently changing information scope.

    ``oracle_sender_hidden`` is retained only as an explicit diagnostic upper
    bound.  The deployable path consumes receiver-local pairwise payloads with
    shape ``[B, receiver, sender, local_dim]``.
    """

    def __init__(self, hidden_dim, message_dim, local_dim=0):
        super().__init__()
        self.message_dim = message_dim
        self.local_dim = int(local_dim)
        # Keep this module name and shape checkpoint-compatible with the legacy
        # sender-hidden diagnostic path.
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, message_dim),
            nn.ReLU(inplace=True),
        )
        self.local_encoder = (
            nn.Sequential(
                nn.Linear(self.local_dim, message_dim),
                nn.ReLU(inplace=True),
            )
            if self.local_dim > 0 else None
        )

    def forward(self, edge_weights, payload, payload_mode="oracle_sender_hidden"):
        if edge_weights.dim() != 3 or edge_weights.size(-1) != edge_weights.size(-2):
            raise ValueError("edge_weights must have shape [B,N,N]")
        if payload_mode == "oracle_sender_hidden":
            if payload.dim() != 3 or payload.shape[:2] != edge_weights.shape[:2]:
                raise ValueError(
                    "oracle sender payload must have shape [B,N,H]"
                )
            encoded = self.encoder(payload).unsqueeze(1).expand(
                -1, edge_weights.size(1), -1, -1
            )
        elif payload_mode == "receiver_local_ally":
            if self.local_encoder is None:
                raise RuntimeError(
                    "receiver_local_ally requires GraphMessage(local_dim > 0)"
                )
            expected_prefix = edge_weights.shape + (self.local_dim,)
            if payload.shape != expected_prefix:
                raise ValueError(
                    "receiver-local payload must have shape [B,N,N,local_dim]"
                )
            encoded = self.local_encoder(payload)
        else:
            raise ValueError(
                "payload_mode must be 'receiver_local_ally' or "
                "'oracle_sender_hidden'"
            )
        normalizer = edge_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return th.einsum("bij,bijd->bid", edge_weights, encoded) / normalizer

    def load_compatible_state_dict(self, state_dict):
        """Load old checkpoints while strictly auditing the new local branch."""
        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing = set()
        if self.local_encoder is not None:
            allowed_missing = {
                "local_encoder.0.weight",
                "local_encoder.0.bias",
            }
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "incompatible GraphMessage checkpoint: missing={} unexpected={}".format(
                    sorted(unexpected_missing), sorted(incompatible.unexpected_keys)
                )
            )
        return incompatible


def sparse_topk_weights(logits, top_k, straight_through=True, min_probability=0.5):
    """Return a row-wise sparse mask with no self edges."""
    n_agents = logits.size(-1)
    k = min(max(int(top_k), 0), max(n_agents - 1, 0))
    probs = th.sigmoid(logits)
    diagonal = th.eye(n_agents, device=logits.device, dtype=th.bool).unsqueeze(0)
    probs = probs.masked_fill(diagonal, 0.0)
    if k == 0:
        return th.zeros_like(probs)

    indices = logits.topk(k, dim=-1).indices
    selected_probs = probs.gather(-1, indices)
    selected = (selected_probs >= float(min_probability)).to(probs.dtype)
    hard = th.zeros_like(probs).scatter_(-1, indices, selected)
    hard = hard.masked_fill(diagonal, 0.0)
    if straight_through:
        return hard + probs - probs.detach()
    return hard


def random_topk_weights(reference, top_k, generator=None):
    """Sample a uniform row-wise K-edge graph with no self edges."""
    if reference.dim() != 3 or reference.size(-1) != reference.size(-2):
        raise ValueError("reference must have shape [B,N,N]")
    n_agents = reference.size(-1)
    k = min(max(int(top_k), 0), max(n_agents - 1, 0))
    if k == 0:
        return th.zeros_like(reference)
    diagonal = th.eye(n_agents, device=reference.device, dtype=th.bool).unsqueeze(0)
    random_scores = th.rand(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    ).masked_fill(diagonal, -1.0)
    indices = random_scores.topk(k, dim=-1).indices
    weights = th.zeros_like(reference).scatter_(-1, indices, 1.0)
    return weights.masked_fill(diagonal, 0.0)


def random_matched_weights(reference, template, generator=None):
    """Sample a random graph with the template's exact per-row edge counts."""
    if reference.shape != template.shape or reference.dim() != 3:
        raise ValueError("reference and template must share shape [B,N,N]")
    n_agents = reference.size(-1)
    diagonal = th.eye(
        n_agents, device=reference.device, dtype=th.bool
    ).unsqueeze(0)
    counts = template.gt(0.5).masked_fill(diagonal, False).sum(
        dim=-1, keepdim=True
    )
    random_scores = th.rand(
        reference.shape, device=reference.device, dtype=reference.dtype,
        generator=generator,
    ).masked_fill(diagonal, -1.0)
    ranks = random_scores.argsort(dim=-1, descending=True).argsort(dim=-1)
    weights = ranks.lt(counts).to(reference.dtype)
    return weights.masked_fill(diagonal, 0.0)


def anti_topk_weights(logits, top_k):
    """Select the lowest-scored non-self edges for intervention tests."""
    if logits.dim() != 3 or logits.size(-1) != logits.size(-2):
        raise ValueError("logits must have shape [B,N,N]")
    n_agents = logits.size(-1)
    k = min(max(int(top_k), 0), max(n_agents - 1, 0))
    if k == 0:
        return th.zeros_like(logits)
    diagonal = th.eye(n_agents, device=logits.device, dtype=th.bool).unsqueeze(0)
    scores = logits.masked_fill(diagonal, float("inf"))
    indices = scores.topk(k, dim=-1, largest=False).indices
    weights = th.zeros_like(logits).scatter_(-1, indices, 1.0)
    return weights.masked_fill(diagonal, 0.0)


def dense_edge_weights(reference):
    """Return the complete directed graph without self edges."""
    if reference.dim() != 3 or reference.size(-1) != reference.size(-2):
        raise ValueError("reference must have shape [B,N,N]")
    n_agents = reference.size(-1)
    diagonal = th.eye(n_agents, device=reference.device, dtype=th.bool).unsqueeze(0)
    return th.ones_like(reference).masked_fill(diagonal, 0.0)
