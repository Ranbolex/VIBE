import torch as th
import torch.nn as nn


class EdgeStudent(nn.Module):
    """Decentralized receiver-side edge predictor.

    Each logit depends on the receiver's local recurrent state and the two agent
    identities. It never consumes a sender hidden state or centralized state.
    Sender features are only used after selection as an explicit graph message.
    """

    def __init__(
            self, hidden_dim, n_agents, embed_dim=64, belief_feature_dim=0,
            receiver_context_dim=0, sender_context_dim=0):
        super().__init__()
        self.n_agents = n_agents
        self.belief_feature_dim = int(belief_feature_dim)
        self.receiver_context_dim = int(receiver_context_dim)
        self.sender_context_dim = int(sender_context_dim)
        input_dim = (
            hidden_dim + self.receiver_context_dim + 2 * n_agents
            + self.belief_feature_dim + self.sender_context_dim
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(
            self, receiver_hidden, pairwise_belief=None, receiver_context=None,
            sender_context=None):
        if receiver_hidden.dim() != 3:
            raise ValueError("receiver_hidden must have shape [B, N, H]")
        batch_size, n_agents, _ = receiver_hidden.shape
        if n_agents != self.n_agents:
            raise ValueError("agent dimension does not match EdgeStudent")

        device = receiver_hidden.device
        identities = th.eye(n_agents, device=device, dtype=receiver_hidden.dtype)
        receiver_h = receiver_hidden.unsqueeze(2).expand(-1, -1, n_agents, -1)
        receiver_ids = identities.view(1, n_agents, 1, n_agents).expand(batch_size, -1, n_agents, -1)
        sender_ids = identities.view(1, 1, n_agents, n_agents).expand(batch_size, n_agents, -1, -1)
        inputs = [receiver_h]
        if self.receiver_context_dim:
            expected = (batch_size, n_agents, self.receiver_context_dim)
            if receiver_context is None or receiver_context.shape != expected:
                raise ValueError(
                    "receiver_context must have shape [B,N,receiver_context_dim]"
                )
            inputs.append(
                receiver_context.unsqueeze(2).expand(-1, -1, n_agents, -1)
            )
        inputs.extend([receiver_ids, sender_ids])
        if self.sender_context_dim:
            expected = (
                batch_size, n_agents, n_agents, self.sender_context_dim
            )
            if sender_context is None or sender_context.shape != expected:
                raise ValueError(
                    "sender_context must have shape [B,N,N,sender_context_dim]"
                )
            inputs.append(sender_context)
        if self.belief_feature_dim:
            expected = (batch_size, n_agents, n_agents, self.belief_feature_dim)
            if pairwise_belief is None or pairwise_belief.shape != expected:
                raise ValueError(
                    "pairwise_belief must have shape [B,N,N,belief_feature_dim]"
                )
            inputs.append(pairwise_belief)
        pair_inputs = th.cat(inputs, dim=-1)
        logits = self.network(pair_inputs).squeeze(-1)

        diagonal = th.eye(n_agents, device=device, dtype=th.bool).unsqueeze(0)
        return logits.masked_fill(diagonal, -1e9)
