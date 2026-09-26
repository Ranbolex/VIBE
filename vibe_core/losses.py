import torch as th
import torch.nn.functional as F


def edge_distillation_loss(logits, labels, transition_mask=None, positive_weight=None):
    """BCE distillation on directed non-self edges.

    ``positive_weight="auto"`` balances the valid teacher labels in the
    current batch.  This avoids an all-negative student when a top-K teacher
    supplies only a small number of positive edges.
    """
    if logits.shape != labels.shape:
        raise ValueError("edge logits and labels must have identical shapes")
    n_agents = logits.size(-1)
    valid = ~th.eye(n_agents, device=logits.device, dtype=th.bool)
    valid = valid.view(*([1] * (logits.dim() - 2)), n_agents, n_agents)
    valid = valid.expand_as(logits)
    if transition_mask is not None:
        mask = transition_mask.bool()
        while mask.dim() < logits.dim():
            mask = mask.unsqueeze(-1)
        valid = valid & mask.expand_as(logits)

    selected_logits = logits[valid]
    selected_labels = labels.to(logits.dtype)[valid]
    if selected_logits.numel() == 0:
        return logits.sum() * 0.0
    pos_weight = None
    if isinstance(positive_weight, str):
        if positive_weight.lower() != "auto":
            raise ValueError("positive_weight must be numeric, None, or 'auto'")
        positive_count = selected_labels.sum()
        negative_count = selected_labels.numel() - positive_count
        pos_weight = th.where(
            positive_count > 0,
            negative_count / positive_count.clamp_min(1.0),
            th.ones_like(positive_count),
        ).detach()
    elif positive_weight is not None:
        pos_weight = th.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(selected_logits, selected_labels, pos_weight=pos_weight)


def edge_delta_rank_loss(
        logits, delta, transition_mask=None, min_advantage=1e-8,
        weight_power=1.0, max_weight=10.0):
    """Utility-aligned row-wise ranking loss for a top-1 Edge Student.

    Hard edge labels contain no sender-order information on rows where every
    Teacher score is zero (or effectively tied). This loss therefore trains
    only receiver rows whose best non-self edge has a measurable advantage
    over the other available senders. Informative rows are weighted by that
    advantage, normalized within the minibatch, so differently scaled Teacher
    regrets contribute comparably while larger utility gaps still matter more.
    """
    if logits.shape != delta.shape:
        raise ValueError("edge logits and Teacher delta must have identical shapes")
    if logits.dim() < 2 or logits.size(-1) != logits.size(-2):
        raise ValueError("edge logits and Teacher delta must end in [N,N]")

    n_agents = logits.size(-1)
    diagonal = th.eye(n_agents, device=logits.device, dtype=th.bool)
    diagonal = diagonal.view(*([1] * (logits.dim() - 2)), n_agents, n_agents)
    valid_edges = (~diagonal).expand_as(logits)
    masked_logits = logits.masked_fill(~valid_edges, -1e9)
    masked_delta = delta.detach().masked_fill(~valid_edges, float("-inf"))

    best_delta, best_sender = masked_delta.max(dim=-1)
    nonself_sum = delta.detach().masked_fill(~valid_edges, 0.0).sum(dim=-1)
    nonself_count = valid_edges.sum(dim=-1).clamp_min(1).to(delta.dtype)
    mean_delta = nonself_sum / nonself_count
    advantage = (best_delta - mean_delta).clamp_min(0.0)

    valid_rows = advantage > float(min_advantage)
    if transition_mask is not None:
        mask = transition_mask.bool()
        while mask.dim() < valid_rows.dim():
            mask = mask.unsqueeze(-1)
        valid_rows = valid_rows & mask.expand_as(valid_rows)
    if not valid_rows.any():
        return logits.sum() * 0.0

    row_loss = F.cross_entropy(
        masked_logits.reshape(-1, n_agents),
        best_sender.reshape(-1),
        reduction="none",
    ).reshape_as(best_sender)
    selected_advantage = advantage[valid_rows]
    power = float(weight_power)
    if power <= 0.0:
        raise ValueError("edge rank weight_power must be positive")
    raw_weights = selected_advantage.pow(power)
    weights = (
        raw_weights / raw_weights.mean().clamp_min(float(min_advantage) ** power)
    )
    if max_weight is not None and float(max_weight) > 0.0:
        weights = weights.clamp(max=float(max_weight))
    weights = weights.detach()
    return (row_loss[valid_rows] * weights).sum() / weights.sum().clamp_min(1.0)


def edge_delta_pairwise_rank_loss(
        logits, delta, transition_mask=None, min_advantage=1e-8,
        weight_power=1.0, max_weight=10.0, temperature=1.0):
    """Pairwise sender-order loss aligned with continuous Teacher utility.

    For every receiver row, each non-self sender pair contributes a logistic
    preference loss whenever the Teacher deltas differ by more than
    ``min_advantage``.  The larger-delta sender is preferred, so the loss
    trains the complete row ordering instead of only the Teacher top-1 edge.
    Tied and transition-masked rows/pairs contribute exactly zero.  Delta is
    detached because it is a frozen Teacher target.
    """
    if logits.shape != delta.shape:
        raise ValueError("edge logits and Teacher delta must have identical shapes")
    if logits.dim() < 2 or logits.size(-1) != logits.size(-2):
        raise ValueError("edge logits and Teacher delta must end in [N,N]")
    if float(weight_power) <= 0.0:
        raise ValueError("edge rank weight_power must be positive")
    if float(temperature) <= 0.0:
        raise ValueError("edge pairwise temperature must be positive")

    n_agents = logits.size(-1)
    diagonal = th.eye(n_agents, device=logits.device, dtype=th.bool)
    diagonal = diagonal.view(*([1] * (logits.dim() - 2)), n_agents, n_agents)
    sender_valid = (~diagonal).expand_as(logits)

    # Use the strict upper triangle so each unordered sender pair is counted
    # once; the sign of the Teacher delta difference supplies its orientation.
    upper = th.triu(
        th.ones(n_agents, n_agents, device=logits.device, dtype=th.bool),
        diagonal=1,
    )
    pair_valid = (
        sender_valid.unsqueeze(-1)
        & sender_valid.unsqueeze(-2)
        & upper.view(*([1] * (logits.dim() - 2)), 1, n_agents, n_agents)
    )
    pair_delta = delta.detach().unsqueeze(-1) - delta.detach().unsqueeze(-2)
    pair_gap = pair_delta.abs()
    pair_valid = pair_valid & pair_gap.gt(float(min_advantage))

    if transition_mask is not None:
        row_mask = transition_mask.bool()
        while row_mask.dim() < logits.dim() - 1:
            row_mask = row_mask.unsqueeze(-1)
        pair_valid = pair_valid & row_mask.unsqueeze(-1).unsqueeze(-1)

    if not pair_valid.any():
        return logits.sum() * 0.0

    pair_logits = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    orientation = pair_delta.sign()
    pair_losses = F.softplus(
        -(orientation * pair_logits) / float(temperature)
    )

    selected_gap = pair_gap[pair_valid]
    raw_weights = selected_gap.pow(float(weight_power))
    weights = raw_weights / raw_weights.mean().clamp_min(1e-12)
    if max_weight is not None and float(max_weight) > 0.0:
        weights = weights.clamp(max=float(max_weight))
    weights = weights.detach()
    return (pair_losses[pair_valid] * weights).sum() / weights.sum().clamp_min(1e-12)


def edge_delta_profile_regression_loss(
        logits, delta, transition_mask=None, min_advantage=1e-8,
        weight_power=1.0, max_weight=10.0, beta=0.5):
    """Regress the continuous within-row Teacher regret profile.

    Absolute Teacher regrets are extremely sparse and differ substantially in
    scale between trajectories. Since execution uses only sender ordering,
    predictions and targets are centered within each receiver row. Targets are
    divided by the Teacher top-vs-mean advantage, preserving relative utility
    without allowing zero-regret rows or raw value scale to dominate training.
    """
    if logits.shape != delta.shape:
        raise ValueError("edge logits and Teacher delta must have identical shapes")
    if logits.dim() < 2 or logits.size(-1) != logits.size(-2):
        raise ValueError("edge logits and Teacher delta must end in [N,N]")
    if float(weight_power) <= 0.0:
        raise ValueError("edge regression weight_power must be positive")
    if float(beta) <= 0.0:
        raise ValueError("edge regression beta must be positive")

    n_agents = logits.size(-1)
    diagonal = th.eye(n_agents, device=logits.device, dtype=th.bool)
    diagonal = diagonal.view(*([1] * (logits.dim() - 2)), n_agents, n_agents)
    valid_edges = (~diagonal).expand_as(logits)
    edge_count = valid_edges.sum(dim=-1).clamp_min(1).to(delta.dtype)

    detached_delta = delta.detach()
    delta_mean = detached_delta.masked_fill(~valid_edges, 0.0).sum(dim=-1) / edge_count
    best_delta = detached_delta.masked_fill(~valid_edges, float("-inf")).max(dim=-1).values
    advantage = (best_delta - delta_mean).clamp_min(0.0)
    informative = advantage > float(min_advantage)
    if transition_mask is not None:
        mask = transition_mask.bool()
        while mask.dim() < informative.dim():
            mask = mask.unsqueeze(-1)
        informative = informative & mask.expand_as(informative)
    if not informative.any():
        return logits.sum() * 0.0

    logits_mean = logits.masked_fill(~valid_edges, 0.0).sum(dim=-1) / edge_count
    centered_logits = logits - logits_mean.unsqueeze(-1)
    normalized_target = (
        (detached_delta - delta_mean.unsqueeze(-1))
        / advantage.clamp_min(float(min_advantage)).unsqueeze(-1)
    )
    edge_loss = F.smooth_l1_loss(
        centered_logits,
        normalized_target,
        reduction="none",
        beta=float(beta),
    )
    edge_loss = edge_loss.masked_fill(~valid_edges, 0.0).sum(dim=-1) / edge_count

    selected_advantage = advantage[informative]
    raw_weights = selected_advantage.pow(float(weight_power))
    weights = raw_weights / raw_weights.mean().clamp_min(1e-12)
    if max_weight is not None and float(max_weight) > 0.0:
        weights = weights.clamp(max=float(max_weight))
    weights = weights.detach()
    return (edge_loss[informative] * weights).sum() / weights.sum().clamp_min(1e-12)


def edge_supervision_loss(
        logits, teacher_edges, delta, transition_mask=None, target_mode="hard",
        positive_weight=None, min_advantage=1e-8, weight_power=1.0,
        max_weight=10.0, pairwise_temperature=1.0,
        value_temperature=0.008, regression_beta=0.5):
    """Shared Teacher-to-Student objective used by every training path.

    Centralizing mode dispatch prevents auxiliary calibration and end-to-end
    policy training from silently interpreting the same configuration flag in
    different ways. Rank-delta delegates to the row-wise loss above, so tied
    or zero-utility rows provide exactly zero supervision.
    """
    if logits.shape != teacher_edges.shape or logits.shape != delta.shape:
        raise ValueError(
            "edge logits, teacher edges, and Teacher delta must have identical shapes"
        )

    mode = str(target_mode).lower()
    if mode == "pairwise_delta":
        return edge_delta_pairwise_rank_loss(
            logits, delta, transition_mask=transition_mask,
            min_advantage=min_advantage, weight_power=weight_power,
            max_weight=max_weight, temperature=pairwise_temperature,
        )
    if mode == "rank_delta":
        return edge_delta_rank_loss(
            logits, delta, transition_mask=transition_mask,
            min_advantage=min_advantage, weight_power=weight_power,
            max_weight=max_weight,
        )
    if mode == "regress_delta":
        return edge_delta_profile_regression_loss(
            logits, delta, transition_mask=transition_mask,
            min_advantage=min_advantage, weight_power=weight_power,
            max_weight=max_weight, beta=regression_beta,
        )
    if mode == "soft_delta":
        temperature = max(float(value_temperature), 1e-8)
        soft_targets = 1.0 - th.exp(-delta.detach().clamp_min(0.0) / temperature)
        n_agents = logits.size(-1)
        diagonal = th.eye(n_agents, device=logits.device, dtype=th.bool)
        edge_valid = (~diagonal).view(
            *([1] * (logits.dim() - 2)), n_agents, n_agents
        ).expand_as(logits)
        if transition_mask is not None:
            mask = transition_mask.bool()
            while mask.dim() < logits.dim():
                mask = mask.unsqueeze(-1)
            edge_valid = edge_valid & mask.expand_as(logits)
        if not edge_valid.any():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            logits[edge_valid], soft_targets[edge_valid]
        )
    if mode == "hierarchical":
        n_agents = logits.size(-1)
        diagonal = th.eye(n_agents, device=logits.device, dtype=th.bool)
        diagonal = diagonal.view(
            *([1] * (logits.dim() - 2)), n_agents, n_agents
        )
        masked_logits = logits.masked_fill(diagonal, -1e9)
        row_targets = teacher_edges.sum(dim=-1).gt(0).to(logits.dtype)
        row_logits = masked_logits.max(dim=-1).values
        row_valid = th.ones_like(row_targets, dtype=th.bool)
        if transition_mask is not None:
            mask = transition_mask.bool()
            while mask.dim() < row_valid.dim():
                mask = mask.unsqueeze(-1)
            row_valid = row_valid & mask.expand_as(row_valid)
        if not row_valid.any():
            return logits.sum() * 0.0
        valid_row_targets = row_targets[row_valid]
        positive_rows = valid_row_targets.sum()
        negative_rows = valid_row_targets.numel() - positive_rows
        row_pos_weight = th.where(
            positive_rows > 0,
            negative_rows / positive_rows.clamp_min(1.0),
            th.ones_like(positive_rows),
        ).detach()
        presence_loss = F.binary_cross_entropy_with_logits(
            row_logits[row_valid], valid_row_targets, pos_weight=row_pos_weight
        )
        positive_row_mask = row_valid & row_targets.bool()
        if positive_row_mask.any():
            sender_targets = teacher_edges.argmax(dim=-1)
            sender_loss = F.cross_entropy(
                masked_logits[positive_row_mask], sender_targets[positive_row_mask]
            )
        else:
            sender_loss = logits.sum() * 0.0
        return presence_loss + sender_loss
    if mode == "hard":
        return edge_distillation_loss(
            logits, teacher_edges, transition_mask=transition_mask,
            positive_weight=positive_weight,
        )
    raise ValueError(
        "target_mode must be 'hard', 'pairwise_delta', 'rank_delta', "
        "'regress_delta', 'soft_delta', or 'hierarchical'"
    )
