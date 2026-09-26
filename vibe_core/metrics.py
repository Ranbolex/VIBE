import torch as th


def _binary_edge_metrics(predictions, targets, valid):
    tp = (predictions & targets & valid).sum().float()
    predicted = (predictions & valid).sum().float()
    positives = (targets & valid).sum().float()
    valid_count = valid.sum().clamp_min(1).float()
    precision = tp / predicted.clamp_min(1.0)
    recall = tp / positives.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    agreement = ((predictions == targets) & valid).sum().float() / valid_count
    predicted_edge_density = predicted / valid_count
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_agreement": agreement,
        "graph_sparsity": 1.0 - predicted_edge_density,
        "predicted_edge_density": predicted_edge_density,
    }


def edge_weight_metrics(weights, labels, transition_mask=None):
    """Compare the graph actually used by the controller with teacher edges."""
    if weights.shape != labels.shape:
        raise ValueError("edge weights and labels must have identical shapes")
    n_agents = weights.size(-1)
    diagonal = th.eye(n_agents, device=weights.device, dtype=th.bool)
    targets = labels.bool()
    valid = (~diagonal).view(*([1] * (weights.dim() - 2)), n_agents, n_agents).expand_as(targets)
    if transition_mask is not None:
        mask = transition_mask.bool()
        while mask.dim() < targets.dim():
            mask = mask.unsqueeze(-1)
        valid = valid & mask.expand_as(targets)
    return _binary_edge_metrics(weights > 0.5, targets, valid)


def edge_classification_metrics(logits, labels, transition_mask=None, top_k=1, threshold=0.5):
    n_agents = logits.size(-1)
    k = min(max(int(top_k), 0), max(n_agents - 1, 0))
    diagonal = th.eye(n_agents, device=logits.device, dtype=th.bool)
    masked_logits = logits.masked_fill(diagonal.view(*([1] * (logits.dim() - 2)), n_agents, n_agents), -1e9)
    predictions = th.zeros_like(labels, dtype=th.bool)
    topk_probability_mean = logits.new_zeros(())
    if k:
        values, indices = masked_logits.topk(k, dim=-1)
        predictions.scatter_(-1, indices, th.sigmoid(values) >= threshold)
    targets = labels.bool()
    valid = (~diagonal).view(*([1] * (logits.dim() - 2)), n_agents, n_agents).expand_as(targets)
    if transition_mask is not None:
        mask = transition_mask.bool()
        while mask.dim() < targets.dim():
            mask = mask.unsqueeze(-1)
        valid = valid & mask.expand_as(targets)
    probabilities = th.sigmoid(logits)
    valid_count = valid.sum().clamp_min(1).float()
    if k:
        selected_valid = valid.gather(-1, indices)
        topk_probability_mean = (
            th.sigmoid(values) * selected_valid.to(values.dtype)
        ).sum() / selected_valid.sum().clamp_min(1).float()
    positives = (targets & valid).sum().float()
    negatives = ((~targets) & valid).sum().float()
    balance_weight = th.where(
        positives > 0,
        negatives / positives.clamp_min(1.0),
        th.ones_like(positives),
    )
    metrics = _binary_edge_metrics(predictions, targets, valid)
    metrics.update({
        "teacher_edge_density": positives / valid_count,
        "edge_balance_positive_weight": balance_weight,
        "edge_probability_mean": (probabilities * valid).sum() / valid_count,
        "edge_positive_probability_mean": (
            probabilities * targets * valid
        ).sum() / positives.clamp_min(1.0),
        "edge_negative_probability_mean": (
            probabilities * (~targets) * valid
        ).sum() / negatives.clamp_min(1.0),
        "edge_topk_probability_mean": topk_probability_mean,
    })
    return metrics


def edge_delta_ranking_metrics(logits, delta, transition_mask=None, min_advantage=1e-8):
    """Evaluate top-1 sender ranking only where Teacher regret is informative."""
    if logits.shape != delta.shape:
        raise ValueError("edge logits and Teacher delta must have identical shapes")
    n_agents = logits.size(-1)
    diagonal = th.eye(n_agents, device=logits.device, dtype=th.bool)
    diagonal = diagonal.view(*([1] * (logits.dim() - 2)), n_agents, n_agents)
    valid_edges = (~diagonal).expand_as(logits)
    masked_delta = delta.masked_fill(~valid_edges, float("-inf"))
    masked_logits = logits.masked_fill(~valid_edges, float("-inf"))
    best_delta, best_sender = masked_delta.max(dim=-1)
    student_sender = masked_logits.argmax(dim=-1)
    mean_delta = delta.masked_fill(~valid_edges, 0.0).sum(dim=-1) / valid_edges.sum(
        dim=-1
    ).clamp_min(1).to(delta.dtype)
    advantage = (best_delta - mean_delta).clamp_min(0.0)
    valid_rows = th.ones_like(advantage, dtype=th.bool)
    if transition_mask is not None:
        mask = transition_mask.bool()
        while mask.dim() < valid_rows.dim():
            mask = mask.unsqueeze(-1)
        valid_rows = valid_rows & mask.expand_as(valid_rows)
    informative = valid_rows & (advantage > float(min_advantage))
    correct = student_sender.eq(best_sender)
    informative_count = informative.sum().clamp_min(1).to(delta.dtype)
    valid_count = valid_rows.sum().clamp_min(1).to(delta.dtype)
    weighted_denominator = advantage.masked_fill(~informative, 0.0).sum().clamp_min(
        float(min_advantage)
    )
    student_delta = delta.gather(-1, student_sender.unsqueeze(-1)).squeeze(-1)
    informative_float = informative.to(delta.dtype)
    correct_float = correct.to(delta.dtype)
    return {
        "informative_row_rate": informative.sum().to(delta.dtype) / valid_count,
        "informative_top1_accuracy": (correct & informative).sum().to(delta.dtype)
        / informative_count,
        "utility_weighted_top1_accuracy": (
            advantage * correct.to(delta.dtype) * informative.to(delta.dtype)
        ).sum()
        / weighted_denominator,
        "informative_student_delta": student_delta.masked_fill(~informative, 0.0).sum()
        / informative_count,
        "informative_teacher_delta": best_delta.masked_fill(~informative, 0.0).sum()
        / informative_count,
        "informative_random_expected_delta": mean_delta.masked_fill(~informative, 0.0).sum()
        / informative_count,
        "valid_row_count": valid_rows.sum().to(delta.dtype),
        "informative_row_count": informative.sum().to(delta.dtype),
        "informative_correct_count": (correct & informative).sum().to(delta.dtype),
        "utility_weighted_correct_sum": (
            advantage * correct_float * informative_float
        ).sum(),
        "informative_advantage_sum": (advantage * informative_float).sum(),
        "informative_student_delta_sum": (student_delta * informative_float).sum(),
        "informative_teacher_delta_sum": (best_delta * informative_float).sum(),
        "informative_random_expected_delta_sum": (mean_delta * informative_float).sum(),
    }


def candidate_action_metrics(candidates, optimal_actions, transition_mask=None):
    """Measure whether a compact candidate set retains a reference joint action."""
    expected_shape = candidates.actions.shape[:-2] + (candidates.actions.size(-1),)
    if optimal_actions.shape != expected_shape:
        raise ValueError("optimal_actions must have shape [B,T,N]")
    matches = candidates.actions.eq(optimal_actions.unsqueeze(-2)).all(dim=-1) & candidates.valid
    retained = matches.any(dim=-1).float()
    precision = matches.sum(dim=-1).float() / candidates.valid.sum(dim=-1).clamp_min(1).float()
    if transition_mask is not None:
        mask = transition_mask.squeeze(-1).to(retained.dtype)
        denominator = mask.sum().clamp_min(1.0)
        retained = (retained * mask).sum() / denominator
        precision = (precision * mask).sum() / denominator
    else:
        retained = retained.mean()
        precision = precision.mean()
    return {
        "candidate_recall": retained,
        "candidate_precision": precision,
        "oracle_best_in_candidate": retained,
    }


def counterfactual_action_metrics(
        base_q, teacher_q, random_q, executed_q, avail_actions, transition_mask=None):
    """Measure how graph interventions change available-action greedy decisions."""
    tensors = (teacher_q, random_q, executed_q, avail_actions)
    if any(tensor.shape != base_q.shape for tensor in tensors):
        raise ValueError("all counterfactual Q tensors and avail_actions must have identical shapes")

    def greedy(q_values):
        return q_values.masked_fill(avail_actions == 0, -1e9).argmax(dim=-1)

    base_actions = greedy(base_q)
    teacher_actions = greedy(teacher_q)
    random_actions = greedy(random_q)
    executed_actions = greedy(executed_q)
    valid = th.ones_like(base_actions, dtype=base_q.dtype)
    if transition_mask is not None:
        mask = transition_mask.to(base_q.dtype)
        while mask.dim() < valid.dim():
            mask = mask.unsqueeze(-1)
        valid = mask.expand_as(valid)
    denominator = valid.sum().clamp_min(1.0)

    def rate(condition):
        return (condition.to(base_q.dtype) * valid).sum() / denominator

    return {
        "teacher_action_flip_rate": rate(teacher_actions != base_actions),
        "random_action_flip_rate": rate(random_actions != base_actions),
        "executed_action_flip_rate": rate(executed_actions != base_actions),
        "teacher_executed_action_agreement": rate(teacher_actions == executed_actions),
    }
