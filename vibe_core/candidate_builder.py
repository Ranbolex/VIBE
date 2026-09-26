from dataclasses import dataclass

import torch as th


@dataclass
class CandidateBatch:
    actions: th.Tensor
    agent_qs: th.Tensor
    valid: th.Tensor
    scores: th.Tensor = None

    @property
    def size(self):
        return self.valid.sum(dim=-1)


class CandidateBuilder:
    """Compact joint-action candidate construction.

    ``local_topk`` uses a bounded beam over per-agent top-k utilities. ``pow_qr``
    uses the same high-recall pool and delegates final ranking to a Q_r scoring
    callback. ``oracle`` accepts an explicit diagnostic candidate tensor.
    """

    def __init__(self, mode="local_topk", top_k=2, top_m=8, pool_size=None,
                 tolerance=0.1):
        if mode not in {"oracle", "local_topk", "pow_qr"}:
            raise ValueError("unknown candidate mode: {}".format(mode))
        self.mode = mode
        self.top_k = int(top_k)
        self.top_m = int(top_m)
        self.pool_size = int(pool_size or max(top_m * 4, top_m))
        self.tolerance = float(tolerance)

    def build(self, agent_qs, avail_actions=None, oracle=None, score_fn=None):
        if agent_qs.dim() != 4:
            raise ValueError("agent_qs must have shape [B,T,N,A]")
        if self.mode == "oracle":
            return self._oracle_candidates(agent_qs, oracle)

        candidates = self._local_topk_candidates(agent_qs, avail_actions)
        if self.mode == "pow_qr":
            if score_fn is None:
                raise ValueError("pow_qr mode requires a score_fn")
            scores = score_fn(candidates.actions, candidates.agent_qs)
            q_values = agent_qs.detach()
            if avail_actions is not None:
                q_values = q_values.masked_fill(avail_actions == 0, -1e9)
            greedy_actions = q_values.argmax(dim=-1)
            return self._retain_pow_members(
                candidates, scores, greedy_actions, self.tolerance
            )
        return candidates

    def _oracle_candidates(self, agent_qs, oracle):
        if oracle is None:
            raise ValueError("oracle candidates are required in oracle mode")
        actions = th.as_tensor(oracle, device=agent_qs.device, dtype=th.long)
        if actions.dim() == 2:
            actions = actions.view(1, 1, *actions.shape).expand(agent_qs.size(0), agent_qs.size(1), -1, -1)
        if actions.dim() != 4:
            raise ValueError("oracle candidates must have shape [M,N] or [B,T,M,N]")
        gathered = self.gather_agent_qs(agent_qs, actions)
        valid = th.ones(actions.shape[:-1], device=agent_qs.device, dtype=th.bool)
        return CandidateBatch(actions=actions, agent_qs=gathered, valid=valid)

    def _local_topk_candidates(self, agent_qs, avail_actions):
        batch_size, timesteps, n_agents, n_actions = agent_qs.shape
        q_values = agent_qs.detach()
        if avail_actions is not None:
            q_values = q_values.masked_fill(avail_actions == 0, -1e9)
        k = min(max(self.top_k, 1), n_actions)
        top_values, top_actions = q_values.topk(k, dim=-1)

        rows_actions = []
        rows_scores = []
        flat_values = top_values.reshape(-1, n_agents, k)
        flat_actions = top_actions.reshape(-1, n_agents, k)
        flat_q_values = q_values.reshape(-1, n_agents, n_actions)
        flat_greedy = q_values.argmax(dim=-1).reshape(-1, n_agents)
        for row_values, row_actions, row_q_values, row_greedy in zip(
                flat_values, flat_actions, flat_q_values, flat_greedy):
            beam_actions = th.empty((1, 0), device=agent_qs.device, dtype=th.long)
            beam_scores = th.zeros(1, device=agent_qs.device, dtype=agent_qs.dtype)
            for agent_id in range(n_agents):
                expanded_actions = beam_actions.unsqueeze(1).expand(-1, k, -1)
                next_actions = row_actions[agent_id].view(1, k, 1).expand(beam_actions.size(0), -1, -1)
                beam_actions = th.cat([expanded_actions, next_actions], dim=-1).reshape(-1, agent_id + 1)
                beam_scores = (beam_scores.unsqueeze(1) + row_values[agent_id].unsqueeze(0)).reshape(-1)
                keep = min(self.pool_size, beam_scores.numel())
                beam_scores, indices = beam_scores.topk(keep)
                beam_actions = beam_actions[indices]
            # ``pow_qr`` needs an over-complete local beam: the centralized
            # score function can only improve the candidate set if it ranks a
            # pool larger than the final retained set.  Previously this branch
            # emitted exactly ``top_m`` candidates and then retained ``top_m``
            # again, making Q_r ranking a no-op.
            final_pool = self.pool_size if self.mode == "pow_qr" else self.top_m
            keep = min(final_pool, beam_scores.numel())
            scores, indices = beam_scores.topk(keep)
            selected_actions = beam_actions[indices]
            if self.mode == "pow_qr" and not selected_actions.eq(
                    row_greedy.unsqueeze(0)).all(dim=-1).any():
                # topk tie ordering differs across PyTorch/CUDA versions.  The
                # POW threshold is defined relative to the decentralized
                # greedy action, so retain that anchor explicitly rather than
                # relying on incidental beam ordering.
                selected_actions[-1] = row_greedy
                scores[-1] = row_q_values.gather(
                    -1, row_greedy.unsqueeze(-1)
                ).sum()
            rows_actions.append(selected_actions)
            rows_scores.append(scores)

        max_m = max(row.size(0) for row in rows_actions)
        actions = th.zeros((len(rows_actions), max_m, n_agents), device=agent_qs.device, dtype=th.long)
        scores = th.full((len(rows_actions), max_m), -1e9, device=agent_qs.device, dtype=agent_qs.dtype)
        valid = th.zeros((len(rows_actions), max_m), device=agent_qs.device, dtype=th.bool)
        for row_id, (row_actions, row_scores) in enumerate(zip(rows_actions, rows_scores)):
            count = row_actions.size(0)
            actions[row_id, :count] = row_actions
            scores[row_id, :count] = row_scores
            valid[row_id, :count] = row_scores > -1e8
        actions = actions.view(batch_size, timesteps, max_m, n_agents)
        scores = scores.view(batch_size, timesteps, max_m)
        valid = valid.view(batch_size, timesteps, max_m)
        gathered = self.gather_agent_qs(agent_qs, actions)
        return CandidateBatch(actions=actions, agent_qs=gathered, valid=valid, scores=scores)

    def _retain_top_m(self, candidates, scores):
        scores = scores.masked_fill(~candidates.valid, -1e9)
        keep = min(self.top_m, scores.size(-1))
        top_scores, indices = scores.topk(keep, dim=-1)
        action_indices = indices.unsqueeze(-1).expand(-1, -1, -1, candidates.actions.size(-1))
        actions = candidates.actions.gather(-2, action_indices)
        agent_qs = candidates.agent_qs.gather(-2, action_indices)
        valid = candidates.valid.gather(-1, indices)
        return CandidateBatch(actions=actions, agent_qs=agent_qs, valid=valid, scores=top_scores)

    def _retain_pow_members(self, candidates, scores, greedy_actions, tolerance):
        """Apply POW membership inside a bounded proposal pool.

        The returned set is an approximation to A_r because the proposal pool
        is bounded, but membership within that pool follows
        Q_r(a) >= Q_r(a_greedy) - C and always retains the greedy anchor.
        """
        if scores.shape != candidates.valid.shape:
            raise ValueError("POW scores must match candidate validity")
        if greedy_actions.shape != candidates.actions.shape[:2] + (
                candidates.actions.size(-1),):
            raise ValueError("greedy_actions must have shape [B,T,N]")

        flat_actions = candidates.actions.reshape(
            -1, candidates.actions.size(-2), candidates.actions.size(-1)
        )
        flat_agent_qs = candidates.agent_qs.reshape_as(flat_actions).to(
            candidates.agent_qs.dtype
        )
        flat_valid = candidates.valid.reshape(-1, candidates.valid.size(-1))
        flat_scores = scores.reshape_as(flat_valid).to(scores.dtype)
        flat_greedy = greedy_actions.reshape(-1, greedy_actions.size(-1))

        rows = flat_actions.size(0)
        keep_max = min(self.top_m, flat_actions.size(1))
        out_actions = th.zeros(
            rows, keep_max, flat_actions.size(-1),
            device=flat_actions.device, dtype=flat_actions.dtype,
        )
        out_agent_qs = th.zeros(
            rows, keep_max, flat_agent_qs.size(-1),
            device=flat_agent_qs.device, dtype=flat_agent_qs.dtype,
        )
        out_scores = th.full(
            (rows, keep_max), -1e9, device=flat_scores.device,
            dtype=flat_scores.dtype,
        )
        out_valid = th.zeros(
            rows, keep_max, device=flat_valid.device, dtype=th.bool
        )

        for row in range(rows):
            valid_indices = flat_valid[row].nonzero(as_tuple=False).squeeze(-1)
            # Replay batches are padded to a common episode length.  After a
            # shorter episode terminates, every action in the padded row is
            # unavailable and the bounded proposal pool is intentionally
            # empty.  Preserve an all-invalid output for that masked row; the
            # greedy-anchor invariant applies only to real, valid states.
            if valid_indices.numel() == 0:
                continue
            anchor_matches = (
                flat_actions[row].eq(flat_greedy[row].unsqueeze(0)).all(dim=-1)
                & flat_valid[row]
            )
            anchor_indices = anchor_matches.nonzero(as_tuple=False).squeeze(-1)
            if anchor_indices.numel() == 0:
                raise RuntimeError("bounded POW proposal pool lost the greedy anchor")
            anchor = anchor_indices[0]
            threshold = flat_scores[row, anchor] - float(tolerance)
            members = valid_indices[
                flat_scores[row, valid_indices] >= threshold
            ]
            member_scores = flat_scores[row, members]
            order = member_scores.argsort(descending=True)
            selected = members[order[:keep_max]]
            if not selected.eq(anchor).any():
                selected[-1] = anchor
                selected = selected[
                    flat_scores[row, selected].argsort(descending=True)
                ]
            count = selected.numel()
            out_actions[row, :count] = flat_actions[row, selected]
            out_agent_qs[row, :count] = flat_agent_qs[row, selected]
            out_scores[row, :count] = flat_scores[row, selected]
            out_valid[row, :count] = True

        prefix = candidates.actions.shape[:2]
        return CandidateBatch(
            actions=out_actions.view(*prefix, keep_max, flat_actions.size(-1)),
            agent_qs=out_agent_qs.view(*prefix, keep_max, flat_agent_qs.size(-1)),
            valid=out_valid.view(*prefix, keep_max),
            scores=out_scores.view(*prefix, keep_max),
        )

    @staticmethod
    def gather_agent_qs(agent_qs, candidates):
        expanded = agent_qs.unsqueeze(-3).expand(-1, -1, candidates.size(-2), -1, -1)
        return expanded.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
