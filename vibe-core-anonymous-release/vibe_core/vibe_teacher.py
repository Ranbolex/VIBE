import torch as th


class VIBETeacher:
    """Centralized decision-regret teacher for directed VIBE edges."""

    def __init__(self, n_agents, edge_budget=1, min_score=1e-8):
        self.n_agents = int(n_agents)
        self.edge_budget = int(edge_budget)
        self.min_score = float(min_score)

    def compute_cct_delta(self, batch_size, timesteps, device, dtype, v_pivotal, lambda_trap, epsilon_n):
        delta = th.zeros(
            batch_size,
            timesteps,
            self.n_agents,
            self.n_agents,
            device=device,
            dtype=dtype,
        )
        # Candidate set {(L,P,L),(R,P,R)} removes the already predictable
        # agent-2 fluctuation. Revealing s matters only in the s=R half.
        pivotal_delta = 0.5 * max(
            float(v_pivotal) - 2.0 * float(epsilon_n) * float(lambda_trap),
            0.0,
        )
        delta[..., 0, 1] = 0.0
        delta[..., 0, 2] = pivotal_delta
        return delta

    @staticmethod
    def evidence_value(payoffs, prior, likelihood):
        """Exact diagnostic VoI on a fixed receiver-decision support.

        Shapes: payoffs [..., decisions, latent_states], prior [..., latent_states],
        likelihood [..., evidence, latent_states] = P(evidence | latent_state).
        Returns expected regret reduction, evidence probabilities and posteriors.
        This oracle-table diagnostic does not replace the benchmark Teacher.
        """
        if payoffs.shape[:-2] != prior.shape[:-1] or likelihood.shape[:-2] != prior.shape[:-1]:
            raise ValueError("batch dimensions must match")
        if payoffs.shape[-1] != prior.shape[-1] or likelihood.shape[-1] != prior.shape[-1]:
            raise ValueError("latent dimensions must match")
        if not all(th.isfinite(x).all() for x in (payoffs, prior, likelihood)):
            raise ValueError("inputs must be finite")
        if (prior < 0).any() or (likelihood < 0).any():
            raise ValueError("probabilities must be nonnegative")
        if not th.allclose(prior.sum(-1), th.ones_like(prior.sum(-1))):
            raise ValueError("prior must sum to one")
        if not th.allclose(likelihood.sum(-2), th.ones_like(likelihood.sum(-2))):
            raise ValueError("likelihood must sum to one over evidence")
        joint = likelihood * prior.unsqueeze(-2)
        evidence_prob = joint.sum(-1)
        posterior = joint / evidence_prob.unsqueeze(-1).clamp_min(th.finfo(joint.dtype).tiny)
        prior_values = (payoffs * prior.unsqueeze(-2)).sum(-1)
        choice = prior_values.argmax(-1)
        posterior_values = th.einsum('...as,...es->...ea', payoffs, posterior)
        selected = posterior_values.gather(-1, choice[..., None, None].expand(*choice.shape, likelihood.shape[-2], 1)).squeeze(-1)
        delta = ((posterior_values.max(-1).values - selected) * evidence_prob).sum(-1)
        return delta, evidence_prob, posterior

    @staticmethod
    def cct_reference_scores(v_pivotal=8.0, lambda_trap=24.0, epsilon_n=0.05):
        """Analytic scores used by the CCT mechanism comparison plots."""
        v = float(v_pivotal)
        coupling = float(lambda_trap)
        epsilon = float(epsilon_n)
        return {
            "action_variance": {
                "agent_2": 0.25 * coupling ** 2,
                "agent_3": 0.25 * v ** 2,
            },
            "belief_value_variance": {
                "agent_2": epsilon * (1.0 - epsilon) * coupling ** 2,
                "agent_3": 0.25 * v ** 2,
            },
            "vibe_delta": {
                "agent_2": 0.0,
                "agent_3": 0.5 * max(v - 2.0 * epsilon * coupling, 0.0),
            },
        }

    def compute_from_candidates(
            self, candidate_values, candidate_agent_qs, candidate_actions, valid,
            uncertainty=None, with_candidate_values=None,
            without_candidate_values=None):
        """Pair-specific empirical with-j/without-j regret approximation.

        Removing sender j's candidate contribution produces the without-j
        decision.  Regret is charged only to receivers whose candidate action
        changes, then weighted by receiver i's residual belief uncertainty
        about sender j.  The expectation over evidence is estimated by replay
        samples.  Diagnostic environments should still use analytic teachers.
        """
        if candidate_values.dim() != 3 or candidate_agent_qs.dim() != 4:
            raise ValueError("invalid candidate value shapes")
        if candidate_actions.shape != candidate_agent_qs.shape:
            raise ValueError("candidate_actions and candidate_agent_qs must have identical shapes")
        if (with_candidate_values is None) != (without_candidate_values is None):
            raise ValueError("with and without candidate values must be supplied together")
        if with_candidate_values is not None:
            return self._compute_from_counterfactual_values(
                candidate_actions, valid, uncertainty,
                with_candidate_values, without_candidate_values,
            )
        full_values = candidate_values.masked_fill(~valid, -1e9)
        full_best, full_choice = full_values.max(dim=-1)
        if uncertainty is None:
            uncertainty = th.ones(
                *candidate_values.shape[:2], self.n_agents, self.n_agents,
                device=candidate_values.device,
                dtype=candidate_values.dtype,
            )
        elif uncertainty.dim() == 3:
            uncertainty = uncertainty.unsqueeze(-2).expand(
                -1, -1, self.n_agents, -1
            )
        expected_uncertainty_shape = candidate_values.shape[:2] + (
            self.n_agents, self.n_agents
        )
        if uncertainty.shape != expected_uncertainty_shape:
            raise ValueError("uncertainty must have shape [B,T,N] or [B,T,N,N]")

        full_action_index = full_choice.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, self.n_agents
        )
        full_actions = candidate_actions.gather(-2, full_action_index).squeeze(-2)
        delta_by_sender = []
        for sender in range(self.n_agents):
            sender_q = candidate_agent_qs[..., sender]
            sender_mean = (sender_q * valid.to(sender_q.dtype)).sum(dim=-1, keepdim=True) / valid.sum(
                dim=-1, keepdim=True
            ).clamp_min(1)
            without_values = full_values - sender_q + sender_mean
            without_choice = without_values.argmax(dim=-1, keepdim=True)
            chosen_under_full = full_values.gather(-1, without_choice).squeeze(-1)
            regret = (full_best - chosen_under_full).clamp_min(0.0)
            without_action_index = without_choice.unsqueeze(-1).expand(
                -1, -1, 1, self.n_agents
            )
            without_actions = candidate_actions.gather(
                -2, without_action_index
            ).squeeze(-2)
            receiver_decision_changed = (full_actions != without_actions).to(
                candidate_values.dtype
            )
            pair_delta = (
                regret.unsqueeze(-1)
                * receiver_decision_changed
                * uncertainty[..., sender]
            )
            delta_by_sender.append(pair_delta)
        delta = th.stack(delta_by_sender, dim=-1)
        diagonal = th.eye(self.n_agents, device=delta.device, dtype=th.bool)
        return delta.masked_fill(diagonal.view(1, 1, self.n_agents, self.n_agents), 0.0)

    def compute_value_difference_variance(
            self, valid, with_candidate_values, without_candidate_values):
        """CASEC-style dispersion of each edge's value contribution.

        The score is the population variance, over the same bounded candidate
        set used by VIBE, of ``Q_with - Q_without``.  This deliberately omits
        the receiver-action switch gate and belief-uncertainty weighting so it
        provides a matched-interface certainty-blind Variance-K comparator.
        """
        if valid.dim() != 3:
            raise ValueError("valid must have shape [B,T,C]")
        expected = valid.shape[:2] + (
            self.n_agents, self.n_agents, valid.size(-1)
        )
        if (
            with_candidate_values.shape != expected
            or without_candidate_values.shape != expected
        ):
            raise ValueError(
                "counterfactual values must have shape [B,T,N,N,C]"
            )
        mask = valid.unsqueeze(-2).unsqueeze(-2).to(
            with_candidate_values.dtype
        )
        count = mask.sum(dim=-1).clamp_min(1.0)
        contribution = with_candidate_values - without_candidate_values
        mean = (contribution * mask).sum(dim=-1) / count
        variance = (
            (contribution - mean.unsqueeze(-1)).square() * mask
        ).sum(dim=-1) / count
        diagonal = th.eye(
            self.n_agents, device=variance.device, dtype=th.bool
        ).view(1, 1, self.n_agents, self.n_agents)
        return variance.masked_fill(diagonal, 0.0)

    def _compute_from_counterfactual_values(
            self, candidate_actions, valid, uncertainty,
            with_candidate_values, without_candidate_values):
        """Score explicit with-evidence and withdrawn-evidence candidate values.

        Both tensors have shape ``[B,T,receiver,sender,candidate]``. This is a
        generic replacement path for environments that can evaluate a genuine
        evidence intervention; the legacy sender-Q subtraction remains the
        fallback for benchmark environments without such a counterfactual.
        """
        expected = candidate_actions.shape[:2] + (
            self.n_agents, self.n_agents, candidate_actions.size(-2)
        )
        if with_candidate_values.shape != expected or without_candidate_values.shape != expected:
            raise ValueError("counterfactual values must have shape [B,T,N,N,C]")
        if uncertainty is None:
            uncertainty = th.ones(
                *candidate_actions.shape[:2], self.n_agents, self.n_agents,
                device=candidate_actions.device, dtype=with_candidate_values.dtype,
            )
        elif uncertainty.dim() == 3:
            uncertainty = uncertainty.unsqueeze(-2).expand(-1, -1, self.n_agents, -1)
        expected_uncertainty = candidate_actions.shape[:2] + (self.n_agents, self.n_agents)
        if uncertainty.shape != expected_uncertainty:
            raise ValueError("uncertainty must have shape [B,T,N] or [B,T,N,N]")

        valid_mask = valid.unsqueeze(-2).unsqueeze(-2)
        full_values = with_candidate_values.masked_fill(~valid_mask, -1e9)
        withdrawn_values = without_candidate_values.masked_fill(~valid_mask, -1e9)
        full_best, full_choice = full_values.max(dim=-1)
        withdrawn_choice = withdrawn_values.argmax(dim=-1)
        withdrawn_under_full = full_values.gather(-1, withdrawn_choice.unsqueeze(-1)).squeeze(-1)
        regret = (full_best - withdrawn_under_full).clamp_min(0.0)

        batch_size, timesteps, _, n_agents = candidate_actions.shape
        expanded_actions = candidate_actions.unsqueeze(-3).unsqueeze(-3).expand(
            -1, -1, n_agents, n_agents, -1, -1
        )
        full_index = full_choice.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, -1, 1, n_agents
        )
        withdrawn_index = withdrawn_choice.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, -1, 1, n_agents
        )
        full_actions = expanded_actions.gather(-2, full_index).squeeze(-2)
        withdrawn_actions = expanded_actions.gather(-2, withdrawn_index).squeeze(-2)
        receiver_index = th.arange(n_agents, device=candidate_actions.device).view(1, 1, n_agents, 1, 1)
        receiver_index = receiver_index.expand(batch_size, timesteps, n_agents, n_agents, 1)
        receiver_changed = full_actions.gather(-1, receiver_index).squeeze(-1) != withdrawn_actions.gather(-1, receiver_index).squeeze(-1)
        diagonal = th.eye(n_agents, device=candidate_actions.device, dtype=th.bool).view(1, 1, n_agents, n_agents)
        delta = regret * receiver_changed.to(regret.dtype) * uncertainty
        return delta.masked_fill(diagonal, 0.0)

    def compute_from_evidence_values(self, candidate_actions, valid, without_values, evidence_values, evidence_probabilities, uncertainty=None):
        """Expected regret before observing a sender's evidence."""
        prefix = candidate_actions.shape[:2] + (self.n_agents, self.n_agents)
        candidates = candidate_actions.size(-2)
        if without_values.shape != prefix + (candidates,) or evidence_values.shape[:-2] != prefix or evidence_values.size(-1) != candidates:
            raise ValueError("invalid evidence counterfactual value shapes")
        if evidence_probabilities.shape != evidence_values.shape[:-1] or not th.allclose(evidence_probabilities.sum(-1), th.ones_like(evidence_probabilities[..., 0])):
            raise ValueError("invalid evidence probabilities")
        if uncertainty is None:
            uncertainty = th.ones(*prefix, device=candidate_actions.device, dtype=without_values.dtype)
        elif uncertainty.dim() == 3:
            uncertainty = uncertainty.unsqueeze(-2).expand(-1, -1, self.n_agents, -1)
        withdrawn = without_values.masked_fill(~valid.unsqueeze(-2).unsqueeze(-2), -1e9)
        observed = evidence_values.masked_fill(~valid.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2), -1e9)
        without_choice = withdrawn.argmax(-1)
        best, full_choice = observed.max(-1)
        under_without = observed.gather(-1, without_choice.unsqueeze(-1).unsqueeze(-1).expand_as(best.unsqueeze(-1))).squeeze(-1)
        regret = (best - under_without).clamp_min(0)
        batch, time, _, agents = candidate_actions.shape
        action_table = candidate_actions.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).expand(-1, -1, agents, agents, observed.size(-2), -1, -1)
        receiver = th.arange(agents, device=candidate_actions.device).view(1, 1, agents, 1, 1, 1).expand(batch, time, agents, agents, observed.size(-2), 1)
        full_actions = action_table.gather(-2, full_choice.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, -1, 1, agents)).squeeze(-2)
        withdrawn_actions = action_table.gather(-2, without_choice.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, observed.size(-2), 1, agents)).squeeze(-2)
        changed = full_actions.gather(-1, receiver).squeeze(-1) != withdrawn_actions.gather(-1, receiver).squeeze(-1)
        delta = (regret * changed.to(regret.dtype) * evidence_probabilities).sum(-1) * uncertainty
        diagonal = th.eye(agents, device=candidate_actions.device, dtype=th.bool).view(1, 1, agents, agents)
        return delta.masked_fill(diagonal, 0.0)

    def cct_expected_evidence_values(self, candidate_actions, epsilon_n, v_pivotal, lambda_trap):
        batch, time, _, agents = candidate_actions.shape
        dtype, device = th.float32, candidate_actions.device
        epsilon, v, trap = float(epsilon_n), float(v_pivotal), float(lambda_trap)
        base = th.tensor([0.5 * v + epsilon * trap, 0.5 * v - epsilon * trap], device=device, dtype=dtype)
        without = base.view(1, 1, 1, 1, 2).expand(batch, time, agents, agents, -1).clone()
        evidence = without.unsqueeze(-2).expand(-1, -1, -1, -1, 2, -1).clone()
        probabilities = th.full((batch, time, agents, agents, 2), 0.5, device=device, dtype=dtype)
        probabilities[..., 0, 1, 0], probabilities[..., 0, 1, 1] = 1.0 - epsilon, epsilon
        evidence[..., 0, 1, :, :] = th.tensor([[0.5*v, 0.5*v], [0.5*v+trap, 0.5*v-trap]], device=device, dtype=dtype)
        evidence[..., 0, 2, :, :] = th.tensor([[v*(1-epsilon)+epsilon*trap, -epsilon*trap], [epsilon*trap, v-epsilon*trap]], device=device, dtype=dtype)
        return without, evidence, probabilities

    def cct_evidence_withdrawal_values(
            self, candidate_actions, states, epsilon_n, v_pivotal, lambda_trap):
        """Return exact CCT candidate values with and without each edge's evidence.

        The receiver-local prior sees neither ``u2`` nor ``s``. For each edge,
        the with-evidence table conditions only on that sender's signal and the
        withdrawn table marginalizes both latent variables under this prior.
        """
        if candidate_actions.size(-1) != 3 or states.size(-1) < 4:
            raise ValueError("CCT evidence withdrawal requires 3-agent CCT states")
        batch_size, timesteps, candidates, n_agents = candidate_actions.shape
        dtype = states.dtype
        device = states.device
        epsilon = float(epsilon_n)
        v = float(v_pivotal)
        coupling = float(lambda_trap)
        base_actions = th.tensor(
            [0.5 * v + epsilon * coupling, 0.5 * v - epsilon * coupling],
            device=device, dtype=dtype,
        ).view(1, 1, 2).expand(batch_size, timesteps, -1)
        action = candidate_actions[..., 0]

        def gather(action_values):
            expanded = action_values.unsqueeze(-2).expand(-1, -1, candidates, -1)
            return expanded.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        with_values = gather(base_actions).unsqueeze(-2).unsqueeze(-2).expand(
            -1, -1, n_agents, n_agents, -1
        ).clone()
        without_values = with_values.clone()

        u2_is_n = states[..., 1] > 0.5
        s_is_r = states[..., 3] > 0.5
        u2_values = th.where(
            u2_is_n.unsqueeze(-1),
            th.tensor([0.5 * v + coupling, 0.5 * v - coupling], device=device, dtype=dtype),
            th.tensor([0.5 * v, 0.5 * v], device=device, dtype=dtype),
        )
        s_values = th.where(
            s_is_r.unsqueeze(-1),
            th.tensor([epsilon * coupling, v - epsilon * coupling], device=device, dtype=dtype),
            th.tensor([v * (1.0 - epsilon) + epsilon * coupling, -epsilon * coupling], device=device, dtype=dtype),
        )
        with_values[..., 0, 1, :] = gather(u2_values)
        with_values[..., 0, 2, :] = gather(s_values)
        return with_values, without_values

    def select_edges(self, delta, top_k=None, positive_only=True):
        if delta.size(-1) != self.n_agents or delta.size(-2) != self.n_agents:
            raise ValueError("delta must end in [N,N]")
        k = min(max(int(self.edge_budget if top_k is None else top_k), 0), max(self.n_agents - 1, 0))
        labels = th.zeros_like(delta)
        if k == 0:
            return labels
        diagonal = th.eye(self.n_agents, device=delta.device, dtype=th.bool)
        scores = delta.masked_fill(diagonal.view(*([1] * (delta.dim() - 2)), self.n_agents, self.n_agents), -1e9)
        values, indices = scores.topk(k, dim=-1)
        if positive_only:
            selected = (values > self.min_score).to(delta.dtype)
        else:
            selected = th.ones_like(values, dtype=delta.dtype)
        labels.scatter_(-1, indices, selected)
        return labels
