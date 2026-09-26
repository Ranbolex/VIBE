import torch as th
import torch.nn as nn


class PairwiseBeliefEncoder(nn.Module):
    """Predict a sender's next action from a receiver-local history.

    The centralized teacher uses the resulting distribution as an executable
    proxy for b_i^{-j}.  No sender hidden state or centralized state is exposed
    to the receiver-side predictor.
    """

    def __init__(self, hidden_dim, n_agents, n_actions, embed_dim=64):
        super().__init__()
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        input_dim = int(hidden_dim) + 2 * self.n_agents
        self.network = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, self.n_actions),
        )

    def forward(self, receiver_hidden):
        if receiver_hidden.dim() != 3:
            raise ValueError("receiver_hidden must have shape [B,N,H]")
        batch_size, n_agents, _ = receiver_hidden.shape
        if n_agents != self.n_agents:
            raise ValueError("agent dimension does not match PairwiseBeliefEncoder")

        identities = th.eye(
            n_agents, device=receiver_hidden.device, dtype=receiver_hidden.dtype
        )
        receiver_h = receiver_hidden.unsqueeze(2).expand(-1, -1, n_agents, -1)
        receiver_ids = identities.view(1, n_agents, 1, n_agents).expand(
            batch_size, -1, n_agents, -1
        )
        sender_ids = identities.view(1, 1, n_agents, n_agents).expand(
            batch_size, n_agents, -1, -1
        )
        pair_inputs = th.cat([receiver_h, receiver_ids, sender_ids], dim=-1)
        return self.network(pair_inputs)
