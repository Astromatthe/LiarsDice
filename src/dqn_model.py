import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Small feedforward network used by training and by loaded opponent DQN bots.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DRONMoE(nn.Module):
    """DRON-MoE network for opponent learning using Mixture-of-Experts."""

    def __init__(self, state_dim: int, action_dim: int, non_opp_dim: int, opp_dim: int, num_experts: int = 3, hidden_dim: int = 128, gate_hidden: int = 64, prefix_dim: int = 10):
        super().__init__()
        self.non_opp_dim = non_opp_dim
        self.opp_dim = opp_dim
        self.num_experts = num_experts
        self.state_dim = state_dim
        self.prefix_dim = prefix_dim

        self.fcs1 = nn.Linear(prefix_dim, hidden_dim)
        self.fcs2 = nn.Linear(hidden_dim, hidden_dim)

        self.fco1 = nn.Linear(opp_dim, gate_hidden)
        self.fco2 = nn.Linear(gate_hidden, num_experts)

        self.experts = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(num_experts)])

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == self.state_dim, (f"Expected state_dim={self.state_dim}, got {x.size(1)}")

        state_part = x[:, :self.prefix_dim]
        opp_part = x[:, self.prefix_dim:self.prefix_dim + self.opp_dim]

        h_s = F.relu(self.fcs1(state_part))
        h_s = F.relu(self.fcs2(h_s))

        h_o = F.relu(self.fco1(opp_part))
        gate_logits = self.fco2(h_o)
        gate_weights = F.softmax(gate_logits, dim=-1)

        expert_qs = torch.stack([exp(h_s) for exp in self.experts], dim=1)

        w = gate_weights.unsqueeze(-1)
        q = (w*expert_qs).sum(dim=1)
        return q