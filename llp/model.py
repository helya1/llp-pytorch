"""Neural network used as instance-level predictor."""
import torch
from torch import nn


class MLP(nn.Module):
    """Instance-level predictor Phi: R^d -> logit of P(Y=1 | X)."""

    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits for a batch of instances, shape (n,)."""
        return self.net(x).squeeze(-1)
