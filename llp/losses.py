"""Bag-level losses for learning from label proportions.

For a bag S_j with known mean m_j of the labels, the model predicts an
instance-level probability p_i = sigmoid(Phi(x_i)). The bag-level prediction is
the average  p_bar_j = mean_i p_i  and is compared with m_j.
"""
import torch
import torch.nn.functional as F


def bag_probability(logits: torch.Tensor) -> torch.Tensor:
    """logits: (n_bags, bag_size) -> predicted proportion per bag, shape (n_bags,)."""
    return torch.sigmoid(logits).mean(dim=1)


def proportion_bce(logits: torch.Tensor, props: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Cross-entropy between the observed and the predicted proportion (DLLP-style loss)."""
    p = bag_probability(logits).clamp(eps, 1 - eps)
    return -(props * torch.log(p) + (1 - props) * torch.log(1 - p)).mean()


def proportion_mse(logits: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
    """Squared error between observed and predicted proportions."""
    return F.mse_loss(bag_probability(logits), props)
