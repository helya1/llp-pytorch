"""Learning from Label Proportions (LLP) with PyTorch."""
from .data import make_dataset, make_bags
from .model import MLP
from .losses import proportion_bce, proportion_mse
from .train import train_llp, train_supervised
from .evaluate import evaluate

__all__ = [
    "make_dataset", "make_bags", "MLP", "proportion_bce", "proportion_mse",
    "train_llp", "train_supervised", "evaluate",
]
