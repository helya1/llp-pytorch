"""Training loops: LLP (bag proportions only) and supervised reference."""
import numpy as np
import torch
from torch import nn

from .losses import proportion_bce, proportion_mse

_LOSSES = {"bce": proportion_bce, "mse": proportion_mse}


def get_device() -> torch.device:
    """Return the GPU if available, else the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_llp(model: nn.Module, features: np.ndarray, bag_idx: np.ndarray, props: np.ndarray,
              epochs: int = 60, lr: float = 1e-3, bags_per_batch: int = 16,
              weight_decay: float = 1e-4, loss: str = "bce", seed: int = 0,
              device=None, verbose: bool = False) -> list[float]:
    """Train the instance-level model using ONLY the bag proportions.

    features: (n, d) array, bag_idx: (n_bags, bag_size) indices into features,
    props: (n_bags,) observed proportions. Returns the loss history.
    """
    device = device or get_device()
    set_seed(seed)
    model.to(device).train()
    x_bags = torch.as_tensor(features[bag_idx], device=device)  # (n_bags, bag_size, d)
    pb = torch.as_tensor(props, device=device)               # (n_bags,)
    n_bags, bag_size, d = x_bags.shape
    loss_fn = _LOSSES[loss]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n_bags, generator=gen)
        total, count = 0.0, 0
        for start in range(0, n_bags, bags_per_batch):
            b = perm[start:start + bags_per_batch].to(device)
            logits = model(x_bags[b].reshape(-1, d)).reshape(len(b), bag_size)
            batch_loss = loss_fn(logits, pb[b])
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total += batch_loss.item() * len(b)
            count += len(b)
        history.append(total / count)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:3d}  bag loss {history[-1]:.4f}")
    return history


def train_supervised(model: nn.Module, features: np.ndarray, y: np.ndarray, epochs: int = 30,
                     lr: float = 1e-3, batch_size: int = 128, weight_decay: float = 1e-4,
                     seed: int = 0, device=None) -> list[float]:
    """Reference model trained with the individual labels (upper bound for LLP)."""
    device = device or get_device()
    set_seed(seed)
    model.to(device).train()
    x_t = torch.as_tensor(features, device=device)
    yt = torch.as_tensor(y, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=gen)
        total = 0.0
        for start in range(0, len(yt), batch_size):
            b = perm[start:start + batch_size].to(device)
            batch_loss = bce(model(x_t[b]), yt[b])
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total += batch_loss.item() * len(b)
        history.append(total / len(yt))
    return history
