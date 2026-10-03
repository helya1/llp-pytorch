import numpy as np
import torch
from torch import nn

from .losses import proportion_bce, proportion_mse

_LOSSES = {"bce": proportion_bce, "mse": proportion_mse}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_llp(model: nn.Module, X: np.ndarray, bag_idx: np.ndarray, props: np.ndarray,
              epochs: int = 60, lr: float = 1e-3, bags_per_batch: int = 16,
              weight_decay: float = 1e-4, loss: str = "bce", seed: int = 0,
              device=None, verbose: bool = False) -> list[float]:
    """Train the instance-level model using ONLY the bag proportions.

    X: (n, d) features, bag_idx: (n_bags, bag_size) indices into X,
    props: (n_bags,) observed proportions. Returns the loss history.
    """
    device = device or get_device()
    set_seed(seed)
    model.to(device).train()
    Xb = torch.as_tensor(X[bag_idx], device=device)          # (n_bags, bag_size, d)
    pb = torch.as_tensor(props, device=device)               # (n_bags,)
    n_bags, bag_size, d = Xb.shape
    loss_fn = _LOSSES[loss]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n_bags, generator=gen)
        total, count = 0.0, 0
        for start in range(0, n_bags, bags_per_batch):
            b = perm[start:start + bags_per_batch].to(device)
            logits = model(Xb[b].reshape(-1, d)).reshape(len(b), bag_size)
            l = loss_fn(logits, pb[b])
            opt.zero_grad()
            l.backward()
            opt.step()
            total += l.item() * len(b)
            count += len(b)
        history.append(total / count)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:3d}  bag loss {history[-1]:.4f}")
    return history


def train_supervised(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 30,
                     lr: float = 1e-3, batch_size: int = 128, weight_decay: float = 1e-4,
                     seed: int = 0, device=None) -> list[float]:
    """Reference model trained with the individual labels (upper bound for LLP)."""
    device = device or get_device()
    set_seed(seed)
    model.to(device).train()
    Xt = torch.as_tensor(X, device=device)
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
            l = bce(model(Xt[b]), yt[b])
            opt.zero_grad()
            l.backward()
            opt.step()
            total += l.item() * len(b)
        history.append(total / len(yt))
    return history
