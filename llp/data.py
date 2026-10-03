"""Synthetic data and bag construction for LLP (numpy / scikit-learn only)."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_dataset(n_samples: int = 20000, n_features: int = 10, n_informative: int = 6,
                 test_size: float = 0.25, seed: int = 0):
    """Binary classification problem, standardised, split into train / test."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative,
        n_redundant=2, n_clusters_per_class=2, class_sep=1.0, flip_y=0.02,
        random_state=seed,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    return (scaler.transform(X_tr).astype(np.float32), y_tr.astype(np.float32),
            scaler.transform(X_te).astype(np.float32), y_te.astype(np.float32))


def make_bags(X: np.ndarray, y: np.ndarray, bag_size: int, rng: np.random.Generator,
              mode: str = "random", noise: float = 0.0):
    """Split the training set into disjoint bags of equal size.

    Only the bag-level mean of y (the label proportion) is returned: individual
    labels are never given to the learner.

    mode:  "random"    -> bags are drawn uniformly at random (weak signal when
                          bags are large, proportions concentrate around the
                          global rate);
           "clustered" -> bags are formed by sorting on a noisy random
                          projection of X, which mimics bags defined by a
                          covariate (site, plot, ...) and gives more varied
                          proportions.
    noise: scale of Laplace noise added to the proportions (noisy measurements),
           the result is clipped to [0, 1].

    Returns (bag_index (n_bags, bag_size), proportions (n_bags,)).
    """
    n = len(y)
    if bag_size < 1 or bag_size > n:
        raise ValueError("bag_size must be in [1, n_samples]")
    if mode == "random":
        order = rng.permutation(n)
    elif mode == "clustered":
        w = rng.normal(size=X.shape[1])
        score = X @ w + rng.normal(scale=X.std() * 0.5, size=n)
        order = np.argsort(score)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    n_bags = n // bag_size
    idx = order[: n_bags * bag_size].reshape(n_bags, bag_size)
    props = y[idx].mean(axis=1)
    if noise > 0:
        props = np.clip(props + rng.laplace(0.0, noise, size=props.shape), 0.0, 1.0)
    return idx, props.astype(np.float32)
