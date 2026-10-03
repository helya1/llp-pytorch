"""Accuracy of LLP as a function of the bag size, versus a supervised reference.

Usage:  python experiments/run_experiments.py --bag-sizes 2 8 32 128 --seeds 5
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llp import MLP, evaluate, make_bags, make_dataset, train_llp, train_supervised  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-sizes", type=int, nargs="+", default=[2, 8, 32, 128])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--mode", choices=["random", "clustered"], default="clustered")
    ap.add_argument("--noise", type=float, default=0.0, help="Laplace noise scale on proportions")
    ap.add_argument("--loss", choices=["bce", "mse"], default="bce")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for seed in range(args.seeds):
        X_tr, y_tr, X_te, y_te = make_dataset(seed=seed)
        sup = MLP(X_tr.shape[1])
        train_supervised(sup, X_tr, y_tr, seed=seed)
        r = evaluate(sup, X_te, y_te)
        rows.append({"method": "supervised", "bag_size": 1, "seed": seed, **r})
        for k in args.bag_sizes:
            rng = np.random.default_rng(seed)
            idx, props = make_bags(X_tr, y_tr, k, rng, mode=args.mode, noise=args.noise)
            model = MLP(X_tr.shape[1])
            train_llp(model, X_tr, idx, props, epochs=args.epochs, loss=args.loss, seed=seed)
            r = evaluate(model, X_te, y_te)
            rows.append({"method": "LLP", "bag_size": k, "seed": seed, **r})
            print(f"seed={seed} bag_size={k:4d}  acc={r['accuracy']:.3f}  auc={r['auc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    summary = df.groupby(["method", "bag_size"])[["accuracy", "auc"]].agg(["mean", "std"]).round(3)
    print(summary)

    llp = df[df.method == "LLP"].groupby("bag_size")["accuracy"].agg(["mean", "std"]).fillna(0)
    sup_mean = df[df.method == "supervised"]["accuracy"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(llp.index, llp["mean"], yerr=llp["std"], marker="o", capsize=3, label="LLP")
    ax.axhline(sup_mean, color="gray", ls="--", label="supervised (individual labels)")
    ax.axhline(0.5, color="red", ls=":", label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bag size")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"LLP vs bag size (bags: {args.mode}, noise: {args.noise})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "accuracy_vs_bag_size.png"), dpi=150)


if __name__ == "__main__":
    main()
