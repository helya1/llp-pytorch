"""Accuracy of LLP as a function of the bag size, versus a supervised reference.

Usage (from the repository root):
    python -m experiments.run_experiments --bag-sizes 2 8 32 128 --seeds 5
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

from llp import MLP, evaluate, make_bags, make_dataset, train_llp, train_supervised

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag-sizes", type=int, nargs="+", default=[2, 8, 32, 128])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--mode", choices=["random", "clustered"], default="clustered")
    ap.add_argument("--noise", type=float, default=0.0, help="Laplace noise scale on proportions")
    ap.add_argument("--loss", choices=["bce", "mse"], default="bce")
    ap.add_argument("--out", default="results")
    return ap.parse_args()


def run_seed(seed: int, args: argparse.Namespace) -> list[dict]:
    """Train the supervised reference and one LLP model per bag size for one seed."""
    x_train, y_train, x_test, y_test = make_dataset(seed=seed)
    rows = []
    reference = MLP(x_train.shape[1])
    train_supervised(reference, x_train, y_train, seed=seed)
    rows.append({"method": "supervised", "bag_size": 1, "seed": seed,
                 **evaluate(reference, x_test, y_test)})
    for bag_size in args.bag_sizes:
        rng = np.random.default_rng(seed)
        idx, props = make_bags(x_train, y_train, bag_size, rng, mode=args.mode, noise=args.noise)
        model = MLP(x_train.shape[1])
        train_llp(model, x_train, idx, props, epochs=args.epochs, loss=args.loss, seed=seed)
        res = evaluate(model, x_test, y_test)
        rows.append({"method": "LLP", "bag_size": bag_size, "seed": seed, **res})
        print(f"seed={seed} bag_size={bag_size:4d}  "
              f"acc={res['accuracy']:.3f}  auc={res['auc']:.3f}")
    return rows


def plot_results(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Accuracy versus bag size, with the supervised reference and chance level."""
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


def main() -> None:
    """Run all experiments, save the CSV table and the figure."""
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        rows += run_seed(seed, args)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    print(df.groupby(["method", "bag_size"])[["accuracy", "auc"]].agg(["mean", "std"]).round(3))
    plot_results(df, args)


if __name__ == "__main__":
    main()
