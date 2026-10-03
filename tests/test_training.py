"""End-to-end test of LLP training."""
# pylint: disable=missing-function-docstring,wrong-import-position,import-error
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from llp import MLP, evaluate, make_bags, make_dataset, train_llp


def test_llp_beats_chance_with_small_bags():
    x_train, y_train, x_test, y_test = make_dataset(n_samples=6000, seed=0)
    idx, props = make_bags(x_train, y_train, 4, np.random.default_rng(0))
    model = MLP(x_train.shape[1])
    hist = train_llp(model, x_train, idx, props, epochs=20, seed=0, device=torch.device("cpu"))
    assert hist[-1] < hist[0]                            # the loss decreases
    assert evaluate(model, x_test, y_test)["accuracy"] > 0.65
