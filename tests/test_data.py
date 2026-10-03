"""Tests for data generation and bag construction."""
# pylint: disable=missing-function-docstring
import numpy as np
import pytest

from llp.data import make_bags, make_dataset


def test_bags_are_disjoint_and_proportions_correct():
    x, y, _, _ = make_dataset(n_samples=2000, seed=1)
    rng = np.random.default_rng(0)
    idx, props = make_bags(x, y, 10, rng)
    assert idx.shape == (len(y) // 10, 10)
    assert len(np.unique(idx)) == idx.size               # disjoint bags
    np.testing.assert_allclose(props, y[idx].mean(axis=1), rtol=1e-6)


@pytest.mark.parametrize("mode", ["random", "clustered"])
def test_noise_keeps_proportions_in_unit_interval(mode):
    x, y, _, _ = make_dataset(n_samples=2000, seed=2)
    _, props = make_bags(x, y, 8, np.random.default_rng(1), mode=mode, noise=0.3)
    assert props.min() >= 0.0 and props.max() <= 1.0


def test_invalid_bag_size():
    x, y, _, _ = make_dataset(n_samples=1000)
    with pytest.raises(ValueError):
        make_bags(x, y, 0, np.random.default_rng(0))
