"""Tests for the bag-level losses."""
# pylint: disable=missing-function-docstring,wrong-import-position,import-error
import pytest

torch = pytest.importorskip("torch")
from llp.losses import bag_probability, proportion_bce, proportion_mse
from llp.model import MLP


def test_bag_probability_is_mean_of_sigmoids():
    logits = torch.tensor([[0.0, 0.0], [10.0, -10.0]])
    p = bag_probability(logits)
    assert torch.allclose(p, torch.tensor([0.5, 0.5]), atol=1e-4)


def test_mse_zero_when_prediction_matches_proportion():
    logits = torch.zeros(3, 4)                           # sigmoid = 0.5 everywhere
    assert proportion_mse(logits, torch.full((3,), 0.5)).item() == pytest.approx(0.0, abs=1e-8)


def test_bce_minimal_at_true_proportion():
    props = torch.tensor([0.25])
    good = torch.log(torch.tensor(1 / 3)) * torch.ones(1, 4)   # sigmoid = 0.25
    bad = torch.zeros(1, 4)                                    # sigmoid = 0.5
    assert proportion_bce(good, props) < proportion_bce(bad, props)


def test_gradient_flows_to_model():
    model = MLP(5)
    x = torch.randn(6, 4, 5)
    loss = proportion_bce(model(x.reshape(-1, 5)).reshape(6, 4), torch.rand(6))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
