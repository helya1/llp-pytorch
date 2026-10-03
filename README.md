# Learning from Label Proportions with PyTorch

A small, tested PyTorch implementation of **Learning from Label Proportions (LLP)**: training an instance-level binary classifier when individual labels are never observed, and only the **mean of the labels of each bag** is known.

> Author: Helya Amiri (M.Sc. Statistics, University of Strasbourg). Personal project built to explore weakly supervised learning.

## Problem

We want a predictor `Phi : R^d -> [0, 1]` of a binary variable `Y` from `X`, but the training set is made of disjoint bags `S_1, ..., S_p`. For each bag we only observe

- the features `x_i` of every element of the bag,
- the label proportion `m_j = (1/|S_j|) * sum_{i in S_j} y_i`.

The individual `y_i` stay hidden (setting of Quadrianto et al., JMLR 2009).

## Method

With `p_i = sigmoid(Phi(x_i))`, the bag-level prediction is the average `p_bar_j = mean_{i in S_j} p_i`. The model is trained by minimising a loss between `p_bar_j` and `m_j`:

- `proportion_bce`: cross-entropy `-(m log p_bar + (1-m) log(1-p_bar))` (DLLP-style loss),
- `proportion_mse`: squared error `(p_bar - m)^2`.

Evaluation is done at the **instance level** on a held-out set with true labels. An MLP trained with the individual labels is provided as a supervised reference.

Two extensions are included to mimic realistic settings:

- **Bag construction**: `random` bags, or `clustered` bags (formed from a noisy projection of `X`, like bags defined by a site or a plot). Large random bags carry little signal because their proportions all concentrate around the global rate.
- **Noisy proportions**: Laplace noise on `m_j` (`--noise`), a simple stand-in for imperfect measurements, as studied in the mutual-contamination / label-noise frameworks (Scott & Zhang 2020; Zhang, Wang & Scott 2022).

## Repository layout

```
llp/
  data.py        synthetic data, bag construction (numpy / scikit-learn)
  model.py       MLP instance-level predictor
  losses.py      bag-level losses
  train.py       LLP training loop and supervised reference
  evaluate.py    accuracy and AUC at the instance level
experiments/
  run_experiments.py   accuracy vs. bag size, several seeds, CSV + figure
tests/                 unit tests (pytest)
results/               outputs of the experiments
```

## Quick start

```bash
git clone https://github.com/helya1/llp-pytorch.git
cd llp-pytorch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                                            # run the tests
python experiments/run_experiments.py --bag-sizes 2 8 32 128 --seeds 5
```

Outputs: `results/results.csv` and `results/accuracy_vs_bag_size.png`.

Useful options: `--mode {random,clustered}`, `--noise 0.1`, `--loss {bce,mse}`, `--epochs 60`.

## Results

Run the command above and paste your table and figure here, for example:

![accuracy vs bag size](results/accuracy_vs_bag_size.png)

| method | bag size | accuracy (mean ± std) |
|---|---|---|
| supervised | 1 | *to fill* |
| LLP | 2 / 8 / 32 / 128 | *to fill* |

## Possible extensions

- Study the effect of proportion noise and compare `bce` vs `mse` losses.
- Variable-size bags (padding / masking) instead of equal-size bags.
- Application to real data (for example signal-like data) in place of the synthetic dataset.

## References

1. N. Quadrianto, A. J. Smola, T. S. Caetano, Q. V. Le. *Estimating labels from label proportions.* JMLR 10, 2009.
2. G. Ardehaly, A. Culotta. *Co-training for demographic classification using deep learning from label proportions.* 2017 (DLLP).
3. C. Scott, J. Zhang. *Learning from label proportions: a mutual contamination framework.* NeurIPS 2020.
4. J. Zhang, Y. Wang, C. Scott. *Learning from label proportions by learning with label noise.* NeurIPS 2022.

## License

MIT
