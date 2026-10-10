"""Selector, pathwise Hedge bound, and realizable weight movement."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import pytest

from hedge_cover import (
    enumerate_hypotheses,
    grid,
    hedge_cover,
    project,
    selector,
)


def test_selector_list_order_and_zero():
    x = jnp.asarray([0.0])
    values = grid(2)
    assert jnp.allclose(values, jnp.asarray([-1.0, 0.0]))

    # project(1) lands on 0, so a function that returns 1 matches label 0.
    neg = lambda x: -1.0
    pos = lambda x: 1.0
    assert float(project(pos(x), values)) == 0.0
    assert selector([neg, pos], [(x, -1.0)], values) is neg
    assert selector([neg, pos], [(x, 0.0)], values) is pos

    # Both functions project to -1. List order picks the first on label -1.
    # The other grid point, 0, matches neither, so the zero function is returned.
    f0 = lambda x: -1.0
    f1 = lambda x: -1.0
    assert float(project(f0(x), values)) == -1.0
    assert float(project(f1(x), values)) == -1.0
    assert selector([f0, f1], [(x, -1.0)], values) is f0

    unmatched = float(values[1])
    assert unmatched == 0.0
    zero = selector([f0, f1], [(x, unmatched)], values)
    assert zero is not f0 and zero is not f1
    assert zero(jnp.asarray([3.0])) == 0.0
    assert float(project(zero(x), values)) == 0.0

    assert selector([f0, f1], [], values) is f0
    assert selector([], [(x, -1.0)], values)(x) == 0.0


def test_lemma_c1_pathwise_bound():
    T = 3
    m = 2
    eta = 0.09
    F = [lambda x: -1.0, lambda x: 0.0, lambda x: 1.0]
    xs = [jnp.asarray([0.0])] * T
    ys = [0.2, -0.4, 0.7]

    result = hedge_cover(F, T, xs, ys, m=m, eta=eta)
    hypotheses = enumerate_hypotheses(T, m)
    assert len(hypotheses) == 27
    expert_predictions = result["expert_predictions"]
    assert expert_predictions.shape == (T, len(hypotheses))

    ys_arr = jnp.asarray(ys, dtype=jnp.float64)
    expert_losses = jnp.sum((expert_predictions - ys_arr[:, None]) ** 2, axis=0)
    assert jnp.allclose(expert_losses, result["losses"], atol=1e-12)

    learner_loss = float(result["learner_loss"])
    recomputed = float(jnp.sum((result["predictions"] - ys_arr) ** 2))
    assert abs(learner_loss - recomputed) <= 1e-12

    for index, hypothesis in enumerate(hypotheses):
        p0_h = result["prior"][hypothesis]
        bound = (1.0 / eta) * math.log(1.0 / p0_h)
        gap = learner_loss - float(expert_losses[index])
        assert gap <= bound + 1e-5, (hypothesis, gap, bound)


def test_realizable_weights_move():
    T = 3
    m = 2
    F = [lambda x: -1.0, lambda x: 0.0, lambda x: 1.0]
    xs = [jnp.asarray([0.0])] * T
    ys = [0.0, 0.0, 0.0]
    result = hedge_cover(F, T, xs, ys, m=m, eta=0.09)

    predictions = result["predictions"]
    assert predictions.shape == (T,)
    assert bool(jnp.all(jnp.isfinite(predictions)))
    assert bool(jnp.all(predictions >= -1.0))
    assert bool(jnp.all(predictions <= 1.0))

    values = grid(m)
    nearest_index = int(jnp.argmin(jnp.abs(values - 0.0)))
    assert float(values[nearest_index]) == 0.0
    always_nearest = tuple([nearest_index] * T)
    assert result["final_weights"][always_nearest] > result["prior"][always_nearest]

    # Empty-support hypothesis consults the selector. Its trajectory stores
    # its own prediction, so it locks onto the first function (-1), not y_t.
    empty = tuple([None] * T)
    empty_index = enumerate_hypotheses(T, m).index(empty)
    empty_preds = result["expert_predictions"][:, empty_index]
    assert jnp.allclose(empty_preds, jnp.asarray([-1.0, -1.0, -1.0]))


def test_enumeration_cap():
    T = 5
    xs = [jnp.asarray([0.0])] * T
    ys = [0.0] * T
    with pytest.raises(ValueError, match="20000"):
        hedge_cover([lambda x: 0.0], T, xs, ys)
