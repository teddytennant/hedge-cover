"""Hedge over selector-consistent hypotheses (Algorithms 1 to 3).

A hypothesis assigns each time ``t = 0, ..., T-1`` either to the selector
(``None``) or to a grid index in ``0 .. m-1``. There are ``(m+1)^T``
hypotheses. The prior is the Algorithm 1 weight that makes the Lemma C.1
normalizer bound hold:

    w(h) = 1 / ((|Th| + 1)^2 * (m + 1)^|Th| * binom(T, |Th|))

with ``p0 = w / sum(w)``. ``|Th|`` counts entries that are not ``None``.
"""

from __future__ import annotations

import itertools
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hedge_cover.experts import selector
from hedge_cover.grid import grid, project

_MAX_HYPOTHESES = 20000


def enumerate_hypotheses(T: int, m: int) -> list[tuple]:
    """Hypotheses as length-``T`` tuples, lexicographic with ``None`` first.

    Each entry is ``None`` (use the selector) or an int in ``0 .. m-1``
    (predict that grid value).
    """
    symbols = [None, *range(m)]
    return list(itertools.product(symbols, repeat=T))


def _support_size(hypothesis) -> int:
    return sum(entry is not None for entry in hypothesis)


def _unnormalized_weight(hypothesis, T: int, m: int) -> float:
    k = _support_size(hypothesis)
    return 1.0 / ((k + 1) ** 2 * (m + 1) ** k * math.comb(T, k))


def prior(T: int, m: int) -> dict:
    """Normalized Algorithm 1 prior, keyed by hypothesis tuple."""
    hypotheses = enumerate_hypotheses(T, m)
    weights = [_unnormalized_weight(h, T, m) for h in hypotheses]
    total = math.fsum(weights)
    return {h: weight / total for h, weight in zip(hypotheses, weights)}


def hedge_cover(F, T, xs, ys, m=None, eta=0.09) -> dict:
    """Run Hedge-Cover for ``T`` rounds on a finite function class.

    ``xs`` is a list of ``T`` 1d arrays and ``ys`` a list of ``T`` labels in
    ``[-1, 1]``. If ``m`` is omitted it defaults to ``2T``, as in the paper.
    ``eta`` defaults to ``0.09``, inside both the lemma range ``eta <= 1/8``
    and the theorem range ``eta < 1/10``.

    Raises ``ValueError`` when ``(m+1)^T`` exceeds 20000.

    Expert predictions are always grid values. A hypothesis that consults the
    selector stores its own prediction, not ``y_t``, on its trajectory.
    """
    if m is None:
        m = 2 * T
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError("T must be a nonnegative int")
    if not isinstance(m, int) or isinstance(m, bool) or m <= 0:
        raise ValueError("m must be a positive int")
    n_hypotheses = (m + 1) ** T
    if n_hypotheses > _MAX_HYPOTHESES:
        raise ValueError(
            f"(m+1)^T = {n_hypotheses} exceeds the enumeration limit of 20000"
        )

    grid_values = grid(m)
    hypotheses = enumerate_hypotheses(T, m)
    p0 = prior(T, m)
    p0_arr = jnp.asarray([p0[h] for h in hypotheses], dtype=jnp.float64)

    trajectories: list[list] = [[] for _ in hypotheses]
    losses = jnp.zeros((len(hypotheses),), dtype=jnp.float64)
    prediction_rows = []
    expert_rows = []

    for t in range(T):
        x_t = xs[t]
        y_hats = []
        for index, hypothesis in enumerate(hypotheses):
            choice = hypothesis[t]
            if choice is not None:
                y_hat_h = grid_values[choice]
            else:
                chosen = selector(F, trajectories[index], grid_values)
                y_hat_h = project(chosen(x_t), grid_values)
            y_hats.append(jnp.asarray(y_hat_h, dtype=jnp.float64))
        y_hats = jnp.stack(y_hats)

        # p_h proportional to p0_h * exp(-eta * L_h), L_h from previous rounds.
        log_weights = jnp.log(p0_arr) - eta * losses
        log_weights = log_weights - jnp.max(log_weights)
        probs = jnp.exp(log_weights)
        probs = probs / jnp.sum(probs)
        y_hat = jnp.sum(probs * y_hats)

        prediction_rows.append(y_hat)
        expert_rows.append(y_hats)

        y_t = jnp.asarray(ys[t], dtype=jnp.float64)
        losses = losses + (y_t - y_hats) ** 2
        for index, y_hat_h in enumerate(y_hats):
            trajectories[index].append((x_t, float(y_hat_h)))

    if prediction_rows:
        predictions = jnp.stack(prediction_rows)
        expert_predictions = jnp.stack(expert_rows)
    else:
        predictions = jnp.zeros((0,), dtype=jnp.float64)
        expert_predictions = jnp.zeros((0, len(hypotheses)), dtype=jnp.float64)

    ys_arr = jnp.asarray(list(ys), dtype=jnp.float64)
    learner_loss = jnp.sum((predictions - ys_arr) ** 2)

    log_weights = jnp.log(p0_arr) - eta * losses
    log_weights = log_weights - jnp.max(log_weights)
    final = jnp.exp(log_weights)
    final = final / jnp.sum(final)
    final_weights = {h: float(final[i]) for i, h in enumerate(hypotheses)}

    return {
        "predictions": predictions,
        "losses": losses,
        "prior": p0,
        "final_weights": final_weights,
        "learner_loss": learner_loss,
        "expert_predictions": expert_predictions,
    }
