"""Covering grid for Hedge-Cover predictions.

The paper writes ``N_m = {-1 + 2j/m : j in [m]}`` with ``|N_m| = m``.
The index set ``[m]`` is ambiguous. This package uses ``j = 0, ..., m-1``,
which has cardinality ``m``, includes the left endpoint ``-1``, and ends at
``1 - 2/m`` (it does not include ``+1``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def grid(m: int) -> jnp.ndarray:
    """Return ``-1 + 2j/m`` for ``j = 0, ..., m-1``, shape ``(m,)``."""
    if not isinstance(m, int) or isinstance(m, bool) or m <= 0:
        raise ValueError("m must be a positive int")
    indices = jnp.arange(m, dtype=jnp.float64)
    return -1.0 + 2.0 * indices / float(m)


def project(y, grid):
    """Nearest value on ``grid``. Ties go to the smaller index."""
    values = jnp.asarray(grid, dtype=jnp.float64)
    point = jnp.asarray(y, dtype=jnp.float64)
    index = jnp.argmin(jnp.abs(values - point))
    return values[index]
