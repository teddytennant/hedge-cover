"""Selector over a finite function class (Definition 2).

The paper's selector ranges over a well-ordering of an infinite class.
Here the well-ordering is list order on a finite list of callables.
Callables are ordinary Python functions; they need not be JAX-transformable.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from hedge_cover.grid import project

_MATCH_TOL = 1e-8


def _zero(_x):
    return 0.0


def selector(F, trajectory, grid_values):
    """First ``f`` in ``F`` consistent with ``trajectory``, else the zero function.

    ``f`` is consistent when ``project(f(x_i), grid_values)`` matches ``y_i``
    within ``1e-8`` for every pair in the trajectory. An empty trajectory is
    consistent with every function, so the first element of ``F`` is returned.
    An empty ``F`` yields the zero function, which always returns ``0.0``.
    """
    if not F:
        return _zero
    if not trajectory:
        return F[0]
    for candidate in F:
        if _consistent(candidate, trajectory, grid_values):
            return candidate
    return _zero


def _consistent(candidate, trajectory, grid_values) -> bool:
    for x_i, y_i in trajectory:
        projected = project(candidate(x_i), grid_values)
        if abs(float(projected) - float(y_i)) > _MATCH_TOL:
            return False
    return True
