"""Grid indexing and nearest-point projection."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp

from hedge_cover import grid, project


def test_grid_four_points():
    values = grid(4)
    expected = jnp.asarray([-1.0, -0.5, 0.0, 0.5])
    assert values.shape == (4,)
    assert jnp.allclose(values, expected, atol=1e-12, rtol=0.0)


def test_project_nearest_and_tie():
    values = grid(4)
    assert float(project(0.2, values)) == 0.0
    assert float(project(0.4, values)) == 0.5

    # Midpoint of index 1 (-0.5) and index 2 (0.0). Smaller index wins.
    midpoint = 0.5 * (float(values[1]) + float(values[2]))
    assert midpoint == -0.25
    assert abs(float(values[1]) - midpoint) == abs(float(values[2]) - midpoint)
    assert float(project(midpoint, values)) == float(values[1])
    assert float(project(-0.25, values)) == -0.5
