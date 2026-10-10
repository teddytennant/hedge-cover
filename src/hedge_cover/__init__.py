"""Hedge-Cover: smoothed online regression over a finite function class."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

from hedge_cover.experts import selector
from hedge_cover.grid import grid, project
from hedge_cover.hedge import enumerate_hypotheses, hedge_cover, prior

__all__ = [
    "enumerate_hypotheses",
    "grid",
    "hedge_cover",
    "prior",
    "project",
    "selector",
]
