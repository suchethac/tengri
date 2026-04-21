"""MCMC backends: NUTS, HMC, Dynamic HMC, GHMC, MCLMC, Ray Tracing, Elliptical Slice.

This module re-exports samplers from their individual backend modules
to maintain backward compatibility with the public API.

Internal infrastructure (shared scan functions, kernel getters, etc.)
is in _shared.py.
"""

from __future__ import annotations

from tengri.inference.backends.mcmc.dynamic_hmc import run_dynamic_hmc
from tengri.inference.backends.mcmc.elliptical_slice import (
    run_elliptical_slice_fitter as run_elliptical_slice,
)
from tengri.inference.backends.mcmc.ghmc import run_ghmc
from tengri.inference.backends.mcmc.hmc import run_hmc
from tengri.inference.backends.mcmc.mclmc import run_adjusted_mclmc, run_mclmc
from tengri.inference.backends.mcmc.nuts import run_nuts
from tengri.inference.backends.mcmc.raytrace import run_raytrace

__all__ = [
    "run_adjusted_mclmc",
    "run_dynamic_hmc",
    "run_elliptical_slice",
    "run_ghmc",
    "run_hmc",
    "run_mclmc",
    "run_nuts",
    "run_raytrace",
]
