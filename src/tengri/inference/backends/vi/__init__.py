# SPDX-License-Identifier: BSD-3-Clause
"""Variational inference backends: native JAX, NIFTy library integration."""

from tengri.inference.backends.vi.gaussian import run_gaussian_vi, run_gaussian_vi_fitter
from tengri.inference.backends.vi.native import run_native_vi
from tengri.inference.backends.vi.nifty import run_nifty_fast_vi, run_nifty_vi

__all__ = [
    "run_gaussian_vi",
    "run_gaussian_vi_fitter",
    "run_native_vi",
    "run_nifty_fast_vi",
    "run_nifty_vi",
]
