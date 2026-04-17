"""Variational inference backends: native JAX, NIFTy library integration."""

from tengri.inference.backends.vi.native import run_native_vi
from tengri.inference.backends.vi.nifty import run_nifty_fast_vi, run_nifty_vi

__all__ = [
    "run_native_vi",
    "run_nifty_fast_vi",
    "run_nifty_vi",
]
