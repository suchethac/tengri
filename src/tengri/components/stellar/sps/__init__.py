# SPDX-License-Identifier: BSD-3-Clause
"""Stellar population synthesis: SSP grids, IMF, and CSP integration."""

from .csp import build_csp_weights
from .dsps_wrapper import (
    SSPData,
    effective_metallicity,
    list_available_ssps,
    list_isochrone_libraries,
    list_spectral_libraries,
    load_ssp,
    mass_remaining_fraction,
    resolve_ssp_path,
)

__all__ = [
    "SSPData",
    "build_csp_weights",
    "effective_metallicity",
    "list_available_ssps",
    "list_isochrone_libraries",
    "list_spectral_libraries",
    "load_ssp",
    "mass_remaining_fraction",
    "resolve_ssp_path",
]


def __getattr__(name):
    """Lazily expose private SSP helpers without polluting the namespace."""
    import warnings

    if name == "_PRECOMPUTED_SSP_DTYPE":
        warnings.warn(
            "_PRECOMPUTED_SSP_DTYPE is private; import from "
            "tengri.components.stellar.sps.dsps_wrapper if you really need it.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .dsps_wrapper import _PRECOMPUTED_SSP_DTYPE

        return _PRECOMPUTED_SSP_DTYPE
    raise AttributeError(name)
