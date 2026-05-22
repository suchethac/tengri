# SPDX-License-Identifier: BSD-3-Clause
"""Cosmology helpers (Planck18 default; differentiable in JAX).

This namespace re-exports the cosmology utilities that were previously
only reachable as ``tengri.utils.cosmology.*``. The functions here are
pure JAX (JIT/grad/vmap-safe) and accept either a :class:`CosmoParams`
instance or rely on the :data:`DEFAULT_COSMO` (Planck18) module default.

Examples
--------
>>> from tengri import cosmology as cosmo
>>> cosmo.luminosity_distance_mpc(0.5)
DeviceArray(2867.4..., dtype=float64)
>>> cosmo.PLANCK18
CosmoParams(Om0=0.315, w0=-1.0, wa=0.0, h=0.674)
"""

from __future__ import annotations

from tengri.utils.cosmology import (
    DEFAULT_COSMO,
    PLANCK18,
    CosmoParams,
    age_at_z,
    age_at_z0,
    angular_diameter_distance,
    angular_diameter_distance_mpc,
    arcsec_per_kpc,
    comoving_distance,
    comoving_distance_mpc,
    comoving_volume_element,
    distance_modulus,
    kpc_per_arcsec,
    lookback_time,
    luminosity_distance,
    luminosity_distance_mpc,
    z_at_cosmic_time,
    z_at_lookback_time,
)

__all__ = [
    "DEFAULT_COSMO",
    "PLANCK18",
    "CosmoParams",
    "age_at_z",
    "age_at_z0",
    "angular_diameter_distance",
    "angular_diameter_distance_mpc",
    "arcsec_per_kpc",
    "comoving_distance",
    "comoving_distance_mpc",
    "comoving_volume_element",
    "distance_modulus",
    "kpc_per_arcsec",
    "lookback_time",
    "luminosity_distance",
    "luminosity_distance_mpc",
    "z_at_cosmic_time",
    "z_at_lookback_time",
]
