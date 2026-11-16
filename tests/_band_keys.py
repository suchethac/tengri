# SPDX-License-Identifier: BSD-3-Clause
"""Cache-key helpers: the keys take a ``BandSpec``; these build one from the loose arguments."""

from __future__ import annotations

from tengri.components.igm import _subband_cache
from tengri.components.stellar.sps import precompute as _pc
from tengri.utils.band_projection import BandSpec


def spec_of(filter_waves, filter_trans, convention, n_subbands=0, lyc_gate=False) -> BandSpec:
    """A :class:`BandSpec` from the loose filter arguments the keys used to take."""
    return BandSpec(
        filters=(list(filter_waves), list(filter_trans)),
        convention=convention,
        n_subbands=n_subbands,
        lyc_gate=lyc_gate,
    )


def ztable_key(
    ssp_data,
    filter_waves,
    filter_trans,
    z_grid,
    apply_igm,
    taylor_correction,
    convention,
    n_subbands=0,
    lyc_gate=False,
) -> str:
    """``_ztable_cache_key`` from the loose arguments."""
    return _pc._ztable_cache_key(
        ssp_data,
        spec_of(filter_waves, filter_trans, convention, n_subbands, lyc_gate),
        z_grid,
        apply_igm,
        taylor_correction,
    )


def band_key(wave_rest, filter_waves, filter_trans, z_grid, igm_model, convention) -> str:
    """``band_factor_key`` from the loose arguments."""
    return _subband_cache.band_factor_key(
        wave_rest, spec_of(filter_waves, filter_trans, convention), z_grid, igm_model
    )
