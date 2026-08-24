# SPDX-License-Identifier: BSD-3-Clause
r"""Broad-line region (BLR) blocks for the composable AGN pipeline.

One file, every BLR option: pick via ``agn={'blr': {'type': ...}}``.
BLR is maskable (Type-1/2). Consolidated 2026-07; registration unchanged.

NAME NOTE: this composable-*block* module shadows the physics kernel
``tengri.components.agn.blr`` one package up: always import by full path,
never a bare ``blr``.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blr import compute_blr_sed
from tengri.components.agn.nlr_cloudy import (
    compute_blr_sed_synthesizer,
    compute_nlr_sed_synthesizer_spectra,
)
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG

__all__ = [
    "blr_analytic_block",
    "blr_synthesizer_block",
    "blr_synthesizer_spectra_block",
]

from tengri.utils.physics_constants import C_AA as _C_AA_PER_S

DEFAULT_F_BOL_5100: float = 9.0


def _resolve_synthesizer_grid(kind: str) -> str:
    """Resolve a Synthesizer AGN grid path for the grid-backed line blocks.

    Searches ``$TENGRI_SYNTHESIZER_AGN_GRID_DIR`` then the repo-default
    ``data/synthesizer_grids/``. ``kind`` is ``"nlr"`` or ``"blr"``.

    These grids are not packaged with tengri (they ship via
    ``synthesizer-download --agn-test-grids``), so a clear error is raised if
    neither location holds ``test_grid_agn-<kind>.hdf5``.
    """
    fname = f"test_grid_agn-{kind}.hdf5"
    candidates = []
    env = os.environ.get("TENGRI_SYNTHESIZER_AGN_GRID_DIR")
    if env:
        candidates.append(Path(env) / fname)
    candidates.append(Path("data/synthesizer_grids") / fname)
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Synthesizer AGN {kind.upper()} grid '{fname}' not found. Set "
        "$TENGRI_SYNTHESIZER_AGN_GRID_DIR or place it in data/synthesizer_grids/ "
        "(fetch via `synthesizer-download --agn-test-grids`)."
    )


@register_agn_block(
    "blr",
    "analytic",
    citation="Krawczyk et al. 2013, ApJS, 206, 4",
    status="production",
    short_doc="Analytic broad-line region with Gaussian broadening",
)
def blr_analytic_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_blr_cf: float = 0.1,
    agn_blr_fwhm_kms: float = 5000.0,
    agn_fe2_strength: float = 0.0,
    agn_blr_line_efficiency: float = 0.08,
    agn_blr_f_bol: float = DEFAULT_F_BOL_5100,
    **_params,
) -> Array:
    r"""Broad-line region emission as a blr-stage block.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
            Rest-frame wavelength [Å].
    agn_log_lbol: float
            Ignored (kept for protocol compatibility: ``l5100_disc`` provides
            the normalization).
    l5100_disc: array, scalar
            :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_blr_cf: float, optional
            BLR covering fraction [0, 1]. Default ``0.1``.
    agn_blr_fwhm_kms: float, optional
            Broad-line FWHM [km/s]. Default ``5000``.
    agn_fe2_strength: float, optional
            :math:`R_{\rm Fe} = F({\rm FeII})/F({\rm H}\beta)`. Default ``0.0``
            (FeII pseudo-continuum disabled).
    agn_blr_line_efficiency: float, optional
            Fraction of intercepted luminosity converted to lines. Default ``0.08``.
    agn_blr_f_bol: float, optional
            Bolometric correction :math:`L_{\rm bol}/\lambda L_\lambda(5100\,\mathrm{\AA})`.
            Default :data:`DEFAULT_F_BOL_5100`.

    Returns
    -------
    L_lambda: ndarray, shape (n_wave,)
            BLR :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    Geometric masking by the torus is **not** applied here. If the
    composable recipe also activates a torus block, double-counting is
    possible for line-of-sight inclination effects; see Section 2 of
    :mod:`tengri.components.agn.unified` for the mask convention.
    """
    del agn_log_lbol  # normalization comes from l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = jnp.asarray(l5100_disc) * agn_blr_f_bol
    L_nu = compute_blr_sed(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_blr_cf,
        fwhm_kms=agn_blr_fwhm_kms,
        agn_fe2_strength=agn_fe2_strength,
        line_efficiency=agn_blr_line_efficiency,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block(
    "blr",
    "synthesizer",
    citation="Lovell et al. 2025 (Open J. Astrophys.); Roper et al. 2026 (JOSS): Synthesizer",
    status="production",
    short_doc="Synthesizer Cloudy grid broad-line region",
)
def blr_synthesizer_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_blr_cf: float = 0.1,
    agn_blr_fwhm_kms: float = 5000.0,
    agn_blr_f_bol: float = DEFAULT_F_BOL_5100,
    agn_blr_logU: float = -1.0,
    agn_blr_logZ: float = -1.8477,
    **_params,
) -> Array:
    r"""BLR lines from the Synthesizer Cloudy grid (grid-backed blr block).

    Broad-line sibling of :func:`nlr_synthesizer_block`, routing to
    :func:`compute_blr_sed_synthesizer` (default FWHM 5000 km/s). Grid path is
    resolved the same way. See that function's Notes for the JIT caveat.

    The photoionization axes are named ``agn_blr_logU/logZ`` (not the galaxy
    ``neb_*`` names) so they survive the AGN component's ``agn_``-prefix filter
    and are drivable through ``SEDModel.build`` (#931); they translate to the
    grid's ``neb_*`` axes internally.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
            Rest-frame wavelength [Å].
    agn_log_lbol: float
            Ignored (bolometric is taken from ``l5100_disc``).
    l5100_disc: array, scalar
            :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_blr_cf, agn_blr_fwhm_kms, agn_blr_f_bol: float
            Covering fraction, broad-line FWHM [km/s], and bolometric correction.
    agn_blr_logU, agn_blr_logZ: float
            Photoionization knobs forwarded to the grid adapter.

    Returns
    -------
    L_lambda: ndarray, shape (n_wave,)
            BLR :math:`L_\lambda` [erg/s/Å].
    """
    del agn_log_lbol
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = jnp.asarray(l5100_disc) * agn_blr_f_bol
    L_nu = compute_blr_sed_synthesizer(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_blr_cf,
        fwhm_kms=agn_blr_fwhm_kms,
        grid_path=_resolve_synthesizer_grid("blr"),
        neb_logU=agn_blr_logU,
        neb_logZ_gas=agn_blr_logZ,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block(
    "blr",
    "synthesizer_spectra",
    citation="Lovell et al. 2025 (Open J. Astrophys.); Roper et al. 2026 (JOSS): Synthesizer",
    status="production",
    short_doc="Synthesizer UnifiedAGN BLR reprocessed nebular spectrum",
)
def blr_synthesizer_spectra_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_blr_cf: float = 0.1,
    agn_blr_logU: float = -1.0,
    agn_blr_logn: float = 4.0,
    agn_blr_logZ: float = -2.0,
    **_params,
) -> Array:
    r"""BLR reprocessed nebular spectrum reproducing Synthesizer's UnifiedAGN.

    Broad-line sibling of :func:`nlr_synthesizer_spectra_block`, reading the
    BLR grid's ``/spectra/nebular`` array. Synthesizer extracts both line regions
    isotropically (grid ``cosine_inclination=0.5``), so this is returned on the
    isotropic channel to reproduce ``UnifiedAGN``'s ``blr`` component (issue #694),
    the physically Type-2-obscured BLR is the ``blr``/``blr_synthesizer`` path.

    The photoionization axes are named ``agn_blr_logU/logn/logZ`` (not the galaxy
    ``neb_*`` names) so they survive the AGN component's ``agn_``-prefix filter
    and are drivable through ``SEDModel.build`` (#931); they translate to the
    grid's ``neb_*`` axes internally.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
            Rest-frame wavelength [Å].
    agn_log_lbol: float
            log10(L_bol / L_sun) of the AGN.
    l5100_disc: array, scalar
            Ignored (bolometric is taken from ``agn_log_lbol``).
    agn_blr_cf: float
            BLR covering fraction.
    agn_blr_logU, agn_blr_logn, agn_blr_logZ: float
            Photoionization knobs forwarded to the grid adapter (log Z absolute).

    Returns
    -------
    tuple of ndarray, shape (n_wave,)
            ``(maskable, isotropic)`` channels; reproduces Synthesizer's isotropic BLR.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed_synthesizer_spectra(
        wave_aa,
        l_disc_bol_erg=l_bol_erg,
        covering_fraction=agn_blr_cf,
        grid_path=_resolve_synthesizer_grid("blr"),
        neb_logU=agn_blr_logU,
        neb_logn=agn_blr_logn,
        neb_logZ_gas=agn_blr_logZ,
        region="blr",
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    return jnp.zeros_like(L_lambda), L_lambda
