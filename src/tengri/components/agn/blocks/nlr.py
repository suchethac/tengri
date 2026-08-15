# SPDX-License-Identifier: BSD-3-Clause
r"""Narrow-line region (NLR) blocks for the composable AGN pipeline.

One file, every NLR option — pick one via ``agn={'nlr': {'type': ...}}``:

======================  ===========================================  ==========
``type``                model                                        physical?
======================  ===========================================  ==========
``analytic``            Richardson+2014 line ratios + Gaussians      empirical
``feltre``              Feltre+2016 CLOUDY photoionization (BEAGLE)   most
``synthesizer``         Synthesizer CLOUDY ``/lines`` grid           physical
``synthesizer_spectra`` Synthesizer UnifiedAGN ``/spectra/nebular``  physical
======================  ===========================================  ==========

All NLR blocks are **isotropic** — the NLR is spatially extended and
illuminated by the intrinsic bolometric ``10**agn_log_lbol * L_sun`` (ADR-0018
§3), so they return ``(0, L_lambda)`` and bypass the runner's Type-1/2 mask.
The ``none`` block lives in :mod:`._protocol`.

Consolidated 2026-07 from the former per-model files (``nlr_analytic``,
``nlr_feltre``, ``nlr_synthesizer``, ``nlr_synthesizer_spectra``, ``nlr_blocks``,
``_nlr_common``) — registration is unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.nlr import compute_nlr_sed
from tengri.components.agn.nlr_cloudy import (
    compute_nlr_sed_cue,
    compute_nlr_sed_feltre,
    compute_nlr_sed_synthesizer,
    compute_nlr_sed_synthesizer_spectra,
    load_cue_agn_weights,
)
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG

__all__ = [
    "nlr_analytic_block",
    "nlr_cue_block",
    "nlr_feltre_block",
    "nlr_synthesizer_block",
    "nlr_synthesizer_spectra_block",
]

from tengri.utils.physics_constants import C_AA as _C_AA_PER_S


def _resolve_synthesizer_grid(kind: str) -> str:
    """Resolve a Synthesizer AGN grid path for the grid-backed line blocks.

    Searches ``$TENGRI_SYNTHESIZER_AGN_GRID_DIR`` then the repo-default
    ``data/synthesizer_grids/``. ``kind`` is ``"nlr"`` or ``"blr"``. These grids
    are not packaged with tengri (they ship via ``synthesizer-download
    --agn-test-grids``), so a clear error is raised if neither location holds
    ``test_grid_agn-<kind>.hdf5``.
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
    "nlr",
    "analytic",
    citation="Richardson et al. 2014, MNRAS, 437, 2376",
    status="production",
    short_doc="Analytic NLR with Richardson+2014 line ratios and Gaussian broadening",
)
def nlr_analytic_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    agn_nlr_line_efficiency: float = 0.10,
    **_params,
) -> tuple[Array, Array]:
    r"""Analytic NLR (Richardson+2014 line ratios + Gaussian broadening).

    Illuminated by the **intrinsic** AGN bolometric (``10**agn_log_lbol *
    L_sun``), so inclination-independent — using the apparent ``l5100_disc``
    would leak the disc's ``cos_inc`` foreshortening into the (isotropic) narrow
    lines. Returns ``(maskable=0, isotropic=L_lambda)``.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm_kms,
        line_efficiency=agn_nlr_line_efficiency,
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    return jnp.zeros_like(L_lambda), L_lambda


@register_agn_block(
    "nlr",
    "feltre",
    citation=(
        "Feltre, Charlot & Gutkin 2016, MNRAS 456, 3354 (arXiv:1511.08217), "
        "'Nuclear activity versus star formation: emission-line diagnostics at "
        "ultraviolet and optical wavelengths'; BEAGLE NLR parity "
        "(Vidal-García et al. 2022, arXiv:2211.13648)"
    ),
    status="experimental",
    short_doc="Feltre+2016 CLOUDY photoionization NLR (BEAGLE parity)",
)
def nlr_feltre_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    agn_nlr_alpha_pl: float = -1.7,
    agn_nlr_logU: float = -2.0,
    agn_nlr_logn: float = 3.0,
    agn_nlr_logZ: float = -1.8477,
    agn_nlr_xi_d: float = 0.3,
    **_params,
) -> tuple[Array, Array]:
    r"""Physical NLR from the Feltre+2016 CLOUDY grid (BEAGLE parity).

    Self-consistent photoionization: the disc's ionizing continuum
    (:math:`f_\nu \propto \nu^{\alpha}`) sets :math:`Q_{\rm H} = \int (L_\nu/h\nu)
    d\nu` from :math:`L_{\rm acc}`, and the Feltre, Charlot & Gutkin (2016)
    CLOUDY c13.03 grid converts :math:`Q_{\rm H}` into emission lines over the
    five grid axes (:math:`\alpha_{\rm pl}, \log U, \log n_{\rm H}, \log Z,
    \xi_d`) — the same grid BEAGLE interpolates. The most physical NLR model.
    Requires ``data/feltre_grid.h5`` (skips gracefully if absent).

    The grid axes are **AGN-specific** (``agn_nlr_logU/logn/logZ``, not the
    galaxy ``neb_*`` names) so they route through the AGN component's ``agn_*``
    parameter sweep — otherwise ``SEDModel.build`` would freeze them at their
    defaults (only ``agn_``-prefixed params reach the runner) and collide with
    the stellar nebular parameters.

    **Energy:** the NLR lines are reprocessed disc ionizing photons, so they
    *will* be debited from the disc under ``agn_norm="conserving"`` once emission
    lines join the Σf covering-fraction ledger (spec Phase 1.x). **Currently the
    lines are additive** — the ledger debits the torus only.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed_feltre(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm_kms,
        alpha_pl=agn_nlr_alpha_pl,
        neb_logU=agn_nlr_logU,
        neb_logn=agn_nlr_logn,
        neb_logZ_gas=agn_nlr_logZ,
        xi_d=agn_nlr_xi_d,
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    return jnp.zeros_like(L_lambda), L_lambda


@register_agn_block(
    "nlr",
    "cue",
    citation=(
        "Li, Y. et al. 2025, ApJ 986, 9 (arXiv:2405.04598), 'The Cue Nebular "
        "Emulator'; disc->Cue->NLR pipeline in the BEAGLE spirit "
        "(Feltre, Charlot & Gutkin 2016)"
    ),
    status="experimental",
    short_doc="Cue emulator AGN-ionized NLR (fast differentiable, BEAGLE-style)",
    template_loader=load_cue_agn_weights,
)
def nlr_cue_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    agn_nlr_alpha_pl: float = -1.7,
    agn_nlr_logU: float = -2.0,
    agn_nlr_logn: float = 3.0,
    agn_nlr_logZ: float = -1.8477,
    templates=None,
    **_params,
) -> tuple[Array, Array]:
    r"""AGN-ionized NLR from the Cue emulator (BEAGLE-style, differentiable).

    Self-consistent photoionization: the disc's ionizing continuum
    (:math:`f_\nu \propto \nu^{\alpha}`) sets :math:`Q_{\rm H}` from
    :math:`L_{\rm acc}`, and the Cue neural-network emulator (Li+2025) predicts
    the AGN-ionized narrow lines — the same disc → :math:`Q_{\rm H}` → nebular
    pipeline as ``nlr='feltre'``, but with Cue's fast differentiable emulator in
    place of a tabulated CLOUDY grid. Requires ``data/cue_weights.npz`` (skips
    gracefully if absent).

    The grid axes are **AGN-specific** (``agn_nlr_logU/logn/logZ``, not the
    galaxy ``neb_*`` names) so they route through the AGN component's ``agn_*``
    parameter sweep — otherwise ``SEDModel.build`` would freeze them at their
    defaults (only ``agn_``-prefixed params reach the runner) and collide with
    the stellar nebular parameters. ``agn_nlr_logZ`` is **absolute**
    :math:`\log_{10} Z` (matching the Feltre block); the adapter converts to
    Cue's native :math:`\log_{10}(Z/Z_\odot)`.

    **Energy:** like the other NLR blocks the lines are reprocessed disc
    ionizing photons — currently additive; they join the Σf covering-fraction
    ledger under ``agn_norm="conserving"`` once emission lines are debited.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed_cue(
        wave_aa,
        _template=templates,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm_kms,
        alpha_pl=agn_nlr_alpha_pl,
        neb_logU=agn_nlr_logU,
        neb_logn=agn_nlr_logn,
        neb_logZ_gas=agn_nlr_logZ,
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    return jnp.zeros_like(L_lambda), L_lambda


@register_agn_block(
    "nlr",
    "synthesizer",
    citation="Lovell et al. 2025 (Open J. Astrophys.); Roper et al. 2026 (JOSS) — Synthesizer",
    status="production",
    short_doc="Synthesizer Cloudy grid narrow-line region",
)
def nlr_synthesizer_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    agn_nlr_logU: float = -2.0,
    agn_nlr_logZ: float = -1.8477,
    **_params,
) -> tuple[Array, Array]:
    r"""NLR lines from the Synthesizer Cloudy ``/lines`` grid.

    Illuminated by the intrinsic bolometric (``10**agn_log_lbol * L_sun``), so
    isotropic. Grid path resolves from ``$TENGRI_SYNTHESIZER_AGN_GRID_DIR`` /
    ``data/synthesizer_grids/``. Backend init reads HDF5 (Python-level) — call
    once eagerly before any ``jax.jit`` over the forward model.

    The photoionization axes are named ``agn_nlr_logU/logZ`` (not the galaxy
    ``neb_*`` names) so they survive the AGN component's ``agn_``-prefix filter
    and are drivable through ``SEDModel.build`` — otherwise they were frozen at
    their defaults (a silent no-op, #931). They translate to the grid's
    ``neb_*`` axes internally.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed_synthesizer(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm_kms,
        grid_path=_resolve_synthesizer_grid("nlr"),
        neb_logU=agn_nlr_logU,
        neb_logZ_gas=agn_nlr_logZ,
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    return jnp.zeros_like(L_lambda), L_lambda


@register_agn_block(
    "nlr",
    "synthesizer_spectra",
    citation="Lovell et al. 2025 (Open J. Astrophys.); Roper et al. 2026 (JOSS) — Synthesizer",
    status="production",
    short_doc="Synthesizer UnifiedAGN NLR reprocessed nebular spectrum",
)
def nlr_synthesizer_spectra_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_logU: float = -2.0,
    # Differs from the declared agn_nlr_logn default (3.0, the Feltre+2016 grid
    # center): this block reproduces Synthesizer's UnifiedAGN, and 4.0 sits at
    # the grid edge, which reads as that code's own NLR-density convention.
    # Left as-is rather than unified — NOT verified against upstream Synthesizer.
    agn_nlr_logn: float = 4.0,
    agn_nlr_logZ: float = -2.0,
    **_params,
) -> tuple[Array, Array]:
    r"""NLR reprocessed nebular spectrum reproducing Synthesizer's UnifiedAGN.

    Reads the grid's ``/spectra/nebular`` array (continuum + lines) — the exact
    product Synthesizer's ``UnifiedAGN`` extracts — instead of re-broadening the
    discrete ``/lines`` table (#694). Isotropic: illuminated by the intrinsic
    bolometric with the grid inclination held at its isotropic node.

    The photoionization axes are named ``agn_nlr_logU/logn/logZ`` (not the galaxy
    ``neb_*`` names) so they survive the AGN component's ``agn_``-prefix filter
    and are drivable through ``SEDModel.build`` (#931); they translate to the
    grid's ``neb_*`` axes internally.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed_synthesizer_spectra(
        wave_aa,
        l_disc_bol_erg=l_bol_erg,
        covering_fraction=agn_nlr_cf,
        grid_path=_resolve_synthesizer_grid("nlr"),
        neb_logU=agn_nlr_logU,
        neb_logn=agn_nlr_logn,
        neb_logZ_gas=agn_nlr_logZ,
        region="nlr",
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    return jnp.zeros_like(L_lambda), L_lambda
