# SPDX-License-Identifier: BSD-3-Clause
"""Lines-stage blocks: BLR (broad-line region) and NLR (narrow-line region).

Wraps :func:`compute_blr_sed` and :func:`compute_nlr_sed` so they fit into
the composable AGN pipeline. The two line regions are normalised differently,
matching their geometry:

* **NLR** — spatially extended and isotropically illuminated, so it is driven by
  the **intrinsic** AGN bolometric luminosity
  (:math:`L_{\\rm bol} = 10^{\\,\\mathrm{agn\\_log\\_lbol}} L_\\odot`). Its flux is
  therefore inclination-independent (the runner's Type-1/2 mask leaves it
  untouched).
* **BLR** — compact and part of the anisotropic central engine. The runner only
  has :math:`\\lambda L_\\lambda(5100\\,\\mathrm{\\AA})` (``l5100_disc``) on hand, so
  the BLR adapter applies a bolometric correction at the boundary,

  .. math::

     L_{\\rm disc, bol} = f_{\\rm bol} \\, \\lambda L_\\lambda(5100\\,\\mathrm{\\AA}),

  with default :math:`f_{\\rm bol} = 9` from Krawczyk+ 2013 (Type-1 quasar
  median), overridable via ``agn_blr_f_bol``. The runner's geometric mask then
  supplies the BLR's inclination dependence.

References
----------
.. [1] Krawczyk, C. M. et al. 2013, ApJS, 206, 4 (quasar SED bolometric
       corrections).
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blocks.masking import split_lines_result
from tengri.components.agn.blr import compute_blr_sed
from tengri.components.agn.nlr import compute_nlr_sed
from tengri.components.agn.nlr_cloudy import (
    compute_blr_sed_synthesizer,
    compute_nlr_sed_synthesizer,
)
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG

__all__: list[str] = []  # registrations only

_C_AA_PER_S: float = 2.99792458e18


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


#: Default bolometric correction :math:`L_{\rm bol}/\lambda L_\lambda(5100\,\mathrm{\AA})`
#: from Krawczyk+ 2013 (2013ApJS..206....4K). Type-1 quasar median.
DEFAULT_F_BOL_5100: float = 9.0


@register_agn_block("lines", "blr")
def blr_lines_block(
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
    r"""Broad-line region emission as a lines-stage block.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored (kept for protocol compatibility — ``l5100_disc`` provides
        the normalisation).
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_blr_cf : float, optional
        BLR covering fraction [0, 1]. Default ``0.1``.
    agn_blr_fwhm_kms : float, optional
        Broad-line FWHM [km/s]. Default ``5000``.
    agn_fe2_strength : float, optional
        :math:`R_{\rm Fe} = F({\rm FeII})/F({\rm H}\beta)`. Default ``0.0``
        (FeII pseudo-continuum disabled).
    agn_blr_line_efficiency : float, optional
        Fraction of intercepted luminosity converted to lines. Default ``0.08``.
    agn_blr_f_bol : float, optional
        Bolometric correction :math:`L_{\rm bol}/\lambda L_\lambda(5100\,\mathrm{\AA})`.
        Default :data:`DEFAULT_F_BOL_5100`.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        BLR :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    Geometric masking by the torus is **not** applied here. If the
    composable recipe also activates a torus block, double-counting is
    possible for line-of-sight inclination effects; see Section 2 of
    :mod:`tengri.components.agn.unified` for the mask convention.
    """
    del agn_log_lbol  # normalisation comes from l5100_disc
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


@register_agn_block("lines", "nlr")
def nlr_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    agn_nlr_line_efficiency: float = 0.10,
    **_params,
) -> Array:
    r"""Narrow-line region emission as a lines-stage block.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        log10 of the AGN bolometric luminosity [Lsun] — the **isotropic**
        illuminator of the NLR.
    l5100_disc : array, scalar
        Ignored (the NLR is illuminated by the inclination-independent
        bolometric, not the apparent — possibly foreshortened — l5100).
    agn_nlr_cf : float, optional
        NLR covering fraction. Default ``0.1``.
    agn_nlr_fwhm_kms : float, optional
        Narrow-line FWHM [km/s]. Default ``500``.
    agn_nlr_line_efficiency : float, optional
        Default ``0.10``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        NLR :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    The NLR is spatially extended and illuminated by the **intrinsic** AGN
    bolometric luminosity (:math:`L_{\rm bol} = 10^{\,\mathrm{agn\_log\_lbol}}
    L_\odot`), so its flux is inclination-independent — matching
    ``unified_nlr_blr``. Using the apparent ``l5100_disc`` would leak an
    inclination-dependent disc's ``cos_inc`` foreshortening into the
    (supposedly isotropic) narrow lines.
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
    # NLR is spatially extended -> isotropic (visible at all inclinations); it is
    # the *isotropic* channel so the runner's Type-1/2 mask leaves it untouched.
    return jnp.zeros_like(L_lambda), L_lambda


@register_agn_block("lines", "nlr_synthesizer")
def nlr_synthesizer_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    neb_logU: float = -2.0,
    neb_logZ_gas: float = -1.8477,
    **_params,
) -> Array:
    r"""NLR lines from the Synthesizer Cloudy grid (grid-backed lines block).

    Routes the composable pipeline to :func:`compute_nlr_sed_synthesizer`, so a
    unified AGN built through ``SEDModel.build`` uses the *same* photoionisation
    grid as the direct adapter (the grids the §9c reproduction panel reads). The
    grid path is resolved from ``$TENGRI_SYNTHESIZER_AGN_GRID_DIR`` /
    ``data/synthesizer_grids/`` (closes the builder-accessibility gap, #588).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored (bolometric is taken from ``l5100_disc``).
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_nlr_cf, agn_nlr_fwhm_kms : float
        Covering fraction and narrow-line FWHM [km/s].
    neb_logU, neb_logZ_gas : float
        Photoionisation knobs forwarded to the grid adapter.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        NLR :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    Like the analytic :func:`nlr_lines_block`, the NLR is illuminated by the
    **intrinsic** bolometric luminosity (``10**agn_log_lbol * L_sun``) and is
    therefore inclination-independent (isotropic). Backend initialisation reads
    HDF5 (Python-level, not JIT-traceable); call once eagerly before any
    ``jax.jit`` over the forward model so the cached backend's interpolation
    stays JIT-safe.
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
        neb_logU=neb_logU,
        neb_logZ_gas=neb_logZ_gas,
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    # Isotropic channel (extended NLR) — bypasses the runner's Type-1/2 mask.
    return jnp.zeros_like(L_lambda), L_lambda


@register_agn_block("lines", "blr_synthesizer")
def blr_synthesizer_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_blr_cf: float = 0.1,
    agn_blr_fwhm_kms: float = 5000.0,
    agn_blr_f_bol: float = DEFAULT_F_BOL_5100,
    neb_logU: float = -1.0,
    neb_logZ_gas: float = -1.8477,
    **_params,
) -> Array:
    r"""BLR lines from the Synthesizer Cloudy grid (grid-backed lines block).

    Broad-line sibling of :func:`nlr_synthesizer_lines_block`, routing to
    :func:`compute_blr_sed_synthesizer` (default FWHM 5000 km/s). Grid path is
    resolved the same way. See that function's Notes for the JIT caveat.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored (bolometric is taken from ``l5100_disc``).
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_blr_cf, agn_blr_fwhm_kms, agn_blr_f_bol : float
        Covering fraction, broad-line FWHM [km/s], and bolometric correction.
    neb_logU, neb_logZ_gas : float
        Photoionisation knobs forwarded to the grid adapter.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
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
        neb_logU=neb_logU,
        neb_logZ_gas=neb_logZ_gas,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("lines", "nlr_blr")
def nlr_blr_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    **params,
) -> Array:
    r"""Combined narrow + broad line region emission (analytic templates).

    A *unified* AGN has both a narrow-line region (NLR) and a broad-line region
    (BLR). The composable ``lines`` slot holds a single selector, so without a
    combined block a disc + torus + NLR + BLR model (the Synthesizer
    ``UnifiedAGN`` decomposition) could not be expressed through
    ``SEDModel.build`` — it had to be hand-assembled from separate calls. This
    block sums :func:`nlr_lines_block` and :func:`blr_lines_block` so one
    composable recipe yields the full line spectrum.

    Each sub-region keeps its own parameters (``agn_nlr_*`` / ``agn_blr_*``);
    they are dispatched to the matching sub-block and the two
    :math:`L_\lambda` are added.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored (normalisation comes from ``l5100_disc``).
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    **params
        ``agn_nlr_*`` and ``agn_blr_*`` knobs, routed to the respective
        sub-blocks (each ignores the other's parameters).

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Summed NLR + BLR :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    Returns the ``(anisotropic, isotropic)`` split: the BLR is the maskable
    (anisotropic) channel and the NLR the isotropic one, so the runner's
    Type-1/2 mask obscures the BLR with the disc while the NLR stays visible.
    """
    a_nlr, i_nlr = split_lines_result(
        nlr_lines_block(wavelength, agn_log_lbol, l5100_disc, **params)
    )
    a_blr, i_blr = split_lines_result(
        blr_lines_block(wavelength, agn_log_lbol, l5100_disc, **params)
    )
    return a_nlr + a_blr, i_nlr + i_blr


@register_agn_block("lines", "nlr_blr_synthesizer")
def nlr_blr_synthesizer_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    **params,
) -> Array:
    r"""Combined NLR + BLR from the Synthesizer Cloudy grids (grid-backed).

    Grid-backed sibling of :func:`nlr_blr_lines_block`: sums
    :func:`nlr_synthesizer_lines_block` and :func:`blr_synthesizer_lines_block`,
    so ``agn={'disc': ..., 'torus': ..., 'lines': {'type': 'nlr_blr_synthesizer'}}``
    builds a unified AGN whose line regions read the *same* photoionisation grids
    as the Synthesizer ``UnifiedAGN`` reproduction — the combination that
    previously required hand-assembling raw adapter calls outside the model.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored (normalisation comes from ``l5100_disc``).
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    **params
        ``agn_nlr_*`` / ``agn_blr_*`` knobs routed to the respective sub-blocks.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Summed NLR + BLR :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    Inherits the grid-backed blocks' JIT caveat: backend init reads HDF5 (not
    JIT-traceable), so build/predict once eagerly before any ``jax.jit``. Returns
    the ``(anisotropic BLR, isotropic NLR)`` split (see :func:`nlr_blr_lines_block`).
    """
    a_nlr, i_nlr = split_lines_result(
        nlr_synthesizer_lines_block(wavelength, agn_log_lbol, l5100_disc, **params)
    )
    a_blr, i_blr = split_lines_result(
        blr_synthesizer_lines_block(wavelength, agn_log_lbol, l5100_disc, **params)
    )
    return a_nlr + a_blr, i_nlr + i_blr
