# SPDX-License-Identifier: BSD-3-Clause
"""QSOgen (Temple+ 2021) decomposed into 5 composable AGN blocks.

The monolithic :func:`tengri.components.agn.qsogen.compute_qsogen_sed`
recipe is internally restructured around :func:`_qsogen_components`, which
returns each spectral piece (continuum, hot dust, emission lines, Balmer
continuum, SMC factor) under a shared joint cont+BB bolometric
normalization. These block adapters expose those pieces under the standard
block-protocol signature so users can mix qsogen continuum with e.g. a
SKIRTOR torus, or pair the qsogen SMC reddening with GRAHSP's BBB.

Bit-for-bit equivalence to ``compute_qsogen_sed`` holds when all 5 blocks
are summed and the SMC factor multiplied, the regression test
``tests/components/agn/test_agn_qsogen_decomposition.py`` pins this.

Block names registered::

    disc           → "qsogen"          (broken power-law continuum)
    blr            → "qsogen"          (Vanden Berk + Baldwin scaling)
    feii           → "qsogen_balmer"   (Balmer continuum; conceptually the
                                        feii-stage analog)
    torus          → "qsogen"          (single-temperature hot-dust BB)
    attenuation    → "qsogen_smc"      (SMC reddening factor)

Note: the disc and torus block share the name ``"qsogen"`` because each
category has its own namespace. There is no collision.

References
----------
.. [1] Temple, M. J., Hewett, P. C. & Banerji, M. 2021, MNRAS, 508, 737,
   https://doi.org/10.1093/mnras/stab2586.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.qsogen import (
    _DEFAULT_BBNORM,
    _DEFAULT_EBV,
    _DEFAULT_EMLINE_SCALE,
    _DEFAULT_PLBRK,
    _DEFAULT_PLSLP1,
    _DEFAULT_PLSLP2,
    _DEFAULT_TBB,
    _qsogen_components,
)

__all__: list[str] = []  # registrations only

from tengri.utils.physics_constants import C_AA as _C_AA_PER_S

# QSOgen-block parameters share a common kwargs subset; defining defaults once
# keeps the 5 wrappers in sync with the upstream module's defaults.
_QSOGEN_KW_DEFAULTS: dict[str, float] = {
    "agn_plslp1": _DEFAULT_PLSLP1,
    "agn_plslp2": _DEFAULT_PLSLP2,
    "agn_plbrk": _DEFAULT_PLBRK,
    "agn_tbb": _DEFAULT_TBB,
    "agn_bbnorm": _DEFAULT_BBNORM,
    "agn_emline_scale": _DEFAULT_EMLINE_SCALE,
    "agn_ebv": _DEFAULT_EBV,
    "agn_bcnorm": 0.0,
}


def _resolve_qsogen_kwargs(params: dict, agn_log_lbol: float) -> dict:
    """Pull QSOgen kwargs from the loose ``**params`` dict with defaults."""
    return dict(
        agn_log_lbol=agn_log_lbol,
        agn_plslp1=params.get("agn_plslp1", _QSOGEN_KW_DEFAULTS["agn_plslp1"]),
        agn_plslp2=params.get("agn_plslp2", _QSOGEN_KW_DEFAULTS["agn_plslp2"]),
        agn_plbrk=params.get("agn_plbrk", _QSOGEN_KW_DEFAULTS["agn_plbrk"]),
        agn_tbb=params.get("agn_tbb", _QSOGEN_KW_DEFAULTS["agn_tbb"]),
        agn_bbnorm=params.get("agn_bbnorm", _QSOGEN_KW_DEFAULTS["agn_bbnorm"]),
        agn_emline_scale=params.get("agn_emline_scale", _QSOGEN_KW_DEFAULTS["agn_emline_scale"]),
        agn_ebv=params.get("agn_ebv", _QSOGEN_KW_DEFAULTS["agn_ebv"]),
        agn_bcnorm=params.get("agn_bcnorm", _QSOGEN_KW_DEFAULTS["agn_bcnorm"]),
    )


def _l_nu_to_l_lambda(L_nu: Array, wave_aa: Array) -> Array:
    return L_nu * _C_AA_PER_S / wave_aa**2


# ──────────────────────────────────────────────────────────────────────
# Disc: broken power-law continuum
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "disc",
    "qsogen",
    citation="Temple et al. 2021, MNRAS, 508, 737",
    status="production",
    short_doc="Temple et al. 2021 QSOgen broken power-law continuum",
)
def qsogen_continuum_block(wavelength: Array, agn_log_lbol: float, **params) -> Array:
    r"""QSOgen broken power-law continuum block.

    The continuum is the *blue + red* power-law (split at ``agn_plbrk``)
    used by Temple+ 2021. Joint cont+BB bolometric normalization is
    re-derived inside :func:`_qsogen_components` so the disc-only output
    here is bit-for-bit identical to the disc piece of the monolithic
    ``compute_qsogen_sed``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    **params
        ``agn_plslp1``, ``agn_plslp2``, ``agn_plbrk``, ``agn_tbb``,
        ``agn_bbnorm``, ``agn_emline_scale``, ``agn_ebv``,
        ``agn_bcnorm``. Unrecognized keys are absorbed silently.
    """
    wave_aa = jnp.asarray(wavelength)
    comps = _qsogen_components(wave_aa, **_resolve_qsogen_kwargs(params, agn_log_lbol))
    return _l_nu_to_l_lambda(comps["continuum"], wave_aa)


# ──────────────────────────────────────────────────────────────────────
# Torus: hot-dust blackbody
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "torus",
    "qsogen",
    citation="Temple et al. 2021, MNRAS, 508, 737",
    status="production",
    short_doc="Temple et al. 2021 QSOgen hot-dust blackbody",
)
def qsogen_hot_dust_block(
    wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **params
) -> Array:
    r"""QSOgen hot-dust blackbody block.

    Single-temperature graybody anchored at 2 µm to ``agn_bbnorm × cont(2µm)``.
    Effective torus temperature ``agn_tbb`` (default ``1240 K``).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol : float
    l5100_disc : array
        Ignored: qsogen's BB anchors directly off the recomputed broken
        power-law continuum.
    **params
        QSOgen kwargs (see :func:`qsogen_continuum_block`).
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    comps = _qsogen_components(wave_aa, **_resolve_qsogen_kwargs(params, agn_log_lbol))
    return _l_nu_to_l_lambda(comps["hot_dust"], wave_aa)


# ──────────────────────────────────────────────────────────────────────
# BLR: Vanden Berk emission lines with Baldwin effect
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "blr",
    "qsogen",
    citation="Temple et al. 2021, MNRAS, 508, 737",
    status="production",
    short_doc="Temple et al. 2021 QSOgen empirical broad-line region",
)
def qsogen_blr_block(wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **params) -> Array:
    r"""QSOgen empirical broad-line region block.

    The complete quasar line spectrum from Vanden Berk template scaled by
    ``agn_emline_scale`` and the Baldwin-effect EW correction (luminous
    quasars have weaker lines). For use with ``nlr={'type': 'none'}``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        log10(L_bol / L_sun) of the AGN.
    l5100_disc : array
        Ignored: qsogen lines anchor on the internally-computed
        normalized continuum at line wavelengths, not on a single 5100 Å
        scalar.
    **params
        QSOgen kwargs.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        BLR :math:`L_\lambda` [erg/s/Å].
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    comps = _qsogen_components(wave_aa, **_resolve_qsogen_kwargs(params, agn_log_lbol))
    return _l_nu_to_l_lambda(comps["emission_lines"], wave_aa)


# ──────────────────────────────────────────────────────────────────────
# FeII slot: Balmer continuum (closest semantic match in the 5-stage
# pipeline; future revisions may grow a dedicated "balmer" category).
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "feii",
    "qsogen_balmer",
    citation="Temple et al. 2021, MNRAS, 508, 737",
    status="production",
    short_doc="Temple et al. 2021 QSOgen Balmer continuum",
)
def qsogen_balmer_block(
    wavelength: Array, agn_log_lbol: float, l5100_disc: Array, **params
) -> Array:
    r"""QSOgen Balmer continuum block.

    Hydrogen recombination continuum below the Balmer edge at 3646 Å,
    scaled by ``agn_bcnorm`` (default 0 = disabled). Categorized as
    ``feii`` because the Balmer continuum is, like FeII, a pseudo-continuum
    pile-up of unresolved transitions on top of the disc.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol : float
    l5100_disc : array
        Ignored.
    **params
        QSOgen kwargs.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    comps = _qsogen_components(wave_aa, **_resolve_qsogen_kwargs(params, agn_log_lbol))
    return _l_nu_to_l_lambda(comps["balmer_continuum"], wave_aa)


# ──────────────────────────────────────────────────────────────────────
# Attenuation; SMC reddening
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "attenuation",
    "qsogen_smc",
    citation="Temple et al. 2021, MNRAS, 508, 737",
    status="production",
    short_doc="Temple et al. 2021 QSOgen SMC reddening factor",
)
def qsogen_smc_block(wavelength: Array, **params) -> Array:
    r"""QSOgen SMC reddening factor block.

    Pure multiplicative factor :math:`10^{-0.4\,A_\lambda}` with
    :math:`A_\lambda = E(B-V)\,R_V^{\rm SMC}\,k(\lambda)` and
    :math:`R_V^{\rm SMC} = 2.93`. Distinct from
    :func:`smc_prevot_block` because the qsogen curve uses a slightly
    different parameterization (Pei 1992 + Temple+ 2021 normalization).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    **params
        ``agn_ebv`` [mag] is consumed; all other QSOgen kwargs are absorbed
        but unused.
    """
    wave_aa = jnp.asarray(wavelength)
    # We only need the smc_factor from _qsogen_components; the rest is
    # discarded but its evaluation cost is small (broken-PL + BB at default
    # log_lbol). For a scalar pure-attenuation block, JIT folds it cleanly.
    comps = _qsogen_components(
        wave_aa,
        **_resolve_qsogen_kwargs(params, agn_log_lbol=DEFAULT_AGN_LOG_LBOL),
    )
    return comps["smc_factor"]
