# SPDX-License-Identifier: BSD-3-Clause
"""Non-GRAHSP block implementations: proof that the block protocol mixes
across models.

Each adapter wraps an existing tengri AGN function (or dust attenuation
law) in the block-protocol signature so users can compose recipes like::

    agn_disc_block = "grahsp_sbpl"  # GRAHSP UV-optical
    agn_nlr_block = "none"  # skip
    agn_blr_block = "none"  # skip
    agn_feii_block = "none"  # skip
    agn_torus_block = "two_temperature"  # tengri's existing 2-T torus
    agn_attenuation_block = "smc_prevot"  # Prevot 1984 SMC

Importing this module side-effects all registrations.

Unit conversions
----------------
The block protocol uses :math:`L_\\lambda` [erg/s/Å]. Tengri's existing
disc / torus functions (``powerlaw_disc``, ``two_temperature_torus``,
etc.) return :math:`L_\\nu` [erg/s/Hz]. The adapters perform the standard
conversion :math:`L_\\lambda = L_\\nu \\, c / \\lambda^2` at the block
boundary.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc import powerlaw_disc
from tengri.components.agn.reddening import redden_disc
from tengri.components.agn.torus import simple_torus, two_temperature_torus
from tengri.components.dust.qsogen_ext import qsogen_quasar_extinction

__all__: list[str] = []  # registrations only

#: Speed of light in Å × Hz, for L_ν → L_λ conversion.
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S

# ──────────────────────────────────────────────────────────────────────
# Disc alternates
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "disc",
    "powerlaw",
    citation="",
    status="production",
    short_doc="Power-law disc with UV cutoff",
)
def powerlaw_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_alpha: float = -1.0,
    agn_T_max: float = 1.0e5,
    **_params,
) -> Array:
    r"""tengri ``powerlaw_disc`` as a disc-stage block.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol: float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_alpha: float, optional
        Power-law spectral index in :math:`L_\nu`. Default ``-1.0``.
    agn_T_max: float, optional
        UV cutoff temperature [K]. Default ``1e5``.

    Returns
    -------
    L_lambda: ndarray, shape (n_wave,)
        Disc :math:`L_\lambda` [erg/s/Å].
    """
    wave_aa = jnp.asarray(wavelength)
    L_nu = powerlaw_disc(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=1.0,
        agn_alpha=agn_alpha,
        agn_T_max=agn_T_max,
    )
    # L_nu [erg/s/Hz] -> L_lambda [erg/s/Å]: L_lambda = L_nu * c / lambda^2.
    return L_nu * _C_AA_PER_S / wave_aa**2


# ──────────────────────────────────────────────────────────────────────
# Torus alternates
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "torus",
    "simple",
    citation="",
    status="production",
    short_doc="Single-temperature graybody torus",
)
def simple_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_T_torus: float = 1000.0,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""tengri ``simple_torus`` (single-temperature graybody) block.

    The torus is normalized by ``agn_torus_frac × 10^agn_log_lbol`` :
    *not* by ``l5100_disc``: so this block ignores the disc 5100Å
    luminosity. That choice matches upstream :func:`unified_agn`.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
    agn_log_lbol: float
    l5100_disc: array_like, scalar
        Ignored (kept for protocol compatibility).
    agn_T_torus: float, optional
        Graybody temperature [K]. Default ``1000``.
    agn_torus_frac: float, optional
        Fraction of :math:`L_{\rm bol}` re-emitted by torus. Default ``0.5``.
    """
    del l5100_disc  # unused: simple torus normalizes off agn_log_lbol directly.
    wave_aa = jnp.asarray(wavelength)
    L_nu = simple_torus(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_T_torus=agn_T_torus,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block(
    "torus",
    "two_temperature",
    citation="",
    status="production",
    short_doc="Two-temperature (hot + warm) graybody torus",
)
def two_temperature_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_T_hot: float = 1200.0,
    agn_T_warm: float = 300.0,
    agn_frac_hot: float = 0.3,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""tengri ``two_temperature_torus`` block.

    Hot + warm dust graybody. As with :func:`simple_torus_block`, the
    normalization comes from ``agn_log_lbol``, not the disc 5100Å
    luminosity.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = two_temperature_torus(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_T_hot=agn_T_hot,
        agn_T_warm=agn_T_warm,
        agn_frac_hot=agn_frac_hot,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


# ──────────────────────────────────────────────────────────────────────
# Attenuation alternates
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "attenuation",
    "smc_prevot",
    citation="Prevot et al. 1984, A&A, 132, 389",
    status="production",
    short_doc="Prevot et al. 1984 SMC extinction curve",
)
def smc_prevot_block(
    wavelength: Array,
    *,
    agn_attenuation_ebv: float = 0.0,
    **_params,
) -> Array:
    r"""Prevot 1984 SMC attenuation curve as an attenuation-stage block.

    Returns the multiplicative factor :math:`10^{-0.4\,A_\lambda}` with

    .. math::

        A_\lambda = k(\lambda)\, R_V\, E(B-V),
        \qquad k(\lambda) = A(\lambda)/A(V),\ R_V = 2.72,

    i.e. exactly the extinction :func:`tengri.components.agn.reddening.redden_disc`
    applies for ``agn_ebv_disc`` (Prevot et al. 1984 [1]_; the prescription
    AGNfitter's ``BBBred_Prevot`` uses for its ``EBVbbb``). Delegating to
    ``redden_disc`` keeps the two disc-reddening paths identical at matched
    :math:`E(B-V)`, the block previously dropped the :math:`R_V` factor and
    under-attenuated by :math:`2.72\times` in magnitudes relative to
    ``agn_ebv_disc``.

    Note AGNfitter evaluates the raw Prevot fit at V
    (:math:`k_{\rm raw}(0.55\,\mu m) \approx 2.468`) instead of pinning the
    published :math:`R_V = 2.72`, so at matched :math:`E(B-V)` its
    :math:`A_\lambda` is a uniform factor :math:`2.468/2.72 \approx 0.907`
    of tengri's: identical spectral shape, a ~10% rescaling of the
    effective :math:`E(B-V)`.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_attenuation_ebv: float, optional
        :math:`E(B-V)` extinction [mag]. Default ``0.0`` (no attenuation).

    References
    ----------
    .. [1] M. L. Prevot et al., "The typical interstellar extinction in the
       Small Magellanic Cloud," A&A, 132, 389 (1984).
    """
    wave_aa = jnp.asarray(wavelength)
    # Single source of truth: redden_disc on a unit SED IS the factor.
    return redden_disc(wave_aa, jnp.ones_like(wave_aa), agn_attenuation_ebv)


@register_agn_block(
    "attenuation",
    "qsogen",
    citation="Temple, Hewett & Banerji 2021, MNRAS, 508, 737",
    status="production",
    short_doc="Temple+2021 empirical quasar extinction curve",
)
def qsogen_quasar_ext_block(
    wavelength: Array,
    *,
    agn_attenuation_ebv: float = 0.0,
    **_params,
) -> Array:
    r"""Temple+2021 empirical *quasar* extinction as an attenuation-stage block.

    This is qsogen's own reddening law (Temple, Hewett & Banerji 2021 [1]_,
    the ``pl_ext_comp_03`` curve), the one qsogen applies to the quasar
    continuum. Returns the multiplicative factor :math:`10^{-0.4\,A_\lambda}`
    with

    .. math::

        A_\lambda = E(B-V)\,\bigl[\,E(\lambda-V)/E(B-V) + R\,\bigr],
        \qquad R = 3.1,

    where the tabulated curve is the *color excess* :math:`E(\lambda-V)/E(B-V)`
    (zero at V) and ``R`` is qsogen's default total-to-selective ratio. This is
    a different law **and** a different convention from ``smc_prevot`` (the SMC
    Prevot fit AGNfitter uses, stored directly as :math:`A_\lambda/E(B-V)`).

    Because it runs at the attenuation stage: after the disc, lines and FeII
    are summed: it reddens the whole quasar spectrum at once, matching qsogen's
    "redden the quasar flux, excluding the host". For a qsogen build with no
    separate torus this is exactly qsogen's reddening; if a torus block is
    present it is reddened too (qsogen bundles the near-IR into the quasar
    model rather than a separate torus). No extinction is applied outside the
    curve's 500–60000 Å domain.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_attenuation_ebv: float, optional
        Quasar color excess :math:`E(B-V)` [mag]. Default ``0.0`` (no
        attenuation, a no-op, :math:`10^0 = 1`).

    Returns
    -------
    ndarray, shape (n_wave,)
        Multiplicative extinction factor :math:`10^{-0.4\,A_\lambda}`.

    References
    ----------
    .. [1] M. J. Temple, P. C. Hewett & M. Banerji, MNRAS, 508, 737 (2021).
       arXiv:2109.04472. https://doi.org/10.1093/mnras/stab2586
    """
    wave_aa = jnp.asarray(wavelength)
    a_over_ebv = qsogen_quasar_extinction(wave_aa)
    return jnp.power(10.0, -0.4 * a_over_ebv * agn_attenuation_ebv)
