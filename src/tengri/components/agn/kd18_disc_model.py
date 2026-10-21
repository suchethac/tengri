# SPDX-License-Identifier: BSD-3-Clause
"""Three-zone accretion disc model (Kubota & Done 2018) SEDModelComponent.

Implements the Kubota & Done (2018) accretion disc model on the
SEDModelComponent contract, enabling use in the model-building API.

This is an opt-in adapter, the existing AGNSEDComponent continues to
support K&D18 through the unified AGN registry.

References
----------
.. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
   of AGN and its implications for the UV/X relation and optical
   variability," MNRAS, 480, 1247 (2018). doi:10.1093/mnras/sty1890.
   arXiv:1804.00171. bibcode:2018MNRAS.480.1247K.
.. [2] A. A. Beloborodov, "Energetic Radiation from Accretion Tori,"
   ApJ 510, L123 (1999). arXiv:astro-ph/9810145.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn._publication import with_lsun_outputs, with_lsun_siblings
from tengri.components.agn.disc import kubota_done_disc as _kubota_done_disc_fn
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig, declared_prior
from tengri.utils.physics_constants import L_SUN
from tengri.utils.scale import log10_magnitude

__all__ = ["KD18Disc"]

#: ``log10(L_sun / (erg/s))`` [dex].
_LOG10_L_SUN: float = math.log10(L_SUN)

#: Single source of truth for every class-level prior below: the shared
#: ``agn_*`` declarations in ``_params.py``. Read once at class-definition
#: time so these class attributes cannot drift from the canonical
#: declaration the way they did before this fix (found by
#: ``tools/check_param_restatements.py``, Task 11 item 5 / fix round 1,
#: R25): every one of the twelve priors below had restated its own literal
#: bounds/default instead of reading them here, and most had drifted --
#: ``log_lbol`` default 11.0 vs canonical 10.0; ``log_mbh`` default 8.0 vs
#: canonical 7.0; ``log_ledd`` bounds (-3, 0) vs canonical (-2, 0.5); and
#: eight more (see the fix-round commit message for the full table).
_LOG_LBOL_PRIOR = declared_prior(_AGN_PARAMS, "agn_log_lbol")
_LOG_MBH_PRIOR = declared_prior(_AGN_PARAMS, "agn_log_mbh")
_LOG_LEDD_PRIOR = declared_prior(_AGN_PARAMS, "agn_log_ledd")
_A_SPIN_PRIOR = declared_prior(_AGN_PARAMS, "agn_a_spin")
_COS_INC_PRIOR = declared_prior(_AGN_PARAMS, "agn_cos_inc")
_F_HARD_PRIOR = declared_prior(_AGN_PARAMS, "agn_f_hard")
_GAMMA_WARM_PRIOR = declared_prior(_AGN_PARAMS, "agn_gamma_warm")
_KT_WARM_PRIOR = declared_prior(_AGN_PARAMS, "agn_kt_warm")
_GAMMA_HARD_PRIOR = declared_prior(_AGN_PARAMS, "agn_gamma_hard")
_KT_HOT_PRIOR = declared_prior(_AGN_PARAMS, "agn_kt_hot")
_R_WARM_RATIO_PRIOR = declared_prior(_AGN_PARAMS, "agn_r_warm_ratio")
_LUM_RATIO_PRIOR = declared_prior(_AGN_PARAMS, "agn_lum_ratio")


@dataclass(frozen=True)
class KD18DiscConfig(SEDComponentConfig):
    """Configuration for K&D18 three-zone accretion disc.

    Parameters
    ----------
    self_consistent_gamma : bool, optional
        If True, derive the hard X-ray photon index Γ_hot self-consistently
        from the Beloborodov (1999) energy-balance relation. If False,
        use the agn_gamma_hard parameter directly. Default: False.
    n_radii : int, optional
        Number of radial grid points for zone integral approximations.
        Default: 50. Higher values increase accuracy but cost.
    """

    self_consistent_gamma: bool = False
    n_radii: int = 50


@dataclass(frozen=True)
class KD18Disc(SEDModelComponent):
    """Kubota & Done (2018) three-zone accretion disc.

    A physically stratified model with outer standard disc, warm Comptonization
    zone, and hot corona. Self-consistent zone radii and temperature profiles
    enable smooth inference gradients across the full parameter space.

    Attributes
    ----------
    name : str
        Component registry key: ``"kd18_disc"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : KD18DiscConfig
        Frozen configuration (self-consistent gamma, radial grid).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    log_mbh : Uniform
        log₁₀(M_BH / M_sun). [dex, 6–10]
    log_ledd : Uniform
        Eddington ratio log₁₀(L / L_Edd). [dex, -2–0.5]
    a_spin : Uniform
        Dimensionless black hole spin (prograde). [dimensionless, 0–0.998]
    cos_inc : Uniform
        Cosine of inclination (1 = face-on, 0 = edge-on). [dimensionless, 0–1]
    f_hard : Uniform
        Fraction of Eddington luminosity in hot corona. [dimensionless, 0–0.1]
    gamma_warm : Uniform
        Photon index of warm Comptonization zone. [dimensionless, 2–3]
    kt_warm : Uniform
        Warm zone electron temperature. [keV, 0.1–1]
    gamma_hard : Uniform
        Hard X-ray photon index (used if self_consistent_gamma=False).
        [dimensionless, 1.5–2.5]
    kt_hot : Uniform
        Hot corona electron temperature. [keV, 50–300]
    r_warm_ratio : Uniform
        Radius ratio R_warm / R_hot. [dimensionless, 1–5]
    lum_ratio : Uniform
        Fraction of bolometric luminosity from disc. [dimensionless, 0–5]

    Cross-component outputs
    -----------------------
    log_L_agn_disc, log_L_agn_disc_los : dex
        log10 of the angle-integrated power and of the line-of-sight isotropic-equivalent
        luminosity [dex re erg/s] of the emitted SED (all three zones), in closed form from
        the zone budget. The sphere-mean power ``log_L_agn_disc`` is the whole budget,
        ``lum_ratio * 10**log_lbol L_sun``: nothing of the model lies outside the emitted SED,
        so there is no ``_unemitted`` piece.

    Notes
    -----
    **JIT-compatible**: yes, predict() is pure JAX.

    **Gradient-safe**: yes, self-consistent zone radii via bisection
    are smooth over the parameter space.

    **Cross-component note**: AGN inclination (cos_inc) is shared with
    torus/corona models and is **never** auto-derived from geometry.
    Declare it as a free parameter independently.

    Examples
    --------
    Minimal model with K&D18 disc::

        from tengri import SEDModel, Fixed, DEFAULT, Uniform, builders
        from tengri.components.agn.kd18_disc_model import KD18Disc, KD18DiscConfig

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(alpha=Fixed(1.5), beta=Fixed(1.0)),
            dust_attenuation={"type": "two_component", "all_params": Fixed(DEFAULT)},
            agn=KD18Disc(config=KD18DiscConfig(self_consistent_gamma=True)),
        )

    See Also
    --------
    tengri.components.agn.disc : Kubota & Done (2018) implementation.
    """

    name = "kd18_disc"
    parameter_prefix = "agn_"
    config: KD18DiscConfig = field(default_factory=KD18DiscConfig)

    # Free parameters: auto-discovered
    log_lbol = Uniform(
        _LOG_LBOL_PRIOR.lo,
        _LOG_LBOL_PRIOR.hi,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=_LOG_LBOL_PRIOR.default,
    )
    log_mbh = Uniform(
        _LOG_MBH_PRIOR.lo,
        _LOG_MBH_PRIOR.hi,
        description="Black hole mass",
        units="dex (M_sun)",
        default=_LOG_MBH_PRIOR.default,
    )
    log_ledd = Uniform(
        _LOG_LEDD_PRIOR.lo,
        _LOG_LEDD_PRIOR.hi,
        description="Eddington ratio",
        units="dex",
        default=_LOG_LEDD_PRIOR.default,
    )
    a_spin = Uniform(
        _A_SPIN_PRIOR.lo,
        _A_SPIN_PRIOR.hi,
        description="Black hole spin parameter",
        units="dimensionless",
        default=_A_SPIN_PRIOR.default,
    )
    cos_inc = Uniform(
        _COS_INC_PRIOR.lo,
        _COS_INC_PRIOR.hi,
        description="Cosine of inclination",
        units="dimensionless",
        default=_COS_INC_PRIOR.default,
    )
    f_hard = Uniform(
        _F_HARD_PRIOR.lo,
        _F_HARD_PRIOR.hi,
        description="Corona luminosity fraction",
        units="dimensionless",
        default=_F_HARD_PRIOR.default,
    )
    gamma_warm = Uniform(
        _GAMMA_WARM_PRIOR.lo,
        _GAMMA_WARM_PRIOR.hi,
        description="Warm Comptonization photon index",
        units="dimensionless",
        default=_GAMMA_WARM_PRIOR.default,
    )
    kt_warm = Uniform(
        _KT_WARM_PRIOR.lo,
        _KT_WARM_PRIOR.hi,
        description="Warm zone electron temperature",
        units="keV",
        default=_KT_WARM_PRIOR.default,
    )
    gamma_hard = Uniform(
        _GAMMA_HARD_PRIOR.lo,
        _GAMMA_HARD_PRIOR.hi,
        description="Hard X-ray photon index",
        units="dimensionless",
        default=_GAMMA_HARD_PRIOR.default,
    )
    kt_hot = Uniform(
        _KT_HOT_PRIOR.lo,
        _KT_HOT_PRIOR.hi,
        description="Hot corona electron temperature",
        units="keV",
        default=_KT_HOT_PRIOR.default,
    )
    r_warm_ratio = Uniform(
        _R_WARM_RATIO_PRIOR.lo,
        _R_WARM_RATIO_PRIOR.hi,
        description="Radius ratio R_warm / R_hot",
        units="dimensionless",
        default=_R_WARM_RATIO_PRIOR.default,
    )
    lum_ratio = Uniform(
        _LUM_RATIO_PRIOR.lo,
        _LUM_RATIO_PRIOR.hi,
        description="Disc luminosity fraction of L_bol",
        units="dimensionless",
        default=_LUM_RATIO_PRIOR.default,
    )

    # Cross-component output
    outputs: ClassVar[dict[str, str]] = with_lsun_outputs(
        {"log_L_agn_disc": "dex", "log_L_agn_disc_los": "dex"}
    )

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        r"""Pure JAX K&D18 three-zone disc prediction.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - log_mbh: log₁₀(M_BH / M_sun)
            - log_ledd: Eddington ratio
            - a_spin: black hole spin
            - cos_inc: cosine of inclination
            - f_hard: hot corona fraction
            - gamma_warm: warm zone photon index
            - kt_warm: warm zone temperature (keV)
            - gamma_hard: hard X-ray index
            - kt_hot: hot corona temperature (keV)
            - r_warm_ratio: R_warm / R_hot
            - lum_ratio: disc luminosity fraction of L_bol [dimensionless]

        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz.
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Unused (AGN disc is self-contained).

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where:

            - sed_out: Updated SED (sed_in + disc contribution).
            - published: ``log_L_agn_disc`` and ``log_L_agn_disc_los`` [dex re erg/s], defined
              below.

        Notes
        -----
        The disc and warm Comptonization zones are optically thick, so their emission is
        proportional to :math:`\cos i`; the hot corona is optically thin and isotropic
        (Kubota & Done 2018 [1]_, Sects. 2.1 and 2.2). With :math:`D_\nu` the
        angle-integrated (two-face) luminosity density of the disc and warm zones, whose power
        is :math:`\int D_\nu\,d\nu = 2\int\sigma T_{\rm NT}^4\,2\pi R\,dR` (their Eq. 2 for
        the hot-flow range), and :math:`H_\nu` the corona,

        .. math::

            L_\nu(i) = 2\cos i\, D_\nu + H_\nu .

        The factor 2 makes the mean of :math:`2\cos i` over :math:`\cos i \in [0, 1]` equal
        one; it is tengri's isotropic-equivalent normalization, not an equation of the paper.
        Two luminosities are published:

        .. math::

            L_{\rm agn\_disc} = \int (D_\nu + H_\nu)\,d\nu ,
            \qquad
            L_{\rm agn\_disc\_los} = \int L_\nu(i)\,d\nu .

        ``log_L_agn_disc`` [dex re erg/s] is the log of the angle-integrated emitted power: it
        is independent of ``cos_inc`` and equals ``lum_ratio`` times the accretion power
        implied by ``log_lbol``. ``log_L_agn_disc_los`` is the log of the line-of-sight
        isotropic-equivalent luminosity of the emitted, inclination-projected SED, a derived
        output; at :math:`\cos i = 0.5` the two coincide. Both are closed forms of the zone
        budget (:func:`_published_log_powers`), which the SED carries exactly: the outer-disc
        rings are normalized by the Planck integral of the spectra they emit, the warm zone by
        the exact integral of its piecewise-linear template, and the corona by the trapezoid in
        :math:`\ln\nu` on its fixed grid, which contains the whole shape. The dense-grid
        integral of the emitted SED matches them to 1e-12. They do not depend on the caller's
        wavelength grid and are formed in log space, so they are finite in pure float32, where
        the erg/s value overflows (NAMING_CONTRACT section 4c). ``-inf`` is an absent disc
        (``lum_ratio = 0``).

        **JIT-compatible**: yes.

        References
        ----------
        .. [1] A. Kubota and C. Done, MNRAS, 480, 1247 (2018), arXiv:1804.00171.

        """

        sed_disc, _, _, corona_fraction = _kubota_done_disc_fn(
            wavelength=wave,
            agn_log_lbol=p["log_lbol"],
            agn_lum_ratio=p["lum_ratio"],
            agn_log_mbh=p["log_mbh"],
            agn_log_ledd=p["log_ledd"],
            agn_a_spin=p["a_spin"],
            agn_cos_inc=p["cos_inc"],
            agn_f_hard=p["f_hard"],
            agn_gamma_warm=p["gamma_warm"],
            agn_kt_warm=p["kt_warm"],
            agn_gamma_hard=p["gamma_hard"],
            agn_kt_hot=p["kt_hot"],
            agn_r_warm_ratio=p["r_warm_ratio"],
            n_radii=self.config.n_radii,
            agn_self_consistent_gamma=self.config.self_consistent_gamma,
            _return_parts=True,
        )
        log_L_disc, log_L_disc_los = _published_log_powers(
            p["log_lbol"], p["lum_ratio"], p["cos_inc"], corona_fraction
        )
        sed_out = sed_in + sed_disc

        return sed_out, with_lsun_siblings(
            {"log_L_agn_disc": log_L_disc, "log_L_agn_disc_los": log_L_disc_los}
        )


def _published_log_powers(log_lbol, lum_ratio, cos_inc, corona_fraction):
    r"""Angle-integrated and line-of-sight power of the emitted K&D18 SED, in closed form.

    The disc and warm zones carry :math:`\int D_\nu\,d\nu = (1-f_c) L` and the corona
    :math:`\int H_\nu\,d\nu = f_c L`, with :math:`L = r\,10^{\log L_{\rm bol}} L_\odot`
    and :math:`f_c` the corona fraction; the emitted spectrum is
    :math:`L_\nu(\mu) = 2\mu D_\nu + H_\nu`, so

    .. math::

        \int_0^1 \!\int L_\nu(\mu)\,d\nu\,d\mu = L ,
        \qquad
        \int L_\nu(\mu)\,d\nu = L\,[2\mu(1-f_c) + f_c] .

    Parameters
    ----------
    log_lbol : array_like
        :math:`\log_{10}(L_{\rm acc}/L_\odot)` [dex].
    lum_ratio : array_like
        Disc luminosity ratio :math:`r` [dimensionless].
    cos_inc : array_like
        :math:`\mu = \cos i` [dimensionless].
    corona_fraction : array_like
        :math:`f_c`, the corona's share of the accretion power [dimensionless]; above 1
        when the hot flow's dissipation exceeds the accretion power, where the disc is
        floored at zero.

    Returns
    -------
    log_disc, log_disc_los : ndarray, shape ()
        :math:`\log_{10}` of the two powers [dex re erg/s]; ``-inf`` at ``lum_ratio = 0``.
    """
    disc_share = jnp.maximum(1.0 - corona_fraction, 0.0)
    log_scale = log10_magnitude(lum_ratio) + log_lbol + _LOG10_L_SUN
    log_disc = log_scale + jnp.log10(disc_share + corona_fraction)
    log_los = log_scale + log10_magnitude(2.0 * cos_inc * disc_share + corona_fraction)
    return log_disc, log_los
