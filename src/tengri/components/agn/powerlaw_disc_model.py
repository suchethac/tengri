# SPDX-License-Identifier: BSD-3-Clause
"""Phenomenological power-law accretion disc model SEDModelComponent.

Implements the power-law accretion disc model on the SEDModelComponent
contract for fast prototyping and simple AGN fits.

This is an opt-in adapter, the existing AGNSEDComponent continues to
support power-law discs through the unified AGN registry.

Notes
-----
This is a simplified model suitable for fast fitting and exploratory work.
For production science, use Kubota & Done (2018) or SKIRTOR torus models.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn._publication import with_lsun_outputs, with_lsun_siblings
from tengri.components.agn.disc import _powerlaw_shape_unit
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig, declared_prior
from tengri.utils.host_array import device_table, host_array
from tengri.utils.physics_constants import C_AA, L_SUN
from tengri.utils.scale import apply_log10_scale, log10_magnitude

__all__ = ["PowerLawDisc"]

#: Single source of truth for every class-level prior below: the shared
#: ``agn_*`` declarations in ``_params.py``. Read once at class-definition
#: time so these class attributes cannot drift from the canonical
#: declaration the way they did before this fix (found by
#: ``tools/check_param_restatements.py``, Task 11 item 5 / fix round 1,
#: R25): ``log_lbol`` default 11.0 vs canonical 10.0, ``alpha`` bounds
#: (-1.5, -0.5) vs canonical (-2, 0), ``lum_ratio`` bounds (0, 1) vs
#: canonical (0, 5).
_LOG_LBOL_PRIOR = declared_prior(_AGN_PARAMS, "agn_log_lbol")
_ALPHA_PRIOR = declared_prior(_AGN_PARAMS, "agn_alpha")
_T_MAX_PRIOR = declared_prior(_AGN_PARAMS, "agn_T_max")
_LUM_RATIO_PRIOR = declared_prior(_AGN_PARAMS, "agn_lum_ratio")

#: Edges of the band the disc is defined on [Angstrom], ascending: the SED is zero outside it.
_PUBLISHED_BAND_AA = np.array([10.0, 1.0e8])

#: Gauss-Legendre panels per decade and points per panel of the band quadrature in ln(nu).
#: The integrand nu**(alpha+1) exp(-h nu / k T) is smooth in ln(nu); measured against 64
#: panels/decade x 32 points over the declared prior box (alpha -2 to 0, T_max 1e4 to 1e6 K),
#: 4 x 16 agrees to 8e-16 at 81 points (float64 rounding).
_BAND_PANELS_PER_DECADE: int = 4
_BAND_GL_ORDER: int = 16

#: ``log10(L_sun / (erg/s))`` [dex].
_LOG10_L_SUN: float = math.log10(L_SUN)


def _band_quadrature(band_aa, panels_per_decade, order):
    """Gauss-Legendre nodes ``nu`` [Hz] and ``d ln(nu)`` weights over a wavelength band [A]."""
    ln_lo = math.log(C_AA / band_aa[1])
    ln_hi = math.log(C_AA / band_aa[0])
    n_panel = round(panels_per_decade * math.log10(band_aa[1] / band_aa[0]))
    edges = np.linspace(ln_lo, ln_hi, n_panel + 1)
    x, w = np.polynomial.legendre.leggauss(order)
    half = 0.5 * (edges[1:] - edges[:-1])[:, None]
    mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
    return np.exp(mid + half * x).ravel(), (half * w * np.ones_like(x)).ravel()


_BAND_NU, _BAND_W = (
    host_array(a)
    for a in _band_quadrature(_PUBLISHED_BAND_AA, _BAND_PANELS_PER_DECADE, _BAND_GL_ORDER)
)
#: Lowest band frequency [Hz], the reference that keeps ``(nu / nu_ref)**alpha`` O(1).
_BAND_NU_REF: float = float(C_AA / _PUBLISHED_BAND_AA[1])


def band_limited_powerlaw_disc(wavelength, agn_log_lbol, agn_lum_ratio, agn_alpha, agn_T_max):
    r"""Power-law disc with exponential cutoff, emitted only inside its band.

    .. math::

        L_\nu = r\,10^{\log L_{\rm bol}} L_\odot\,
            \frac{\nu^\alpha e^{-h\nu/k_B T_{\rm max}}}
                 {\int_{\nu_1}^{\nu_2} \nu'^\alpha e^{-h\nu'/k_B T_{\rm max}}\,d\nu'}
            \quad (\nu_1 \le \nu \le \nu_2), \qquad L_\nu = 0 \ \text{otherwise},

    with :math:`\nu_1 = c/10^8` A and :math:`\nu_2 = c/10` A the band edges [Hz], :math:`r`
    the luminosity ratio [dimensionless], :math:`\alpha` the index [dimensionless],
    :math:`T_{\rm max}` the cutoff temperature [K], and :math:`h`, :math:`k_B` the Planck
    and Boltzmann constants. The band integral is Gauss-Legendre in :math:`\ln\nu`
    (``_BAND_PANELS_PER_DECADE`` panels per decade of ``_BAND_GL_ORDER`` points), converged
    to float64 rounding, so the SED carries exactly :math:`r\,10^{\log L_{\rm bol}} L_\odot`.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelengths [Angstrom].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)` [dex].
    agn_lum_ratio : float
        Disc luminosity ratio [dimensionless].
    agn_alpha : float
        Power-law index of :math:`L_\nu` [dimensionless].
    agn_T_max : float
        Cutoff temperature [K].

    Returns
    -------
    ndarray, shape (n_wave,)
        :math:`L_\nu` [erg/s/Hz], zero outside 10 A to 1e8 A.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes. The shape is that of ``powerlaw_disc`` in
    ``tengri.components.agn.disc``, a phenomenological form with no derivation in the
    literature, so there is no paper equation to cite. **Approximation**: the power law does
    not converge at long wavelength (:math:`\nu L_\nu` is constant at :math:`\alpha = -1`),
    so the band is part of the model's definition and the SED is truncated at its edges. The
    luminosity scale is applied as a log10 offset, so no erg/s-scale value is formed
    (float32-safe).
    """
    wave = jnp.asarray(wavelength)
    nu = C_AA / wave
    band_nu = device_table(_BAND_NU).astype(wave.dtype)
    band_w = device_table(_BAND_W).astype(wave.dtype)
    # int shape dnu = nu_ref**(alpha+1) sum_k w_k (nu_k/nu_ref)**(alpha+1) e^{-x_k}; the sum is
    # peak-factored (stop_gradient: the value does not depend on the factor) to stay O(1).
    terms = _powerlaw_shape_unit(band_nu, agn_alpha, agn_T_max, _BAND_NU_REF) * (
        band_nu / _BAND_NU_REF
    )
    peak = jax.lax.stop_gradient(jnp.max(terms))
    log_integral = jnp.log10(jnp.sum(band_w * terms / peak)) + jnp.log10(peak)
    log_scale = agn_log_lbol + _LOG10_L_SUN - log_integral - math.log10(_BAND_NU_REF)
    shape = _powerlaw_shape_unit(nu, agn_alpha, agn_T_max, _BAND_NU_REF)
    in_band = (wave >= _PUBLISHED_BAND_AA[0]) & (wave <= _PUBLISHED_BAND_AA[1])
    return agn_lum_ratio * jnp.where(in_band, apply_log10_scale(shape, log_scale), 0.0)


@dataclass(frozen=True)
class PowerLawDiscConfig(SEDComponentConfig):
    """Configuration for power-law AGN disc (minimal; no state)."""

    pass


@dataclass(frozen=True)
class PowerLawDisc(SEDModelComponent):
    """Simple power-law accretion disc with exponential UV cutoff.

    A fast phenomenological disc model: L_ν ∝ ν^α exp(-hν / k_B T_max) between 10 A and
    1e8 A, zero outside (see :func:`band_limited_powerlaw_disc`).
    Suitable for rapid fitting and when fine spectral details are not required.

    Attributes
    ----------
    name : str
        Component registry key: ``"powerlaw_disc"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : PowerLawDiscConfig
        Frozen configuration (unused; here for consistency).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    alpha : Uniform
        Power-law spectral index. [dimensionless, -2–0]
    T_max : Uniform
        UV cutoff temperature. [K, 10^4–10^6]
    lum_ratio : Uniform
        Fraction of bolometric luminosity from disc. [dimensionless, 0–5]

    Cross-component outputs
    -----------------------
    log_L_agn_disc : dex
        log10 of the bolometric luminosity of the disc [dex re erg/s]: the frequency integral
        of the emitted SED, which is zero outside 10 A - 1e8 A and carries the whole budget
        ``lum_ratio * 10**log_lbol L_sun``, formed in log space so it is finite in pure
        float32 (NAMING_CONTRACT section 4c). ``-inf`` is an absent disc. The disc has no
        inclination, so there is no ``_los`` sibling.

    Notes
    -----
    **JIT-compatible**: yes, predict() is pure JAX.

    **Approximation**: This model is a simplified phenomenological
    representation. It does not capture multi-zone temperature structure,
    soft X-ray excess, or hard X-ray corona. Use for fast prototyping only.

    Examples
    --------
    Minimal model with power-law disc::

        from tengri import SEDModel, Fixed, DEFAULT, Uniform, builders
        from tengri.components.agn.powerlaw_disc_model import PowerLawDisc

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(alpha=Fixed(1.5), beta=Fixed(1.0)),
            dust_attenuation={"type": "two_component", "all_params": Fixed(DEFAULT)},
            agn=PowerLawDisc(),
        )
    """

    name = "powerlaw_disc"
    parameter_prefix = "agn_"
    config: PowerLawDiscConfig = PowerLawDiscConfig()

    # Free parameters: auto-discovered
    log_lbol = Uniform(
        _LOG_LBOL_PRIOR.lo,
        _LOG_LBOL_PRIOR.hi,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=_LOG_LBOL_PRIOR.default,
    )
    alpha = Uniform(
        _ALPHA_PRIOR.lo,
        _ALPHA_PRIOR.hi,
        description="Power-law spectral index",
        units="dimensionless",
        default=_ALPHA_PRIOR.default,
    )
    T_max = Uniform(
        _T_MAX_PRIOR.lo,
        _T_MAX_PRIOR.hi,
        description="UV cutoff temperature",
        units="K",
        default=_T_MAX_PRIOR.default,
    )
    lum_ratio = Uniform(
        _LUM_RATIO_PRIOR.lo,
        _LUM_RATIO_PRIOR.hi,
        description="Disc luminosity fraction of L_bol",
        units="dimensionless",
        default=_LUM_RATIO_PRIOR.default,
    )

    # Cross-component output
    outputs: ClassVar[dict[str, str]] = with_lsun_outputs({"log_L_agn_disc": "dex"})

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX power-law disc prediction.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - alpha: power-law index
            - T_max: UV cutoff temperature (K)
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
            - published: {"log_L_agn_disc": log10 of the bolometric disc luminosity [dex re
              erg/s], ``log10(lum_ratio) + log_lbol + log10(L_sun)``}. The SED is zero outside
              10 A - 1e8 A and normalized so that its integral is this whole budget, so the
              closed form is the integral of the emitted SED (no ``_unemitted`` piece). The
              band is part of the definition: the power-law tail (L_nu nu ~ const at long
              wavelength) does not converge beyond it.

        """

        disc = band_limited_powerlaw_disc(
            wave, p["log_lbol"], p["lum_ratio"], p["alpha"], p["T_max"]
        )
        # The SED is normalized so its band integral is the whole budget: closed form.
        log_L_disc = log10_magnitude(p["lum_ratio"]) + p["log_lbol"] + _LOG10_L_SUN
        sed_out = sed_in + disc

        return sed_out, with_lsun_siblings({"log_L_agn_disc": log_L_disc})
