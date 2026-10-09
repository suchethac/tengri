# SPDX-License-Identifier: BSD-3-Clause
"""CMB heating and contrast for the tabulated dust-emission models (#2766).

da Cunha et al. (2013) [1]_ show that at high redshift the cosmic microwave
background (i) heats the dust and (ii) is the background the dust emission is
measured against. The analytic models (``modified_blackbody``, ``graybody``,
``casey2012``) carry both effects unconditionally. This module supplies the
same two effects to the tabulated models, as an opt-in
(``dust_emission={'cmb': True}``), by reusing the very functions the analytic
closures call
(``tengri.components.dust.emission._physics.cmb_corrected_temperature``,
Eq. 12, and
``tengri.components.dust.emission._physics.cmb_contrast_factor``, Eq. 18),
so one CMB temperature and one set of equations serve every model.

What a tabulated model can support depends on whether it has a dust temperature:

* **Single-temperature libraries** (``schreiber2016``, ``schreiber2018``) take
  the dust temperature ``T`` as a parameter. Heating and contrast are both
  applied as in the paper: the template is read at the heated temperature
  :math:`T_{\\rm d}(z)` and scaled by the luminosity boost the paper derives
  (see :func:`cmb_heating_luminosity_boost`), then multiplied by the contrast.
* **Radiation-field libraries** (Draine & Li, Dale et al., THEMIS, Astrodust,
  Dale & Helou / Chary & Elbaz) have no dust temperature: a spectrum is a
  mixture of grains at many temperatures set by a radiation field. Heating has
  no template-level form there (it would shift each grain population by its own
  amount, and the stochastically heated PAH and small-grain emission by none),
  so only the contrast is applied, with the temperature read off the template
  by :func:`template_far_ir_temperature`.

References
----------
.. [1] da Cunha, E., Groves, B., Walter, F., et al., 2013, "On the Effect of
   the Cosmic Microwave Background in High-redshift (Sub-)millimeter
   Observations", ApJ, 766, 13. arXiv:1302.0844.
   https://doi.org/10.1088/0004-637X/766/1/13
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.dust._params import ANALYTIC_BETA_IR_DEFAULT
from tengri.components.dust.emission._physics import (
    cmb_contrast_factor,
    planck_bnu,
)
from tengri.utils.physics_constants import AA_TO_CM, C_CGS

__all__ = [
    "CMB_FAR_IR_MIN_AA",
    "CMB_TEMPLATE_BETA_IR",
    "cmb_heating_luminosity_boost",
    "cmb_observed_emission",
    "template_far_ir_temperature",
]

#: Emissivity index :math:`\beta` in Eq. 12 and in the template temperature
#: estimate. The tabulated models carry no :math:`\beta` of their own, so this
#: is the value the analytic models default to (and the one the issue's
#: contrast table was computed with). [dimensionless]
CMB_TEMPLATE_BETA_IR: float = ANALYTIC_BETA_IR_DEFAULT

#: Shortest rest-frame wavelength used to read a temperature off a template.
#: Below it the stochastically heated PAH and small-grain emission, which no
#: single temperature describes, dominates the power. [Angstrom]
CMB_FAR_IR_MIN_AA: float = 5.0e5

#: Temperature axis of the modified-blackbody lookup that inverts the template
#: statistic; spans every dust temperature the CMB can matter for. [K]
_T_LOOKUP_K = np.geomspace(4.0, 250.0, 96)

_THZ = 1.0e12  # Hz per THz: keeps the frequency moments O(1) in float32


def cmb_heating_luminosity_boost(
    T_dust: jnp.ndarray,
    T_eff: jnp.ndarray,
    beta: float = CMB_TEMPLATE_BETA_IR,
) -> jnp.ndarray:
    r"""Factor by which CMB heating raises the intrinsic dust luminosity.

    Parameters
    ----------
    T_dust : array_like
        Intrinsic (:math:`z = 0`) dust temperature. [K]
    T_eff : array_like
        CMB-heated dust temperature :math:`T_{\rm d}(z)` from
        ``tengri.components.dust.emission._physics.cmb_corrected_temperature``. [K]
    beta : float
        Dust emissivity index. [dimensionless]

    Returns
    -------
    ndarray
        :math:`L_{\rm dust}(z) / L_{\rm dust}(z=0)`, at least 1. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes; ``T_dust`` is floored at 1 K, as in
    ``tengri.components.dust.emission._physics.cmb_corrected_temperature``.

    With the stellar heating unchanged, the dust also absorbs the CMB, so its
    total emitted power grows with temperature as Eq. 11 of da Cunha et al.
    (2013) [1]_ (:math:`\int \nu^\beta B_\nu(T)\, d\nu \propto T^{4+\beta}`):

    .. math::

        \frac{L_{\rm dust}(z)}{L_{\rm dust}(0)}
        = \left[\frac{T_{\rm d}(z)}{T_{\rm d}^{z=0}}\right]^{4+\beta}
        = 1 + \left(\frac{T_{\rm CMB}^{z=0}}{T_{\rm d}^{z=0}}\right)^{4+\beta}
        \left[(1+z)^{4+\beta} - 1\right]

    The paper states this as point (ii) of Sec. 2.2. It is the energy that the
    contrast factor (Eq. 18) later removes again from what is observed.

    References
    ----------
    .. [1] da Cunha et al. 2013, ApJ, 766, 13. https://doi.org/10.1088/0004-637X/766/1/13
    """
    # Scale floor: dust_T has prior 20-80 K (library 15-99 K); the 1 K floor never binds.
    return (T_eff / jnp.maximum(T_dust, 1.0)) ** (4.0 + beta)


def template_far_ir_temperature(
    wave_aa: jnp.ndarray,
    lnu: jnp.ndarray,
    beta: float = CMB_TEMPLATE_BETA_IR,
) -> jnp.ndarray:
    r"""Dust temperature of a tabulated far-IR spectrum, as a modified blackbody.

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Rest-frame wavelength grid, ascending. [Å]
    lnu : array_like, shape (n_wave,)
        Template :math:`L_\nu`, any positive scale. [erg/s/Hz]
    beta : float
        Emissivity index of the reference modified blackbody. [dimensionless]

    Returns
    -------
    ndarray
        The temperature of the :math:`\nu^\beta B_\nu(T)` spectrum whose
        far-IR mean frequency equals the template's. [K] A template with no
        power at :math:`\lambda \ge 50\,\mu{\rm m}` on this grid returns the
        top of the lookup (250 K), for which the contrast factor is ~1.

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes; piecewise linear in the template (the inversion is
    a ``jnp.interp``), constant where the statistic leaves the lookup range.

    **Approximation**: a radiation-field template is not a single-temperature
    spectrum, so no unique :math:`T_{\rm d}` exists. This is the temperature of
    the modified blackbody with the same far-IR mean frequency,

    .. math::

        \langle\nu\rangle = \frac{\int_{\lambda \ge 50\,\mu{\rm m}} \nu\, L_\nu\, d\nu}
        {\int_{\lambda \ge 50\,\mu{\rm m}} L_\nu\, d\nu},

    with the same band cut applied to the reference spectrum, so the cut
    cancels for a template that is a modified blackbody with this ``beta``
    (the estimate recovers its temperature). The result is the temperature of
    the far-IR emission; it is not a mass-weighted temperature, and the warm
    stochastic emission longward of 50 micron biases it high by a few K.

    Approximation of Eq. 18 in da Cunha et al. (2013): exact for a
    single-temperature modified blackbody (temperature recovered to < 0.03 %);
    for a two-temperature mixture the mean-frequency temperature leans to the
    hot component while the CMB penalty is set by the cold one, so the detected
    flux is overestimated by about 5-15 % at z = 4 and 20-100 % at z = 6 when a
    20-25 K cold component carries a large share (2-8 % when it is a 5 % hot
    component).

    Implements the estimate from the template's own far-IR moment rather than a
    catalog of per-model temperatures, so one definition serves every library.
    """
    wave = jnp.asarray(wave_aa)
    nu_thz = C_CGS / (wave * AA_TO_CM) / _THZ
    band = wave >= CMB_FAR_IR_MIN_AA

    def _mean_nu(shape: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # nu falls as wavelength rises, so both integrals carry the same sign.
        top = jnp.max(jnp.where(band, shape, 0.0))
        shape_n = jnp.where(band, shape / jnp.where(top > 0.0, top, 1.0), 0.0)
        den = -jnp.trapezoid(shape_n, nu_thz)
        num = -jnp.trapezoid(shape_n * nu_thz, nu_thz)
        return num, den

    temps = jnp.asarray(_T_LOOKUP_K, dtype=wave.dtype)
    emissivity = (nu_thz * _THZ) ** beta

    def _reference(T: jnp.ndarray) -> jnp.ndarray:
        num, den = _mean_nu(emissivity * planck_bnu(wave, T))
        return num / den

    reference = jax.vmap(_reference)(temps)
    num, den = _mean_nu(jnp.asarray(lnu))
    has_power = den > 0.0
    mean_nu = jnp.where(has_power, num / jnp.where(has_power, den, 1.0), reference[-1])
    return jnp.interp(mean_nu, reference, temps)


def cmb_observed_emission(
    wave_aa: jnp.ndarray,
    emission: jnp.ndarray,
    redshift: jnp.ndarray,
    T_contrast: jnp.ndarray,
) -> jnp.ndarray:
    r"""Multiply a dust emission spectrum by the CMB contrast factor.

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    emission : array_like, shape (n_wave,)
        Intrinsic dust :math:`L_\nu`. [erg/s/Hz]
    redshift : array_like
        Source redshift. [dimensionless]
    T_contrast : array_like
        Dust temperature :math:`T_{\rm d}(z)` entering Eq. 18. [K]

    Returns
    -------
    ndarray, shape (n_wave,)
        The part of the emission detectable against the CMB. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    **Gradient-safe**: yes, see
    ``tengri.components.dust.emission._physics.cmb_contrast_factor``.

    .. math::

        L_\nu^{\rm obs} = L_\nu \left[1 - \frac{B_\nu(T_{\rm CMB}(z))}
        {B_\nu(T_{\rm d}(z))}\right]

    Eq. 18 of da Cunha et al. (2013) [1]_, evaluated at rest-frame
    frequency, where :math:`T_{\rm CMB}(z) = T_{\rm CMB}^{z=0}(1+z)`.

    References
    ----------
    .. [1] da Cunha et al. 2013, ApJ, 766, 13. https://doi.org/10.1088/0004-637X/766/1/13
    """
    return emission * cmb_contrast_factor(wave_aa, T_contrast, redshift)
