# SPDX-License-Identifier: BSD-3-Clause
"""Analytic dust models carry da Cunha et al. (2013) CMB luminosity boost (#2806).

``modified_blackbody``, ``graybody`` and ``casey2012`` heat their dust to
``T_d(z)`` (Eq. 12) and must emit the CMB energy the dust absorbs, so the
emitted total is ``L_absorbed * [T_d(z)/T_d0]^(4+beta)`` (Sect. 2.2, Eqs. 11
and 17). The contrast of Eq. 18 is applied afterwards and is not part of the
emitted power, so every integral below divides the contrast out first.

Every expected value is written here from the paper's formulas in numpy, never
read back from the code under test. The only shared quantity is the CMB
temperature, which is the unified :data:`tengri.utils.physics_constants.T_CMB_0`
(Fixsen 2009), and a test pins that constant directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.emission import _cmb, _physics
from tengri.components.dust.emission.analytic._closures import (
    casey2012,
    graybody,
    modified_blackbody,
)
from tengri.utils import cosmology
from tengri.utils.physics_constants import T_CMB_0

T_D0_K = 20.0
BETA = 1.8
Z_HIGH = 6.0
C_CGS_NUM = 2.99792458e10
T_CMB_FIXSEN_K = 2.7255  # Fixsen 2009; the value T_CMB_0 must carry

MODELS = (
    ("modified_blackbody", modified_blackbody),
    ("graybody", graybody),
    ("casey2012", casey2012),
)


def _wave_grid_aa() -> np.ndarray:
    """Rest-frame grid from 1 um to 1 cm, which holds the bulk of the power at 20 K."""
    return np.geomspace(1.0e4, 1.0e8, 6000)


def _t_heated_k(z: float) -> float:
    """Eq. 12 of da Cunha et al. (2013): T_d(z)^(4+b) = T_d0^(4+b) + T_CMB(z)^(4+b) - T_CMB0."""
    e = 4.0 + BETA
    t_cmb_z = T_CMB_FIXSEN_K * (1.0 + z)
    return (T_D0_K**e + t_cmb_z**e - T_CMB_FIXSEN_K**e) ** (1.0 / e)


def _contrast_numpy(wave_aa: np.ndarray, t_eff: float, z: float) -> np.ndarray:
    """Eq. 18: C = 1 - B(T_CMB(z)) / B(T_eff), with nu^3 canceling in the ratio."""
    h_over_k_cm = 4.799243073e-11  # h/k_B [s K], Planck and Boltzmann CODATA 2018
    nu = C_CGS_NUM / (wave_aa * 1.0e-8)
    x_eff = h_over_k_cm * nu / t_eff
    x_cmb = h_over_k_cm * nu / (T_CMB_FIXSEN_K * (1.0 + z))
    # B(T_CMB)/B(T_eff) = (e^x_eff - 1)/(e^x_cmb - 1)
    #   = exp(x_eff - x_cmb) (1 - e^-x_eff)/(1 - e^-x_cmb),
    # the second form stable for x_cmb > x_eff at every wavelength.
    ratio = np.exp(x_eff - x_cmb) * (-np.expm1(-x_eff)) / (-np.expm1(-x_cmb))
    return 1.0 - ratio


def _emitted_integral(wave_aa: np.ndarray, out: np.ndarray, z: float) -> float:
    """Frequency integral of the pre-contrast emitted spectrum, L_nu / C_nu."""
    t_eff = _t_heated_k(z)
    contrast = _contrast_numpy(wave_aa, t_eff, z)
    emitted = np.asarray(out) / contrast
    nu = C_CGS_NUM / (wave_aa * 1.0e-8)
    return float(-np.trapezoid(emitted, nu))


def _closure_output(fn, wave_aa: np.ndarray, z: float) -> np.ndarray:
    return np.asarray(fn(jnp.asarray(wave_aa), 1.0, dust_T=T_D0_K, dust_beta_ir=BETA, redshift=z))


@pytest.mark.regression_paper
@pytest.mark.parametrize(("name", "fn"), MODELS, ids=[m[0] for m in MODELS])
def test_emitted_power_ratio_is_heated_boost_eq11(name, fn):
    """Emitted power at z = 6 over z = 0 equals [T_d(6)/T_d0]^(4+beta), Eq. 11 (#2806).

    Conservation of absorbed power: with the contrast divided out, the integral
    of the emitted spectrum is ``L_absorbed`` times the boost, so the ratio at
    the same ``L_absorbed`` is the boost alone. The expected value is the paper's
    Eq. 12 temperature inserted into Eq. 11. Tolerance rtol 1e-6, which is looser
    than the float64 round-off of the trapezoid on this grid.
    """
    wave = _wave_grid_aa()
    i_high = _emitted_integral(wave, _closure_output(fn, wave, Z_HIGH), Z_HIGH)
    i_zero = _emitted_integral(wave, _closure_output(fn, wave, 0.0), 0.0)
    expected = (_t_heated_k(Z_HIGH) / T_D0_K) ** (4.0 + BETA)
    assert i_high / i_zero == pytest.approx(expected, rel=1e-6)
    assert 1.5 < expected < 2.0  # sanity: ~1.76 for 20 K at z = 6, beta = 1.8


@pytest.mark.limit
@pytest.mark.parametrize(("name", "fn"), MODELS, ids=[m[0] for m in MODELS])
def test_no_heating_at_z0_emits_exactly_l_absorbed(name, fn):
    """At z = 0 there is no CMB heating: the emitted total is L_absorbed (#2806).

    Limit: the boost is exactly 1 at z = 0, so the pre-contrast integral equals
    the absorbed luminosity on the grid to float64 round-off.
    """
    wave = _wave_grid_aa()
    i_zero = _emitted_integral(wave, _closure_output(fn, wave, 0.0), 0.0)
    assert i_zero == pytest.approx(1.0, rel=1e-10)


@pytest.mark.limit
def test_boost_is_exactly_one_at_z0():
    """The closed-form boost is a bit-exact 1.0 at z = 0 (#2806).

    The bracket ``(1 + z)^(4+beta) - 1`` vanishes exactly at z = 0, so the boost
    is 1.0 with no branch. Every z = 0 output is then bit-identical to the
    pre-#2806 code at the same T_CMB,0.
    """
    boost = _physics.cmb_luminosity_boost(jnp.asarray(T_D0_K), jnp.asarray(0.0), BETA)
    assert float(boost) == 1.0


#: Schreiber 2016 CMB output, sampled at 1, 10, 250, 500 and 1000 um rest on the
#: grid ``np.geomspace(1e3, 3e7, 4000)``, captured once at commit d0c7c4d0d (#2806),
#: i.e. before the single-helper consolidation, with the same T_CMB0 = 2.7255 K. A
#: relative change of 1e-12 or more would mean the consolidation moved Schreiber.
_SCHREIBER_PRE_CONSOLIDATION = {
    ("schreiber2016", 35.0, 4.0): (
        8.600009205691639e-18,
        2.706996734043223e-15,
        2.5149958069289976e-14,
        2.8843073691739244e-15,
        2.7084580446479655e-16,
    ),
    ("schreiber2016", 25.0, 6.0): (
        1.0343067311084527e-17,
        3.1217187626327653e-15,
        5.0380982468218995e-14,
        5.888791879477046e-15,
        5.519857409689982e-16,
    ),
    ("schreiber2016", 50.0, 10.0): (
        9.027684718701017e-18,
        3.245272454077447e-15,
        4.7768895923367924e-15,
        4.3779951040651277e-16,
        3.8532434487345546e-17,
    ),
}
_SCHREIBER_INDICES = (893, 1786, 3035, 3304, 3573)


@pytest.mark.regression_paper
def test_schreiber_cmb_output_matches_pre_consolidation_values():
    """Schreiber's CMB output is unchanged by the consolidated boost helper (#2806).

    The Schreiber path now multiplies by ``cmb_luminosity_boost(T_d, z, beta)`` at
    the same redshift its heated temperature came from. Its sampled output matches
    the values captured before the change to 1e-12 relative.
    """
    from tengri.forward.component_factory import _resolve_registry_component

    wave = jnp.geomspace(1.0e3, 3.0e7, 4000)
    for (name, t_dust, z), expected in _SCHREIBER_PRE_CONSOLIDATION.items():
        comp = _resolve_registry_component("dust_emission", name, config=None)
        comp.cmb = True
        lib = _resolve_registry_component("dust_emission", name, config=None).load(None)
        sed, _ = comp.predict(
            {"T": t_dust, "f_pah": 0.05, "redshift": z},
            jnp.zeros_like(wave),
            wave,
            L_ir=1.0,
            templates=lib,
        )
        got = np.asarray(sed)[list(_SCHREIBER_INDICES)]
        np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-12, atol=0.0)


@pytest.mark.gradient
def test_schreiber_redshift_gradient_at_z0_is_finite_and_nonzero():
    """Schreiber's CMB output has a finite, nonzero z-derivative at z = 0 (#2806).

    The boost bracket ``(1+z)^(4+beta) - 1`` vanishes at z = 0 but its slope does
    not, and the contrast's z-dependence is live there, so the derivative of the
    observed flux at 500 um is nonzero. A branch-selected boost would zero the slope.
    """
    from tengri.forward.component_factory import _resolve_registry_component

    wave = jnp.geomspace(1.0e3, 3.0e7, 4000)
    comp = _resolve_registry_component("dust_emission", "schreiber2016", config=None)
    comp.cmb = True
    lib = _resolve_registry_component("dust_emission", "schreiber2016", config=None).load(None)

    def flux_500(z):
        sed, _ = comp.predict(
            {"T": 35.0, "f_pah": 0.05, "redshift": z},
            jnp.zeros_like(wave),
            wave,
            L_ir=1.0,
            templates=lib,
        )
        return jnp.interp(5.0e6, wave, sed)

    g = float(jax.grad(flux_500)(jnp.asarray(0.0)))
    assert np.isfinite(g) and g != 0.0


@pytest.mark.gradient
@pytest.mark.parametrize(("name", "fn"), MODELS, ids=[m[0] for m in MODELS])
def test_gradients_wrt_dust_temperature_and_redshift_are_finite(name, fn):
    """The emitted total is differentiable in T_d and z at z = 0 and z = 6 (#2806).

    The gradient of the summed output with respect to ``dust_T`` and ``redshift``
    is finite and not identically zero at both redshifts. At z = 0 the
    autodiff gradient in z matches the one-sided finite difference toward z > 0,
    so the boost contributes its slope at the boundary and no branch zeroes it.
    """
    wave = jnp.asarray(_wave_grid_aa())

    def total(t_d, z):
        return jnp.sum(fn(wave, 1.0, dust_T=t_d, dust_beta_ir=BETA, redshift=z))

    grad_t = jax.grad(total, argnums=0)
    grad_z = jax.grad(total, argnums=1)
    for z in (0.0, Z_HIGH):
        gt = np.asarray(grad_t(jnp.asarray(T_D0_K), jnp.asarray(z)))
        gz = np.asarray(grad_z(jnp.asarray(T_D0_K), jnp.asarray(z)))
        assert np.all(np.isfinite(gt)) and np.all(np.isfinite(gz))
        assert np.any(gt != 0.0) and np.any(gz != 0.0)
    h = 1.0e-6
    fd = (
        float(total(jnp.asarray(T_D0_K), jnp.asarray(h)))
        - float(total(jnp.asarray(T_D0_K), jnp.asarray(0.0)))
    ) / h
    assert float(grad_z(jnp.asarray(T_D0_K), jnp.asarray(0.0))) == pytest.approx(fd, rel=1e-4)


@pytest.mark.contract
def test_one_boost_helper_serves_schreiber_and_analytic():
    """One CMB luminosity boost helper, shared by the tabulated and analytic paths (#2806).

    The old helper taking the heated temperature is gone; ``_cmb`` re-exports the
    single ``cmb_luminosity_boost``.
    """
    assert _cmb.cmb_luminosity_boost is _physics.cmb_luminosity_boost
    assert not hasattr(_physics, "cmb_heating_luminosity_boost")


@pytest.mark.contract
def test_t_cmb0_is_one_constant_at_2_7255_k():
    """T_CMB0 is one constant, 2.7255 K (Fixsen 2009), used by physics and cosmology (#2806).

    The per-cosmology ``Tcmb0`` entries (WMAP5 2.725 K) are separate published
    values and are not changed here.
    """
    assert T_CMB_0 == T_CMB_FIXSEN_K
    assert _physics._T_CMB_0 == T_CMB_FIXSEN_K
    assert cosmology._T_REF_K == T_CMB_FIXSEN_K
