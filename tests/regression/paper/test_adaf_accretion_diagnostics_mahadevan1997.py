# SPDX-License-Identifier: BSD-3-Clause
"""ADAF accretion-side diagnostics (Mahadevan 1997, Eq. 4; Sect. 5.2) (#2842).

Paper: R. Mahadevan, ApJ 477, 585 (1997), arXiv:astro-ph/9609107.

``agn_log_lbol`` of an ADAF is the radiated luminosity :math:`L_{\\rm ADAF} = Q_{ie} + \\delta
Q^+` (Eqs. 13, 14), and the accretion rate follows from it (Sect. 5.2). The paper scales the
rate by the Eddington rate of Eq. 4,

    Mdot = mdot Mdot_Edd,   Mdot_Edd = L_Edd / (eta_eff c^2) = 1.39e18 m g/s,   eta_eff = 0.1,

so the ADAF's accretion power is Mdot c^2 = mdot * 1.39e18 * m * c^2 [erg/s]. The composable AGN
publishes ``agn_adaf_mdot`` (mdot, dimensionless) and ``log_L_agn_adaf_accretion``
(log10 of that power in erg/s) beside the unchanged radiated ``log_L_agn_bol``.

Everything below is held to the independent numpy / scipy solver of ``tests/_adaf_reference.py``
(exact K_2, its own Eq. 20 solve and a brentq inversion of Eqs. 13 and 14) and to the paper's
Eq. 4 constants written out in the test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.adaf import adaf_accretion_diagnostics
from tengri.components.agn.component import AGNSEDComponent, AGNSEDComponentConfig
from tengri.protocols.component import ForwardState
from tengri.utils.physics_constants import L_SUN
from tests import _adaf_reference as ref

pytestmark = pytest.mark.regression_paper

# Mahadevan 1997, Eq. 4.
_MDOT_EDD_PER_MSUN = 1.39e18  # g/s per solar mass
_C = 2.99792458e10  # cm/s

# (log10 L_ADAF / L_sun, log10 M / M_sun, alpha, beta, delta): the default point and box edges.
_POINTS = [
    (8.0, 7.0, 0.3, 0.5, 0.1),
    (9.0, 7.0, 0.3, 0.5, 0.1),
    (8.0, 10.0, 0.5, 0.1, 0.5),
    (8.0, 10.0, 0.05, 0.9, 1e-3),
    (8.5, 8.5, 0.4, 0.3, 0.05),
    (9.5, 9.0, 0.4, 0.7, 0.3),
]


def _reference(point):
    log_l, log_m, alpha, beta, delta = point
    m = 10.0**log_m
    mdot, _, clipped = ref.ref_mdot_from_luminosity(10.0**log_l * L_SUN, m, alpha, beta, delta)
    assert not clipped, point
    return mdot, m


@pytest.mark.parametrize("point", _POINTS)
def test_mdot_is_the_independent_inversion_of_the_radiated_luminosity(point):
    """mdot equals a scipy inversion of L_ADAF = Q_ie + delta Q+ (Eqs. 13, 14) to 1e-6."""
    log_l, log_m, alpha, beta, delta = point
    mdot, _ = adaf_accretion_diagnostics(log_l, log_m, alpha, beta, delta)
    assert float(mdot) == pytest.approx(_reference(point)[0], rel=1e-6)


@pytest.mark.parametrize("point", _POINTS)
def test_accretion_power_is_mdot_times_the_eddington_accretion_power_of_eq4(point):
    """log10(Mdot c^2) = log10(mdot * 1.39e18 m * c^2), the Eq. 4 constant, to 1e-6 dex."""
    log_l, log_m, alpha, beta, delta = point
    mdot_ref, m = _reference(point)
    _, log_power = adaf_accretion_diagnostics(log_l, log_m, alpha, beta, delta)
    expected = np.log10(mdot_ref * _MDOT_EDD_PER_MSUN * m * _C**2)
    assert float(log_power) == pytest.approx(expected, abs=1e-6)


def test_critical_rate_clip_carries_through_to_the_accretion_power():
    """Above the luminosity of mdot_crit = 0.28 alpha^2 (Eq. 52) mdot is the critical rate."""
    log_m, alpha, beta, delta = 8.0, 0.3, 0.5, 0.1
    m = 10.0**log_m
    lum_crit, _ = ref.ref_luminosity(0.28 * alpha**2, m, alpha, beta, delta)
    log_l = np.log10(3.0 * lum_crit / L_SUN)
    mdot, log_power = adaf_accretion_diagnostics(log_l, log_m, alpha, beta, delta)
    assert float(mdot) == 0.28 * alpha**2
    assert float(log_power) == pytest.approx(
        np.log10(0.28 * alpha**2 * _MDOT_EDD_PER_MSUN * m * _C**2), abs=1e-12
    )


def test_float32_accretion_diagnostics_are_finite_and_accurate_over_the_box():
    """Pure float32 reproduces float64 over the declared box (1e-4, the ADAF float32 tolerance)."""
    rng = np.random.default_rng(2842)
    for _ in range(12):
        log_l, log_m = rng.uniform(8.0, 14.0), rng.uniform(6.0, 10.0)
        alpha, beta = rng.uniform(0.05, 0.5), rng.uniform(0.1, 0.9)
        delta = 10.0 ** rng.uniform(-3.0, np.log10(0.5))
        mdot64, power64 = adaf_accretion_diagnostics(log_l, log_m, alpha, beta, delta)
        with jax.enable_x64(False):
            mdot32, power32 = adaf_accretion_diagnostics(
                jnp.float32(log_l),
                jnp.float32(log_m),
                jnp.float32(alpha),
                jnp.float32(beta),
                jnp.float32(delta),
                dtype=jnp.float32,
            )
            mdot32, power32 = float(mdot32), float(power32)
        assert np.isfinite(mdot32) and np.isfinite(power32)
        assert mdot32 == pytest.approx(float(mdot64), rel=1e-4)
        assert 10.0 ** (power32 - float(power64)) == pytest.approx(1.0, abs=1e-4)


def _adaf_component():
    return AGNSEDComponent(
        config=AGNSEDComponentConfig(
            model="composable",
            agn_disc_block="adaf",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_norm="independent",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
        )
    )


def test_composable_agn_publishes_the_accretion_side_beside_the_radiated_luminosity():
    """The composable ADAF publishes mdot and Mdot c^2; ``log_L_agn_bol`` stays the radiated L."""
    log_l, log_m, alpha, beta, delta = 9.0, 7.5, 0.3, 0.5, 0.1
    params = {
        "agn_log_lbol": jnp.asarray(log_l),
        "agn_log_mbh": jnp.asarray(log_m),
        "agn_adaf_alpha": jnp.asarray(alpha),
        "agn_adaf_beta": jnp.asarray(beta),
        "agn_adaf_delta": jnp.asarray(delta),
    }
    out = _adaf_component().apply(ForwardState(wave=jnp.geomspace(100.0, 1.0e8, 200)), params)
    mdot_ref, m = _reference((log_l, log_m, alpha, beta, delta))
    assert float(out.derived["agn_adaf_mdot"]) == pytest.approx(mdot_ref, rel=1e-6)
    accretion = float(out.derived["log_L_agn_adaf_accretion"])
    assert accretion == pytest.approx(
        np.log10(mdot_ref * _MDOT_EDD_PER_MSUN * m * _C**2), abs=1e-6
    )
    # The parameter stays the radiated luminosity: 10**log_l L_sun, not the accretion power.
    assert float(out.derived["log_L_agn_bol"]) == pytest.approx(log_l + np.log10(L_SUN), abs=1e-12)
    assert accretion > float(out.derived["log_L_agn_bol"])


def test_other_disc_blocks_do_not_publish_adaf_diagnostics():
    """A non-ADAF composition carries no ``agn_adaf_mdot`` (the key is ADAF-specific)."""
    component = AGNSEDComponent(
        config=AGNSEDComponentConfig(
            model="composable",
            agn_disc_block="multicolor",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_norm="independent",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
        )
    )
    params = {"agn_log_lbol": jnp.asarray(11.0), "agn_log_mbh": jnp.asarray(8.0)}
    out = component.apply(ForwardState(wave=jnp.geomspace(100.0, 1.0e8, 200)), params)
    assert "agn_adaf_mdot" not in out.derived
