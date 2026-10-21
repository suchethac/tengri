# SPDX-License-Identifier: BSD-3-Clause
"""The standalone power-law AGN disc closes its energy ledger on the accretion power.

``PowerLawDisc`` publishes ``log_L_agn_disc`` [dex re erg/s] and ``L_agn_disc_lsun``. The
ledger conventions of NAMING_CONTRACT section 4c ("Standalone discs"):

* C1: the disc has no inclination, so ``log_L_agn_disc`` is its sphere mean and there is no
  ``_los`` key;
* C2: ``log_L_agn_disc`` (there is no unemitted part: the SED is the whole model) equals
  ``lum_ratio * 10**agn_log_lbol L_sun`` to 1e-10;
* C3: it equals the dense-grid integral of the emitted SED to 1e-6, and the SED is exactly
  zero outside the band 10 A to 1e8 A it is defined on;
* C4: the keys agree between pure float32 and float64 to 1.5e-5, and the ``_lsun`` sibling is
  the ``log`` key in solar luminosities.

The dense integral is the trapezoid in :math:`\\ln\\nu` on 1e5 log-uniform nodes with the band
edges as nodes, independent of the component's own quadrature (a power law between the 8001
nodes of its normalization). The measured C3 floor is 3.4e-7: the normalization's interpolant
against the smooth shape for the steepest cut-off of the prior box.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.physics_constants import L_SUN
from tests.physics._agn_published import (
    box_points,
    build_component,
    declared_box,
    predict,
    relative_error,
)

pytestmark = pytest.mark.contract

_C_AA = 2.99792458e18

#: Band the disc is defined on [Angstrom]; nodes at both edges.
_DENSE = np.geomspace(10.0, 1.0e8, 100_001)
#: Wavelengths outside the band [Angstrom].
_OUTSIDE_BAND = np.array([1.0e-3, 1.0, 9.999, 1.0001e8, 1.0e9, 1.0e10])
_CALLER = np.geomspace(100.0, 1.0e6, 30)

_C2_RTOL = 1.0e-10
_C3_RTOL = 1.0e-6
_F32_RTOL = 1.5e-5


@pytest.fixture(scope="module")
def powerlaw():
    """The component and its box points (center, every edge, 6 draws), ``lum_ratio > 0``."""
    comp = build_component("powerlaw")
    pts = box_points(declared_box(comp), 6, seed=2026)
    return comp, [p for p in pts if p["lum_ratio"] > 0.0]


def _predict64(comp, params, wave):
    """Float64 SED on ``wave`` [erg/s/Hz] and the published values as floats."""
    w = jnp.asarray(wave, dtype=jnp.float64)
    p = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
    sed, pub = comp.predict(p, jnp.zeros_like(w), w)
    return np.asarray(sed), {k: float(v) for k, v in pub.items()}


def _ln_nu_power(sed, wave):
    """Trapezoid in ln(nu) of L_nu nu [erg/s] on ``wave`` [Angstrom]."""
    nu = _C_AA / np.asarray(wave)
    order = np.argsort(nu)
    return float(np.trapezoid((np.asarray(sed) * nu)[order], np.log(nu[order])))


@pytest.mark.conservation
def test_published_power_is_the_accretion_budget(powerlaw):
    """C1, C2: one isotropic key, equal to ``lum_ratio 10**log_lbol L_sun`` to 1e-10."""
    comp, pts = powerlaw
    for params in pts:
        _, pub = _predict64(comp, params, _CALLER)
        assert {k for k in pub if k.startswith("log_L_agn_")} == {"log_L_agn_disc"}
        budget = params["lum_ratio"] * 10.0 ** params["log_lbol"] * L_SUN
        err = abs(10.0 ** pub["log_L_agn_disc"] / budget - 1.0)
        assert err < _C2_RTOL, f"powerlaw at {params}: key / budget - 1 = {err:.2e}"


@pytest.mark.conservation
def test_emitted_sed_integrates_to_its_published_key(powerlaw):
    """C3: the dense integral of the emitted SED over its band equals the key to 1e-6."""
    comp, pts = powerlaw
    for params in pts:
        sed, pub = _predict64(comp, params, _DENSE)
        err = abs(_ln_nu_power(sed, _DENSE) / 10.0 ** pub["log_L_agn_disc"] - 1.0)
        assert err < _C3_RTOL, f"powerlaw at {params}: integral / key - 1 = {err:.2e}"


def test_powerlaw_disc_emits_nothing_outside_its_band(powerlaw):
    """C3: the SED is exactly zero outside 10 A to 1e8 A, so nothing escapes the ledger."""
    comp, pts = powerlaw
    for params in pts:
        sed, _ = _predict64(comp, params, _OUTSIDE_BAND)
        assert np.all(sed == 0.0), f"powerlaw at {params}: outside-band L_nu {sed}"


def test_published_keys_are_float32_accurate(powerlaw):
    """C4: every published key in pure float32 matches float64 to 1.5e-5."""
    comp, pts = powerlaw
    for params in pts:
        pub32, _ = predict(comp, params, float32=True)
        pub64, _ = predict(comp, params, float32=False)
        assert pub32.keys() == pub64.keys()
        for key, v64 in pub64.items():
            err = relative_error(key, pub32[key], v64)
            assert err < _F32_RTOL, f"powerlaw.{key} at {params}: float32 rel err {err:.2e}"


def test_lsun_sibling_is_the_log_key_in_solar_units(powerlaw):
    """C4: ``L_agn_disc_lsun`` = ``10**log_L_agn_disc / L_sun``."""
    comp, pts = powerlaw
    _, pub = _predict64(comp, pts[0], _CALLER)
    assert pub["L_agn_disc_lsun"] == pytest.approx(
        10.0 ** pub["log_L_agn_disc"] / L_SUN, rel=1e-12
    )


def test_absent_disc_publishes_minus_infinity_and_emits_nothing(powerlaw):
    """``lum_ratio = 0``: the ``log`` key is ``-inf``, the sibling 0, the SED zero."""
    comp, pts = powerlaw
    sed, pub = _predict64(comp, {**pts[0], "lum_ratio": 0.0}, _CALLER)
    assert np.all(sed == 0.0)
    assert pub["log_L_agn_disc"] == -np.inf
    assert pub["L_agn_disc_lsun"] == 0.0


def test_truncated_disc_keeps_finite_gradients(powerlaw):
    """The band truncation leaves finite gradients in alpha, T_max and log_lbol."""
    comp, pts = powerlaw
    wave = jnp.asarray(np.geomspace(1.0, 1.0e9, 200))
    base = {k: jnp.asarray(v) for k, v in pts[0].items()}

    def total(alpha, t_max, log_lbol):
        p = {**base, "alpha": alpha, "T_max": t_max, "log_lbol": log_lbol}
        sed, pub = comp.predict(p, jnp.zeros_like(wave), wave)
        return jnp.sum(jnp.log(jnp.where(sed > 0, sed, 1.0))) + pub["log_L_agn_disc"]

    grads = jax.grad(total, argnums=(0, 1, 2))(base["alpha"], base["T_max"], base["log_lbol"])
    assert all(np.isfinite(float(g)) for g in grads), f"non-finite gradient: {grads}"
    assert all(float(g) != 0.0 for g in grads), f"a parameter lost its gradient: {grads}"
