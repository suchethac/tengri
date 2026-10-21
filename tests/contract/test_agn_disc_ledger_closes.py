# SPDX-License-Identifier: BSD-3-Clause
"""The standalone AGN discs close their energy ledger on the accretion power.

Kubota & Done 2018 (``KD18Disc``) and the band-limited power law (``PowerLawDisc``) publish
``log_L_agn_disc`` [dex re erg/s], the KD18 disc also ``log_L_agn_disc_los``. The ledger
conventions of NAMING_CONTRACT section 4c ("Standalone discs"):

* C1: ``log_L_agn_disc`` is the sphere mean :math:`\\int_0^1 L(\\mu)\\,d\\mu` of the emitted
  power, :math:`\\mu = \\cos i`, checked against a Gauss-Legendre quadrature in :math:`\\mu`
  of dense-grid integrals of the emitted SED;
* C2: the published sphere-mean pieces (there is no unemitted part: neither model has power
  outside the SED it emits) add up to ``lum_ratio * 10**agn_log_lbol L_sun`` to 1e-10;
* C3: each emitted key equals the dense-grid integral of its own SED to 1e-6;
* C4: the keys agree between pure float32 and float64 to 1.5e-5, and each ``_lsun`` sibling
  is the ``log`` key in solar luminosities.

The dense integral is the trapezoid in :math:`\\ln\\nu` on a log-uniform grid of 1e5 nodes,
independent of the components' own quadratures (closed-form Planck ring powers, the exact
integral of the piecewise-linear warm template, the corona's :math:`\\ln\\nu` trapezoid and
the power law's Gauss-Legendre band rule).
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

#: Component -> its published sphere-mean pieces (C2).
_PIECES = {"kd18": ("disc",), "powerlaw": ("disc",)}

#: Rest-frame grids of the dense reference integral [Angstrom]. The KD18 grid spans the AGN
#: publication range (it contains the whole three-zone spectrum); the power-law grid has the
#: band edges as nodes, and the SED outside the band is checked to be exactly zero.
_DENSE = {
    "kd18": np.geomspace(1.0e-3, 1.0e10, 100_001),
    "powerlaw": np.geomspace(10.0, 1.0e8, 100_001),
}
_OUTSIDE_BAND = np.array([1.0e-3, 1.0, 9.999, 1.0001e8, 1.0e9, 1.0e10])

_C2_RTOL = 1.0e-10
_C3_RTOL = 1.0e-6
_F32_RTOL = 1.5e-5


def _points(name, n_random):
    """Box center, every edge and ``n_random`` draws, with a nonzero luminosity ratio."""
    comp = build_component(name)
    pts = box_points(declared_box(comp), n_random, seed=2026)
    return comp, [p for p in pts if p["lum_ratio"] > 0.0]


@pytest.fixture(scope="module", params=sorted(_PIECES))
def disc(request):
    """``(name, component, box points)`` for each standalone disc."""
    comp, pts = _points(request.param, n_random=4)
    return request.param, comp, pts


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


def _budget(params):
    """``lum_ratio * 10**log_lbol L_sun`` [erg/s]."""
    return params["lum_ratio"] * 10.0 ** params["log_lbol"] * L_SUN


@pytest.mark.conservation
def test_published_pieces_add_up_to_the_accretion_power(disc):
    """C2: sum of the sphere-mean pieces = ``lum_ratio 10**log_lbol L_sun`` to 1e-10."""
    name, comp, pts = disc
    for params in pts:
        _, pub = _predict64(comp, params, np.geomspace(100.0, 1.0e6, 30))
        log_keys = {k for k in pub if k.startswith("log_L_agn_")}
        assert not any(k.endswith("_unemitted") for k in log_keys), log_keys
        total = sum(10.0 ** pub[f"log_L_agn_{piece}"] for piece in _PIECES[name])
        err = abs(total / _budget(params) - 1.0)
        assert err < _C2_RTOL, f"{name} at {params}: ledger / budget - 1 = {err:.2e}"


@pytest.mark.conservation
def test_emitted_sed_integrates_to_its_published_key(disc):
    """C3: the dense integral of the emitted SED equals its key to 1e-6."""
    name, comp, pts = disc
    key = "log_L_agn_disc_los" if name == "kd18" else "log_L_agn_disc"
    for params in pts:
        sed, pub = _predict64(comp, params, _DENSE[name])
        err = abs(_ln_nu_power(sed, _DENSE[name]) / 10.0 ** pub[key] - 1.0)
        assert err < _C3_RTOL, f"{name}.{key} at {params}: integral / key - 1 = {err:.2e}"


def test_powerlaw_disc_emits_nothing_outside_its_band():
    """The power-law SED is exactly zero outside 10 A to 1e8 A (the band it is defined on)."""
    comp, pts = _points("powerlaw", n_random=2)
    for params in pts:
        sed, _ = _predict64(comp, params, _OUTSIDE_BAND)
        assert np.all(sed == 0.0), f"powerlaw at {params}: outside-band L_nu {sed}"


def test_kd18_disc_power_is_the_sphere_mean_of_the_emitted_power():
    """C1: ``log_L_agn_disc`` = Gauss-Legendre mean over cos i of dense SED integrals."""
    comp, pts = _points("kd18", n_random=1)
    x, w = np.polynomial.legendre.leggauss(6)
    mu, weight = 0.5 * (x + 1.0), 0.5 * w
    for params in pts[:3]:
        powers = [
            _ln_nu_power(
                _predict64(comp, {**params, "cos_inc": m}, _DENSE["kd18"])[0], _DENSE["kd18"]
            )
            for m in mu
        ]
        mean = float(np.sum(weight * np.asarray(powers)))
        _, pub = _predict64(comp, params, _DENSE["kd18"][::1000])
        err = abs(mean / 10.0 ** pub["log_L_agn_disc"] - 1.0)
        assert err < _C3_RTOL, f"kd18 at {params}: sphere mean / log_L_agn_disc - 1 = {err:.2e}"


def test_published_keys_are_float32_accurate(disc):
    """C4: every published key in pure float32 matches float64 to 1.5e-5."""
    name, comp, pts = disc
    for params in pts:
        pub32, _ = predict(comp, params, float32=True)
        pub64, _ = predict(comp, params, float32=False)
        assert pub32.keys() == pub64.keys()
        for key, v64 in pub64.items():
            err = relative_error(key, pub32[key], v64)
            assert err < _F32_RTOL, f"{name}.{key} at {params}: float32 rel err {err:.2e}"


def test_lsun_siblings_are_the_log_keys_in_solar_units(disc):
    """C4: ``L_agn_<piece>_lsun`` = ``10**log_L_agn_<piece> / L_sun``, for every piece."""
    _name, comp, pts = disc
    _, pub = _predict64(comp, pts[0], np.geomspace(100.0, 1.0e6, 30))
    for key, value in pub.items():
        if key.startswith("log_L_agn_"):
            sibling = f"{key[len('log_') :]}_lsun"
            assert pub[sibling] == pytest.approx(10.0**value / L_SUN, rel=1e-12), sibling


def test_absent_disc_publishes_minus_infinity_and_emits_nothing(disc):
    """``lum_ratio = 0``: every ``log`` key is ``-inf``, every sibling 0, the SED zero."""
    _name, comp, pts = disc
    sed, pub = _predict64(comp, {**pts[0], "lum_ratio": 0.0}, np.geomspace(100.0, 1.0e6, 30))
    assert np.all(sed == 0.0)
    for key, value in pub.items():
        assert value == (-np.inf if key.startswith("log_") else 0.0), (key, value)


def test_powerlaw_gradient_is_finite_across_the_band():
    """The truncated power law keeps finite gradients in alpha, T_max and log_lbol."""
    comp, pts = _points("powerlaw", n_random=0)
    wave = jnp.asarray(np.geomspace(1.0, 1.0e9, 200))

    def total(alpha, t_max, log_lbol):
        p = {**{k: jnp.asarray(v) for k, v in pts[0].items()}}
        p.update(alpha=alpha, T_max=t_max, log_lbol=log_lbol)
        sed, _ = comp.predict(p, jnp.zeros_like(wave), wave)
        return jnp.sum(jnp.log(jnp.where(sed > 0, sed, 1.0)))

    grads = jax.grad(total, argnums=(0, 1, 2))(
        jnp.asarray(pts[0]["alpha"]), jnp.asarray(pts[0]["T_max"]), jnp.asarray(10.0)
    )
    assert all(np.isfinite(float(g)) for g in grads), grads
