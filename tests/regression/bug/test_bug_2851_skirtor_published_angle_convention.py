# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR keeps its own anisotropy, and its published powers follow the #2678 convention (#2851).

An AGN's luminosity parameter is the angle-integrated power, and the line-of-sight
isotropic-equivalent luminosity is a separate derived output (KD18 publishes
``log_L_agn_disc`` and ``log_L_agn_disc_los`` that way). SKIRTOR's library holds a different
disc and dust spectrum at every inclination, and each record is stored divided by its own dust
integral (``spectra/norm`` holds the divisor), so the flux ratio between inclinations survives
only in ``norm``. The component multiplies the records by ``norm``, interpolates (PCHIP,
node-exact), and scales by ONE factor per parameter point so that the sphere-mean escaping
disc plus the sphere-mean dust

.. math:: \\bar P = \\int_0^1 P(\\mu)\\,d\\mu, \\qquad \\mu = \\cos i,

(the mean of the line-of-sight power over the sphere, by symmetry about the equator) equals
``10**agn_log_lbol * L_sun * torus_frac`` (``test_bug_2851_skirtor_energy_conservation``). The
published keys are

* ``log_L_agn_disc``, ``log_L_agn_torus``: the sphere mean of the line-of-sight power, by
  Gauss-Legendre on each interval between the template's own inclination nodes; independent
  of ``cos_inc``; the disc is net of the power the polar dust takes from the cone;
* ``log_L_agn_polar_dust``: the power the polar dust absorbs from the disc light emitted into
  the polar cone, :math:`\\int_{\\sin \\Phi}^1 A(\\mu)\\,d\\mu` with :math:`\\Phi` the torus
  half-opening angle, re-emitted isotropically; independent of ``cos_inc``;
* the ``_los`` siblings: the power of the SED emitted along ``cos_inc``, after the polar dust's
  Type-1-masked attenuation. The polar re-emission is isotropic, so its ``_los`` key equals
  the plain one.

The references are independent of the component: ``RawSkirtor`` reads the file with ``h5py``
and integrates the power laws in closed form; the angle integrals are per-interval
Gauss-Legendre of order 20 of the dense-grid line-of-sight pieces.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data
from tests.physics._agn_publication import PUB_COARSE, ln_nu_integral
from tests.physics._agn_published import build_component, declared_box
from tests.physics._skirtor_reference import (
    RawSkirtor,
    cone_absorbed,
    cone_intervals,
    emitted_pieces,
)

pytestmark = pytest.mark.regression_bug

_KEYS = ("disc", "torus", "polar_dust")
_PARAM_SETS = {
    "center_with_polar_dust": {"polar_ebv": 0.2},
    "open_torus_shifted_disc": {
        "oa_skirtor": 70.0,
        "tau_skirtor": 4.0,
        "delta": -0.5,
        "polar_ebv": 0.1,
    },
}
_RTOL = 1.0e-6
_GL_ORDER = 20
#: Wide log grid of the reference integrals; the analytic disc is zero below 8 nm and above
#: 1e6 nm, so each edge is bracketed by nodes.
_EDGES_AA = (80.0, 1.0e7)
_WIDE = np.unique(
    np.concatenate(
        [
            np.geomspace(1.0, 1.0e10, 40_000),
            [x for e in _EDGES_AA for x in (e * (1 - 1e-9), e, e * (1 + 1e-9))],
        ]
    )
)


@pytest.fixture(scope="module")
def skirtor():
    comp = build_component("skirtor")
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in declared_box(comp).items()}
    return comp, mid


def _published(comp, params, wave=PUB_COARSE):
    w = jnp.asarray(wave)
    _, pub = comp.predict(
        {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}, jnp.zeros_like(w), w
    )
    return {k: float(v) for k, v in pub.items()}


def _los_powers_from_templates(comp, params, mu):
    """Powers [erg/s] along ``mu``: the disc before and after the polar attenuation, the dust."""
    pieces = emitted_pieces(
        comp,
        {k: jnp.asarray(v, dtype=jnp.float64) for k, v in {**params, "cos_inc": mu}.items()},
        _WIDE,
    )
    return np.array(
        [
            ln_nu_integral(pieces["disc"], _WIDE),
            ln_nu_integral(pieces["disc_attenuated"], _WIDE),
            ln_nu_integral(pieces["dust"], _WIDE),
        ]
    )


def _sphere_mean_reference(comp, params):
    """Disc, torus and polar powers [erg/s]: per-interval Gauss-Legendre order 20 of the pieces.

    The published disc is the sphere mean of the escaping disc less the power the polar dust
    absorbs from the cone, so that the three add up to the accretion power.
    """
    nodes = np.asarray(comp.data.native_cos_inc)
    edges = np.concatenate([[0.0], nodes[1:]])
    x, w = np.polynomial.legendre.leggauss(_GL_ORDER)
    mean = np.zeros(3)
    for a, b in itertools.pairwise(edges):
        for xi, wi in zip(x, w, strict=True):
            mean += (
                0.5
                * (b - a)
                * wi
                * _los_powers_from_templates(comp, params, 0.5 * (b - a) * xi + 0.5 * (a + b))
            )
    wide_params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
    polar = (
        cone_absorbed(comp, wide_params, _WIDE, order=_GL_ORDER)
        if params["polar_ebv"] > 0
        else 0.0
    )
    return np.array([mean[0] - polar, mean[2], polar])


@pytest.mark.parametrize("name", sorted(_PARAM_SETS))
def test_angle_integrated_powers_do_not_depend_on_inclination(skirtor, name):
    """``log_L_agn_*`` are the same at cos i = 0.05, 0.4 and 0.9; the ``_los`` ones are not."""
    comp, mid = skirtor
    base = {**mid, **_PARAM_SETS[name]}
    runs = [_published(comp, {**base, "cos_inc": c}) for c in (0.05, 0.4, 0.9)]
    for key in _KEYS:
        values = [r[f"log_L_agn_{key}"] for r in runs]
        assert max(values) - min(values) < 1e-12, (key, values)
    los_disc = [r["log_L_agn_disc_los"] for r in runs]
    assert max(los_disc) - min(los_disc) > 1e-3, (
        f"the line-of-sight disc power is flat: {los_disc}"
    )


@pytest.mark.parametrize("name", sorted(_PARAM_SETS))
def test_angle_integrated_powers_equal_the_gauss_legendre_integral_of_the_pieces(skirtor, name):
    """``log_L_agn_*`` equal an independent order-20 per-interval Gauss-Legendre integral (1e-6).

    The pieces are the component's disc, dust and cone-absorbed power rebuilt on a wide dense
    grid at each Gauss node; the nodes of the reference are not those of the component's rule.
    """
    comp, mid = skirtor
    params = {**mid, **_PARAM_SETS[name]}
    reference = _sphere_mean_reference(comp, params)
    pub = _published(comp, params)
    for i, key in enumerate(_KEYS):
        got = 10.0 ** pub[f"log_L_agn_{key}"]
        err = abs(got / reference[i] - 1.0)
        assert err < _RTOL, f"{name} {key}: published/integral - 1 = {err:.2e}"


@pytest.mark.parametrize("cos_inc", [0.1, 0.5, 0.95])
def test_line_of_sight_powers_equal_the_template_value_at_the_sightline(skirtor, cos_inc):
    """``log_L_agn_*_los`` equal the dense integral of the SED from the templates (1e-5)."""
    comp, mid = skirtor
    params = {**mid, "polar_ebv": 0.2, "cos_inc": cos_inc}
    powers = _los_powers_from_templates(comp, params, cos_inc)
    expected = powers[[1, 2]]
    pub = _published(comp, params)
    for i, key in enumerate(("disc", "torus")):
        err = abs(10.0 ** pub[f"log_L_agn_{key}_los"] / expected[i] - 1.0)
        assert err < 1.0e-5, f"cos_inc={cos_inc} {key}: rel {err:.2e}"


def test_polar_reemission_does_not_depend_on_the_sightline(skirtor):
    """The polar power is the disc light emitted into the cone, so it is the same at every cos i.

    The graybody re-emission is isotropic, so the line-of-sight key is the plain key. Both match
    the cone integral of the disc emitted along each direction, whatever the observer sees.
    """
    comp, mid = skirtor
    base = {**mid, "polar_ebv": 0.2}
    runs = [_published(comp, {**base, "cos_inc": c}) for c in (0.02, 0.3, 0.6, 0.95)]
    plain = [r["log_L_agn_polar_dust"] for r in runs]
    los = [r["log_L_agn_polar_dust_los"] for r in runs]
    assert max(plain) - min(plain) < 1e-12, plain
    assert los == plain
    cone = cone_absorbed(
        comp, {k: jnp.asarray(v, dtype=jnp.float64) for k, v in base.items()}, _WIDE, order=8
    )
    assert abs(10.0 ** plain[0] / cone - 1.0) < _RTOL


def test_polar_cone_integral_is_the_edge_weighted_sphere_share(skirtor):
    """The cone rule integrates ``int_{sin(oa)}^1 d(mu)`` exactly for a constant integrand."""
    comp, _ = skirtor
    for oa in (10.0, 38.0, 80.0):
        _, weight = cone_intervals(comp.data.native_cos_inc, oa, 8)
        assert weight.sum() == pytest.approx(1.0 - np.sin(np.radians(oa)), abs=1e-14)


@pytest.fixture(scope="module")
def raw_library():
    path = find_data("skirtor_templates_v3.h5")
    if path is None:
        pytest.skip("SKIRTOR v3 templates unavailable")
    return RawSkirtor(str(path))


def test_line_of_sight_powers_keep_the_relative_flux_between_inclinations(skirtor, raw_library):
    """Disc and torus power along each library inclination follow the raw records (1e-12).

    The reference reads ``skirtor_templates_v3.h5`` with ``h5py``, multiplies the stored spectra
    by their ``norm`` records (the stored spectra are divided by the model's dust integral),
    interpolates the five parameter axes with SciPy's PCHIP and integrates the power laws in
    closed form. The interpolation is node-exact in cos i, so the published power at a library
    inclination IS the record's. Ratios to the face-on node are compared: the overall scale is
    the normalization's. If every inclination were rescaled on its own, as the stored records
    are, the torus ratios would all be 1.
    """
    comp, mid = skirtor
    params = {**mid, "polar_ebv": 0.0, "delta": 0.0}
    point = tuple(
        mid[k] for k in ("tau_skirtor", "p_skirtor", "q_skirtor", "oa_skirtor", "radius_ratio")
    )
    nodes = np.asarray(comp.data.native_cos_inc)
    got = {"disc": [], "dust": []}
    ref = {"disc": [], "dust": []}
    for mu in nodes:
        pub = _published(comp, {**params, "cos_inc": mu})
        got["disc"].append(10.0 ** pub["log_L_agn_disc_los"])
        got["dust"].append(10.0 ** pub["log_L_agn_torus_los"])
        ref["disc"].append(
            raw_library.power("disk", point, mu, lower=_EDGES_AA[0], upper=_EDGES_AA[1])
        )
        ref["dust"].append(raw_library.power("dust", point, mu))
    for key in ("disc", "dust"):
        g, r = np.array(got[key]), np.array(ref[key])
        err = np.max(np.abs((g / g[-1]) / (r / r[-1]) - 1.0))
        assert err < 1.0e-12, f"{key}: relative flux between inclinations off by {err:.2e}"
    flat = np.array(
        [
            raw_library.power("dust", point, mu, per_inclination_norm=True)
            / raw_library.power("dust", point, nodes[-1], per_inclination_norm=True)
            for mu in nodes
        ]
    )
    anisotropic = np.array(ref["dust"]) / ref["dust"][-1]
    # Each stored record integrates to one unit by the trapezoid, so on its own scale the
    # dust power is flat to the trapezoid-versus-power-law difference (a few 1e-3).
    assert np.max(np.abs(flat - 1.0)) < 1e-2
    assert anisotropic.min() < 0.7, "the library dust is not anisotropic: the check is vacuous"


def test_skirtor_predict_is_jittable_and_vmappable_over_inclination(skirtor):
    """``predict`` traces under ``jit`` (the disc breakpoints are read at compile time)."""
    comp, mid = skirtor
    wave = jnp.asarray(PUB_COARSE)
    params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in mid.items()}

    @jax.jit
    def los_disc(cos_inc):
        _, pub = comp.predict({**params, "cos_inc": cos_inc}, jnp.zeros_like(wave), wave)
        return pub["log_L_agn_disc_los"]

    values = jax.vmap(los_disc)(jnp.asarray([0.1, 0.5, 0.9]))
    direct = [
        _published(comp, {**mid, "cos_inc": c})["log_L_agn_disc_los"] for c in (0.1, 0.5, 0.9)
    ]
    np.testing.assert_allclose(np.asarray(values), direct, rtol=0, atol=1e-10)


@pytest.mark.parametrize("delta", [0.0, 0.6])
def test_sed_gradient_in_torus_frac_survives_zero(skirtor, delta):
    """d(SED)/d(torus_frac) is the disc plus torus at frac = 0, matching finite differences.

    Every piece is linear in the fraction, so the derivative at 0 is the unit-fraction SED. A
    guard that selects a constant where the disc power vanishes drops the disc term there.
    """
    comp, mid = skirtor
    wave = jnp.asarray(PUB_COARSE)
    params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in {**mid, "delta": delta}.items()}

    def sed(frac):
        out, _ = comp.predict({**params, "torus_frac": frac}, jnp.zeros_like(wave), wave)
        return out

    h = 1.0e-8
    for frac0, fd in (
        (0.0, lambda: (sed(h) - sed(0.0)) / h),
        (h, lambda: (sed(h + 0.5 * h) - sed(h - 0.5 * h)) / h),
    ):
        grad = np.asarray(jax.jacfwd(sed)(jnp.asarray(frac0)))
        np.testing.assert_allclose(grad, np.asarray(fd()), rtol=1e-6, atol=0.0)
        assert np.all(np.isfinite(grad)) and grad.max() > 0.0
