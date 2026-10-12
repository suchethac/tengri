# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR published luminosities follow the angle-integrated convention of #2678 (#2851).

An AGN's luminosity parameter is the angle-integrated power, and the line-of-sight
isotropic-equivalent luminosity is a separate derived output (KD18 publishes
``log_L_agn_disc`` and ``log_L_agn_disc_los`` that way). SKIRTOR's tabulated disc is an
anisotropic emitter: the library holds a different spectrum at every inclination, so the power
of the SED the component emits changes with ``cos_inc``. The published keys are

* ``log_L_agn_disc``, ``log_L_agn_torus``, ``log_L_agn_polar_dust``: the mean over the sphere of
  the line-of-sight power, :math:`\\tfrac12\\int_0^\\pi L(\\theta)\\sin\\theta\\,d\\theta =
  \\int_0^1 L(\\mu)\\,d\\mu` with :math:`\\mu = \\cos i` (symmetry about the equator), by
  Gauss-Legendre on each interval between the template's own inclination nodes; independent
  of ``cos_inc``;
* the ``_los`` siblings: the power of the SED emitted along ``cos_inc``.

The reference integral over :math:`\\mu` is computed here from the raw templates, with the
component's disc re-tilt written out independently
(``test_agn_template_grid_independence._skirtor_emitted_pieces``), by Simpson's rule on 81
inclinations. The quadrature error of the component's rule, measured against an 801-point
Simpson reference of the same interpolant over five parameter sets, is 7e-5 (disc), 2e-8
(torus) and 1e-3 (polar-dust power); the tolerances below are those errors plus the
reference's own.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.physics._agn_publication import PUB_COARSE, ln_nu_integral
from tests.physics._agn_published import build_component, declared_box
from tests.physics.test_agn_template_grid_independence import _skirtor_emitted_pieces

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
#: Tolerances on the angle-integrated powers against the 81-point reference (see the docstring).
_RTOL_MEAN = {"disc": 3.0e-4, "torus": 5.0e-6, "polar_dust": 3.0e-3}
_N_MU = 81
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


def _los_power_from_templates(comp, params, mu):
    """Powers [erg/s] of the disc, torus dust and polar re-emission emitted along ``mu``."""
    pieces = _skirtor_emitted_pieces(
        comp,
        {k: jnp.asarray(v, dtype=jnp.float64) for k, v in {**params, "cos_inc": mu}.items()},
        _WIDE,
    )
    return np.array(
        [
            ln_nu_integral(pieces["disc"], _WIDE),
            ln_nu_integral(pieces["dust"], _WIDE),
            ln_nu_integral(pieces["polar"], _WIDE),
        ]
    )


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
def test_angle_integrated_powers_equal_the_integral_over_mu_of_the_raw_templates(skirtor, name):
    """``log_L_agn_*`` equal Simpson's integral over cos i of the powers from the templates."""
    comp, mid = skirtor
    params = {**mid, **_PARAM_SETS[name]}
    mu = np.linspace(0.0, 1.0, _N_MU)
    los = np.array([_los_power_from_templates(comp, params, m) for m in mu])
    step = mu[1] - mu[0]
    simpson = step / 3.0 * (los[0] + los[-1] + 4.0 * los[1:-1:2].sum(0) + 2.0 * los[2:-1:2].sum(0))
    pub = _published(comp, params)
    for i, key in enumerate(_KEYS):
        got = 10.0 ** pub[f"log_L_agn_{key}"]
        err = abs(got / simpson[i] - 1.0)
        assert err < _RTOL_MEAN[key], f"{name} {key}: published/integral - 1 = {err:.2e}"


@pytest.mark.parametrize("cos_inc", [0.1, 0.5, 0.95])
def test_line_of_sight_powers_equal_the_template_value_at_the_sightline(skirtor, cos_inc):
    """``log_L_agn_*_los`` equal the dense integral of the SED from the templates (1e-5)."""
    comp, mid = skirtor
    params = {**mid, "polar_ebv": 0.2, "cos_inc": cos_inc}
    expected = _los_power_from_templates(comp, params, cos_inc)
    pub = _published(comp, params)
    for i, key in enumerate(_KEYS):
        err = abs(10.0 ** pub[f"log_L_agn_{key}_los"] / expected[i] - 1.0)
        assert err < 1.0e-5, f"cos_inc={cos_inc} {key}: rel {err:.2e}"


def test_torus_power_varies_with_inclination_at_the_normalization_level_only(skirtor):
    """The torus dust is renormalized per inclination: its line-of-sight power moves by < 0.5%.

    That small spread is the angle-integrated convention's reason for ``log_L_agn_torus_los``
    existing at all; the angle-integrated torus power sits inside it.
    """
    comp, mid = skirtor
    runs = [_published(comp, {**mid, "cos_inc": c}) for c in np.linspace(0.0, 1.0, 9)]
    los = np.array([10.0 ** r["log_L_agn_torus_los"] for r in runs])
    assert 1e-4 < los.max() / los.min() - 1.0 < 5e-3, los
    mean = 10.0 ** runs[0]["log_L_agn_torus"]
    assert los.min() * (1 - 1e-12) <= mean <= los.max() * (1 + 1e-12)


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
