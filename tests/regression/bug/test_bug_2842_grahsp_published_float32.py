# SPDX-License-Identifier: BSD-3-Clause
"""The standalone GRAHSP component is finite and accurate in pure float32 (#2842).

Its free parameter ``agn_grahsp_l5100`` was a linear ``LogUniform(1e38, 1e50)`` erg/s input:
at the prior's geometric mean (1e44 erg/s) the bare value is ``inf`` in float32, so the SED and
every published luminosity were NaN. The parameter is ``agn_grahsp_log_l5100`` (dex), the
luminosities are published as ``log_L_agn_bol`` / ``log_L_agn_torus`` / ``log_L_agn_absorbed``
(NAMING_CONTRACT section 4c), and ``L_agn_bol`` stays as the one erg/s key the radio and X-ray
components read (it is ``inf`` in float32, where its log companion is the value).

Float32 is selected with ``jax.enable_x64(False)`` inside the test; ``JAX_ENABLE_X64=0`` is
never set for pytest.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.grahsp import GRAHSPSEDComponent, GRAHSPSEDComponentConfig
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.regression_bug

FLOAT32_RTOL = (
    1.0e-4  # the ADAF float32 shape tolerance, tests/physics/agn/test_adaf_float32_shape.py
)
_WAVE = np.geomspace(100.0, 1.0e6, 300)
_LOG_KEYS = ("log_L_agn_bol", "log_L_agn_torus", "log_L_agn_absorbed")

_CONFIGS = {
    "default": {},
    "mn12_torus": {"torus_model": "mn12"},
    "netzer_disc": {"disc_model": "netzer"},
    "no_attenuation": {"apply_attenuation": False},
    "type2": {"agn_type": 2},
    "veroncetty_feii": {"feii_template": "veroncetty2004"},
}


def _box(comp):
    box = {}
    for decl in comp.declared_parameters():
        prior = decl.prior
        lo = getattr(prior, "lo", None)
        box[decl.name] = (
            (float(prior.default),) * 2 if lo is None else (float(lo), float(prior.hi))
        )
    return box


def _points(box, n_random, seed):
    rng = np.random.default_rng(seed)
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in box.items()}
    points = [dict(mid)]
    points.extend({**mid, k: edge} for k, (lo, hi) in box.items() for edge in (lo, hi))
    points.extend({k: rng.uniform(lo, hi) for k, (lo, hi) in box.items()} for _ in range(n_random))
    return points


def _apply(comp, templates, params, *, float32):
    dtype = jnp.float32 if float32 else jnp.float64
    with jax.enable_x64(not float32):
        wave = jnp.asarray(_WAVE, dtype=dtype)
        p = {k: jnp.asarray(v, dtype=dtype) for k, v in params.items()}
        state = ForwardState(wave=wave, sed_intrinsic=jnp.zeros_like(wave), derived={})
        out = comp.apply(state, p, templates)
        derived = {k: float(out.derived[k]) for k in ("L_agn_bol", *_LOG_KEYS)}
        return derived, np.asarray(out.sed_intrinsic, dtype=np.float64)


@pytest.fixture(scope="module", params=sorted(_CONFIGS))
def component(request):
    comp = GRAHSPSEDComponent(config=GRAHSPSEDComponentConfig(**_CONFIGS[request.param]))
    return request.param, comp, comp.precompute()


def test_published_luminosities_and_sed_are_finite_and_accurate_in_float32(component):
    """Over the declared box (center, edges, draws) float32 equals float64 to FLOAT32_RTOL."""
    name, comp, templates = component
    for point in _points(_box(comp), 4, seed=2842):
        d64, sed64 = _apply(comp, templates, point, float32=False)
        d32, sed32 = _apply(comp, templates, point, float32=True)
        assert np.all(np.isfinite(sed32)), f"{name}: non-finite float32 SED at {point}"
        significant = sed64 > 1.0e-3 * sed64.max()
        sed_err = np.max(np.abs(sed32[significant] / sed64[significant] - 1.0))
        assert sed_err < FLOAT32_RTOL, f"{name}: float32 SED rel {sed_err:.2e}"
        for key in _LOG_KEYS:
            if not np.isfinite(d64[key]):
                # A power that is exactly zero (no attenuation absorbs nothing) is -inf in both.
                assert d32[key] == d64[key] == -np.inf, f"{name}.{key} at {point}"
                continue
            assert np.isfinite(d32[key]), f"{name}.{key} = {d32[key]} at {point}"
            err = abs(10.0 ** (d32[key] - d64[key]) - 1.0)
            assert err < FLOAT32_RTOL, f"{name}.{key}: rel {err:.2e} at {point}"


def test_erg_per_s_bol_is_the_documented_float64_convenience(component):
    """``L_agn_bol`` is ``10**log_L_agn_bol`` in float64 and ``inf`` (not NaN) in float32."""
    _, comp, templates = component
    point = {k: 0.5 * (lo + hi) for k, (lo, hi) in _box(comp).items()}
    d64, _ = _apply(comp, templates, point, float32=False)
    d32, _ = _apply(comp, templates, point, float32=True)
    assert d64["L_agn_bol"] == pytest.approx(10.0 ** d64["log_L_agn_bol"], rel=1e-12)
    assert d32["L_agn_bol"] == np.inf
    assert 43.0 < d32["log_L_agn_bol"] < 46.0


def test_sed_scales_linearly_with_l5100():
    """The SED and every published power scale as ``10**d`` for ``log_l5100 -> log_l5100 + d``.

    This is the property the float32 evaluation at a reference ``l5100`` relies on.
    """
    comp = GRAHSPSEDComponent()
    templates = comp.precompute()
    point = {k: 0.5 * (lo + hi) for k, (lo, hi) in _box(comp).items()}
    base, sed = _apply(comp, templates, point, float32=False)
    shifted = {**point, "agn_grahsp_log_l5100": point["agn_grahsp_log_l5100"] + 2.0}
    up, sed_up = _apply(comp, templates, shifted, float32=False)
    np.testing.assert_allclose(sed_up, 100.0 * sed, rtol=1e-12)
    for key in _LOG_KEYS:
        assert up[key] - base[key] == pytest.approx(2.0, abs=1e-12)


def test_retired_linear_l5100_name_warns_and_is_translated():
    """The linear ``agn_grahsp_l5100`` still works for the deprecation period, with a warning."""
    comp = GRAHSPSEDComponent()
    templates = comp.precompute()
    point = {k: 0.5 * (lo + hi) for k, (lo, hi) in _box(comp).items()}
    new, sed_new = _apply(comp, templates, point, float32=False)
    legacy = {k: v for k, v in point.items() if k != "agn_grahsp_log_l5100"}
    legacy["agn_grahsp_l5100"] = 10.0 ** point["agn_grahsp_log_l5100"]
    with pytest.warns(DeprecationWarning, match="agn_grahsp_log_l5100"):
        old, sed_old = _apply(comp, templates, legacy, float32=False)
    np.testing.assert_allclose(sed_old, sed_new, rtol=1e-12)
    assert old["log_L_agn_bol"] == pytest.approx(new["log_L_agn_bol"], abs=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _apply(comp, templates, point, float32=False)  # the new name is silent
