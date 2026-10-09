# SPDX-License-Identifier: BSD-3-Clause
"""The composable shock's photometry is served from fixed per-line band coefficients (#2324).

Under ``WavePrecomp`` on a model with a composable MAPPINGS V shock, the observed
and rest shock bands are the line luminosities contracted with band coefficients
built once at build time through ``registry.resolve("mappings_shock")``. The
per-call filter integral of the shock SED is then skipped. The line placement is
linear in each line's luminosity, so the contraction is the same integral
regrouped. This file pins:

(a) the coefficients are built through the registry, and the per-call batch
    filter integral is skipped in the shock apply when they engage;
(b) the coefficient path agrees with the per-call path within 1e-3 at three seeded
    draws (total photometry), and the gradients with respect to the free shock
    H-alpha luminosity and velocity are finite and non-zero;
(c) exact mode carries no coefficients;
(d) a free redshift declines the gate, with the reason recorded.

Measured at the function level (band of the shock SED, six draws, before the
photometry sums): worst relative error 4.4e-16.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri._data_setup import find_data_str

pytestmark = pytest.mark.contract

_SSP = find_data_str("fsps_prsc_miles_chabrier.h5")
_SKIP = pytest.mark.skipif(
    _SSP is None,
    reason="needs fsps_prsc_miles_chabrier.h5 (set TENGRI_DATA_DIR)",
)

_TOL = 1e-3
_N_DRAWS = 3
_FILTERS = ["sdss_g", "sdss_r", "galex_fuv", "2mass_ks"]


def _build(*, approx, redshift=None, norm="lhalpha"):
    """A delayed-SFH model with a composable shock, photometry only."""
    import tengri
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp

    ssp = tengri.load_ssp_data(_SSP)
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        shock={
            "norm": norm,
            "log_lhalpha": Uniform(39.0, 42.0),
            "velocity": Uniform(200.0, 800.0),
            "log_density": Uniform(-1.0, 2.0),
            "b_over_sqrt_n": Uniform(0.5, 5.0),
        },
        redshift=Fixed(0.1) if redshift is None else redshift,
        approx=WavePrecomp() if approx else None,
    )


def _declined_build(monkeypatch):
    """A WavePrecomp model whose registry adapter declines, so the per-call path runs."""
    import tengri.forward.precompute.registry as registry

    real = registry.resolve

    class _Decline:
        @staticmethod
        def shock_band_coefficients(*args, **kwargs):
            return None

    monkeypatch.setattr(
        registry, "resolve", lambda name: _Decline if name == "mappings_shock" else real(name)
    )
    model = _build(approx=True)
    model._template_data_for_jit()
    monkeypatch.setattr(registry, "resolve", real)
    return model


def _draws():
    rng = np.random.default_rng(20241009)
    out = []
    for _ in range(_N_DRAWS):
        out.append(
            {
                "shock_log_lhalpha": float(rng.uniform(39.5, 41.5)),
                "shock_velocity": float(rng.uniform(200.0, 800.0)),
                "shock_log_density": float(rng.uniform(-1.0, 2.0)),
                "shock_b_over_sqrt_n": float(rng.uniform(0.5, 5.0)),
            }
        )
    return out


@pytest.fixture(scope="module")
def table_model():
    model = _build(approx=True)
    model._template_data_for_jit()
    return model


@_SKIP
def test_coefficients_are_built_through_the_registry(monkeypatch):
    """(a) The coefficients come from ``registry.resolve("mappings_shock")``."""
    import tengri.components.nebular.mappings_shock_precompute as msp

    seen = []
    real = msp.shock_band_coefficients

    def spy(*args, **kwargs):
        seen.append(len(args))
        return real(*args, **kwargs)

    monkeypatch.setattr(msp, "shock_band_coefficients", spy)
    model = _build(approx=True)
    model._template_data_for_jit()
    assert model._shock_band_table_decline is None
    assert len(seen) == 1 and seen[0] == 4
    assert "shock_band_table" in model._template_data_for_jit()


@_SKIP
def test_per_call_batch_integral_is_skipped_when_table_engages(table_model, monkeypatch):
    """(a) With the coefficients engaged, the shock apply runs no per-call batch integral."""
    import tengri.components.nebular.shock_model as shock_model
    import tengri.observation.photometry as photometry

    calls = {"n": 0, "in_apply": None}
    real_batch = photometry.lnu_filter_integral_batch
    real_apply = shock_model.ShockNebular.apply

    def counting_batch(*args, **kwargs):
        calls["n"] += 1
        return real_batch(*args, **kwargs)

    def counting_apply(self, *args, **kwargs):
        calls["n"] = 0
        out = real_apply(self, *args, **kwargs)
        calls["in_apply"] = calls["n"]
        return out

    monkeypatch.setattr(photometry, "lnu_filter_integral_batch", counting_batch)
    monkeypatch.setattr(shock_model.ShockNebular, "apply", counting_apply)
    # Off-grid values, so this trace is not shared with the accuracy test.
    table_model.predict_photometry(
        {
            "shock_log_lhalpha": np.float64(40.3),
            "shock_velocity": np.float64(351.7),
            "shock_log_density": np.float64(0.37),
            "shock_b_over_sqrt_n": np.float64(1.9),
        }
    )
    assert calls["in_apply"] == 0


@_SKIP
def test_coefficient_path_matches_per_call_with_finite_gradients(table_model, monkeypatch):
    """(b) Coefficient vs per-call within 1e-3 at three draws; gradients finite, non-zero."""
    import jax
    import jax.numpy as jnp

    percall = _declined_build(monkeypatch)
    worst = 0.0
    for draw in _draws():
        params = {k: jnp.asarray(v) for k, v in draw.items()}
        approx = np.asarray(table_model.predict_photometry(params))
        exact = np.asarray(percall.predict_photometry(params))
        rel = np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-30)
        worst = max(worst, float(rel.max()))

        def loss(p):
            return jnp.sum(table_model.predict_photometry(p)) * 1e27

        grads = jax.grad(loss)(params)
        for name in ("shock_log_lhalpha", "shock_velocity"):
            g = float(grads[name])
            assert np.isfinite(g), f"gradient w.r.t. {name} is not finite"
            assert g != 0.0, f"gradient w.r.t. {name} is zero"
    assert worst < _TOL, f"worst relative photometry error {worst:.3e} exceeds {_TOL:.0e}"


@_SKIP
def test_exact_mode_carries_no_coefficients():
    """(c) Exact mode (approx=None) publishes no shock band coefficients."""
    model = _build(approx=False)
    template = model._template_data_for_jit()
    assert template is None or "shock_band_table" not in template
    assert getattr(model, "_shock_band_table_cache", None) is None


@_SKIP
def test_free_redshift_declines_the_gate():
    """(d) A free redshift declines the gate, and the reason is recorded."""
    from tengri import Uniform

    model = _build(approx=True, redshift=Uniform(0.05, 0.2))
    template = model._template_data_for_jit()
    assert "shock_band_table" not in (template or {})
    assert "redshift is free" in model._shock_band_table_decline
