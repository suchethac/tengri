# SPDX-License-Identifier: BSD-3-Clause
"""The CLOUDY age-resolved band table is wired through the registry (#2324).

``WavePrecomp`` on a CLOUDY nebular model builds the band table through
``registry.resolve("cloudy").precompute`` and publishes the observed and rest
band rows from its contraction over ages, skipping the per-call filter
integrals. This file pins the four things that make that safe:

(a) the table is built through the registry, and the per-call nebular filter
    integral is skipped when it engages;
(b) with free ``neb_logU`` and ``neb_logZ_gas``, the table path agrees with the
    per-call path within 1e-3 at three seeded draws, and the gradients with
    respect to both are finite and non-zero. Measured worst off-node relative
    error of the total photometry at these draws: 9.6e-6 (the band-level worst
    over five draws is 1.7e-4, in test_nebular_cloudy_band_table_accuracy);
(c) exact mode carries no table and stays at the exact path;
(d) a free ``neb_eline_sigma_kms`` declines the gate, with the reason recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri._data_setup import find_data_str

_SSP = find_data_str("fsps_prsc_miles_chabrier.h5")
_GRID = find_data_str("cloudy_grid_prsc.h5")
_SKIP = pytest.mark.skipif(
    _SSP is None or _GRID is None,
    reason="needs fsps_prsc_miles_chabrier.h5 and cloudy_grid_prsc.h5 (set TENGRI_DATA_DIR)",
)

_TOL = 1e-3
_N_DRAWS = 3
_FILTERS = ["sdss_g", "sdss_r", "galex_fuv", "2mass_ks"]
_GAS_PRIOR = (-1.3, 0.2)
_LOGU_PRIOR = (-4.0, -1.0)


def _build(*, approx, neb_extra=None, redshift=0.1):
    """A delayed-SFH CLOUDY model on the prsc SSP with photometry only."""
    import tengri
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp

    ssp = tengri.load_ssp_data(_SSP)
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    neb = {
        "type": "cloudy",
        "neb_logU": Uniform(*_LOGU_PRIOR),
        "neb_logZ_gas": Uniform(*_GAS_PRIOR),
        "all_params": Fixed(DEFAULT),
    }
    neb.update(neb_extra or {})
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb=neb,
        redshift=Fixed(redshift),
        approx=WavePrecomp() if approx else None,
    )


def _percall_model(monkeypatch):
    """A WavePrecomp model whose registry adapter declines, so the per-call path runs."""
    import tengri.forward.precompute.registry as registry

    real = registry.resolve

    class _Decline:
        @staticmethod
        def precompute(*args, **kwargs):
            return None

    monkeypatch.setattr(
        registry, "resolve", lambda name: _Decline if name == "cloudy" else real(name)
    )
    model = _build(approx=True)
    model._template_data_for_jit()
    monkeypatch.setattr(registry, "resolve", real)
    return model


@pytest.fixture(scope="module")
def table_model():
    model = _build(approx=True)
    model._template_data_for_jit()
    return model


@pytest.fixture(scope="module")
def percall_model():
    import tengri.forward.precompute.registry as registry

    real = registry.resolve

    class _Decline:
        @staticmethod
        def precompute(*args, **kwargs):
            return None

    registry.resolve = lambda name: _Decline if name == "cloudy" else real(name)
    try:
        model = _build(approx=True)
        model._template_data_for_jit()
    finally:
        registry.resolve = real
    return model


def _draws():
    rng = np.random.default_rng(20241009)
    out = []
    for _ in range(_N_DRAWS):
        out.append(
            {
                "neb_logU": float(rng.uniform(*_LOGU_PRIOR)),
                "neb_logZ_gas": float(rng.uniform(*_GAS_PRIOR)),
            }
        )
    return out


@_SKIP
def test_table_is_built_through_the_registry(monkeypatch):
    """(a) The table comes from ``registry.resolve("cloudy").precompute``."""
    import tengri.components.nebular.cloudy_precompute as cp
    import tengri.forward.precompute.registry as registry

    seen = []
    real_precompute = cp.precompute

    def spy(*args, **kwargs):
        seen.append(sorted(kwargs))
        return real_precompute(*args, **kwargs)

    monkeypatch.setattr(cp, "precompute", spy)
    model = _build(approx=True)
    model._template_data_for_jit()
    assert model._cloudy_band_table_decline is None
    assert len(seen) == 1
    assert {"backend", "wave", "line_sigma_kms"} <= set(seen[0])
    assert "nebular_band_table" in model._template_data_for_jit()
    assert registry.resolve("cloudy") is cp


@_SKIP
def test_per_call_integral_is_skipped_when_table_engages(table_model, monkeypatch):
    """(a) With the table engaged, the nebular apply runs no per-call filter integral."""
    import tengri.observation.photometry as photometry
    from tengri.components.nebular.component import NebularSEDComponent

    calls = {"n": 0}
    real_integral = photometry.lnu_filter_integral
    real_apply = NebularSEDComponent.apply

    def counting_integral(*args, **kwargs):
        calls["n"] += 1
        return real_integral(*args, **kwargs)

    def counting_apply(self, *args, **kwargs):
        calls["n"] = 0
        out = real_apply(self, *args, **kwargs)
        calls["in_apply"] = calls["n"]
        return out

    monkeypatch.setattr(photometry, "lnu_filter_integral", counting_integral)
    monkeypatch.setattr(NebularSEDComponent, "apply", counting_apply)
    # Off-node draw, so the trace is not shared with the accuracy test.
    table_model.predict_photometry(
        {"neb_logU": np.float64(-2.25), "neb_logZ_gas": np.float64(-0.5)}
    )
    assert calls.get("in_apply") == 0


@_SKIP
def test_table_matches_per_call_within_tolerance_with_finite_gradients(table_model, percall_model):
    """(b) Table vs per-call within 1e-3 at three draws; gradients finite, non-zero."""
    import jax
    import jax.numpy as jnp

    worst = 0.0
    for draw in _draws():
        params = {k: jnp.asarray(v) for k, v in draw.items()}
        approx = np.asarray(table_model.predict_photometry(params))
        exact = np.asarray(percall_model.predict_photometry(params))
        rel = np.abs(approx - exact) / np.maximum(np.abs(exact), 1e-30)
        worst = max(worst, float(rel.max()))

        def loss(p):
            return jnp.sum(table_model.predict_photometry(p)) * 1e27

        grads = jax.grad(loss)(params)
        for name in ("neb_logU", "neb_logZ_gas"):
            g = float(grads[name])
            assert np.isfinite(g), f"gradient w.r.t. {name} is not finite"
            assert g != 0.0, f"gradient w.r.t. {name} is zero"
    assert worst < _TOL, f"worst relative photometry error {worst:.3e} exceeds {_TOL:.0e}"


@_SKIP
def test_exact_mode_carries_no_table_and_stays_exact():
    """(c) Exact mode (approx=None) publishes no band table and is not table-driven."""
    model = _build(approx=False)
    template = model._template_data_for_jit()
    assert template is None or "nebular_band_table" not in template
    assert getattr(model, "_cloudy_band_table_cache", None) is None


@_SKIP
def test_free_eline_sigma_declines_the_gate():
    """(d) A free neb_eline_sigma_kms declines the gate, and the reason is recorded."""
    import tengri
    from tengri import Uniform

    model = _build(approx=True, neb_extra={"neb_eline_sigma_kms": Uniform(50.0, 150.0)})
    template = model._template_data_for_jit()
    assert "nebular_band_table" not in (template or {})
    assert "neb_eline_sigma_kms" in model._cloudy_band_table_decline
    assert tengri is not None
