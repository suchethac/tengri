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
(d) a free ``neb_eline_sigma_kms`` declines the gate, with the reason recorded;
(e) a CLOUDY backend that is not ``grid_interp="linear"`` declines the gate, because the
    table's contraction reproduces only the linear scheme.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri._data_setup import find_data_str
from tests._cloudy_grid import synthetic_ionizing_ssp, write_synthetic_cloudy_grid

pytestmark = pytest.mark.contract

#: The real grid is built locally from raw CLOUDY output and is not tracked or
#: downloadable, so only the real-data case gates on it. The synthetic case
#: below carries no such gate and runs on every runner.
_GRID = find_data_str("cloudy_grid_prsc.h5")
_REAL_SKIP = pytest.mark.skipif(
    _GRID is None,
    reason="needs cloudy_grid_prsc.h5 (scripts/convert_fsps_cloudy_grid.py; not downloadable)",
)

_TOL = 1e-3
_N_DRAWS = 3
_FILTERS = ["sdss_g", "sdss_r", "galex_fuv", "2mass_ks"]
_GAS_PRIOR = (-1.3, 0.2)
_LOGU_PRIOR = (-4.0, -1.0)


@pytest.fixture(scope="module", params=["synthetic", pytest.param("real", marks=_REAL_SKIP)])
def cloudy_source(request, tmp_path_factory):
    """The SSP and CLOUDY grid the model is built on.

    ``synthetic`` writes a deterministic CLOUDY-format grid and uses an SSP with
    a physical ionizing flux, so the wiring runs with no external data. ``real``
    takes the bare-stellar SSP from ``download_ssp`` (the helper the CI-run
    shock test uses) and the locally built CLOUDY grid.
    """
    if request.param == "synthetic":
        grid = write_synthetic_cloudy_grid(tmp_path_factory.mktemp("cloudy") / "grid.h5")
        return {"ssp": synthetic_ionizing_ssp(), "grid": str(grid)}
    import tengri

    ssp_path = tengri.download_ssp("fsps_prsc_miles_chabrier", progress=False)
    return {"ssp": tengri.load_ssp_data(str(ssp_path)), "grid": _GRID}


def _build(source, *, approx, neb_extra=None, redshift=0.1):
    """A delayed-SFH CLOUDY model on the given SSP and grid with photometry only."""
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp

    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    neb = {
        "type": "cloudy",
        "grid": source["grid"],
        "neb_logU": Uniform(*_LOGU_PRIOR),
        "neb_logZ_gas": Uniform(*_GAS_PRIOR),
        "all_params": Fixed(DEFAULT),
    }
    neb.update(neb_extra or {})
    return SEDModel.build(
        ssp_data=source["ssp"],
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb=neb,
        redshift=Fixed(redshift),
        approx=WavePrecomp() if approx else None,
    )


@pytest.fixture(scope="module")
def table_model(cloudy_source):
    model = _build(cloudy_source, approx=True)
    model._template_data_for_jit()
    return model


@pytest.fixture(scope="module")
def percall_model(cloudy_source):
    import tengri.forward.precompute.registry as registry

    real = registry.resolve

    class _Decline:
        @staticmethod
        def precompute(*args, **kwargs):
            return None

    registry.resolve = lambda name: _Decline if name == "cloudy" else real(name)
    try:
        model = _build(cloudy_source, approx=True)
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


def test_table_is_built_through_the_registry(cloudy_source, monkeypatch):
    """(a) The table comes from ``registry.resolve("cloudy").precompute``."""
    import tengri.components.nebular.cloudy_precompute as cp
    import tengri.forward.precompute.registry as registry

    seen = []
    real_precompute = cp.precompute

    def spy(*args, **kwargs):
        seen.append(sorted(kwargs))
        return real_precompute(*args, **kwargs)

    monkeypatch.setattr(cp, "precompute", spy)
    model = _build(cloudy_source, approx=True)
    model._template_data_for_jit()
    assert model._cloudy_band_table_decline is None
    assert len(seen) == 1
    assert {"backend", "wave", "line_sigma_kms"} <= set(seen[0])
    assert "nebular_band_table" in model._template_data_for_jit()
    assert registry.resolve("cloudy") is cp


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


def test_exact_mode_carries_no_table_and_stays_exact(cloudy_source):
    """(c) Exact mode (approx=None) publishes no band table and is not table-driven."""
    model = _build(cloudy_source, approx=False)
    template = model._template_data_for_jit()
    assert template is None or "nebular_band_table" not in template
    assert getattr(model, "_cloudy_band_table_cache", None) is None


def test_free_eline_sigma_declines_the_gate(cloudy_source):
    """(d) A free neb_eline_sigma_kms declines the gate, and the reason is recorded."""
    import tengri
    from tengri import Uniform

    model = _build(
        cloudy_source, approx=True, neb_extra={"neb_eline_sigma_kms": Uniform(50.0, 150.0)}
    )
    template = model._template_data_for_jit()
    assert "nebular_band_table" not in (template or {})
    assert "neb_eline_sigma_kms" in model._cloudy_band_table_decline
    assert tengri is not None


def test_non_linear_grid_interp_declines_the_gate(cloudy_source, monkeypatch):
    """(e) A CLOUDY backend on grid_interp='triweight' declines the table, reason recorded."""
    import functools

    import tengri.components.nebular as nebular_pkg
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend

    monkeypatch.setattr(
        nebular_pkg,
        "CloudyGridBackend",
        functools.partial(CloudyGridBackend, grid_interp="triweight"),
    )
    model = _build(cloudy_source, approx=True)
    template = model._template_data_for_jit()
    assert "nebular_band_table" not in (template or {})
    assert "grid_interp='triweight'" in model._cloudy_band_table_decline
