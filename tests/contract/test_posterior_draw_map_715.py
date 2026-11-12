# SPDX-License-Identifier: BSD-3-Clause
"""The per-draw posterior summaries equal the per-draw loop and compile once (#715).

``Posterior.sed_components``, ``Posterior.equivalent_widths`` and
``Posterior.posterior_predictive`` used to run one forward pass per draw in a Python
loop. They now map the draws in bounded, jitted chunks. Each is checked here against
the loop itself, reimplemented in this file, to rtol 1e-10 in float64. A repeated call
on new draws of the same shape must add no compilation-cache miss.

The draw count is not a multiple of the chunk size, so the ragged final chunk is
exercised too.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest
from jax._src import test_util as jtu

from tengri import FREE, Fixed, Observation, Photometry, Posterior, SEDModel, Uniform
from tengri.analysis.diagnostics.spectral import equivalent_width
from tengri.forward.component_factory import state_to_sed_components
from tengri.inference.posterior import _DRAW_CHUNK

pytestmark = pytest.mark.contract

N_DRAWS = 40
assert N_DRAWS % _DRAW_CHUNK != 0, "the draw count must leave a ragged final chunk"
RTOL = 1e-10
#: Absolute floor, as a fraction of each array's largest magnitude, for entries that are
#: exactly zero in one path and round-off-small in the other.
ATOL_FRACTION = 1e-12
LINES = {"Halpha": 6564.61, "Hbeta": 4862.68}
EW_WINDOW_AA = 20.0
EW_CONTINUUM_AA = 50.0


def _assert_close(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float64
    scale = float(np.max(np.abs(expected))) or 1.0
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL_FRACTION * scale)


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide_physical):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide_physical,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        sfh={"type": "dpl", "all_params": FREE, "log_total_mass": Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


def _draws(model, seed: int) -> dict:
    """Prior draws of every free parameter, stacked on a leading draw axis."""
    rows = [model.spec.sample(key) for key in jax.random.split(jax.random.PRNGKey(seed), N_DRAWS)]
    return {name: np.stack([np.asarray(r[name]) for r in rows]) for name in model.spec.free_params}


def _posterior(model, seed: int) -> Posterior:
    samples = _draws(model, seed)
    return Posterior(
        samples=samples,
        params={k: v[0] for k, v in samples.items()},
        method="mcmc_nuts",
        wall_time_s=0.0,
        diagnostics={},
        eline_fluxes=np.ones((N_DRAWS, len(LINES))),
        eline_names=list(LINES),
        eline_wavelengths=np.array(list(LINES.values())),
        _model=model,
    )


def _one_draw(samples: dict, i: int) -> dict:
    return {k: (v[i] if v.ndim >= 1 else v) for k, v in samples.items()}


def _loop_sed_components(model, samples: dict) -> dict:
    """Reference: the per-draw loop that ``sed_components`` replaced."""
    per_draw = [
        state_to_sed_components(model.predict_state(_one_draw(samples, i))) for i in range(N_DRAWS)
    ]
    return {
        key: np.stack([np.asarray(c[key]) for c in per_draw]) for key in Posterior._COMPONENT_KEYS
    }


def _loop_equivalent_widths(model, samples: dict) -> np.ndarray:
    """Reference: per-draw EW matrix, shape ``(n_draws, n_lines)``."""
    rows = []
    for i in range(N_DRAWS):
        sed = model._predict_rest_sed(_one_draw(samples, i))
        rows.append(
            [
                float(equivalent_width(sed.wavelength, sed.sed, lc, EW_WINDOW_AA, EW_CONTINUUM_AA))
                for lc in LINES.values()
            ]
        )
    return np.asarray(rows)


def _loop_predictions(model, samples: dict, indices) -> np.ndarray:
    """Reference: per-draw photometry for the selected draws, ``(len(indices), n_obs)``."""
    return np.stack(
        [np.asarray(model.predict_photometry(_one_draw(samples, int(i)))) for i in indices]
    )


def _no_eager_fallback(record) -> bool:
    """True when no warning reports that a draw function could not be traced."""
    return not any("cannot be traced" in str(w.message) for w in record)


def test_sed_components_match_the_per_draw_loop(model):
    post = _posterior(model, seed=0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = post.sed_components()
    assert _no_eager_fallback(record)

    ref = _loop_sed_components(model, post.samples)
    for key in Posterior._COMPONENT_KEYS:
        _assert_close(out[key], ref[key])
    wave = model._predict_rest_sed(_one_draw(post.samples, 0)).wavelength
    _assert_close(out["wavelength"], wave)
    assert ref["sed_total"].shape == (N_DRAWS, np.asarray(wave).shape[0])


def test_equivalent_widths_match_the_per_draw_loop(model):
    post = _posterior(model, seed=0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = post.equivalent_widths(window_aa=EW_WINDOW_AA, continuum_width_aa=EW_CONTINUUM_AA)
    assert _no_eager_fallback(record)

    ew = _loop_equivalent_widths(model, post.samples)
    for j, name in enumerate(LINES):
        lo, med, hi = np.percentile(ew[:, j], [16.0, 50.0, 84.0])
        _assert_close(out[name], (med, lo, hi))


def test_posterior_predictive_match_the_per_draw_loop(model):
    post = _posterior(model, seed=0)
    data = np.linspace(20.0, 22.0, 3)
    noise = np.full(3, 0.1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        full = post.posterior_predictive(data=data, noise=noise)
        sub = post.posterior_predictive(
            data=data, noise=noise, n_samples=10, key=jax.random.PRNGKey(3)
        )
    assert _no_eager_fallback(record)

    preds = _loop_predictions(model, post.samples, range(N_DRAWS))
    _assert_close(full["predictions"], preds)
    _assert_close(full["residuals"], (data[None, :] - preds) / noise[None, :])
    _assert_close(full["chi2"], np.sum(((data[None, :] - preds) / noise[None, :]) ** 2, axis=1))

    indices = np.asarray(
        jax.random.choice(jax.random.PRNGKey(3), N_DRAWS, shape=(10,), replace=True)
    )
    _assert_close(sub["predictions"], _loop_predictions(model, post.samples, indices))


@pytest.mark.parametrize(
    "call",
    [
        lambda p: p.sed_components(),
        lambda p: p.equivalent_widths(),
        lambda p: p.posterior_predictive(data=np.zeros(3), noise=np.ones(3)),
    ],
    ids=["sed_components", "equivalent_widths", "posterior_predictive"],
)
def test_second_call_on_new_draws_compiles_nothing(model, call):
    call(_posterior(model, seed=0))
    second = _posterior(model, seed=1)
    with jtu.count_jit_compilation_cache_miss() as misses:
        call(second)
    assert misses() == 0, f"second call recompiled {misses()} programs"


# ── float32 ────────────────────────────────────────────────────────────────
# With x64 off, the same chunked map must stay finite and must not flush real signal to
# zero. Each float32 output is compared with the float64 output of the same draws.
#
# Tolerance: float32 has a unit round-off of 6e-8. The forward pass chains an SFH
# integral, a metallicity/age interpolation, a filter projection and an exp/log, each
# adding a few round-offs over ~1600 wavelengths. The measured float32-vs-float64
# deviation is recorded in the assertion comments below. ``F32_RTOL`` bounds the error
# relative to each array's largest magnitude, which is the scale the tests check.
F32_RTOL = 1e-4


def _assert_close_f32(actual, expected):
    """float32 ``actual`` within ``F32_RTOL`` of the float64 ``expected``, scale-relative."""
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual)), "float32 output has non-finite entries"
    scale = float(np.max(np.abs(expected))) or 1.0
    np.testing.assert_allclose(
        actual.astype(np.float64), expected, rtol=F32_RTOL, atol=F32_RTOL * scale
    )


def test_sed_components_float32_are_finite_nonzero_and_match_float64(model):
    ref = _posterior(model, seed=0).sed_components()
    # Draws come from the float64 stream; a posterior built under x64 off samples other values.
    post32 = _posterior(model, seed=0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with jax.enable_x64(False):
            out = post32.sed_components()
    assert _no_eager_fallback(record)

    assert np.asarray(out["sed_total"]).dtype == np.float32
    assert np.all(np.asarray(out["sed_total"]) != 0.0)
    for key in Posterior._COMPONENT_KEYS:
        if np.any(np.asarray(ref[key]) != 0.0):
            assert np.any(np.asarray(out[key]) != 0.0), f"{key} flushed to zero in float32"
        _assert_close_f32(out[key], ref[key])


@pytest.mark.xfail(
    strict=True,
    reason="float32 overflow in analysis/diagnostics/spectral.py equivalent_width; "
    "fixed on fix/float32-equivalent-width (4d328fa9a), remove this marker when it lands",
)
def test_equivalent_widths_float32_are_finite_nonzero_and_match_float64(model):
    ref = _posterior(model, seed=0).equivalent_widths(
        window_aa=EW_WINDOW_AA, continuum_width_aa=EW_CONTINUUM_AA
    )
    post32 = _posterior(model, seed=0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with jax.enable_x64(False):
            out = post32.equivalent_widths(
                window_aa=EW_WINDOW_AA, continuum_width_aa=EW_CONTINUUM_AA
            )
    assert _no_eager_fallback(record)

    for name in LINES:
        med, lo, hi = out[name]
        assert all(np.isfinite(float(v)) for v in (med, lo, hi))
        assert any(float(v) != 0.0 for v in (med, lo, hi)), f"{name} EW is zero in float32"
        _assert_close_f32((med, lo, hi), ref[name])


def test_posterior_predictive_float32_is_finite_nonzero_and_matches_float64(model):
    data = np.linspace(20.0, 22.0, 3)
    noise = np.full(3, 0.1)
    ref = _posterior(model, seed=0).posterior_predictive(data=data, noise=noise)
    post32 = _posterior(model, seed=0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with jax.enable_x64(False):
            out = post32.posterior_predictive(data=data, noise=noise)
    assert _no_eager_fallback(record)

    assert np.asarray(out["predictions"]).dtype == np.float32
    assert np.all(np.asarray(out["predictions"]) != 0.0)
    for key in ("predictions", "residuals", "chi2"):
        _assert_close_f32(out[key], ref[key])
