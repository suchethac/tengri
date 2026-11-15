# SPDX-License-Identifier: BSD-3-Clause
"""Chunked per-draw maps: plotting, SBI draws and the ``vmap_chunked`` kernel cache (#3359).

Contract for three things that changed together:

1. :func:`tengri.utils.batching.vmap_chunked` keeps no state between calls. The
   compiled kernel is cached per model by :func:`tengri.forward.draw_maps.map_draws`,
   keyed by method name and ``compile_signature()``: a stale kernel is never served
   after a structural mutation, and the model is collectable once dropped.
2. The posterior plots (``posterior_plot_sfh``, ``posterior_plot_sed``,
   ``plot_sfh``, ``SEDModel.plot_sfh_posterior``) and their helper
   :func:`tengri.forward.draw_maps.map_draws` reproduce the per-draw Python loop
   they replaced, in float64, across a ragged final chunk.
3. ``generate_sbi_training_data`` evaluates the forward model through
   ``vmap_chunked``, and its result does not depend on ``chunk_size``.

The draw count (37) is deliberately not a multiple of the chunk sizes used, so a
chunk-boundary bug cannot hide in an exact division.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

from tengri import DEFAULT, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.forward.draw_maps import DRAW_CHUNK, map_draws
from tengri.utils.batching import vmap_chunked

pytestmark = pytest.mark.contract

N_DRAWS = 37
RTOL = 1e-10


# ── fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(DEFAULT),
    )


@pytest.fixture(scope="module")
def samples():
    key = jax.random.PRNGKey(3359)
    return {"sfh_dpl_log_total_mass": jax.random.uniform(key, (N_DRAWS,), minval=9.0, maxval=11.0)}


@pytest.fixture(scope="module")
def posterior(model, samples):
    from tengri.inference.posterior import Posterior

    return Posterior(
        samples=samples,
        params={k: v[0] for k, v in samples.items()},
        method="test",
        wall_time_s=0.0,
        diagnostics={},
        _model=model,
    )


def _loop(fn, samples):
    """The per-draw Python loop that the chunked map replaces."""
    n = next(iter(samples.values())).shape[0]
    return np.array([np.asarray(fn({k: v[i] for k, v in samples.items()})) for i in range(n)])


def _assert_band(ax, lo, hi):
    """Both 16th and 84th percentile curves appear, verbatim, in ``ax``'s fill band.

    Matplotlib's polygon layout is not a contract, so the check is that each expected
    curve is a contiguous run of the band's vertex y-values, within float64 rtol.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    poly = next(c for c in ax.collections if isinstance(c, PolyCollection))
    ys = np.concatenate([p.vertices[:, 1] for p in poly.get_paths()])
    for name, curve in (("lo", lo), ("hi", hi)):
        # The upper edge is traversed in reverse to close the polygon.
        found = False
        for cand in (curve, curve[::-1]):
            windows = sliding_window_view(ys, len(cand))
            found = found or bool(
                np.all(np.isclose(windows, cand, rtol=RTOL, atol=0.0), axis=1).any()
            )
        assert found, f"{name} percentile curve not found in the plotted band"


# ── 1. vmap_chunked keeps no state; owners cache their own kernels ─


def test_vmap_chunked_keeps_no_state_between_calls():
    """Same callable twice still recompiles: nothing global caches ``fn``."""
    traces = []

    def fn(p):
        traces.append(1)  # runs at trace time only
        return p["x"] ** 2

    batch = {"x": jnp.arange(9.0)}  # chunks of 4, 4, 1: two distinct shapes per call
    first = vmap_chunked(fn, chunk_size=4)(batch)
    per_call = len(traces)
    second = vmap_chunked(fn, chunk_size=4)(batch)
    assert len(traces) == 2 * per_call
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_map_draws_reuses_the_owner_kernel_for_an_unchanged_model(model, samples):
    map_draws(model, "_sfr_mean_draw", samples)
    first = vars(model)["_draw_kernels"]["_sfr_mean_draw"][1]
    map_draws(model, "_sfr_mean_draw", samples)
    assert vars(model)["_draw_kernels"]["_sfr_mean_draw"][1] is first


def test_map_draws_stale_kernel_is_replaced_when_the_signature_changes(
    synthetic_ssp_wide, synthetic_tophat_obs, samples
):
    model = _build_cue_model(synthetic_ssp_wide, synthetic_tophat_obs)
    wl = np.asarray([6564.61, 4862.68])
    map_draws(model, "predict_photometry", samples)
    first = vars(model)["_draw_kernels"]["predict_photometry"]
    model.enable_fast_nebular(wl, n_grid=8)
    got = map_draws(model, "predict_photometry", samples)
    assert vars(model)["_draw_kernels"]["predict_photometry"][0] != first[0]
    np.testing.assert_allclose(got, _loop(model.predict_photometry, samples), rtol=RTOL, atol=0.0)


def test_ragged_chunk_matches_unchunked_float64():
    def fn(p):
        return jnp.sin(p["x"]) * p["y"] + p["x"] ** 3

    batch = {"x": jnp.linspace(-2.0, 2.0, N_DRAWS), "y": jnp.linspace(0.1, 3.0, N_DRAWS)}
    full = np.asarray(jax.vmap(fn)(batch))
    for chunk in (1, 8, 16, N_DRAWS, 1000):
        out = np.asarray(vmap_chunked(fn, chunk_size=chunk)(batch))
        assert out.shape == full.shape
        np.testing.assert_allclose(out, full, rtol=RTOL, atol=0.0)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError, match="chunk_size"):
        vmap_chunked(lambda p: p["x"], chunk_size=0)


# ── 2. plotting helpers and posterior plots ───────────────────────


def test_map_draws_sfr_mean_matches_loop(model, samples):
    assert N_DRAWS % DRAW_CHUNK != 0  # the ragged chunk is exercised
    got = map_draws(model, "_sfr_mean_draw", samples)
    ref = _loop(model._sfr_mean_draw, samples)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=0.0)


def test_map_draws_sfr_full_matches_loop(model, samples):
    got = map_draws(model, "_sfr_full_draw", samples)
    ref = _loop(model._sfr_full_draw, samples)
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=0.0)


def test_map_draws_rest_sed_matches_loop(model, samples):
    got = map_draws(model, "_rest_sed_draw", samples)
    ref = _loop(model._rest_sed_draw, samples)
    assert got.shape == (N_DRAWS, ref.shape[1])
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=0.0)


def test_posterior_plot_sfh_band_matches_loop(posterior, model, samples):
    from tengri.analysis.plotting.convergence import posterior_plot_sfh

    fig, ax = posterior_plot_sfh(posterior)
    try:
        ref = _loop(model._sfr_mean_draw, samples)
        _assert_band(ax, np.percentile(ref, 16, axis=0), np.percentile(ref, 84, axis=0))
    finally:
        plt.close(fig)


def test_posterior_plot_sed_bands_match_loop(posterior, model, samples):
    from tengri.analysis.plotting.sed import posterior_plot_sed

    fig, axes = posterior_plot_sed(posterior)
    try:
        sed_ref = _loop(model._rest_sed_draw, samples)
        sfh_ref = _loop(model._sfr_mean_draw, samples)
        _assert_band(axes.flat[0], *np.percentile(sed_ref, (16, 84), axis=0))
        _assert_band(axes.flat[1], *np.percentile(sfh_ref, (16, 84), axis=0))
    finally:
        plt.close(fig)


def test_plot_sfh_band_matches_loop(posterior, model, samples):
    from tengri.analysis.plotting.sfh import plot_sfh

    fig, ax = plt.subplots()
    try:
        plot_sfh(model, posterior, ax=ax, show_draws=False)
        ref = _loop(model._sfr_full_draw, samples)
        _assert_band(ax, *np.percentile(ref, (16, 84), axis=0))
    finally:
        plt.close(fig)


def test_plot_sfh_posterior_band_matches_loop(posterior, model, samples):
    fig, ax = plt.subplots()
    try:
        model.plot_sfh_posterior(posterior, ax=ax, n_draws=5)
        ref = _loop(model._sfr_full_draw, samples)
        _assert_band(ax, *np.percentile(ref, (16, 84), axis=0))
    finally:
        plt.close(fig)


# ── 3. SBI draws ──────────────────────────────────────────────────


def test_sbi_training_data_independent_of_chunk_size(model):
    from tengri.inference.backends.sbi import generate_sbi_training_data

    key = jax.random.PRNGKey(7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small = generate_sbi_training_data(
            model, model.spec, n_samples=N_DRAWS, key=key, chunk_size=8
        )
        large = generate_sbi_training_data(
            model, model.spec, n_samples=N_DRAWS, key=key, chunk_size=1024
        )
    assert small["x"].shape == large["x"].shape == (N_DRAWS, 5)
    np.testing.assert_allclose(np.asarray(small["x"]), np.asarray(large["x"]), rtol=RTOL, atol=0.0)
    np.testing.assert_array_equal(np.asarray(small["theta"]), np.asarray(large["theta"]))


# ── 4. owner-scoped kernels: staleness and retention (#3359 review) ──


def _build_cue_model(ssp, obs):
    """A Cue-nebular model on the WavePrecomp LUT, so enable_fast_nebular moves photometry."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(DEFAULT),
    )


def _batch(samples):
    return {k: jnp.asarray(v) for k, v in samples.items()}


def test_compile_relevant_mutation_is_not_served_a_stale_kernel(
    synthetic_ssp_wide, synthetic_tophat_obs, samples
):
    """Mutate the model between two mapped calls; the second must equal the eager loop.

    ``enable_fast_nebular`` moves the LUT photometry (about 2e-6 relative here), so the
    mapped function is the bound ``predict_photometry``, which is what the cache keys on.
    """
    wl = np.asarray([6564.61, 4862.68])  # Halpha, Hbeta (vacuum)
    model = _build_cue_model(synthetic_ssp_wide, synthetic_tophat_obs)
    vmap_chunked(model.predict_photometry, chunk_size=DRAW_CHUNK)(_batch(samples))  # warm

    before = _loop(model.predict_photometry, samples)
    sig_before = model.compile_signature()
    model.enable_fast_nebular(wl, n_grid=8)
    assert model.compile_signature() != sig_before
    after = _loop(model.predict_photometry, samples)
    # The mutation must move the output past rtol 1e-10, or this test cannot discriminate.
    assert not np.allclose(before, after, rtol=RTOL, atol=0.0)

    got = np.asarray(
        vmap_chunked(model.predict_photometry, chunk_size=DRAW_CHUNK)(_batch(samples))
    )
    np.testing.assert_allclose(got, after, rtol=RTOL, atol=0.0)


def test_mutated_model_retires_the_kernel_filed_under_its_old_signature(
    synthetic_ssp_wide, synthetic_tophat_obs, samples
):
    """A second model must not be traced over a first model that was mutated in place.

    The owner files one kernel per compile signature and that kernel reads the model
    that built it at trace time. ``first`` builds the kernel, is mutated before the
    kernel is ever traced, and ``second`` (same signature as ``first`` had) then calls
    it: without retirement ``second`` is traced over the fast-nebular ``first``.
    """
    from tengri.inference._model_cache import _default_owner

    _default_owner.clear()  # a kernel an earlier test already traced would hide the defect
    wl = np.asarray([6564.61, 4862.68])  # Halpha, Hbeta (vacuum)
    first = _build_cue_model(synthetic_ssp_wide, synthetic_tophat_obs)
    first._get_or_build_predict_observables_jit()  # built, not traced
    first.enable_fast_nebular(wl, n_grid=8)
    fast = _loop(first.predict_photometry, samples)

    second = _build_cue_model(synthetic_ssp_wide, synthetic_tophat_obs)
    exact = _loop(second.predict_photometry, samples)
    assert not np.allclose(exact, fast, rtol=RTOL, atol=0.0)

    second.enable_fast_nebular(wl, n_grid=8)
    np.testing.assert_allclose(
        _loop(second.predict_photometry, samples), fast, rtol=RTOL, atol=0.0
    )


def test_mapped_kernel_does_not_keep_the_model_alive(
    synthetic_ssp_wide, synthetic_tophat_obs, samples
):
    """After a mapped call and ``del``, nothing global still references the model."""
    import gc
    import weakref

    model = _build_cue_model(synthetic_ssp_wide, synthetic_tophat_obs)
    vmap_chunked(model._rest_sed_draw, chunk_size=DRAW_CHUNK)(_batch(samples))
    ref = weakref.ref(model)
    del model
    gc.collect()
    assert ref() is None


def test_owner_cached_kernel_does_not_keep_the_model_alive(
    synthetic_ssp_wide, synthetic_tophat_obs, samples
):
    """The kernel cached on the model by ``map_draws`` is collected with the model."""
    import gc
    import weakref

    model = _build_cue_model(synthetic_ssp_wide, synthetic_tophat_obs)
    map_draws(model, "_rest_sed_draw", samples)
    assert "_draw_kernels" in vars(model)
    ref = weakref.ref(model)
    del model
    gc.collect()
    assert ref() is None
