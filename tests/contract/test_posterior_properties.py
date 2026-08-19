# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the property catalog on Posterior + ``vmap_chunked`` (#1048).

API Phase 4 of #1043. The contract (§1) is **"same names, more axes"**: the keys
a ``Prediction`` answers to are exactly the keys a ``Posterior`` answers to — a
scalar becomes ``(n_samples,)`` and nothing else changes.

These tests pin three things:

* the **topology lift** — identical keys, one extra axis, values equal to the
  model evaluated sample-by-sample;
* **``vmap_chunked``** — the result must not depend on ``chunk_size`` (a
  chunk-boundary bug is otherwise invisible: it produces plausible numbers);
* the **``Posterior.derived`` deprecation** — warns, and still returns the old
  five keys bit-exactly (contract §7 demands a bit-exact shim).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, SEDModel, Uniform

pytestmark = pytest.mark.contract


# ── fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=FIXED,
    )


@pytest.fixture(scope="module")
def samples(model):
    """A stand-in posterior sample block: 37 draws — deliberately NOT a multiple
    of any chunk size we test, so a chunk-boundary bug cannot hide in an exact
    division."""
    key = jax.random.PRNGKey(0)
    return {"sfh_dpl_log_total_mass": jax.random.uniform(key, (37,), minval=9.0, maxval=11.0)}


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


@pytest.fixture(scope="module")
def model_with_spectroscopy(synthetic_ssp_wide, synthetic_tophat_obs):
    """Photometry + spectroscopy, redshift FIXED — so the draws legitimately omit it."""
    from tengri import Fixed, Observation, Spectroscopy

    obs = Observation(
        photometry=synthetic_tophat_obs.photometry,
        spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4000.0, 8000.0, 64)),
    )
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


@pytest.fixture(scope="module")
def spec_posterior(model_with_spectroscopy, samples):
    from tengri.inference.posterior import Posterior

    return Posterior(
        samples=samples,
        params={k: v[0] for k, v in samples.items()},
        method="test",
        wall_time_s=0.0,
        diagnostics={},
        _model=model_with_spectroscopy,
    )


# ── vmap_chunked ─────────────────────────────────────────────────


# Chunking is not bit-neutral: XLA compiles a DIFFERENT kernel per batch shape,
# so a chunk of 8 and a batch of 37 reassociate their reductions differently and
# the last bit can move (measured: ~1 ULP on sfr_100myr, exact on stellar_mass).
# The contract is therefore "independent of chunk_size to a few ULP" — still a
# strong test, because a real chunk-boundary bug (dropped final chunk, misordered
# concat, reused slice) produces O(1) errors, not 1e-16.
_CHUNK_RTOL = 1e-14


def test_vmap_chunked_result_is_independent_of_chunk_size(model, samples):
    """The whole point. A chunk-boundary bug yields plausible-but-wrong numbers.

    37 draws over chunk sizes 1/8/16/64 exercises: every-chunk-partial, a ragged
    final chunk, and one chunk larger than the whole batch.
    """
    from tengri import vmap_chunked

    def fn(p):
        return model.predict_properties(p, names=("stellar_mass",))["stellar_mass"]

    ref = jax.vmap(fn)(samples)  # the unchunked truth
    assert ref.shape == (37,)

    for chunk in (1, 8, 16, 64):
        got = vmap_chunked(fn, chunk_size=chunk)(samples)
        assert got.shape == (37,), f"chunk={chunk} changed the shape"
        np.testing.assert_allclose(
            np.asarray(got),
            np.asarray(ref),
            rtol=_CHUNK_RTOL,
            err_msg=f"chunk_size={chunk} changed the values beyond ULP",
        )


def test_vmap_chunked_handles_pytree_outputs(model, samples):
    """Results concatenate along the draw axis for every leaf, not just bare arrays."""
    from tengri import vmap_chunked

    def fn(p):
        return model.predict_properties(p, names=("stellar_mass", "sfr_100myr"))

    got = vmap_chunked(fn, chunk_size=8)(samples)
    ref = jax.vmap(fn)(samples)

    assert set(got) == set(ref) == {"stellar_mass", "sfr_100myr"}
    for k in ref:
        assert got[k].shape == (37,)
        np.testing.assert_allclose(np.asarray(got[k]), np.asarray(ref[k]), rtol=_CHUNK_RTOL)


def test_vmap_chunked_preserves_draw_order(model, samples):
    """Ordering is the chunk-boundary failure mode that tolerances cannot catch.

    A concat that reorders chunks still yields the right *multiset* of values, so
    an unordered comparison would pass. Pin the per-draw correspondence.
    """
    from tengri import vmap_chunked

    def fn(p):
        return model.predict_properties(p, names=("stellar_mass",))["stellar_mass"]

    got = np.asarray(vmap_chunked(fn, chunk_size=8)(samples))

    # Independent reference: the model, one draw at a time, in order.
    for i in (0, 7, 8, 31, 36):  # chunk starts, chunk ends, and the ragged tail
        one = model.predict_properties(
            {k: v[i] for k, v in samples.items()}, names=("stellar_mass",)
        )["stellar_mass"]
        np.testing.assert_allclose(got[i], float(one), rtol=_CHUNK_RTOL)


def test_vmap_chunked_falls_back_to_eager_when_fn_is_not_jittable(samples):
    """Contract: a non-jittable fn must never crash — it degrades to an eager loop.

    Probed ONCE, not per sample.
    """
    from tengri import vmap_chunked

    calls = {"n": 0}

    def not_jittable(p):
        calls["n"] += 1
        # A Python bool on a traced value: fine eagerly, ConcretizationTypeError
        # under jit. This is the shape of the real non-jittable nebular backends.
        v = p["sfh_dpl_log_total_mass"]
        if float(v) > 10.0:
            return jnp.asarray(1.0)
        return jnp.asarray(0.0)

    with pytest.warns(UserWarning, match="eager"):
        got = vmap_chunked(not_jittable, chunk_size=8)(samples)

    assert got.shape == (37,)
    expected = np.where(np.asarray(samples["sfh_dpl_log_total_mass"]) > 10.0, 1.0, 0.0)
    np.testing.assert_array_equal(np.asarray(got), expected)


def test_the_eager_fallback_announces_itself(samples):
    """Contract (#1048, #1128): the fallback must not be silent.

    The eager path is ~7x slower. A user who lands on it silently has no way to
    learn why their posterior lift crawls — and the fallback exists precisely for
    configurations nobody chose deliberately.
    """
    from tengri import vmap_chunked

    def not_jittable(p):
        return jnp.asarray(1.0 if float(p["sfh_dpl_log_total_mass"]) > 10.0 else 0.0)

    with pytest.warns(UserWarning) as record:
        vmap_chunked(not_jittable, chunk_size=8)(samples)

    assert len(record) == 1, "warn ONCE per callable, not once per draw"
    msg = str(record[0].message)
    assert "eager" in msg
    assert "ConcretizationTypeError" in msg, "the warning must name what made it non-jittable"


def test_the_probe_warns_once_not_once_per_chunk(samples):
    """The jittability probe is once per callable — so is its warning."""
    from tengri import vmap_chunked

    def not_jittable(p):
        return jnp.asarray(1.0 if float(p["sfh_dpl_log_total_mass"]) > 10.0 else 0.0)

    mapped = vmap_chunked(not_jittable, chunk_size=4)  # 37 draws / 4 = 10 chunks
    with pytest.warns(UserWarning) as record:
        mapped(samples)
        mapped(samples)  # called twice: still one warning, the probe is settled

    assert len(record) == 1


def test_a_genuine_bug_is_not_misfiled_as_not_jittable(samples):
    """Contract (#1128): only *tracing* failures mean "not jittable".

    ``except Exception`` swallowed everything — so a real bug (a typo'd key, a
    tracer leak, a shape mismatch) was silently reclassified as a fact of life and
    routed around forever. It must propagate instead.
    """
    from tengri import vmap_chunked

    def buggy(p):
        return p["a_key_that_does_not_exist"] * 2.0

    with pytest.raises(KeyError, match="a_key_that_does_not_exist"):
        vmap_chunked(buggy, chunk_size=8)(samples)


# ── the topology lift: same names, more axes ─────────────────────


def test_posterior_properties_have_the_same_keys_as_the_model(model, posterior):
    """Contract §1: identical keys on every topology."""
    assert set(posterior.properties.keys()) == set(model.available_properties)


def test_posterior_properties_add_exactly_one_axis(model, posterior, samples):
    """Scalar on a Prediction -> (n_samples,) on a Posterior. Values must agree.

    The reference is built INDEPENDENTLY — the model evaluated one sample at a
    time in a Python loop — not by re-calling the batched path under test.
    """
    got = posterior.properties["stellar_mass"]
    assert got.shape == (37,)

    expected = np.array(
        [
            float(
                model.predict_properties(
                    {k: v[i] for k, v in samples.items()}, names=("stellar_mass",)
                )["stellar_mass"]
            )
            for i in range(37)
        ]
    )
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-12)


def test_posterior_property_attribute_sugar(posterior):
    """``post.stellar_mass`` is ``post.properties["stellar_mass"]`` — as on Prediction."""
    np.testing.assert_array_equal(
        np.asarray(posterior.stellar_mass), np.asarray(posterior.properties["stellar_mass"])
    )


def test_unknown_property_raises_and_lists_the_alternatives(posterior):
    """Contract §1: never NaN/None — raise, and say what IS available."""
    with pytest.raises(KeyError, match="stellar_mass"):
        posterior.properties["stellar_masss"]


def test_ci_matches_direct_percentiles(posterior):
    """``.ci()`` is a convenience over np.percentile — it must not invent its own."""
    lo, med, hi = posterior.properties.ci("stellar_mass")
    arr = np.asarray(posterior.properties["stellar_mass"])

    exp_lo, exp_med, exp_hi = np.percentile(arr, [16.0, 50.0, 84.0])
    np.testing.assert_allclose([lo, med, hi], [exp_lo, exp_med, exp_hi], rtol=1e-12)
    assert lo <= med <= hi


def test_ci_level_is_honored(posterior):
    """A wider credible interval must actually be wider."""
    lo68, _, hi68 = posterior.properties.ci("stellar_mass", level=0.68)
    lo95, _, hi95 = posterior.properties.ci("stellar_mass", level=0.95)

    assert lo95 < lo68
    assert hi95 > hi68

    arr = np.asarray(posterior.properties["stellar_mass"])
    np.testing.assert_allclose([lo95, hi95], np.percentile(arr, [2.5, 97.5]), rtol=1e-12)


def test_to_dict_exports_the_requested_names(posterior):
    out = posterior.properties.to_dict(names=("stellar_mass", "sfr_100myr"))
    assert set(out) == {"stellar_mass", "sfr_100myr"}
    assert all(v.shape == (37,) for v in out.values())


# ── MAP posteriors (samples is None) keep the same keys, no axis ──


def test_map_posterior_keeps_the_same_keys_as_scalars(model):
    """A MAP fit has no samples. Same names — zero extra axes, not a crash."""
    from tengri.inference.posterior import Posterior

    params = {"sfh_dpl_log_total_mass": jnp.asarray(10.0)}
    post = Posterior(
        samples=None,
        params=params,
        method="map",
        wall_time_s=0.0,
        diagnostics={},
        _model=model,
    )

    assert set(post.properties.keys()) == set(model.available_properties)
    value = post.properties["stellar_mass"]
    assert np.ndim(value) == 0

    expected = model.predict_properties(params, names=("stellar_mass",))["stellar_mass"]
    np.testing.assert_array_equal(np.asarray(value), np.asarray(expected))


# ── the Posterior.derived deprecation (contract §7) ──────────────


def test_derived_warns_and_points_at_properties(posterior):
    with pytest.warns(DeprecationWarning, match="properties"):
        _ = posterior.derived


def test_derived_is_a_bit_exact_shim(model, posterior, samples):
    """Contract §7: a deprecated method keeps its EXACT old numbers for one cycle.

    ``derived`` routes through ``predict_sfh_quantities`` — a separate internal
    recompute path from the orchestrator ``state_to_*`` functions the property
    catalog is pinned to (the Phase 1 finding). So this asserts bit-equality
    against the OLD path, deliberately NOT against ``.properties``: re-routing
    the shim through the catalog would silently shift every user's numbers.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        got = posterior.derived

    assert set(got) == {
        "stellar_mass",
        "stellar_mass_surviving",
        "sfr_100myr",
        "sfr_10myr",
        "ssfr",
    }

    ref = jax.vmap(model.predict_sfh_quantities)(samples)
    np.testing.assert_array_equal(np.asarray(got["stellar_mass"]), np.asarray(ref.stellar_mass))
    np.testing.assert_array_equal(np.asarray(got["sfr_100myr"]), np.asarray(ref.sfr_100myr))


# ── the galaxy axis: CatalogPosterior ────────────────────────────


def test_catalog_properties_stack_over_galaxies(model, samples):
    """Contract §1 on the catalog topology: same names, a leading galaxy axis.

    A CatalogPosterior is a LIST of independent Posteriors (each galaxy fit
    separately), so the lift is a stack over galaxies, not a vmap.
    """
    from tengri.inference.catalog_fitter import CatalogPosterior
    from tengri.inference.posterior import Posterior

    posts = [
        Posterior(
            samples={k: v + float(g) for k, v in samples.items()},
            params={k: v[0] for k, v in samples.items()},
            method="test",
            wall_time_s=0.0,
            diagnostics={},
            _model=model,
        )
        for g in range(3)
    ]
    cat = CatalogPosterior(posteriors=posts, method="test", n_galaxies=3)

    assert set(cat.properties.keys()) == set(model.available_properties)

    got = cat.properties["stellar_mass"]
    assert got.shape == (3, 37), "expected (n_galaxies, n_samples)"

    # Each row must be that galaxy's own posterior — composed independently.
    for g in range(3):
        np.testing.assert_allclose(
            got[g], np.asarray(posts[g].properties["stellar_mass"]), rtol=1e-12
        )

    # More mass in later galaxies (we shifted log_total_mass up by g).
    assert got[2].mean() > got[0].mean()

    ci = cat.properties.ci("stellar_mass")
    assert ci.shape == (3, 3), "expected (n_galaxies, 3) for (lo, med, hi)"


# ── string indexing: cat["name"] is the median convenience (spec §9.2, #1368) ──


def _three_galaxy_catalog(model, samples):
    from tengri.inference.catalog_fitter import CatalogPosterior
    from tengri.inference.posterior import Posterior

    posts = [
        Posterior(
            samples={k: v + float(g) for k, v in samples.items()},
            params={k: v[0] for k, v in samples.items()},
            method="test",
            wall_time_s=0.0,
            diagnostics={},
            _model=model,
        )
        for g in range(3)
    ]
    return posts, CatalogPosterior(posteriors=posts, method="test", n_galaxies=3)


def test_catalog_getitem_string_is_the_median_convenience(model, samples):
    """cat["stellar_mass"] → per-galaxy medians, shape (N,) — spec §9.2 (#1368).

    Positional indexing must keep working unchanged: cat[0] is the first
    galaxy's Posterior. The two never collide — one key type each.
    """
    posts, cat = _three_galaxy_catalog(model, samples)

    med = cat["stellar_mass"]
    assert np.shape(med) == (3,), "expected one median per galaxy"
    per_galaxy = np.asarray(cat.properties["stellar_mass"])  # (3, n_samples)
    np.testing.assert_allclose(np.asarray(med), np.median(per_galaxy, axis=1), rtol=1e-12)

    assert cat[0] is posts[0]
    assert cat[-1] is posts[-1]
    assert cat[1:3] == posts[1:3]


def test_catalog_getitem_string_answers_from_summary_store():
    """store="summary" has no per-galaxy posteriors — the medians must come
    from the stored percentile block (same median column to_table uses)."""
    from tengri.inference.catalog_fitter import CatalogPosterior

    pct = {"stellar_mass": np.array([[9.0, 10.0, 11.0], [8.0, 9.0, 10.0]])}
    cat = CatalogPosterior(
        posteriors=[], method="test", n_galaxies=2, store="summary", percentiles=pct
    )
    np.testing.assert_allclose(np.asarray(cat["stellar_mass"]), [10.0, 9.0])


def test_catalog_getitem_unknown_name_names_the_available_keys(model, samples):
    """An unknown property name raises KeyError that teaches what exists."""
    _, cat = _three_galaxy_catalog(model, samples)

    with pytest.raises(KeyError, match="stellar_mass"):
        cat["definitely_not_a_property"]


# ── observables over the sample axis (contract §3: exact by default) ──


def test_observables_shape_and_exactness(model, posterior, samples):
    """Exact by default: each draw's bands equal the model's own exact photometry.

    The reference is composed INDEPENDENTLY — ``model.predict(p).photometry()``,
    the Prediction surface — not by re-calling the batched path under test.
    """
    got = posterior.observables()
    n_filters = model.observation.photometry.n_filters
    assert got.shape == (37, n_filters)

    for i in (0, 18, 36):
        one = model.predict({k: v[i] for k, v in samples.items()}).photometry()
        np.testing.assert_allclose(np.asarray(got[i]), np.asarray(one), rtol=1e-12)


def test_observables_thins_to_n_draws(posterior, model):
    got = posterior.observables(n_draws=5, key=jax.random.PRNGKey(1))
    assert got.shape == (5, model.observation.photometry.n_filters)


def test_observables_fast_is_opt_in_not_default(posterior):
    """``approx=True`` must be a *different* code path, or the flag is a lie.

    On a model built with no ``approx=``, the lean path and the exact path
    coincide numerically — so this pins the CONTRACT (both run, both finite,
    same shape) rather than asserting a difference that this model cannot show.
    """
    exact = posterior.observables(n_draws=4, key=jax.random.PRNGKey(2))
    fast = posterior.observables(n_draws=4, key=jax.random.PRNGKey(2), approx=True)

    assert exact.shape == fast.shape
    assert np.all(np.isfinite(exact)) and np.all(np.isfinite(fast))
    # This model carries no build-time LUT, so the two paths must agree here.
    # (A WavePrecomp model is where they diverge — that is Phase 2's territory.)
    np.testing.assert_allclose(np.asarray(fast), np.asarray(exact), rtol=1e-10)


def test_observables_filters_means_the_same_thing_as_on_prediction(model, posterior):
    """Contract (#1129): ``filters=`` must not mean two different things.

    ``Prediction.photometry(filters=["sdss_g"])`` took filter *names*;
    ``Posterior.observables(filters=...)`` took a Photometry *object*. Same
    keyword, same concept, two incompatible types — in the API-consistency
    campaign's own surface. Both now route through one normalizer.
    """
    from tengri.observation.photometry_config import Photometry

    names = ["sdss_g", "sdss_r"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the one-time runtime-photometry notice
        by_name = posterior.observables(filters=names)
        by_object = posterior.observables(filters=Photometry.from_names(names))

    assert by_name.shape == (len(posterior.samples["sfh_dpl_log_total_mass"]), 2)
    np.testing.assert_allclose(np.asarray(by_name), np.asarray(by_object), rtol=1e-12)


def test_spectra_lifts_the_spectrum_over_draws(model_with_spectroscopy, spec_posterior):
    """Contract (#1048, #1129): the seam must give spectrum draws, not only bands.

    #1048 asked ``observables`` for "band/spectrum sample arrays"; only bands
    shipped. For a spectroscopic fit the spectrum draws ARE the SED plot, and
    without them callers loop ``predict_spectrum`` per draw — the exact memory
    problem ``vmap_chunked`` was added to solve.
    """
    spec = spec_posterior.spectra()
    n_draws = spec_posterior.samples["sfh_dpl_log_total_mass"].shape[0]

    assert spec.ndim == 2
    assert spec.shape[0] == n_draws
    assert np.all(np.isfinite(spec))

    # Exact by default: the same kernel Prediction.spectrum uses, per draw.
    first = {k: v[0] for k, v in spec_posterior.samples.items()}
    expected = model_with_spectroscopy.predict(first).spectrum()
    np.testing.assert_allclose(np.asarray(spec[0]), np.asarray(expected), rtol=1e-10)


def test_spectra_honors_a_fixed_redshift(model_with_spectroscopy, spec_posterior):
    """The draws carry only free params — a Fixed redshift must still reach the projector.

    Not a hypothetical: the identical bug was introduced in ``observables`` while
    fixing it in ``Prediction`` (#1124), and was still live in
    ``measure_line_fluxes`` (#1127). Both lifts now share ``_draws_for_lift``.

    The exact spectrum path runs ``Observation.predict``, which takes the
    luminosity distance from the params **dict** — so if the draws did not carry
    the resolved redshift, every draw would come back at 10 pc.
    """
    assert "redshift" not in spec_posterior.samples  # vacuity guard: it IS omitted

    spec = np.asarray(spec_posterior.spectra())
    first = {k: v[0] for k, v in spec_posterior.samples.items()}

    at_z = np.asarray(model_with_spectroscopy.predict({**first, "redshift": 0.5}).spectrum())
    at_zero = np.asarray(model_with_spectroscopy.predict({**first, "redshift": 0.0}).spectrum())

    # Power check: z must genuinely move the spectrum, or this proves nothing.
    assert np.nanmax(np.abs(at_zero / at_z)) > 1e3

    np.testing.assert_allclose(spec[0], at_z, rtol=1e-10)


def test_spectra_without_spectroscopy_raises_clearly(posterior):
    """No spectroscopy is a user error, not a silent empty array."""
    with pytest.raises(RuntimeError, match="spectroscopy"):
        posterior.spectra()


def test_spectra_fast_is_opt_in_and_not_a_dropped_kwarg(spec_posterior):
    """``approx=True`` on a model with no SpectrumPrecomp must RAISE, not silently
    hand back the exact answer.

    A ``fast`` flag that is accepted and then ignored is the dropped-kwarg
    failure mode: the user believes they opted into the LUT, the docstring says
    they did, and the number says otherwise.
    """
    with pytest.raises(ValueError, match="SpectrumPrecomp"):
        spec_posterior.spectra(approx=True)


# ── the population topology ──────────────────────────────────────


def test_population_properties_merge_the_shared_hyperparameters(model, samples):
    """A per-galaxy block is NOT a complete parameter set in a hierarchical fit.

    The shared hyperparameters must be merged in, or the properties silently
    answer a different question. Here the model has no shared params, so the
    merge is a no-op numerically — what this pins is that the galaxy axis exists,
    carries the right keys, and equals each galaxy's own posterior.
    """
    from tengri.inference.hierarchical import PopulationPosterior
    from tengri.inference.posterior import Posterior

    individual = [{k: v + float(g) for k, v in samples.items()} for g in range(2)]
    pop = PopulationPosterior(
        shared_samples={},
        shared_params={},
        individual_samples=individual,
        method="test",
        _model=model,
    )

    assert set(pop.properties.keys()) == set(model.available_properties)

    got = pop.properties["stellar_mass"]
    assert got.shape == (2, 37)

    for g in range(2):
        ref = Posterior(
            samples=individual[g],
            params={},
            method="test",
            wall_time_s=0.0,
            diagnostics={},
            _model=model,
        )
        np.testing.assert_allclose(got[g], np.asarray(ref.properties["stellar_mass"]), rtol=1e-12)


def test_population_without_individual_samples_raises_clearly(model):
    """No per-galaxy samples -> a clear error, never a silent empty array."""
    from tengri.inference.hierarchical import PopulationPosterior

    pop = PopulationPosterior(
        shared_samples={"x": jnp.ones(3)},
        shared_params={},
        individual_samples=None,
        method="test",
        _model=model,
    )
    with pytest.raises(RuntimeError, match="no per-galaxy samples"):
        pop.properties["stellar_mass"]


def test_properties_agree_with_derived_to_tolerance(posterior):
    """The two paths are NOT bit-identical (different recompute routes) but must agree.

    Pinning the measured agreement documents the ULP-level divergence instead of
    asserting a false bit-equality — the same discipline Phase 0/1 used.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = np.asarray(posterior.derived["stellar_mass"])

    new = np.asarray(posterior.properties["stellar_mass"])
    np.testing.assert_allclose(new, old, rtol=1e-10)
