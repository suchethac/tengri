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
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
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

    got = vmap_chunked(not_jittable, chunk_size=8)(samples)

    assert got.shape == (37,)
    expected = np.where(np.asarray(samples["sfh_dpl_log_total_mass"]) > 10.0, 1.0, 0.0)
    np.testing.assert_array_equal(np.asarray(got), expected)


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
