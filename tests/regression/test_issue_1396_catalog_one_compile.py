# SPDX-License-Identifier: BSD-3-Clause
"""One compile for the whole catalog, at fixed ``n_t`` (#1396 acceptance).

The load-bearing performance claim behind simulation catalogs: ``WavePrecomp``
bakes the SFH-independent SSP x filter integral, so a tabulated history changes
only the (met, age) weights, and the *whole catalog* should be one compiled
program rather than a per-galaxy recompile — the #1316 cliff the catalog path
exists to remove.

**Why this test needs care.** A Python-level call counter measures nothing here:
``jax.vmap`` is not a compile boundary, so counting invocations reports the
*chunk count*, and such a test is green while measuring nothing. What actually
happens without ``jit`` is worse than "a few" compiles — ``vmap`` alone
dispatches op by op, so every primitive is compiled separately and keyed on its
own shapes. Measured on this model before the fix: **236 compiles** for a bare
``vmap(predict_photometry)`` versus **1** for ``jit(vmap(...))``, and a ragged
trailing chunk is a second set of shapes that pays the whole cost again.

So the assertions here are on the number of **compiled programs** — read from
the catalog's own memoized ``jit`` wrapper, which is the object whose cache the
claim is about — plus a global compile counter as the guard against the jit
being removed entirely.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.regression_bug

_Z_OBS = 0.05
_T_GYR = np.concatenate([np.array([0.0]), np.linspace(1.0, 13.0, 39)])


def _cache_size(cached):
    """Number of compiled programs held by the catalog's batched callable.

    ``_cache_size`` is JAX-internal, so this distinguishes the two ways it can
    go missing — they need opposite fixes and must not share a message:

    * the callable is not jitted at all (someone dropped the ``jax.jit``), which
      is a **source** regression;
    * it is jitted but JAX moved the accessor, which is a **test** repair.

    Either way it fails rather than skips. The acceptance criterion here is a
    compile count, and a silently skipped count test is precisely the invisible
    coverage this suite exists to prevent.
    """
    if not hasattr(cached, "lower"):
        raise AssertionError(
            f"the catalog's batched callable is not jitted (got "
            f"{type(cached).__name__}). jax.vmap alone is not a compile "
            f"boundary — it dispatches op by op, which measured 236 compiles on "
            f"this model. Restore jax.jit(jax.vmap(...)) in Catalog._batched."
        )
    if not hasattr(cached, "_cache_size"):
        raise AssertionError(
            f"jax {jax.__version__} no longer exposes _cache_size() on a jitted "
            f"callable; re-point this helper at the current accessor."
        )
    return cached._cache_size()


@pytest.fixture
def fwd_table(synthetic_ssp_wide, synthetic_tophat_obs):
    from tengri import FIXED, ForwardModel, SEDModel
    from tengri.parameters.priors import Fixed, Uniform

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sed = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "table"},
            dust={
                "type": "two_component",
                "all_params": FIXED,
                "tau_bc": 0.5,
                "tau_diff": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Fixed(_Z_OBS),
        )
        return ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)


def _catalog(fwd, n):
    """A tabulated catalog of n galaxies on a shared (fixed) n_t grid."""
    from tengri import Catalog

    n_t = _T_GYR.shape[0]
    shape = np.ones(n_t)
    shape[0] = 0.0
    t = np.broadcast_to(_T_GYR, (n, n_t)).copy()
    sfr = np.stack([shape * (1.0 + i) for i in range(n)])
    return Catalog.from_histories(fwd, t_gyr=t, sfr=sfr, params={"dust_tau_diff": np.full(n, 0.2)})


def test_whole_catalog_is_one_compile(fwd_table):
    """Eight galaxies in two chunks must compile exactly one program."""
    cat = _catalog(fwd_table, 8)
    cat.predict(chunk_size=4)

    assert _cache_size(cat._batched_cache["photometry"]) == 1


def test_a_ragged_chunk_division_is_still_one_compile(fwd_table):
    """chunk_size=3 over 8 galaxies would be widths 3 and 2 — padding makes it one.

    This is the ``n_pad`` motivation stated on the galaxy axis: the XLA cache
    keys on shape, so an uneven trailing chunk is a second program and doubles
    the compile cost. Without the padding this assertion reads 2.
    """
    cat = _catalog(fwd_table, 8)
    cat.predict(chunk_size=3)

    assert _cache_size(cat._batched_cache["photometry"]) == 1


def test_padding_does_not_change_the_answer(fwd_table):
    """The padded rows must be discarded, not averaged in or left attached."""
    cat = _catalog(fwd_table, 8)
    whole = np.asarray(cat.predict(chunk_size=8))
    ragged = np.asarray(cat.predict(chunk_size=3))

    assert ragged.shape == whole.shape == (8, fwd_table.observation.photometry.n_filters)
    assert np.allclose(ragged / whole, 1.0, rtol=1e-12)


def test_repeating_a_prediction_compiles_nothing_new(fwd_table):
    """The memoized wrapper must survive across calls, or every call recompiles."""
    cat = _catalog(fwd_table, 8)
    cat.predict(chunk_size=4)
    before = _cache_size(cat._batched_cache["photometry"])

    cat.predict(chunk_size=4)

    assert _cache_size(cat._batched_cache["photometry"]) == before == 1


def test_n_pad_lets_different_sized_catalogs_share_one_program(fwd_table):
    """Catalogs of different N share a cache entry when padded to a common size.

    Without n_pad each distinct galaxy count is its own leading dimension and so
    its own program; with it, both compile to the same shape. Asserted on a
    shared wrapper, so the two catalogs must genuinely reuse one entry.
    """
    cat_a = _catalog(fwd_table, 5)
    cat_b = _catalog(fwd_table, 8)
    # Share one memoized wrapper between them, which is what "share a cache
    # entry" has to mean — two Catalog objects otherwise hold separate jits.
    cat_b._batched_cache = cat_a._batched_cache

    cat_a.predict(chunk_size=16, n_pad=16)
    cat_b.predict(chunk_size=16, n_pad=16)

    assert _cache_size(cat_a._batched_cache["photometry"]) == 1


def test_n_pad_below_the_catalog_size_is_refused(fwd_table):
    """n_pad pads up to a shared size; it must never silently truncate."""
    cat = _catalog(fwd_table, 8)
    with pytest.raises(ValueError, match="smaller than the catalog"):
        cat.predict(n_pad=3)


def test_the_batched_call_is_jitted_not_dispatched_op_by_op(fwd_table):
    """Guard the 236x, self-calibrated against a bare vmap on the SAME shape.

    An earlier version of this test compared a *warm* repeat against zero, and
    it passed with the jit removed: op-level caches are global and keyed on
    shape, so a second identically-shaped call costs nothing either way. That
    measured memoization, not jitting.

    The discriminator has to be a **cold** shape, and the honest bound is the
    bare-vmap cost itself rather than a number hard-coded from one machine. The
    jitted path builds one program; the unjitted path compiles every primitive
    separately, so the gap is orders of magnitude, not a few counts.
    """
    from jax._src import test_util as jtu

    width = 7  # not used by any other test here, so it is a cold shape
    cat = _catalog(fwd_table, width)
    columns, _n = cat._prediction_columns(None)

    with jtu.count_jit_compilation_cache_miss() as jit_count:
        cat.predict(chunk_size=width)
    jitted = jit_count()

    # Same columns, same shape, but dispatched op by op.
    with jtu.count_jit_compilation_cache_miss() as raw_count:
        jax.vmap(fwd_table.predict_photometry)(columns)
    raw = raw_count()

    assert raw > 20, (
        f"the calibration arm compiled only {raw} programs, so this test cannot "
        f"give a verdict. Most likely the batched path is no longer jitted and "
        f"already compiled these very ops in the first arm (see the other "
        f"failures in this file); otherwise the forward model or JAX's dispatch "
        f"changed and the guard needs revisiting."
    )
    assert jitted * 10 < raw, (
        f"the batched call compiled {jitted} programs against {raw} for a bare "
        f"vmap on the same shape — it is being dispatched op by op, so the jit "
        f"in Catalog._batched is gone"
    )


def test_simulate_channels_get_separate_cache_entries(fwd_table):
    """Photometry, lines and properties are different programs — and stay separate.

    Sharing one entry across channels would return one channel's numbers for
    another. Each is tagged, and the line tag carries the line set, so asking
    for a different set cannot reuse the wrong program.
    """
    cat = _catalog(fwd_table, 4)
    cat.simulate(lines=("Halpha",), properties=("stellar_mass",), chunk_size=4)

    tags = set(cat._batched_cache)
    assert "photometry" in tags
    assert any(t.startswith("lines:") for t in tags)
    assert any(t.startswith("properties:") for t in tags)
    for tag in tags:
        assert _cache_size(cat._batched_cache[tag]) == 1

    cat.simulate(lines=("Halpha", "OIII_5007"), chunk_size=4)
    line_tags = {t for t in cat._batched_cache if t.startswith("lines:")}
    assert len(line_tags) == 2, f"a different line set reused a cache entry: {line_tags}"
