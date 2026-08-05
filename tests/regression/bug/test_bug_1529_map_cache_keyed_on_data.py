# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the MAP-init cache must not cross datasets (#1529).

``_maybe_map_init`` caches the MAP point in the per-model namespace returned by
``ModelCacheOwner.get_or_compile_model``, a ``WeakKeyDictionary`` keyed on the
**model object**. Nothing in that key mentions the data. The documented and
intended win is cheap: a repeated fit of the same target skips a fresh MAP run.

The loop that breaks it is the ordinary one for a catalog::

    model = build_model(ssp)  # built once, deliberately
    for i in range(n_galaxies):
        Fitter(model, flux[i], err[i]).run("mcmc_nuts")

Galaxy 0 populates the cache. Galaxies 1..N-1 then start their chains at galaxy
0's MAP -- a point that can sit far out in their own posterior tails, or outside
the region their sampler can recover from at all.

Observed while fitting a 2048-galaxy mock bank: six of eight NUTS fits died,
each returning in ~5 s against a normal ~600 s, with R-hat up to 10.74 and
**zero divergences**. Zero divergences reads as health; it equally describes a
sampler that never took a step.

The fix keys reuse on a content fingerprint of ``(data, noise)``, so the
speed-up survives for a genuine refit of the same target and cannot leak across
targets. These tests use stub fitters rather than a real model so they run in
the PR gate, where SSP data is absent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference._sample_utils import _maybe_map_init

pytestmark = pytest.mark.regression_bug


class _Model:
    """Weakref-able stand-in; the cache keys on object identity alone."""


class _MapResult:
    def __init__(self, params):
        self.params = params
        self.diagnostics = {"final_loss": 0.0}


class _StubFitter:
    """Minimal surface ``_maybe_map_init`` touches, counting its MAP runs."""

    def __init__(self, model, data, noise):
        self.model = model
        self.data = jnp.asarray(data)
        self.noise = jnp.asarray(noise)
        self.spec = None
        self._free_names = ()
        self.n_map_runs = 0

    def _run_map(self, *, key, n_steps, n_restarts, verbose):
        self.n_map_runs += 1
        # A MAP that depends on the data, so a leaked cache entry is visible
        # in the returned value and not only in the run counter.
        return _MapResult({"a": jnp.asarray(float(jnp.sum(self.data)))})

    def _unbounded_from_posterior(self, posterior):
        return dict(posterior.params)


def _init(fitter):
    params, _ = _maybe_map_init(fitter, jax.random.PRNGKey(0), None, False)
    return params


class TestTheCacheDoesNotCrossDatasets:
    def test_a_different_dataset_runs_its_own_map(self):
        """LOAD-BEARING. Neuter: drop the fingerprint comparison.

        Without it the second fitter reports ``n_map_runs == 0`` and silently
        starts from the first galaxy's MAP.
        """
        model = _Model()
        first = _StubFitter(model, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        second = _StubFitter(model, [9.0, 9.0, 9.0], [0.1, 0.1, 0.1])

        _init(first)
        _init(second)

        assert second.n_map_runs == 1, "second dataset reused the first one's MAP"

    def test_the_returned_init_belongs_to_the_second_dataset(self):
        """The run counter alone could pass while the wrong value is returned."""
        model = _Model()
        first = _StubFitter(model, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        second = _StubFitter(model, [9.0, 9.0, 9.0], [0.1, 0.1, 0.1])

        _init(first)
        got = _init(second)

        assert float(got["a"]) == pytest.approx(27.0), (
            "init came from the first dataset (sum 6.0), not the second (27.0)"
        )

    def test_noise_alone_is_enough_to_separate_two_targets(self):
        """Two galaxies can share fluxes and differ only in their errors; the
        posterior differs, so the MAP must not be shared."""
        model = _Model()
        first = _StubFitter(model, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        second = _StubFitter(model, [1.0, 2.0, 3.0], [5.0, 5.0, 5.0])

        _init(first)
        _init(second)

        assert second.n_map_runs == 1, "differing noise did not invalidate the cache"


class TestTheCacheStillWorks:
    def test_the_same_data_reuses_the_cached_map(self):
        """The feature this cache exists for. Losing it would turn a silent
        correctness bug into a silent performance one."""
        model = _Model()
        first = _StubFitter(model, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        again = _StubFitter(model, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])

        _init(first)
        _init(again)

        assert again.n_map_runs == 0, "an identical refit paid for a fresh MAP"

    def test_equal_values_from_a_different_array_object_still_hit(self):
        """Fingerprint on content, not on object identity -- data reloaded from
        disk between runs is a hit, which is the documented cross-session win."""
        model = _Model()
        first = _StubFitter(model, np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.1, 0.1]))
        again = _StubFitter(model, jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([0.1, 0.1, 0.1]))

        _init(first)
        _init(again)

        assert again.n_map_runs == 0

    def test_a_separate_model_object_is_unaffected(self):
        """Per-model isolation is the pre-existing contract and must hold."""
        first = _StubFitter(_Model(), [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        second = _StubFitter(_Model(), [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])

        _init(first)
        _init(second)

        assert second.n_map_runs == 1

    def test_explicit_init_from_still_short_circuits(self):
        """``init_from`` precedes the cache in the documented resolution order."""
        fitter = _StubFitter(_Model(), [1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
        params, _ = _maybe_map_init(
            fitter, jax.random.PRNGKey(0), _MapResult({"a": jnp.asarray(1.5)}), False
        )
        assert fitter.n_map_runs == 0
        assert float(params["a"]) == pytest.approx(1.5)
