# SPDX-License-Identifier: BSD-3-Clause
"""Tree-depth diagnostics and the shared max_num_doublings default.

NUTS at ``max_num_doublings=10`` saturated its trajectory cap on 46% of
iterations on the heavy-tailed StudentT SFR-ratio geometry of the
nonparametric SFH priors (19-band continuity fit, D=9, 2026-08-18) — 118 s
of wall dominated by 1023-gradient trajectories. Lowering the default was
measured and *rejected*: cap 6 cut the wall to 11 s but collapsed min-ESS
93 → 5, strictly worse per effective sample. ``dense_mass_matrix=True`` was
the original recommendation and is no longer: re-measured on the same fit it
buys wall time (35 s against 69 s) with 23 divergences per 400 draws against
6 for the diagonal, and the saturation itself is largely a model defect, since
bin edges past the age of the universe leave prior-only bins (#1975). What
ships instead:

1. the pure stats helper that turns per-iteration trajectory-expansion
   counts into ``posterior.diagnostics`` entries — depth saturation was
   previously invisible,
2. the saturation warning gate (deep caps only — a deliberately low cap
   is a wall-time bound working, not a pathology), and
3. one shared default across every NUTS entry point, so the single-fit
   path, the catalog engine, the batched-vmap path, and the prewarm
   compile (which hardcoded its own ``10``) cannot drift.
"""

import inspect
import warnings

import jax.numpy as jnp
import pytest

from tengri.config.exceptions import NUTSTreeDepthWarning
from tengri.inference.backends.mcmc._shared import DEFAULT_MAX_NUM_DOUBLINGS
from tengri.inference.backends.mcmc.nuts import (
    _SATURATION_WARN_MIN_CAP,
    _tree_depth_stats,
    _warn_if_tree_depth_saturated,
)


class TestTreeDepthStats:
    def test_stats_values_hand_computed(self):
        # 4 iterations: depths 2, 6, 6, 6 at a cap of 6 -> 3/4 saturated.
        expansions = jnp.array([2, 6, 6, 6])
        stats = _tree_depth_stats(expansions, max_num_doublings=6)
        assert stats["max_num_doublings"] == 6
        assert stats["tree_depth_max"] == 6
        assert stats["tree_depth_mean"] == pytest.approx(5.0)
        assert stats["frac_max_depth"] == pytest.approx(0.75)

    def test_no_saturation(self):
        stats = _tree_depth_stats(jnp.array([1, 2, 3]), max_num_doublings=10)
        assert stats["frac_max_depth"] == 0.0
        assert stats["tree_depth_max"] == 3

    def test_values_are_python_scalars(self):
        # Diagnostics dicts must survive pickling / repr without JAX arrays.
        stats = _tree_depth_stats(jnp.array([4, 4]), max_num_doublings=4)
        assert isinstance(stats["tree_depth_mean"], float)
        assert isinstance(stats["frac_max_depth"], float)
        assert isinstance(stats["tree_depth_max"], int)
        assert isinstance(stats["max_num_doublings"], int)


class TestSaturationWarning:
    def _stats(self, frac, doublings):
        return {
            "max_num_doublings": doublings,
            "tree_depth_mean": float(doublings),
            "tree_depth_max": doublings,
            "frac_max_depth": frac,
        }

    def test_warns_on_saturated_deep_cap(self):
        with pytest.warns(NUTSTreeDepthWarning, match="max_num_doublings"):
            _warn_if_tree_depth_saturated(self._stats(0.5, 10))

    def test_silent_below_saturation_threshold(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_tree_depth_saturated(self._stats(0.1, 10))

    def test_silent_at_low_cap(self):
        # A low cap that saturates is a deliberate wall-time bound doing its
        # job (at a known ESS cost), not a pathology to nag about.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_tree_depth_saturated(self._stats(1.0, _SATURATION_WARN_MIN_CAP - 1))


class TestDefaultConsistency:
    def test_run_nuts_default(self):
        from tengri.inference.backends.mcmc.nuts import run_nuts

        default = inspect.signature(run_nuts).parameters["max_num_doublings"].default
        assert default == DEFAULT_MAX_NUM_DOUBLINGS

    def test_catalog_engine_default(self):
        from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

        default = (
            inspect.signature(build_catalog_mcmc_engine).parameters["max_num_doublings"].default
        )
        assert default == DEFAULT_MAX_NUM_DOUBLINGS

    def test_the_callers_cap_reaches_the_runner(self, monkeypatch):
        """A caller's ``max_num_doublings`` arrives at the NUTS runner unchanged.

        Replaces a scan of ``inference/fitter`` for the literal text
        ``kwargs.get("max_num_doublings", DEFAULT_MAX_NUM_DOUBLINGS)``. That scan
        could not tell forwarding from a hardcoded default: an expression matching
        it in a branch that never runs satisfies the grep, and the value the
        runner actually receives was never looked at.

        Two things make the recorder work, both non-obvious enough to record:

        1. ``BackendEntry`` captures its ``runner`` **by value at registration**,
           so patching ``nuts.run_nuts`` does nothing -- dispatch never looks the
           name up again -- and the entry is a frozen dataclass, so the handle is
           ``dataclasses.replace`` into the registry dict.
        2. ``Fitter.run`` *calls* some attributes it reads, so the stub's fallback
           must be callable as well as falsy; a plain ``None`` fallback dies with
           ``TypeError`` before the runner is reached.

        Scope, stated because it is narrower than the name suggests: this pins the
        forwarding of an *explicitly passed* cap. It does not pin the omitted case
        -- measured on this stub the runner then receives ``None`` rather than
        ``DEFAULT_MAX_NUM_DOUBLINGS``, stable across stub variations, which may be
        a real path difference or an artifact of the synthetic fitter. Asserting
        either reading would be guessing. The constant itself is already pinned as
        each runner's own signature default by the two tests above.
        """
        import dataclasses

        from tengri.inference._backend_registry import _BACKENDS
        from tengri.inference.fitter import Fitter

        class _Reached(Exception):
            """Private marker: raised by the stub runner once it has recorded."""

        class _Nothing:
            """Callable and falsy, so it is harmless in call and branch positions."""

            def __call__(self, *args, **kwargs):
                return None

            def __bool__(self):
                return False

        class _Spec:
            stochastic = False
            free_params = ("a",)
            all_params = ("a",)
            n_grid = 8
            n_free = 1

            def __getattr__(self, name):
                return None

        class _StubFitter:
            spec = _Spec()
            _lut_bias_checked = True

            def __getattr__(self, name):
                return _Nothing()

        received = {}

        def _record(*args, **kwargs):
            received.update(kwargs)
            raise _Reached

        entry = _BACKENDS["mcmc_nuts"]
        monkeypatch.setitem(_BACKENDS, "mcmc_nuts", dataclasses.replace(entry, runner=_record))

        sentinel_cap = DEFAULT_MAX_NUM_DOUBLINGS + 3
        with pytest.raises(_Reached):
            Fitter.run(_StubFitter(), "mcmc_nuts", max_num_doublings=sentinel_cap)

        assert "max_num_doublings" in received, (
            "the cap never reached the NUTS runner: Fitter accepted the argument "
            "and the sampler it dispatches to never saw it"
        )
        assert received["max_num_doublings"] == sentinel_cap, (
            f"the caller asked for max_num_doublings={sentinel_cap} and the runner "
            f"received {received['max_num_doublings']!r} -- a hardcoded default has "
            f"replaced the forwarded value"
        )
