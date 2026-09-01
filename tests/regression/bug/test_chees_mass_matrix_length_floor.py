# SPDX-License-Identifier: BSD-3-Clause
"""ChEES's ensemble mass matrix cannot be traced with BlackJAX's length floor on.

``run_chees``'s docstring exposes ``mass_matrix_estimation="diagonal"`` "so the
ablation is re-runnable from a call rather than an edit". It was not: BlackJAX
1.6.2 enables its trajectory-length floor **exactly when** a mass matrix is being
estimated, and that branch calls ``float(step_size_ma)`` on a traced array
(``blackjax/adaptation/chees_adaptation.py``, in ``run``). The pair therefore
raises ``ConcretizationTypeError`` under *any* ``jit`` -- a single fit as much as
a catalog ``vmap``, and independently of tengri, since the failure reproduces on
a bare Gaussian with no tengri model in the trace.

Every tengri ChEES entry point is jitted (``_chees_scan`` carries the
``jax.jit``), so the option was unreachable in practice. ``_chees_scan`` now
turns the floor off whenever a mass matrix is estimated, and warns, because
disabling half of an algorithm silently is worse than a slow ablation.

These tests pin all three halves: that the upstream combination really is
untraceable (so the workaround is not cargo-cult), that tengri's path runs, and
that it says so.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def _gaussian(pos, data_args):
    return -0.5 * jnp.sum(pos**2) + 0.0 * data_args


def _scan_args(mass_matrix_estimation):
    """Arguments for a minimal 3-D ``_chees_scan`` call."""
    return (
        jnp.zeros(3),  # init_flat
        jax.random.PRNGKey(0),  # warmup_key
        jax.random.split(jax.random.PRNGKey(1), 20).reshape(1, 20, 2),
        _gaussian,
        jnp.asarray(0.0),  # data_args
        20,  # n_warmup
        4,  # n_ensemble
        1,  # n_chains
        20,  # n_iter
        0.1,  # jitter_scale
        1.0,  # jitter_amount
        0.651,  # target_accept_rate
        16,  # max_leapfrog_steps
        0.05,  # learning_rate
        mass_matrix_estimation,
        None,  # chain_jitter
    )


class TestTheUpstreamCombinationIsUntraceable:
    """Pinned against BlackJAX directly, so a fixed release is *noticed*.

    If BlackJAX repairs the ``float()`` call this test starts failing, which is
    the signal to delete tengri's workaround rather than carry it forever.
    """

    def test_diagonal_mass_with_the_length_floor_cannot_be_jitted(self):
        import optax
        from blackjax import chees_adaptation
        from blackjax.adaptation.base import get_filter_adapt_info_fn

        def run(mass, floor):
            warmup = chees_adaptation(
                lambda p: -0.5 * jnp.sum(p**2),
                num_chains=4,
                jitter_amount=1.0,
                target_acceptance_rate=0.651,
                max_leapfrog_steps=16,
                adaptation_info_fn=get_filter_adapt_info_fn(),
                mass_matrix_estimation=mass,
                _length_floor=floor,
            )
            ensemble = jax.random.normal(jax.random.PRNGKey(0), (4, 3))
            (_states, params), _ = warmup.run(
                jax.random.PRNGKey(1),
                ensemble,
                0.1,
                optax.adam(0.05),
                num_steps=20,
                max_sampling_steps=20,
            )
            return params["step_size"]

        with pytest.raises(jax.errors.ConcretizationTypeError):
            jax.jit(lambda: run("diagonal", True))()

        # And the floor is the whole difference -- same call, floor off, traces.
        assert jnp.isfinite(jax.jit(lambda: run("diagonal", False))())

    def test_the_default_configuration_is_unaffected(self):
        """``mass_matrix_estimation=None`` never enables the floor, so it traces."""
        positions = jax.jit(lambda: _chees(None))()
        assert positions.shape == (1, 20, 3)


def _chees(mass_matrix_estimation):
    from tengri.inference.backends.mcmc._shared import _chees_scan

    return _chees_scan(*_scan_args(mass_matrix_estimation))[0]


class TestTengriRunsItAndSaysWhatItCost:
    def test_the_ablation_runs_and_names_what_it_cost(self):
        """It must run, and a caller must learn this is a different sampler.

        Asserted in one block rather than two because the warning is emitted at
        **trace** time -- see the next test -- so a second call would not repeat
        it and a second ``pytest.warns`` would fail for the wrong reason.
        """
        with pytest.warns(UserWarning, match="trajectory-length floor") as caught:
            positions = _chees("diagonal")

        assert positions.shape == (1, 20, 3)
        assert bool(jnp.all(jnp.isfinite(positions)))
        text = " ".join(str(w.message) for w in caught)
        assert "NOT the same sampler" in text
        assert "ablation, not a configuration" in text

    def test_the_warning_fires_once_per_compilation_not_once_per_call(self):
        """``_chees_scan`` is jitted, so the Python body runs only on a trace.

        Recorded because it is a real limit on the warning's reach rather than a
        defect to fix: a caller who runs the ablation in a loop sees it once, and
        a caller whose program was already traced may not see it at all. The
        docstring in ``run_chees`` is what carries the caveat for them.
        """
        import warnings as _w

        _chees("diagonal")  # first call: traces, warns
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            _chees("diagonal")  # second call: cache hit, silent
        assert not [w for w in caught if "trajectory-length floor" in str(w.message)]

    def test_the_default_does_not_warn(self):
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            _chees(None)
        assert not [w for w in caught if "trajectory-length floor" in str(w.message)]
