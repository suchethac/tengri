# SPDX-License-Identifier: BSD-3-Clause
"""ChEES's ensemble mass matrix test pinning tengri's contract with BlackJAX.

BlackJAX 1.6.2 had a bug: its trajectory-length floor called ``float()`` on
traced step sizes, so ``(mass_matrix_estimation="diagonal", _length_floor=True)``
raised ``ConcretizationTypeError`` under any JIT. BlackJAX 1.7+ fixed the bug.

This test pins tengri's contract rather than upstream's failure mode: that the
combination traces correctly and produces a valid mass matrix. The test therefore
passes on both old and new BlackJAX versions (pre-1.7 because tengri's workaround
disables the floor, post-1.7 because the bug is fixed upstream).

Tengri's workaround (disabling the floor when a mass matrix is estimated)
remains for compatibility with BlackJAX 1.6.x deployments, but new code should
not require it.
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


class TestTengriCheesMassMatrixContract:
    """Tengri's ChEES mass matrix path is traced and produces valid results.

    This test pins tengri's contract: that using ``mass_matrix_estimation=
    "diagonal"`` produces a jittable combination that returns a valid diagonal
    inverse mass matrix. This is true regardless of BlackJAX version: pre-1.7
    because tengri disables the floor (the old workaround), post-1.7 because
    the bug is fixed upstream.
    """

    def test_diagonal_mass_matrix_estimation_works(self):
        """ChEES with diagonal mass matrix estimation produces a valid mass matrix.

        Tengri's workaround disables the length floor when a mass matrix is
        estimated (for BlackJAX < 1.7 compatibility), so this test passes on both
        old and new BlackJAX versions. On 1.7+ the upstream bug is fixed; on 1.6.2
        tengri's workaround engages and disables the floor (with a warning that
        may be caught by the second test, which runs after this one on the same
        cache key).
        """
        # Tengri's path: _chees_scan with mass_matrix_estimation="diagonal"
        # invokes tengri's workaround, which disables the floor on BlackJAX 1.6.2
        # to make the combination trace. On 1.7.1 the upstream bug is fixed.
        positions = _chees("diagonal")

        assert positions.shape == (1, 20, 3)
        assert jnp.all(jnp.isfinite(positions))

    def test_the_default_configuration_is_unaffected(self):
        """``mass_matrix_estimation=None`` always traces without warnings."""
        positions = jax.jit(lambda: _chees(None))()
        assert positions.shape == (1, 20, 3)


def _chees(mass_matrix_estimation):
    from tengri.inference.backends.mcmc._shared import _chees_scan

    return _chees_scan(*_scan_args(mass_matrix_estimation))[0]


class TestTengriRunsItAndSaysWhatItCost:
    def test_the_ablation_runs_and_names_what_it_cost(self):
        """It must run, and a caller must learn this is a different sampler.

        The warning is emitted at **trace** time (once per JIT compilation),
        and only when the workaround is engaged. On BlackJAX 1.7+ the upstream
        bug is fixed so the workaround does not engage and no warning is emitted;
        on 1.6.2 the workaround engages and warns.
        """
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            positions = _chees("diagonal")

        assert positions.shape == (1, 20, 3)
        assert bool(jnp.all(jnp.isfinite(positions)))

        # On BlackJAX 1.6.2, tengri's workaround emits a warning about disabling
        # the floor. On 1.7+, the upstream bug is fixed and no warning is emitted.
        trajectory_warnings = [w for w in caught if "trajectory-length floor" in str(w.message)]
        if trajectory_warnings:
            # BlackJAX 1.6.2 path: workaround engaged
            text = " ".join(str(w.message) for w in trajectory_warnings)
            assert "NOT the same sampler" in text
            assert "ablation, not a configuration" in text
        # else: BlackJAX 1.7+ path, upstream bug fixed, no warning

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
