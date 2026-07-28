# SPDX-License-Identifier: BSD-3-Clause
"""Regression: a chain that never moved must not report as converged (#1438, #1437).

Split-R-hat compares within- and between-chain variance. Both are zero for a
chain that never moved, so a **frozen** chain scores ``R-hat ~ 1.0`` -- the best
possible value. The one honest signal, the divergence count, sits beside it and
reads as a footnote.

``Posterior.rhat()`` additionally drops zero-variance parameters by design, so a
*fully* frozen chain returns an **empty dict**. The documented idiom --
``max(float(v) for v in rhat().values())``, used by the spine notebooks and by
``rhat``'s own docstring example -- then raises::

    ValueError: max() iterable argument is empty

which is how this surfaced: as a crash in ``notebooks/06_fitting_spectroscopy``
whose message says nothing about sampling.

Measured cause (fixed-length HMC on that notebook's model, one fresh process per
arm): the posterior has a sharp step-size cliff, and whichever arm adapts above
it freezes.

    precondition  warmup   step     divergences  unique draws
    off             300    0.041      0            568
    off            1000    0.063    544              1
    on              300    0.125    600              1
    on             1000    0.047      0            572

Landing above the cliff is a legitimate outcome of adaptation. Reporting it as
convergence is not. These tests pin the reporting, which is the part that made a
dead fit look alive.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.regression_bug


def _frozen(n_draw: int = 200, names=("a", "b", "c")) -> Posterior:
    """A posterior whose every draw is identical -- the frozen-chain shape."""
    return Posterior(
        samples={n: np.full(n_draw, 0.3) for n in names},
        params={n: 0.3 for n in names},
        method="mcmc_hmc",
        wall_time_s=1.0,
        diagnostics={"n_divergent": n_draw, "n_samples": n_draw},
    )


def _moving(n_draw: int = 200) -> Posterior:
    rng = np.random.default_rng(0)
    return Posterior(
        samples={"a": rng.normal(size=n_draw), "b": rng.normal(size=n_draw)},
        params={"a": 0.0, "b": 0.0},
        method="mcmc_hmc",
        wall_time_s=1.0,
        diagnostics={"n_divergent": 0, "n_samples": n_draw},
    )


class TestRhatRefusesToBeSilentlyEmpty:
    def test_a_frozen_chain_raises_instead_of_returning_an_empty_dict(self):
        """LOAD-BEARING. Neuter: drop the all-static check from ``rhat``.

        Without it this returns ``{}`` and the caller's ``max()`` raises a
        message about an empty iterable, which names neither the chain nor the
        sampler.
        """
        with pytest.raises(ValueError) as excinfo:
            _frozen().rhat()
        message = str(excinfo.value).lower()
        assert "did not move" in message or "frozen" in message
        assert "max() iterable" not in message

    def test_the_error_names_the_draw_count_that_explains_it(self):
        """A frozen chain is nearly always all-divergent; say so, so the reader
        has somewhere to go rather than just being told the fit is dead."""
        with pytest.raises(ValueError) as excinfo:
            _frozen(150).rhat()
        assert "150" in str(excinfo.value), "divergence/draw count not surfaced"

    def test_a_moving_chain_is_unaffected(self):
        """The ordinary path must not regress."""
        rh = _moving().rhat()
        assert set(rh) == {"a", "b"}
        assert all(np.isfinite(v) for v in rh.values())

    def test_a_partially_static_chain_still_reports_the_moving_parameters(self):
        """Only a *fully* frozen chain is a dead fit.

        A single pinned parameter alongside moving ones is ordinary -- dropping
        it and reporting the rest is the documented behavior and must stay.
        """
        rng = np.random.default_rng(1)
        post = Posterior(
            samples={"moves": rng.normal(size=200), "pinned": np.full(200, 1.0)},
            params={"moves": 0.0, "pinned": 1.0},
            method="mcmc_hmc",
            wall_time_s=1.0,
            diagnostics={"n_divergent": 0},
        )
        rh = post.rhat()
        assert "moves" in rh
        assert "pinned" not in rh

    def test_the_documented_idiom_now_fails_with_a_useful_message(self):
        """``max(rhat().values())`` is what the notebooks and the docstring do.

        It must fail saying the chain never moved, not that a builtin got an
        empty argument.
        """
        with pytest.raises(ValueError) as excinfo:
            max(float(v) for v in _frozen().rhat().values())
        assert "did not move" in str(excinfo.value).lower()
