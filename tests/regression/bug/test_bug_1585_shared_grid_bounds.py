# SPDX-License-Identifier: BSD-3-Clause
"""Regression: SharedGrid.uniform must reject bounds it cannot represent (#1585).

``SharedGrid.uniform`` guarded the **upper** tau bound with care — a message
citing the 1e20 yr underflow threshold, the age of the universe, and a units-error
hint — and left every other bound completely unchecked. The grid is geometric in
tau and the quadrature runs in ``log(tau)``, so the unchecked bounds do not fail
loudly; they produce an intact-looking, entirely NaN grid::

    tau = jnp.geomspace(tau_bounds_yr[0], tau_bounds_yr[1], n_tau)  # NaN from a
    d_log_tau = (log(tau_bounds_yr[1]) - log(tau_bounds_yr[0])) / n_tau  # negative start

Measured on the four widths shipped in the ESS-vs-breadth sweep, whose bounds
scale symmetrically about the truth (``350 -/+ 245 * width`` Myr) and therefore
cross zero at ``width >= 1.43`` — two of the four are past it:

===== ======================= ============== ================ =======
width tau bounds [Myr]        tau nodes NaN  log_volume NaN   raised
===== ======================= ============== ================ =======
0.5   (227.5, 472.5)          0/5            0/25             --
1.0   (105.0, 595.0)          0/5            0/25             --
2.0   (-140.0, 840.0)         5/5            25/25            **no**
4.0   (-630.0, 1330.0)        5/5            25/25            **no**
===== ======================= ============== ================ =======

``log_prior`` stays finite (0/25 NaN) over that all-NaN grid: with the default
``tau_prior="log_uniform"`` its weights are ``zeros`` by construction, so a
caller spot-checking the prior sees a clean normalized array describing garbage.
A derived field cannot report corruption it does not depend on.

Both axes are positive by construction. ``sigma`` is a modulation amplitude in
dex and enters ``ou_logpdf`` only as ``var = (sigma * ln 10)**2``, which is
**even** — so a negative node is a bit-identical mirror of ``|sigma|`` and puts
posterior mass on unphysical support, while ``sigma = 0`` divides by that
variance and returns NaN outright. ``tau`` is a correlation timescale whose DRW
kernel ``exp(-|t_i - t_j| / tau)`` has no negative branch.

These tests build grids only — no SSP data, no fit — so they run in the PR gate
rather than the SSP-gated slow tier where the sweep that found this lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.inference.population import SharedGrid, ou_logpdf

pytestmark = pytest.mark.regression_bug

#: Bounds that are healthy on both axes — the control for every case below.
_GOOD = {"sigma_bounds": (0.1, 1.0), "tau_bounds_yr": (1.0e7, 5.0e8), "n_sigma": 5, "n_tau": 5}

#: Times spaced commensurately with the tau grid. At ``dt >> tau`` the lag-one
#: correlation ``exp(-dt/tau)`` underflows to exactly 0 and unrelated nodes
#: become bit-identical for reasons that have nothing to do with the bounds.
_TIMES_YR = np.linspace(0.0, 2.0e9, 12)
_FIELD = np.linspace(-0.4, 0.4, 12)


def _grid(**overrides):
    return SharedGrid.uniform(**{**_GOOD, "n_sigma": 5, "n_tau": 5, **overrides})


class TestTauBounds:
    """The axis whose upper end was guarded and whose lower end was not."""

    def test_a_negative_lower_bound_raises(self):
        """The sweep's own w=2.0 setting: 350 -/+ 490 Myr crosses zero."""
        with pytest.raises(ValueError, match="tau_bounds_yr"):
            _grid(tau_bounds_yr=(-1.4e8, 8.4e8))

    def test_a_zero_lower_bound_raises(self):
        """``log(0)`` is -inf and ``geomspace`` from 0 is degenerate. Zero is a
        NaN source, not merely an edge, so the guard is ``<= 0`` not ``< 0``."""
        with pytest.raises(ValueError, match="tau_bounds_yr"):
            _grid(tau_bounds_yr=(0.0, 5.0e8))

    def test_equal_bounds_raise(self):
        """``d_log_tau`` is 0, so every ``log_volume`` entry is ``log(0) = -inf``
        and the quadrature weight of every node vanishes."""
        with pytest.raises(ValueError, match="tau_bounds_yr"):
            _grid(tau_bounds_yr=(5.0e8, 5.0e8))

    def test_inverted_bounds_raise(self):
        """``d_log_tau`` is negative, so ``log_volume`` is NaN — the tau nodes
        themselves stay finite, which is exactly why this went unnoticed."""
        with pytest.raises(ValueError, match="tau_bounds_yr"):
            _grid(tau_bounds_yr=(5.0e8, 1.0e7))

    def test_the_upper_bound_guard_still_fires(self):
        """The one guard that existed. Do not regress it while adding siblings."""
        with pytest.raises(ValueError, match=r"1\.00e\+20|underflow"):
            _grid(tau_bounds_yr=(1.0e7, 1.0e21))


class TestSigmaBounds:
    """The axis that had no guard at either end."""

    @pytest.mark.parametrize("lower", [-0.39, -1.38, 0.0])
    def test_a_nonpositive_lower_bound_raises(self, lower):
        """-0.39 and -1.38 are the sweep's w=2.0 and w=4.0 lower bounds."""
        with pytest.raises(ValueError, match="sigma_bounds"):
            _grid(sigma_bounds=(lower, 1.59))

    def test_equal_bounds_raise(self):
        """``d_sigma`` is 0 and every ``log_volume`` entry is ``-inf``."""
        with pytest.raises(ValueError, match="sigma_bounds"):
            _grid(sigma_bounds=(0.5, 0.5))

    def test_inverted_bounds_raise(self):
        with pytest.raises(ValueError, match="sigma_bounds"):
            _grid(sigma_bounds=(1.0, 0.1))


class TestWhyTheBoundsMustBePositive:
    """Pin the reasons, so the guards cannot be relaxed without confronting them."""

    @pytest.mark.parametrize("sigma", [0.25, 0.6, 1.3])
    def test_the_density_is_even_in_sigma(self, sigma):
        """``var = (sigma ln10)**2`` is even, so a negative node is a bit-identical
        mirror of its positive twin. Quadrature there is not merely unphysical:
        it double-counts the amplitude axis."""
        mean = -0.5 * (sigma * np.log(10.0)) ** 2
        positive = float(ou_logpdf(_FIELD, mean, sigma, 2.0e8, _TIMES_YR))
        negative = float(ou_logpdf(_FIELD, mean, -sigma, 2.0e8, _TIMES_YR))
        assert positive == negative

    def test_zero_sigma_makes_the_density_nan(self):
        """The variance is the denominator; a tiny positive sigma stays finite."""
        assert not np.isfinite(float(ou_logpdf(_FIELD, 0.0, 0.0, 2.0e8, _TIMES_YR)))
        assert np.isfinite(float(ou_logpdf(_FIELD, 0.0, 1.0e-12, 2.0e8, _TIMES_YR)))


class TestItDoesNotOverreach:
    def test_healthy_bounds_build_a_finite_grid(self):
        """The control. Every array finite, on both axes."""
        grid = _grid()
        for name in ("sigma", "tau_yr", "log_prior", "log_volume"):
            assert np.all(np.isfinite(np.asarray(getattr(grid, name)))), name

    def test_a_very_small_positive_tau_is_allowed(self):
        """The guard is positivity, not a physical-plausibility opinion. Short
        timescales are legitimate; only the non-positive branch is rejected."""
        grid = _grid(tau_bounds_yr=(1.0e3, 1.0e5))
        assert np.all(np.isfinite(np.asarray(grid.tau_yr)))
        assert np.all(np.isfinite(np.asarray(grid.log_volume)))

    def test_the_uniform_tau_prior_still_builds(self):
        """The non-default prior branch shares the bounds path."""
        grid = _grid(tau_prior="uniform")
        assert np.all(np.isfinite(np.asarray(grid.log_prior)))

    def test_the_message_names_the_axis_and_the_failure_mode(self):
        """An error that says only 'invalid bounds' sends the reader back to the
        source. #1575 shipped advice that could not reach its own mode; the
        lesson was that the message must name the remedy."""
        with pytest.raises(ValueError) as excinfo:
            _grid(tau_bounds_yr=(-1.4e8, 8.4e8))
        message = str(excinfo.value)
        assert "-1.4" in message or "-1.40" in message
        assert "NaN" in message
        assert "positive" in message
