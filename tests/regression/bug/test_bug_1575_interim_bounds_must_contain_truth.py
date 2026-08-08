# SPDX-License-Identifier: BSD-3-Clause
"""Regression: fit_interim must reject a truth outside its own bounds (#1575).

``make_population`` validates each injected truth against the model's prior
(``assert_truth_against_model``). ``fit_interim`` then **overrides** the shared
PSD priors with ``interim_bounds`` — and nothing re-validated the truth against
the bounds actually used.

The N=8 PSD pilot did exactly that: mocks generated at ``tau_true_myr = 350``,
fitted with ``tau_bounds_myr = (50.0, 200.0)``. The truth sits 1.75x outside the
support the fit is allowed to reach, so the optimizer walks to the boundary —
which is at infinity in the unbounded parameterization — and every one of 8 MAP
restarts returns a non-finite loss::

    ValueError: all 8 MAP restarts diverged to a non-finite loss (NaN or inf)

That message is generically correct and locally useless: its advice (smaller
``learning_rate``, more ``n_restarts``) cannot reach a mode that is not in the
support. It also arrives **50 minutes** into the run, after 8 HMC fits.

Two guards existed and neither covered this. The truths were checked against the
*nominal* bounds (10, 500) — where 350 passes — and the interim bounds were then
narrowed to (50, 200) without re-checking. One axis guarded against one set of
bounds while the fit used another.

It stayed invisible for a second reason: before #1562, the MAP cache was keyed on
the model, so only galaxy 0 ever ran a MAP and galaxies 1-7 reused its cached
point. **Seven of eight galaxies were never fit at all**, so the divergence had
nowhere to surface.

These tests use a stub mock rather than a real model, so they run in the PR gate
rather than the SSP-gated slow tier where the original failure hid.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from tengri.inference.population.interim import _assert_truth_within_interim_bounds

pytestmark = pytest.mark.regression_bug


class _Mock:
    """Only the surface the guard reads: per-galaxy injected truths."""

    def __init__(self, sigma=0.6, tau_myr=350.0, n=3):
        self.truth_params = [
            {"sfh_field_psd_sigma": sigma, "sfh_field_psd_tau_myr": tau_myr} for _ in range(n)
        ]


_PILOT_BOUNDS = {"sigma_bounds": (0.5, 1.5), "tau_bounds_myr": (50.0, 200.0)}


class TestTheTruthMustBeReachable:
    def test_a_tau_truth_outside_the_bounds_raises(self):
        """LOAD-BEARING. Neuter: drop the tau branch.

        Without it the pilot's tau=350 against (50, 200) runs for 50 minutes and
        then reports 8 diverged MAP restarts, which names neither tau nor the
        bounds.
        """
        with pytest.raises(ValueError) as excinfo:
            _assert_truth_within_interim_bounds(_Mock(tau_myr=350.0), _PILOT_BOUNDS)
        message = str(excinfo.value)
        assert "350" in message, "the offending truth is not quoted"
        assert "200" in message, "the bound it violates is not quoted"
        assert "tau" in message.lower()

    def test_a_sigma_truth_outside_the_bounds_raises(self):
        """The same axis guard, on the parameter that is actually identified."""
        with pytest.raises(ValueError) as excinfo:
            _assert_truth_within_interim_bounds(
                _Mock(sigma=2.0), {"sigma_bounds": (0.5, 1.5), "tau_bounds_myr": (50.0, 500.0)}
            )
        assert "sigma" in str(excinfo.value).lower()

    def test_the_message_says_what_to_change(self):
        """A guard that names the failure but not the remedy just relocates the
        confusion. Either the truth moves or the bounds widen; say so."""
        with pytest.raises(ValueError) as excinfo:
            _assert_truth_within_interim_bounds(_Mock(tau_myr=350.0), _PILOT_BOUNDS)
        message = str(excinfo.value).lower()
        assert "interim_bounds" in message
        assert "widen" in message or "outside" in message

    def test_a_truth_inside_the_bounds_passes(self):
        """The ordinary path stays silent, or every legitimate fit raises."""
        _assert_truth_within_interim_bounds(_Mock(sigma=0.6, tau_myr=150.0), _PILOT_BOUNDS)

    def test_the_bounds_are_inclusive_at_the_edge(self):
        """A truth exactly on a bound is inside the support, not outside it.
        Raising there would reject a legitimate edge-case fixture."""
        _assert_truth_within_interim_bounds(_Mock(sigma=0.5, tau_myr=200.0), _PILOT_BOUNDS)


class TestItDoesNotOverreach:
    def test_a_mock_without_truths_is_not_an_error(self):
        """Real data has no injected truth. The guard must skip, not raise —
        otherwise it breaks the only case that matters scientifically."""

        class _NoTruths:
            truth_params: ClassVar[list] = []

        _assert_truth_within_interim_bounds(_NoTruths(), _PILOT_BOUNDS)

    def test_a_mock_missing_the_psd_keys_is_not_an_error(self):
        """A population whose truths omit the PSD parameters is out of scope for
        this check, not a failure of it."""

        class _OtherTruths:
            truth_params: ClassVar[list] = [{"met_logzsol": -0.3}]

        _assert_truth_within_interim_bounds(_OtherTruths(), _PILOT_BOUNDS)

    def test_it_checks_every_galaxy_not_only_the_first(self):
        """Truths are per-galaxy. Checking galaxy 0 alone is the same class of
        mistake as fitting galaxy 0 and reusing its MAP for the rest (#1529)."""
        mock = _Mock(tau_myr=150.0, n=3)
        mock.truth_params[2]["sfh_field_psd_tau_myr"] = 350.0
        with pytest.raises(ValueError) as excinfo:
            _assert_truth_within_interim_bounds(mock, _PILOT_BOUNDS)
        assert "350" in str(excinfo.value)

    def test_an_array_valued_truth_is_handled(self):
        """``truth_params`` values arrive as 0-d arrays from the sampler."""
        mock = _Mock()
        mock.truth_params[0]["sfh_field_psd_tau_myr"] = np.asarray(350.0)
        with pytest.raises(ValueError):
            _assert_truth_within_interim_bounds(mock, _PILOT_BOUNDS)
