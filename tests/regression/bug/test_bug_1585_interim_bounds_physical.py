# SPDX-License-Identifier: BSD-3-Clause
"""Regression: fit_interim must reject unphysical interim bounds (#1585 sibling).

#1585 guarded ``SharedGrid.uniform``, the *reweighting* grid. ``fit_interim``
builds the *interim priors* from the same ``interim_bounds`` dict and was not
guarded — so the expensive HMC ran first, on a prior admitting unphysical
support, and only the cheap grid construction afterwards complained.

``Uniform`` already rejects ``lo >= hi``, so the degenerate and inverted cases
were covered. Non-positive lower bounds were not. Measured on ``ou_logpdf``
with a 12-node field:

====================== ============================ ==========================
bound                  ou_logpdf                    consequence
====================== ============================ ==========================
``tau = 2e8 yr``       -15.402943475118297          healthy reference
``tau = -1e6 yr``      **nan**                      undefined; HMC diverges
``tau = 0 yr``         -17.966906776193017          **finite but degenerate**
``sigma = +0.6``       -15.402943475118297          healthy reference
``sigma = -0.6``       -15.402943475118297          **bit-identical mirror**
``sigma = 0``          **nan**                      variance is the denominator
====================== ============================ ==========================

Two of those fail quietly rather than loudly, which is why this needed a guard
rather than a NaN check:

* ``tau = 0`` does **not** NaN. ``rho = exp(-dt/0) = 0``, so the field
  degenerates to independent draws — a real, silently-wrong model, not an error.
* ``sigma < 0`` returns a density **bit-identical** to ``|sigma|`` because sigma
  reaches the kernel only through ``var = (sigma ln10)**2``, which is even. The
  sampler explores a spurious mirror mode and the marginal in sigma is corrupt.

The ESS-vs-breadth sweep reaches both: its bounds scale symmetrically about the
truth and cross zero at ``width >= 1.21`` (sigma) and ``>= 1.43`` (tau).

These tests pass ``model=None``. That is the assertion, not a shortcut: the
check must fire before anything expensive is touched. #1575 shipped the opposite
— a bounds problem that surfaced 50 minutes into a run, after 8 HMC fits.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from tengri.inference.population.interim import fit_interim

pytestmark = pytest.mark.regression_bug

_GOOD = {"sigma_bounds": (0.01, 1.0), "tau_bounds_myr": (10.0, 500.0)}


class _Mock:
    """Population stub with no injected truths, so only the bounds check runs."""

    truth_params: ClassVar[list] = []
    table: ClassVar[list] = []


def _fit(**bounds_overrides):
    return fit_interim(
        None,
        _Mock(),
        key=None,
        interim_bounds={**_GOOD, **bounds_overrides},
    )


class TestTauBounds:
    @pytest.mark.parametrize("lower", [-140.0, -630.0, 0.0])
    def test_a_nonpositive_lower_bound_raises(self, lower):
        """-140 and -630 Myr are the sweep's own w=2.0 and w=4.0 lower bounds."""
        with pytest.raises(ValueError, match="tau_bounds_myr"):
            _fit(tau_bounds_myr=(lower, 840.0))

    def test_the_message_explains_the_degeneracy_not_just_the_sign(self):
        """tau=0 does not NaN — it silently removes the correlation. A message
        that only says 'must be positive' leaves the reader to discover that."""
        with pytest.raises(ValueError) as excinfo:
            _fit(tau_bounds_myr=(0.0, 840.0))
        assert "correlation" in str(excinfo.value).lower()


class TestSigmaBounds:
    @pytest.mark.parametrize("lower", [-0.39, -1.38, 0.0])
    def test_a_nonpositive_lower_bound_raises(self, lower):
        """-0.39 and -1.38 dex are the sweep's w=2.0 and w=4.0 lower bounds."""
        with pytest.raises(ValueError, match="sigma_bounds"):
            _fit(sigma_bounds=(lower, 1.59))

    def test_the_message_explains_the_mirror(self):
        """A negative sigma is not merely unphysical — it is indistinguishable
        from its positive twin, so the sampler cannot tell the modes apart."""
        with pytest.raises(ValueError) as excinfo:
            _fit(sigma_bounds=(-0.39, 1.59))
        assert "mirror" in str(excinfo.value).lower()


class TestItFailsBeforeAnythingExpensive:
    def test_it_raises_without_touching_the_model(self):
        """``model=None`` would AttributeError the moment the function reached
        ``model.log_age_grid``. Reaching ValueError instead proves the check
        runs first — the #1575 lesson, where the equivalent failure cost 50 min
        of HMC before surfacing."""
        with pytest.raises(ValueError):
            _fit(tau_bounds_myr=(-140.0, 840.0))


class TestItDoesNotOverreach:
    def test_healthy_bounds_pass_the_check(self):
        """Must get PAST validation and fail later on ``model=None``, proving
        the guard did not simply reject everything."""
        with pytest.raises(AttributeError):
            _fit()

    def test_a_narrow_positive_range_is_allowed(self):
        """The guard is positivity, not an opinion about how tight a prior may
        be. ``Uniform`` already rejects ``lo >= hi``; this must not add more."""
        with pytest.raises(AttributeError):
            _fit(sigma_bounds=(0.59, 0.61), tau_bounds_myr=(349.0, 351.0))
