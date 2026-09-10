# SPDX-License-Identifier: BSD-3-Clause
"""``shock_log_density`` frees under the wildcard at its measured populated
range; ``shock_b_over_sqrt_n`` stays pinned (#2065/#2066).

Companion to ``tests/regression/bug/test_bug_2065_shock_silent_zero.py``
(the build-time coverage guard the free_prior's honesty depends on) and to
``docs/internal/specs/2026-09-05-shock-family-interp-diagnosis.md`` (why B stays
inert: case (c), a 2-D-coupled sparse-grid kernel limitation, not fixed here).
"""

from __future__ import annotations

import warnings

import pytest

import tengri
from tengri import FREE, Fixed, Uniform
from tengri.config.exceptions import WildcardPartialFreeWarning


def _free_shock(**shock_kwargs):
    """``shock={..., 'all_params': FREE}`` on an otherwise-minimal spec."""
    from tengri.config.exceptions import DefaultFixedParametersWarning

    with warnings.catch_warnings():
        # SFH's own "no all_params disposition" advisory, unrelated to shock;
        # filtered by exact category so the shock WildcardPartialFreeWarning
        # under test still reaches an enclosing ``pytest.warns``.
        warnings.filterwarnings("ignore", category=DefaultFixedParametersWarning)
        return tengri.parse_groups(
            sfh={"type": "dpl"},
            shock={"norm": "frac", "all_params": FREE, **shock_kwargs},
            redshift=Fixed(0.1),
        )


def test_shock_log_density_frees_at_the_measured_declared_interval():
    """``all_params: FREE`` opens ``shock_log_density`` at exactly the
    interval declared on ``SHOCK_PARAMS`` -- the measured solar-abundance
    populated envelope, [-2, 3] log10(cm^-3) (see ``_params.py`` for the
    per-abundance table and the measurement method)."""
    spec = _free_shock()
    assert "shock_log_density" in spec.free_params
    assert spec.get_distribution("shock_log_density") == Uniform(-2.0, 3.0)


def test_shock_b_over_sqrt_n_stays_pinned_under_the_same_wildcard():
    """The B axis has no declared free_prior (case (c), #2066): it must stay
    ``Fixed`` at its registry default under the same wildcard call that frees
    density -- this is a partial free, not a failure to free anything."""
    with pytest.warns(WildcardPartialFreeWarning, match=r"shock_b_over_sqrt_n"):
        spec = _free_shock()
    assert "shock_b_over_sqrt_n" not in spec.free_params
    dist = spec.get_distribution("shock_b_over_sqrt_n")
    assert dist.is_fixed
    assert float(dist.value) == pytest.approx(1.0)


def test_shock_group_free_composition_is_exactly_density_and_not_b():
    """The wildcard's partial-free composition, in one place: density frees,
    B does not. A prior version of this contract pinned BOTH -- this is the
    updated composition after #2065 gave density a measured free_prior."""
    with pytest.warns(WildcardPartialFreeWarning):
        spec = _free_shock()
    assert "shock_log_density" in spec.free_params
    assert "shock_b_over_sqrt_n" not in spec.free_params
