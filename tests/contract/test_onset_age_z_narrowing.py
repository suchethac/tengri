# SPDX-License-Identifier: BSD-3-Clause
"""Redshift-aware onset-age priors: the parse-time z-narrowing pass.

``sfh_exp_start_gyr`` / ``sfh_dexp_start_gyr`` / ``sfh_const_start_gyr`` each
declare a static ``free_prior`` ceiling of today's cosmic age
(``_AGE_UNIV_GYR``, z=0) in ``components/stellar/sfh/registry.py`` -- the
widest value that is ever correct, mirroring the ``default=_AGE_UNIV_GYR``
convention already used on ``sfh_dpl_age_gyr`` and friends. That declaration
alone is not enough: a bound generous enough for z~0 admits SF-onset draws
at z=2 where star formation never happens (a zero-mass galaxy, zero flux,
zero gradient). ``parameters/groups.py``'s ``_narrow_free_priors_to_z``
closes that gap by narrowing the declared range to ``age_at_z(z)`` at parse
time, whenever the build's redshift floor is knowable.

Every expectation here is computed via ``tengri.utils.cosmology.age_at_z``,
never hardcoded -- these are pinned to the *mechanism*, not to today's
cosmological constants.
"""

from __future__ import annotations

import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Uniform
from tengri.utils.cosmology import age_at_z, age_at_z0

pytestmark = pytest.mark.contract


def _build_dexp(**overrides):
    """Parse a bare dexp SFH with 'all_params': FREE, no dust/nebular noise."""
    kwargs = {
        "sfh": {"type": "dexp", "all_params": FREE},
        "dust_attenuation": {"type": "none"},
        "neb": {"type": "none"},
    }
    kwargs.update(overrides)
    return tengri.parse_groups(**kwargs)


# ── (a) the wildcard frees the onset, capped at age_at_z(z) ────────────────


def test_dexp_onset_freed_with_z_aware_ceiling():
    """'all_params': FREE frees sfh_dexp_start_gyr, capped at age_at_z(0.5)."""
    spec = _build_dexp(redshift=Fixed(0.5))

    assert "sfh_dexp_start_gyr" in spec.free_params
    lo, hi = spec.get_distribution("sfh_dexp_start_gyr").bounds
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(float(age_at_z(0.5)), rel=1e-9)


# ── (b) monotonicity: a lower redshift leaves more cosmic time available ───


def test_cap_is_monotonic_in_redshift():
    hi_z05 = _build_dexp(redshift=Fixed(0.5)).get_distribution("sfh_dexp_start_gyr").bounds[1]
    hi_z005 = _build_dexp(redshift=Fixed(0.05)).get_distribution("sfh_dexp_start_gyr").bounds[1]

    assert hi_z005 > hi_z05
    assert hi_z005 == pytest.approx(float(age_at_z(0.05)), rel=1e-9)
    assert hi_z05 == pytest.approx(float(age_at_z(0.5)), rel=1e-9)


# ── (c) a free redshift narrows against its own FLOOR ───────────────────────


def test_free_redshift_caps_at_its_own_floor():
    """Uniform(0, 20) redshift narrows against z=0 -- i.e. no real narrowing.

    The floor of the redshift prior is what a build can guarantee about every
    galaxy this model will ever be asked to fit: at least as young as z=0.
    """
    spec = _build_dexp(redshift=Uniform(0.0, 20.0))

    hi = spec.get_distribution("sfh_dexp_start_gyr").bounds[1]
    declared_ceiling = round(float(age_at_z0()), 3)  # matches _AGE_UNIV_GYR's own rounding
    assert hi == pytest.approx(declared_ceiling, rel=1e-9)
    # Genuinely un-narrowed: the declared range already sat inside the cap,
    # so provenance carries no "_zcap" suffix.
    assert spec._group_provenance["sfh_dexp_start_gyr"] == "wildcard_free"


# ── (d) a user's own explicit prior is never touched ────────────────────────


def test_explicit_user_prior_is_never_narrowed():
    spec = _build_dexp(
        sfh={"type": "dexp", "start_gyr": Uniform(0.0, 5.0), "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.5),
    )

    lo, hi = spec.get_distribution("sfh_dexp_start_gyr").bounds
    assert (lo, hi) == (0.0, 5.0)
    assert spec._group_provenance["sfh_dexp_start_gyr"] == "user_prior"


# ── (e) round-trip: a z-narrowed spec still reads as 'all_params': FREE ────


def test_zcap_provenance_collapses_like_grid_provenance():
    """``_base_provenance`` treats '_zcap' exactly like '_grid': stripped.

    Isolated at the resolver level (mirrors how test_wildcard_frees_something.py
    tests ``_check_wildcard_freed_something`` directly) so this does not
    depend on the unrelated met_* collapsing limitation documented on
    ``parameters_to_groups`` (a bare 'sfh' wildcard never collapses cleanly
    because of it, independent of z-narrowing).
    """
    from tengri.parameters.groups import _analyze_wildcard_intent, _base_provenance

    assert _base_provenance("wildcard_free_zcap") == "wildcard_free"
    assert _base_provenance("user_free_zcap") == "user_free"

    provenance = {
        "sfh_dexp_log_total_mass": "wildcard_free",
        "sfh_dexp_start_gyr": "wildcard_free_zcap",
    }
    intent = _analyze_wildcard_intent(list(provenance), spec=None, provenance=provenance)
    assert intent is FREE


def test_z_narrowed_spec_round_trips_as_all_params_free():
    """End-to-end round-trip, with met split into its own group (met_mode != 'delta')

    so the pre-existing met_* mixing (see ``parameters_to_groups``'s docstring
    example) cannot confound the assertion: this isolates the z-narrowing
    suffix as the only thing that could prevent collapsing.
    """
    spec = _build_dexp(
        met={"type": "ramp", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.5),
    )
    assert spec._group_provenance["sfh_dexp_start_gyr"] == "wildcard_free_zcap"

    groups = spec.to_groups()
    sfh_group = groups["sfh"]
    assert sfh_group == {"type": "dexp", "all_params": FREE}

    roundtripped = tengri.parse_groups(**groups)
    assert roundtripped.free_params == spec.free_params
    lo, hi = roundtripped.get_distribution("sfh_dexp_start_gyr").bounds
    assert (lo, hi) == spec.get_distribution("sfh_dexp_start_gyr").bounds


# ── (f) const: positive floor (ordering constraint) + z-aware ceiling ──────


def test_const_onset_frees_with_positive_floor_and_cap():
    spec = tengri.parse_groups(
        sfh={"type": "const", "all_params": FREE},
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )

    lo, hi = spec.get_distribution("sfh_const_start_gyr").bounds
    assert lo == pytest.approx(0.01)
    assert hi == pytest.approx(float(age_at_z(0.5)), rel=1e-9)

    # Ordering constraint vs sfh_const_end_gyr (Fixed(0.0)) still holds --
    # Parameters.__init__'s _validate_orderings would have raised otherwise.
    end_lo = spec.get_distribution("sfh_const_end_gyr").bounds[0]
    assert lo > end_lo
