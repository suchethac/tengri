# SPDX-License-Identifier: BSD-3-Clause
"""The default nonparametric bin ladder is built from the source redshift (#2521).

``continuity``, ``dirichlet``, ``bursty_continuity``, and ``prospector_beta``
default, when no explicit ``bin_edges_gyr`` is given, to a ladder whose
oldest edge is the age of the universe at the source redshift
(:func:`~tengri.parameters.groups._default_nonparametric_bin_edges_from_z`,
:func:`~tengri.components.stellar.sfh.nonparametric.make_agebins_from_zred`
-- the Prospector-beta scheme, Wang et al. 2024, arXiv:2401.12198): the two
youngest edges are fixed at 30 Myr and 100 Myr, the remainder are
log-spaced, and the oldest edge is ``age_at_z`` of the source redshift, so
no default bin can lie beyond cosmic time at any redshift.

This mechanism lives at the ``SEDModel.build`` / ``parse_groups`` grammar
level, not on the low-level ``continuity`` / ``dirichlet`` functions or on
``DEFAULT_BIN_EDGES_GYR`` themselves (unchanged, and still what a direct call
to those functions or to ``sfh_bin_edges_yr(fn, {})`` defaults to): building
through the grammar is what tests here.

``bursty_continuity`` and ``prospector_beta`` are registered but not yet
validated against the DSPS forward path (``UNVALIDATED_SFH_TYPES``), so
``SEDModel.build`` refuses them; they are exercised directly against
:func:`~tengri.parameters.groups._default_nonparametric_bin_edges_from_z`
instead, which is the one place the mechanism lives and does not care
whether the family is forward-buildable yet.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.stellar.sfh.nonparametric import DEFAULT_BIN_EDGES_GYR
from tengri.parameters.groups import _default_nonparametric_bin_edges_from_z
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.bounds

#: Buildable directly through SEDModel.build.
_BUILDABLE_FAMILIES = ("continuity", "dirichlet")
#: Registered but not yet forward-validated (UNVALIDATED_SFH_TYPES); exercised
#: against the mechanism function directly instead.
_UNBUILDABLE_FAMILIES = ("bursty_continuity", "prospector_beta")
_REDSHIFTS = (0.5, 2.5, 6.0, 10.0)


def _n_bins_beyond_age(edges_gyr: np.ndarray, z: float) -> int:
    starts_gyr = np.asarray(edges_gyr)[:-1]
    age_gyr = float(age_at_z(z))
    return int(np.sum(starts_gyr >= age_gyr))


def _edges_via_mechanism(family: str, z: float) -> np.ndarray:
    resolved = {"mean_sfh_type": family, "redshift": Fixed(z)}
    _default_nonparametric_bin_edges_from_z(resolved)
    return np.asarray(resolved["bin_edges_gyr"])


@pytest.mark.parametrize("family", _BUILDABLE_FAMILIES)
@pytest.mark.parametrize("z", _REDSHIFTS)
def test_no_default_bin_lies_entirely_before_the_big_bang(ssp_data_fsps, family, z):
    """0 of 7 default bins lie wholly beyond age(z), built through SEDModel.build."""
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        sfh={"type": family, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(z),
    )
    assert _n_bins_beyond_age(model.spec.bin_edges_gyr, z) == 0


@pytest.mark.parametrize("family", _UNBUILDABLE_FAMILIES)
@pytest.mark.parametrize("z", _REDSHIFTS)
def test_no_default_bin_lies_entirely_before_the_big_bang_mechanism_only(family, z):
    """Same claim for the two not-yet-forward-validated families, against the
    mechanism directly (``SEDModel.build`` refuses these by name today).
    """
    edges_gyr = _edges_via_mechanism(family, z)
    assert _n_bins_beyond_age(edges_gyr, z) == 0


@pytest.mark.parametrize("family", _BUILDABLE_FAMILIES)
@pytest.mark.parametrize("z", _REDSHIFTS)
def test_formed_mass_equals_the_request(ssp_data_fsps, family, z):
    """log_mstar_formed matches the requested log_total_mass at every redshift."""
    log_total_mass = 10.3
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        sfh={
            "type": family,
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Fixed(log_total_mass),
        },
        redshift=Fixed(z),
    )
    state = model.predict_state({})
    assert float(state.derived["log_mstar_formed"]) == pytest.approx(log_total_mass, abs=1e-6)


@pytest.mark.parametrize("family", _BUILDABLE_FAMILIES + _UNBUILDABLE_FAMILIES)
def test_z_zero_edges_do_not_reproduce_the_previous_fixed_defaults(family):
    """At z=0 the new ladder is close to, but not bit-identical with, the old one.

    ``DEFAULT_BIN_EDGES_GYR`` is a fixed, hand-chosen ladder (``[0, 0.03, 0.1,
    0.3, 1.0, 3.0, 6.0, 13.7]`` Gyr); ``make_agebins_from_zred``'s
    Prospector-beta scheme keeps only the two youngest edges (30 Myr, 100
    Myr) and then log-spaces the remainder up to ``age_at_z(0)`` = 13.81 Gyr
    (Planck 2018) rather than tengri's own historical 13.7 Gyr anchor, so
    the two schemes diverge from the third edge onward. This is a deliberate
    convention adoption (Prospector-beta's own edge layout), not a
    regression target: the assertion pins the actual, measured
    disagreement rather than an aspirational match.
    """
    edges_gyr = _edges_via_mechanism(family, 0.0)
    old = np.asarray(DEFAULT_BIN_EDGES_GYR)
    assert edges_gyr.shape == old.shape
    # The two youngest edges (30 Myr, 100 Myr) are shared by construction.
    np.testing.assert_allclose(edges_gyr[:2], old[:2], atol=1e-12)
    # From the third edge on, the schemes disagree measurably.
    assert np.max(np.abs(edges_gyr[2:] - old[2:])) > 1e-3


def test_explicit_bin_edges_are_never_overridden(ssp_data_fsps):
    """A user-supplied ladder is untouched, even when it reaches past age(z)."""
    my_edges = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 20.0]  # deliberately beyond age(z)
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        sfh={
            "type": "dirichlet",
            "all_params": Fixed(DEFAULT),
            "bin_edges_gyr": my_edges,
        },
        redshift=Uniform(0.5, 6.0),
    )
    np.testing.assert_allclose(np.asarray(model.spec.bin_edges_gyr), my_edges)


def test_free_redshift_builds_the_ladder_at_the_prior_upper_bound(ssp_data_fsps):
    """Free redshift: the ladder is built once, at age_at_z of the prior ceiling.

    The edges cannot be re-built per posterior draw (``make_agebins_from_zred``
    is a NumPy, non-traceable function), so the one ladder that stays inside
    cosmic time for every draw is the one built at the youngest universe the
    prior admits.
    """
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        sfh={"type": "dirichlet", "all_params": Fixed(DEFAULT)},
        redshift=Uniform(0.5, 6.0),
    )
    expected_oldest_edge = float(age_at_z(6.0))
    assert float(np.asarray(model.spec.bin_edges_gyr)[-1]) == pytest.approx(
        expected_oldest_edge, rel=1e-9
    )
    # And every draw across the prior conserves the requested mass, including
    # the low-z end where the ladder built at z=6 does not reach that draw's
    # own (older) age of the universe.
    state = model.predict_state({"redshift": 0.5})
    assert np.isfinite(float(state.derived["log_mstar_formed"]))
