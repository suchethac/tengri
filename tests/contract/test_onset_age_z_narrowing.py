# SPDX-License-Identifier: BSD-3-Clause
"""Redshift and the SF-onset timing parameters, under the fraction contract (#2748).

A free onset is its fraction of age(z), so its support cannot exceed the age of
the universe at any redshift and needs no parse-time narrowing. An absolute Gyr
onset is a shim: a ``Fixed`` value is kept as an absolute time and is checked
against age(z) at a fixed redshift (refused) or a free redshift (warned); a free
prior on the Gyr name is refused and points at its fraction.

Every expectation is computed via ``tengri.utils.cosmology.age_at_z``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

import tengri
from tengri import FREE, Fixed, Uniform
from tengri.config.exceptions import ConfigError, FreeRedshiftOnsetCeilingWarning
from tengri.utils.cosmology import age_at_z


def _dexp(sfh_extra, redshift):
    return tengri.parse_groups(
        sfh={"type": "dexp", **sfh_extra},
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=redshift,
    )


def test_free_onset_fraction_is_bounded_by_the_age_of_the_universe():
    """A freed fraction of age(z) has its upper bound at 1, at every redshift."""
    spec = _dexp({"all_params": FREE}, Uniform(0.5, 12.0, default=5.0))
    frac = spec.get_distribution("sfh_dexp_start_frac")
    assert frac.hi <= 1.0 + 1e-12
    assert "sfh_dexp_start_gyr" not in spec.free_params


def test_free_onset_fraction_gives_the_same_bound_at_every_redshift():
    """The fraction maps to Gyr through age(z): hi * age(z) never exceeds age(z)."""
    spec = _dexp({"all_params": FREE}, Uniform(0.5, 12.0, default=5.0))
    hi = spec.get_distribution("sfh_dexp_start_frac").hi
    for z in (0.5, 2.0, 10.0):
        assert hi * float(age_at_z(z)) <= float(age_at_z(z)) + 1e-12


def test_free_prior_on_the_gyr_name_is_refused_and_names_the_fraction():
    with pytest.raises(ConfigError, match="sfh_dexp_start_frac"):
        _dexp({"start_gyr": Uniform(0.5, 13.0)}, Fixed(0.5))


def test_fixed_gyr_above_the_age_at_a_fixed_redshift_is_refused():
    age = float(age_at_z(2.0))
    with pytest.raises(ConfigError, match="later than the age of the universe"):
        _dexp({"start_gyr": Fixed(age + 1.0)}, Fixed(2.0))


def test_fixed_gyr_within_the_age_at_a_fixed_redshift_is_accepted():
    age = float(age_at_z(2.0))
    spec = _dexp({"start_gyr": Fixed(0.5 * age)}, Fixed(2.0))
    assert float(spec.get_distribution("sfh_dexp_start_gyr").value) == pytest.approx(0.5 * age)


def test_fixed_gyr_above_the_age_at_a_free_redshift_warns():
    with pytest.warns(FreeRedshiftOnsetCeilingWarning, match="sfh_dexp_start_gyr"):
        _dexp({"start_gyr": Fixed(13.5)}, Uniform(0.5, 12.0, default=5.0))


def test_fixed_gyr_shim_is_kept_as_an_absolute_time_not_a_fraction():
    """The shim holds the absolute value; its fraction is not in the spec."""
    spec = _dexp({"start_gyr": Fixed(3.0)}, Fixed(0.5))
    assert "sfh_dexp_start_gyr" in spec.fixed_params
    assert "sfh_dexp_start_frac" not in spec.fixed_params
    assert "sfh_dexp_start_frac" not in spec.free_params
