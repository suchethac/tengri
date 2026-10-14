# SPDX-License-Identifier: BSD-3-Clause
"""``FreeRedshiftOnsetCeilingWarning`` fires only for a Fixed absolute Gyr onset (#2521, #2748).

An onset given as a fraction of age(z) cannot exceed the age of the universe at
any redshift, so a free redshift with the default prior never warns. The warning
is kept for a ``Fixed`` absolute Gyr onset (the shim of its fraction), which
does not scale with age(z): at a free redshift a draw near the younger-universe
end can place it before the Big Bang.
"""

from __future__ import annotations

import warnings

import pytest

import tengri
from tengri import FREE, Fixed, Uniform
from tengri.config.exceptions import FreeRedshiftOnsetCeilingWarning

pytestmark = pytest.mark.contract


def _build(sfh_extra, redshift):
    return tengri.parse_groups(
        sfh={"type": "dexp", **sfh_extra},
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=redshift,
    )


def _ceiling_warnings(sfh_extra, redshift):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(sfh_extra, redshift)
    return [w for w in caught if issubclass(w.category, FreeRedshiftOnsetCeilingWarning)]


def test_fires_for_a_fixed_gyr_onset_above_age_at_a_wide_free_redshift():
    assert _ceiling_warnings({"start_gyr": Fixed(13.5)}, Uniform(0.5, 12.0, default=5.0))


def test_does_not_fire_for_a_free_fraction_at_a_wide_free_redshift():
    assert not _ceiling_warnings({"all_params": FREE}, Uniform(0.5, 12.0, default=5.0))


def test_does_not_fire_for_a_fixed_redshift():
    assert not _ceiling_warnings({"start_gyr": Fixed(0.2)}, Fixed(0.5))


def test_does_not_fire_when_the_fixed_onset_is_within_the_age_at_the_upper_redshift():
    assert not _ceiling_warnings({"start_gyr": Fixed(0.2)}, Uniform(0.5, 12.0, default=5.0))
