# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #2184 / #2748: the quench-ordering guard sees wildcard-freed timings.

The post-starburst flexible zone is ``[tlast, tflex]`` (Suess et al. 2022,
ApJ 935, 146; arXiv:2207.02883, Sect. 3.1.4 and Table 3). The build refuses a
pair that can cross, because a crossing gives the flexible bin a negative width
while the mass still closes and nothing is raised.

Since #2748 ``all_params: FREE`` frees the ``*_frac`` timings, with a prior that
reaches about 1 Gyr in age. The guard used to read only the keys the user wrote,
so ``tflex_gyr=Fixed(0.7)`` beside the wildcard-freed ``tlast_frac`` built
silently. It now reads the resolved priors.
"""

import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform
from tengri.components.stellar.sfh.registry import _AGE_FRAC_PRIORS
from tengri.config.exceptions import ConfigError

pytestmark = pytest.mark.regression_bug

#: Ceiling of the ``tlast_frac`` prior: 1.0 Gyr over the age of the universe.
_TLAST_FRAC_CEIL = _AGE_FRAC_PRIORS["sfh_psb2022_tlast_frac"].bounds[1]


def _build(ssp, *, redshift=None, **sfh_extra):
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={"type": "psb_suess2022", **sfh_extra},
        redshift=Fixed(0.05) if redshift is None else redshift,
    )


@pytest.mark.parametrize("key", ["tflex_gyr", "sfh_psb2022_tflex_gyr"])
def test_wildcard_freed_tlast_frac_crossing_a_gyr_pin_is_refused(synthetic_ssp_wide, key):
    """tflex pinned at 0.7 Gyr with tlast_frac freed by the wildcard reaches 0.96 Gyr."""
    with pytest.raises(ValueError, match="negative width"):
        _build(synthetic_ssp_wide, **{key: Fixed(0.7)}, all_params=FREE)


def test_wildcard_crossing_is_refused_at_a_free_redshift(synthetic_ssp_wide):
    """A free redshift widens the age range; the crossing stays reachable."""
    with pytest.raises(ValueError, match="negative width"):
        _build(
            synthetic_ssp_wide,
            redshift=Uniform(0.05, 0.5, default=0.1),
            tflex_gyr=Fixed(0.7),
            all_params=FREE,
        )


def test_wildcard_tlast_gyr_pin_crossing_the_freed_tflex_frac_is_refused(synthetic_ssp_wide):
    """The mirror: tlast pinned at 1.0 Gyr, tflex_frac freed by the wildcard.

    The freed tflex_frac floor is 1.0 Gyr over the age at z = 0, which is below
    1.0 Gyr at z = 0.05, so the zone can cross.
    """
    with pytest.raises(ValueError, match="negative width"):
        _build(synthetic_ssp_wide, tlast_gyr=Fixed(1.0), all_params=FREE)


def test_wildcard_free_prior_that_cannot_reach_the_pin_builds(synthetic_ssp_wide):
    """tflex pinned at 1.5 Gyr clears everything the freed tlast_frac can reach."""
    _build(synthetic_ssp_wide, tflex_gyr=Fixed(1.5), all_params=FREE)
    _build(synthetic_ssp_wide, sfh_psb2022_tflex_gyr=Fixed(1.5), all_params=FREE)


def test_wildcard_with_a_fraction_pin_above_the_ceiling_builds(synthetic_ssp_wide):
    """A tflex_frac pin past the tlast_frac ceiling is ordered in fractions, not refused."""
    _build(synthetic_ssp_wide, tflex_frac=Fixed(1.5 * _TLAST_FRAC_CEIL), all_params=FREE)


def test_wildcard_with_a_fraction_pin_inside_the_prior_is_refused(synthetic_ssp_wide):
    """tflex_frac pinned at half the tlast_frac ceiling can be crossed by the freed tlast_frac."""
    with pytest.raises(ValueError, match="negative width"):
        _build(synthetic_ssp_wide, tflex_frac=Fixed(0.5 * _TLAST_FRAC_CEIL), all_params=FREE)


def test_user_written_mixed_units_are_still_an_error(synthetic_ssp_wide):
    """A fraction and a Gyr time both written by the user stay a ConfigError."""
    with pytest.raises(ConfigError, match="mixed"):
        _build(
            synthetic_ssp_wide,
            tlast_frac=Uniform(0.001, 0.05),
            tflex_gyr=Fixed(2.0),
            all_params=Fixed(DEFAULT),
        )
