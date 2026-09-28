# SPDX-License-Identifier: BSD-3-Clause
r"""Regression tests for #2204 — metallicity-history bins must fit cosmic age at z_floor.

Mechanism
---------

The metallicity-history bins mode assigns metallicity in six fixed lookback-time bins
whose edges are a hard-coded z=0 ladder (_DEFAULT_MET_BIN_EDGES_LOG_YR in Gyr):

  [1 Myr, 31.6 Myr, 316 Myr, 1 Gyr, 3.16 Gyr, 7.94 Gyr, 13.8 Gyr]

These bins are [start, end] lookback intervals. At high redshifts where the cosmic age
is younger than a bin's START, that bin becomes unreachable (it lies before the Big
Bang). The check evaluates reachability at the LOWEST redshift the prior admits:
for Fixed(z), that is z; for Uniform(z_min, z_max), that is z_min. This ensures
defaults build and high-z builds that would be unreachable are refused early.

Measured (FSPS/MILES grid, real SSP, tsnorm SFH):

  z=0.1:  age = 12.47 Gyr, all bin starts <= 7.94 Gyr < age (reachable)
  z=3.0:  age = 2.15 Gyr, bins [3.16,7.94] and [7.94,13.8] have starts >= 2.15 (unreachable)
  Uniform(0.05, 4.0): z_floor = 0.05, age ≈ 13.8 Gyr, all bins reachable
  Uniform(2.5, 4.0): z_floor = 2.5, age ≈ 3.2 Gyr, bins at 3.16+ Gyr unreachable

https://github.com/suchethac/tengri/issues/2204
"""

from __future__ import annotations

import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.config.exceptions import ParameterError
from tengri.parameters.priors import Uniform
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug


class TestMetBinsCosmicAgeBuild:
    """Metallicity-history bins must fit within age_at_z at the lowest admitted redshift.

    Reachability is judged at the lowest redshift the prior admits: for Fixed(z),
    that is z; for Uniform(z_min, z_max), that is z_min. The check compares only
    bin STARTS (lower edges) against cosmic age; bin ENDS can exceed cosmic age.
    """

    def test_fixed_low_redshift_builds(self, synthetic_ssp, simple_observation):
        """FIXED(0.1) BUILDS: bin starts [0.00, ..., 7.94] < age_at_z(0.1)=12.47 Gyr.

        At redshift 0.1, cosmic age is 12.47 Gyr. All bin lower edges are
        <= 7.94 Gyr < 12.47 Gyr, so all bins are reachable.
        """
        z_low = 0.1
        expected_age = age_at_z(z_low)
        assert expected_age > 7.94, f"Test setup: z={z_low} should give age > 7.94 Gyr"

        obs = simple_observation
        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
            met={"type": "bins", "all_params": Fixed(DEFAULT), "met_bin_0": -0.3},
            redshift=Fixed(z_low),
        )
        assert model is not None

    def test_fixed_high_redshift_refuses(self, synthetic_ssp, simple_observation):
        """FIXED(3.0) REFUSES: bin starts [3.16, 7.94] >= age_at_z(3.0)=2.15 Gyr.

        At redshift 3.0, cosmic age is 2.15 Gyr. Bins starting at 3.16 and 7.94 Gyr
        lie before the Big Bang (their starts exceed cosmic age). Build raises
        ParameterError naming the unreachable bins [3.16, 7.94] and [7.94, 13.80].
        """
        z_high = 3.0
        expected_age = age_at_z(z_high)
        assert expected_age < 3.16, f"Test setup: z={z_high} should give age < 3.16 Gyr"

        obs = simple_observation

        with pytest.raises(ParameterError) as exc_info:
            SEDModel.build(
                ssp_data=synthetic_ssp,
                observation=obs,
                sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
                met={"type": "bins", "all_params": Fixed(DEFAULT), "met_bin_0": -0.3},
                redshift=Fixed(z_high),
            )

        error_msg = str(exc_info.value)
        # Error should name the unreachable bins [3.16, 7.94] and [7.94, 13.80]
        assert "[3.1" in error_msg or "[3.16" in error_msg, (
            f"Error should name bins starting at 3.16 Gyr; got: {error_msg}"
        )
        assert "[7.9" in error_msg or "[7.94" in error_msg, (
            f"Error should name bins starting at 7.94 Gyr; got: {error_msg}"
        )
        # Error should name cosmic age and cite #2433
        assert f"{expected_age:.2f}" in error_msg or f"{expected_age:.1f}" in error_msg, (
            f"Error should name cosmic age ({expected_age:.2f} Gyr); got: {error_msg}"
        )
        assert "2433" in error_msg, f"Error should reference issue #2433; got: {error_msg}"

    def test_free_narrow_low_redshift_builds(self, synthetic_ssp, simple_observation):
        """FREE UNIFORM(0.05, 4.0) BUILDS: z_floor=0.05, age ≈13.8 Gyr.

        Free redshift priors use the lower bound (z_floor) as the reachability floor.
        With Uniform(0.05, 4.0), z_floor = 0.05 and age ≈ 13.8 Gyr.
        All bin starts <= 7.94 Gyr, so build succeeds.
        """
        obs = simple_observation
        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
            met={"type": "bins", "all_params": Fixed(DEFAULT), "met_bin_0": -0.3},
            redshift=Uniform(0.05, 4.0),
        )
        assert model is not None

    def test_free_narrow_high_redshift_refuses(self, synthetic_ssp, simple_observation):
        """FREE UNIFORM(2.5, 4.0) REFUSES: z_floor=2.5, age ≈3.2 Gyr.

        With Uniform(2.5, 4.0), z_floor = 2.5 and age ≈ 3.2 Gyr.
        Bins starting at 3.16 and 7.94 Gyr exceed this, so build refuses.
        """
        obs = simple_observation

        with pytest.raises(ParameterError) as exc_info:
            SEDModel.build(
                ssp_data=synthetic_ssp,
                observation=obs,
                sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
                met={"type": "bins", "all_params": Fixed(DEFAULT), "met_bin_0": -0.3},
                redshift=Uniform(2.5, 4.0),
            )

        error_msg = str(exc_info.value)
        # Verify the error names unreachable bins and #2433
        assert "[3.1" in error_msg or "[3.16" in error_msg, (
            f"Error should name bins starting at 3.16 Gyr; got: {error_msg}"
        )
        assert "2433" in error_msg, f"Error should reference issue #2433; got: {error_msg}"
