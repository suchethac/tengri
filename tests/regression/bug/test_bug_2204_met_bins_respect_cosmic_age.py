# SPDX-License-Identifier: BSD-3-Clause
r"""Regression tests for #2204 — metallicity-history bin edges must fit cosmic age.

Mechanism
---------

The metallicity-history bins mode assigns metallicity in six fixed lookback-time bins
whose edges are a hard-coded z=0 ladder (_DEFAULT_MET_BIN_EDGES_LOG_YR in Gyr):

  [1 Myr, 31.6 Myr, 316 Myr, 1 Gyr, 3.16 Gyr, 7.94 Gyr, 13.8 Gyr]

These edges are in absolute lookback time, not scaled to age_at_z(z). At high
redshifts where the cosmic age is younger than the oldest bin edges, those bins
become unreachable (they lie before the Big Bang). The fix is to REFUSE at build
time with a ParameterError naming the unreachable bin edge(s) and cosmic age,
pointing the user to met_bin_edges_log_yr as the remedy.

Measured (FSPS/MILES grid, real SSP, tsnorm SFH):

  z=0.0:  age = 13.81 Gyr, bin 5 reachable (max edge 7.94 Gyr < 13.81)
  z=0.5:  age =  8.61 Gyr, bin 5 reachable (max edge 7.94 Gyr < 8.61)
  z=0.66: age =  7.55 Gyr, bin 5 unreachable (max edge 7.94 Gyr > 7.55)
  z=1.0:  age =  5.87 Gyr, bins 4+5 unreachable (edges 3.16+ Gyr exceed 5.87)

https://github.com/suchethac/tengri/issues/2204
"""

from __future__ import annotations

import pytest

from tengri import SEDModel, Fixed, Uniform, DEFAULT
from tengri.config.exceptions import ParameterError
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug


class TestMetBinsCosmicAgeBuild:
    """Metallicity-history bins must fit within age_at_z at the model redshift."""

    def test_met_bins_refused_at_high_z_buildi(self, synthetic_ssp, simple_observation):
        """HIGH-Z BUILD REFUSES: bins unreachable at z=0.66 (age=7.55 Gyr).

        Bin 5 spans lookback 7.94-13.8 Gyr. At z=0.66, age_at_z=7.55 Gyr,
        so the entire bin 5 lies before the Big Bang. SEDModel.build raises
        ParameterError naming the unreachable bin edge (7.94 Gyr) and cosmic
        age (7.55 Gyr), pointing to met_bin_edges_log_yr as the remedy.
        """
        # z=0.66: age_at_z = 7.55 Gyr (unreachable for bin 5 max edge 7.94)
        z_high = 0.66
        expected_age = age_at_z(z_high)
        assert expected_age < 7.94, f"Test setup: z={z_high} should give age < 7.94 Gyr"

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
        # Error should name the unreachable bin edge and cosmic age
        assert "7.94" in error_msg or "7.9" in error_msg, (
            f"Error should name bin edge (7.94 Gyr); got: {error_msg}"
        )
        assert f"{expected_age:.2f}" in error_msg or f"{expected_age:.1f}" in error_msg, (
            f"Error should name cosmic age ({expected_age:.2f} Gyr); got: {error_msg}"
        )
        assert "met_bin_edges_log_yr" in error_msg, (
            f"Error should point to met_bin_edges_log_yr as remedy; got: {error_msg}"
        )

    def test_met_bins_accepted_at_low_z(self, synthetic_ssp, simple_observation):
        """LOW-Z BUILD SUCCEEDS: bins reachable at z=0 (age=13.81 Gyr).

        At z=0, all bins fit inside age_at_z. Build should succeed.
        """
        z_low = 0.0
        expected_age = age_at_z(z_low)
        assert expected_age >= 13.8, f"Test setup: z={z_low} should give age >= 13.8 Gyr"

        obs = simple_observation
        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
            met={"type": "bins", "all_params": Fixed(DEFAULT), "met_bin_0": -0.3},
            redshift=Fixed(z_low),
        )
        assert model is not None
