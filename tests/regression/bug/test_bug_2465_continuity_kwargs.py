# SPDX-License-Identifier: BSD-3-Clause
"""continuity() rejects unknown keyword arguments.

Regression for #2465. The continuity() function silently swallowed unknown
keyword arguments like n_pts=200 or obsolete ratio_7=0.1 (for a 5-edge grid),
causing shape mismatches between sfr_bins and mass_unnorm. Fixed by validating
kwargs against the valid ratio_0..ratio_{n_bins-2} set at function entry and
raising TypeError with the list of unknown keys.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.stellar.sfh.nonparametric import continuity

pytestmark = [pytest.mark.regression_bug]


def test_continuity_valid_ratios_returns_finite_sfr():
    """continuity() with valid ratio_0..ratio_2 returns finite non-negative SFR.

    A 5-edge bin grid (bin_edges_gyr with 5 elements) has n_bins=4 and accepts
    ratio_0, ratio_1, ratio_2. The returned SFR must be non-negative and finite,
    matching the input age shape and consistent with the total formed mass.
    """
    bin_edges_gyr = jnp.array([0.0, 0.1, 0.5, 2.0, 6.0])
    ages_yr = jnp.linspace(0.0, 6.0e9, 50)
    log_total_mass = 10.0

    sfr = continuity(
        ages_yr,
        log_total_mass=log_total_mass,
        bin_edges_gyr=bin_edges_gyr,
        ratio_0=0.2,
        ratio_1=-0.1,
        ratio_2=0.05,
    )

    assert sfr.shape == ages_yr.shape, f"SFR shape {sfr.shape} != ages shape {ages_yr.shape}"
    assert jnp.all(sfr >= 0.0), "SFR has negative values"
    assert jnp.all(jnp.isfinite(sfr)), "SFR has NaN or Inf values"

    # For a piecewise-constant SFH, SFR at the bin midpoints times the bin
    # widths sums to exactly the declared total formed mass.
    bin_widths_yr = jnp.diff(bin_edges_gyr) * 1e9
    midpoints_yr = (bin_edges_gyr[:-1] + bin_edges_gyr[1:]) / 2.0 * 1e9
    sfr_mid = continuity(
        midpoints_yr,
        log_total_mass=log_total_mass,
        bin_edges_gyr=bin_edges_gyr,
        ratio_0=0.2,
        ratio_1=-0.1,
        ratio_2=0.05,
    )
    total_formed = float(jnp.sum(sfr_mid * bin_widths_yr))
    assert total_formed == pytest.approx(10.0**log_total_mass, rel=1e-6)


def test_continuity_unknown_ratio_raises_typeerror():
    """continuity() raises TypeError when unknown ratio_* kwarg is supplied.

    For a 5-edge bin grid (n_bins=4), supplying ratio_7 (beyond ratio_2) or
    obsolete parameter names like n_pts=200 must raise TypeError naming the
    bad keys.
    """
    bin_edges_gyr = jnp.array([0.0, 0.1, 0.5, 2.0, 6.0])
    ages_yr = jnp.linspace(0.0, 6.0e9, 50)

    # Test case 1: ratio index out of range
    with pytest.raises(
        TypeError,
        match=r"continuity\(\) got unexpected keyword argument\(s\) \['ratio_7'\]",
    ):
        continuity(
            ages_yr,
            log_total_mass=10.0,
            bin_edges_gyr=bin_edges_gyr,
            ratio_7=0.1,
        )

    # Test case 2: obsolete n_pts parameter
    with pytest.raises(
        TypeError,
        match=r"continuity\(\) got unexpected keyword argument\(s\) \['n_pts'\]",
    ):
        continuity(
            ages_yr,
            log_total_mass=10.0,
            bin_edges_gyr=bin_edges_gyr,
            n_pts=200,
        )

    # Test case 3: multiple unknown kwargs
    with pytest.raises(
        TypeError,
        match=r"continuity\(\) got unexpected keyword argument\(s\) \['n_pts', 'ratio_5'\]",
    ):
        continuity(
            ages_yr,
            log_total_mass=10.0,
            bin_edges_gyr=bin_edges_gyr,
            n_pts=200,
            ratio_5=0.3,
        )
