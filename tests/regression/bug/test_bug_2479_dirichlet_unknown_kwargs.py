# SPDX-License-Identifier: BSD-3-Clause
"""dirichlet() rejects unknown keyword arguments.

Regression for #2479. The dirichlet() function silently swallowed unknown
keyword arguments like z_frac_6=0.1 (for a 7-bin grid with valid z_frac_0..z_frac_5),
causing shape mismatches or silent failures. Fixed by validating
kwargs against the valid z_frac_0..z_frac_{n_bins-2} set at function entry and
raising TypeError with the list of unknown keys.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.stellar.sfh.nonparametric import dirichlet

pytestmark = [pytest.mark.regression_bug]


def test_dirichlet_valid_z_fracs_returns_finite_sfr():
    """dirichlet() with valid z_frac_0..z_frac_5 returns finite non-negative SFR.

    An 8-edge bin grid (bin_edges_gyr with 8 elements) has n_bins=7 and accepts
    z_frac_0 through z_frac_5 (six auxiliary variables). The returned SFR must
    be non-negative and finite, matching the input age shape and consistent
    with the total formed mass.
    """
    bin_edges_gyr = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.7])
    ages_yr = jnp.linspace(0.0, 13.7e9, 50)
    log_total_mass = 10.0

    sfr = dirichlet(
        ages_yr,
        log_total_mass=log_total_mass,
        bin_edges_gyr=bin_edges_gyr,
        z_frac_0=0.2,
        z_frac_1=0.3,
        z_frac_2=0.4,
        z_frac_3=0.5,
        z_frac_4=0.6,
        z_frac_5=0.7,
    )

    assert sfr.shape == ages_yr.shape, f"SFR shape {sfr.shape} != ages shape {ages_yr.shape}"
    assert jnp.all(sfr >= 0.0), "SFR has negative values"
    assert jnp.all(jnp.isfinite(sfr)), "SFR has NaN or Inf values"

    # For a piecewise-constant SFH, the SFR integrates to the total mass.
    # Verify using trapezoid rule with relative tolerance 1e-5.
    bin_widths_yr = jnp.diff(bin_edges_gyr) * 1e9
    midpoints_yr = (bin_edges_gyr[:-1] + bin_edges_gyr[1:]) / 2.0 * 1e9
    sfr_mid = dirichlet(
        midpoints_yr,
        log_total_mass=log_total_mass,
        bin_edges_gyr=bin_edges_gyr,
        z_frac_0=0.2,
        z_frac_1=0.3,
        z_frac_2=0.4,
        z_frac_3=0.5,
        z_frac_4=0.6,
        z_frac_5=0.7,
    )
    total_formed = float(jnp.sum(sfr_mid * bin_widths_yr))
    assert total_formed == pytest.approx(10.0**log_total_mass, rel=1e-5)


def test_dirichlet_unknown_z_frac_raises_typeerror():
    """dirichlet() raises TypeError when unknown z_frac_* kwarg is supplied.

    For an 8-edge bin grid (n_bins=7), supplying z_frac_6 (beyond z_frac_5) or
    other unknown parameter names must raise TypeError naming the bad keys.
    """
    bin_edges_gyr = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.7])
    ages_yr = jnp.linspace(0.0, 13.7e9, 50)

    # Test case 1: z_frac index out of range (7th fraction when only 6 expected)
    with pytest.raises(
        TypeError,
        match=r"dirichlet\(\) got unexpected keyword argument\(s\) \['z_frac_6'\]",
    ):
        dirichlet(
            ages_yr,
            log_total_mass=10.0,
            bin_edges_gyr=bin_edges_gyr,
            z_frac_0=0.5,
            z_frac_1=0.5,
            z_frac_2=0.5,
            z_frac_3=0.5,
            z_frac_4=0.5,
            z_frac_5=0.5,
            z_frac_6=0.1,
        )

    # Test case 2: misspelled parameter name
    with pytest.raises(
        TypeError,
        match=r"dirichlet\(\) got unexpected keyword argument\(s\) \['zfrac_0'\]",
    ):
        dirichlet(
            ages_yr,
            log_total_mass=10.0,
            bin_edges_gyr=bin_edges_gyr,
            zfrac_0=0.5,
        )

    # Test case 3: multiple unknown kwargs
    with pytest.raises(
        TypeError,
        match=r"dirichlet\(\) got unexpected keyword argument\(s\) \['n_pts', 'z_frac_6'\]",
    ):
        dirichlet(
            ages_yr,
            log_total_mass=10.0,
            bin_edges_gyr=bin_edges_gyr,
            n_pts=200,
            z_frac_6=0.3,
        )
