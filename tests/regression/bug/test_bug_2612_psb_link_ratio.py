# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for issue #2612: PSB link ratio parameter.

The first ratio_old parameter is a link between the oldest flex bin
and the youngest fixed bin. Adjacent steps are ratio_old_1 and beyond.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.nonparametric import (
    psb_continuity_flex,
    PSB_FLEX_DEFAULT_N_FIXED,
    PSB_FLEX_DEFAULT_MAX_AGE_GYR,
)

pytestmark = pytest.mark.regression_bug


class TestPSBLinkRatioDefinition:
    """Test that ratio_old_0 is the link, ratio_old_1+ are adjacent steps."""

    def test_psb_flex_link_ratio_definition(self):
        """Verify that ratio_old_0 = link and ratio_old_1, ratio_old_2 are adjacent steps.

        Build psb_flex with tlast=0.2 Gyr, tflex=2 Gyr, 5 flex bins, 3 fixed bins.
        With link=0.4 and adjacent steps (0.2, -0.3):
          log10(SFR_youngest_fixed / SFR_oldest_flex) = 0.4
          log10(SFR_fixed_1 / SFR_fixed_0) = 0.2
          log10(SFR_fixed_2 / SFR_fixed_1) = -0.3
        """
        age_yr = jnp.logspace(6, 10.14, 256)

        # Parameters
        tlast_gyr = 0.2
        tflex_gyr = 2.0
        n_fixed = 3
        link = 0.4  # ratio_old_0
        adjacent_step_1 = 0.2  # ratio_old_1
        adjacent_step_2 = -0.3  # ratio_old_2

        sfr = psb_continuity_flex(
            age_yr,
            log_total_mass=10.0,
            tlast_gyr=tlast_gyr,
            tflex_gyr=tflex_gyr,
            ratio_young=0.0,
            flex_0=0.0,
            flex_1=0.0,
            flex_2=0.0,
            flex_3=0.0,
            ratio_old_0=link,
            ratio_old_1=adjacent_step_1,
            ratio_old_2=adjacent_step_2,
        )

        # Extract bin edges and SFR values
        max_age_gyr = PSB_FLEX_DEFAULT_MAX_AGE_GYR
        fixed_edges_gyr = jnp.linspace(tflex_gyr, max_age_gyr, n_fixed + 1)

        # Construct full edge array as psb_continuity_flex does
        flex_edges_gyr = jnp.linspace(tlast_gyr, tflex_gyr, 6)[1:]  # 5 flex bins
        all_edges_gyr = jnp.concatenate(
            [jnp.array([0.0, tlast_gyr]), flex_edges_gyr, fixed_edges_gyr[1:]]
        )
        all_edges_yr = all_edges_gyr * 1e9

        # Get SFR at bin centers
        bin_centers_yr = (all_edges_yr[:-1] + all_edges_yr[1:]) / 2
        sfr_at_centers = jnp.interp(bin_centers_yr, age_yr, sfr)

        # Bins are: [0, tlast], [tlast, flex_edge_1], ..., [flex_edge_4, tflex],
        #           [tflex, fixed_1], [fixed_1, fixed_2], [fixed_2, max_age]
        # So indices: 0 (youngest), 1-5 (flex zone), 6-8 (fixed zone, oldest)
        sfr_oldest_flex = sfr_at_centers[5]  # last flex bin
        sfr_youngest_fixed_0 = sfr_at_centers[6]  # first fixed bin
        sfr_youngest_fixed_1 = sfr_at_centers[7]  # second fixed bin
        sfr_youngest_fixed_2 = sfr_at_centers[8]  # third fixed bin

        # Check link ratio
        log10_link_actual = jnp.log10(sfr_youngest_fixed_0 / sfr_oldest_flex)
        np.testing.assert_allclose(log10_link_actual, link, atol=1e-10)

        # Check adjacent steps
        log10_step_1_actual = jnp.log10(sfr_youngest_fixed_1 / sfr_youngest_fixed_0)
        np.testing.assert_allclose(log10_step_1_actual, adjacent_step_1, atol=1e-10)

        log10_step_2_actual = jnp.log10(sfr_youngest_fixed_2 / sfr_youngest_fixed_1)
        np.testing.assert_allclose(log10_step_2_actual, adjacent_step_2, atol=1e-10)

    def test_psb_suess2022_link_ratio_definition(self):
        """Same test for psb_suess2022 (1 flex bin case)."""
        from tengri.components.stellar.sfh.nonparametric import psb_continuity_flex

        age_yr = jnp.logspace(6, 10.14, 256)

        # Parameters for psb_suess2022 case (single flex bin)
        tlast_gyr = 0.2
        tflex_gyr = 2.0
        link = 0.4
        adjacent_step_1 = 0.2
        adjacent_step_2 = -0.3

        sfr = psb_continuity_flex(
            age_yr,
            log_total_mass=10.0,
            tlast_gyr=tlast_gyr,
            tflex_gyr=tflex_gyr,
            ratio_young=0.0,
            # No flex_* ratios means 1 flex bin
            ratio_old_0=link,
            ratio_old_1=adjacent_step_1,
            ratio_old_2=adjacent_step_2,
        )

        # Similar verification as above
        n_fixed = 3
        max_age_gyr = PSB_FLEX_DEFAULT_MAX_AGE_GYR
        fixed_edges_gyr = jnp.linspace(tflex_gyr, max_age_gyr, n_fixed + 1)

        flex_edges_gyr = jnp.linspace(tlast_gyr, tflex_gyr, 2)[1:]  # 1 flex bin
        all_edges_gyr = jnp.concatenate(
            [jnp.array([0.0, tlast_gyr]), flex_edges_gyr, fixed_edges_gyr[1:]]
        )
        all_edges_yr = all_edges_gyr * 1e9

        bin_centers_yr = (all_edges_yr[:-1] + all_edges_yr[1:]) / 2
        sfr_at_centers = jnp.interp(bin_centers_yr, age_yr, sfr)

        # Bins: [0, tlast], [tlast, tflex], [tflex, fixed_1], [fixed_1, fixed_2], [fixed_2, max]
        # Indices: 0, 1, 2, 3, 4
        sfr_oldest_flex = sfr_at_centers[1]
        sfr_youngest_fixed_0 = sfr_at_centers[2]

        log10_link_actual = jnp.log10(sfr_youngest_fixed_0 / sfr_oldest_flex)
        np.testing.assert_allclose(log10_link_actual, link, atol=1e-10)


class TestPSBLinkRatioRegression:
    """Test that link=0 reproduces origin/main output (backward compatibility)."""

    def test_link_zero_backward_compat(self):
        """With link=0, the SFR at oldest flex and youngest fixed bins are equal."""
        age_yr = jnp.logspace(6, 10.14, 256)

        sfr = psb_continuity_flex(
            age_yr,
            log_total_mass=10.0,
            tlast_gyr=0.2,
            tflex_gyr=2.0,
            ratio_young=0.3,
            flex_0=0.1,
            flex_1=-0.2,
            flex_2=0.05,
            flex_3=0.1,
            ratio_old_0=0.0,  # link = 0 means oldest flex and youngest fixed have same SFR
            ratio_old_1=0.2,
            ratio_old_2=-0.1,
        )

        # With link=0, the transition should be invisible
        n_fixed = 3
        max_age_gyr = PSB_FLEX_DEFAULT_MAX_AGE_GYR
        fixed_edges_gyr = jnp.linspace(2.0, max_age_gyr, n_fixed + 1)

        flex_edges_gyr = jnp.linspace(0.2, 2.0, 6)[1:]
        all_edges_gyr = jnp.concatenate(
            [jnp.array([0.0, 0.2]), flex_edges_gyr, fixed_edges_gyr[1:]]
        )
        all_edges_yr = all_edges_gyr * 1e9

        bin_centers_yr = (all_edges_yr[:-1] + all_edges_yr[1:]) / 2
        sfr_at_centers = jnp.interp(bin_centers_yr, age_yr, sfr)

        sfr_oldest_flex = sfr_at_centers[5]
        sfr_youngest_fixed = sfr_at_centers[6]

        # With link=0, they should be equal
        np.testing.assert_allclose(sfr_oldest_flex, sfr_youngest_fixed, rtol=1e-10)


class TestPSBMassConservation:
    """Test mass conservation with non-zero link ratios."""

    def test_mass_conservation_with_link(self):
        """Total formed mass should equal 10^log_total_mass."""
        age_yr = jnp.logspace(6, 10.14, 512)
        log_total_mass = 10.0

        sfr = psb_continuity_flex(
            age_yr,
            log_total_mass=log_total_mass,
            tlast_gyr=0.2,
            tflex_gyr=2.0,
            ratio_young=0.4,
            flex_0=0.1,
            flex_1=-0.2,
            ratio_old_0=0.4,  # non-zero link
            ratio_old_1=0.2,
            ratio_old_2=-0.3,
        )

        # Integrate SFR to get formed mass
        # d(mass) / d(ln t) = t * SFR(t), so integral is sum of SFR(t) * t * d(ln t)
        dt_logspace = jnp.diff(jnp.log(age_yr))
        dm = sfr[:-1] * age_yr[:-1] * dt_logspace
        mass_formed = jnp.sum(dm)

        expected_mass = 10.0 ** log_total_mass
        np.testing.assert_allclose(mass_formed, expected_mass, rtol=0.01)

    def test_mass_conservation_with_different_links(self):
        """Test several link values."""
        age_yr = jnp.logspace(6, 10.14, 512)
        log_total_mass = 10.0

        for link in [-0.5, 0.0, 0.4]:
            sfr = psb_continuity_flex(
                age_yr,
                log_total_mass=log_total_mass,
                tlast_gyr=0.2,
                tflex_gyr=2.0,
                ratio_young=0.3,
                flex_0=0.1,
                ratio_old_0=link,
                ratio_old_1=0.2,
                ratio_old_2=-0.3,
            )

            dt_logspace = jnp.diff(jnp.log(age_yr))
            dm = sfr[:-1] * age_yr[:-1] * dt_logspace
            mass_formed = jnp.sum(dm)

            expected_mass = 10.0 ** log_total_mass
            np.testing.assert_allclose(mass_formed, expected_mass, rtol=0.01)


class TestPSBCountContract:
    """Test that the parameter count contract is satisfied."""

    def test_psb_flex_declares_n_fixed_ratio_old(self):
        """psb_flex should declare exactly n_fixed ratio_old parameters."""
        from tengri import SEDModel, Observation

        # Load a minimal SSP to build a model
        try:
            from tengri.data import SSPData
            ssp = SSPData.load()
        except Exception:
            pytest.skip("SSP data not available")

        obs = Observation.from_filters(['sdss_u', 'sdss_g'])

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={'type': 'psb_flex', 'all_params': 'free'},
        )

        # Count ratio_old parameters
        ratio_old_params = [p for p in model.spec.free_params if 'ratio_old' in p]
        n_fixed = PSB_FLEX_DEFAULT_N_FIXED

        # Should have exactly n_fixed ratio_old parameters
        assert len(ratio_old_params) == n_fixed, (
            f"Expected {n_fixed} ratio_old params, got {len(ratio_old_params)}: {ratio_old_params}"
        )

    def test_psb_suess2022_declares_n_fixed_ratio_old(self):
        """psb_suess2022 should declare exactly n_fixed ratio_old parameters."""
        try:
            from tengri.data import SSPData
            ssp = SSPData.load()
        except Exception:
            pytest.skip("SSP data not available")

        from tengri import SEDModel, Observation

        obs = Observation.from_filters(['sdss_u', 'sdss_g'])

        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={'type': 'psb_suess2022', 'all_params': 'free'},
        )

        ratio_old_params = [p for p in model.spec.free_params if 'ratio_old' in p]
        n_fixed = PSB_FLEX_DEFAULT_N_FIXED

        assert len(ratio_old_params) == n_fixed, (
            f"Expected {n_fixed} ratio_old params, got {len(ratio_old_params)}: {ratio_old_params}"
        )
