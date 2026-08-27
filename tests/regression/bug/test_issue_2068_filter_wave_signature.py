# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #2068: compile_signature must key filter wavelength content.

Issue: SEDModel.compile_signature() keys filter_wave_shape (SHAPE only) but not
filter_wave_values (CONTENT). Two models with identical transmission and point
count but different wavelength grids produce identical signatures, causing the
second model to silently reuse the first model's compiled photometry closure
which has baked the first model's wavelengths. The user-visible defect is
that photometry or spectrum computed with the wrong wavelength grid is
silently served.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import FilterCurve, Photometry

pytestmark = pytest.mark.regression_bug


class TestIssue2068FilterWaveSignature:
    """Regression tests for filter wavelength content in compile_signature.

    These tests verify that compile_signature includes filter and spectroscopy
    wavelength content, preventing silent cache collisions where models with
    identical shapes but different wavelength grids reuse compiled kernels
    that have baked in the first model's wavelengths.
    """

    def test_filter_wavelength_shift_changes_signature(self, synthetic_ssp):
        """Shifting filter wavelengths by +500 A changes the signature.

        Same transmission and point count, but wavelengths differ by +500 A.
        This should produce different signatures after the fix.

        Before the fix: signatures identical, photometry from first model is silently reused.
        After the fix: signatures differ, each model gets its own compiled kernel.
        """
        wave_base = jnp.linspace(4000, 5000, 50)
        wave_shifted = jnp.linspace(4500, 5500, 50)
        trans = jnp.ones(50, dtype=jnp.float64) * 0.5

        filters_base = [FilterCurve(wave=wave_base, trans=trans, name="optical")]
        filters_shifted = [FilterCurve(wave=wave_shifted, trans=trans, name="optical")]

        obs_base = Observation(photometry=Photometry(filters=filters_base))
        obs_shifted = Observation(photometry=Photometry(filters=filters_shifted))

        spec = Parameters(
            redshift=0.3,
            sfh_dpl_alpha=1.5,
            sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=2.0,
            sfh_dpl_log_total_mass=10.5,
            met_logzsol=0.0,
            dust_tau_bc=0.1,
            dust_tau_diff=0.2,
        )

        model_base = SEDModel(spec, synthetic_ssp, observation=obs_base)
        model_shifted = SEDModel(spec, synthetic_ssp, observation=obs_shifted)

        sig_base = model_base.compile_signature()
        sig_shifted = model_shifted.compile_signature()

        assert sig_base != sig_shifted, "Filter wavelength shift should change the signature"

    def test_identical_filter_content_produces_equal_signatures(self, synthetic_ssp):
        """Two models with identical filter content should produce equal signatures.

        This is the baseline case: identical wavelengths + transmission should
        produce identical signatures and reuse the compiled kernel safely.
        """
        wave = jnp.linspace(4000, 5000, 50)
        trans = jnp.ones(50, dtype=jnp.float64) * 0.5

        filters_1 = [FilterCurve(wave=wave, trans=trans, name="optical")]
        filters_2 = [FilterCurve(wave=wave, trans=trans, name="optical")]

        obs_1 = Observation(photometry=Photometry(filters=filters_1))
        obs_2 = Observation(photometry=Photometry(filters=filters_2))

        spec = Parameters(
            redshift=0.3,
            sfh_dpl_alpha=1.5,
            sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=2.0,
            sfh_dpl_log_total_mass=10.5,
            met_logzsol=0.0,
            dust_tau_bc=0.1,
            dust_tau_diff=0.2,
        )

        model_1 = SEDModel(spec, synthetic_ssp, observation=obs_1)
        model_2 = SEDModel(spec, synthetic_ssp, observation=obs_2)

        sig_1 = model_1.compile_signature()
        sig_2 = model_2.compile_signature()

        assert sig_1 == sig_2, "Identical filter content must produce equal signatures"

    def test_filter_transmission_change_produces_different_signature(self, synthetic_ssp):
        """Changing transmission (not just wavelength) also changes signature.

        Verifies that both filter_wave_id and filter_trans_id are in the signature.
        """
        wave = jnp.linspace(4000, 5000, 50)
        trans_05 = jnp.ones(50, dtype=jnp.float64) * 0.5
        trans_04 = jnp.ones(50, dtype=jnp.float64) * 0.4

        filters_05 = [FilterCurve(wave=wave, trans=trans_05, name="optical")]
        filters_04 = [FilterCurve(wave=wave, trans=trans_04, name="optical")]

        obs_05 = Observation(photometry=Photometry(filters=filters_05))
        obs_04 = Observation(photometry=Photometry(filters=filters_04))

        spec = Parameters(
            redshift=0.3,
            sfh_dpl_alpha=1.5,
            sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=2.0,
            sfh_dpl_log_total_mass=10.5,
            met_logzsol=0.0,
            dust_tau_bc=0.1,
            dust_tau_diff=0.2,
        )

        model_05 = SEDModel(spec, synthetic_ssp, observation=obs_05)
        model_04 = SEDModel(spec, synthetic_ssp, observation=obs_04)

        sig_05 = model_05.compile_signature()
        sig_04 = model_04.compile_signature()

        assert sig_05 != sig_04, "Different transmission should produce different signatures"

    def test_filter_point_count_change_produces_different_signature(self, synthetic_ssp):
        """Changing filter point count also changes signature.

        Same wavelength range and transmission value, but different sampling.
        """
        trans = 0.5

        wave_50 = jnp.linspace(4000, 5000, 50)
        trans_50 = jnp.ones(50, dtype=jnp.float64) * trans

        wave_60 = jnp.linspace(4000, 5000, 60)
        trans_60 = jnp.ones(60, dtype=jnp.float64) * trans

        filters_50 = [FilterCurve(wave=wave_50, trans=trans_50, name="optical")]
        filters_60 = [FilterCurve(wave=wave_60, trans=trans_60, name="optical")]

        obs_50 = Observation(photometry=Photometry(filters=filters_50))
        obs_60 = Observation(photometry=Photometry(filters=filters_60))

        spec = Parameters(
            redshift=0.3,
            sfh_dpl_alpha=1.5,
            sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=2.0,
            sfh_dpl_log_total_mass=10.5,
            met_logzsol=0.0,
            dust_tau_bc=0.1,
            dust_tau_diff=0.2,
        )

        model_50 = SEDModel(spec, synthetic_ssp, observation=obs_50)
        model_60 = SEDModel(spec, synthetic_ssp, observation=obs_60)

        sig_50 = model_50.compile_signature()
        sig_60 = model_60.compile_signature()

        assert sig_50 != sig_60, (
            "Different filter point counts should produce different signatures"
        )

    def test_second_model_gets_its_own_photometry_both_orders(self, synthetic_ssp):
        """Two models (optical-first, then IR) get their own photometry despite cache collision.

        Tests the user-visible defect directly: photometry computed with the wrong
        wavelength grid is silently served when signatures collide.

        Using issue #2068 exact spec:
        - DPL SFH with all params Fixed
        - 3000-10000 A SSP grid
        - Optical filters: centers [4000, 5750, 8250], widths [500, 750, 750]
        - IR filters: centers [5e4, 2e5, 1e6], widths [1e4, 5e4, 3e5]
          (entirely outside the SSP grid)
        - Both orderings (optical-then-IR AND IR-then-optical) in one test

        Before the fix: signatures identical, second model's photometry is
        silently computed with first model's wavelengths, producing wrong results.
        After the fix: signatures differ, each model gets correct photometry.
        """
        from tengri import Parameters

        spec = Parameters(
            sfh_dpl_alpha=1.5,
            sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=5.0,
            sfh_dpl_log_total_mass=10.0,
            sfh_dpl_age_gyr=5.0,
            met_logzsol=-0.5,
            dust_tau_bc=0.3,
            dust_tau_diff=0.2,
            dust_slope=-0.7,
            redshift=0.3,
        )

        # Optical filters: centers [4000, 5750, 8250], widths [500, 750, 750]
        opt_filters = [
            FilterCurve(
                wave=jnp.linspace(4000 - 250, 4000 + 250, 50),
                trans=jnp.ones(50) * 0.5,
                name="opt_4000",
            ),
            FilterCurve(
                wave=jnp.linspace(5750 - 375, 5750 + 375, 50),
                trans=jnp.ones(50) * 0.5,
                name="opt_5750",
            ),
            FilterCurve(
                wave=jnp.linspace(8250 - 375, 8250 + 375, 50),
                trans=jnp.ones(50) * 0.5,
                name="opt_8250",
            ),
        ]

        # IR filters: centers [5e4, 2e5, 1e6], widths [1e4, 5e4, 3e5]
        ir_filters = [
            FilterCurve(
                wave=jnp.linspace(5e4 - 5e3, 5e4 + 5e3, 50),
                trans=jnp.ones(50) * 0.5,
                name="ir_50k",
            ),
            FilterCurve(
                wave=jnp.linspace(2e5 - 2.5e4, 2e5 + 2.5e4, 50),
                trans=jnp.ones(50) * 0.5,
                name="ir_200k",
            ),
            FilterCurve(
                wave=jnp.linspace(1e6 - 1.5e5, 1e6 + 1.5e5, 50),
                trans=jnp.ones(50) * 0.5,
                name="ir_1m",
            ),
        ]

        # Order 1: optical first, then IR
        opt_obs = Observation(photometry=Photometry(filters=opt_filters))
        opt_model = SEDModel(spec, synthetic_ssp, observation=opt_obs)
        opt_phot = opt_model.predict_photometry(spec.get_fixed_values())

        ir_obs = Observation(photometry=Photometry(filters=ir_filters))
        ir_model = SEDModel(spec, synthetic_ssp, observation=ir_obs)
        ir_phot = ir_model.predict_photometry(spec.get_fixed_values())

        # Order 2: IR first, then optical (in same process to test cache collision)
        ir_obs_2 = Observation(photometry=Photometry(filters=ir_filters))
        ir_model_2 = SEDModel(spec, synthetic_ssp, observation=ir_obs_2)
        ir_phot_2 = ir_model_2.predict_photometry(spec.get_fixed_values())

        opt_obs_2 = Observation(photometry=Photometry(filters=opt_filters))
        opt_model_2 = SEDModel(spec, synthetic_ssp, observation=opt_obs_2)
        opt_phot_2 = opt_model_2.predict_photometry(spec.get_fixed_values())

        # Assertions: photometry must differ and IR must be zero
        assert not bool(jnp.all(opt_phot == ir_phot)), "Optical and IR photometry should differ"
        assert bool(jnp.all(ir_phot == 0.0)), (
            "IR photometry should be zero (bands outside SSP grid)"
        )
        assert bool(jnp.all(opt_phot > 0.0)), "Optical photometry should be positive"

        # Also check the reverse order
        assert not bool(jnp.all(opt_phot_2 == ir_phot_2)), (
            "Reversed order: Optical and IR photometry should differ"
        )
        assert bool(jnp.all(ir_phot_2 == 0.0)), "Reversed order: IR photometry should be zero"
        assert bool(jnp.all(opt_phot_2 > 0.0)), (
            "Reversed order: Optical photometry should be positive"
        )

    def test_shifted_filters_same_transmission_get_own_photometry(self, synthetic_ssp):
        """Shifting filter wavelengths by +500 A changes photometry.

        Same transmission and point count, but wavelengths differ by +500 A.
        This should produce different photometry after the fix.

        Before the fix: signatures identical, second model's photometry
        is silently computed with first model's wavelengths.
        """
        from tengri import Parameters

        spec = Parameters(
            sfh_dpl_alpha=1.5,
            sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=5.0,
            sfh_dpl_log_total_mass=10.0,
            sfh_dpl_age_gyr=5.0,
            met_logzsol=-0.5,
            dust_tau_bc=0.3,
            dust_tau_diff=0.2,
            dust_slope=-0.7,
            redshift=0.3,
        )

        # Optical filters at centers [4000, 5750, 8250], widths [500, 750, 750]
        opt_filters = [
            FilterCurve(
                wave=jnp.linspace(4000 - 250, 4000 + 250, 50),
                trans=jnp.ones(50) * 0.5,
                name="opt_4000",
            ),
            FilterCurve(
                wave=jnp.linspace(5750 - 375, 5750 + 375, 50),
                trans=jnp.ones(50) * 0.5,
                name="opt_5750",
            ),
            FilterCurve(
                wave=jnp.linspace(8250 - 375, 8250 + 375, 50),
                trans=jnp.ones(50) * 0.5,
                name="opt_8250",
            ),
        ]

        # Same filters but shifted +500 A: centers [4500, 6250, 8750], widths [500, 750, 750]
        shifted_filters = [
            FilterCurve(
                wave=jnp.linspace(4500 - 250, 4500 + 250, 50),
                trans=jnp.ones(50) * 0.5,
                name="shifted_4500",
            ),
            FilterCurve(
                wave=jnp.linspace(6250 - 375, 6250 + 375, 50),
                trans=jnp.ones(50) * 0.5,
                name="shifted_6250",
            ),
            FilterCurve(
                wave=jnp.linspace(8750 - 375, 8750 + 375, 50),
                trans=jnp.ones(50) * 0.5,
                name="shifted_8750",
            ),
        ]

        opt_obs = Observation(photometry=Photometry(filters=opt_filters))
        opt_model = SEDModel(spec, synthetic_ssp, observation=opt_obs)
        opt_phot = opt_model.predict_photometry(spec.get_fixed_values())

        shifted_obs = Observation(photometry=Photometry(filters=shifted_filters))
        shifted_model = SEDModel(spec, synthetic_ssp, observation=shifted_obs)
        shifted_phot = shifted_model.predict_photometry(spec.get_fixed_values())

        assert not bool(jnp.all(opt_phot == shifted_phot)), (
            "Shifted filter wavelengths should produce different photometry"
        )
