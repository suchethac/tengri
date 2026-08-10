# SPDX-License-Identifier: BSD-3-Clause
"""Test precompute↔runtime equivalence for AGN-nebular emitters (BLR, NLR-Gaussian).

Verifies that precompute lookups return per-filter photometry matching the
runtime full-wavelength evaluation to within 1e-3 relative tolerance.

.. warning::

   **Quarantined — this file has not run for some time (#1660).** All three
   tests skipped with "SSP data not available", which was false: the SSP data
   was present. They call ``SEDModel(base_spec, filter_list, ...)``, but the
   second positional is now ``ssp_data`` (order: ``spec, ssp_data, filters``),
   so the filter list landed in the SSP slot, construction raised
   ``AttributeError``, and ``except (FileNotFoundError, AttributeError)``
   reported it as missing data.

   It is skipped rather than repaired in place because updating it mechanically
   would test the wrong thing twice over. The ``precompute=True/False`` axis it
   toggles is documented as legacy and explicitly *not* the LUT path — that is
   ``approx=WavePrecomp()``. And on the modern axis the naive rewrite is
   vacuous: measured, an ``analytic`` BLR changes the photometry in the sixth
   significant figure, while the exact-vs-LUT difference is identical to four
   significant figures with and without it. The comparison would be dominated
   by the stellar continuum and would say nothing about the emitter.

   A real replacement needs a configuration where the line emitter dominates a
   band, and must assert that dominance before comparing. See #1660.
"""

import numpy as np
import pytest

pytestmark = [
    pytest.mark.contract,
    pytest.mark.skip(
        reason=(
            "quarantined: stale API made this silently inert, and updating it "
            "mechanically would compare continuum to continuum — see #1660"
        )
    ),
]

from tengri import Parameters, SEDModel
from tengri.config import AGNConfig


class TestAGNNebularPrecomputeEquivalence:
    """Test that precompute and runtime paths agree on AGN-nebular emitters."""

    @pytest.fixture
    def base_spec(self):
        """Default Parameters with basic settings."""
        return Parameters(redshift=0.1)

    @pytest.fixture
    def filter_list(self):
        """Simple mock filter list for testing."""
        # Import from actual filter library
        from tengri.observation.filters import load_filter_set

        filter_names = ["sdss_u", "sdss_g", "sdss_r"]
        return load_filter_set(filter_names)

    def test_default_off_regression_blr_disabled(self, base_spec, filter_list):
        """With BLR disabled, photometry is independent of BLR precompute.

        This is a regression test: enabling the precompute infrastructure
        should not change results when the feature is off.
        """
        cfg_off = AGNConfig(agn_blr_enabled=False)

        # Build two models: one with precompute, one without.
        # Both should give identical results when BLR is disabled.
        try:
            model_no_precomp = SEDModel(
                base_spec, filter_list, agn_config=cfg_off, precompute=False
            )
            model_with_precomp = SEDModel(
                base_spec, filter_list, agn_config=cfg_off, precompute=True
            )
        except (FileNotFoundError, AttributeError):
            # Expected if SSP data missing
            pytest.skip("SSP data not available")
            return

        # Create a simple input: minimal SFR, solar metallicity
        params = {
            "sfr_tsnorm": 1.0,
            "log_z_abs": -1.8477,
            "tau_v": 0.1,
            "agn_log_lbol": 11.0,
            "agn_alpha": -1.5,
            "agn_blr_cf": 0.1,
        }

        try:
            phot_no_precomp = model_no_precomp(params)
            phot_with_precomp = model_with_precomp(params)
        except (ValueError, RuntimeError, KeyError):
            # Expected if model incomplete
            pytest.skip("Model evaluation not fully implemented")
            return

        # With BLR disabled, both should be bit-identical
        np.testing.assert_array_equal(
            phot_no_precomp,
            phot_with_precomp,
            err_msg="BLR-disabled photometry should be identical regardless of precompute",
        )

    def test_blr_precompute_runtime_equivalence(self, base_spec, filter_list):
        """BLR precompute lookup should agree with runtime evaluation to 1e-3 rel tol.

        This is the load-bearing equivalence test. If precompute and runtime
        disagree significantly, the precompute consumer is incorrect.
        """
        cfg = AGNConfig(agn_blr_enabled=True)

        # Build models with and without precompute
        try:
            model_runtime = SEDModel(base_spec, filter_list, agn_config=cfg, precompute=False)
            model_precomp = SEDModel(base_spec, filter_list, agn_config=cfg, precompute=True)
        except (FileNotFoundError, AttributeError):
            pytest.skip("SSP data not available")
            return

        params = {
            "sfr_tsnorm": 1.0,
            "log_z_abs": -1.8477,
            "tau_v": 0.1,
            "agn_log_lbol": 11.0,
            "agn_alpha": -1.5,
            "agn_blr_cf": 0.1,
            "agn_fe2_strength": 0.5,
        }

        try:
            phot_runtime = model_runtime(params)
            phot_precomp = model_precomp(params)
        except (ValueError, RuntimeError, KeyError):
            pytest.skip("Model evaluation not fully implemented")
            return

        # Compute relative error
        # Avoid division-by-zero by using maximum of abs values
        numerator = np.abs(phot_runtime - phot_precomp)
        denominator = np.maximum(np.abs(phot_runtime), np.abs(phot_precomp))
        denominator = np.maximum(denominator, 1e-30)  # Avoid exact zero
        rel_error = numerator / denominator

        # Report max relative error for debugging
        max_rel_error = np.max(rel_error)
        print(f"BLR max rel error: {max_rel_error:.4e}")

        # Assert agreement to 1e-3 relative tolerance
        np.testing.assert_array_less(
            rel_error,
            1e-3,
            err_msg=f"BLR precompute↔runtime disagreement exceeds 1e-3; "
            f"max rel error = {max_rel_error:.4e}",
        )

    def test_nlr_gaussian_precompute_runtime_equivalence(self, base_spec, filter_list):
        """NLR-Gaussian precompute should agree with runtime to 1e-3 rel tol."""
        cfg = AGNConfig(agn_nlr_gaussian_enabled=True)

        try:
            model_runtime = SEDModel(base_spec, filter_list, agn_config=cfg, precompute=False)
            model_precomp = SEDModel(base_spec, filter_list, agn_config=cfg, precompute=True)
        except (FileNotFoundError, AttributeError):
            pytest.skip("SSP data not available")
            return

        params = {
            "sfr_tsnorm": 1.0,
            "log_z_abs": -1.8477,
            "tau_v": 0.1,
            "agn_log_lbol": 11.0,
            "agn_alpha": -1.5,
            "agn_nlr_cf": 0.2,
        }

        try:
            phot_runtime = model_runtime(params)
            phot_precomp = model_precomp(params)
        except (ValueError, RuntimeError, KeyError):
            pytest.skip("Model evaluation not fully implemented")
            return

        # Compute relative error
        numerator = np.abs(phot_runtime - phot_precomp)
        denominator = np.maximum(np.abs(phot_runtime), np.abs(phot_precomp))
        denominator = np.maximum(denominator, 1e-30)
        rel_error = numerator / denominator

        max_rel_error = np.max(rel_error)
        print(f"NLR-Gaussian max rel error: {max_rel_error:.4e}")

        np.testing.assert_array_less(
            rel_error,
            1e-3,
            err_msg=f"NLR-Gaussian precompute↔runtime disagreement exceeds 1e-3; "
            f"max rel error = {max_rel_error:.4e}",
        )
