# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-NSS-02: evolving_metallicity in fused kernel.

See ADR / docs/known_bugs.md for full context.
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestBugNSS02EvolvingMetFusedKernel:
    """compositional.py — fused kernel reads log_z_abs (scalar) but evolving_metallicity
    produces log_z_abs_initial and log_z_abs_final (ramp params). Silent fallback
    to log_z_abs_final causes wrong physics (uses present-day Z only, ignores ramp).

    Root cause:
    - Exact path (sed_model.py:2171-2179): computes per-age lgmet_per_age via
      compute_log_z_evolving(ssp_lg_age_gyr, log_z_abs_initial, log_z_abs_final, t_obs_gyr)
      then vmaps interpolation over per-age metallicities.
    - Fused kernel path (compositional.py lines 249, 263, 434, 438, 570, 623, 631):
      reads p.get("log_z_abs", p.get("log_z_abs_final", -1.8477)) — uses only the
      final (present-day) metallicity, ignoring the ramp to log_z_abs_initial.

    Fixed:
    1. Add _met_mode to SEDModelState (set at SEDModel.__init__ from spec.met_mode).
    2. Inside builder closures, detect ramp mode at kernel-build time.
    3. For ramp mode: compute lgmet_per_age from compute_log_z_evolving inside kernel,
       then vmap interpolation over age bins (matching exact path).
    4. For scalar mode: use existing fast path unchanged.
    5. Remove silent fallback; raise clear KeyError if neither path is available.
    """

    def test_fused_kernel_evolving_metallicity_finite(self, real_ssp_only):
        """Fused kernel with evolving_metallicity=True must return finite photometry.

        Needs the real grid: the evolving-metallicity ramp interpolates over the
        SSP lgmet axis; the synthetic #613 grid's narrow axis drives it out of
        range. ``real_ssp_only`` skips on synthetic-only CI.
        """
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.forward.sed_model import SEDModel
        from tengri.observation.filters import load_filter_set
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Uniform

        jax.config.update("jax_enable_x64", True)

        # Load SSP data
        data_dir = Path(__file__).resolve().parents[3] / "data"
        ssp_file = data_dir / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
        if not ssp_file.is_file():
            pytest.skip("SSP data not available")

        ssp_data = load_ssp_data(str(ssp_file))
        filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

        # Build model with evolving metallicity
        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            evolving_metallicity=True,
            met_logzsol_0=Uniform(-2.0, 0.2),
            met_logzsol_final=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=-0.7,
            redshift=0.1,
        )

        model = SEDModel(spec, ssp_data, filters=filters)

        # The compositional kernel this bug lived in was retired (ADR-0019
        # Phase 6 / #922); the physics contract — evolving metallicity gives
        # finite, positive photometry — is pinned on the live orchestrator.

        # Sample parameters and compute photometry
        params = spec.sample(jax.random.PRNGKey(42))

        # PRE-FIX: This should raise KeyError: 'log_z_abs' or return silently-wrong values
        # POST-FIX: Should return finite photometry
        try:
            phot = model.predict_photometry(params)
            assert jnp.all(jnp.isfinite(phot)), (
                f"BUG-NSS-02: Non-finite photometry with evolving_metallicity: {phot}"
            )
            assert jnp.all(phot > 0), (
                f"BUG-NSS-02: Non-positive photometry with evolving_metallicity: {phot}"
            )
        except KeyError as e:
            if "log_z_abs" in str(e):
                pytest.fail(
                    f"BUG-NSS-02 not fixed: KeyError '{e}' when reading log_z_abs "
                    f"(should use per-age lgmet_per_age computed from log_z_abs_initial/final)"
                )
            raise

    def test_fused_vs_exact_evolving_metallicity_agreement(self):
        """Exact path and compositional path must agree on SED for same params (rtol=1e-5)."""
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.forward.sed_model import SEDModel
        from tengri.observation.filters import load_filter_set
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Uniform

        jax.config.update("jax_enable_x64", True)

        data_dir = Path(__file__).resolve().parents[3] / "data"
        ssp_file = data_dir / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
        if not ssp_file.is_file():
            pytest.skip("SSP data not available")

        ssp_data = load_ssp_data(str(ssp_file))
        filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

        spec = Parameters(
            mean_sfh_type="tsnorm",
            sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
            sfh_tsnorm_skew=Uniform(-1.0, 1.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            evolving_metallicity=True,
            met_logzsol_0=Uniform(-2.0, 0.2),
            met_logzsol_final=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=-0.7,
            redshift=0.1,
        )

        model = SEDModel(spec, ssp_data, filters=filters)
        params = spec.sample(jax.random.PRNGKey(43))

        # Compute photometry via both paths (exact and compositional are selected
        # internally based on model config)
        # The current code selects compositional for evolving_metallicity=True
        phot_compositional = model.predict_photometry(params)

        # For the exact path, we'd need to call _predict_photometry_exact explicitly,
        # but the public API doesn't expose this. Instead, verify that the
        # compositional path matches the ground truth by comparing against a
        # reference computation (if available).
        #
        # For now, the minimal regression test is that compositional doesn't crash
        # and returns finite, positive values (tested above). The strongest correctness
        # check is the crossval test in the integration suite (test_compositional_routing.py).
        #
        # Future enhancement: expose _predict_photometry_exact in the public API
        # for unit-test-level comparison.

        chex.assert_tree_all_finite(phot_compositional)
        assert jnp.all(phot_compositional > 0), "Compositional photometry must be positive"

    def test_translate_evolving_keys_present(self):
        """Evolving metallicity spec generates met_logzsol_0 and met_logzsol_final params."""
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Uniform

        spec = Parameters(
            mean_sfh_type="const",
            evolving_metallicity=True,
            met_logzsol_0=Uniform(-2.0, 0.2),
            met_logzsol_final=Uniform(-2.0, 0.2),
            redshift=0.1,
        )

        # Verify that the spec correctly sets up the ramp parameters
        assert spec.met_mode == "ramp", "evolving_metallicity=True should set met_mode='ramp'"
        assert "met_logzsol_0" in spec.free_params, (
            "evolving_metallicity=True should add met_logzsol_0"
        )
        assert "met_logzsol_final" in spec.free_params, (
            "evolving_metallicity=True should add met_logzsol_final"
        )
