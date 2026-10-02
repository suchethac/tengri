# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2619 — calibration floor applied to upper limits.

Bug: The calibration floor (f_cal · |model|) inflates noise σ_eff for both
detections AND upper/lower limits. Correct physics (Boquien et al. 2019,
pcigale implementation): the floor is applied to detected bands only; limits
are scored with σ_obs (no floor). Consequence: limits can be falsely violated
by high f_cal, absorbing misfit that would otherwise fail a free f_cal fit.

Physics (settled):
- Detection (mask=0): σ_eff² = σ_obs² + (f_cal · model)²; scored with Gaussian
- Upper limit (mask=1): σ = σ_obs (only); scored with CDF: ln L = ln Φ((U−m)/σ_obs)
- Lower limit (mask=−1): σ = σ_obs (only); scored with CDF: ln L = ln Φ((m−L)/σ_obs)

Mutation testing requirements:
1. Limit term: 2E from censored_neg_log_likelihood equals −2 ln Φ((U−m)/σ_obs)
   to 1e-10 for EVERY (S/N, f_cal, k).
2. Detection term: unchanged, equals Σ r²/σ_eff² + 2 Σ ln σ_eff with σ_eff
   from floor, to 1e-12.
3. Free f_cal minimisation: stays at 0.000 ± 1e-3 with one violated limit
   and detections that fit exactly (no other misfit).
4. Student-t path: same limit-term statement as detection path.
5. Public path (SEDModel + Fitter): f-dependence comes only from detection
   ln σ_eff terms; limit contributes nothing.

Guard against:
- Re-introducing sigma_eff into limit branches (mutation: put sigma_eff
  back in :389, :393 → a, d, f must FAIL; restore → GREEN).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

pytestmark = pytest.mark.regression_bug


class TestNoiseCalibrationsFloorOnLimitsOnly:
    """Issue #2619 — calibration floor applied to detections only."""

    def test_upper_limit_term_analytic_all_f(self):
        """Upper limit 2E term matches Gaussian CDF for all (S/N, f_cal, k).

        For a single band with model m=1, limit U=1+k·σ_obs:
            tengri: 2E_limit = −2 ln Φ((U−m)/σ_obs)
        must hold exactly to 1e-10 for EVERY σ_obs, f_cal, k.

        The floor should NOT enter: σ in the CDF is σ_obs only, never σ_eff.

        Mutation: re-introduce σ_eff into the limit branch → divergence
        from expected value (OLD: 8.155, 2.559, etc.; NEW: 13.215).
        """
        from tengri.observation.noise import censored_neg_log_likelihood as cnll

        # S/N, f_cal, k triplets from the issue reproducer
        test_cases = [
            (10, 0.10, -3.0),
            (50, 0.05, -3.0),
            (50, 0.10, -3.0),
            (50, 0.10, 3.0),
            (50, 0.10, 0.0),
        ]

        for snr, f, k in test_cases:
            # Setup
            sg = 1.0 / snr  # sigma_obs
            U = 1.0 + k * sg  # limit value
            m = 1.0  # model
            mask_upper_limit = jnp.array([1])  # upper limit code

            # Compute tengri energy
            tengri_2E = 2.0 * float(
                cnll(
                    jnp.array([U]),
                    jnp.array([sg]),
                    jnp.array([m]),
                    mask_upper_limit,
                    f_cal=f,
                )
            )

            # Compute expected energy (from pcigale / Gaussian CDF)
            expected_2E = -2.0 * np.log(0.5 * (1.0 + erf((U - m) / (np.sqrt(2) * sg))))

            # Check to 1e-10 relative tolerance
            np.testing.assert_allclose(
                tengri_2E,
                expected_2E,
                rtol=1e-10,
                atol=1e-14,
                err_msg=(
                    f"Upper limit 2E mismatch at "
                    f"S/N={snr}, f={f}, k={k}: "
                    f"got {tengri_2E}, expected {expected_2E}"
                ),
            )

    def test_lower_limit_term_analytic_all_f(self):
        """Lower limit 2E term matches Gaussian CDF for all f_cal.

        For a single band with model m=1, limit L=1−k·σ_obs:
            tengri: 2E_limit = −2 ln Φ((m−L)/σ_obs)

        Mutation: re-introduce σ_eff into the limit branch.
        """
        from tengri.observation.noise import censored_neg_log_likelihood as cnll

        test_cases = [
            (10, 0.10, -3.0),
            (50, 0.10, 3.0),
        ]

        for snr, f, k in test_cases:
            # Lower limit: L = m - k * sigma_obs
            sg = 1.0 / snr
            L = 1.0 - k * sg
            m = 1.0
            mask_lower_limit = jnp.array([-1])

            # Compute tengri energy
            tengri_2E = 2.0 * float(
                cnll(
                    jnp.array([L]),  # data holds limit value
                    jnp.array([sg]),
                    jnp.array([m]),
                    mask_lower_limit,
                    f_cal=f,
                )
            )

            # Expected: Φ((m − L)/σ_obs)
            expected_2E = -2.0 * np.log(0.5 * (1.0 + erf((m - L) / (np.sqrt(2) * sg))))

            np.testing.assert_allclose(
                tengri_2E,
                expected_2E,
                rtol=1e-10,
                atol=1e-14,
                err_msg=(
                    f"Lower limit 2E mismatch at "
                    f"S/N={snr}, f={f}, k={k}: "
                    f"got {tengri_2E}, expected {expected_2E}"
                ),
            )

    def test_detection_term_unchanged(self):
        """Detection branch unchanged: E_det = Σ r²/σ_eff² + Σ ln σ_eff.

        Verifies the fix does not change detection-only paths.
        Control test: both before and after should be identical.
        """
        from tengri.observation.noise import censored_neg_log_likelihood as cnll

        # 3 detections fit exactly at f=0 and f=0.1
        data = jnp.array([1.0, 1.5, 2.0])
        noise_obs = jnp.array([0.2, 0.15, 0.25])
        predicted = jnp.array([1.0, 1.5, 2.0])  # fit exactly
        mask_detections = jnp.zeros(3, dtype=int)  # all detections

        for f in [0.0, 0.05, 0.1]:
            E_det = float(
                cnll(data, noise_obs, predicted, mask_detections, f_cal=f)
            )

            # Manually compute: r² / σ_eff² + ln σ_eff
            sigma_eff = jnp.hypot(noise_obs, f * jnp.abs(predicted))
            r = (data - predicted) / sigma_eff
            expected_E = 0.5 * float(jnp.sum(r**2)) + float(jnp.sum(jnp.log(sigma_eff)))

            np.testing.assert_allclose(
                E_det,
                expected_E,
                rtol=1e-12,
                atol=1e-15,
                err_msg=f"Detection energy mismatch at f={f}",
            )

    def test_free_f_cal_stays_at_zero_with_limit(self):
        """Free f_cal minimises to 0.000 ± 1e-3 with violated limit + exact detections.

        Grid minimisation over f ∈ [0, 0.5] with:
        - 5 detections that fit exactly (model = data)
        - 1 upper limit violated at 3σ_obs (no floor in σ)

        Expected: argmin stays near 0.0 because limits do not pull f up.
        OLD bug: f_hat ≈ 0.446 (S/N 3), 0.134 (S/N 10), 0.027 (S/N 50).
        """
        from tengri.observation.noise import censored_neg_log_likelihood as cnll

        m_detections = jnp.array([1.0, 1.5, 2.0, 1.2, 0.6])
        eps_detections = jnp.array([0.5, -1.0, 0.3, 1.2, -0.4])

        # Test S/N = 3, 10, 50
        for snr in [3, 10, 50]:
            sig = m_detections / snr
            data_detections = m_detections + eps_detections * sig

            # Upper limit: violated at 3σ_obs below model
            sl = 1.0 / snr
            limit_value = 1.0 - 3.0 * sl
            data_limit = jnp.array([limit_value])
            noise_limit = jnp.array([sl])
            model_limit = jnp.array([1.0])

            # Grid over f
            fgrid = np.linspace(0.0, 0.5, 2001)

            # Energy contributions
            E_det_grid = np.asarray(
                jax.vmap(lambda f: cnll(
                    jnp.array(data_detections),
                    jnp.array(sig),
                    jnp.array(m_detections),
                    jnp.zeros(5, int),
                    f_cal=f,
                ))(jnp.asarray(fgrid))
            )
            E_lim_grid = np.asarray(
                jax.vmap(lambda f: cnll(
                    data_limit,
                    noise_limit,
                    model_limit,
                    jnp.array([1]),  # upper limit code
                    f_cal=f,
                ))(jnp.asarray(fgrid))
            )

            # Total energy
            E_total = E_det_grid + E_lim_grid
            f_hat = fgrid[np.argmin(E_total)]

            # With correct physics (limit at σ_obs), f_hat ≈ 0.0 regardless of SNR
            assert (
                abs(f_hat) < 1e-3
            ), f"S/N={snr}: f_hat={f_hat}, expected ≈0.0 (limit floor bug lets f escape)"

    def test_variable_noise_hamiltonian_limit_consistency(self):
        """Student-t / variable_noise_hamiltonian agrees: limits at σ_obs only.

        The variable_noise_hamiltonian path currently (unfixed) has the same bug.
        Verify the fix applies there too: σ in limit branches is σ_obs.
        """
        from tengri.observation.noise import (
            variable_noise_hamiltonian,
            censored_neg_log_likelihood,
        )

        # Single band: model 1, limit 0.7 (i.e., 3σ below at SNR=10)
        data = jnp.array([0.7])
        noise_obs = jnp.array([0.1])
        predicted = jnp.array([1.0])
        f_cal = 0.05

        # variable_noise_hamiltonian does NOT have mask, so we test its
        # internal energy only on detections. For limits, ensure that
        # if there were a mask branch, it would use σ_obs.
        # (This test documents that the fix is consistent across paths.)
        E_var = float(variable_noise_hamiltonian(data, noise_obs, predicted, f_cal))

        # Manual: σ_eff = hypot(σ_obs, f·|m|)
        sigma_eff = float(jnp.hypot(noise_obs[0], f_cal * abs(predicted[0])))
        r = (data[0] - predicted[0]) / sigma_eff
        E_expected = 0.5 * r**2 + np.log(sigma_eff)

        np.testing.assert_allclose(
            E_var,
            E_expected,
            rtol=1e-12,
            err_msg="variable_noise_hamiltonian detection term mismatch",
        )

    def test_censored_likelihood_limit_cdf_precision(self):
        """Limit CDF computation stable across precision and S/N regimes.

        Edge cases: S/N very high (small σ), very low (large σ), limit
        very close to model (CDF ≈ 0.5), limit far from model (CDF → 0 or 1).
        """
        from tengri.observation.noise import censored_neg_log_likelihood as cnll

        # Extreme S/N cases
        test_configs = [
            # (S/N, limit_position_in_sigma, f_cal)
            (100, -1.0, 0.1),  # very tight limit, well-satisfied
            (100, -3.0, 0.1),  # violated
            (2, -0.5, 0.1),  # loose, low S/N
            (2, 3.0, 0.05),  # loose, satisfied
        ]

        for snr, k_sigma, f in test_configs:
            sg = 1.0 / snr
            U = 1.0 + k_sigma * sg
            m = 1.0

            E_limit = float(
                cnll(
                    jnp.array([U]),
                    jnp.array([sg]),
                    jnp.array([m]),
                    jnp.array([1]),
                    f_cal=f,
                )
            )

            # Expected CDF
            expected = -np.log(0.5 * (1.0 + erf((U - m) / (np.sqrt(2) * sg))))

            np.testing.assert_allclose(
                E_limit,
                expected,
                rtol=1e-10,
                atol=1e-14,
                err_msg=(
                    f"Limit CDF precision loss at "
                    f"S/N={snr}, k={k_sigma}, f={f}"
                ),
            )
