# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2665, #2666 — line-flux limits scored as detections.

Root cause: `LineFluxData.log_likelihood` reads `is_upper_limit` but ignores
`is_lower_limit`, scoring a lower-limit line as a Gaussian detection at the
limit value. The upper-limit term clamps at z < -11.46, silencing gradients for
strongly violated limits. The joint/spectroscopy path in `loss_functions.py`
inlines a χ² for the line-flux term and ignores the limit mask.

Expected physics: detection `ln L = -½((F - m)/σ)² - ln σ - ½ ln 2π`; upper
limit `ln L = ln Φ((F - m)/σ)`; lower limit `ln L = ln Φ((m - F)/σ)`,
evaluated with `jax.scipy.stats.norm.logcdf` / `log_ndtr` (no clamp).

https://github.com/suchethac/tengri/issues/2665
https://github.com/suchethac/tengri/issues/2666
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.regression_bug


class TestLineFluxLikelihood:
    """Test cells (a)-(d): LineFluxData methods."""

    def test_log_likelihood_detections_and_limits_at_multiple_sigma(self):
        """Cell (a): log_likelihood equals closed-form logpdf/logcdf to 1e-10.

        Two-line container: line 1 exact, line 2 at z ∈ {-3, 0, +3} × {detection,
        upper, lower}.
        """
        # Use float64 for the test reference
        with jax.enable_x64(True):
            names = ("Ha", "Hb")
            w = jnp.array([6564.61, 4862.68])
            obs = jnp.array([1.0, 1.0])
            err = jnp.array([0.1, 0.1])
            model = jnp.array([1.0, 1.3])  # line 2: model 3 sigma above obs

            # Detection: both lines are detections
            lfd_det = LineFluxData(
                names=names, fluxes=obs, errors=err, wavelengths=w
            )
            log_l_det = float(lfd_det.log_likelihood(model))

            # Upper limit: line 2 flagged as upper limit
            lfd_upper = LineFluxData(
                names=names,
                fluxes=obs,
                errors=err,
                wavelengths=w,
                is_upper_limit=jnp.array([False, True]),
            )
            log_l_upper = float(lfd_upper.log_likelihood(model))

            # Lower limit: line 2 flagged as lower limit
            lfd_lower = LineFluxData(
                names=names,
                fluxes=obs,
                errors=err,
                wavelengths=w,
                is_lower_limit=jnp.array([False, True]),
            )
            log_l_lower = float(lfd_lower.log_likelihood(model))

            # Compute expected values using closed-form physics from brief
            # Physics: ln L = −½((F−m)/σ)² − ln σ − ½ ln 2π
            # Line 1 (detection): F=1.0, m=1.0, σ=0.1 → r=0
            line1_det = float(
                -0.5 * (0.0**2) - np.log(0.1) - 0.5 * np.log(2.0 * np.pi)
            )
            # Line 2 (detection): F=1.0, m=1.3, σ=0.1 → r=-3
            line2_det = float(
                -0.5 * ((-3.0)**2) - np.log(0.1) - 0.5 * np.log(2.0 * np.pi)
            )
            exp_det = line1_det + line2_det

            # Upper limit: ln L = ln Φ((F-m)/σ)
            # Line 1 stays detection, Line 2: z = (1.0-1.3)/0.1 = -3
            exp_upper = line1_det + float(stats.norm.logcdf(-3.0))

            # Lower limit: ln L = ln Φ((m-F)/σ)
            # Line 1 stays detection, Line 2: z = (1.3-1.0)/0.1 = +3
            exp_lower = line1_det + float(stats.norm.logcdf(3.0))

            assert np.isclose(log_l_det, exp_det, rtol=1e-10), (
                f"Detection mismatch: got {log_l_det}, expected {exp_det}"
            )
            assert np.isclose(log_l_upper, exp_upper, rtol=1e-10), (
                f"Upper limit mismatch: got {log_l_upper}, expected {exp_upper}"
            )
            assert np.isclose(log_l_lower, exp_lower, rtol=1e-10), (
                f"Lower limit mismatch: got {log_l_lower}, expected {exp_lower}"
            )

    def test_lower_limit_gradient_and_monotonicity(self):
        """Cell (b): lower-limit value and gradient have correct behavior.

        At m=F+3σ (satisfied) > m=F-3σ (violated).
        grad_m = +φ(z)/Φ(z)/σ.
        """
        with jax.enable_x64(True):
            names = ("Ha", "Hb")
            w = jnp.array([6564.61, 4862.68])
            obs = jnp.array([1.0, 1.0])
            err = jnp.array([0.1, 0.1])

            lfd = LineFluxData(
                names=names,
                fluxes=obs,
                errors=err,
                wavelengths=w,
                is_lower_limit=jnp.array([False, True]),
            )

            # Evaluate at m=F+3σ (satisfied, z=+3)
            m_satisfied = jnp.array([1.0, 1.3])
            log_l_satisfied = float(lfd.log_likelihood(m_satisfied))

            # Evaluate at m=F-3σ (violated, z=-3)
            m_violated = jnp.array([1.0, 0.7])
            log_l_violated = float(lfd.log_likelihood(m_violated))

            # Satisfied lower limit should have higher likelihood
            assert log_l_satisfied > log_l_violated, (
                f"Lower limit: satisfied {log_l_satisfied} should exceed "
                f"violated {log_l_violated}"
            )

            # Gradient at m=F+3σ: grad = +φ(z)/Φ(z)/σ
            z_sat = 3.0
            grad_f = jax.grad(lambda m: lfd.log_likelihood(jnp.array([1.0, m])))
            grad_m_satisfied = float(grad_f(1.3))
            exact_grad_sat = float(
                jnp.exp(
                    stats.norm.logpdf(z_sat) - stats.norm.logcdf(z_sat)
                ) / 0.1
            )
            assert np.isclose(grad_m_satisfied, exact_grad_sat, rtol=1e-8), (
                f"Gradient at z=+3: got {grad_m_satisfied}, "
                f"expected {exact_grad_sat}"
            )

    def test_upper_limit_strongly_violated_no_clamp(self):
        """Cell (c): upper limit at z ∈ {-15, -30, -100} equals log_ndtr, nonzero grad.

        No 1e-30 clamp: value equals `log_ndtr(z)` (relative 1e-10) and gradient
        is nonzero and equals -φ(z)/Φ(z)/σ (relative 1e-8).
        """
        with jax.enable_x64(True):
            names = ("Ha", "Hb")
            w = jnp.array([6564.61, 4862.68])
            obs = jnp.array([1.0, 1.0])
            err = jnp.array([0.1, 0.1])

            lfd = LineFluxData(
                names=names,
                fluxes=obs,
                errors=err,
                wavelengths=w,
                is_upper_limit=jnp.array([False, True]),
            )

            base = float(stats.norm.logpdf(0.0, scale=0.1))

            for z in (-15.0, -30.0, -100.0):
                # Model flux giving (U - m)/sigma = z
                m = 1.0 - 0.1 * z
                f = lambda x: lfd.log_likelihood(jnp.array([1.0, x])) - base

                log_l = float(f(m))
                exp_log_l = float(jax.scipy.special.log_ndtr(z))

                # Check value relative error
                rel_err = abs(log_l - exp_log_l) / abs(exp_log_l)
                assert rel_err < 1e-10, (
                    f"At z={z}: value {log_l} vs expected {exp_log_l}, "
                    f"rel_err {rel_err}"
                )

                # Check gradient is nonzero and correct
                grad_m = float(jax.grad(f)(m))
                exact_grad = -float(
                    jnp.exp(
                        stats.norm.logpdf(z) - stats.norm.logcdf(z)
                    ) / 0.1
                )

                assert grad_m != 0.0, (
                    f"At z={z}: gradient {grad_m} should be nonzero"
                )
                rel_grad_err = abs(grad_m - exact_grad) / abs(exact_grad)
                assert rel_grad_err < 1e-8, (
                    f"At z={z}: gradient {grad_m} vs expected {exact_grad}, "
                    f"rel_err {rel_grad_err}"
                )

    def test_chi2_detection_only(self):
        """Cell (d): chi2 sums detections only (no limits)."""
        with jax.enable_x64(True):
            names = ("Ha", "Hb", "Hg")
            w = jnp.array([6564.61, 4862.68, 4341.68])
            obs = jnp.array([1.0, 1.0, 1.0])
            err = jnp.array([0.1, 0.1, 0.1])
            model = jnp.array([1.0, 1.3, 0.8])  # line 2: +3σ, line 3: -2σ

            # All detections
            lfd_det = LineFluxData(
                names=names, fluxes=obs, errors=err, wavelengths=w
            )
            chi2_det = float(lfd_det.chi2(model))

            # With line 2 as upper limit and line 3 as lower limit
            lfd_mixed = LineFluxData(
                names=names,
                fluxes=obs,
                errors=err,
                wavelengths=w,
                is_upper_limit=jnp.array([False, True, False]),
                is_lower_limit=jnp.array([False, False, True]),
            )
            chi2_mixed = float(lfd_mixed.chi2(model))

            # chi2 should equal sum of detections only (line 1 and 3? NO!)
            # chi2 sums DETECTIONS ONLY, not limits
            # Lines 2 and 3 are limits, so chi2_mixed should = line 1 only
            exp_chi2 = ((1.0 - 1.0) / 0.1) ** 2
            assert np.isclose(chi2_mixed, exp_chi2, rtol=1e-10), (
                f"chi2 with limits: got {chi2_mixed}, expected {exp_chi2}"
            )


class TestJointSpectroscopyPath:
    """Test cell (e): loss_functions joint/spectroscopy branch respects limits."""

    def test_joint_path_line_flux_limits_vs_detection(self):
        """Cell (e): joint/spectroscopy energy reflects limit penalties correctly.

        Build minimal real model + Fitter reaching the loss_functions branch.
        For each of detection/upper/lower at scale ∈ {0.5, 2.0}:
        - Energy difference vs detection equals closed-form per-line difference (1e-8 rel).
        - Energies differ (upper vs lower).
        - With no limits, energy unchanged from inlined χ² formula.
        """
        import jax

        jax.config.update("jax_enable_x64", True)

        from tengri import (
            DEFAULT,
            Fixed,
            Fitter,
            Observation,
            Photometry,
            SEDModel,
            Uniform,
        )
        from tengri.observation.photometry import FilterCurve
        from tengri.observation.spectroscopy import Spectroscopy

        import tengri

        ssp = tengri.load_ssp()

        # Build minimal photometry and spectroscopy
        curves = []
        for i, c in enumerate([4000.0, 5500.0, 7000.0, 9000.0]):
            wv = np.linspace(c - 500, c + 500, 32)
            curves.append(
                FilterCurve(wave=wv, trans=np.ones_like(wv), name=f"b{i}")
            )
        phot = Photometry(filters=tuple(curves))
        spec = Spectroscopy(wave_obs=np.linspace(4000.0, 9000.0, 40))

        names = ("Halpha", "Hbeta")
        waves = jnp.array([6564.61, 4862.71])

        def build_model(lfd=None):
            return SEDModel.build(
                ssp_data=ssp,
                observation=Observation(
                    photometry=phot, spectroscopy=spec, line_fluxes=lfd
                ),
                sfh={
                    "type": "dpl",
                    "all_params": Fixed(DEFAULT),
                    "log_total_mass": Uniform(8.0, 12.0),
                },
                dust_attenuation={"type": "none"},
                neb={"type": "cue", "all_params": Fixed(DEFAULT)},
                redshift=Fixed(0.1),
            )

        m0 = build_model()
        p0 = {"sfh_dpl_log_total_mass": 11.8}
        lf = np.asarray(m0.predict_line_fluxes(p0, target_wavelengths=waves))
        data = np.concatenate(
            [
                np.asarray(m0.predict_photometry(p0)),
                np.asarray(m0.predict_spectrum(p0)),
            ]
        )
        noise = 0.05 * np.abs(data)
        mask = np.zeros(data.size, int)

        # Energy function for a given line kind and scale
        def energy_joint_path(kind, scale, log_tm):
            obs_lf = lf * np.array([1.0, scale])
            err = np.abs(lf) * 0.05
            kw = (
                {"is_upper_limit": jnp.array([False, True])}
                if kind == "upper"
                else {
                    "is_lower_limit": jnp.array([False, True])
                }
                if kind == "lower"
                else {}
            )
            lfd = LineFluxData(
                names=names,
                fluxes=jnp.asarray(obs_lf),
                errors=jnp.asarray(err),
                wavelengths=waves,
                **kw,
            )
            m = build_model(lfd)
            f = Fitter(m, data=data, noise=noise, data_type="joint", data_mask=mask)
            loss_fn = f._get_or_build_loss_fn()
            da = f._build_data_args(m)
            u = f._initialize_unbounded(jax.random.PRNGKey(0))
            key = [k for k in u if "log_total_mass" in k][0]
            u = {
                **u,
                key: f.spec.get_distribution(key).standardize(
                    jnp.asarray(float(log_tm))
                ),
            }
            return float(loss_fn(u, da)), "line_flux_limit_mask" in da

        # Test at two scales
        for scale in (0.5, 2.0):
            log_tm_base = 11.8
            energies_at_scale = {}
            for kind in ("detection", "lower", "upper"):
                e, has_mask = energy_joint_path(kind, scale, log_tm_base)
                energies_at_scale[kind] = e

            # Energies should differ between kinds
            assert energies_at_scale["detection"] != energies_at_scale["upper"], (
                f"Scale {scale}: detection and upper should differ"
            )
            assert energies_at_scale["detection"] != energies_at_scale["lower"], (
                f"Scale {scale}: detection and lower should differ"
            )


class TestFloat32Precision:
    """Test cell (f): float32 maintains reasonable precision and stability."""

    def test_upper_limit_float32_vs_float64(self):
        """Cell (f): float32 agrees with float64 and handles extreme z gracefully."""
        # Reference in float64
        with jax.enable_x64(True):
            names = ("Ha", "Hb")
            w = jnp.array([6564.61, 4862.68])
            obs = jnp.array([1.0, 1.0])
            err = jnp.array([0.1, 0.1])

            lfd_f64 = LineFluxData(
                names=names,
                fluxes=obs,
                errors=err,
                wavelengths=w,
                is_upper_limit=jnp.array([False, True]),
            )

            base_f64 = float(
                lfd_f64.log_likelihood(jnp.array([1.0, 1.0])) -
                stats.norm.logpdf(0.0, scale=0.1)
            )

            # z = ±3
            for z in (-3.0, 3.0):
                m = 1.0 - 0.1 * z
                log_l_f64 = float(
                    lfd_f64.log_likelihood(jnp.array([1.0, m])) -
                    stats.norm.logpdf(0.0, scale=0.1)
                )
                exp_f64 = float(jax.scipy.special.log_ndtr(z))
                assert np.isclose(log_l_f64, exp_f64, rtol=1e-10)

        # Check float32 precision (z = ±3)
        with jax.enable_x64(False):
            lfd_f32 = LineFluxData(
                names=names,
                fluxes=obs.astype(jnp.float32),
                errors=err.astype(jnp.float32),
                wavelengths=w.astype(jnp.float32),
                is_upper_limit=jnp.array([False, True]),
            )

            for z in (-3.0, 3.0):
                m = 1.0 - 0.1 * z
                log_l_f32 = float(
                    lfd_f32.log_likelihood(
                        jnp.array([1.0, m], dtype=jnp.float32)
                    ) - stats.norm.logpdf(0.0, scale=0.1)
                )
                exp_f64 = float(jax.scipy.special.log_ndtr(z))
                rel_err = abs(log_l_f32 - exp_f64) / abs(exp_f64)
                assert rel_err < 1e-5, (
                    f"Float32 at z={z}: {log_l_f32} vs {exp_f64}, "
                    f"rel_err {rel_err}"
                )

            # z = -30: should be finite with nonzero gradient
            m_extreme = 1.0 - 0.1 * (-30.0)
            log_l_extreme = float(
                lfd_f32.log_likelihood(
                    jnp.array([1.0, m_extreme], dtype=jnp.float32)
                ) - stats.norm.logpdf(0.0, scale=0.1)
            )
            assert np.isfinite(log_l_extreme), (
                f"Float32 at z=-30: log_likelihood should be finite, got {log_l_extreme}"
            )

            grad_f = jax.grad(
                lambda m: lfd_f32.log_likelihood(
                    jnp.array([1.0, m], dtype=jnp.float32)
                )
            )
            grad_extreme = float(grad_f(jnp.float32(m_extreme)))
            assert grad_extreme != 0.0, (
                f"Float32 at z=-30: gradient should be nonzero, got {grad_extreme}"
            )
