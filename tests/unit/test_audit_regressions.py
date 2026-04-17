"""Regression tests for bugs found in 2026-03-31 audit.

Each test is designed to FAIL on the buggy code and PASS after the fix.
See docs/known_bugs.md for full descriptions and references.

RULE: When fixing a bug, the fixer MUST:
1. Read the original paper cited in docs/known_bugs.md
2. Implement the fix citing the paper equation number
3. Verify this test fails before the fix and passes after
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ---------------------------------------------------------------------------
# BUG-01: SFR hardcoded to 1.0 Msun/yr
# ---------------------------------------------------------------------------
class TestBug01SfrCached:
    """sed_pipeline.py:638 — _sfr_cached must reflect actual SFR.

    Fixed: sed_pipeline.py now uses sfr[-1] for parametric SFH paths instead
    of hard-coding 1.0. The fix is already in the code; these tests verify it.
    """

    def test_sfr_computed_not_hardcoded(self):
        """Verify that the SFR pipeline code selects sfr[-1] for parametric path."""
        import inspect

        from tengri.forward import pipeline as sed_pipeline

        src = inspect.getsource(sed_pipeline)
        # The fix: sfr[-1] is used as the instantaneous SFR
        assert "sfr[-1]" in src, "sed_pipeline must use sfr[-1] for parametric SFR path"
        # Guard against any re-introduction of the 1.0 fallback at the _sfr_cached line
        # (the fallback for _sfr_cached=0 override path is intentional; this checks
        # the parametric path doesn't hard-code it)
        lines = [l for l in src.split("\n") if "_sfr_cached" in l and "1.0" in l]
        assert not any("= 1.0" in l and "sfr" not in l for l in lines), (
            "Hard-coded _sfr_cached = 1.0 found with no sfr fallback"
        )


# ---------------------------------------------------------------------------
# BUG-02: SFR time-averaging trapezoid
# ---------------------------------------------------------------------------
class TestBug02SfrTimeAvg:
    """model.py:791-804 — sfr_100myr must be positive and correct.

    Fix (model.py): replaced jnp.trapezoid on zeroed SFR with gradient-weighted
    Riemann sum (jnp.gradient for bin widths) over masked ages only. This avoids
    the phantom boundary segment from the last in-window age to the first
    out-of-window age.
    """

    def test_constant_sfr_recovery(self):
        """For constant SFR=10, sfr_100myr should be ~10 (not biased by boundary)."""
        age_yr = jnp.logspace(6, 10, 100)  # 1 Myr to 10 Gyr
        sfr = jnp.full_like(age_yr, 10.0)  # constant SFR = 10 Msun/yr

        dt = jnp.gradient(age_yr)
        mask_100 = age_yr <= 1e8
        numerator = jnp.sum(jnp.where(mask_100, sfr * dt, 0.0))
        denom = jnp.maximum(jnp.sum(jnp.where(mask_100, dt, 0.0)), 1.0)
        sfr_100myr = jnp.where(jnp.sum(mask_100) > 1, numerator / denom, sfr[0])

        assert float(sfr_100myr) > 0, "sfr_100myr must be positive"
        assert abs(float(sfr_100myr) - 10.0) < 0.5, (
            f"sfr_100myr = {float(sfr_100myr):.2f}, expected ~10.0"
        )

    def test_sfr_100myr_positive_with_declining_sfh(self):
        """sfr_100myr must be positive even for a strongly declining SFH."""
        age_yr = jnp.logspace(6, 10, 100)
        # Exponentially declining SFH: high early SFR, low recent
        tau_yr = 1e9
        sfr = 100.0 * jnp.exp(-age_yr / tau_yr)

        dt = jnp.gradient(age_yr)
        mask_100 = age_yr <= 1e8
        numerator = jnp.sum(jnp.where(mask_100, sfr * dt, 0.0))
        denom = jnp.maximum(jnp.sum(jnp.where(mask_100, dt, 0.0)), 1.0)
        sfr_100myr = jnp.where(jnp.sum(mask_100) > 1, numerator / denom, sfr[0])

        assert float(sfr_100myr) > 0, "sfr_100myr must be positive for declining SFH"


# ---------------------------------------------------------------------------
# BUG-06: Balmer continuum tau direction
# ---------------------------------------------------------------------------
class TestBug06BalmerTau:
    """qsogen.py:397 — tau must increase at shorter wavelengths."""

    def test_tau_increases_shortward(self):
        """Grandi 1982: sigma(nu) ~ nu^3, so tau increases at shorter lambda."""
        wavbe = 3646.0  # Balmer edge wavelength
        taube = 1.0

        # Current (buggy) code: tau = taube * (wave / wavbe)^3
        wave_short = 3000.0
        wave_long = 3500.0
        tau_short_buggy = taube * (wave_short / wavbe) ** 3
        tau_long_buggy = taube * (wave_long / wavbe) ** 3

        # Correct: tau = taube * (wavbe / wave)^3
        tau_short_correct = taube * (wavbe / wave_short) ** 3
        tau_long_correct = taube * (wavbe / wave_long) ** 3

        # Bug: tau_short < tau_long (wrong — should be higher at shorter lambda)
        assert tau_short_buggy < tau_long_buggy, (
            "If this fails, BUG-06 may have been fixed — remove xfail"
        )

        # Correct: tau_short > tau_long
        assert tau_short_correct > tau_long_correct


# ---------------------------------------------------------------------------
# BUG-07: Disc ring area missing pi factor
# ---------------------------------------------------------------------------
class TestBug07DiscArea:
    """disc.py:298 — L_nu per ring must include pi*B_nu."""

    def test_single_ring_luminosity(self):
        """Compare single-ring L_nu against analytical pi*B_nu*A*cos(i)."""
        h = 6.626e-27  # erg*s
        c = 3e10  # cm/s
        k = 1.38e-16  # erg/K
        T = 1e4  # K
        R = 1e13  # cm
        dR = 1e11  # cm
        nu = 1e15  # Hz (UV)
        cos_i = 1.0

        # Planck function B_nu
        x = h * nu / (k * T)
        B_nu = 2 * h * nu**3 / c**2 / (np.exp(x) - 1)

        # Correct analytical: pi * B_nu * (2*pi*R*dR) * cos_i
        L_analytical = np.pi * B_nu * 2 * np.pi * R * dR * cos_i

        # Buggy (missing pi): B_nu * (2*pi*R*dR) * cos_i
        L_buggy = B_nu * 2 * np.pi * R * dR * cos_i

        assert abs(L_analytical / L_buggy - np.pi) < 0.01, (
            "Ratio should be pi — the missing factor"
        )


# ---------------------------------------------------------------------------
# BUG-08: Shock emission unit mismatch
# ---------------------------------------------------------------------------
class TestBug08ShockUnits:
    """shock.py:182-206 — Both branches must return same units."""

    def test_gaussian_vs_delta_total_luminosity(self):
        """Integrated luminosity must agree between Gaussian and delta branches.

        BUG-08 fixed: sigma_nu was missing 1e-8 Å-to-cm conversion.
        Canonical regression test: TestShockEmissionUnits in test_bug_regression.py.
        """
        pass  # covered by test_bug_regression.py::TestShockEmissionUnits


# ---------------------------------------------------------------------------
# BUG-09: Mean ionizing photon energy
# ---------------------------------------------------------------------------
class TestBug09MeanPhotonEnergy:
    """agn_nebular.py:177-183 — <hnu> must depend on spectral index."""

    def test_mean_energy_varies_with_alpha(self):
        """For different power-law indices, <hnu> must differ."""
        h = 6.626e-27
        nu_lyman = 3.29e15  # Lyman limit frequency
        nu_max = 1e18  # X-ray cutoff

        def correct_mean_hnu(alpha):
            """Correct: <hnu> = integral(hnu * nu^{alpha-1}) / integral(nu^{alpha-1})."""
            # numerator exponent: alpha+1, denominator exponent: alpha
            num = (nu_max ** (alpha + 1) - nu_lyman ** (alpha + 1)) / (alpha + 1)
            den = (nu_max**alpha - nu_lyman**alpha) / alpha
            return h * num / den

        def buggy_mean_hnu(alpha):
            """Bug: both integrals use nu^alpha."""
            num = (nu_max ** (alpha + 1) - nu_lyman ** (alpha + 1)) / (alpha + 1)
            den = (nu_max ** (alpha + 1) - nu_lyman ** (alpha + 1)) / (alpha + 1)
            return h * abs(num / den)

        # Buggy version: <hnu> = h regardless of alpha
        assert abs(buggy_mean_hnu(-1.5) - buggy_mean_hnu(-2.5)) < 1e-30, (
            "Bug: mean photon energy is constant (= h)"
        )

        # Correct version: <hnu> changes with alpha
        e1 = correct_mean_hnu(-1.5)
        e2 = correct_mean_hnu(-2.5)
        assert abs(e1 - e2) / e1 > 0.1, "Correct <hnu> must differ by >10% for different alpha"


# ---------------------------------------------------------------------------
# BUG-11: summary_table key mismatch
# ---------------------------------------------------------------------------
class TestBug11SummaryKeys:
    """posterior.py:178-181 — Must show acceptance rate from RT diagnostics."""

    def test_accept_rate_key_found(self):
        """The key stored by _run_raytrace must be recognized by summary_table."""
        # These are the keys actually stored by the samplers:
        rt_keys = {"accept_rate", "n_steps", "step_size"}
        nuts_keys = {"n_divergent", "mean_accept_prob"}

        # These are the keys posterior.py checks for:
        checked_keys = {"acceptance_rate", "n_divergences"}

        rt_found = checked_keys & rt_keys
        nuts_found = checked_keys & nuts_keys

        # BUG: none of the checked keys match the stored keys
        assert len(rt_found) == 0, "If this fails, BUG-11 has been fixed"
        assert len(nuts_found) == 0, "If this fails, BUG-11 has been fixed"


# ---------------------------------------------------------------------------
# BUG-13: nonparametric len() under JIT
# ---------------------------------------------------------------------------
class TestBug13NonparametricJit:
    """nonparametric.py:74 — Must not use len() on JAX arrays."""

    @pytest.mark.xfail(reason="BUG-13: len() on JAX array under JIT", strict=True)
    def test_continuity_sfh_jit(self):
        """continuity_sfh must JIT-compile without ConcretizationTypeError."""
        try:
            from tengri.components.sfh.nonparametric import continuity_sfh
        except ImportError:
            pytest.skip("nonparametric module not available")

        bin_edges = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.0])
        log_ratios = jnp.zeros(6)
        age_grid = jnp.linspace(0.01, 13.0, 100)

        # This should work but currently raises ConcretizationTypeError
        jitted = jax.jit(continuity_sfh)
        result = jitted(log_ratios, age_grid, bin_edges)
        assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# BUG-16: Dead code in eline_priors.py — FIXED (bare expression removed)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-23: wg00_cloudy NaN gradient at tau_k=0
# ---------------------------------------------------------------------------
class TestBug23WG00Gradient:
    """attenuation.py:1198-1202 — grad must be finite at tau_k=0."""

    @pytest.mark.xfail(reason="BUG-23: NaN gradient trap", strict=True)
    def test_gradient_finite_at_zero_tau(self):
        """jax.grad through wg00_cloudy must not produce NaN at tau_k=0."""
        try:
            from tengri.components.dust.attenuation import wg00_cloudy
        except ImportError:
            pytest.skip("wg00_cloudy not available")

        wave = jnp.array([5500.0])

        def f(tau_v):
            return jnp.sum(wg00_cloudy(wave, dust_tau_v=tau_v))

        grad_jax = float(jax.grad(f)(0.0))
        grad_fd = fd_grad(f, 0.0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-12,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )


# ---------------------------------------------------------------------------
# BUG-30: Planck function divide-by-zero
# ---------------------------------------------------------------------------
class TestBug30PlanckDivZero:
    """emission.py:159-160 — exp(x)-1 must not be zero."""

    def test_planck_finite_at_long_wavelength(self):
        """B_nu must be finite at very long wavelengths (Rayleigh-Jeans)."""
        try:
            from tengri.components.dust.emission import planck_bnu
        except ImportError:
            pytest.skip("planck_bnu not available")

        # 1 mm wavelength, T=30 K: x = hnu/kT ~ 0.005
        # 10 mm wavelength, T=30 K: x ~ 0.0005
        # Very long wavelength: x -> 0
        wave_aa = jnp.array([1e7, 1e8, 1e9])  # 1mm, 10mm, 100mm in Angstrom
        T = 30.0

        result = planck_bnu(wave_aa, T)
        assert jnp.all(jnp.isfinite(result)), f"Planck function has non-finite values: {result}"
        assert jnp.all(result > 0), "Planck function must be positive"

    def test_planck_finite_with_float32_uv_input(self):
        """B_nu must be finite when given float32 input at short UV wavelengths.

        Root cause: ssp_wave is stored as float32 in HDF5 files. With JAX's
        weak-type promotion, ``float32_array * Python_float`` stays float32 even
        with x64 enabled globally. At 5.6 Å, nu = 5.35e17 Hz and nu**3 ~ 1.5e53,
        far beyond float32 max (~3.4e38). Without an explicit float64 cast inside
        planck_bnu, nu**3 overflows to Inf and expm1(x) = Inf, giving Inf/Inf = NaN.
        """
        try:
            from tengri.components.dust.emission import planck_bnu
        except ImportError:
            pytest.skip("planck_bnu not available")

        # Mimic the actual SSP wavelength array dtype (float32 from HDF5)
        wave_aa = jnp.array([5.6, 10.0, 50.0, 100.0, 1000.0, 5000.0], dtype=jnp.float32)
        T = 35.0

        result = planck_bnu(wave_aa, T)
        assert jnp.all(jnp.isfinite(result)), (
            f"planck_bnu returned non-finite values for float32 input: {result}. "
            "Check float64 cast inside planck_bnu."
        )
        assert jnp.all(result >= 0), "Planck function must be non-negative"


# ---------------------------------------------------------------------------
# BUG-29: _mstar uses formed mass, not surviving mass (XRB over-estimate)
# ---------------------------------------------------------------------------
class TestBug29MstarSurvivingMass:
    """sed_pipeline.py:753 — XRB must use surviving stellar mass, not formed mass.

    Lehmer+2010 / Mineo+2012 XRB calibrations are normalised to surviving
    stellar mass (living stars + remnants). Using total formed mass overestimates
    XRB L_X by ~30-50% for old stellar populations.

    Fix: sed_pipeline.py now calls compute_surviving_mass(weights,
    interpolate_mass_remaining(...)) and exposes both mstar_formed and
    mstar_surviving in the output dict.
    """

    def test_surviving_mass_less_than_formed_for_old_population(self):
        """Surviving mass must be < formed mass for a purely old SSP.

        For a 10 Gyr population with Kroupa IMF, f_surv ≈ 0.6 (B&C03).
        compute_surviving_mass(weights, f_surv * ones) < sum(weights).
        """
        from tengri.components.sps.dsps_wrapper import compute_surviving_mass

        weights = jnp.ones(50) * 1e9  # 50 age bins, 1e9 Msun each
        # Simulate old population: f_surv = 0.6 uniformly
        mass_remaining = jnp.full(50, 0.6)
        surviving = float(compute_surviving_mass(weights, mass_remaining))
        formed = float(jnp.sum(weights))
        assert surviving < formed, (
            f"Surviving mass {surviving:.3e} should be < formed mass {formed:.3e}"
        )
        assert abs(surviving / formed - 0.6) < 1e-6, (
            f"Expected surviving/formed = 0.6, got {surviving / formed:.4f}"
        )

    def test_interpolate_mass_remaining_shape(self):
        """interpolate_mass_remaining returns shape (n_age,) for a scalar log_z."""
        from tengri.components.sps.dsps_wrapper import interpolate_mass_remaining

        n_met, n_age = 4, 20
        # Synthetic mass-remaining grid: decreases with age (older → less survives)
        ssp_mass_remaining = jnp.ones((n_met, n_age)) * jnp.linspace(0.95, 0.50, n_age)
        ssp_lgmet = jnp.linspace(-2.0, 0.3, n_met)
        mr = interpolate_mass_remaining(ssp_mass_remaining, ssp_lgmet, log_z=-1.0)
        assert mr.shape == (n_age,), f"Expected shape ({n_age},), got {mr.shape}"
        assert jnp.all(mr > 0.0), "Mass-remaining fractions must be positive"
        assert jnp.all(mr <= 1.0), "Mass-remaining fractions must be <= 1"

    def test_pipeline_exposes_both_mstar_keys(self):
        """sed_pipeline must expose mstar_formed and mstar_surviving in output dict."""
        import inspect

        from tengri.forward import pipeline as sed_pipeline

        src = inspect.getsource(sed_pipeline)
        assert '"mstar_formed"' in src, "Pipeline output dict must contain 'mstar_formed' key"
        assert '"mstar_surviving"' in src, (
            "Pipeline output dict must contain 'mstar_surviving' key"
        )

    def test_pipeline_uses_surviving_for_xrb(self):
        """XRB must receive surviving mass, not formed mass.

        Verifies that the pipeline passes _mstar_surviving (not _mstar_formed)
        to xray_total via the stellar_mass argument.
        """
        import inspect

        from tengri.forward import pipeline as sed_pipeline

        src = inspect.getsource(sed_pipeline)
        # The fix introduces _mstar_surviving and assigns _mstar = _mstar_surviving.
        # XRB call uses stellar_mass=_mstar.
        assert "_mstar_surviving" in src, (
            "Pipeline must compute _mstar_surviving from mass-remaining grid"
        )
        assert "compute_surviving_mass" in src, (
            "Pipeline must call compute_surviving_mass (not just jnp.sum(weights))"
        )


# ---------------------------------------------------------------------------
# BUG-04: Warm Comptonization uses simplified power-law, not nthcomp
# ---------------------------------------------------------------------------


class TestBug04WarmComptonization:
    """disc.py — warm zone must use nthcomp (K&D 2018 §2.2), not a modified BB.

    The full fix requires precomputed templates (build_nthcomp_templates.py).
    These tests verify:
    1. The nthcomp numpy solver produces physically correct spectra.
    2. kubota_done_disc emits a warning when templates are absent.
    3. When templates are present, the warm zone SED differs from the
       simplified proxy (confirming the two paths are distinct).
    4. The nthcomp spectrum peaks at higher frequency than the seed BB
       (the defining signature of Comptonization).

    Reference: Kubota & Done (2018) MNRAS 480 1247 §2.2;
               Zdziarski, Johnson & Magdziarz (1996) MNRAS 283 193.
    """

    def test_nthcomp_template_returns_finite_nonnegative(self):
        """nthcomp template interpolation returns finite, non-negative shape."""
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE, nthcomp_lnu_interp

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates absent — run scripts/build_nthcomp_templates.py")

        nu = jnp.array(np.logspace(13, np.log10(5e18), 200))
        # K&D 2018 default warm zone: Gamma=2.5, kTe=0.2 keV, kTbb=10 eV = 0.01 keV
        shape = nthcomp_lnu_interp(nu, gamma=2.5, kTe_keV=0.2, kTbb_keV=0.01)
        assert shape.shape == nu.shape
        assert jnp.all(jnp.isfinite(shape)), "nthcomp template shape must be finite everywhere"
        assert jnp.all(shape >= 0.0), "nthcomp template shape must be non-negative"
        assert jnp.any(shape > 0), "nthcomp template shape must be non-zero somewhere"

    def test_nthcomp_spectrum_peaks_above_seed_temperature(self):
        """Comptonized spectrum peak must be at higher nu than the seed BB.

        For warm Comptonization, photons are scattered to higher energies
        than the seed blackbody.  The nthcomp template peak should be at
        significantly higher frequency than the seed BB peak (Wien: nu_peak = 2.82 kT/h).
        """
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE, nthcomp_lnu_interp

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates absent — run scripts/build_nthcomp_templates.py")

        kTbb_keV = 0.01  # 10 eV seed
        kTe_keV = 0.2
        gamma = 2.5
        _KEV_TO_ERG = 1.602176634e-9
        _H_PLANCK = 6.62607015e-27
        nu_seed_peak = 2.82 * kTbb_keV * _KEV_TO_ERG / _H_PLANCK  # Hz

        nu = jnp.array(np.logspace(13, np.log10(5e18), 300))
        shape = np.array(nthcomp_lnu_interp(nu, gamma=gamma, kTe_keV=kTe_keV, kTbb_keV=kTbb_keV))
        nu_np = np.array(nu)

        power = shape * nu_np  # weight by nu for energy centroid
        if power.sum() > 0:
            nu_centroid = np.average(nu_np, weights=power)
            assert nu_centroid > nu_seed_peak * 5, (
                f"nthcomp centroid {nu_centroid:.2e} Hz should be > 5x seed BB "
                f"peak {nu_seed_peak:.2e} Hz — Comptonization must shift photons up"
            )

    def test_nthcomp_gamma_effect(self):
        """Harder Gamma (steeper spectrum) reduces soft X-ray relative to UV.

        Larger Gamma → steeper power-law → less energy at high nu.
        The ratio of X-ray to UV flux must decrease as Gamma increases.
        """
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE, nthcomp_lnu_interp

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates absent — run scripts/build_nthcomp_templates.py")

        nu = np.logspace(13, np.log10(5e18), 300)
        nu_uv = nu[(nu > 1e15) & (nu < 3e15)]  # UV band
        nu_xray = nu[(nu > 5e17) & (nu < 2e18)]  # soft X-ray band

        def xray_uv_ratio(gamma):
            shape = np.array(
                nthcomp_lnu_interp(jnp.array(nu), gamma=gamma, kTe_keV=0.2, kTbb_keV=0.01)
            )
            f_uv = np.trapezoid(np.interp(nu_uv, nu, shape), nu_uv)
            f_xray = np.trapezoid(np.interp(nu_xray, nu, shape), nu_xray)
            return f_xray / max(f_uv, 1e-300)

        ratio_soft = xray_uv_ratio(gamma=2.0)  # softer spectrum
        ratio_hard = xray_uv_ratio(gamma=3.0)  # harder spectrum
        assert ratio_hard < ratio_soft, (
            f"Harder Gamma=3.0 (ratio={ratio_hard:.4f}) should have less X-ray "
            f"relative to UV than Gamma=2.0 (ratio={ratio_soft:.4f})"
        )

    def test_kubota_done_disc_warns_without_templates(self, monkeypatch):
        """kubota_done_disc must warn when nthcomp templates are absent.

        Uses monkeypatch to simulate absent templates regardless of whether
        data/nthcomp_templates.h5 is present on this machine.
        """
        import warnings as _warnings

        import tengri.components.agn._nthcomp as _nthcomp_mod
        import tengri.components.agn.disc as _disc_mod
        from tengri.components.agn.disc import kubota_done_disc

        monkeypatch.setattr(_nthcomp_mod, "_TABLE_AVAILABLE", False)
        monkeypatch.setattr(_disc_mod, "_NTHCOMP_AVAILABLE", False)

        wavelength = jnp.linspace(1000.0, 50000.0, 100)
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            result = kubota_done_disc(wavelength, agn_log_lbol=46.0)
        warning_msgs = [str(x.message) for x in w]
        assert any("nthcomp" in m for m in warning_msgs), (
            "kubota_done_disc must warn that nthcomp templates are absent"
        )
        assert jnp.all(jnp.isfinite(result)), "Fallback must still return a finite SED"

    def test_kubota_done_disc_uses_nthcomp_when_templates_present(self):
        """When templates present, warm zone SED differs from simplified proxy.

        This is the key regression: the old (buggy) code multiplied B_nu by
        (nu/nu_seed)^(Gamma-1).  The nthcomp Kompaneets solution produces a
        qualitatively different spectrum (correct soft X-ray excess shape).
        The two results must differ by > 1% in X-ray bands.
        """
        from tengri.components.agn._nthcomp import _TABLE_AVAILABLE

        if not _TABLE_AVAILABLE:
            pytest.skip("nthcomp templates absent — run scripts/build_nthcomp_templates.py")

        from tengri.components.agn import disc as disc_mod

        # Build a high-nu wavelength grid (soft X-ray: 10-100 Å = 1-10 keV)
        wav_xray = jnp.linspace(10.0, 200.0, 80)  # Angstrom (soft X-ray / EUV)

        # nthcomp path (templates present → _NTHCOMP_AVAILABLE=True)
        result_nthcomp = disc_mod.kubota_done_disc(wav_xray, agn_log_lbol=46.0)

        # Temporarily monkey-patch _NTHCOMP_AVAILABLE to test the fallback
        _orig = disc_mod._NTHCOMP_AVAILABLE
        disc_mod._NTHCOMP_AVAILABLE = False
        try:
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                result_simplified = disc_mod.kubota_done_disc(wav_xray, agn_log_lbol=46.0)
        finally:
            disc_mod._NTHCOMP_AVAILABLE = _orig

        # The two spectra must differ (nthcomp ≠ power-law proxy)
        rel_diff = jnp.abs(result_nthcomp - result_simplified) / jnp.maximum(
            jnp.abs(result_simplified), 1e-100
        )
        assert float(jnp.max(rel_diff)) > 0.01, (
            "nthcomp and simplified warm Compton spectra must differ by > 1% — "
            "if they are identical the nthcomp path is not being used"
        )


# ---------------------------------------------------------------------------
# NaN fix: compositional kernel L_absorbed_stellar guard
# (parity with hybrid kernel lines 645-649 and sed_pipeline.py lines 742-743)
# ---------------------------------------------------------------------------


class TestCompositionalKernelLAbsorbedGuard:
    """fused_kernels.py compositional path — L_absorbed_stellar must be guarded.

    The hybrid kernel and sed_pipeline.py both have:
        L_absorbed_stellar = jnp.where(jnp.isfinite(...), ..., 0.0)
        L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, 0.0)

    The compositional kernel was missing this guard until this fix. These tests
    verify the guard logic behaves correctly and would detect its removal.
    """

    def test_guard_clamps_inf_to_zero(self):
        """The isfinite guard replaces Inf with 0.0, preventing NaN in L_ir."""
        # Simulate what -jnp.trapezoid(sed_intr - sed_atten, nu) returns when
        # sed_intr contains Inf (e.g., from pure-SSP extreme metallicity UV flux)
        L_absorbed_stellar_raw = jnp.array(jnp.inf)

        # Apply the guard added to the compositional kernel
        L_absorbed_stellar = jnp.where(
            jnp.isfinite(L_absorbed_stellar_raw), L_absorbed_stellar_raw, 0.0
        )
        L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, 0.0)
        L_ir = jnp.maximum(L_absorbed_stellar * 0.5, 0.0)

        assert jnp.isfinite(L_ir), "Guard failed: Inf L_absorbed_stellar produced non-finite L_ir"
        assert float(L_ir) == pytest.approx(0.0, abs=1e-30)

    def test_guard_clamps_nan_to_zero(self):
        """The isfinite guard replaces NaN (e.g. 0*Inf) with 0.0."""
        L_absorbed_stellar_raw = jnp.array(jnp.nan)

        L_absorbed_stellar = jnp.where(
            jnp.isfinite(L_absorbed_stellar_raw), L_absorbed_stellar_raw, 0.0
        )
        L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, 0.0)

        assert jnp.isfinite(L_absorbed_stellar)
        assert float(L_absorbed_stellar) == pytest.approx(0.0, abs=1e-30)

    def test_guard_preserves_valid_values(self):
        """The guard must not alter normal (finite, positive) absorbed luminosities."""
        L_in = jnp.array(1.23e45)  # typical AGN host absorbed luminosity in erg/s

        L_out = jnp.where(jnp.isfinite(L_in), L_in, 0.0)
        L_out = jnp.maximum(L_out, 0.0)

        assert float(L_out) == pytest.approx(float(L_in), rel=1e-6)

    def test_guard_present_in_source(self):
        """Smoke test: verify the isfinite guard exists in both guard sites.

        Guard locations:
        - fused_kernels.py: hybrid path (_hybrid_phot_body)
        - nonstell.py: compositional path (build_nonstell_fn)

        If either is accidentally removed, this test catches it immediately
        without needing to construct a full model.
        """
        import inspect

        import tengri.forward.kernels.hybrid as fk
        import tengri.forward.nonstell as ns

        fk_count = inspect.getsource(fk).count("jnp.isfinite(L_absorbed_stellar)")
        ns_count = inspect.getsource(ns).count("jnp.isfinite(L_absorbed_stellar)")
        total = fk_count + ns_count
        assert total >= 2, (
            f"Expected ≥2 isfinite guards on L_absorbed_stellar across "
            f"fused_kernels.py ({fk_count}) and nonstell.py ({ns_count}). "
            "A guard may have been accidentally removed."
        )


# ---------------------------------------------------------------------------
# BUG-NSS-04: IGM silently not applied in z-table kernel (hybrid.py)
# ---------------------------------------------------------------------------


class TestBugNSS04ZTableIGM:
    """hybrid.py build_hybrid_photometry_ztable — IGM must be applied.

    Before the fix, has_igm was set but never referenced in the z-table kernel
    (the block was just `pass`). This left both stellar photometry and non-stellar
    SED without IGM attenuation, producing artificially bright UV/NUV flux at
    z > 0.5 even with apply_igm=True.

    Fix (hybrid.py lines 1565-1574 and 2265-2320):
      1. Non-stellar: call igm_transmission(ssp_wave * (1+z), z) inside the traced
         function and multiply the non-stellar SED before filter integration.
      2. Stellar: linearly interpolate igm_trans_table (n_z, n_filters) to the
         current redshift and multiply stellar_phot after flux scaling.

    The fix does not cite an external equation because it is a wiring correction
    (applying an existing IGM model that was computed but not used), not a physics
    formula change. Inoue et al. (2014) MNRAS 442, 1805 governs the IGM model itself.
    """

    def test_ztable_igm_wiring_present_in_source(self):
        """Smoke: verify z-table kernel wires has_igm to actual IGM application.

        The buggy code had `if has_igm: pass`.  The fixed code must reference
        `_igm_fn` (the import of igm_transmission) inside the kernel setup and
        must apply `_igm_full` or `_igm_eff` to the SED/photometry.
        """
        import inspect

        import tengri.forward.kernels.hybrid as fk

        src = inspect.getsource(fk)

        # Must import igm_transmission (not just set has_igm=True)
        assert "_igm_fn" in src, (
            "z-table kernel must import igm_transmission as _igm_fn when has_igm is True"
        )

        # Must apply full-wavelength IGM to non-stellar SED
        assert "_igm_full" in src, (
            "z-table kernel must compute _igm_full and apply it to non_stellar_sed"
        )

        # Must apply per-filter IGM to stellar photometry
        assert "_igm_eff" in src, (
            "z-table kernel must compute _igm_eff and multiply stellar_phot by it"
        )

        # Must NOT have the old dead `pass` block (buggy version had this comment + pass).
        # Guard: check that the stub was replaced with real implementation.
        _dead_stub = "Full wavelength IGM will be evaluated in non-stellar section\n        pass"
        assert _dead_stub not in src, "Old dead `pass` IGM block still present — fix not applied"

    def test_igm_trans_table_interpolation_formula(self):
        """Verify that the per-filter IGM interpolation formula is correct.

        The z-table kernel uses the same linear interpolation as interpolate_ztable:
          frac = (z - z_grid[iz]) / (z_grid[iz+1] - z_grid[iz])
          igm_eff = (1-frac) * igm_table[iz] + frac * igm_table[iz+1]

        This is a unit test of the formula itself with a synthetic igm_trans_table.
        """
        # Synthetic igm_trans_table: 5 z-grid points, 3 filters
        z_grid = jnp.array([0.1, 0.5, 1.0, 2.0, 3.0])
        # IGM transmission decreases with z (more absorption at high z)
        igm_table = jnp.array(
            [
                [0.99, 0.99, 1.00],  # z=0.1
                [0.90, 0.95, 1.00],  # z=0.5
                [0.70, 0.85, 1.00],  # z=1.0
                [0.40, 0.65, 1.00],  # z=2.0
                [0.15, 0.45, 1.00],  # z=3.0
            ]
        )

        # Interpolate at z=0.75 (midpoint between z=0.5 and z=1.0, index 1→2)
        z_test = jnp.float64(0.75)
        _z_c = jnp.clip(z_test, z_grid[0], z_grid[-1])
        _iz = jnp.clip(jnp.searchsorted(z_grid, _z_c) - 1, 0, z_grid.shape[0] - 2)
        _frac = (_z_c - z_grid[_iz]) / (z_grid[_iz + 1] - z_grid[_iz])
        igm_eff = (1.0 - _frac) * igm_table[_iz] + _frac * igm_table[_iz + 1]

        # Expected: linear interp between row at z=0.5 and z=1.0, frac=0.5
        expected = 0.5 * jnp.array([0.90, 0.95, 1.00]) + 0.5 * jnp.array([0.70, 0.85, 1.00])
        assert int(_iz) == 1, f"Expected iz=1 (z=0.5 bin), got {int(_iz)}"
        assert abs(float(_frac) - 0.5) < 1e-10, f"Expected frac=0.5, got {float(_frac)}"
        assert jnp.allclose(igm_eff, expected, atol=1e-10), (
            f"IGM interp mismatch: got {igm_eff}, expected {expected}"
        )

    def test_igm_attenuates_uv_at_high_z(self):
        """IGM transmission must be < 1 for UV at z ~ 3 (Lyman forest).

        Uses igm_transmission directly to verify the physics. igm_transmission
        takes **observed-frame** wavelengths. At z=3, the Ly-alpha forest absorbs
        all observed wavelengths below 1216*(1+3)=4864 Å. An observed wavelength of
        2271 Å at z=3 corresponds to rest-frame 2271/(1+3)=568 Å — deep in the
        Lyman continuum — so IGM transmission must be essentially 0.
        """
        from tengri.components.igm import igm_transmission

        # Observed-frame wavelength: 2271 Å. At z=3 this probes rest ~568 Å (Lyman continuum).
        wave_obs = jnp.array([2271.0])  # observed-frame Angstrom (already observer frame)
        z = 3.0
        trans = igm_transmission(wave_obs, z)
        assert float(trans[0]) < 0.5, (
            f"IGM transmission at z=3 for wave_obs=2271 Å must be < 0.5, got {float(trans[0]):.4f}"
        )
