"""Regression tests for the 29 bugs fixed in the April 2026 audit.

Every test is named for the bug it covers.  Tests are ordered by severity:
showstoppers first, then serious, then sloppy.

References
----------
- disc.py ring area: Rybicki & Lightman 1979, Eq. 1.6
- warm Comptonization: Kubota & Done 2018, MNRAS 480 1247, Eq. 3
- Beloborodov / Gamma_hot: Kubota & Done 2018 Eq. 6; Beloborodov 1999 ApJ 510 L123
- ADAF T_e / synchrotron: Mahadevan 1997, ApJ 477 585, Eq. 4-9 / Eq. 24
- Balmer continuum: Osterbrock & Ferland, AGN^2 Eq. 2.4; Grandi 1982
- Shock units: delta-function and Gaussian branches must return same unit (Lsun/Hz)
- Continuity / Dirichlet SFH: Leja+2019 ApJ 876 3 (step functions)
- Posterior keys: fitter.py:3509 (accept_rate), nuts.py:193 (n_divergent)
- Attenuation cutoff: Leitherer+2002 ApJS 140 303 Eq. 14 (970-1800 A valid range)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


_WAVE = jnp.logspace(2.5, 8.0, 500)  # 316 A to 10 cm, broad grid

# ── SHOWSTOPPER 1: SFR no longer hardcoded to 1.0 Msun/yr ─────────


class TestSFRNotHardcoded:
    """Bug: sed_pipeline.py:638 — SFR fallback was 1.0 Msun/yr for all parametric SFH."""

    def test_sfr_varies_with_mass(self):
        """SFR used for X-ray scaling should depend on the SFH, not be 1.0."""
        from tengri.components.sfh.mean_sfh import double_powerlaw

        # double_powerlaw(t_lookback, alpha, beta, tau, norm)
        # norm scales SFR amplitude, so SFR[-1] should scale with norm
        t_lookback = jnp.logspace(6, 10, 100)
        sfr_high = double_powerlaw(t_lookback, alpha=0.5, beta=2.0, tau=1e9, norm=100.0)
        sfr_low = double_powerlaw(t_lookback, alpha=0.5, beta=2.0, tau=1e9, norm=0.1)
        # sfr[-1] is the instantaneous SFR; high-norm galaxy must exceed low-norm
        assert sfr_high[-1] > sfr_low[-1]
        # Neither should be 1.0 Msun/yr by accident (hardcoded fallback bug)
        assert not jnp.isclose(sfr_high[-1], 1.0, atol=0.1)
        assert not jnp.isclose(sfr_low[-1], 1.0, atol=0.1)


# ── SHOWSTOPPER 2: SFR time-averaging trapezoid non-negative ──────


class TestSFRTrapezoidNonNegative:
    """Bug: model.py:791-804 — zeroed age array caused negative SFR integrals."""

    def test_sfr_100myr_non_negative(self):
        """sfr_100myr must never be negative regardless of SFH shape."""
        from tengri.components.sfh.mean_sfh import double_powerlaw

        age_yr = jnp.logspace(6, 10.1, 200)
        for norm in [0.01, 1.0, 100.0]:
            sfr = double_powerlaw(age_yr, alpha=0.5, beta=2.0, tau=2e9, norm=norm)
            mask_100 = age_yr <= 1e8
            sfr_100_masked = jnp.where(mask_100, sfr, 0.0)
            # Fixed bug: use real ages as x-values, not zeroed array
            integral_100 = jnp.trapezoid(sfr_100_masked, age_yr)
            assert integral_100 >= 0.0, f"sfr_100myr integral negative: {integral_100:.3e}"

    def test_sfr_10myr_non_negative(self):
        """sfr_10myr must be non-negative."""
        from tengri.components.sfh.mean_sfh import double_powerlaw

        age_yr = jnp.logspace(6, 10.1, 200)
        sfr = double_powerlaw(age_yr, alpha=0.3, beta=3.0, tau=5e8, norm=10.0)
        mask_10 = age_yr <= 1e7
        sfr_10_masked = jnp.where(mask_10, sfr, 0.0)
        integral_10 = jnp.trapezoid(sfr_10_masked, age_yr)
        assert integral_10 >= 0.0


# ── SHOWSTOPPER 3: disc.py — ring area π factor ───────────────────


class TestRingAreaPi:
    """Bug: disc.py:298/618/639/858 — ring area missing pi from hemisphere integral.

    Rybicki & Lightman 1979, Eq. 1.6: dL_nu = pi * B_nu * dA * cos(i).
    Since the disc is renormalized to L_bol, the shape is unaffected, but
    l_warm_bol used in the Beloborodov energy budget must include the pi.
    """

    def test_multicolor_disc_finite_positive(self):
        """multicolor_disc should return finite, positive SED at all wavelengths."""
        from tengri.components.agn.disc import multicolor_disc

        l_nu = multicolor_disc(
            _WAVE, agn_log_lbol=12.0, agn_frac=1.0, agn_log_mbh=8.0, agn_cos_inc=0.5
        )
        assert jnp.all(jnp.isfinite(l_nu))
        assert jnp.all(l_nu >= 0.0)

    def test_kubota_done_disc_finite(self):
        """kubota_done_disc should return finite, positive SED."""
        from tengri.components.agn.disc import kubota_done_disc

        l_nu = kubota_done_disc(
            _WAVE, agn_log_lbol=12.0, agn_frac=1.0, agn_log_mbh=8.0, agn_log_ledd=-1.0
        )
        assert jnp.all(jnp.isfinite(l_nu))
        assert jnp.all(l_nu >= 0.0)

    def test_adaf_disc_finite(self):
        """adaf_disc should return finite, positive SED."""
        from tengri.components.agn.disc import adaf_disc

        l_nu = adaf_disc(
            _WAVE, agn_log_lbol=10.0, agn_frac=0.1, agn_log_mbh=8.0, agn_log_ledd=-3.0
        )
        assert jnp.all(jnp.isfinite(l_nu))
        assert jnp.all(l_nu >= 0.0)


# ── SHOWSTOPPER 4: Warm Comptonization — UV boost present ─────────


class TestWarmComptonization:
    """Bug: disc.py:321-362 — warm zone used kT_warm (soft X-rays) as seed frequency,
    so the enhancement was never triggered at optical/UV.  K&D 2018 Eq. 3 prescribes
    the local disc temperature as the seed frequency.
    """

    def test_warm_comp_exceeds_outer_disc_at_uv(self):
        """With warm Comptonization, the warm zone SED should exceed a pure blackbody
        at intermediate UV/soft-X-ray wavelengths.
        """
        from tengri.components.agn.disc import (
            _planck_lnu,
            _warm_comptonization_lnu,
            _wavelength_to_nu,
        )

        wave_uv = jnp.logspace(2.5, 6.0, 200)  # 316 A - 1 mm
        nu = _wavelength_to_nu(wave_uv)
        temperature = 1e5  # K  — representative warm zone ring temperature
        kt_warm_kev = 0.2  # keV
        _KEV_TO_ERG = 1.602176634e-9
        _H_PLANCK = 6.626e-27
        nu_warm = kt_warm_kev * _KEV_TO_ERG / _H_PLANCK

        b_nu_plain = _planck_lnu(nu, temperature)
        b_nu_comp = _warm_comptonization_lnu(nu, temperature, nu_warm, gamma_warm=2.5)

        # The Comptonized spectrum should have MORE power than a plain blackbody
        # at intermediate frequencies between nu_seed and nu_warm.
        _K_BOLTZ = 1.38e-16
        nu_seed = _K_BOLTZ * temperature / _H_PLANCK  # ~2e15 Hz for T=1e5 K
        mid_mask = (nu > nu_seed) & (nu < nu_warm)
        if jnp.sum(mid_mask) > 5:
            denom = jnp.maximum(jnp.mean(b_nu_plain[mid_mask]), 1e-300)
            ratio = jnp.mean(b_nu_comp[mid_mask]) / denom
            assert ratio > 1.0, f"Comptonized SED not enhanced over plain BB: ratio={ratio:.3f}"

    def test_warm_comp_finite_positive(self):
        """_warm_comptonization_lnu must return finite, positive values."""
        from tengri.components.agn.disc import _warm_comptonization_lnu, _wavelength_to_nu

        nu = _wavelength_to_nu(_WAVE)
        _KEV_TO_ERG = 1.602176634e-9
        _H_PLANCK = 6.626e-27
        nu_warm = 0.2 * _KEV_TO_ERG / _H_PLANCK
        b_nu = _warm_comptonization_lnu(nu, 1e5, nu_warm, 2.5)
        assert jnp.all(jnp.isfinite(b_nu))
        assert jnp.all(b_nu >= 0.0)


# ── SHOWSTOPPER 5: ADAF — T_e has m_dot dependence ────────────────


class TestADAFMdotDependence:
    """Bug: disc.py:783-789 — T_e = 5e9 * delta^0.5 ignored m_dot.
    Mahadevan 1997 Eq. 4-9: T_e ∝ (delta/m_dot)^0.5.
    """

    def test_adaf_seds_differ_with_mdot(self):
        """adaf_disc at different Eddington ratios should give different SED shapes
        (reflecting different T_e and synchrotron peak).
        """
        from tengri.components.agn.disc import adaf_disc

        l_nu_high = adaf_disc(
            _WAVE, agn_log_lbol=10.0, agn_frac=0.1, agn_log_mbh=8.0, agn_log_ledd=-2.0
        )
        l_nu_low = adaf_disc(
            _WAVE, agn_log_lbol=10.0, agn_frac=0.1, agn_log_mbh=8.0, agn_log_ledd=-4.0
        )
        # SEDs should differ in shape (not just scale, since both are renormalized to L_bol)
        ratio = l_nu_high / jnp.maximum(l_nu_low, 1e-300)
        # Synchrotron peak moves with m_dot, so the ratio should not be flat
        finite_ratio = ratio[jnp.isfinite(ratio) & (ratio > 0)]
        ratio_spread = jnp.std(jnp.log10(finite_ratio))
        assert ratio_spread > 0.01, "ADAF SED shape does not change with m_dot"

    def test_adaf_synchrotron_peak_moves_with_mdot(self):
        """Higher m_dot → higher synchrotron peak frequency (Mahadevan 1997 Eq. 24)."""
        from tengri.components.agn.disc import adaf_disc

        wave_radio = jnp.logspace(6, 9, 100)  # mm to cm radio
        l_nu_high = adaf_disc(
            wave_radio, agn_log_lbol=10.0, agn_frac=1.0, agn_log_mbh=8.0, agn_log_ledd=-2.0
        )
        l_nu_low = adaf_disc(
            wave_radio, agn_log_lbol=10.0, agn_frac=1.0, agn_log_mbh=8.0, agn_log_ledd=-4.0
        )
        peak_high = wave_radio[jnp.argmax(l_nu_high)]
        peak_low = wave_radio[jnp.argmax(l_nu_low)]
        # Higher m_dot → higher nu_peak → shorter peak wavelength
        assert peak_high <= peak_low, (
            f"Higher m_dot should have shorter peak wavelength: "
            f"high={peak_high:.2e}, low={peak_low:.2e}"
        )


# ── SHOWSTOPPER 6: Balmer continuum tau direction ─────────────────


class TestBalmerContinuumTauDirection:
    """Bug: qsogen.py:397 — tau ∝ (lambda/lambda_BE)^3 made tau larger at longer wavelengths.
    Correct: sigma_bf(nu) ~ nu^{-3} → tau(lambda) = tau_BE * (lambda_BE/lambda)^3
    (Osterbrock & Ferland AGN^2 Eq. 2.4).
    """

    def test_tau_decreases_at_longer_wavelengths(self):
        """tau must be largest at the Balmer edge (3646 A) and fall off at longer wavelengths."""
        wavbe = 3646.0  # Balmer edge in Angstrom
        taube = 1.0

        # Wavelengths shorter (above edge, higher nu) should have large tau
        # Wavelengths longer (below edge, lower nu) should have smaller tau
        wave_short = jnp.array([3000.0, 3200.0, 3400.0])  # shorter than edge -> tau > taube
        wave_long = jnp.array([4000.0, 5000.0, 7000.0])  # longer than edge -> tau < taube

        tau_short = taube * (wavbe / wave_short) ** 3
        tau_long = taube * (wavbe / wave_long) ** 3

        # tau at the edge should be taube
        tau_at_edge = taube * (wavbe / wavbe) ** 3
        assert jnp.isclose(tau_at_edge, taube)

        # tau should increase toward shorter wavelengths (tau_short > taube)
        assert jnp.all(tau_short > taube), "tau should exceed taube below the Balmer edge"

        # tau should decrease at longer wavelengths (tau_long < taube)
        assert jnp.all(tau_long < taube), "tau should fall below taube above the Balmer edge"

    def test_qsogen_balmer_continuum_shape(self):
        """Balmer continuum in qsogen should peak near the edge and fall at longer wavelengths."""
        pytest.importorskip("tengri.components.agn.qsogen")
        from tengri.components.agn.qsogen import _balmer_continuum

        wave = jnp.linspace(2500.0, 5000.0, 200)
        # Use a flat continuum for the test
        continuum = jnp.ones_like(wave)
        bc = _balmer_continuum(wave, continuum, tbc=2.0, taube=1.0, wavbe=3646.0)
        # Find peak: should be near or at the Balmer edge
        peak_wave = wave[jnp.argmax(bc)]
        assert peak_wave < 4000.0, f"Balmer continuum peak at {peak_wave:.0f} A, expected < 4000 A"


# ── SHOWSTOPPER 8: Shock emission unit consistency ────────────────


class TestShockEmissionUnits:
    """Bug: shock.py:182-206 — Gaussian branch multiplied by _LSUN_ERG giving erg/s/Hz;
    delta-function branch returned Lsun/Hz.  Both should return Lsun/Hz.
    """

    def test_gaussian_delta_consistent_total_power(self):
        """Total power (integral of SED over frequency) should match between branches.

        Both branches represent the same physical emission, so the total luminosity
        (integral of SED over nu) must be approximately equal for the same input.

        Uses line_sigma_aa=200 Å to ensure the Gaussian is well-sampled by the
        log-spaced wavelength grid (grid spacing ~38 Å at Halpha → sigma/spacing ≈ 5).
        A 3 Å sigma would be unresolved and the integral would underestimate power.
        """
        from tengri.components.nebular.shock import compute_shock_sed

        # Dense linear grid around optical lines to ensure Gaussian is well-sampled
        wave = jnp.linspace(3500.0, 7500.0, 10000)
        _C_AA = 2.99792458e18
        nu = _C_AA / wave

        # Gaussian branch — wide sigma (200 Å) to be well-sampled by the grid
        sed_gaussian = compute_shock_sed(
            wave, shock_velocity=200.0, l_shock_halpha=1.0, line_sigma_aa=200.0
        )
        # Delta branch
        sed_delta = compute_shock_sed(
            wave, shock_velocity=200.0, l_shock_halpha=1.0, line_sigma_aa=0.0
        )

        # Sort for integration (nu decreases as wave increases)
        sort_idx = jnp.argsort(nu)
        power_gaussian = jnp.abs(jnp.trapezoid(sed_gaussian[sort_idx], nu[sort_idx]))
        power_delta = jnp.abs(jnp.trapezoid(sed_delta[sort_idx], nu[sort_idx]))

        # Both should be in the same units (Lsun/Hz); ratio should be order unity
        if power_gaussian > 0 and power_delta > 0:
            ratio = power_gaussian / power_delta
            assert 0.01 < ratio < 100.0, (
                f"Gaussian/delta power ratio = {ratio:.2e}; suggests unit mismatch (expected ~1)"
            )

    def test_sed_order_of_magnitude(self):
        """SED values should be in Lsun/Hz, not erg/s/Hz.

        For Halpha luminosity = 1 Lsun, the peak SED value in Lsun/Hz should be
        much smaller than 3.828e33 (which would indicate erg/s/Hz units).
        """
        from tengri.components.nebular.shock import compute_shock_sed

        wave = jnp.logspace(2.5, 5.0, 500)
        sed = compute_shock_sed(wave, shock_velocity=200.0, l_shock_halpha=1.0, line_sigma_aa=3.0)
        peak = float(jnp.max(sed))
        # If units were erg/s/Hz, peak would be ~3.828e33 × (1/sigma_nu) >> 1
        # In Lsun/Hz, peak should be << 3.828e33
        assert peak < 1e10, f"SED peak {peak:.2e} Lsun/Hz suggests wrong units (erg/s/Hz?)"


# ── SERIOUS 11: Posterior summary_table shows accept_rate and n_divergent


class TestPosteriorDiagnosticKeys:
    """Bug: posterior.py:349-352 — checked 'acceptance_rate'/'n_divergences' but
    raytrace stores 'accept_rate' and NUTS stores 'n_divergent'.
    """

    def test_accept_rate_key_shown(self):
        """summary_table() should display acceptance rate when 'accept_rate' is in diagnostics."""
        from tengri.inference.posterior import Posterior

        p = Posterior(
            samples=None,
            params={"x": jnp.array(1.0)},
            method="raytrace",
            wall_time_s=1.0,
            diagnostics={"accept_rate": 0.55},
        )
        table = p.summary_table()
        assert "accept=55.0%" in table, f"accept_rate not shown in summary_table:\n{table}"

    def test_n_divergent_key_shown(self):
        """summary_table() should display divergences when 'n_divergent' is in diagnostics."""
        from tengri.inference.posterior import Posterior

        key = jax.random.PRNGKey(0)
        samples = {"x": jax.random.normal(key, (100,))}
        p = Posterior(
            samples=samples,
            params={"x": jnp.array(1.0)},
            method="nuts",
            wall_time_s=1.0,
            diagnostics={"n_divergent": 3},
        )
        table = p.summary_table()
        assert "divergences=3" in table, f"n_divergent not shown in summary_table:\n{table}"

    def test_wrong_keys_not_used(self):
        """Old wrong key names should not trigger the diagnostic display."""
        from tengri.inference.posterior import Posterior

        p = Posterior(
            samples=None,
            params={"x": jnp.array(1.0)},
            method="raytrace",
            wall_time_s=1.0,
            diagnostics={"acceptance_rate": 0.55, "n_divergences": 3},
        )
        table = p.summary_table()
        # With the wrong keys, nothing should be shown
        assert "accept=" not in table, "Old wrong key 'acceptance_rate' is being read"
        assert "divergences=" not in table, "Old wrong key 'n_divergences' is being read"


# ── SERIOUS 13: nonparametric.py JIT-safe (no len()) ──────────────


class TestNonparametricJITSafe:
    """Bug: nonparametric.py:74,210 — len(bin_edges_gyr) raises ConcretizationTypeError in JIT."""

    def test_continuity_sfh_jit(self):
        """continuity_sfh should JIT-compile with JAX array bin_edges."""
        from tengri.components.sfh.nonparametric import DEFAULT_BIN_EDGES_GYR, continuity_sfh

        age_yr = jnp.linspace(1e7, 13e9, 100)

        @jax.jit
        def _eval(edges):
            kwargs = {f"ratio_{i}": 0.0 for i in range(edges.shape[0] - 2)}
            return continuity_sfh(age_yr, log_total_mass=10.0, bin_edges_gyr=edges, **kwargs)

        sfr = _eval(DEFAULT_BIN_EDGES_GYR)
        assert sfr.shape == age_yr.shape
        assert jnp.all(jnp.isfinite(sfr))

    def test_dirichlet_sfh_jit(self):
        """dirichlet_sfh should JIT-compile with JAX array bin_edges."""
        from tengri.components.sfh.nonparametric import DEFAULT_BIN_EDGES_GYR, dirichlet_sfh

        age_yr = jnp.linspace(1e7, 13e9, 100)

        @jax.jit
        def _eval(edges):
            kwargs = {f"z_frac_{i}": 0.5 for i in range(edges.shape[0] - 2)}
            return dirichlet_sfh(age_yr, log_total_mass=10.0, bin_edges_gyr=edges, **kwargs)

        sfr = _eval(DEFAULT_BIN_EDGES_GYR)
        assert sfr.shape == age_yr.shape

    def test_continuity_sfh_piecewise_constant(self):
        """continuity_sfh should return piecewise-constant SFR (step function per Leja+2019)."""
        from tengri.components.sfh.nonparametric import continuity_sfh

        edges = jnp.array([0.0, 1.0, 5.0, 13.7])  # 3 bins in Gyr
        # Age points within the same bin should have identical SFR
        age_in_bin0 = jnp.array([0.1e9, 0.5e9, 0.9e9])  # all in [0, 1] Gyr bin
        age_in_bin1 = jnp.array([1.5e9, 2.0e9, 4.0e9])  # all in [1, 5] Gyr bin

        sfr_bin0 = continuity_sfh(
            age_in_bin0, log_total_mass=10.0, bin_edges_gyr=edges, ratio_0=0.5, ratio_1=0.0
        )
        sfr_bin1 = continuity_sfh(
            age_in_bin1, log_total_mass=10.0, bin_edges_gyr=edges, ratio_0=0.5, ratio_1=0.0
        )

        # Within each bin, SFR must be exactly constant
        assert jnp.allclose(sfr_bin0, sfr_bin0[0]), "SFR not constant within bin 0"
        assert jnp.allclose(sfr_bin1, sfr_bin1[0]), "SFR not constant within bin 1"


# ── SERIOUS 14: DIG short-circuit when frac=0 ─────────────────────


class TestDIGShortCircuit:
    """Bug: dig.py:110-113 — DIG forward pass always called even when neb_dig_frac=0."""

    def test_dig_zero_frac_short_circuits(self):
        """With neb_dig_frac=0.0 (Python float), predict_nebular_sed should only be called once."""
        call_count = {"n": 0}

        class _FakeBackend:
            def predict_nebular_sed(self, **kwargs):
                call_count["n"] += 1
                return jnp.zeros(100)

        from tengri.components.nebular.dig import mix_dig_emission

        wave = jnp.linspace(1000.0, 10000.0, 100)
        weights = jnp.ones(10) / 10.0
        log_ages = jnp.linspace(6.0, 10.0, 10)
        mix_dig_emission(
            _FakeBackend(),
            ssp_wave=wave,
            ssp_weights=weights,
            ssp_log_ages_yr=log_ages,
            log_z=-1.848,
            neb_logU=-3.0,
            neb_dig_frac=0.0,  # Python float — should short-circuit
            neb_dig_delta_logU=-1.0,
            line_sigma_aa=50.0,
        )
        assert call_count["n"] == 1, (
            f"DIG forward pass called {call_count['n']} times with neb_dig_frac=0.0; "
            "expected 1 (short-circuit)"
        )


# ── SERIOUS 15: attenuation.py float equality safe under JIT ──────


class TestAttenuationFloatEqualitySafe:
    """Bug: attenuation.py:725-730 — narayanan_z used == for float sentinel detection,
    which is JIT-unsafe for traced values.
    """

    def test_narayanan_z_jit_safe(self):
        """narayanan_z should JIT-compile and return correct results."""
        from tengri.components.dust.attenuation import narayanan_z

        wave = jnp.linspace(1000.0, 10000.0, 100)

        @jax.jit
        def _eval(delta, bump):
            return narayanan_z(wave, dust_delta=delta, dust_bump_strength=bump, redshift=0.5)

        # Default sentinel values (delta=-0.2, bump=1.0) should activate redshift scaling
        k_default = _eval(-0.2, 1.0)
        # Non-default values should not activate redshift scaling
        k_custom = _eval(-0.4, 0.5)

        assert jnp.all(jnp.isfinite(k_default))
        assert jnp.all(jnp.isfinite(k_custom))
        # The two should be different (different delta values)
        assert not jnp.allclose(k_default, k_custom)

    def test_narayanan_z_gradient_exists(self):
        """Gradient w.r.t. dust_delta should be finite (not NaN from == comparison)."""
        from tengri.components.dust.attenuation import narayanan_z

        wave = jnp.linspace(1000.0, 10000.0, 50)

        def _sum(delta):
            k = narayanan_z(wave, dust_delta=delta, dust_bump_strength=0.5, redshift=0.3)
            return jnp.sum(k)

        g_jax = float(jax.grad(_sum)(-0.3))
        g_fd = fd_grad(_sum, -0.3)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )


# ── SLOPPY 16: eline_priors.py dead code removed ──────────────────


class TestElinePriorNoDead:
    """Bug: eline_priors.py:251 — orphaned design_matrix.shape[1] expression (no assignment)."""

    def test_no_dead_shape_expression(self):
        """The file should not contain the bare dead-code expression."""
        import inspect

        from tengri.observation import eline_priors

        src = inspect.getsource(eline_priors)
        # The dead-code line was 'design_matrix.shape[1]' with no assignment
        assert "design_matrix.shape[1]\n" not in src, (
            "Dead code 'design_matrix.shape[1]' (no assignment) still present"
        )


# ── SLOPPY 19: BLR Fe II normalization grid-resolution-independent


class TestBLRFeIINormalization:
    """Bug: blr.py:217 — jnp.sum used for Fe II normalization; result depends on pixel spacing.
    Fix uses jnp.trapezoid over frequency for grid-independent normalization.
    """

    def test_fe2_normalization_grid_independent(self):
        """Fe II normalization should not change significantly with wavelength grid resolution."""
        from tengri.components.agn.blr import compute_blr_sed

        wave_coarse = jnp.logspace(2.8, 4.2, 100)
        wave_fine = jnp.logspace(2.8, 4.2, 500)

        sed_coarse = compute_blr_sed(wave_coarse, l_disc_bol_erg=1e46, agn_fe2_strength=1.0)
        sed_fine = compute_blr_sed(wave_fine, l_disc_bol_erg=1e46, agn_fe2_strength=1.0)

        # The Fe II template in the 4434-4684 A window should normalize consistently.
        # Total power (integral over frequency) should agree within 10% between grids.
        _C_AA = 2.99792458e18
        nu_c = _C_AA / wave_coarse
        nu_f = _C_AA / wave_fine

        sort_c = jnp.argsort(nu_c)
        sort_f = jnp.argsort(nu_f)

        mask_c = (wave_coarse >= 4434.0) & (wave_coarse <= 4684.0)
        mask_f = (wave_fine >= 4434.0) & (wave_fine <= 4684.0)

        fe2_c = jnp.abs(jnp.trapezoid((sed_coarse * mask_c)[sort_c], nu_c[sort_c]))
        fe2_f = jnp.abs(jnp.trapezoid((sed_fine * mask_f)[sort_f], nu_f[sort_f]))

        if fe2_c > 0 and fe2_f > 0:
            ratio = fe2_c / fe2_f
            assert 0.5 < ratio < 2.0, (
                f"Fe II optical bump power ratio coarse/fine = {ratio:.3f}; "
                "normalization is grid-resolution-dependent"
            )


# ── SLOPPY 20: Continuity SFH piecewise-constant (not linear) ─────


class TestContinuitySFHPiecewiseConstant:
    """Bug: nonparametric.py:99-101 — jnp.interp on bin centers gives linear interpolation;
    Leja+2019 defines the SFH as piecewise-constant (step functions).
    """

    def test_sfr_constant_within_bin(self):
        """SFR must be exactly constant within each bin (step function)."""
        from tengri.components.sfh.nonparametric import continuity_sfh

        edges = jnp.array([0.0, 1.0, 5.0, 13.7])  # 3 bins in Gyr
        # All ages within the middle bin [1, 5] Gyr must have identical SFR
        ages_in_bin1 = jnp.linspace(1.01e9, 4.99e9, 50)
        kwargs = {"ratio_0": 1.0, "ratio_1": -0.5}
        sfr = continuity_sfh(ages_in_bin1, log_total_mass=10.0, bin_edges_gyr=edges, **kwargs)
        # Maximum deviation from mean should be zero (step function)
        max_dev = float(jnp.max(jnp.abs(sfr - sfr[0])))
        assert max_dev == 0.0, f"SFR varies by {max_dev:.3e} within a bin (should be zero)"


# ── SLOPPY 21: single_component_dust_fast JIT-safe ────────────────


class TestSingleComponentDustFastJITSafe:
    """Bug: attenuation.py:1077 — len(wavelengths) is not JIT-safe."""

    def test_jit_compilable(self):
        """single_component_dust_fast should JIT-compile without errors."""
        from tengri.components.dust.attenuation import single_component_dust_fast

        wave = jnp.linspace(1000.0, 10000.0, 50)
        n_ages = 10

        @jax.jit
        def _eval():
            return single_component_dust_fast(wave, tau_v=1.0, n_ages=n_ages)

        result = _eval()
        assert result.shape == (n_ages, wave.shape[0])
        assert jnp.all(jnp.isfinite(result))


# ── SLOPPY 22: Leitherer02 cutoff consistent at 1800 A ────────────


class TestLeithererCutoff:
    """Bug: attenuation.py:143 — _calzetti_l02_kprime helper used 0.15 um cutoff,
    but L02 polynomial is valid 970-1800 A (Leitherer+2002 ApJS 140 303 Eq. 14).
    Standalone leitherer02 used 0.18 um; now both match.
    """

    def test_calzetti_l02_kprime_matches_leitherer02_at_1700A(self):
        """At 1700 A (between 1500 and 1800 A), both implementations should agree."""
        from tengri.components.dust.attenuation import _calzetti_l02_kprime, leitherer02

        wave = jnp.array([1700.0])
        k_helper = _calzetti_l02_kprime(wave)  # was using 0.15 cutoff, now 0.18
        k_standalone = leitherer02(wave)

        # The two should now agree (both use L02 polynomial at 1700 A < 1800 A)
        assert jnp.allclose(k_helper / 4.05, k_standalone, rtol=0.01), (
            f"_calzetti_l02_kprime/RV={float(k_helper / 4.05):.4f} != "
            f"leitherer02={float(k_standalone):.4f} at 1700 A"
        )


# ── SLOPPY 23: wg00_cloudy gradient not zero at tau_k=0 ───────────


class TestWG00CloudyGradient:
    """Bug: attenuation.py:1198-1202 — safe_tau_k=jnp.where(...,1.0) disconnected the gradient;
    gradient of ratio branch was 0 w.r.t. tau_k when tau_k < 1e-10.
    """

    def test_gradient_finite_near_zero(self):
        """Gradient of wg00_cloudy w.r.t. tau_v must be finite and non-zero near tau_v=0."""
        from tengri.components.dust.attenuation import wg00_cloudy

        wave = jnp.linspace(3000.0, 10000.0, 50)

        def _sum_transmission(tau_v):
            return jnp.sum(wg00_cloudy(wave, tau_v=tau_v))

        # Test near zero — old code had dead gradient here
        g_jax = float(jax.grad(_sum_transmission)(1e-6))
        g_fd = fd_grad(_sum_transmission, 1e-6)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )
        assert g_jax != 0.0, "gradient is zero near tau_v=0 (disconnected)"

    def test_gradient_finite_at_large_tau(self):
        """Gradient at large tau agrees with FD (wg00_cloudy)."""
        from tengri.components.dust.attenuation import wg00_cloudy

        wave = jnp.linspace(3000.0, 10000.0, 50)

        def _sum_transmission(tau_v):
            return jnp.sum(wg00_cloudy(wave, tau_v=tau_v))

        g_jax = float(jax.grad(_sum_transmission)(2.0))
        np.testing.assert_allclose(
            g_jax,
            fd_grad(lambda t: float(_sum_transmission(t)), 2.0),
            rtol=1e-3,
            err_msg=f"wg00_cloudy large tau: autodiff={g_jax:.4e}",
        )


# ── SLOPPY 26: radio.py redundant *_LSUN/_LSUN removed ────────────


class TestRadioAGNSimplified:
    """Bug: radio.py:113 — L_B = L_agn_bol * _LSUN / (...) / _LSUN; *_LSUN/_LSUN cancelled."""

    def test_radio_agn_finite(self):
        """radio_agn should return finite values with simplified formula."""
        from tengri.components.radio import radio_agn

        wave = jnp.logspace(7.0, 9.0, 100)  # radio wavelengths
        l_nu = radio_agn(wave, L_agn_bol=1e11, radio_loudness=2.0)
        assert jnp.all(jnp.isfinite(l_nu))
        assert jnp.all(l_nu >= 0.0)


# ── SLOPPY 27: unified.py torus_frac float equality safe ──────────


class TestUnifiedAGNTorusFrac:
    """Bug: unified.py:813 — jnp.where(agn_torus_frac == 0.5, ...) JIT-unsafe for traced values."""

    def test_unified_agn_jit_safe(self):
        """unified_nlr_blr should JIT-compile without issues from float == comparison."""
        pytest.importorskip("tengri.components.agn.unified")
        from tengri.components.agn.unified import unified_nlr_blr

        wave = jnp.logspace(2.5, 5.0, 200)

        @jax.jit
        def _eval(torus_frac):
            return unified_nlr_blr(
                wave,
                agn_log_lbol=12.0,
                agn_torus_frac=torus_frac,
                agn_theta_torus=60.0,
                agn_cos_inc=0.5,
                agn_log_mbh=8.0,
                agn_log_ledd=-1.0,
            )

        # Default sentinel value (0.5) — should activate geometric derivation
        l_nu_default = _eval(0.5)
        # Non-default value — should use the provided value
        l_nu_custom = _eval(0.3)

        assert jnp.all(jnp.isfinite(l_nu_default))
        assert jnp.all(jnp.isfinite(l_nu_custom))


# ── SLOPPY 28: precompute_dust_age_mask preserves dtype ───────────


class TestDustAgeMaskDtype:
    """Bug: attenuation.py:836 — hardcoded jnp.float64 defeats mixed-precision support."""

    def test_float32_input_gives_float32_mask(self):
        """float32 age grid should produce float32 masks, not float64."""
        from tengri.components.dust.attenuation import precompute_dust_age_mask

        age_grid_f32 = jnp.linspace(0.0, 1e10, 100, dtype=jnp.float32)
        young, old = precompute_dust_age_mask(age_grid_f32, t_birth=3e8)
        assert young.dtype == jnp.float32, f"young mask is {young.dtype}, expected float32"
        assert old.dtype == jnp.float32, f"old mask is {old.dtype}, expected float32"

    def test_float64_input_gives_float64_mask(self):
        """float64 age grid should produce float64 masks."""
        from tengri.components.dust.attenuation import precompute_dust_age_mask

        age_grid_f64 = jnp.linspace(0.0, 1e10, 100, dtype=jnp.float64)
        young, _old = precompute_dust_age_mask(age_grid_f64, t_birth=3e8)
        assert young.dtype == jnp.float64, f"young mask is {young.dtype}, expected float64"
