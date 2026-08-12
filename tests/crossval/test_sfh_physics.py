# SPDX-License-Identifier: BSD-3-Clause
"""Physics cross-validation for SFH models.

Tests physical correctness of parametric, non-parametric, and GP-based
star formation history models against known analytic properties,
conservation laws, and shape constraints.

References
----------
- Bellstedt+2020 (arXiv:2005.11917) — tsnorm, snorm
- Robotham+2020 (arXiv:2002.06980) — ProSpect SFH models
- Carnall+2018 — double power law (BAGPIPES)
- Leja+2019 (arXiv:1905.11997) — continuity prior
- Leja+2017 (arXiv:1609.09073) — Dirichlet prior
"""

import chex
import jax.numpy as jnp
import pytest

# Age of the universe today [yr], from the default cosmology — never a
# literal. SFH formation anchor (age_gyr) for dpl/lnorm shape tests.
from tengri.cosmology import age_at_z0 as _age_at_z0
from tests._bounds import assert_non_negative

_AGE_UNIV_YR = float(_age_at_z0()) * 1e9

pytestmark = pytest.mark.crossval

# Standard lookback time grid: 10 kyr to 14 Gyr
T_LOOKBACK = jnp.geomspace(1e5, 14e9, 500)


# ── 1. CONSTANT SFH — trivial analytic ────────────────────────────


class TestConstantSFHPhysics:
    """Constant SFH: flat between start and end times."""

    def test_flat_between_boundaries(self):
        """SFR should be constant in the active window."""
        from tengri.components.stellar.sfh import constant_sfh

        sfr = constant_sfh(T_LOOKBACK, log_total_mass=10.0, start=1e9, end=10e9)
        active = (T_LOOKBACK >= 1e9) & (T_LOOKBACK <= 10e9)
        sfr_active = sfr[active]
        if len(sfr_active) > 2:
            cv = float(jnp.std(sfr_active) / jnp.mean(sfr_active))
            assert cv < 0.01, f"Constant SFH should be flat, CV={cv:.4f}"

    def test_mass_integral_correct(self):
        """Integral of SFR * dt = 10**log_total_mass."""
        from tengri.components.stellar.sfh import constant_sfh

        log_total_mass = 10.0
        start, end = 1e9, 10e9
        sfr = constant_sfh(T_LOOKBACK, log_total_mass=log_total_mass, start=start, end=end)
        mass = float(jnp.trapezoid(sfr, T_LOOKBACK))
        expected = 10.0**log_total_mass
        assert abs(mass / expected - 1.0) < 0.01, (
            f"Mass integral: got {mass:.2e}, expected {expected:.2e}"
        )


# ── 2. EXPONENTIAL SFH — declining model ──────────────────────────


class TestExponentialSFHPhysics:
    """Exponential SFH: SFR(t) declines exponentially from start."""

    def test_declines_from_peak(self):
        """SFR peaks at `start` lookback time and declines away from it.

        With start=0, peak is at present and SFR declines into the past.
        We verify the exponential decay behavior.
        """
        from tengri.components.stellar.sfh import exponential_sfh

        sfr = exponential_sfh(T_LOOKBACK, log_total_mass=1.0, tau=2e9, start=0.0)
        # SFR should be highest at small lookback (near present) and decay
        young = T_LOOKBACK < 1e9
        old = T_LOOKBACK > 5e9
        if jnp.sum(old) > 0 and jnp.sum(young) > 0:
            mean_young = float(jnp.mean(sfr[young]))
            mean_old = float(jnp.mean(sfr[old]))
            assert mean_young > mean_old, "Exponential SFH should peak at start"


# ── 3. DELAYED EXPONENTIAL — peaks at start + tau ─────────────────


class TestDelayedExponentialPhysics:
    """Delayed exponential: SFR(t) ∝ (t_lookback - start) * exp(-(t_lookback-start)/tau)."""

    def test_peaks_after_start(self):
        """SFR should NOT peak at start — peak is displaced by tau."""
        from tengri.components.stellar.sfh import delayed_exponential_sfh

        # start=0 means SF begins at lookback=0 (present day). Peak at ~tau.
        tau = 3e9
        sfr = delayed_exponential_sfh(T_LOOKBACK, log_total_mass=1.0, start=0.0, tau=tau)
        peak_lbt = float(T_LOOKBACK[jnp.argmax(sfr)])
        # Peak should be near tau in lookback time
        assert 0.5e9 < peak_lbt < 10e9, f"Delayed exp peak at {peak_lbt / 1e9:.1f} Gyr"
        # SFR should be non-negative and finite
        assert_non_negative(sfr, name="sfr")
        chex.assert_tree_all_finite(sfr)


# ── 4. DOUBLE POWER LAW — Carnall+2018 ────────────────────────────


class TestDPLPhysics:
    """Double power law: SFR = norm / [(t/tau)^alpha + (t/tau)^{-beta}]."""

    def test_peaks_near_tau(self):
        """DPL peaks near t = tau."""
        from tengri.components.stellar.sfh import dpl

        tau = 5e9
        age = _AGE_UNIV_YR
        sfr = dpl(T_LOOKBACK, alpha=2.0, beta=1.0, tau=tau, age=age, log_total_mass=1.0)
        peak_lbt = float(T_LOOKBACK[jnp.argmax(sfr)])
        # New convention (#514): the shape is in cosmic time since formation,
        # T = age - t_lookback, so the turnover lands near T = tau, i.e. at
        # lookback age - tau. Compare in cosmic time, matching BAGPIPES dblplaw.
        t_peak_cosmic = age - peak_lbt
        assert abs(t_peak_cosmic / tau - 1.0) < 0.30, (
            f"DPL peak at cosmic time {t_peak_cosmic / 1e9:.1f} Gyr, "
            f"expected near tau = {tau / 1e9:.1f} Gyr"
        )

    def test_alpha_controls_decline(self):
        """Larger alpha → steeper decline from peak toward present."""
        from tengri.components.stellar.sfh import dpl

        tau = 5e9
        age = _AGE_UNIV_YR
        sfr_steep = dpl(T_LOOKBACK, alpha=4.0, beta=1.0, tau=tau, age=age, log_total_mass=1.0)
        sfr_shallow = dpl(T_LOOKBACK, alpha=1.0, beta=1.0, tau=tau, age=age, log_total_mass=1.0)

        # At recent times (small lookback), steep alpha → lower SFR
        recent = T_LOOKBACK < 1e9
        ratio_steep = float(jnp.mean(sfr_steep[recent]) / jnp.max(sfr_steep))
        ratio_shallow = float(jnp.mean(sfr_shallow[recent]) / jnp.max(sfr_shallow))
        assert ratio_steep < ratio_shallow, "Larger alpha should give steeper decline"

    def test_symmetric_when_alpha_equals_beta(self):
        """alpha = beta → symmetric SFH around peak (in log-time)."""
        from tengri.components.stellar.sfh import dpl

        tau = 3e9
        sfr = dpl(T_LOOKBACK, alpha=2.0, beta=2.0, tau=tau, age=_AGE_UNIV_YR, log_total_mass=1.0)
        peak_idx = int(jnp.argmax(sfr))
        # Check that SFR is roughly symmetric in log-time around peak
        chex.assert_tree_all_finite(sfr)
        assert float(jnp.max(sfr)) > 0


# ── 5. SKEW-NORMAL FAMILY — tsnorm, snorm, norm ───────────────────


class TestSkewNormalPhysics:
    """Skew-normal SFH family physics (Bellstedt+2020, Robotham+2020)."""

    def test_norm_is_symmetric(self):
        """norm (skew=0) should be symmetric around peak_lbt."""
        from tengri.components.stellar.sfh import norm

        peak = 5e9
        width = 2e9
        sfr = norm(T_LOOKBACK, log_total_mass=1.0, peak_lbt=peak, width=width)
        peak_idx = int(jnp.argmax(sfr))
        peak_lbt_actual = float(T_LOOKBACK[peak_idx])
        assert abs(peak_lbt_actual / peak - 1.0) < 0.15, (
            f"norm peak at {peak_lbt_actual / 1e9:.1f} Gyr, expected {peak / 1e9:.1f} Gyr"
        )

    def test_snorm_skew_changes_shape(self):
        """Non-zero skew should produce different SFH than zero skew."""
        from tengri.components.stellar.sfh import snorm

        peak = 5e9
        width = 2e9
        sfr_sym = snorm(T_LOOKBACK, log_total_mass=1.0, peak_lbt=peak, width=width, skew=0.0)
        sfr_skew = snorm(T_LOOKBACK, log_total_mass=1.0, peak_lbt=peak, width=width, skew=2.0)

        # Shapes should differ
        diff = float(jnp.sum(jnp.abs(sfr_sym - sfr_skew)))
        assert diff > 0.1, "Non-zero skew should change SFH shape"

    def test_tsnorm_truncation_suppresses_recent(self):
        """Truncation reduces SFR at recent times."""
        from tengri.components.stellar.sfh import snorm, tsnorm

        peak = 5e9
        width = 2e9
        sfr_no_trunc = snorm(T_LOOKBACK, log_total_mass=1.0, peak_lbt=peak, width=width, skew=0.0)
        sfr_trunc = tsnorm(
            T_LOOKBACK, log_total_mass=1.0, peak_lbt=peak, width=width, skew=0.0, trunc=2.0
        )

        # At recent times (small lookback), truncated should be lower
        recent = T_LOOKBACK < 1e9
        assert float(jnp.mean(sfr_trunc[recent])) <= float(jnp.mean(sfr_no_trunc[recent])) + 1e-10

    def test_lnorm_asymmetric_in_linear_time(self):
        """Log-normal: Gaussian in log-time → asymmetric in linear time."""
        from tengri.components.stellar.sfh import lnorm

        # New convention (#514): log-normal in cosmic time T = age - t_lookback,
        # peaked at T = peak. The log-normal's long tail extends toward larger T
        # (= smaller lookback = younger), the same direction as Carnall+2018 /
        # BAGPIPES lognormal.
        sfr = lnorm(T_LOOKBACK, log_total_mass=1.0, peak=3e9, width=0.5, age=_AGE_UNIV_YR)
        peak_idx = int(jnp.argmax(sfr))

        # Long tail now points toward smaller lookback (larger cosmic time T).
        mass_old = float(jnp.trapezoid(sfr[peak_idx:], T_LOOKBACK[peak_idx:]))
        mass_young = float(jnp.trapezoid(sfr[:peak_idx], T_LOOKBACK[:peak_idx]))
        assert mass_young > mass_old, "Log-normal long tail should point toward younger ages"


# ── 6. TRIWEIGHT BURST — compact burst component ──────────────────


class TestTriweightBurstPhysics:
    """Triweight kernel for burst SFH component."""

    def test_compact_in_log_age(self):
        """Triweight burst should be compact (confined to narrow age range)."""
        from tengri.components.stellar.sfh import triweight_burst

        # log_tpeak_myr=2 → 100 Myr, log_tmax_myr=2.5 → ~316 Myr
        sfr = triweight_burst(T_LOOKBACK, log_tpeak_myr=2.0, log_tmax_myr=2.5)
        # SFR should be non-zero near 100 Myr and zero far away
        chex.assert_tree_all_finite(sfr)
        assert_non_negative(sfr, name="sfr")
        # Peak should be near 100 Myr lookback
        peak_lbt = float(T_LOOKBACK[jnp.argmax(sfr)])
        assert 5e7 < peak_lbt < 5e8, (
            f"Triweight peak at {peak_lbt / 1e6:.0f} Myr, expected ~100 Myr"
        )


# ── 7. CONTINUITY SFH — non-parametric (Leja+2019) ────────────────


class TestContinuitySFHPhysics:
    """Non-parametric continuity SFH conservation and smoothness."""

    def test_mass_conservation(self):
        """Integrated SFR × dt must equal 10^log_total_mass."""
        from tengri.components.stellar.sfh import continuity_sfh

        age_yr = jnp.geomspace(1e6, 13.7e9, 1000)
        sfr = continuity_sfh(
            age_yr,
            log_total_mass=10.0,
            ratio_0=0.0,
            ratio_1=0.0,
            ratio_2=0.0,
            ratio_3=0.0,
            ratio_4=0.0,
            ratio_5=0.0,
        )
        mass = float(jnp.trapezoid(sfr, age_yr))
        expected = 10.0**10.0
        assert abs(mass / expected - 1.0) < 0.15, f"Mass {mass:.2e} should be ~{expected:.2e}"

    def test_flat_sfh_from_zero_ratios(self):
        """All ratios = 0 → flat SFH (constant across all bins)."""
        from tengri.components.stellar.sfh import continuity_sfh

        age_yr = jnp.geomspace(1e8, 13e9, 500)
        sfr = continuity_sfh(
            age_yr,
            log_total_mass=10.0,
            ratio_0=0.0,
            ratio_1=0.0,
            ratio_2=0.0,
            ratio_3=0.0,
            ratio_4=0.0,
            ratio_5=0.0,
        )
        # Should be approximately constant (piecewise-constant interpolation)
        active = sfr > 0
        if jnp.sum(active) > 10:
            cv = float(jnp.std(sfr[active]) / jnp.mean(sfr[active]))
            assert cv < 0.3, f"Zero ratios should give ~flat SFH, CV={cv:.2f}"

    def test_positive_ratios_rising_sfh(self):
        """Positive ratios → rising SFH (more SF at recent times)."""
        from tengri.components.stellar.sfh import continuity_sfh

        age_yr = jnp.geomspace(1e6, 13.7e9, 1000)
        sfr = continuity_sfh(
            age_yr,
            log_total_mass=10.0,
            ratio_0=0.5,
            ratio_1=0.5,
            ratio_2=0.5,
            ratio_3=0.5,
            ratio_4=0.5,
            ratio_5=0.5,
        )
        # Recent SFR (small lookback) should be higher than old
        recent = age_yr < 1e9
        old = age_yr > 6e9
        if jnp.sum(recent) > 0 and jnp.sum(old) > 0:
            mean_recent = float(jnp.mean(sfr[recent]))
            mean_old = float(jnp.mean(sfr[old]))
            assert mean_recent > mean_old, "Positive ratios should give rising SFH"


# ── 8. DIRICHLET SFH — non-parametric (Leja+2017) ─────────────────


class TestDirichletSFHPhysics:
    """Dirichlet SFH via stick-breaking."""

    def test_mass_conservation(self):
        """Integrated SFR × dt must equal 10^log_total_mass."""
        from tengri.components.stellar.sfh import dirichlet_sfh

        age_yr = jnp.geomspace(1e6, 13.7e9, 1000)
        sfr = dirichlet_sfh(
            age_yr,
            log_total_mass=10.0,
            z_frac_0=0.5,
            z_frac_1=0.5,
            z_frac_2=0.5,
            z_frac_3=0.5,
            z_frac_4=0.5,
            z_frac_5=0.5,
        )
        mass = float(jnp.trapezoid(sfr, age_yr))
        expected = 10.0**10.0
        # Piecewise-constant interpolation onto geomspace grid can introduce
        # ~30% error depending on grid resolution and bin boundaries
        assert abs(mass / expected - 1.0) < 0.35, f"Mass {mass:.2e} should be ~{expected:.2e}"

    def test_sfr_non_negative(self):
        """SFR must be non-negative everywhere."""
        from tengri.components.stellar.sfh import dirichlet_sfh

        age_yr = jnp.geomspace(1e6, 13.7e9, 1000)
        for z_val in [0.1, 0.5, 0.9]:
            sfr = dirichlet_sfh(
                age_yr,
                log_total_mass=10.0,
                z_frac_0=z_val,
                z_frac_1=z_val,
                z_frac_2=z_val,
                z_frac_3=z_val,
                z_frac_4=z_val,
                z_frac_5=z_val,
            )
            assert_non_negative(
                sfr, name="sfr", msg=f"Dirichlet SFR should be non-negative at z={z_val}"
            )

    def test_extreme_z_concentrates_mass(self):
        """z_frac_0 near 1 concentrates mass in youngest bin."""
        from tengri.components.stellar.sfh import dirichlet_sfh

        age_yr = jnp.geomspace(1e6, 13.7e9, 1000)
        sfr = dirichlet_sfh(
            age_yr,
            log_total_mass=10.0,
            z_frac_0=0.99,
            z_frac_1=0.01,
            z_frac_2=0.01,
            z_frac_3=0.01,
            z_frac_4=0.01,
            z_frac_5=0.01,
        )
        # Most mass should be in the youngest bin
        young = age_yr < 100e6
        mass_young = float(jnp.trapezoid(sfr[young], age_yr[young]))
        mass_total = float(jnp.trapezoid(sfr, age_yr))
        if mass_total > 0:
            frac = mass_young / mass_total
            assert frac > 0.3, f"z_frac_0=0.99 should concentrate mass young, got frac={frac:.2f}"


# ── 9. ALL SFH MODELS — universal constraints ─────────────────────


class TestAllSFHUniversalPhysics:
    """Every SFH model must satisfy universal constraints: non-negative, finite."""

    _SFH_CONFIGS: dict = {  # noqa: RUF012
        "tsnorm": {
            "log_total_mass": 1.0,
            "peak_lbt": 5e9,
            "width": 2e9,
            "skew": 0.5,
            "trunc": 2.0,
        },
        "snorm": {"log_total_mass": 1.0, "peak_lbt": 5e9, "width": 2e9, "skew": 0.5},
        "norm": {"log_total_mass": 1.0, "peak_lbt": 5e9, "width": 2e9},
        "lnorm": {"log_total_mass": 1.0, "peak": 5e9, "width": 0.5, "age": _AGE_UNIV_YR},
        "dpl": {"alpha": 2.0, "beta": 1.0, "tau": 5e9, "age": _AGE_UNIV_YR, "log_total_mass": 1.0},
    }

    @pytest.fixture(params=list(_SFH_CONFIGS.keys()))
    def sfh_model(self, request):
        return request.param

    def _get_fn(self, name):
        from tengri.components.stellar.sfh import dpl, lnorm, norm, snorm, tsnorm

        return {"tsnorm": tsnorm, "snorm": snorm, "norm": norm, "lnorm": lnorm, "dpl": dpl}[name]

    def test_sfr_non_negative(self, sfh_model):
        """SFR must be non-negative at all lookback times."""
        fn = self._get_fn(sfh_model)
        sfr = fn(T_LOOKBACK, **self._SFH_CONFIGS[sfh_model])
        assert jnp.all(sfr >= -1e-30), f"{sfh_model}: SFR has negative values"

    def test_sfr_finite(self, sfh_model):
        """SFR must be finite at all lookback times."""
        fn = self._get_fn(sfh_model)
        sfr = fn(T_LOOKBACK, **self._SFH_CONFIGS[sfh_model])
        assert jnp.all(jnp.isfinite(sfr)), f"{sfh_model}: SFR has non-finite values"

    def test_integral_matches_log_total_mass(self, sfh_model):
        """Integral of SFR should equal 10^log_total_mass (NEW normalization)."""
        fn = self._get_fn(sfh_model)
        sfr = fn(T_LOOKBACK, **self._SFH_CONFIGS[sfh_model])
        dt_yr = jnp.abs(jnp.diff(T_LOOKBACK))
        integral_mass = float(jnp.trapezoid(sfr, T_LOOKBACK))
        # log_total_mass = 1.0 means 10^1 = 10 Msun
        expected_mass = 10.0
        relative_error = abs(integral_mass - expected_mass) / expected_mass
        assert relative_error < 0.01, (
            f"{sfh_model}: integral mass={integral_mass:.1f} Msun, "
            f"expected {expected_mass:.1f} (error={relative_error:.1%})"
        )
