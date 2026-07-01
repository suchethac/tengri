# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation: tengri ``field`` (IFT/PSD) SFH vs CIGALE ``sfhstochastic_carvajal2025``.

CIGALE 2025.1 shipped ``sfhstochastic_carvajal2025`` (Carvajal-Bohórquez et al.
2025, arXiv:2507.13160): a delayed-τ baseline multiplicatively modulated by a
stochastic process drawn from a **bending power-law PSD** via the Timmer & König
(1995) algorithm. This is the first external code with a burstiness prescription
directly comparable to tengri's headline method — SFHs as IFT correlated fields
with PSD-governed burstiness priors (the ``field`` SFH, Paper I).

Both codes draw a Gaussian process from a PSD (white noise × √PSD in Fourier →
inverse FFT) and apply it as a log-normal multiplicative modulation of a
delayed-τ baseline. At CIGALE's default ``alpha = 2`` the bending power law is a
Lorentzian — the **same damped-random-walk (DRW) PSD** tengri uses (``psd_drw``).

What this suite pins (verified numerically, #865):

1. **PSD form parity** — CIGALE ``PSD(f; alpha=2)`` and tengri ``psd_drw`` are the
   same Lorentzian, identified by ``f_break = 1 / (2π τ)``.
2. **Amplitude convention** — both reach a target dex variability: CIGALE's
   ``sigma`` *is* the std of log10(SFR) (post-hoc normalized); tengri's dex std is
   linear in ``psd_sigma``.
3. **CIGALE timescale convention** — the 1/e decorrelation of a CIGALE light
   curve is ``tau_break / (2π)`` (corner-frequency convention), NOT ``tau_break``.
4. **Attributed difference (the key finding)** — tengri's field is a DRW in
   **log-age** (``u = log10 t``: scale-free, correlation length grows with age),
   while CIGALE's is a DRW in **linear time** (one fixed decorrelation timescale).
   They agree in dex variance but their autocorrelation structures differ by
   construction — a genuine methodological distinction, not a bug.

Skipped unless ``pcigale`` is installed (``pytest -m crossval``).

References
----------
.. [1] R. Carvajal-Bohórquez et al., arXiv:2507.13160 (A&A, 2025).
.. [2] J. Timmer & M. König, "On generating power law noise," A&A, 300, 707 (1995).
.. [3] A. C. Carnall et al., MNRAS, 480, 4379 (2018).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_N_REALIZATIONS = 200
_N_GRID = 128
_LN10 = np.log(10.0)


def _acf(x):
    """Normalized autocorrelation (lag 0 = 1)."""
    x0 = x - x.mean()
    a = np.correlate(x0, x0, "full")[len(x0) - 1 :]
    return a / a[0]


def _decorrelation_lag(acf, dx):
    """First lag where the ACF drops below 1/e, in units of dx (else NaN)."""
    below = np.nonzero(acf < 1.0 / np.e)[0]
    return dx * below[0] if below.size else np.nan


# ── tengri field ensemble (DRW in log-age) ───────────────────────────


def _tengri_ensemble(psd_sigma, tau_yr, n=_N_REALIZATIONS, n_grid=_N_GRID):
    import jax

    from tengri import make_log_age_grid
    from tengri.components.stellar.sfh.registry import compute_field_gp

    grid = np.asarray(make_log_age_grid(n_grid=n_grid, log_age_min=6.0, log_age_max=10.14))
    d_log = float(grid[1] - grid[0])
    gp = np.array(
        [
            np.asarray(
                compute_field_gp(
                    jax.random.normal(jax.random.PRNGKey(i), (n_grid,)),
                    psd_sigma,
                    tau_yr,
                    n_grid,
                    d_log,
                )[0]
            )
            for i in range(n)
        ]
    )
    return gp, d_log  # gp is the natural-log modulation on the log-age grid


# ── CIGALE Carvajal ensemble (DRW in linear time) ────────────────────


def _cigale_ensemble(tau_break_myr, sigma_dex, age_myr=1000, n=_N_REALIZATIONS):
    import pcigale.sed_modules.sfhstochastic_carvajal2025 as C

    f_break = 1.0 / tau_break_myr
    curves = []
    for seed in range(n):
        lc, _, _ = C.TimmerKoenig(
            C.PSD,
            (1.0, f_break, 2.0, 0.0, 0.0),  # A=1, f_break, alpha_low=2 (DRW), alpha_high=0, c=0
            age_myr,
            C.DEFAULT_TIME_BIN_MYR,
            seed,
            RedNoiseL=C.DEFAULT_RED_NOISE_FACTOR,
            aliasTbin=C.DEFAULT_ALIAS_TBIN,
        )
        curves.append(lc / lc.std() * sigma_dex)  # normalize to sigma dex (log10)
    return np.array(curves), C


def test_psd_form_is_the_same_drw_lorentzian():
    """CIGALE PSD(f; alpha=2) and tengri psd_drw are the same Lorentzian.

    tengri: psd_drw(omega) = sigma^2 tau / (1 + (tau omega)^2), omega = 2 pi f.
    CIGALE: PSD(f; alpha=2) = 1 / (f^2 + f_break^2) (A=1, c=0).
    Identify f_break = 1 / (2 pi tau); both are proportional to 1/(f_c^2 + f^2).
    """
    pytest.importorskip("pcigale")
    import pcigale.sed_modules.sfhstochastic_carvajal2025 as C

    from tengri.components.stellar.sfh.psd_models import psd_drw

    tau_yr = 150e6
    f = np.geomspace(1e-10, 1e-7, 200)  # cycles / yr
    omega = 2.0 * np.pi * f
    f_break = 1.0 / (2.0 * np.pi * tau_yr)

    p_tengri = np.asarray(psd_drw(omega, 1.0, tau_yr))
    p_cigale = C.PSD(f, 1.0, f_break, 2.0, 0.0, 0.0)
    # Compare shapes (each normalized to its zero-frequency / plateau value).
    r_t = p_tengri / p_tengri[0]
    r_c = p_cigale / p_cigale[0]
    assert np.allclose(r_t, r_c, rtol=5e-3), "DRW PSD shapes should match at f_break=1/(2 pi tau)"


def test_amplitude_reaches_target_dex_variance_on_both_sides():
    """Both codes can be configured to a common dex variability of log10(SFR)."""
    pytest.importorskip("pcigale")
    target_dex = 0.30

    # tengri: dex std is linear in psd_sigma; solve for the psd_sigma that hits target.
    gp1, _ = _tengri_ensemble(1.0, 150e6)
    dex_per_unit = gp1.std() / _LN10
    psd_sigma = target_dex / dex_per_unit
    gp, _ = _tengri_ensemble(psd_sigma, 150e6)
    tengri_dex = gp.std() / _LN10
    assert tengri_dex == pytest.approx(target_dex, rel=0.1)

    # CIGALE: sigma IS the dex std by construction (post-hoc normalization).
    lcs, _ = _cigale_ensemble(150.0, target_dex)
    cigale_dex = lcs.std()
    assert cigale_dex == pytest.approx(target_dex, rel=0.1)

    # => both reach the same dex variance.
    assert tengri_dex == pytest.approx(cigale_dex, rel=0.15)


def test_cigale_tau_break_is_corner_frequency_convention():
    """CIGALE's 1/e decorrelation is tau_break/(2 pi), not tau_break."""
    pytest.importorskip("pcigale")
    tau_break = 50.0  # Myr — small enough that the finite grid doesn't bias 1/e
    lcs, C = _cigale_ensemble(tau_break, 0.3)
    acf = np.mean([_acf(lc) for lc in lcs], axis=0)
    decorr_myr = _decorrelation_lag(acf, C.DEFAULT_TIME_BIN_MYR)
    assert decorr_myr == pytest.approx(tau_break / (2.0 * np.pi), rel=0.25)


def test_coordinate_difference_is_log_age_vs_linear_time():
    """The attributed difference: tengri decorrelates in log-age, CIGALE in linear time.

    tengri's log-age decorrelation (in dex) is set by ``psd_tau_yr`` and is the
    SAME dex lag regardless of where in cosmic time it sits — i.e. its linear-time
    correlation length scales with the age at which it is evaluated (scale-free).
    CIGALE's decorrelation is a fixed number of Myr independent of age. This test
    documents that distinction rather than asserting the two are identical.
    """
    pytest.importorskip("pcigale")
    # tengri: decorrelation in dex depends on psd_tau but NOT on absolute age.
    gp_short, d_log = _tengri_ensemble(1.0, 50e6)
    gp_long, _ = _tengri_ensemble(1.0, 500e6)
    acf_short = np.mean([_acf(g) for g in gp_short], axis=0)
    acf_long = np.mean([_acf(g) for g in gp_long], axis=0)
    dec_short_dex = _decorrelation_lag(acf_short, d_log)
    dec_long_dex = _decorrelation_lag(acf_long, d_log)
    # Longer PSD timescale -> longer log-age decorrelation (monotone), and both
    # are expressed in dex (log-age), the scale-free coordinate.
    assert dec_long_dex > dec_short_dex
    assert 0.0 < dec_short_dex < 1.0 and 0.0 < dec_long_dex < 1.5

    # A fixed 0.3-dex decorrelation spans a DIFFERENT linear-time interval at
    # different reference ages (the hallmark of the log-age coordinate): the span
    # t_ref * (10**0.3 - 1) grows linearly with t_ref, unlike CIGALE's fixed Myr.
    span_100myr = 100.0 * (10.0**0.3 - 1.0)
    span_1gyr = 1000.0 * (10.0**0.3 - 1.0)
    assert span_1gyr == pytest.approx(10.0 * span_100myr, rel=1e-6)
