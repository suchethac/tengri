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
2. **Amplitude convention** — both use the same meaning of ``sigma``: after the
   #865 linear-time fix, tengri's ``psd_sigma`` *is* the std of log10(SFR) in dex,
   identical to CIGALE's ``sigma`` (post-hoc normalized).
3. **CIGALE timescale convention** — the 1/e decorrelation of a CIGALE light
   curve is ``tau_break / (2π)`` (corner-frequency convention), NOT ``tau_break``.
4. **Agreement (the key result)** — tengri's field is now a DRW stationary in
   **linear (physical) time**, the same as CIGALE: its decorrelation is a fixed
   number of Myr at every age (where the log grid resolves it), not a fixed
   number of dex. Before #865 tengri applied the DRW in log-age (scale-free,
   correlation length growing with age); that was corrected so the two codes'
   burstiness prescriptions now match in both amplitude and autocorrelation.

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
                    log_age_grid=grid,
                )[0]
            )
            for i in range(n)
        ]
    )
    # gp is the natural-log modulation on the log-age grid; the process is now a
    # DRW stationary in LINEAR time, so return the physical times too.
    return gp, grid, 10.0**grid


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


def test_amplitude_is_dex_directly_on_both_sides():
    """psd_sigma is now the dex std of log10(SFR) directly — matching CIGALE.

    After #865's linear-time fix the field GP's natural-log variance is exactly
    (psd_sigma * ln10)^2, so tengri's ``psd_sigma`` == std of log10(SFR) in dex,
    the same meaning as CIGALE's ``sigma``.
    """
    pytest.importorskip("pcigale")
    target_dex = 0.30

    gp, _, _ = _tengri_ensemble(target_dex, 150e6)
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


def test_tengri_decorrelation_is_fixed_in_linear_time_like_cigale():
    """tengri's field now decorrelates at a FIXED physical timescale, matching CIGALE.

    After #865's linear-time fix, the field GP is a DRW stationary in cosmic time,
    so its 1/e decorrelation is ~the same number of Myr at young and old ages
    (where the log grid resolves it) — NOT a fixed number of dex that stretches
    with age. This is the parity confirmation: both codes are linear-time DRWs.
    """
    pytest.importorskip("pcigale")
    tau_myr = 150.0
    gp, _grid, t = _tengri_ensemble(0.3, tau_myr * 1e6)
    mod = gp - gp.mean(axis=0)

    def phys_decorr_myr(t0):
        i0 = int(np.argmin(np.abs(t - t0)))
        corr = np.array([np.corrcoef(mod[:, i0], mod[:, j])[0, 1] for j in range(i0, len(t))])
        below = np.nonzero(corr < 1.0 / np.e)[0]
        return (t[i0 + below[0]] - t[i0]) / 1e6 if below.size else np.nan

    dec_young = phys_decorr_myr(30e6)  # 30 Myr — grid step << tau, well resolved
    dec_mid = phys_decorr_myr(300e6)  # 300 Myr — still resolved
    # Fixed in LINEAR time: young and mid decorrelations agree within a factor ~2
    # (residual is grid resolution, not the age-stretch of the old log-age model,
    # which drifted by ~10x). Both are of order the physical tau, not << or >>.
    assert 0.3 * tau_myr < dec_young < 3.0 * tau_myr
    assert 0.3 * tau_myr < dec_mid < 3.0 * tau_myr
    # must not scale ~10x with age (the old log-age behavior)
    assert dec_mid / dec_young < 3.0
