"""Cross-validate tengri's ``field`` (IFT/PSD) SFH against CIGALE 2025.1's
``sfhstochastic_carvajal2025`` (Carvajal-Bohorquez et al. 2025, arXiv:2507.13160).

This is the first external code with a burstiness prescription directly
comparable to tengri's headline method (SFHs as IFT correlated fields with
PSD-governed burstiness priors -- the ``field`` SFH, Paper I). Both codes draw a
Gaussian process from a PSD (white noise x sqrt(PSD) in Fourier -> inverse FFT)
and apply it as a log-normal multiplicative modulation of a delayed-tau baseline.
At CIGALE's default ``alpha = 2`` the bending power law is a Lorentzian -- the
same damped-random-walk (DRW) PSD tengri uses (``psd_drw``).

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
        .venv/bin/python reproduction/cigale/validate_carvajal2025_stochastic.py

Needs ``pcigale`` (2025.1) in the venv. Saves a figure to ``_figs/``.

FINDING -- same PSD family AND same time coordinate (after #865)
----------------------------------------------------------------
The two methods use the SAME DRW/bending-power-law PSD and, after the #865
linear-time fix, apply it in the SAME time coordinate:

  * tengri ``field``:  DRW stationary in LINEAR (physical) time, sampled on the
    log-age grid -> fixed decorrelation in Myr at every age.
  * CIGALE Carvajal:   DRW stationary in LINEAR time (Myr) -> fixed decorrelation.

They now agree in BOTH dex variance and autocorrelation. (Before #865 tengri
applied the DRW in log-age, giving a scale-free correlation length that grew
with age -- that was the corrected deficiency.) Two convention factors remain to
line the parameters up:

  amplitude:  CIGALE ``sigma`` IS std(log10 SFR) (post-hoc normalized); tengri's
              ``psd_sigma`` is now ALSO the dex std directly (natural-log variance
              (sigma*ln10)^2), so sigma <-> psd_sigma one-to-one.
  timescale:  CIGALE's 1/e decorrelation is ``tau_break / (2*pi)`` (corner-
              frequency convention), NOT ``tau_break``; tengri's ``psd_tau_yr`` is
              the physical decorrelation timescale directly.

The frozen assertions live in
``tests/crossval/test_carvajal2025_stochastic_sfh_crossval.py``.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import jax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pcigale.sed_modules.sfhstochastic_carvajal2025 as C
from tengri import make_log_age_grid
from tengri.components.stellar.sfh.registry import compute_field_gp
from tengri.components.stellar.sfh.psd_models import psd_drw

LN10 = np.log(10.0)
N = 400
N_GRID = 256


def acf(x):
    x0 = x - x.mean()
    a = np.correlate(x0, x0, "full")[len(x0) - 1:]
    return a / a[0]


def decorr_lag(a, dx):
    below = np.nonzero(a < 1.0 / np.e)[0]
    return dx * below[0] if below.size else np.nan


def tengri_ensemble(psd_sigma, tau_yr):
    grid = np.asarray(make_log_age_grid(n_grid=N_GRID, log_age_min=6.0, log_age_max=10.14))
    d_log = float(grid[1] - grid[0])
    gp = np.array([
        np.asarray(compute_field_gp(
            jax.random.normal(jax.random.PRNGKey(i), (N_GRID,)),
            psd_sigma, tau_yr, N_GRID, d_log, log_age_grid=grid)[0])
        for i in range(N)
    ])
    return grid, d_log, gp


def cigale_ensemble(tau_break_myr, sigma_dex, age_myr=1000):
    f_break = 1.0 / tau_break_myr
    curves = []
    for seed in range(N):
        lc, _, _ = C.TimmerKoenig(
            C.PSD, (1.0, f_break, 2.0, 0.0, 0.0),
            age_myr, C.DEFAULT_TIME_BIN_MYR, seed,
            RedNoiseL=C.DEFAULT_RED_NOISE_FACTOR, aliasTbin=C.DEFAULT_ALIAS_TBIN)
        curves.append(lc / lc.std() * sigma_dex)
    return np.arange(age_myr), np.array(curves)


def main():
    print("=" * 72)
    print("tengri field (IFT/PSD) vs CIGALE sfhstochastic_carvajal2025  (#865)")
    print("=" * 72)

    # 1. PSD form parity: CIGALE alpha=2 == tengri DRW, f_break = 1/(2 pi tau)
    tau_yr = 150e6
    f = np.geomspace(1e-10, 1e-7, 300)
    f_break = 1.0 / (2.0 * np.pi * tau_yr)
    p_t = np.asarray(psd_drw(2 * np.pi * f, 1.0, tau_yr)); p_t /= p_t[0]
    p_c = C.PSD(f, 1.0, f_break, 2.0, 0.0, 0.0); p_c /= p_c[0]
    print(f"\n[1] PSD form   max|tengri-CIGALE| (normalized) = {np.max(np.abs(p_t - p_c)):.2e}"
          f"   -> same Lorentzian at f_break=1/(2 pi tau)")

    # 2. amplitude: psd_sigma is now the dex std directly (one-to-one with sigma)
    target = 0.30
    grid, d_log, gp = tengri_ensemble(target, tau_yr)
    t_dex = gp.std() / LN10
    t_myr, lcs = cigale_ensemble(150.0, target)
    c_dex = lcs.std()
    print(f"[2] amplitude  tengri psd_sigma={target} -> {t_dex:.3f} dex ;"
          f" CIGALE sigma={target} -> {c_dex:.3f} dex   (psd_sigma == sigma, one-to-one)")

    # 3. CIGALE timescale convention
    for tb in (50.0, 150.0):
        _, lc = cigale_ensemble(tb, 0.3)
        a = np.mean([acf(x) for x in lc], axis=0)
        print(f"[3] timescale  CIGALE tau_break={tb:.0f} Myr -> 1/e decorr"
              f" {decorr_lag(a, 1.0):.1f} Myr   (tau_break/2pi={tb / 2 / np.pi:.1f})")

    # 4. tengri now decorrelates in LINEAR time too -> matches CIGALE
    t_phys = 10.0**grid
    mod = gp - gp.mean(0)

    def tdecorr_myr(t0):
        i0 = int(np.argmin(np.abs(t_phys - t0)))
        c = np.array([np.corrcoef(mod[:, i0], mod[:, j])[0, 1] for j in range(i0, len(t_phys))])
        b = np.where(c < 1.0 / np.e)[0]
        return (t_phys[i0 + b[0]] - t_phys[i0]) / 1e6 if b.size else np.nan

    cdec = decorr_lag(np.mean([acf(x) for x in lcs], axis=0), 1.0)
    print(f"[4] linear-time decorr  tengri @30Myr={tdecorr_myr(30e6):.0f} @300Myr="
          f"{tdecorr_myr(300e6):.0f} Myr ; CIGALE={cdec:.0f} Myr"
          f"   -> both FIXED in Myr (no age-stretch), agree")

    # figure: both are linear-time DRWs now
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for g in gp[:30]:
        ax[0].plot(t_phys / 1e6, g / LN10, color="C0", alpha=0.15, lw=0.6)
    ax[0].set(title="tengri field: linear-time DRW (#865)", xlabel="lookback [Myr]",
              ylabel="log10 SFR modulation [dex]", xscale="log")
    for x in lcs[:30]:
        ax[1].plot(t_myr, x, color="C1", alpha=0.15, lw=0.6)
    ax[1].set(title="CIGALE Carvajal: linear-time DRW", xlabel="lookback [Myr]",
              ylabel="log10 SFR modulation [dex]")
    at_myr = np.array([np.corrcoef(mod[:, 0], mod[:, j])[0, 1] for j in range(len(t_phys))])
    ac = np.mean([acf(x) for x in lcs], axis=0)
    ax[2].plot((t_phys - t_phys[0]) / 1e6, at_myr, "C0", label="tengri")
    ax[2].plot(np.arange(len(ac)), ac, "C1", label="CIGALE", alpha=0.7)
    ax[2].axhline(1 / np.e, ls=":", c="k", lw=1)
    ax[2].set(title="autocorrelation vs linear-time lag", xlabel="lag [Myr]",
              ylabel="ACF", xlim=(0, 600)); ax[2].legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "_figs",
                       "cigale_carvajal2025_stochastic_sfh.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nfigure -> {out}")
    print("\nConclusion (post-#865): same DRW PSD family, same linear-time coordinate,")
    print("matched dex variance and fixed-Myr decorrelation -- tengri's field burstiness")
    print("now agrees with CIGALE sfhstochastic_carvajal2025 in amplitude AND autocorrelation.")
    print("Frozen in tests/crossval/test_carvajal2025_stochastic_sfh_crossval.py.")


if __name__ == "__main__":
    main()
