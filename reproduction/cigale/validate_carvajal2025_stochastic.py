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

FINDING -- same PSD family, different time coordinate
-----------------------------------------------------
The two methods use the SAME DRW/bending-power-law PSD, and they agree in dex
variance, but they apply it in DIFFERENT time coordinates:

  * tengri ``field``:  DRW in LOG-AGE  (u = log10 t)  -> scale-free burstiness;
    the correlation length in linear time GROWS with the age at which it sits.
  * CIGALE Carvajal:   DRW in LINEAR TIME (Myr)        -> one fixed decorrelation
    timescale, independent of age.

So a "matched" pair agrees in the amplitude sense but their autocorrelation
structures differ by construction -- a genuine methodological distinction, not a
bug. Two convention factors must be handled to compare them at all:

  amplitude:  CIGALE ``sigma`` IS std(log10 SFR) (post-hoc normalized);
              tengri dex std is LINEAR in ``psd_sigma`` (~0.30 * psd_sigma at
              tau=150 Myr on the default log-age grid) -- set psd_sigma to hit
              the target dex.
  timescale:  CIGALE's 1/e decorrelation is ``tau_break / (2*pi)`` (corner-
              frequency convention), NOT ``tau_break``; tengri's is expressed in
              dex (log-age) and set by ``psd_tau_yr``.

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
            psd_sigma, tau_yr, N_GRID, d_log)[0])
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

    # 2. amplitude: reach a common target dex variance
    target = 0.30
    _, _, gp1 = tengri_ensemble(1.0, tau_yr)
    psd_sigma = target / (gp1.std() / LN10)
    grid, d_log, gp = tengri_ensemble(psd_sigma, tau_yr)
    t_dex = gp.std() / LN10
    t_myr, lcs = cigale_ensemble(150.0, target)
    c_dex = lcs.std()
    print(f"[2] amplitude  tengri psd_sigma={psd_sigma:.3f} -> {t_dex:.3f} dex ;"
          f" CIGALE sigma={target} -> {c_dex:.3f} dex   (both hit target {target})")

    # 3. CIGALE timescale convention
    for tb in (50.0, 150.0):
        _, l = cigale_ensemble(tb, 0.3)
        a = np.mean([acf(x) for x in l], axis=0)
        print(f"[3] timescale  CIGALE tau_break={tb:.0f} Myr -> 1/e decorr"
              f" {decorr_lag(a, 1.0):.1f} Myr   (tau_break/2pi={tb / 2 / np.pi:.1f})")

    # 4. coordinate difference (the finding)
    at = np.mean([acf(g) for g in gp], axis=0)
    ac = np.mean([acf(x) for x in lcs], axis=0)
    tdec = decorr_lag(at, d_log)
    print(f"[4] coordinate tengri decorr = {tdec:.3f} dex (LOG-AGE, scale-free) ;"
          f" CIGALE decorr = {decorr_lag(ac, 1.0):.1f} Myr (LINEAR-TIME, fixed)")
    print(f"    a {tdec:.2f}-dex lag spans {100 * (10**tdec - 1):.0f} Myr at age 100 Myr"
          f" but {1000 * (10**tdec - 1):.0f} Myr at age 1 Gyr (age-dependent).")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for g in gp[:30]:
        ax[0].plot(grid, g / LN10, color="C0", alpha=0.15, lw=0.6)
    ax[0].set(title="tengri field: log-age DRW", xlabel="log10(age/yr)",
              ylabel="log10 SFR modulation [dex]")
    for x in lcs[:30]:
        ax[1].plot(t_myr, x, color="C1", alpha=0.15, lw=0.6)
    ax[1].set(title="CIGALE Carvajal: linear-time DRW", xlabel="lookback [Myr]",
              ylabel="log10 SFR modulation [dex]")
    # ACF shapes on each code's own lag index (visual comparison only -- the
    # x-axes are DIFFERENT coordinates: dex for tengri, Myr for CIGALE).
    nlag = 60
    ax[2].plot(np.arange(nlag), at[:nlag], "C0", label="tengri (lag in dex)")
    ax[2].plot(np.arange(nlag), ac[:nlag], "C1", label="CIGALE (lag in Myr)", alpha=0.7)
    ax[2].axhline(1 / np.e, ls=":", c="k", lw=1)
    ax[2].set(title="autocorrelation (native coords)", xlabel="lag [grid steps]",
              ylabel="ACF"); ax[2].legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "_figs",
                       "cigale_carvajal2025_stochastic_sfh.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nfigure -> {out}")
    print("\nConclusion: same DRW PSD family + matched dex variance, but tengri applies")
    print("it in log-age (scale-free) and CIGALE in linear time (fixed) -- attributed,")
    print("not a bug. Frozen in tests/crossval/test_carvajal2025_stochastic_sfh_crossval.py.")


if __name__ == "__main__":
    main()
