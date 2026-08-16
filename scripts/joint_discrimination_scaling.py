"""How well does joint-mode (rich phot + 4 lines) discriminate σ_PSD and τ_PSD?

Methodology
-----------
For each (σ_PSD, τ_PSD) in a 2D grid we draw M=2048 mocks from the prior
(holding σ, τ pinned) and compute the joint observable per galaxy. We then
ask: given a population of N galaxies, at what N does the population mean
of each observable distinguish (σ_A, τ) from (σ_B, τ) — and similarly for τ?

Per-observable z-score for a 2-sample test on log10(flux):

    z(N) = (μ_A - μ_B) / sqrt((s_A^2 + s_B^2) / N)

where μ, s are the per-galaxy log-flux mean and std evaluated under the
same nuisance prior.

The combined z over all 14 observables (assuming independence — upper bound
on info) is sqrt(Σ z_i^2). Threshold: z=3 → 3σ discrimination.

Two figures:
- ``joint_disc_sigma.png`` : z(N) curves for (σ=1 vs σ=2) at multiple τ values
- ``joint_disc_tau.png``   : z(N) curves for (τ=20 vs τ=100, etc.) at σ=2

Also reports projected N for 3σ per pair, which is the key "how much would
you need" number.

Reference
---------
Burnham et al. 2026 (arXiv:2601.20930) demonstrate that with full JWST/NIRSpec
spectra at z~4, N≈500 is enough to distinguish FIRE-2 vs Illustris-like SFR
PSDs at >99% confidence. Their per-galaxy info is much higher than ours
(full spectra ≫ 10 photo bands + 4 line fluxes); this script quantifies
how much of the gap our joint mode closes vs photometry alone.
"""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri  # noqa: F401
from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
from tengri.sps.dsps_wrapper import load_ssp_data

jax.config.update("jax_enable_x64", True)

SSP_FILE = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
FILTERS = [
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
]
LINE_NAMES = ["Halpha", "Hbeta", "OIII_5007", "OII_3727"]
LINE_WAVES_REST_AA = jnp.array([6564.61, 4862.68, 5008.24, 3727.09])

Z_FIX = 0.1
LINE_WINDOW_AA = 30.0
LINE_NPIX = 41

# (σ, τ) grid for prior-predictive draws
SIGMA_GRID = (0.5, 1.0, 2.0, 3.0)
TAU_GRID = (5.0, 20.0, 100.0, 300.0)
M_MOCK = 1024  # mocks per cell

OUT_DIR = Path("analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_spec(psd_sigma: float, psd_tau_myr: float) -> Parameters:
    return Parameters(
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Uniform(-3.0, 3.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        sfh_field_psd_sigma=Fixed(psd_sigma),
        sfh_field_psd_tau_myr=Fixed(psd_tau_myr),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(Z_FIX),
        mean_sfh_type=["tsnorm", "field"],
        n_grid=64,
    )


def build_joint_predict(model: SEDModel):
    line_centers_obs = LINE_WAVES_REST_AA * (1.0 + Z_FIX)
    waves_per_line = jnp.stack(
        [jnp.linspace(c - LINE_WINDOW_AA, c + LINE_WINDOW_AA, LINE_NPIX) for c in line_centers_obs]
    )
    waves_concat = waves_per_line.reshape(-1)

    @jax.jit
    def joint_predict(params):
        phot = model.predict_photometry(params)
        spec = model.predict_spectrum(params, waves_concat)
        spec_per = spec.reshape(LINE_WAVES_REST_AA.shape[0], LINE_NPIX)
        cont = 0.5 * (spec_per[:, 0] + spec_per[:, -1])
        line_flux = jax.vmap(lambda f, w, c: jnp.trapezoid(f - c, w))(
            spec_per, waves_per_line, cont
        )
        return phot, line_flux

    return joint_predict


def draw_pool(sigma: float, tau: float, ssp_data, obs: Observation, m: int = M_MOCK):
    spec = make_spec(sigma, tau)
    model = SEDModel(spec, ssp_data, observation=obs)
    predict = build_joint_predict(model)

    key = jax.random.PRNGKey(int(1000 * sigma + tau))
    log_obs = np.zeros((m, len(FILTERS) + len(LINE_NAMES)))
    for i in range(m):
        k = jax.random.fold_in(key, i)
        params = model.spec.sample(k)
        params["sfh_field_psd_sigma"] = jnp.array(sigma)
        params["sfh_field_psd_tau_myr"] = jnp.array(tau)
        phot, line_flux = predict(params)
        full = np.concatenate([np.asarray(phot), np.asarray(line_flux)])
        log_obs[i] = np.log10(np.abs(full) + 1e-40)
    return log_obs


def z_per_obs(pool_a: np.ndarray, pool_b: np.ndarray, n: int) -> np.ndarray:
    """Per-observable z-score for distinguishing pool_a from pool_b at sample size N."""
    mu_a = np.nanmean(pool_a, axis=0)
    mu_b = np.nanmean(pool_b, axis=0)
    s_a = np.nanstd(pool_a, axis=0)
    s_b = np.nanstd(pool_b, axis=0)
    se = np.sqrt((s_a**2 + s_b**2) / n)
    return (mu_a - mu_b) / np.maximum(se, 1e-20)


def combined_z(z_obs: np.ndarray) -> float:
    """Combined z assuming independent observables (upper bound on info)."""
    return float(np.sqrt(np.nansum(z_obs**2)))


def project_n_for_threshold(
    pool_a: np.ndarray, pool_b: np.ndarray, z_target: float = 3.0
) -> float:
    """N where combined z reaches z_target. Closed-form: combined z scales
    as sqrt(N), so N_target = N_ref * (z_target / z_at_N_ref)^2."""
    n_ref = 100
    z_ref = combined_z(z_per_obs(pool_a, pool_b, n_ref))
    if z_ref <= 0:
        return float("inf")
    return n_ref * (z_target / z_ref) ** 2


def plot_disc_curves(pairs: list[tuple], pools: dict, label_for: callable, title: str, fname: str):
    """Z vs N curves for a list of (key_a, key_b) pairs."""
    fig, (ax_combined, ax_per_obs) = plt.subplots(1, 2, figsize=(14, 5.5))
    n_grid = np.logspace(1, 5, 50)  # N = 10 .. 1e5

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(pairs)))

    n_obs_total = len(FILTERS) + len(LINE_NAMES)

    for c, (key_a, key_b) in zip(colors, pairs):
        z_at_100 = z_per_obs(pools[key_a], pools[key_b], 100)
        cz_at_100 = combined_z(z_at_100)
        # combined z scales as sqrt(N/100)
        cz_curve = cz_at_100 * np.sqrt(n_grid / 100)
        n_3sig = project_n_for_threshold(pools[key_a], pools[key_b], 3.0)
        ax_combined.plot(
            n_grid,
            cz_curve,
            color=c,
            label=f"{label_for(key_a)} vs {label_for(key_b)}  (N₃σ={n_3sig:.0f})",
        )

        # Per-observable bar (z @ N=500, Burnham-comparable)
        z500 = z_per_obs(pools[key_a], pools[key_b], 500)
        ax_per_obs.plot(
            np.arange(n_obs_total),
            np.abs(z500),
            "o-",
            color=c,
            label=f"{label_for(key_a)} vs {label_for(key_b)}",
            markersize=4,
        )

    ax_combined.axhline(3, color="black", lw=0.8, ls="--", alpha=0.6, label="3σ threshold")
    ax_combined.axhline(5, color="gray", lw=0.8, ls=":", alpha=0.5, label="5σ threshold")
    ax_combined.set_xscale("log")
    ax_combined.set_yscale("log")
    ax_combined.set_xlabel("N (galaxies)")
    ax_combined.set_ylabel("Combined z (sqrt of sum of squared per-obs z)")
    ax_combined.set_title("Combined population z vs N (assumes obs independent)")
    ax_combined.grid(alpha=0.3, which="both")
    ax_combined.legend(fontsize=8, loc="lower right")

    ax_per_obs.axhline(1, color="gray", ls=":", alpha=0.5)
    ax_per_obs.axhline(3, color="black", ls="--", alpha=0.6)
    ax_per_obs.set_xticks(np.arange(n_obs_total))
    ax_per_obs.set_xticklabels(FILTERS + LINE_NAMES, rotation=45, ha="right", fontsize=8)
    ax_per_obs.set_ylabel("|z| at N=500")
    ax_per_obs.set_title("Per-observable contribution at N=500\n(Burnham et al. 2026 use N≈500)")
    ax_per_obs.grid(alpha=0.3)
    ax_per_obs.legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / fname, dpi=130)
    fig.savefig(OUT_DIR / fname.replace(".png", ".pdf"))
    plt.close(fig)


def main() -> None:
    print("Loading SSP data...")
    ssp_data = load_ssp_data(SSP_FILE)
    obs = Observation(photometry=Photometry.from_names(FILTERS))

    pools: dict = {}
    for sigma in SIGMA_GRID:
        for tau in TAU_GRID:
            print(f"\n=== Drawing M={M_MOCK} mocks at σ={sigma}, τ={tau} Myr ===")
            pools[(sigma, tau)] = draw_pool(sigma, tau, ssp_data, obs)

    # σ-discrimination at fixed τ=20 (the truth from earlier benches)
    sigma_pairs_at_tau20 = [
        ((0.5, 20.0), (1.0, 20.0)),
        ((0.5, 20.0), (2.0, 20.0)),
        ((0.5, 20.0), (3.0, 20.0)),
        ((1.0, 20.0), (2.0, 20.0)),
        ((1.0, 20.0), (3.0, 20.0)),
        ((2.0, 20.0), (3.0, 20.0)),
    ]
    plot_disc_curves(
        sigma_pairs_at_tau20,
        pools,
        lambda k: f"σ={k[0]}",
        title=r"σ_PSD discrimination at fixed τ=20 Myr  (joint = 10 phot + 4 lines)",
        fname="joint_disc_sigma.png",
    )

    # τ-discrimination at fixed σ=2.0 (the truth from earlier benches)
    tau_pairs_at_sig2 = [
        ((2.0, 5.0), (2.0, 20.0)),
        ((2.0, 5.0), (2.0, 100.0)),
        ((2.0, 5.0), (2.0, 300.0)),
        ((2.0, 20.0), (2.0, 100.0)),
        ((2.0, 20.0), (2.0, 300.0)),
        ((2.0, 100.0), (2.0, 300.0)),
    ]
    plot_disc_curves(
        tau_pairs_at_sig2,
        pools,
        lambda k: f"τ={k[1]:.0f}",
        title=r"τ_PSD discrimination at fixed σ=2.0  (joint = 10 phot + 4 lines)",
        fname="joint_disc_tau.png",
    )

    # === Headline summary table ===
    print("\n=== σ_PSD discrimination at τ=20 Myr ===")
    print(
        f"{'pair':30s} | {'z@N=128':>9s} | {'z@N=512':>9s} | {'z@N=2048':>9s} | {'N for 3σ':>10s}"
    )
    for ka, kb in sigma_pairs_at_tau20:
        z128 = combined_z(z_per_obs(pools[ka], pools[kb], 128))
        z512 = combined_z(z_per_obs(pools[ka], pools[kb], 512))
        z2k = combined_z(z_per_obs(pools[ka], pools[kb], 2048))
        n3 = project_n_for_threshold(pools[ka], pools[kb], 3.0)
        print(
            f"σ={ka[0]}↔{kb[0]} (τ=20)              | "
            f"{z128:9.2f} | {z512:9.2f} | {z2k:9.2f} | {n3:10.0f}"
        )

    print("\n=== τ_PSD discrimination at σ=2.0 ===")
    print(
        f"{'pair':30s} | {'z@N=128':>9s} | {'z@N=512':>9s} | {'z@N=2048':>9s} | {'N for 3σ':>10s}"
    )
    for ka, kb in tau_pairs_at_sig2:
        z128 = combined_z(z_per_obs(pools[ka], pools[kb], 128))
        z512 = combined_z(z_per_obs(pools[ka], pools[kb], 512))
        z2k = combined_z(z_per_obs(pools[ka], pools[kb], 2048))
        n3 = project_n_for_threshold(pools[ka], pools[kb], 3.0)
        print(
            f"τ={ka[1]:.0f}↔{kb[1]:.0f} Myr (σ=2)            | "
            f"{z128:9.2f} | {z512:9.2f} | {z2k:9.2f} | {n3:10.0f}"
        )

    print(f"\nFigures: {OUT_DIR}/joint_disc_sigma.png  and  {OUT_DIR}/joint_disc_tau.png")


if __name__ == "__main__":
    main()
