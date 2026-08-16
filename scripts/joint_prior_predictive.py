"""Prior predictive separability test: does joint mode carry σ_PSD info?

For each σ_PSD ∈ {0.5, 1.0, 2.0, 3.5} we draw N mock galaxies from the prior,
holding ``sfh_field_psd_sigma`` fixed. The joint observable is:
    - 10-band rich photometry (FUV/NUV/SDSS/JHKs)
    - 4 emission-line fluxes (Hα, Hβ, [OIII] 5007, [OII] 3727), integrated
      from the SSP-baked spectrum at line centers
       — same recipe as ``benchmark_population_native.py``.

Two diagnostic plots:

1. **Per-observable distributions**: for each of the 14 observable components,
   overlay histograms across the four σ_PSD values. Heavy overlap = no info.

2. **Population-statistics vs σ_PSD**: for each observable, plot
   ``std-across-galaxies`` and ``mean-across-galaxies`` vs σ_PSD. If the std
   *grows* with σ_PSD, that observable carries information; if it's flat, it
   doesn't.

KS-test p-values quantify per-observable separability.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

import tengri  # noqa: F401  (auto-enables persistent cache)
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
LINE_WAVES_REST_AA = jnp.array([6564.61, 4862.68, 5008.24, 3727.09])  # vacuum

SIGMA_VALUES = (0.5, 1.0, 2.0, 3.5)
N_MOCK = 256
Z_FIX = 0.1
LINE_WINDOW_AA = 30.0
LINE_NPIX = 41

OUT_DIR = Path("analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_spec(psd_sigma_fixed: float) -> Parameters:
    """Spec with σ_PSD pinned; everything else from prior."""
    return Parameters(
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Uniform(-3.0, 3.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        sfh_field_psd_sigma=Fixed(psd_sigma_fixed),
        sfh_field_psd_tau_myr=Fixed(20.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(Z_FIX),
        mean_sfh_type=["tsnorm", "field"],
        n_grid=64,
    )


def build_joint_predict(model: SEDModel):
    """Return jitted ``params -> (phot[10], line_flux[4])``."""
    line_centers_obs = LINE_WAVES_REST_AA * (1.0 + Z_FIX)
    waves_per_line = jnp.stack(
        [jnp.linspace(c - LINE_WINDOW_AA, c + LINE_WINDOW_AA, LINE_NPIX) for c in line_centers_obs]
    )  # (4, 41)
    waves_concat = waves_per_line.reshape(-1)

    @jax.jit
    def joint_predict(params):
        phot = model.predict_photometry(params)
        spec = model.predict_spectrum(params, waves_concat)
        spec_per = spec.reshape(LINE_WAVES_REST_AA.shape[0], LINE_NPIX)
        cont = 0.5 * (spec_per[:, 0] + spec_per[:, -1])

        def _trap(f, w, c):
            return jnp.trapezoid(f - c, w)

        line_flux = jax.vmap(_trap)(spec_per, waves_per_line, cont)
        return phot, line_flux

    return joint_predict


def draw_mocks(sigma: float, ssp_data, obs: Observation, n_mock: int = N_MOCK):
    spec = make_spec(sigma)
    model = SEDModel(spec, ssp_data, observation=obs)
    predict = build_joint_predict(model)

    key = jax.random.PRNGKey(42)
    phots = np.zeros((n_mock, len(FILTERS)))
    lines = np.zeros((n_mock, len(LINE_NAMES)))

    for i in range(n_mock):
        k = jax.random.fold_in(key, i)
        params = model.spec.sample(k)
        params["sfh_field_psd_sigma"] = jnp.array(sigma)
        params["sfh_field_psd_tau_myr"] = jnp.array(20.0)
        phot, line_flux = predict(params)
        phots[i] = np.asarray(phot)
        lines[i] = np.asarray(line_flux)

    return phots, lines


def plot_separability(all_phot, all_lines):
    """3×5 grid: histogram per observable, one curve per σ value."""
    fig, axes = plt.subplots(3, 5, figsize=(16, 9))
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(SIGMA_VALUES)))

    n_phot = len(FILTERS)
    for i, name in enumerate(FILTERS + LINE_NAMES):
        ax = axes[i]
        for j, sigma in enumerate(SIGMA_VALUES):
            if i < n_phot:
                arr = all_phot[sigma][:, i]
            else:
                arr = all_lines[sigma][:, i - n_phot]
            log_vals = np.log10(np.abs(arr) + 1e-40)
            log_vals = log_vals[np.isfinite(log_vals)]
            ax.hist(
                log_vals,
                bins=30,
                alpha=0.45,
                color=colors[j],
                label=rf"$\sigma_{{\rm PSD}}={sigma}$",
            )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(r"$\log_{10}$(flux)", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for k in range(n_phot + len(LINE_NAMES), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        "Joint-mode prior predictive: per-observable distribution by "
        r"$\sigma_{\rm PSD}$  (N=256 mocks each)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "joint_prior_predictive_hist.png", dpi=130)
    fig.savefig(OUT_DIR / "joint_prior_predictive_hist.pdf")
    plt.close(fig)


def plot_population_stats(all_phot, all_lines):
    """std-across-galaxies and mean vs σ_PSD per observable."""
    n_phot = len(FILTERS)
    n_obs = n_phot + len(LINE_NAMES)
    sigmas = np.array(SIGMA_VALUES)

    stds = np.zeros((n_obs, len(sigmas)))
    means = np.zeros((n_obs, len(sigmas)))

    for i, name in enumerate(FILTERS + LINE_NAMES):
        for j, sigma in enumerate(sigmas):
            if i < n_phot:
                arr = all_phot[sigma][:, i]
            else:
                arr = all_lines[sigma][:, i - n_phot]
            log_vals = np.log10(np.abs(arr) + 1e-40)
            log_vals = log_vals[np.isfinite(log_vals)]
            stds[i, j] = np.std(log_vals)
            means[i, j] = np.mean(log_vals)

    fig, (ax_std, ax_mean) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.cm.tab20

    for i, name in enumerate(FILTERS + LINE_NAMES):
        c = cmap(i / n_obs)
        ls = "-" if i < n_phot else "--"
        ax_std.plot(sigmas, stds[i], "o", linestyle=ls, color=c, label=name, markersize=5)
        ax_mean.plot(
            sigmas, means[i] - means[i, 0], "o", linestyle=ls, color=c, label=name, markersize=5
        )

    ax_std.set_xlabel(r"$\sigma_{\rm PSD}$ (mock truth)")
    ax_std.set_ylabel(r"std across galaxies of $\log_{10}$(observable)")
    ax_std.set_title("Population scatter vs σ_PSD\n(growth = info available)")
    ax_std.grid(alpha=0.3)
    ax_std.legend(fontsize=7, ncol=2, loc="best")

    ax_mean.set_xlabel(r"$\sigma_{\rm PSD}$ (mock truth)")
    ax_mean.set_ylabel(
        r"$\Delta$ mean $\log_{10}$(observable)  (relative to $\sigma_{\rm PSD}=0.5$)"
    )
    ax_mean.set_title("Mean shift vs σ_PSD\n(systematic = identifiable)")
    ax_mean.grid(alpha=0.3)
    ax_mean.axhline(0, color="black", lw=0.5)

    fig.suptitle(
        "If population scatter does not grow with σ_PSD, the snapshot data\n"
        "carries no information beyond the prior — this is the symptom.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "joint_prior_predictive_popstats.png", dpi=130)
    fig.savefig(OUT_DIR / "joint_prior_predictive_popstats.pdf")
    plt.close(fig)

    return stds, means


def report_ks(all_phot, all_lines):
    n_phot = len(FILTERS)
    pairs = [(SIGMA_VALUES[0], SIGMA_VALUES[-1]), (SIGMA_VALUES[1], SIGMA_VALUES[2])]

    print("\n=== KS test (per-observable, log10 of flux) ===")
    print(f"{'observable':18s} | " + "  |  ".join([f"σ={a}↔{b} (KS, p)" for a, b in pairs]))
    for i, name in enumerate(FILTERS + LINE_NAMES):
        out = []
        for sig_a, sig_b in pairs:
            if i < n_phot:
                xa = all_phot[sig_a][:, i]
                xb = all_phot[sig_b][:, i]
            else:
                xa = all_lines[sig_a][:, i - n_phot]
                xb = all_lines[sig_b][:, i - n_phot]
            la = np.log10(np.abs(xa) + 1e-40)
            lb = np.log10(np.abs(xb) + 1e-40)
            la = la[np.isfinite(la)]
            lb = lb[np.isfinite(lb)]
            ks_stat, p = ks_2samp(la, lb)
            out.append(f"{ks_stat:5.2f},{p:.2e}")
        print(f"{name:18s} | " + "  |  ".join(out))


def main() -> None:
    print("Loading SSP data...")
    ssp_data = load_ssp_data(SSP_FILE)
    obs = Observation(photometry=Photometry.from_names(FILTERS))

    all_phot: dict[float, np.ndarray] = {}
    all_lines: dict[float, np.ndarray] = {}

    for sigma in SIGMA_VALUES:
        print(f"\n=== Drawing N={N_MOCK} mocks at σ_PSD={sigma} ===")
        phot, lines = draw_mocks(sigma, ssp_data, obs)
        all_phot[sigma] = phot
        all_lines[sigma] = lines
        # Sanity: a single observable's median + spread
        print(
            f"  log10 Hα flux: median={np.median(np.log10(np.abs(lines[:, 0]) + 1e-40)):.3f}, "
            f"std={np.std(np.log10(np.abs(lines[:, 0]) + 1e-40)):.3f}"
        )

    plot_separability(all_phot, all_lines)
    print(f"\nWrote {OUT_DIR / 'joint_prior_predictive_hist.png'}")

    stds, means = plot_population_stats(all_phot, all_lines)
    print(f"Wrote {OUT_DIR / 'joint_prior_predictive_popstats.png'}")

    report_ks(all_phot, all_lines)

    # Headline summary: max relative change in scatter across σ range
    n_phot = len(FILTERS)
    print("\n=== Population-scatter sensitivity (std growth ratio σ=3.5 / σ=0.5) ===")
    for i, name in enumerate(FILTERS + LINE_NAMES):
        ratio = stds[i, -1] / max(stds[i, 0], 1e-6)
        marker = "  <-- info!" if ratio > 1.5 else ""
        print(f"  {name:18s}  std ratio = {ratio:5.2f}{marker}")


if __name__ == "__main__":
    main()
