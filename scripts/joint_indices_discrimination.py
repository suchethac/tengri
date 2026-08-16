"""How much does adding absorption indices help discriminate σ_PSD and τ_PSD?

Compares four observable scenarios for population-level discrimination of
(σ_PSD, τ_PSD), all under nuisance prior marginalization (Monte Carlo over
peak-SFR, dust, metallicity, etc.):

    A. Hα/FUV ratio alone     (1 number per galaxy)
       — classic Weisz/Faisst/Wang burstiness diagnostic, ~nuisance-clean.

    B. Joint  (current)       (14 numbers: 10 phot bands + 4 line fluxes)

    C. Joint + Lick indices   (19: B + D4000, Mg b, Hβ_abs, Hδ_A, Hα/FUV)

    D. Full spectrum          (binned 4000-7500 Å @ 30 Å = 117 pixels)
       — Burnham et al. 2026 use spectra of similar info content at z~4.

For each scenario we report combined population z-score for σ=1↔2 (truth=2)
and τ=20↔100 (truth=20) at N ∈ {128, 512, 2048}, and projected N for 3σ.

The key experiment: as we add more spectral info, does the τ_PSD
discrimination floor drop toward the Burnham-class N≈500 regime?
"""

from __future__ import annotations

import os
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

# Lick-style absorption indices (rest-frame). Definitions from Worthey/Lick:
#   feat: feature band [lo, hi] in Å
#   blue, red: pseudo-continuum bands
# D4000 (Bruzual 1983, Balogh narrow Dn4000 variant)
INDEX_DEFS = {
    "D4000": {"feat": (4050, 4250), "blue": (3750, 3950), "red": None},
    "Hdelta_A": {"feat": (4083, 4122), "blue": (4041, 4079), "red": (4128, 4161)},
    "Hbeta_abs": {"feat": (4848, 4877), "blue": (4827, 4847), "red": (4877, 4892)},
    "Mgb": {"feat": (5160, 5193), "blue": (5142, 5161), "red": (5191, 5206)},
}
# We'll sample each band at 21 points -> max 4 indices * 3 bands * 21 = 252 pts
INDEX_NPIX = 21

# Full-spectrum bins (rest-frame): 4000-7500 Å @ 30 Å = 117 pixels
FULL_SPEC_REST = jnp.linspace(4000.0, 7500.0, 117)
FULL_SPEC_OBS = FULL_SPEC_REST * (1.0 + Z_FIX)

SIGMA_GRID = (1.0, 2.0)  # truth=2, alt=1
TAU_GRID = (20.0, 100.0)  # truth=20, alt=100
M_MOCK = 1024

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


def _build_index_band_grid() -> tuple[jnp.ndarray, dict]:
    """Concat the wavelength grid for all index bands (observed-frame).

    Returns (wave_grid, layout) where layout describes slice indices."""
    layout: dict = {}
    waves = []
    cursor = 0
    for name, defs in INDEX_DEFS.items():
        layout[name] = {}
        for band_key in ("feat", "blue", "red"):
            if defs[band_key] is None:
                continue
            lo, hi = defs[band_key]
            obs = jnp.linspace(lo, hi, INDEX_NPIX) * (1.0 + Z_FIX)
            layout[name][band_key] = (cursor, cursor + INDEX_NPIX)
            waves.append(obs)
            cursor += INDEX_NPIX
    return jnp.concatenate(waves), layout


def _compute_indices(
    spec_at_index_grid: jnp.ndarray, layout: dict, wave_grid: jnp.ndarray
) -> jnp.ndarray:
    """Compute Lick-style indices from spectrum sampled at all index bands.

    For 3-band Lick: index = ∫(1 - F/F_pseudo) dλ over feat
    For 2-band D4000-like (no red): D4000 = mean(F_red_band) / mean(F_blue_band)
    Returns array shape (n_indices,) in input order.
    """
    out = []
    names = list(INDEX_DEFS.keys())
    for name in names:
        feat_slice = layout[name]["feat"]
        feat_F = spec_at_index_grid[feat_slice[0] : feat_slice[1]]
        feat_w = wave_grid[feat_slice[0] : feat_slice[1]]
        blue_slice = layout[name]["blue"]
        blue_F = spec_at_index_grid[blue_slice[0] : blue_slice[1]]
        blue_w = wave_grid[blue_slice[0] : blue_slice[1]]
        if name == "D4000":
            # Bruzual-style ratio
            d4 = jnp.mean(feat_F * feat_w**2) / jnp.maximum(jnp.mean(blue_F * blue_w**2), 1e-40)
            out.append(d4)
        else:
            red_slice = layout[name]["red"]
            red_F = spec_at_index_grid[red_slice[0] : red_slice[1]]
            red_w = wave_grid[red_slice[0] : red_slice[1]]
            blue_mid = 0.5 * (blue_w[0] + blue_w[-1])
            red_mid = 0.5 * (red_w[0] + red_w[-1])
            blue_F_mean = jnp.mean(blue_F)
            red_F_mean = jnp.mean(red_F)
            # Linear pseudo-continuum across feat
            slope = (red_F_mean - blue_F_mean) / (red_mid - blue_mid)
            cont_at_feat = blue_F_mean + slope * (feat_w - blue_mid)
            ew = jnp.trapezoid(1.0 - feat_F / jnp.maximum(cont_at_feat, 1e-40), feat_w)
            out.append(ew)
    return jnp.array(out)


def build_extended_predict(model: SEDModel):
    """Return predict(params) -> (phot, line_flux, indices, full_spec)."""
    line_centers_obs = LINE_WAVES_REST_AA * (1.0 + Z_FIX)
    waves_per_line = jnp.stack(
        [jnp.linspace(c - LINE_WINDOW_AA, c + LINE_WINDOW_AA, LINE_NPIX) for c in line_centers_obs]
    )
    waves_lines_concat = waves_per_line.reshape(-1)

    waves_index, index_layout = _build_index_band_grid()
    waves_full = FULL_SPEC_OBS

    # All wavelengths concat for one predict call
    n_lines = waves_lines_concat.shape[0]
    n_index = waves_index.shape[0]
    n_full = waves_full.shape[0]
    waves_all = jnp.concatenate([waves_lines_concat, waves_index, waves_full])

    @jax.jit
    def predict(params):
        phot = model.predict_photometry(params)
        spec_all = model.predict_spectrum(params, waves_all)
        spec_lines = spec_all[:n_lines]
        spec_indices = spec_all[n_lines : n_lines + n_index]
        spec_full = spec_all[n_lines + n_index :]

        spec_per_line = spec_lines.reshape(LINE_WAVES_REST_AA.shape[0], LINE_NPIX)
        cont = 0.5 * (spec_per_line[:, 0] + spec_per_line[:, -1])
        line_flux = jax.vmap(lambda f, w, c: jnp.trapezoid(f - c, w))(
            spec_per_line, waves_per_line, cont
        )

        indices = _compute_indices(spec_indices, index_layout, waves_index)

        return phot, line_flux, indices, spec_full

    return predict


def draw_pool(sigma: float, tau: float, ssp_data, obs: Observation):
    spec = make_spec(sigma, tau)
    model = SEDModel(spec, ssp_data, observation=obs)
    predict = build_extended_predict(model)

    key = jax.random.PRNGKey(int(1000 * sigma + tau))
    n_phot = len(FILTERS)
    n_lines = len(LINE_NAMES)
    n_idx = len(INDEX_DEFS)
    n_full = FULL_SPEC_OBS.shape[0]

    arr_phot = np.zeros((M_MOCK, n_phot))
    arr_lines = np.zeros((M_MOCK, n_lines))
    arr_indices = np.zeros((M_MOCK, n_idx))
    arr_full = np.zeros((M_MOCK, n_full))

    for i in range(M_MOCK):
        k = jax.random.fold_in(key, i)
        params = model.spec.sample(k)
        params["sfh_field_psd_sigma"] = jnp.array(sigma)
        params["sfh_field_psd_tau_myr"] = jnp.array(tau)
        phot, lines, indices, full = predict(params)
        arr_phot[i] = np.asarray(phot)
        arr_lines[i] = np.asarray(lines)
        arr_indices[i] = np.asarray(indices)
        arr_full[i] = np.asarray(full)
    return {"phot": arr_phot, "lines": arr_lines, "indices": arr_indices, "full": arr_full}


# ── Observable-set assembly ─────────────────────────────────────────────────


def make_observable_set(pool: dict, scenario: str) -> np.ndarray:
    """Return (M, n_obs) array of log10 observables for the given scenario."""
    log_phot = np.log10(np.abs(pool["phot"]) + 1e-40)
    log_lines = np.log10(np.abs(pool["lines"]) + 1e-40)
    indices = pool["indices"]  # not in log; D4000 is a ratio, EWs are signed
    log_full = np.log10(np.abs(pool["full"]) + 1e-40)

    halpha_idx = LINE_NAMES.index("Halpha")
    fuv_idx = FILTERS.index("galex_fuv")
    halpha_uv = log_lines[:, halpha_idx : halpha_idx + 1] - log_phot[:, fuv_idx : fuv_idx + 1]

    if scenario == "halpha_uv":
        return halpha_uv
    if scenario == "joint":
        return np.concatenate([log_phot, log_lines], axis=1)
    if scenario == "joint_indices":
        return np.concatenate([log_phot, log_lines, indices, halpha_uv], axis=1)
    if scenario == "full_spec":
        return log_full
    raise ValueError(scenario)


SCENARIO_LABELS = {
    "halpha_uv": "Hα/FUV alone (1 obs)",
    "joint": "joint: 10 phot + 4 lines (14)",
    "joint_indices": "joint + Lick indices + Hα/FUV (19)",
    "full_spec": f"full spectrum, 4000-7500Å @ 30Å ({FULL_SPEC_OBS.shape[0]})",
}
SCENARIOS = list(SCENARIO_LABELS.keys())


# ── Z-score machinery ──────────────────────────────────────────────────────


def z_per_obs(arr_a: np.ndarray, arr_b: np.ndarray, n: int) -> np.ndarray:
    mu_a = np.nanmean(arr_a, axis=0)
    mu_b = np.nanmean(arr_b, axis=0)
    s_a = np.nanstd(arr_a, axis=0)
    s_b = np.nanstd(arr_b, axis=0)
    se = np.sqrt((s_a**2 + s_b**2) / n)
    return (mu_a - mu_b) / np.maximum(se, 1e-20)


def combined_z(z: np.ndarray) -> float:
    return float(np.sqrt(np.nansum(z**2)))


def project_n(arr_a: np.ndarray, arr_b: np.ndarray, z_target: float = 3.0) -> float:
    n_ref = 100
    z_ref = combined_z(z_per_obs(arr_a, arr_b, n_ref))
    if z_ref <= 0:
        return float("inf")
    return n_ref * (z_target / z_ref) ** 2


def main() -> None:
    print("Loading SSP data...")
    ssp_data = load_ssp_data(SSP_FILE)
    obs = Observation(photometry=Photometry.from_names(FILTERS))

    cells = [(s, t) for s in SIGMA_GRID for t in TAU_GRID]
    pools: dict = {}
    for sigma, tau in cells:
        print(f"  M={M_MOCK} mocks  σ={sigma}  τ={tau} Myr ...")
        pools[(sigma, tau)] = draw_pool(sigma, tau, ssp_data, obs)

    # ── Discrimination tests ─────────────────────────────────────────────
    sigma_pair = ((1.0, 20.0), (2.0, 20.0))  # σ=1 vs σ=2 at truth-τ
    tau_pair = ((2.0, 20.0), (2.0, 100.0))  # τ=20 vs τ=100 at truth-σ

    print("\n" + "=" * 90)
    print("σ_PSD discrimination: σ=1 vs σ=2 at τ=20 Myr  (nuisance-marginalized)")
    print("=" * 90)
    print(f"{'scenario':40s} | {'z@128':>7s} | {'z@512':>7s} | {'z@2048':>7s} | {'N for 3σ':>10s}")
    sigma_results = {}
    for sc in SCENARIOS:
        a = make_observable_set(pools[sigma_pair[0]], sc)
        b = make_observable_set(pools[sigma_pair[1]], sc)
        z128 = combined_z(z_per_obs(a, b, 128))
        z512 = combined_z(z_per_obs(a, b, 512))
        z2k = combined_z(z_per_obs(a, b, 2048))
        n3 = project_n(a, b)
        sigma_results[sc] = (z128, z512, z2k, n3)
        print(f"{SCENARIO_LABELS[sc]:40s} | {z128:7.2f} | {z512:7.2f} | {z2k:7.2f} | {n3:10.0f}")

    print("\n" + "=" * 90)
    print("τ_PSD discrimination: τ=20 vs τ=100 Myr at σ=2.0  (nuisance-marginalized)")
    print("=" * 90)
    print(f"{'scenario':40s} | {'z@128':>7s} | {'z@512':>7s} | {'z@2048':>7s} | {'N for 3σ':>10s}")
    tau_results = {}
    for sc in SCENARIOS:
        a = make_observable_set(pools[tau_pair[0]], sc)
        b = make_observable_set(pools[tau_pair[1]], sc)
        z128 = combined_z(z_per_obs(a, b, 128))
        z512 = combined_z(z_per_obs(a, b, 512))
        z2k = combined_z(z_per_obs(a, b, 2048))
        n3 = project_n(a, b)
        tau_results[sc] = (z128, z512, z2k, n3)
        print(f"{SCENARIO_LABELS[sc]:40s} | {z128:7.2f} | {z512:7.2f} | {z2k:7.2f} | {n3:10.0f}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    n_grid = np.logspace(1, 5, 80)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(SCENARIOS)))

    for ax, results, title in [
        (axes[0], sigma_results, r"σ_PSD: $\sigma=1 \leftrightarrow \sigma=2$ at $\tau=20$ Myr"),
        (axes[1], tau_results, r"τ_PSD: $\tau=20 \leftrightarrow \tau=100$ Myr at $\sigma=2$"),
    ]:
        for c, sc in zip(colors, SCENARIOS):
            z_at_128 = results[sc][0]
            curve = z_at_128 * np.sqrt(n_grid / 128.0)
            n3 = results[sc][3]
            label = SCENARIO_LABELS[sc] + f"   N₃σ={n3:.0f}"
            ax.plot(n_grid, curve, color=c, label=label, lw=1.6)
        ax.axhline(3, color="black", ls="--", lw=0.8, alpha=0.6, label="3σ threshold")
        ax.axhline(5, color="gray", ls=":", lw=0.8, alpha=0.5, label="5σ threshold")
        ax.axvline(500, color="orange", ls=":", lw=0.8, alpha=0.5, label="Burnham N=500")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (galaxies)")
        ax.set_ylabel("Combined z (nuisance-marginalized)")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        "Discrimination scaling vs observable richness  "
        "(prior-marginalized over peak SFR, dust, metallicity)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "joint_indices_discrimination.png", dpi=130)
    fig.savefig(OUT_DIR / "joint_indices_discrimination.pdf")
    plt.close(fig)

    print(f"\nWrote {OUT_DIR}/joint_indices_discrimination.png")


if __name__ == "__main__":
    main()
