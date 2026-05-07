"""
Age-Dust-Metallicity Degeneracy with UV Break
===============================================

Demonstrates the famous age-dust-metallicity degeneracy in broadband
photometry. Two galaxies with identical SDSS ugriz photometry — one old
and dust-poor, one young and dust-rich — reveal dramatically different
stellar ages, dust content, and metallicities. Adding GALEX FUV/NUV
photometry breaks this degeneracy, illustrating why UV coverage is
critical for accurate stellar population age dating.

Synthetic data: SDSS+GALEX matched at z=0.1.
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
    setup_style,
)

setup_style()


def _find_ssp():
    """Locate SSP data from project root or docs/ (sphinx-gallery) cwd."""
    name = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    for p in [
        Path("data") / name,
        Path("../data") / name,
        Path("../../data") / name,
        Path("../../../data") / name,
    ]:
        if p.exists():
            return str(p)
    return None


SSP_PATH = _find_ssp()
if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

ssp = load_ssp_data(SSP_PATH)

_FILTER_DIR = next(
    (
        str(d)
        for d in [
            Path("data/filters"),
            Path("../data/filters"),
            Path("../../data/filters"),
            Path("../../../data/filters"),
        ]
        if d.exists()
    ),
    "data/filters",
)

# --- Setup: SDSS 5-band model ---
spec_sdss = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

obs_sdss = Observation(
    photometry=Photometry.from_names(
        ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"], cache_dir=_FILTER_DIR
    ),
)

model_sdss = SEDModel(spec_sdss, ssp, observation=obs_sdss)

# Setup: SDSS + GALEX model ---
obs_uv = Observation(
    photometry=Photometry.from_names(
        ["galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
        cache_dir=_FILTER_DIR,
    ),
)

model_uv = SEDModel(spec_sdss, ssp, observation=obs_uv)

# --- Generate two-galaxy scenario ---
key = jax.random.PRNGKey(42)

# Galaxy A: old, dust-free, high metallicity
params_a = {
    "sfh_tsnorm_peak_lbt_gyr": 9.0,  # Peak at 9 Gyr ago (old)
    "sfh_tsnorm_width_gyr": 0.5,  # Single brief burst
    "sfh_tsnorm_log_peak_sfr": 0.5,
    "sfh_tsnorm_skew": -0.5,
    "sfh_tsnorm_trunc": 2.0,
    "met_logzsol": 0.0,  # Solar metallicity
    "dust_tau_bc": 0.05,  # Minimal dust
    "dust_tau_diff": 0.02,
}

# Galaxy B: young, dusty, low metallicity
params_b = {
    "sfh_tsnorm_peak_lbt_gyr": 2.0,  # Peak at 2 Gyr ago (young)
    "sfh_tsnorm_width_gyr": 1.5,  # Extended burst
    "sfh_tsnorm_log_peak_sfr": -0.5,
    "sfh_tsnorm_skew": 0.5,
    "sfh_tsnorm_trunc": 4.0,
    "met_logzsol": -0.5,  # Sub-solar
    "dust_tau_bc": 1.5,  # Heavy dust
    "dust_tau_diff": 0.8,
}

# Generate SDSS photometry (match both galaxies)
mock_a_sdss = model_sdss.mock(params_a, snr=50.0, key=jax.random.fold_in(key, 0))
mock_b_sdss = model_sdss.mock(params_b, snr=50.0, key=jax.random.fold_in(key, 1))

# Scale galaxy B to match SDSS flux of galaxy A (degeneracy)
flux_ratio = np.mean(np.array(mock_a_sdss.flux_obs) / np.array(mock_b_sdss.flux_obs))
scaled_flux_b = np.array(mock_b_sdss.flux_obs) * flux_ratio
scaled_noise_b = np.array(mock_b_sdss.noise) * flux_ratio

# Generate UV photometry with original params
mock_a_uv = model_uv.mock(params_a, snr=50.0, key=jax.random.fold_in(key, 2))
mock_b_uv = model_uv.mock(params_b, snr=50.0, key=jax.random.fold_in(key, 3))

# --- Figure: left panel (SDSS), right panel (SDSS+GALEX) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# --- Left: SDSS only (degenerate) ---
wave_sdss = np.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0])
band_names = ["u", "g", "r", "i", "z"]

# Plot both galaxies' SDSS photometry (indistinguishable)
ax1.errorbar(
    wave_sdss,
    mock_a_sdss.flux_obs,
    yerr=mock_a_sdss.noise,
    fmt="o",
    color="C0",
    ms=7,
    capsize=4,
    lw=2.0,
    label="Galaxy A (old+dust-free)",
    zorder=5,
)
ax1.errorbar(
    wave_sdss,
    scaled_flux_b,
    yerr=scaled_noise_b,
    fmt="s",
    color="C3",
    ms=7,
    capsize=4,
    lw=2.0,
    alpha=0.7,
    label="Galaxy B (young+dusty)",
    zorder=4,
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax1.set_ylabel(r"$f_\nu$ [arbitrary]", fontsize=11)
ax1.set_title("SDSS photometry only\n(indistinguishable)", fontsize=12, fontweight="bold")
ax1.legend(frameon=False, fontsize=9.5, loc="upper right")
ax1.grid(True, alpha=0.3, which="both")
ax1.set_xticks(wave_sdss)
ax1.set_xticklabels(band_names)

# --- Right: SDSS + GALEX (degenerate broken) ---
wave_galex = np.array([1516.0, 2267.0])  # FUV, NUV rest-frame
wave_uv_combined = np.concatenate([wave_galex, wave_sdss])
band_names_uv = ["FUV", "NUV", *band_names]

flux_a_uv_all = np.concatenate([mock_a_uv.flux_obs[:2], mock_a_sdss.flux_obs])
noise_a_uv_all = np.concatenate([mock_a_uv.noise[:2], mock_a_sdss.noise])

# Scale galaxy B's UV to match SDSS (partial matching)
flux_b_uv_all = np.concatenate([mock_b_uv.flux_obs[:2], scaled_flux_b])
noise_b_uv_all = np.concatenate([mock_b_uv.noise[:2], scaled_noise_b])

ax2.errorbar(
    wave_uv_combined,
    flux_a_uv_all,
    yerr=noise_a_uv_all,
    fmt="o",
    color="C0",
    ms=7,
    capsize=4,
    lw=2.0,
    label="Galaxy A (old, clean)",
    zorder=5,
)
ax2.errorbar(
    wave_uv_combined,
    flux_b_uv_all,
    yerr=noise_b_uv_all,
    fmt="s",
    color="C3",
    ms=7,
    capsize=4,
    lw=2.0,
    alpha=0.7,
    label="Galaxy B (young, dusty)",
    zorder=4,
)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
ax2.set_ylabel(r"$f_\nu$ [arbitrary]", fontsize=11)
ax2.set_title("SDSS + GALEX photometry\n(degeneracy broken)", fontsize=12, fontweight="bold")
ax2.legend(frameon=False, fontsize=9.5, loc="upper right")
ax2.grid(True, alpha=0.3, which="both")
ax2.set_xticks(wave_uv_combined)
ax2.set_xticklabels(band_names_uv)

fig.suptitle(
    "Age-Dust-Metallicity Degeneracy: UV as a Diagnostic",
    fontsize=13,
    fontweight="bold",
    y=1.00,
)
fig.tight_layout()

outdir = (
    Path(__file__).resolve().parent.parent.parent / "figures" if "__file__" in dir() else Path(".")
)
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(outdir / "usecase_age_dust_degeneracy.png"), dpi=150, bbox_inches="tight")
plt.show()
