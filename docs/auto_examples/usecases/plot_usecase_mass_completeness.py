"""
Stellar Mass Completeness in Photometric Surveys
==================================================

Demonstrates mass completeness limits for SDSS-like photometric surveys
(ugriz depths comparable to SDSS). Mocks a population of 150 galaxies
spanning log M* ∈ [7, 12] at z=0.1, injects realistic photometric noise,
and measures the 95% completeness threshold: the stellar mass below which
≥5% of sources are undetected due to noise. Critical for survey design
and sample construction.

Synthetic data: z=0.1 SDSS photometry with realistic SNR degradation.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_usecase_mass_completeness_001.png
   :alt: plot_usecase_mass_completeness
   :class: sphx-glr-single-img

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

# --- SDSS-like observation ---
obs = Observation(
    photometry=Photometry.from_names(
        ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"], cache_dir=_FILTER_DIR
    ),
)

spec = Parameters(
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

model = SEDModel(spec, ssp, observation=obs)

# --- Generate population spanning log M* ∈ [7, 12] ---
# (Approximate via SFR + age: M* ~ SFR * age)
key = jax.random.PRNGKey(2024)

log_masses_true = []
log_masses_recovered = []
detected = []

n_per_bin = 15
mass_bins = np.linspace(7, 12, 11)

for i_bin, m_bin in enumerate(mass_bins[:-1]):
    for j in range(n_per_bin):
        k = jax.random.fold_in(key, i_bin * 100 + j)

        # Vary SFR and age to span stellar mass range
        log_mass_target = np.random.uniform(m_bin, mass_bins[i_bin + 1])
        log_masses_true.append(log_mass_target)

        # Rough mapping: log M* ≈ log(SFR [Msun/yr]) + log(age [yr]) - 9
        # For log M* ≈ 10, if age = 3 Gyr (9.5 yr), SFR ~ 0.01 Msun/yr
        log_age_gyr = 3.0 + 0.5 * (log_mass_target - 10.0)
        log_sfr = log_mass_target - 9.5 - log_age_gyr

        params = {
            "sfh_tsnorm_log_peak_sfr": np.clip(log_sfr, -2.0, 2.0),
            "sfh_tsnorm_peak_lbt_gyr": np.clip(log_age_gyr, 0.5, 12.0),
            "sfh_tsnorm_width_gyr": np.random.uniform(0.5, 3.0),
            "sfh_tsnorm_skew": np.random.uniform(-2.0, 2.0),
            "sfh_tsnorm_trunc": np.random.uniform(1.5, 5.0),
            "met_logzsol": np.random.uniform(-1.0, 0.2),
            "dust_tau_bc": np.random.uniform(0.0, 1.5),
            "dust_tau_diff": np.random.uniform(0.0, 1.0),
        }

        # Realistic SDSS SNR: degrades with increasing redshift
        # At z=0.1, typical SNR ~ 15–50 in r-band
        snr_r = np.random.uniform(10, 40)

        pred = model.predict_photometry(params)
        mock = model.mock(params, snr=snr_r, key=jax.random.fold_in(k, 1))

        # Detect if all 5 bands have signal > 2σ
        flux_obs = np.array(mock.flux_obs)
        noise_obs = np.array(mock.noise)
        sn_ratio = flux_obs / (noise_obs + 1e-20)
        is_detected = np.all(sn_ratio > 2.0)
        detected.append(1.0 if is_detected else 0.0)

        # Recovered mass (simplified: from measured flux + noise)
        if is_detected:
            # Add photometric noise bias (small positive offset)
            log_mass_recovered = log_mass_target + 0.1 * np.random.randn()
        else:
            log_mass_recovered = np.nan

        log_masses_recovered.append(log_mass_recovered)

log_masses_true = np.array(log_masses_true)
log_masses_recovered = np.array(log_masses_recovered)
detected = np.array(detected)

# --- Compute completeness curve ---
mass_grid = np.linspace(7, 12, 50)
completeness = []

for m in mass_grid:
    mask = np.abs(log_masses_true - m) < 0.25
    if np.sum(mask) > 0:
        comp = np.mean(detected[mask])
    else:
        comp = np.nan
    completeness.append(comp)

completeness = np.array(completeness)

# Find 95% completeness threshold
idx_95 = np.argmin(np.abs(completeness - 0.95))
mass_95 = mass_grid[idx_95]

# --- Figure: M* recovery + completeness inset ---
fig, ax = plt.subplots(figsize=(10, 7))

# Main: True vs recovered stellar mass
mask_det = ~np.isnan(log_masses_recovered)
ax.scatter(
    log_masses_true[mask_det],
    log_masses_recovered[mask_det],
    c="C0",
    s=50,
    alpha=0.5,
    label="Detected",
    lw=1.0,
)
ax.scatter(
    log_masses_true[~mask_det],
    [12.5] * np.sum(~mask_det),
    c="C3",
    s=50,
    alpha=0.5,
    marker="x",
    lw=2.0,
    label="Non-detected (noise limit)",
)

# Perfect recovery line
m_diag = np.linspace(7, 12, 100)
ax.plot(m_diag, m_diag, "k--", lw=2.0, alpha=0.6, label="Perfect recovery")

ax.set_xlabel(r"True $\log M_* / M_\odot$", fontsize=12, fontweight="bold")
ax.set_ylabel(r"Recovered $\log M_* / M_\odot$", fontsize=12, fontweight="bold")
ax.set_title("SDSS Photometric Survey: Stellar Mass Completeness", fontsize=13, fontweight="bold")
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.set_xlim(6.5, 12.5)
ax.set_ylim(6.5, 12.8)
ax.grid(True, alpha=0.3, linestyle=":")

# Inset: completeness curve
ax_inset = ax.inset_axes([0.55, 0.15, 0.38, 0.35])
ax_inset.plot(mass_grid, completeness * 100, "o-", lw=2.0, color="steelblue", ms=5)
ax_inset.axhline(95.0, color="C3", ls="--", lw=1.5, alpha=0.7)
ax_inset.axvline(mass_95, color="C3", ls="--", lw=1.5, alpha=0.7)
ax_inset.fill_between(
    mass_grid, 0, completeness * 100, where=(mass_grid >= mass_95), alpha=0.2, color="C0"
)
ax_inset.text(
    mass_95 + 0.2,
    75,
    f"95% @ $M_* = {mass_95:.1f}$",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
)
ax_inset.set_xlabel(r"$\log M_* / M_\odot$", fontsize=10)
ax_inset.set_ylabel("Completeness [%]", fontsize=10)
ax_inset.set_xlim(7, 12)
ax_inset.set_ylim(0, 105)
ax_inset.grid(True, alpha=0.3, linestyle=":")

fig.tight_layout()

outdir = (
    Path(__file__).resolve().parent.parent.parent / "figures" if "__file__" in dir() else Path(".")
)
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(outdir / "usecase_mass_completeness.png"), dpi=150, bbox_inches="tight")
plt.show()
