"""
JWST NIRSpec PRISM vs G395M grating: Hα + [NII] resolution comparison
======================================================================

A z=5 JADES-like star-forming galaxy observed with JWST NIRSpec in two modes:
PRISM (R~100, low-resolution) and G395M grating (R~1000, medium-resolution).
The Hα line at rest 6564.61 Å appears as a single blob in PRISM but resolves
into three peaks in the grating: Hα + [NII] λλ6549,6585 Å doublet.

Demonstrates instrumental resolution effects on emission-line diagnostics
and spectral feature recovery. Based on JADES survey (Eisenstein+2023).

References
----------
.. [1] Jakobsen et al. (2022). JWST/NIRSpec in the Infrared. Space Telescope
       Science Institute Technical Report.
.. [2] Eisenstein et al. (2023). The JADES Survey: First Spectroscopic
       Redshifts. arXiv:2306.02465.
.. [3] Cameron et al. (2023). The assembly of metals in galaxies at z~3–5.
       MNRAS (submitted).

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# ============================================================================
# Model Setup: z=5 JADES-like star-forming galaxy
# ============================================================================
# Bare-stellar SSP with Cue nebular backend (EoR-standard).
# Parameters tuned to JADES median redshift (z~5, Eisenstein+2023) and
# typical metallicity for early-universe star-forming galaxies (log Z_gas ~ -0.5,
# Cameron+2023).

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# z=5 star-forming model: moderate ongoing star formation, modest dust.
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "tau_gyr": 0.3,
        "log_total_mass": 10.0,
        "alpha": 2.5,
        "beta": 1.8,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.06, "tau_bc": 0.10},
    neb={"type": "cue", "all_params": tengri.FIXED, "logZ_gas": -0.5, "logU": -1.5},
    redshift=tengri.Fixed(5.0),
)

p = dict(model.spec.sample(jax.random.PRNGKey(42)))

# ============================================================================
# Rest-frame SED from the model
# ============================================================================
z = 5.0
rest_to_obs = 1.0 + z

out = model.predict(p)
wave_rest = np.asarray(model.wavelengths)  # Rest-frame Å
sed_rest = np.asarray(out.rest_sed())  # erg/s/Hz

# Hα region in rest frame: 6400–6700 Å
mask_rest = (wave_rest >= 6400.0) & (wave_rest <= 6700.0)
wave_rest_ha = wave_rest[mask_rest]
sed_rest_ha = sed_rest[mask_rest]

# Observed-frame Hα complex: 6400–6700 Å rest → 38400–40200 Å observed
wave_obs_ha = wave_rest_ha * rest_to_obs

# Normalize to continuum for display
sed_rest_ha_norm = sed_rest_ha / np.median(sed_rest_ha)

# ============================================================================
# Resolution convolution: PRISM (R~100) and grating (R~1000)
# ============================================================================


def convolve_with_resolution(wave, sed, R):
    """Convolve SED with Gaussian LSF for constant resolution R.

    Parameters
    ----------
    wave : ndarray
        Wavelength grid [Angstrom].
    sed : ndarray
        SED [arbitrary units, normalized].
    R : float
        Spectral resolution (lambda / delta_lambda).

    Returns
    -------
    ndarray
        Convolved SED.

    """
    dlam_pix = np.mean(np.diff(wave))
    dlam = np.mean(wave) / R
    sigma_pix = dlam / (2.355 * dlam_pix)
    return gaussian_filter1d(sed, sigma=sigma_pix)


# PRISM: R~100 (low-resolution, blends lines)
sed_prism = convolve_with_resolution(wave_obs_ha, sed_rest_ha_norm, R=100.0)

# G395M grating: R~1000 (medium-resolution, resolves triplet)
sed_grating = convolve_with_resolution(wave_obs_ha, sed_rest_ha_norm, R=1000.0)

# ============================================================================
# Plot: Two-panel comparison
# ============================================================================

fig, (ax_prism, ax_grating) = plt.subplots(
    2,
    1,
    figsize=(10, 7),
    sharex=True,
)

# --- Panel 1: PRISM (low-resolution) ---
ax_prism.plot(
    wave_obs_ha,
    sed_prism,
    lw=2.0,
    color="C0",
    label=r"$R \approx 100$ (PRISM)",
)
ax_prism.fill_between(wave_obs_ha, sed_prism, alpha=0.3, color="C0")

# Emission line markers (vacuum wavelengths in rest frame)
ha_rest = 6562.80
nii_6549_rest = 6548.05
nii_6585_rest = 6583.46

ax_prism.axvline(ha_rest * rest_to_obs, ls=":", lw=1.5, color="gray", alpha=0.6)
ax_prism.axvline(nii_6549_rest * rest_to_obs, ls=":", lw=1.0, color="gray", alpha=0.4)
ax_prism.axvline(nii_6585_rest * rest_to_obs, ls=":", lw=1.0, color="gray", alpha=0.4)

ax_prism.text(
    ha_rest * rest_to_obs,
    0.90,
    r"H$\alpha$",
    fontsize=9,
    ha="center",
    color="gray",
    transform=ax_prism.get_xaxis_transform(),
)

ax_prism.set_ylabel("Normalized Flux", fontsize=11)
ax_prism.text(
    0.05,
    0.95,
    r"PRISM ($R \approx 100$): blended",
    transform=ax_prism.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
ax_prism.legend(frameon=False, loc="upper right", fontsize=10)
ax_prism.set_ylim(0.7, 1.35)

ax_grating.plot(
    wave_obs_ha,
    sed_grating,
    lw=2.0,
    color="C1",
    label=r"$R \approx 1000$ (G395M)",
)
ax_grating.fill_between(wave_obs_ha, sed_grating, alpha=0.3, color="C1")

colors_lines = ["C2", "gray", "C2"]
line_rest = [nii_6549_rest, ha_rest, nii_6585_rest]
line_names = [r"[N\,II]", r"H$\alpha$", r"[N\,II]"]

for lam_rest, name, color in zip(line_rest, line_names, colors_lines):
    lam_obs = lam_rest * rest_to_obs
    ax_grating.axvline(lam_obs, ls=":", lw=1.5, color=color, alpha=0.6)
    ax_grating.text(
        lam_obs,
        0.90,
        name,
        fontsize=9,
        ha="center",
        color=color,
        transform=ax_grating.get_xaxis_transform(),
    )

ax_grating.set_xlabel(r"Observed wavelength [$\AA$]", fontsize=11)
ax_grating.set_ylabel("Normalized Flux", fontsize=11)
ax_grating.text(
    0.05,
    0.95,
    r"G395M ($R \approx 1000$): resolved",
    transform=ax_grating.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
ax_grating.legend(frameon=False, loc="upper right", fontsize=10)
ax_grating.set_ylim(0.7, 1.35)

# Common x-axis limits: rest 6500–6700 Å → observed 39000–40200 Å
ax_grating.set_xlim(wave_obs_ha.min(), wave_obs_ha.max())

fig.tight_layout()
plt.savefig("plot_nirspec_prism_vs_grating.png", dpi=150, bbox_inches="tight")
