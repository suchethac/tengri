"""
Fisher Information Ellipses from the Hessian
=============================================

The Fisher Information Matrix quantifies which linear combinations of
parameters are constrained by data — and which are degenerate. Tengri's
fully differentiable forward model makes it trivial to compute the Fisher
matrix at any point in parameter space.

This plot demonstrates the classic age-dust degeneracy in galaxy SED
fitting: at fixed stellar mass, older stars + more dust produce the
same multiwavelength SED as younger stars + less dust. The principal axes
of the Fisher ellipse (eigenvectors of the Hessian of the likelihood) reveal
the most-constrained and least-constrained linear combinations.

We compute the Fisher matrix via finite-difference approximation to the
Hessian: F_ij ≈ (∂²χ² / ∂θ_i ∂θ_j) evaluated at the fiducial point. The
resulting ellipse's eccentricity and orientation tell us exactly which
parameter combinations observations can break.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*wNE.*")

# Load a star-forming galaxy SED library
BARE = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Common fiducial parameters for fair comparison across filter sets
FIDUCIAL_PARAMS = {
    "sfh_dpl_alpha": 1.5,
    "sfh_dpl_beta": 2.0,
    "sfh_dpl_tau_gyr": 1.0,
    "sfh_dpl_age_gyr": 3.0,
    "sfh_dpl_log_total_mass": 0.5,
    "dust_tau_bc": 0.3,
    "dust_tau_diff": 0.1,
    "redshift": 0.1,
    "met_logzsol": 0.0,
}

# Single observation config to demonstrate the concept
FILTERS = ["sdss_u", "sdss_r", "2mass_j"]

# Build observation
observation = tengri.Observation(photometry=tengri.Photometry.from_names(FILTERS))

# Create a minimal model: DPL SFH + Calzetti dust + Cue nebular (fixed)
model = tengri.SEDModel.build(
    ssp_data=BARE,
    observation=observation,
    sfh={"type": "dpl", "all_params": tengri.FREE},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "all_params": tengri.FREE,
        "emission": {"type": "dale2014", "all_params": tengri.FIXED},
    },
    neb={"type": "cue", "all_params": tengri.FIXED},
    redshift=tengri.Fixed(0.1),
)

# Generate synthetic observations at the fiducial point
out = model.predict_photometry(FIDUCIAL_PARAMS)
flux_fiducial = np.asarray(out)
snr = 50.0  # S/N per filter
noise = flux_fiducial / snr
noise_inv = 1.0 / noise**2

# Extract the two degenerate parameters
free_names = ["sfh_dpl_log_total_mass", "dust_tau_bc"]
flat_fiducial = np.array([FIDUCIAL_PARAMS[free_names[0]], FIDUCIAL_PARAMS[free_names[1]]])

# Compute Fisher matrix via finite differences
delta = 1e-5
fisher = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        corners = []
        for di in [0, 1]:
            for dj in [0, 1]:
                flat_params = flat_fiducial.copy()
                if di == 1:
                    flat_params[i] += delta
                if dj == 1:
                    flat_params[j] += delta

                # Reconstruct full parameter dict
                p = dict(FIDUCIAL_PARAMS)
                p[free_names[0]] = float(flat_params[0])
                p[free_names[1]] = float(flat_params[1])

                # Compute chi²
                out = model.predict_photometry(p)
                flux_model = np.asarray(out)
                chi2 = np.sum((flux_model - flux_fiducial) ** 2 * noise_inv)
                corners.append(0.5 * chi2)

        # Second mixed partial: (L_++ - L_+0 - L_0+ + L_00) / (δ²)
        l_pp, l_p0, l_0p, l_00 = corners[3], corners[1], corners[2], corners[0]
        fisher[i, j] = (l_pp - l_p0 - l_0p + l_00) / (delta**2)

# Eigendecomposition
eigenvals, eigenvecs = np.linalg.eigh(fisher)

# Covariance matrix = inverse of Fisher
cov = np.linalg.inv(fisher)

# Plot the ellipse: 2-sigma confidence (95%)
angle_rad = np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])
angle_deg = float(np.degrees(angle_rad))

scale = 2.0  # 2-sigma
width = 2.0 * scale * np.sqrt(1.0 / eigenvals[0])
height = 2.0 * scale * np.sqrt(1.0 / eigenvals[1])

from matplotlib.patches import Ellipse

fig, ax = plt.subplots(figsize=(8.0, 6.0))

ellipse = Ellipse(
    xy=(float(flat_fiducial[0]), float(flat_fiducial[1])),
    width=float(width),
    height=float(height),
    angle=angle_deg,
    facecolor="#1f77b4",
    alpha=0.15,
    edgecolor="#1f77b4",
    linewidth=2.5,
)
ax.add_patch(ellipse)

# Overlay the principal axes for visualization
a_major = np.sqrt(1.0 / eigenvals[0])
a_minor = np.sqrt(1.0 / eigenvals[1])
cos_a = np.cos(angle_rad)
sin_a = np.sin(angle_rad)

ax.arrow(
    flat_fiducial[0],
    flat_fiducial[1],
    2.0 * a_major * cos_a,
    2.0 * a_major * sin_a,
    head_width=0.05,
    head_length=0.08,
    fc="#d62728",
    ec="#d62728",
    lw=1.5,
    label="Major axis (least constrained)",
)
ax.arrow(
    flat_fiducial[0],
    flat_fiducial[1],
    -2.0 * a_minor * sin_a,
    2.0 * a_minor * cos_a,
    head_width=0.05,
    head_length=0.08,
    fc="#2ca02c",
    ec="#2ca02c",
    lw=1.5,
    label="Minor axis (most constrained)",
)

# Fiducial point
ax.plot(flat_fiducial[0], flat_fiducial[1], "ko", markersize=6, label="Fiducial")

ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-0.15, 0.65)
ax.set_xlabel(r"$\log_{10}(\dot{M}_\star / M_\odot\,\mathrm{yr}^{-1})$ (SFR)", fontsize=11)
ax.set_ylabel(r"Dust optical depth $\tau_\mathrm{BC}$ (Calzetti)", fontsize=11)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

ax.text(
    0.05,
    0.95,
    f"Fisher Information Ellipse (95% / 2σ)\nz=0.1, {len(FILTERS)} filters",
    transform=ax.transAxes,
    fontsize=11,
    weight="bold",
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
)

fig.tight_layout()
plt.savefig("plot_gradient_degeneracy_direction.png", dpi=150, bbox_inches="tight")
