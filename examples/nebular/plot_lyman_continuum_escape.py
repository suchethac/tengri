"""
Lyman-continuum escape fraction reshapes the SED around the 912 A edge
=======================================================================

We zoom on the Lyman-continuum region (rest 800-1300 A) and sweep
the escape fraction f_esc to show how the 912 A discontinuity
deepens as more ionising photons leave the ISM unabsorbed. Companion
to ``plot_fesc_sweep.py``, which projects the same physics into
optical line-ratio diagnostics.

References
----------
.. [1] Li et al. 2024, "Cue: A fast neural network emulator for
    nebular emission lines", ApJ, 969, 28
.. [2] Steidel et al. 2018, "The Low-z Lyman Continuum Survey",
    ApJ, 869, 123
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Load bare-stellar SSP (required by Cue)
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Build model for a young, star-forming galaxy
# Double power-law SFH tuned to emphasize young stellar population
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 1.0,
        "beta": 2.5,
        "tau_gyr": 0.3,
        "log_peak_sfr": 1.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED, "neb_fesc": tengri.Uniform(0.0, 1.0), "neb_logU": -2.5, "neb_logZ_gas": -0.5},
    redshift=tengri.Fixed(0.05),
)

# Sample baseline parameters
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Escape fraction grid: 0, 0.1, 0.3, 0.5, 0.7
fesc_values = np.array([0.0, 0.1, 0.3, 0.5, 0.7])
norm = mpl.colors.Normalize(vmin=fesc_values.min(), vmax=fesc_values.max())
cmap = plt.get_cmap("Purples")

fig, ax = plt.subplots(figsize=(6.5, 4.2))

sed_at_fesc = {}
for fesc in fesc_values:
    params = {**baseline, "neb_fesc": jnp.float64(fesc)}
    out = model.predict_obs_sed(params)
    wave_obs = np.asarray(out.wavelength)
    sed = np.asarray(out.sed)

    # Store for verification
    sed_at_fesc[fesc] = (wave_obs, sed)

    # Compute nu * L_nu for SED plotting
    nu = 2.998e18 / wave_obs  # frequency in Hz
    nu_l_nu = nu * sed

    ax.loglog(wave_obs, nu_l_nu, color=cmap(norm(fesc)), lw=1.4, label=f"$f_{{\\mathrm{{esc}}}} = {fesc:.1f}$")

# Set limits to focus on Lyman continuum and edge region (obs 840–1350 Å, rest-frame equivalent at z=0.05)
ax.set_xlim(840, 1350)
ax.set_ylim(1e39, 1e45)
ax.set_xlabel(r"Observed-frame wavelength $\lambda_{\mathrm{obs}}$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

# Add vertical line at Lyman edge observed-frame
lyman_edge_obs = 912 * 1.05  # Rest 912 A at z=0.05
ax.axvline(lyman_edge_obs, color="gray", linestyle="--", alpha=0.5, lw=1.0)
ax.text(lyman_edge_obs, 2e39, "Lyman edge\n(obs ~958 Å)", ha="center", fontsize=9, color="gray")

# Add colorbar instead of legend
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$f_{\mathrm{esc}}$", fontsize=11)

fig.tight_layout()
plt.savefig("plot_lyman_continuum_escape.png", dpi=150, bbox_inches="tight")
print("Saved plot_lyman_continuum_escape.png")

# Verification: Print flux ratios above/below Lyman edge (observed-frame)
lyman_edge_obs = 912 * 1.05  # Rest 912 A at z=0.05
for fesc in fesc_values:
    wave_obs, sed = sed_at_fesc[fesc]
    above_edge = sed[wave_obs > lyman_edge_obs]
    below_edge = sed[wave_obs < lyman_edge_obs]
    if len(above_edge) > 0 and len(below_edge) > 0:
        ratio = np.mean(below_edge) / np.mean(above_edge)
        print(f"f_esc={fesc:.1f}: LyC/non-LyC flux ratio = {ratio:.2f}")
        print(f"  Mean LyC sed: {np.mean(below_edge):.3e}, Mean non-LyC sed: {np.mean(above_edge):.3e}")
