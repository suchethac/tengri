"""
Photo-z degeneracy: chi² landscape over redshift and stellar mass
==================================================================

Galaxy photometry is degenerate in redshift and stellar mass — the same
galaxy can look identical at different redshifts if the mass is adjusted.

References: Bolzonella+2000; Brammer+2008.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*FutureWarning.*")

# %% Build the template model ONCE.
#
# Redshift and stellar mass are the two axes of the grid — so they are *free
# parameters*, not structural choices, and both vary inside a single model.
# There is no reason to rebuild anything as we walk the grid: a free parameter
# takes a different value on every evaluation, which is exactly what a grid is.
# The rest of the template (SFH shape, dust) is held fixed.

BANDS = [
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "vista_y",
    "vista_j",
    "vista_h",
    "vista_ks",
]

Z_MIN, Z_MAX = 0.5, 5.0
LOGM_MIN, LOGM_MAX = 9.0, 11.5

obs = tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))

model = tengri.SEDModel.build(
    tengri.load_ssp("fsps_prsc_miles_chabrier"),
    observation=obs,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "peak_lbt_gyr": 3.0,
        "width_gyr": 2.0,
        "log_total_mass": tengri.Uniform(LOGM_MIN, LOGM_MAX),  # free: the grid's mass axis
        "skew": 0.3,
        "trunc": 10.0,
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_diff": 0.3,
        "tau_bc": 0.1,
        "slope": -0.7,
    },
    neb={"type": "cue", "all_params": tengri.FIXED},
    redshift=tengri.Uniform(Z_MIN, Z_MAX),
)

# %% Mock a galaxy at z_true = 2.5 with known mass.

z_true = 2.5
log_mstar_true = 10.5  # ~3.2e10 Msun

key = jax.random.PRNGKey(42)
truth_params = dict(model.spec.sample(key))
truth_params.update(
    sfh_tsnorm_peak_lbt_gyr=3.0,
    sfh_tsnorm_width_gyr=2.0,
    sfh_tsnorm_log_total_mass=log_mstar_true,
    sfh_tsnorm_skew=0.3,
    sfh_tsnorm_trunc=10.0,
    dust_tau_diff=0.3,
    dust_tau_bc=0.1,
    dust_slope=-0.7,
    neb_logU=-3.0,
    neb_logZ_gas=0.0,
    redshift=z_true,
)

mock = model.mock(truth_params, snr=10.0, key=key)
flux_obs = np.asarray(mock.flux_obs)
noise_obs = np.asarray(mock.noise)

# %% Compute χ² on a 2D grid of (z, log_mstar).
#
# `predict_photometry` is the lean, vmap-safe surface, so the whole grid is one
# compiled call. We map over the mass axis and sequence the redshift axis with
# `lax.map`: a flat 65x65 vmap would materialize every SED at once (>10 GB),
# whereas one row at a time peaks near 3 GB.

z_grid = np.linspace(Z_MIN, Z_MAX, 65)
log_mstar_grid = np.linspace(LOGM_MIN, LOGM_MAX, 65)

fixed_params = {
    k: v for k, v in truth_params.items() if k not in ("redshift", "sfh_tsnorm_log_total_mass")
}


def _flux(z, log_mstar):
    """Predicted photometry at one (z, M*) point."""
    return model.predict_photometry(
        {**fixed_params, "redshift": z, "sfh_tsnorm_log_total_mass": log_mstar}
    )


_row = jax.vmap(_flux, in_axes=(None, 0))  # over mass, at fixed z
_grid = jax.jit(lambda zs, ms: jax.lax.map(lambda z: _row(z, ms), zs))

flux_pred = np.asarray(_grid(jnp.asarray(z_grid), jnp.asarray(log_mstar_grid)))

chi2_grid = np.sum(((flux_obs - flux_pred) / noise_obs) ** 2, axis=-1)

# %% Plot χ² surface with contours.

fig, ax = plt.subplots(figsize=(7.0, 5.5))

# Normalize χ² relative to minimum for better visibility.
chi2_min = np.min(chi2_grid)
chi2_norm = chi2_grid - chi2_min

# Heatmap: log scale to show structure across orders of magnitude.
im = ax.contourf(
    log_mstar_grid,
    z_grid,
    chi2_norm,
    levels=np.logspace(-1, 3, 30),
    norm=plt.matplotlib.colors.LogNorm(vmin=0.5, vmax=500),
    cmap="viridis",
)

# Contours at 1σ, 2σ, 3σ (Δχ² = 2.28, 6.18, 11.83 for 2 DoF).
sigma_levels = [2.28, 6.18, 11.83]
cs = ax.contour(
    log_mstar_grid,
    z_grid,
    chi2_norm,
    levels=sigma_levels,
    colors="white",
    linewidths=0.8,
    linestyles=["solid", "dashed", "dotted"],
)

# Manual contour labels (avoid matplotlib's automatic labels).
ax.clabel(cs, inline=True, fontsize=7, fmt=r"$%g\sigma$")

# Mark true (z, M*).
ax.plot(
    log_mstar_true,
    z_true,
    "r*",
    markersize=18,
    markeredgecolor="white",
    markeredgewidth=0.8,
    label=f"True: z={z_true}, M*=10$^{{{log_mstar_true:.1f}}}$ M$_\\odot$",
)

# Colorbar.
cbar = fig.colorbar(im, ax=ax, label=r"$\Delta\chi^2$ (relative to minimum)", pad=0.01)

ax.set_xlabel(r"$\log_{10}(M_* / M_\odot)$")
ax.set_ylabel(r"Redshift $z$")
ax.legend(loc="upper left", fontsize=8, frameon=False)
ax.set_xlim(log_mstar_grid.min(), log_mstar_grid.max())
ax.set_ylim(z_grid.min(), z_grid.max())

plt.savefig("plot_photoz_chi2_grid.png", dpi=150, bbox_inches="tight")
