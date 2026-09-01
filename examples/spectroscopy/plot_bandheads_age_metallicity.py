"""
Stellar absorption bandheads: age and metallicity anti-correlation
===================================================================

Stellar absorption features, especially the Mg b and Fe5270 line strengths,
encode both age and metallicity in a classical anti-correlation pattern:
at fixed metallicity, both features strengthen with age (population becomes
older, cooler); at fixed age, increasing metallicity also strengthens the
features (enhanced α-element abundances + stronger metal absorption).

This two-panel comparison shows:

- **Left panel**: Mg b and Fe5270 equivalent widths (pseudo-EW in Ångströms)
  versus single-burst age at solar metallicity (Z/Z☉ = 0.0 dex).

- **Right panel**: same age range but across metallicities
  (Z/Z☉ = -1.0, -0.5, 0.0, +0.3 dex), illustrating the strength of the
  age–metallicity degeneracy.

Both indices are computed from high-resolution spectra (R=2000) in the
5050–5350 Å rest-frame optical window using Trager+1998 and Worthey+1994

line definitions. Reproduces the cluster age-dating diagnostics of
Trager+2000 [1]_.

References
----------
.. [1] Trager et al. 2000, "Stellar Population Models from 0.3 to 10 Microns
   Based on the Revised Padova Tracks," ApJ 535, 775.
   https://doi.org/10.1086/308859

.. [2] Worthey et al. 1994, "Old Stellar Populations. I. Empirical Models
   and Population Diagnostics," ApJS 94, 687.
   https://doi.org/10.1086/192220
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import Fixed, Observation, Spectroscopy
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

C_AA_PER_S = 2.998e18


def mgb_pseudo_ew(wave: np.ndarray, f_lambda: np.ndarray) -> float:
    r"""
    Mg b equivalent width using Trager+1998 line/continuum windows.

    The Mg b feature is measured as the equivalent width in the 5100–5300 Å
    region. Line core: 5150–5200 Å. Continuum anchors: 5100–5150 Å (blue),
    5200–5300 Å (red).

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Rest-frame wavelengths [Å].
    f_lambda : ndarray, shape (n_wave,)
        Specific flux density [erg/s/cm^3/Å], converted from L_nu.

    Returns
    -------
    ew : float
        Pseudo-equivalent width [Å] (positive for absorption).
    """
    line = (wave >= 5150.0) & (wave <= 5200.0)
    blue = (wave >= 5100.0) & (wave <= 5150.0)
    red = (wave >= 5200.0) & (wave <= 5300.0)

    cont_blue = np.mean(f_lambda[blue]) if blue.sum() > 0 else 1.0
    cont_red = np.mean(f_lambda[red]) if red.sum() > 0 else 1.0

    # Linear interpolation of continuum across line window
    lam_blue = 0.5 * (5100.0 + 5150.0)
    lam_red = 0.5 * (5200.0 + 5300.0)
    slope = (cont_red - cont_blue) / (lam_red - lam_blue)
    cont = cont_blue + slope * (wave[line] - lam_blue)

    delta = wave[line][1] - wave[line][0] if line.sum() > 1 else 1.0
    return float(np.sum(np.maximum(1.0 - f_lambda[line] / np.maximum(cont, 1e-30), 0.0)) * delta)


def fe5270_pseudo_ew(wave: np.ndarray, f_lambda: np.ndarray) -> float:
    r"""
    Fe5270 equivalent width using Worthey+1994 window.

    The Fe5270 feature is measured in the 5235–5285 Å region (narrow core).
    Continuum anchors: 5245–5275 Å (approximately symmetric).

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Rest-frame wavelengths [Å].
    f_lambda : ndarray, shape (n_wave,)
        Specific flux density [erg/s/cm^3/Å], converted from L_nu.

    Returns
    -------
    ew : float
        Pseudo-equivalent width [Å] (positive for absorption).
    """
    line = (wave >= 5235.0) & (wave <= 5285.0)
    cont = (wave >= 5245.0) & (wave <= 5275.0)

    cont_level = np.mean(f_lambda[cont]) if cont.sum() > 0 else 1.0
    delta = wave[line][1] - wave[line][0] if line.sum() > 1 else 1.0
    fe_ew = 1.0 - f_lambda[line] / np.maximum(cont_level, 1e-30)
    return float(np.sum(np.maximum(fe_ew, 0.0)) * delta)


# ============================================================================
# Left panel: age vs indices at fixed solar metallicity
# ============================================================================

ssp = tengri.load_ssp()
REDSHIFT = 0.05

# Wavelength range for bandhead measurements (observer frame)
wave_obs_min = 5050.0 * (1.0 + REDSHIFT)
wave_obs_max = 5350.0 * (1.0 + REDSHIFT)
wave_obs = np.linspace(wave_obs_min, wave_obs_max, 600)

# Create observation object with spectroscopy
obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))

# Age grid for left panel
ages_gyr = np.geomspace(1.0, 13.0, 18)
mgb_at_age = np.empty_like(ages_gyr)
fe5270_at_age = np.empty_like(ages_gyr)

# Build model at fixed parameters, then vary age
model_solar = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "peak_lbt_gyr": tengri.Uniform(0.03, 13.0),
        "width_gyr": 0.05,
        "log_total_mass": 10.0,
        "skew": 0.0,
        "trunc": 13.0,
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "tau_diff": 0.0,
        "tau_bc": 0.0,
    },
    redshift=Fixed(REDSHIFT),
    observation=obs,
)
baseline_solar = dict(model_solar.spec.sample(jax.random.PRNGKey(0)))

# Predict spectra across ages
for i, age in enumerate(ages_gyr):
    params = {**baseline_solar, "sfh_tsnorm_peak_lbt_gyr": jnp.float64(age)}

    pred = model_solar.predict(params)
    wave_rest = np.asarray(model_solar.wavelengths)
    l_nu = np.asarray(pred.rest_sed())
    f_lambda = l_nu * C_AA_PER_S / (wave_rest**2)

    mgb_at_age[i] = mgb_pseudo_ew(wave_rest, f_lambda)
    fe5270_at_age[i] = fe5270_pseudo_ew(wave_rest, f_lambda)

# ============================================================================
# Right panel: metallicity grid at multiple ages
# ============================================================================

ages_multimet = np.array([1.0, 2.0, 5.0, 8.0, 11.0])
met_dex = np.array([-1.0, -0.5, 0.0, 0.3])

mgb_grid = np.empty((len(ages_multimet), len(met_dex)))
fe5270_grid = np.empty((len(ages_multimet), len(met_dex)))

for j, age in enumerate(ages_multimet):
    for k, met in enumerate(met_dex):
        # Build model with fixed age, variable metallicity
        # The nested-dict builder supports met_logzsol as a direct parameter
        model_met = tengri.SEDModel.build(
            ssp,
            sfh={
                "type": "tsnorm",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "peak_lbt_gyr": age,
                "width_gyr": 0.05,
                "log_total_mass": 10.0,
                "skew": 0.0,
                "trunc": 13.0,
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_diff": 0.0,
                "tau_bc": 0.0,
            },
            redshift=Fixed(REDSHIFT),
            observation=obs,
        )
        baseline_met = dict(model_met.spec.sample(jax.random.PRNGKey(0)))

        # Set metallicity
        params_met = {**baseline_met, "met_logzsol": jnp.float64(met)}

        pred = model_met.predict(params_met)
        wave_rest = np.asarray(model_met.wavelengths)
        l_nu = np.asarray(pred.rest_sed())
        f_lambda = l_nu * C_AA_PER_S / (wave_rest**2)

        mgb_grid[j, k] = mgb_pseudo_ew(wave_rest, f_lambda)
        fe5270_grid[j, k] = fe5270_pseudo_ew(wave_rest, f_lambda)

# ============================================================================
# Plot
# ============================================================================

fig, (ax_age, ax_met) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"wspace": 0.30})

# Left: age at fixed metallicity
ax_age.plot(ages_gyr, mgb_at_age, "o-", color="C2", lw=1.8, markersize=5, label=r"Mg $b$")
ax_age.plot(ages_gyr, fe5270_at_age, "s-", color="C1", lw=1.8, markersize=5, label=r"Fe5270")
ax_age.set_xscale("log")
ax_age.set_xlabel(r"Single-burst age  [Gyr]")
ax_age.set_ylabel(r"Pseudo-equivalent width  [$\mathrm{\AA}$]")
ax_age.text(
    0.05,
    0.95,
    r"$Z/Z_\odot = 0.0$",
    transform=ax_age.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
ax_age.legend(loc="lower right", fontsize=10)
ax_age.grid(True, alpha=0.3, linestyle="--")

# Right: metallicity at multiple fixed ages
cmap = plt.get_cmap("coolwarm")
norm = plt.Normalize(vmin=ages_multimet.min(), vmax=ages_multimet.max())

for j, age in enumerate(ages_multimet):
    color = cmap(norm(age))
    ax_met.plot(
        met_dex,
        mgb_grid[j, :],
        "o-",
        color=color,
        lw=1.6,
        markersize=5,
        label=f"{age:.1f} Gyr",
    )

ax_met.set_xlabel(r"Metallicity  [$\log_{10}(Z/Z_\odot)$]")
ax_met.set_ylabel(r"Mg $b$ pseudo-EW  [$\mathrm{\AA}$]")
ax_met.text(
    0.05,
    0.95,
    "Mg $b$ across metallicity",
    transform=ax_met.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)
ax_met.legend(loc="lower right", fontsize=9, title="Age", title_fontsize=9)
ax_met.grid(True, alpha=0.3, linestyle="--")

plt.savefig("plot_bandheads_age_metallicity.png", dpi=150, bbox_inches="tight")
