"""
Lyman-alpha equivalent width peaks at young ages, varies with gas metallicity
==============================================================================

Lyα rest-frame wavelength is 1216 Å (vacuum). EW peaks at 3–5 Myr when O-type
stars dominate ionization, then decays past 10 Myr. Higher metallicity
suppresses ionizing photon production, reducing peak EW.

References
----------

Schaerer 2003, A&A, 397, 527 (ionizing photon production in massive starburst populations).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

wave_lya = 1216.0
C_AA_PER_S = 2.998e18
ages_myr = np.logspace(0.0, np.log10(30), 18)
met_logzsol = np.array([-0.3, 0.0, 0.3])

fig, ax = plt.subplots(figsize=(7, 5))
cmap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(vmin=met_logzsol.min(), vmax=met_logzsol.max())

for met in met_logzsol:
    ew_lya = []

    for age_myr in ages_myr:
        log_total_mass_age = np.log10(age_myr * 1e6)
        sfh_config_age = {
            "type": "const",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "log_total_mass": log_total_mass_age,
            "start_gyr": age_myr / 1e3,
            "end_gyr": 0.0,
        }

        neb_config = {
            "type": "cue",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "logZ_gas": met,
            "logU": -2.0,
            "fesc": 0.0,
            "fesc_lya": 0.0,
        }

        dust_config = {
            "law": "power_law",
            "type": "two_component",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "tau_diff": 0.0,
            "tau_bc": 0.0,
        }

        model = tengri.SEDModel.build(
            ssp,
            sfh=sfh_config_age,
            dust_attenuation=dust_config,
            neb=neb_config,
            redshift=tengri.Fixed(0.0),
        )

        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        lines = model.predict(params).lines
        sed_result = model.predict(params)

        lya_lum = float(lines.lya)
        wave = np.asarray(model.wavelengths)
        sed = np.asarray(sed_result.rest_sed())

        idx_lya = np.argmin(np.abs(wave - wave_lya))
        continu_at_lya = sed[idx_lya]

        if continu_at_lya > 0:
            continu_lambda = continu_at_lya * C_AA_PER_S / (wave_lya**2)
            ew = lya_lum / continu_lambda
        else:
            ew = np.nan

        ew_lya.append(ew)

    ew_lya = np.asarray(ew_lya)
    color = cmap(norm(met))
    ax.plot(ages_myr, ew_lya, "o-", color=color, markersize=5, lw=1.4)

ax.set_xlabel(r"Population age [Myr]")
ax.set_ylabel(r"Lyα equivalent width [Å]")
ax.set_xscale("log")
ax.set_xlim(0.7, 35)
ax.grid(True, alpha=0.2, which="both")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\log Z_{\rm gas} / Z_{\odot}$")

fig.tight_layout()
plt.savefig("plot_lyalpha_ew_vs_age.png", dpi=150, bbox_inches="tight")
