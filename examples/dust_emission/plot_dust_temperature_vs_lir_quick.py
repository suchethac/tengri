r"""Quick test of the L_IR–T_dust example"""

import os
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, load_ssp_data
from tengri.analysis.plotting import setup_style

setup_style()

_ssp_name = "fsps_prsc_miles_chabrier.h5"
_repo_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / _ssp_name).exists())
ssp = load_ssp_data(str(_repo_root / "data" / _ssp_name))

photometry = Photometry.from_names(["galex_nuv", "sdss_g", "sdss_i", "wise_w3"])
observation = Observation(photometry=photometry)

z_gal = 0.05
sfr_grid = np.array([1.0, 10.0, 100.0])  # Just 3 points for quick test

lir_values = []
tdust_values = []
sfr_values = []
c_aa_per_s = 2.99792458e18

for idx, sfr in enumerate(sfr_grid):
    print(f"SFR {idx+1}/3: {sfr} Msun/yr")

    groups = {
        "sfh": {
            "type": "dpl",
            "log_peak_sfr": Fixed(np.log10(sfr)),
            "alpha": Fixed(1.0),
            "beta": Fixed(1.0),
            "tau_gyr": Fixed(1.0),
        },
        "dust": {
            "type": "two_component",
            "law_bc": "calzetti",
            "tau_bc": Fixed(0.3),
            "tau_diff": Fixed(0.1),
            "slope": Fixed(-0.7),
            "emission": {
                "type": "modified_blackbody",
                "T": Fixed(35.0),
                "beta_ir": Fixed(1.8),
            },
        },
        "neb": {"type": "cue"},
        "redshift": Fixed(z_gal),
        "apply_igm": False,
    }

    spec = Parameters.from_groups(**groups)
    model = SEDModel(spec, ssp, observation=observation)

    truth = {
        "sfh_dpl_log_peak_sfr": np.log10(sfr),
        "sfh_dpl_alpha": 1.0,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 1.0,
        "met_logzsol": -0.1,
        "dust_tau_bc": 0.3,
        "dust_tau_diff": 0.1,
        "dust_slope": -0.7,
        "dust_T": 35.0,
        "dust_beta_ir": 1.8,
        "redshift": z_gal,
    }

    sed = model.predict_rest_sed(truth)
    wave_rest_aa = np.asarray(sed.wavelength)
    lnu_rest = np.asarray(sed.sed)

    mask_ir = (wave_rest_aa >= 8.0e4) & (wave_rest_aa <= 1.0e7)

    if np.sum(mask_ir) > 1:
        nu = c_aa_per_s / wave_rest_aa
        wave_ir = wave_rest_aa[mask_ir]
        nu_ir = c_aa_per_s / wave_ir
        lnu_ir = lnu_rest[mask_ir]

        sort_idx = np.argsort(nu_ir)
        nu_ir_sorted = nu_ir[sort_idx]
        lnu_ir_sorted = lnu_ir[sort_idx]

        lir = np.trapz(lnu_ir_sorted, nu_ir_sorted)
    else:
        lir = np.trapz(lnu_rest[mask_ir], wave_rest_aa[mask_ir])

    if np.sum(mask_ir) > 1:
        wave_ir = wave_rest_aa[mask_ir]
        lnu_ir = lnu_rest[mask_ir]
        peak_idx = np.argmax(lnu_ir)
        wave_peak_aa = wave_ir[peak_idx]
        wave_peak_um = wave_peak_aa / 1e4
        tdust = 2898.0 / wave_peak_um
    else:
        tdust = 35.0

    lir_values.append(lir)
    tdust_values.append(tdust)
    sfr_values.append(sfr)

lir_values = np.array(lir_values)
tdust_values = np.array(tdust_values)
sfr_values = np.array(sfr_values)

lsun = 3.839e33
lir_lsun = lir_values / lsun

print(f"\nResults:")
print(f"  L_IR [Lsun]: {lir_lsun}")
print(f"  T_dust [K]: {tdust_values}")
print(f"  SFR [Msun/yr]: {sfr_values}")

fig, ax = plt.subplots(figsize=(9, 6))

scatter = ax.scatter(
    np.log10(lir_lsun),
    tdust_values,
    c=np.log10(sfr_values),
    s=200,
    alpha=0.6,
    cmap="viridis",
    edgecolors="black",
    linewidth=1.0,
)

lir_range = np.logspace(9, 11, 100)
t_sym = 27.0 + 6.5 * (np.log10(lir_range) - 11.5)
ax.plot(np.log10(lir_range), t_sym, "r--", linewidth=2.0, label="Symeonidis+2013 fit", alpha=0.8)

ax.set_xlabel(r"$\log_{10}(L_{\rm IR} / L_\odot)$", fontsize=11)
ax.set_ylabel(r"$T_{\rm dust}$ [K]", fontsize=11)
ax.set_xlim(8, 11)
ax.set_ylim(25, 40)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", frameon=False, fontsize=10)

cbar = plt.colorbar(scatter, ax=ax, label=r"$\log_{10}(\mathrm{SFR} / M_\odot \, \mathrm{yr}^{-1})$")

fig.tight_layout()
plt.savefig("plot_dust_temperature_vs_lir_quick.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: plot_dust_temperature_vs_lir_quick.png")
