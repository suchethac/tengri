"""
Nebular Emission Backends
==========================

Compare tengri's three nebular emission backends: BakedIn (SSP-embedded),
CloudyGrid (tabulated photoionization), and Cue (neural emulator).
Shows how each backend predicts emission lines in the optical window.
"""

import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri import Fixed, Model, ParamSpec, load_ssp_data

# --- Check for SSP data ---
SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if not os.path.exists(SSP_PATH):
    sys.exit(f"SSP data not found at {SSP_PATH}. Run from the project root.")

ssp_data = load_ssp_data(SSP_PATH)

# Shared galaxy parameters: young, star-forming, no dust
shared_params = dict(
    sfh_tsnorm_log_peak_sfr=Fixed(1.5),
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),
    sfh_tsnorm_width_gyr=Fixed(0.5),
    sfh_tsnorm_skew=Fixed(0.0),
    sfh_tsnorm_trunc=Fixed(5.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(0.0),
    dust_tau_diff=Fixed(0.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.0),
)

# --- Backend 1: BakedIn (nebular pre-computed in SSP) ---
spec_baked = ParamSpec(**shared_params)
model_baked = Model(spec_baked, ssp_data)
params_baked = {k: float(v.value) for k, v in shared_params.items()}
sed_baked = model_baked.predict_sed(params_baked)
wave = ssp_data.ssp_wave

# --- Backend 2: CloudyGrid (if available) ---
cloudy_path = "data/cloudy_grid.h5"
sed_cloudy = None
if os.path.exists(cloudy_path):
    from tengri.models.nebular import CloudyGridBackend

    spec_cloudy = ParamSpec(**shared_params, neb_logU=Fixed(-3.0), neb_logZ_gas=Fixed(-0.3))
    model_cloudy = Model(spec_cloudy, ssp_data, nebular_backend=CloudyGridBackend(cloudy_path))
    params_cloudy = {k: float(v.value) for k, v in shared_params.items()}
    params_cloudy.update({"neb_logU": -3.0, "neb_logZ_gas": -0.3})
    sed_cloudy = model_cloudy.predict_sed(params_cloudy)

# --- Plot: optical emission line region ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel 1: H-beta + [O III]
regions = [
    (axes[0], 4700, 5100, r"H$\beta$ + [O III]",
     {"H$\\beta$": 4861, "[O III]": 5007}),
    (axes[1], 6400, 6750, r"H$\\alpha$ Region",
     {"H$\\alpha$": 6563}),
]

for ax, wmin, wmax, title, lines in regions:
    mask = (wave > wmin) & (wave < wmax)
    ax.plot(np.array(wave[mask]), np.array(sed_baked[mask]),
            "k-", lw=1.2, label="BakedIn (default)")
    if sed_cloudy is not None:
        ax.plot(np.array(wave[mask]), np.array(sed_cloudy[mask]),
                "C1--", lw=1.2, label="CloudyGrid")
    for lbl, lam in lines.items():
        ax.axvline(lam, ls=":", color="C3", lw=0.7, alpha=0.6)
        ax.text(lam + 5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
                lbl, fontsize=8, color="C3", va="top")
    ax.set_xlabel(r"Rest Wavelength [$\AA$]")
    ax.set_ylabel(r"$L_\nu$ [arbitrary]")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)

fig.suptitle("Nebular Emission: Backend Comparison", fontsize=12, y=1.02)
fig.tight_layout()
plt.savefig("nebular_backends.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Summary ---
print("Backend comparison:")
print("  BakedIn  : 0 extra params, fastest, fixed logU and Z_gas")
print("  CloudyGrid: 2 extra params (logU, Z_gas), tabulated CLOUDY grids")
print("  Cue      : 12 extra params (abundances), neural net emulator")
if sed_cloudy is None:
    print(f"\n  Note: CloudyGrid not shown (grid file not found at {cloudy_path})")
