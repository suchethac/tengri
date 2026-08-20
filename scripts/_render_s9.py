"""Standalone §9 figure renderer — pulls the §9 cell from the
reproduction notebook so we can re-render after audit fixes without
running the whole 1200-line notebook.
"""

import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from reproduction.cigale._drivers import cigale_driver as C, units as U
from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

SSP_PATH = Path("reproduction/cigale/_drivers/data/bc03_from_cigale.h5")
ssp = load_ssp_data(str(SSP_PATH))

_LOG10_ZSUN = -1.8477
MET_LOGZSOL_FIDUCIAL = float(np.log10(0.02) - _LOG10_ZSUN)
# Updated to match CIGALE's actual modified_starburst behavior
# (single Calzetti screen at E_BV_stars = 0.44 × 0.3 = 0.132 across all ages)
_E_BV_STARS = 0.44 * 0.3
TAU_DIFF_FIDUCIAL = 4.05 * _E_BV_STARS / 1.086
TAU_BC_FIDUCIAL = 0.0

MET_FIDUCIAL = {"logzsol": Fixed(MET_LOGZSOL_FIDUCIAL), "*": FIXED}

_sfh_args_d = (
    "sfhdelayed",
    dict(
        tau_main=1000,
        age_main=5000,
        tau_burst=50,
        age_burst=20,
        f_burst=0.0,
        sfr_A=1.0,
        normalise=True,
    ),
)

# CIGALE: stellar+dust (no AGN) baseline
sed_c_base = C.run_chain(
    [
        _sfh_args_d,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
    ]
)
w_base, L_base = C.to_lnu(sed_c_base)

# tengri: stellar+dust baseline
m_agn_base = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        "*": FIXED,
        "emission": {"type": "dale2014", "*": FIXED},
    },
    redshift=Fixed(0.0),
)
s_agn_base = m_agn_base.predict_state({})

# CIGALE: full chain + SKIRTOR
sed_skirtor = C.run_chain(
    [
        _sfh_args_d,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("dale2014", dict(alpha=2.0)),
        (
            "skirtor2016",
            dict(
                t=7,
                pl=1.0,
                q=1.0,
                oa=40,
                R=20,
                Mcl=0.97,
                i=30,
                disk_type=1,
                delta=0,
                fracAGN=0.3,
                lambda_fracAGN="0/0",
                law=0,
                EBV=0.03,
                temperature=100.0,
                emissivity=1.6,
            ),
        ),
    ]
)
w_skirt, L_skirt = C.to_lnu(sed_skirtor)

# tengri: full chain + AGN
m_agn = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        "*": FIXED,
        "emission": {"type": "dale2014", "*": FIXED},
    },
    agn={
        "type": "composable",
        "disc": {"type": "schartmann2005", "*": FIXED},
        "torus": {"type": "skirtor", "*": FIXED},
        "agn_log_lbol": Fixed(-0.42),
        "agn_fracAGN": Fixed(0.3),
        "*": FIXED,
    },
    redshift=Fixed(0.0),
)
s_agn = m_agn.predict_state({})

print(f"tengri L_absorbed = {float(s_agn.derived['L_absorbed']) / 3.828e33:.4f} L_sun")
print(f"tengri L_agn_bol = {float(s_agn.derived['L_agn_bol']) / 3.828e33:.4f} L_sun")

# Two-panel figure
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax in (ax_l, ax_r):
    ax.set_xlabel(r"$\lambda$ [Å]")
    ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
    ax.set_xscale("log")
    ax.set_yscale("log")

ax_l.set_title("pcigale  + SKIRTOR2016 (i = 30°, τ_9.7 = 7)  [CIGALE]")
ax_r.set_title("tengri  agn[schartmann disc + skirtor torus + polar BB]")

# CIGALE side
L_skirt_only = np.maximum(L_skirt - U.regrid(w_base, L_base, w_skirt), 1e-50)
ax_l.plot(w_base, L_base, "k--", linewidth=1.0, alpha=0.5, label="stellar + dust")
ax_l.plot(w_skirt, L_skirt, "C0-", linewidth=1.5, alpha=0.7, label="stellar + dust + SKIRTOR")
ax_l.plot(w_skirt, L_skirt_only, "C0:", linewidth=1.5, label="SKIRTOR component only")
ax_l.legend(fontsize=9, loc="lower left")
ax_l.grid(True, alpha=0.3)

# tengri side
L_t_agn_only = np.maximum(np.asarray(s_agn.derived["sed_agn"]), 1e-50)
ax_r.plot(
    s_agn_base.wave,
    s_agn_base.sed_intrinsic,
    "k--",
    linewidth=1.0,
    alpha=0.5,
    label="stellar + dust",
)
ax_r.plot(
    s_agn.wave, s_agn.sed_intrinsic, "C1-", linewidth=1.5, alpha=0.7, label="stellar + dust + AGN"
)
ax_r.plot(
    s_agn.wave, L_t_agn_only, "C1:", linewidth=1.5, label="composable disc + SKIRTOR torus only"
)
ax_r.legend(fontsize=9, loc="lower left")
ax_r.grid(True, alpha=0.3)

# Bound axes consistently
_xmin = float(min(w_skirt.min(), float(np.asarray(s_agn.wave).min())))
_xmax = float(max(w_skirt.max(), float(np.asarray(s_agn.wave).max())))
_ymax = max(float(np.asarray(L_skirt).max()), float(np.asarray(s_agn.sed_intrinsic).max()))
for ax in (ax_l, ax_r):
    ax.set_xlim(_xmin, _xmax)
    ax.set_ylim(_ymax * 1e-6, _ymax * 2)

fig.tight_layout()
out_path = Path("reproduction/cigale/_figs/cigale_09_agn_skirtor.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved {out_path}")
