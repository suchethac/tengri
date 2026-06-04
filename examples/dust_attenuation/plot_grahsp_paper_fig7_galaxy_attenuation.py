"""
GRAHSP Fig. 7 reproduction: attenuation of the galaxy model
============================================================

Reproduction of Fig. 7 of Buchner et al. (2024, GRAHSP): a star-forming
galaxy SED from intrinsic (dark blue) to strongly attenuated (dark red) as the
diffuse colour excess E(B-V) is swept from 0.01 to 10. Energy balance routes
the attenuated UV/optical light into the far-IR dust bump (Dale 2014), so the
curves pivot about the FIR peak while the UV is progressively suppressed.

The extremely attenuated, low-metallicity starbursting galaxy **Haro 11**
(photometry from NED, mirroring Lyu et al. 2016) is overplotted as a dashed
red curve for reference.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

import tengri
from tengri import FIXED, Fixed, SEDModel
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_NM_HZ = 2.99792458e17
SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
HARO11 = Path(__file__).parent / "data" / "haro11_ned_sed.txt"
R_V = 4.05  # Calzetti: A_V = R_V * E(B-V); tau_V = A_V / 1.086

ssp = tengri.load_ssp(SSP_PATH)
wave_aa = jnp.logspace(np.log10(800.0), np.log10(1.0e7), 3000)  # 0.08 - 1000 um
wave_um = np.asarray(wave_aa) / 1e4

ebv_grid = np.logspace(np.log10(0.01), np.log10(10.0), 9)
norm = colors.LogNorm(vmin=0.01, vmax=10.0)
cmap = cm.get_cmap("RdBu_r")


def nu_Lnu(lnu):
    return np.asarray(lnu) * (C_NM_HZ / (np.asarray(wave_aa) * 0.1))


fig, ax = plt.subplots(figsize=(6.6, 7.6))

norm_ref = None
for ebv in ebv_grid:
    tau_diff = R_V * ebv / 1.086
    model = SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "*": FIXED,
            "tau_gyr": 5.0,
            "age_gyr": 3.0,
            "log_total_mass": 10.0,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "*": FIXED,
            "tau_bc": 0.3,  # fixed birth-cloud baseline (stabilises the FIR peak)
            "tau_diff": tau_diff,  # diffuse screen scales with E(B-V)
            "emission": {"type": "dale2014", "*": FIXED},
        },
        redshift=Fixed(0.01),
    )
    params = model.spec.get_fixed_values()
    rest = model.predict_rest_sed(params, wave_aa)
    lnu = np.asarray(rest[1] if np.ndim(rest) == 2 else rest)
    lflam = nu_Lnu(lnu)
    if norm_ref is None:
        # Normalise so the FIR dust peak sits near ~5 (paper scaling).
        fir = (wave_um > 30) & (wave_um < 300)
        norm_ref = lflam[fir].max() / 5.0
    ax.plot(wave_um, lflam / norm_ref, color=cmap(norm(ebv)), lw=1.4, zorder=3)

# Haro 11 overlay: raw NED photometry has many measurements per band (plus
# radio + upper limits), so bin into log-wavelength medians for a smooth,
# representative SED (cf. the Lyu+ 2016 model curve in the paper).
if HARO11.exists():
    h = np.loadtxt(HARO11)
    hw, hf = h[:, 0], h[:, 1]
    good = np.isfinite(hf) & (hf > 0) & (hw > 0.1) & (hw < 500.0)
    hw, hf = hw[good], hf[good]
    edges = np.logspace(np.log10(0.1), np.log10(500.0), 20)
    centers, meds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (hw >= lo) & (hw < hi)
        if sel.sum() >= 2:  # require >=2 measurements to suppress single outliers
            centers.append(np.sqrt(lo * hi))
            meds.append(np.median(hf[sel]))
    centers, meds = np.array(centers), np.array(meds)
    # Despike: 2 passes dropping bins >0.5 dex from a 3-point rolling median.
    for _ in range(2):
        logm = np.log10(meds)
        roll = np.array([np.median(logm[max(0, i - 1) : i + 2]) for i in range(len(logm))])
        keep = np.abs(logm - roll) < 0.5
        centers, meds = centers[keep], meds[keep]
    fir = (centers > 40) & (centers < 200)
    if fir.any():
        meds = meds / (np.median(meds[fir]) / 5.0)
    # Final guard: in lambda*F_lambda a dusty starburst cannot exceed its FIR
    # bump, so drop residual NED outlier bins above 3x the FIR peak (~5).
    ok = meds < 15.0
    centers, meds = centers[ok], meds[ok]
    ax.plot(
        centers,
        meds,
        color="red",
        ls="--",
        lw=1.8,
        marker="o",
        ms=3.5,
        label="Haro 11 (NED)",
        zorder=5,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1, 1000.0)
ax.set_ylim(1e-3, 1e2)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda F_\lambda$ [arb.]")
ax.legend(loc="lower center", frameon=True, fontsize=10)

secax = ax.secondary_xaxis(
    "top", functions=(lambda x: C_NM_HZ / 1e3 / x, lambda nu: C_NM_HZ / 1e3 / nu)
)
secax.set_xlabel("Frequency [Hz]")

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
cbar.set_label("E(B-V)")
cbar.set_ticks([0.01, 0.3, 10.0])
cbar.set_ticklabels(["0.01", "0.3", "10"])

fig.tight_layout()
plt.show()
