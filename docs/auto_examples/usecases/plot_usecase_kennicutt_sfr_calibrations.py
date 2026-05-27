"""
Kennicutt+1998 SFR calibrations: UV, Hα, and L_IR
====================================================

Three of the most-used star-formation-rate indicators agree only for
specific assumed SFHs. We mock a constant-SFR galaxy across SFR = 0.01
to 100 M☉/yr and read each indicator out:

- ``L_FUV(1500 Å)`` — Kennicutt 1998, SFR/L_FUV = 1.4 × 10⁻²⁸
- ``L_Hα`` — Kennicutt 1998, SFR/L_Hα = 7.9 × 10⁻⁴²
- ``L_IR(8-1000 μm)`` — Kennicutt 1998, SFR/L_IR = 4.5 × 10⁻⁴⁴

All three should recover the input SFR within a factor of ~2 if the
SFH is constant for ≳ 100 Myr. Departures from the 1:1 line for any
indicator measure the bias from violating the constant-SFR
assumption — see also ``plot_usecase_sfr_indicator_compare.py``.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18
L_SUN_ERG_S = 3.839e33

# Kennicutt 1998 (Salpeter IMF) calibrations
K98_UV = 1.4e-28  # M_sun / yr / (erg / s / Hz)
K98_HA = 7.9e-42  # M_sun / yr / (erg / s)
K98_IR = 4.5e-44  # M_sun / yr / (erg / s)

SSP = tengri.load_ssp()


def _measure(log_sfr):
    model = tengri.SEDModel.build(
        SSP,
        sfh={"type": "const", "*": tengri.FIXED, "log_sfr": float(log_sfr)},
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 0.0,
            "tau_bc": 0.0,
            "emission": {"type": "dale2014", "*": tengri.FIXED},
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)
    wave = np.asarray(out.wavelength)
    l_nu = np.asarray(out.sed)
    i1500 = int(np.argmin(np.abs(wave - 1500)))
    L_uv = float(l_nu[i1500])
    ir = (wave > 8e4) & (wave < 1e7)
    nu_ir = C_AA_PER_S / wave[ir]
    order = np.argsort(nu_ir)
    L_ir = float(np.trapezoid(l_nu[ir][order], nu_ir[order]))
    # crude Hα flux: integrate line over a ±20 Å window minus median continuum
    line = (wave > 6543) & (wave < 6583)
    cont = (wave > 6510) & (wave < 6540)
    f_lam = l_nu * C_AA_PER_S / wave**2
    cont_lvl = float(np.median(f_lam[cont]))
    delta = float(wave[line][1] - wave[line][0]) if line.sum() > 1 else 1.0
    L_ha = float(np.sum(np.maximum(f_lam[line] - cont_lvl, 0.0)) * delta)
    return L_uv, L_ha, L_ir


log_sfr_grid = np.linspace(-2.0, 2.0, 9)
L_uv = np.empty_like(log_sfr_grid)
L_ha = np.empty_like(log_sfr_grid)
L_ir = np.empty_like(log_sfr_grid)
for i, s in enumerate(log_sfr_grid):
    L_uv[i], L_ha[i], L_ir[i] = _measure(s)

sfr_uv = K98_UV * L_uv
sfr_ha = K98_HA * L_ha
sfr_ir = K98_IR * L_ir
sfr_in = 10.0**log_sfr_grid

fig, ax = plt.subplots(figsize=(6.4, 5.0))
diag = np.array([1e-3, 1e3])
ax.plot(diag, diag, color="0.55", lw=0.8, ls="--", label="1:1")
ax.loglog(sfr_in, sfr_uv, "o-", color="#3377cc", lw=1.6, ms=6, label=r"K98 UV (1500 Å)")
ax.loglog(
    sfr_in,
    np.where(sfr_ha > 0, sfr_ha, np.nan),
    "s-",
    color="#cc3333",
    lw=1.6,
    ms=6,
    label=r"K98 H$\alpha$",
)
ax.loglog(sfr_in, sfr_ir, "^-", color="#cc8833", lw=1.6, ms=6, label=r"K98 $L_{\rm IR}$")
ax.set(
    xlim=(1e-2, 1e2),
    ylim=(1e-3, 1e3),
    xlabel=r"input SFR  [$M_\odot\,$yr$^{-1}$]",
    ylabel=r"SFR from K98 indicator  [$M_\odot\,$yr$^{-1}$]",
)
ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.tight_layout()
plt.savefig("plot_usecase_kennicutt_sfr_calibrations.png", dpi=150, bbox_inches="tight")
