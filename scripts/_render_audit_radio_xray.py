"""Render §10 radio + §11 X-ray audit figures — tengri vs CIGALE/X-CIGALE.

Produces two PNGs:
- ``reproduction/cigale/_figs/cigale_audit_10_radio.png``
- ``reproduction/cigale/_figs/cigale_audit_11_xray.png``

Each shows two panels: SEDs overlaid in the relevant band, and the
tengri/CIGALE ratio. Mirrors the numerical results printed by
``reproduction/cigale/_drivers/consistency_audit.py`` §10 / §11.

Run::

    PYTHONPATH=src:. .venv/bin/python \\
        scripts/_render_audit_radio_xray.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reproduction.cigale._drivers import cigale_driver as C
from tengri.components.radio.radio import radio_sfr_bell2003
from tengri.components.xray.xray import xray_total

OUT = Path("reproduction/cigale/_figs")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# §10 radio
# ---------------------------------------------------------------------------
sed_c10 = C.run_chain(
    [
        (
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
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("dale2014", dict(alpha=2.0)),
        ("radio", dict(qir_sf=2.58, alpha_sf=0.8, R_agn=0.0, alpha_agn=0.7)),
    ]
)
wave_c10, L_c10 = C.to_lnu(sed_c10)
L_dust_erg = float(sed_c10.info["dust.luminosity"]) * 1e7

mask_radio = (wave_c10 >= 1e8) & (wave_c10 <= 1e10)
wave_r = wave_c10[mask_radio]
L_radio_c = L_c10[mask_radio]
L_radio_t = np.asarray(radio_sfr_bell2003(wave_r, L_ir=L_dust_erg, q_ir=2.58, alpha_sf=0.8))

fig, (ax_sed, ax_ratio) = plt.subplots(
    2,
    1,
    figsize=(6.0, 5.5),
    sharex=True,
    gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.07},
)
nu_r = 2.99792458e18 / wave_r
ax_sed.loglog(wave_r / 1e8, nu_r * L_radio_c, color="C0", lw=2, label="CIGALE radio (SF)")
ax_sed.loglog(
    wave_r / 1e8, nu_r * L_radio_t, color="C3", lw=1.4, ls="--", label="tengri radio_sfr_bell2003"
)
ax_sed.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
ax_sed.set_title("§10 Radio SF synchrotron — Bell+2003, q$_{IR}$=2.58, α=0.8")
ax_sed.legend(loc="upper right", frameon=False)
ax_sed.grid(True, which="both", alpha=0.25)

ratio = L_radio_t / L_radio_c
ax_ratio.semilogx(wave_r / 1e8, ratio, color="k", lw=1.4)
ax_ratio.axhline(1.0, color="grey", lw=0.8, ls=":")
ax_ratio.axhspan(0.95, 1.05, color="grey", alpha=0.15, label="±5%")
ax_ratio.set_xlim(1, 100)
ax_ratio.set_ylim(0.9, 1.1)
ax_ratio.set_xlabel(r"$\lambda$ [cm]")
ax_ratio.set_ylabel("tengri / CIGALE")
ax_ratio.text(
    0.02,
    0.92,
    f"median = {np.median(ratio):.3f}",
    transform=ax_ratio.transAxes,
    fontsize=9,
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8),
)
ax_ratio.grid(True, which="both", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "cigale_audit_10_radio.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'cigale_audit_10_radio.png'}  median = {np.median(ratio):.4f}")


# ---------------------------------------------------------------------------
# §11 X-ray
# ---------------------------------------------------------------------------
sed_c11 = C.run_chain(
    [
        (
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
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
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
        (
            "yang20",
            dict(
                gam=1.8,
                E_cut=300,
                alpha_ox=-1.5,
                max_dev_alpha_ox=999,
                angle_coef="0.5 & 0",
                det_lmxb=0.0,
                det_hmxb=0.0,
            ),
        ),
    ]
)
wave_c11, L_c11 = C.to_lnu(sed_c11)

sfr = float(sed_c11.info["sfh.sfr100Myrs"])
mstar = float(sed_c11.info["stellar.m_star"])
age_m_star_gyr = float(sed_c11.info["stellar.age_m_star"]) * 1e-3
metallicity_z = float(sed_c11.info["stellar.metallicity"])
Lnu_2500_30deg = float(sed_c11.info["agn.intrin_Lnu_2500A_30deg"]) * 1e7
cos_inc = float(np.cos(np.deg2rad(float(sed_c11.info["agn.i"]))))

mask_xray = (wave_c11 >= 1.24) & (wave_c11 <= 124.0)
wave_x = wave_c11[mask_xray]
L_xray_c = L_c11[mask_xray]
L_xray_t = np.asarray(
    xray_total(
        wave_x,
        sfr=sfr,
        stellar_mass=mstar,
        metallicity_z=metallicity_z,
        stellar_age_gyr=age_m_star_gyr,
        l_2500_30deg=Lnu_2500_30deg,
        gamma_hmxb=2.0,
        gamma_lmxb=1.6,
        gamma_agn=1.8,
        E_cut=300.0,
        delta_alpha_ox=0.0,
        cos_inc=cos_inc,
        apply_anisotropy=True,
        a1=0.5,
        a2=0.0,
        log_nh=20.0,
    )
)

# Convert λ [Å] → E [keV]
E_keV = 12.398 / wave_x

fig, (ax_sed, ax_ratio) = plt.subplots(
    2,
    1,
    figsize=(6.0, 5.5),
    sharex=True,
    gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.07},
)
nu_x = 2.99792458e18 / wave_x
ax_sed.loglog(E_keV, nu_x * L_xray_c, color="C0", lw=2, label="CIGALE yang20 (X-CIGALE)")
ax_sed.loglog(E_keV, nu_x * L_xray_t, color="C3", lw=1.4, ls="--", label="tengri xray_total")
ax_sed.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
ax_sed.set_title("§11 X-ray — Yang+2020 corona + Lehmer+2016 XRB, Γ=1.8, αox=−1.5")
ax_sed.legend(loc="lower right", frameon=False)
ax_sed.axvspan(0.5, 2.0, color="C0", alpha=0.08, label="0.5–2 keV")
ax_sed.axvspan(2.0, 10.0, color="C1", alpha=0.08, label="2–10 keV")
ax_sed.grid(True, which="both", alpha=0.25)

ratio = L_xray_t / L_xray_c
ax_ratio.semilogx(E_keV, ratio, color="k", lw=1.4)
ax_ratio.axhline(1.0, color="grey", lw=0.8, ls=":")
ax_ratio.axhspan(0.95, 1.05, color="grey", alpha=0.15)
ax_ratio.set_xlim(0.1, 10.0)
ax_ratio.set_ylim(0.9, 1.1)
ax_ratio.set_xlabel("E [keV]")
ax_ratio.set_ylabel("tengri / CIGALE")
soft = (E_keV >= 0.5) & (E_keV <= 2.0)
hard = (E_keV >= 2.0) & (E_keV <= 10.0)
note = f"0.5–2 keV: {np.median(ratio[soft]):.3f}\n2–10 keV: {np.median(ratio[hard]):.3f}"
ax_ratio.text(
    0.02,
    0.92,
    note,
    transform=ax_ratio.transAxes,
    fontsize=9,
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8),
)
ax_ratio.grid(True, which="both", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "cigale_audit_11_xray.png", dpi=140, bbox_inches="tight")
print(
    f"wrote {OUT / 'cigale_audit_11_xray.png'}  "
    f"soft = {np.median(ratio[soft]):.4f}  hard = {np.median(ratio[hard]):.4f}"
)
