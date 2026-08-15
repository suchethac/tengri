#!/usr/bin/env python3
r"""Build themis_templates.h5 directly from the FSPS/DustEM THEMIS grids.

The canonical THEMIS (Jones+2017) IR emission is produced by DustEM
(ias.u-psud.fr/DUSTEM). FSPS ships the pre-run DustEM output as
``$SPS_HOME/dust/dustem/THEMIS_MW3.1_{00..100}.dat`` — one file per qhac
(small a-C(:H) mass fraction), each column-paired into a delta-U (single
radiation field at U=U_min) template and a power-law (U_min..U_max, alpha=2)
template per U_min. This is the *direct* source (not CIGALE's repackaging).

Unlike the previous build, the two components are kept with their **real
relative power**: the power-law column carries DustEM's true higher
luminosity per H, so a dust-mass fraction ``gamma`` warms the SED correctly.
single_u is normalised to unit frequency integral; powerlaw is normalised by
the *same* per-(qhac,U_min) factor, so it integrates to the real ratio
∫powerlaw / ∫single_u. The loader passes these through unchanged
(spectra_unit flag) and the forward renormalises the mix to L_absorbed.

References: Jones, Köhler, Ysard et al. 2017 A&A 602 A46; Conroy+2009 (FSPS).
"""

import os
import sys

import h5py
import numpy as np

SPS = os.environ.get("SPS_HOME")
if not SPS:
    raise SystemExit(
        "SPS_HOME is not set. Point it at your FSPS checkout, e.g.\n"
        "    export SPS_HOME=/path/to/fsps"
    )
DUSTEM = os.path.join(SPS, "dust", "dustem")
# FSPS THEMIS grids (sps_vars.f90, THEMIS block)
QPAH = np.array([0.02, 0.06, 0.10, 0.14, 0.17, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40]) / 2.2 * 100
UMIN = np.array(
    [
        0.1,
        0.12,
        0.15,
        0.17,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1.0,
        1.2,
        1.5,
        1.7,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        10.0,
        12.0,
        15.0,
        17.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        50.0,
        80.0,
    ]
)
TAGS = ["00", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]  # sorted -> QPAH order
assert len(TAGS) == len(QPAH)
nq, nu = len(QPAH), len(UMIN)

files = [os.path.join(DUSTEM, f"THEMIS_MW3.1_{t}.dat") for t in TAGS]
for f in files:
    if not os.path.isfile(f):
        sys.exit(f"missing {f}")

# read wavelength + n_wave from first file
d0 = np.loadtxt(files[0])
wave_um = d0[:, 0]
n_wave = wave_um.shape[0]
ncol = d0.shape[1]
assert ncol == 1 + 2 * nu, f"ncol {ncol} != 1+2*{nu}"

C_CGS = 2.99792458e10
AA_TO_CM = 1e-8
wave_aa = wave_um * 1e4
nu_hz = C_CGS / (wave_aa * AA_TO_CM)  # descending for ascending wave

single = np.zeros((nq, nu, n_wave))
power = np.zeros((nq, nu, n_wave))
real_ratio = np.zeros((nq, nu))
for iq, fpath in enumerate(files):
    d = np.loadtxt(fpath)
    assert np.allclose(d[:, 0], wave_um)
    for iu in range(nu):
        s_raw = d[:, 1 + 2 * iu]  # delta-U (single) at U=U_min
        p_raw = d[:, 2 + 2 * iu]  # power-law U_min..U_max (alpha=2)
        s_int = -np.trapezoid(s_raw, nu_hz)
        p_int = -np.trapezoid(p_raw, nu_hz)
        real_ratio[iq, iu] = p_int / s_int if s_int > 0 else 1.0
        # single -> unit integral; power -> same scale (carries real ratio)
        single[iq, iu] = s_raw / s_int if s_int > 0 else s_raw
        power[iq, iu] = p_raw / s_int if s_int > 0 else p_raw

print(f"THEMIS from FSPS: nq={nq} numin={nu} nwave={n_wave}")
print(
    f"real powerlaw/single ratio: U_min=0.1 -> {real_ratio[:, 0].mean():.2f}, "
    f"U_min=1.0 -> {real_ratio[:, 13].mean():.2f}, U_min=10 -> {real_ratio[:, 26].mean():.2f}"
)
print(f"single distinct from powerlaw? {not np.allclose(single, power)}")

out = "data/themis_templates.h5"
# unlink the committed symlink/file so we write a fresh local file
if os.path.islink(out) or os.path.isfile(out):
    os.remove(out)
with h5py.File(out, "w") as f:
    f.create_dataset("wavelength_aa", data=wave_aa)
    f.create_dataset("umin_grid", data=UMIN)
    f.create_dataset("qhac_grid", data=QPAH)
    f.create_dataset("single_u", data=single)
    f.create_dataset("powerlaw", data=power)  # carries the REAL relative power
    f.attrs["spectra_unit"] = "L_nu normalized (integral over nu = 1)"
    f.attrs["model"] = "THEMIS (Jones+2017) via FSPS/DustEM direct grids"
    f.attrs["source"] = "$SPS_HOME/dust/dustem/THEMIS_MW3.1_*.dat (DustEM, ias.u-psud.fr/DUSTEM)"
    f.attrs["paper"] = "Jones, Köhler, Ysard et al. 2017 A&A 602 A46"
    f.attrs["alpha_powerlaw"] = 2.0
    f.attrs["note"] = (
        "single_u integrates to 1; powerlaw integrates to the real "
        "DustEM ratio (∫powerlaw/∫single_u), so gamma is a mass fraction."
    )
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
