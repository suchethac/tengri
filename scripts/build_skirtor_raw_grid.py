# SPDX-License-Identifier: BSD-3-Clause
"""Build a faithful SKIRTOR (Stalevski 2016) HDF5 grid from the raw ``.dat`` files.

Unlike the v3 grid (which dropped the ``R`` axis — only R=20 — and *reconstructed*
the total SED as disk+dust, omitting the scattered-dust column and the published
radiative-transfer total), this preserves:

- the full parameter grid **including the R (outer/inner radius ratio) axis**;
- the published **radiative-transfer total** SED (``.dat`` column 2) — the exact
  quantity codes that read SKIRTOR directly (e.g. ProSpect) use;
- all components: disk (direct+scattered stellar), dust (thermal+scattered),
  transparent.

The ``.dat`` columns (Stalevski 2016 release, also bundled with CIGALE):
    1 lambda [micron]    2 total lF_l    3 direct stellar    4 scattered stellar
    5 total dust         6 scattered dust   7 transparent     (all lF_l, W/m^2)

Stored as ``L_nu`` shape (``nu L_nu = lambda F_lambda``; ``L_nu = lF_l * lambda /
c``), relative — absolute scale is applied at use (scale total to L_bol).

Layout (tengri standard, mirrors skirtor_templates_v3.h5 + the R axis and a
``total_emission`` spectrum):

    grid/{tau_97, p, q, opening_angle, radius_ratio, cos_inclination}
    spectra/{total_emission, disk_emission, dust_emission, transparent_emission}
    wavelength
    metadata (attrs)

Usage::

    python scripts/build_skirtor_raw_grid.py \
        --input-dir <dat-dir> --output data/skirtor_raw_v4.h5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np

_FN = re.compile(r"t(\d+)_p([\d.]+)_q([\d.]+)_oa(\d+)_R(\d+)_Mcl([\d.]+)_i(\d+)_sed\.dat")
_C_AA_PER_S = 2.99792458e18


def _read_dat(path: Path):
    """Fast read of a SKIRTOR ``.dat`` file -> (wave_micron, cols[1:7] as 2D)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(x) for x in line.split()])
    a = np.asarray(rows)
    return a[:, 0], a  # wave[micron], full columns


def build(input_dir: Path, output: Path) -> None:
    files = sorted(input_dir.glob("t*_sed.dat"))
    if not files:
        raise SystemExit(f"No t*_sed.dat files in {input_dir}")

    meta = []
    for f in files:
        m = _FN.match(f.name)
        if m:
            ta, p, q, oa, R, _mcl, i = m.groups()
            meta.append((float(ta), float(p), float(q), float(oa), float(R), float(i), f))
    ta_ax = np.array(sorted({x[0] for x in meta}))
    p_ax = np.array(sorted({x[1] for x in meta}))
    q_ax = np.array(sorted({x[2] for x in meta}))
    oa_ax = np.array(sorted({x[3] for x in meta}))
    R_ax = np.array(sorted({x[4] for x in meta}))
    i_ax = np.array(sorted({x[5] for x in meta}))
    print(f"axes: ta{ta_ax} p{p_ax} q{q_ax} oa{oa_ax} R{R_ax} i{i_ax}  ({len(meta)} files)")

    # Reference wavelength grid from the first file.
    wl_um0, _ = _read_dat(meta[0][6])
    wave_aa = wl_um0 * 1e4
    nw = wave_aa.shape[0]
    shape = (len(ta_ax), len(p_ax), len(q_ax), len(oa_ax), len(R_ax), len(i_ax), nw)
    total = np.zeros(shape, np.float64)
    disk = np.zeros(shape, np.float64)
    dust = np.zeros(shape, np.float64)
    trans = np.zeros(shape, np.float64)

    def idx(ax, v):
        return int(np.argmin(np.abs(ax - v)))

    for n, (ta, p, q, oa, R, i, f) in enumerate(meta):
        wl_um, c = _read_dat(f)
        lam = wl_um * 1e4
        jac = lam / _C_AA_PER_S  # lambda F_lambda -> L_nu shape
        it = (
            idx(ta_ax, ta),
            idx(p_ax, p),
            idx(q_ax, q),
            idx(oa_ax, oa),
            idx(R_ax, R),
            idx(i_ax, i),
        )
        total[it] = c[:, 1] * jac
        disk[it] = (c[:, 2] + c[:, 3]) * jac
        dust[it] = (c[:, 4] + c[:, 5]) * jac
        trans[it] = c[:, 6] * jac
        if n % 2000 == 0:
            print(f"  {n}/{len(meta)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h:
        g = h.create_group("grid")
        g.create_dataset("tau_97", data=ta_ax)
        g.create_dataset("p", data=p_ax)
        g.create_dataset("q", data=q_ax)
        g.create_dataset("opening_angle", data=oa_ax)
        g.create_dataset("radius_ratio", data=R_ax)
        g.create_dataset("cos_inclination", data=np.cos(np.radians(i_ax)))
        g.create_dataset("inclination_deg", data=i_ax)
        s = h.create_group("spectra")
        s.create_dataset("total_emission", data=total, compression="gzip")
        s.create_dataset("disk_emission", data=disk, compression="gzip")
        s.create_dataset("dust_emission", data=dust, compression="gzip")
        s.create_dataset("transparent_emission", data=trans, compression="gzip")
        h.create_dataset("wavelength", data=wave_aa)
        h.attrs["model_name"] = "SKIRTOR (Stalevski et al. 2012, 2016)"
        h.attrs["version"] = 4
        h.attrs["wavelength_unit"] = "Angstrom"
        h.attrs["spectra_unit"] = (
            "L_nu shape (relative; nu L_nu = lambda F_lambda from .dat col 2-7)"
        )
        h.attrs["reference"] = "Stalevski et al. 2012 MNRAS 420 2756; 2016 MNRAS 458 2288"
        h.attrs["processing"] = (
            "Faithful: full R axis; published RT total (.dat col2) preserved; "
            "disk=direct+scattered stellar (col3+4); dust=thermal+scattered (col5+6); "
            "no analytic-disc substitution, no per-component renormalisation."
        )
        h.attrs["created_by"] = "scripts/build_skirtor_raw_grid.py"
    print(f"wrote {output}  shape {shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", default="data/skirtor_raw_v4.h5")
    a = ap.parse_args()
    build(Path(a.input_dir), Path(a.output))
