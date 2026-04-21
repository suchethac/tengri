#!/usr/bin/env python3
"""Build data/feltre_grid.h5 from NEOGAL Feltre+2016 ASCII grids.

Downloads
---------
The raw ASCII grids were downloaded from::

    http://www.iap.fr/neogal/ewExternalFiles/AGN_NLR_nebular_feltre16.tar.gz

They are stored at ``data/neogal/AGN_NLR_nebular_feltre16/nlr_nebular_Z*.txt``.

Normalization convention
------------------------
The NEOGAL files store line luminosities in **erg/s** normalized to an AGN
accretion luminosity of ``L_acc = 10^45 erg/s``.  Line luminosities scale
linearly with L_acc (Feltre et al. 2016, Section 2.1).

The ``FeltreNLRBackend`` in ``agn_nebular.py`` expects the grid in a different
normalization: ``log10(L_Hβ / Q_H)`` and ``line_ratios = L_line / L_Hβ``.
This script performs the conversion.

Conversion
----------
``L_Hβ / Q_H`` is related to the stored value ``L_Hβ_norm`` (erg/s per
L_acc = 10^45 erg/s) via the ionizing photon rate:

    Q_H_norm = f_ion(α) × 10^45 erg/s / <hν_ion(α)>

where ``f_ion`` is the fraction of the bolometric luminosity emitted as
ionizing photons (below 912 Å) and ``<hν_ion>`` is the mean ionizing photon
energy, both depending on the EUV power-law slope α.  This calculation uses
the same ``_log_qh_from_lacc`` function as ``agn_nebular.py``.

HDF5 schema (data/feltre_grid.h5)
----------------------------------
Group "/feltre":
  alpha_axis        : (4,)         float64  ionizing slope axis [-2.0,-1.7,-1.4,-1.2]
  logUs_axis        : (9,)         float64  log(U_S) axis (ascending)
  logn_axis         : (3,)         float64  log(n_H) axis
  logZ_axis         : (16,)        float64  log10(Z) axis
  xi_d_axis         : (3,)         float64  dust-to-metal ratio axis
  line_wavelengths_aa : (20,)      float64  vacuum wavelengths [Angstrom]
  line_names        : (20,)        str      line identifiers
  logHB_per_logq    : (4,9,3,16,3) float64  log10(L_Hβ / Q_H) [erg/photon]
  line_ratios       : (4,9,3,16,3,20) float64  L_line / L_Hβ (dimensionless)

References
----------
Feltre, Charlot & Gutkin 2016, MNRAS, 456, 3354 (arXiv:1511.08217).

Usage
-----
    python scripts/build_feltre_grid.py
    python scripts/build_feltre_grid.py \\
        --input data/neogal/AGN_NLR_nebular_feltre16/ \\
        --output data/feltre_grid.h5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# ── Physical constants (cgs) ──────────────────────────────────────
_C_CGS: float = 2.99792458e10  # speed of light [cm/s]
_H_PLANCK: float = 6.62607015e-34 * 1e7  # Planck constant [erg·s]
_NU_LYMAN: float = _C_CGS / (911.76e-8)  # Lyman limit frequency [Hz]
_RYDBERG_ERG: float = 2.1799e-11  # 13.6 eV [erg]

# Vacuum wavelengths [Angstrom] for the 20 Feltre+2016 emission lines.
# Source: NIST ASD (Kramida et al. 2023, https://physics.nist.gov/asd)
_LINE_WAVELENGTHS_AA: list[float] = [
    3727.09,  # [OII]3727 (doublet centroid)
    4861.33,  # Hbeta
    4958.92,  # [OIII]4959
    5006.84,  # [OIII]5007
    6300.30,  # [OI]6300
    6548.05,  # [NII]6548
    6564.61,  # Halpha (vacuum)
    6583.45,  # [NII]6584
    6716.44,  # [SII]6717
    6730.82,  # [SII]6731
    1240.81,  # NV1240 (doublet centroid)
    1548.20,  # CIV1548
    1550.77,  # CIV1551
    1640.40,  # HeII1640
    1660.81,  # OIII]1661
    1666.15,  # OIII]1666
    1882.71,  # [SiIII]1883
    1892.03,  # SiIII]1888
    1906.68,  # [CIII]1907
    1908.73,  # CIII]1910
]

_LINE_NAMES: list[str] = [
    "[OII]3727",
    "Hbeta",
    "[OIII]4959",
    "[OIII]5007",
    "[OI]6300",
    "[NII]6548",
    "Halpha",
    "[NII]6584",
    "[SII]6717",
    "[SII]6731",
    "NV1240",
    "CIV1548",
    "CIV1551",
    "HeII1640",
    "OIII]1661",
    "OIII]1666",
    "[SiIII]1883",
    "SiIII]1888",
    "[CIII]1907",
    "CIII]1910",
]


def _z_from_filename(fpath: Path) -> float:
    """Parse metallicity from filename like nlr_nebular_Z014.txt → 0.014."""
    m = re.search(r"_Z(\d+)\.txt$", fpath.name)
    if m is None:
        raise ValueError(f"Cannot parse Z from filename: {fpath.name}")
    digits = m.group(1)
    return float("0." + digits)


def _log_qh_from_lacc_alpha(l_acc_erg: float, alpha_pl: float) -> float:
    """Compute log10(Q_H) from accretion luminosity and EUV slope.

    Uses the same analytic approximation as ``agn_nebular.py``.
    """
    nu_lyman = _NU_LYMAN
    nu_max = _C_CGS / 1e-8  # 1 Angstrom
    nu_min = _C_CGS / (10.0e-4)  # 10 micron

    a = alpha_pl
    ap1 = a + 1.0
    safe_ap1 = ap1 if abs(ap1) > 1e-8 else 1e-8

    int_total = (nu_max**safe_ap1 - nu_min**safe_ap1) / safe_ap1
    int_ion = (nu_max**safe_ap1 - nu_lyman**safe_ap1) / safe_ap1
    f_ion = abs(int_ion / int_total)
    f_ion = max(0.01, min(1.0, f_ion))

    safe_a = a if abs(a) > 1e-8 else 1e-8
    int_num = int_ion
    int_den = (nu_max**safe_a - nu_lyman**safe_a) / safe_a
    mean_hnu = _H_PLANCK * abs(int_num / int_den)
    mean_hnu = max(mean_hnu, _RYDBERG_ERG)

    l_ion = f_ion * l_acc_erg
    q_h = l_ion / mean_hnu
    return np.log10(max(q_h, 1.0))


def _load_one_z_file(fpath: Path) -> np.ndarray:
    """Load a single Feltre+2016 Z-file into a numpy array.

    Returns
    -------
    data : (N_rows, 24) ndarray
        Columns: [log_Us, xid, nh, alpha, 20 line luminosities erg/s]
    """
    rows = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows)


def build(
    input_dir: Path,
    output_path: Path,
) -> None:
    """Build data/feltre_grid.h5 from NEOGAL ASCII files.

    Parameters
    ----------
    input_dir : Path
        Directory containing ``nlr_nebular_Z*.txt`` files.
    output_path : Path
        Destination HDF5 file.
    """
    files = sorted(input_dir.glob("nlr_nebular_Z*.txt"))
    if not files:
        print(f"ERROR: No nlr_nebular_Z*.txt files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} metallicity files in {input_dir}")

    # ── Parse axis values from first file ────────────────────────
    data0 = _load_one_z_file(files[0])
    alpha_axis = np.unique(data0[:, 3])
    logUs_axis = np.unique(data0[:, 0])
    logn_axis = np.unique(np.log10(data0[:, 2]))
    xi_d_axis = np.unique(data0[:, 1])

    alpha_axis = np.sort(alpha_axis)  # ascending: -2.0, -1.7, -1.4, -1.2
    logUs_axis = np.sort(logUs_axis)  # ascending: -5.0 ... -1.0
    logn_axis = np.sort(logn_axis)  # ascending: log10(100), log10(1000), log10(10000)
    xi_d_axis = np.sort(xi_d_axis)  # ascending: 0.1, 0.3, 0.5

    # ── Parse metallicity axis ────────────────────────────────────
    z_values = [_z_from_filename(f) for f in files]
    z_values = np.array(sorted(z_values))
    logZ_axis = np.log10(z_values)

    n_alpha = len(alpha_axis)
    n_logUs = len(logUs_axis)
    n_logn = len(logn_axis)
    n_logZ = len(logZ_axis)
    n_xid = len(xi_d_axis)
    n_lines = len(_LINE_WAVELENGTHS_AA)

    print(f"Grid shape: ({n_alpha}, {n_logUs}, {n_logn}, {n_logZ}, {n_xid})")
    print(f"  alpha:  {alpha_axis}")
    print(f"  logUs:  {logUs_axis}")
    print(f"  logn_H: {logn_axis} (log10 cm^-3)")
    print(f"  logZ:   {logZ_axis}")
    print(f"  xi_d:   {xi_d_axis}")

    # ── Allocate output arrays ────────────────────────────────────
    # Normalization: line luminosities in NEOGAL are erg/s per L_acc = 10^45
    # We store:
    #   logHB_per_logq  = log10(L_Hβ / Q_H)   [erg/photon]
    #   line_ratios     = L_line / L_Hβ
    logHB_per_logq = np.full((n_alpha, n_logUs, n_logn, n_logZ, n_xid), np.nan)
    line_ratios = np.full((n_alpha, n_logUs, n_logn, n_logZ, n_xid, n_lines), np.nan)

    # Reference accretion luminosity [erg/s]
    L_ACC_NORM = 1e45

    # ── Fill grid ─────────────────────────────────────────────────
    for iz, (_z_val, fpath) in enumerate(zip(sorted(z_values), files)):
        data = _load_one_z_file(fpath)
        for row in data:
            log_us, xi_d, n_h, alpha = row[0], row[1], row[2], row[3]
            line_lum = row[4:]  # erg/s per L_acc=1e45

            # Find axis indices
            i_alpha = np.argmin(np.abs(alpha_axis - alpha))
            i_logUs = np.argmin(np.abs(logUs_axis - log_us))
            i_logn = np.argmin(np.abs(logn_axis - np.log10(n_h)))
            i_xid = np.argmin(np.abs(xi_d_axis - xi_d))

            # Hβ is column index 1 (0-indexed from line columns)
            l_hb_erg = line_lum[1]

            # Compute Q_H for this alpha at L_acc = 10^45 erg/s
            log_qh = _log_qh_from_lacc_alpha(L_ACC_NORM, alpha)
            q_h = 10.0**log_qh

            # Store normalized quantities
            if l_hb_erg > 0 and q_h > 0:
                logHB_per_logq[i_alpha, i_logUs, i_logn, iz, i_xid] = np.log10(l_hb_erg) - log_qh
                # line_ratios = L_line / L_Hβ
                line_ratios[i_alpha, i_logUs, i_logn, iz, i_xid, :] = line_lum / l_hb_erg
            else:
                logHB_per_logq[i_alpha, i_logUs, i_logn, iz, i_xid] = -99.0
                line_ratios[i_alpha, i_logUs, i_logn, iz, i_xid, :] = 0.0

    nan_frac = np.isnan(logHB_per_logq).sum() / logHB_per_logq.size
    if nan_frac > 0:
        print(f"WARNING: {nan_frac:.1%} of grid points are NaN (unfilled).")

    # ── Write HDF5 ────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["source"] = "Feltre, Charlot & Gutkin 2016, MNRAS 456, 3354 (arXiv:1511.08217)"
        f.attrs["neogal_url"] = (
            "http://www.iap.fr/neogal/ewExternalFiles/AGN_NLR_nebular_feltre16.tar.gz"
        )
        f.attrs["cloudy_version"] = "c13.03"
        f.attrs["normalization"] = (
            "logHB_per_logq = log10(L_Hbeta [erg/s] / Q_H [photons/s]); "
            "line_ratios = L_line / L_Hbeta"
        )

        grp = f.create_group("feltre")
        grp.create_dataset("alpha_axis", data=alpha_axis)
        grp.create_dataset("logUs_axis", data=logUs_axis)
        grp.create_dataset("logn_axis", data=logn_axis)
        grp.create_dataset("logZ_axis", data=logZ_axis)
        grp.create_dataset("xi_d_axis", data=xi_d_axis)
        grp.create_dataset("line_wavelengths_aa", data=np.array(_LINE_WAVELENGTHS_AA))
        dt = h5py.special_dtype(vlen=str)
        ds = grp.create_dataset("line_names", shape=(n_lines,), dtype=dt)
        for i, name in enumerate(_LINE_NAMES):
            ds[i] = name
        grp.create_dataset("logHB_per_logq", data=logHB_per_logq)
        grp.create_dataset("line_ratios", data=line_ratios)

    print(f"\nWrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    print("Run validation with:")
    print(f"  python scripts/download_feltre_grid.py --validate --output {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data/feltre_grid.h5 from NEOGAL Feltre+2016 ASCII grids.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/neogal/AGN_NLR_nebular_feltre16"),
        help="Directory containing nlr_nebular_Z*.txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/feltre_grid.h5"),
        help="Output path for feltre_grid.h5.",
    )
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
