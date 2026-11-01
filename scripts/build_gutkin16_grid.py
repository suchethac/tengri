#!/usr/bin/env python3
r"""Build data/gutkin16_grid.h5 from the public NEOGAL Gutkin+2016 star-forming grids.

Downloads
---------
The constant-SFR nebular tables were downloaded from the NEOGAL distribution::

    http://www.iap.fr/neogal/sf-models.html

They are stored, untouched, at ``data/neogal/nebular_emission_Z*.txt``. Each file
holds one metallicity. Columns are ``log(Us)``, ``xid``, ``nh``, ``(C/O)/(C/O)sol``,
``mup`` and then 18 line luminosities in L_sun per (M_sun/yr) for a constant star
formation rate sustained over 100 Myr (Cloudy c13.03).

Normalization convention
------------------------
For each node the stored value is the line-to-ionizing-photon ratio. With
:math:`L^{\rm tab}_{\rm line}` the tabulated luminosity [L_sun / (M_sun/yr)] and
:math:`Q^{\rm tab}_H(Z, m_{\rm up})` the table's own ionizing photon rate per unit
SFR [photons/s / (M_sun/yr)]:

.. math::

    \log_{10}\frac{L_{\rm line}}{Q_H}
    = \log_{10}\!\left(\max(L^{\rm tab}_{\rm line}, 10^{-30})\,
      L_\odot\right) - \log_{10} Q^{\rm tab}_H(Z, m_{\rm up})

where :math:`L_\odot = 3.826\times10^{33}` erg/s. The ionizing photon rate is
anchored to Case B recombination, not to a stellar population model: at
:math:`\log U = -4`, :math:`n_H = 100`, :math:`({\rm C/O})/({\rm C/O})_\odot = 1`,
the H-beta luminosity is extrapolated linearly in dust-to-metal ratio to
:math:`\xi_d \to 0`, and divided by the Case B H-beta yield per ionizing photon
:math:`4.78\times10^{-13}` erg (Osterbrock & Ferland 2006, Table 4.4, T = 1e4 K).
A diagnostic Q_H computed from the BC03 SSP grid is stored alongside, for
cross-checking only; it does not enter the fitted grid.

Missing nodes
-------------
Nineteen (logU, xi_d, n_H, C/O, m_up) nodes are absent from the text files. Each is
filled from the tabulated logU row nearest to it in the same column. A missing
interior logU row has two equidistant neighbors; the lower logU row is taken.
Filled nodes are flagged in ``filled_mask``.

The Z = 0.040 table has m_up = 300 byte-identical to m_up = 100. This is a
property of the public table, recorded in the ``mup300_equals_mup100_at_logZ``
attribute rather than corrected.

Wavelengths
-----------
Labels at or above 2000 A are the NIST air wavelengths of their lines, converted
once to vacuum with :func:`tengri.utils.air_vacuum.air_to_vac`. Labels below
2000 A are stored as vacuum values directly. Vacuum is the library convention.

HDF5 schema (data/gutkin16_grid.h5)
-----------------------------------
Group "/gutkin16", float64 unless noted:
  logZ_axis              : (14,)            log10(Z), ascending
  logU_axis              : (7,)             log10(U_S)
  xi_d_axis              : (3,)             dust-to-metal ratio
  log_nH_axis            : (2,)             log10(n_H [cm^-3]) = [2, 3]
  co_ratio_axis          : (9,)             (C/O)/(C/O)_sun
  mup_axis               : (2,)             upper mass cutoff [M_sun] = [100, 300]
  line_names             : (18,)            str
  line_wavelengths_label_aa : (18,)         nominal line label [Angstrom]
  line_wavelengths_aa    : (18,)            vacuum wavelength [Angstrom]
  log_line_per_qh        : (14,7,3,2,9,2,18) log10(L_line / Q_H) [erg/photon]
  log_qh_per_sfr_g16     : (14,2)           log10(Q_H / SFR) [photons/s/(M_sun/yr)]
  log_qh_per_sfr_bc03    : (14,)            log10(Q_H / SFR), BC03 diagnostic; NaN if absent
  filled_mask            : (14,7,3,2,9,2)   bool, True where a node was filled

Attributes are written on both the file and the group.

References
----------
Gutkin, Charlot & Bruzual 2016, MNRAS, 462, 1757, doi:10.1093/mnras/stw1716
(arXiv:1607.06086), "Modelling the nebular emission from primeval to present-day
star-forming galaxies".
Osterbrock, D. E. & Ferland, G. J. 2006, Astrophysics of Gaseous Nebulae and
Active Galactic Nuclei, 2nd ed. (University Science Books), Table 4.4.

Usage
-----
    python scripts/build_gutkin16_grid.py \
        --ssp /path/to/bc03_pdva_stelib_chabrier.h5
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import h5py
import numpy as np

from tengri import load_ssp_data
from tengri.components.nebular._shared import compute_qh_log10
from tengri.utils.air_vacuum import air_to_vac

# ── Constants ─────────────────────────────────────────────────────
_LSUN_ERG: float = 3.826e33  # L_sun [erg/s]
_HB_PER_QH_ERG: float = 4.78e-13  # Case B H-beta energy per ionizing photon [erg]
_NH_KEEP: tuple[float, ...] = (100.0, 1000.0)  # fully tabulated n_H [cm^-3]
_NH_DROPPED: tuple[float, ...] = (10.0, 10000.0)  # sparsely tabulated, not kept
_T_QH_AGE_YR: float = 1e8  # 100 Myr constant-SFR window [yr]
_N_EXPECTED_FILLED: int = 19
_CASE_B_LOGU_INDEX: int = 0  # logU = -4
_CASE_B_NH_INDEX: int = 0  # n_H = 100
_CASE_B_CO_INDEX: int = 7  # (C/O) = 1.00
_HBETA_LINE_INDEX: int = 1
_NEOGAL_TAG = "Gutkin+2016 NEOGAL star-forming nebular tables"

# (name, label [Angstrom], wavelength [Angstrom], is_air)
# Labels at or above 2000 A carry NIST air wavelengths (converted once below);
# the rest are vacuum values as stored.
_LINES: tuple[tuple[str, float, float, bool], ...] = (
    # [O II] 3726 component (NEOGAL's label names the doublet); NIST air 3726.032 A.
    ("[OII]3727", 3727.0, 3726.032, True),
    ("Hbeta", 4861.0, 4861.33, True),
    ("[OIII]4959", 4959.0, 4958.91, True),
    ("[OIII]5007", 5007.0, 5006.84, True),
    ("[NII]6548", 6548.0, 6548.05, True),
    ("Halpha", 6563.0, 6562.80, True),
    ("[NII]6584", 6584.0, 6583.45, True),
    ("[SII]6717", 6717.0, 6716.44, True),
    ("[SII]6731", 6731.0, 6730.82, True),
    ("NV1240", 1240.0, 1240.81, False),  # NV doublet centroid
    ("CIV1548", 1548.0, 1548.19, False),
    ("CIV1551", 1551.0, 1550.77, False),
    ("HeII1640", 1640.0, 1640.42, False),
    ("OIII]1661", 1661.0, 1660.81, False),
    ("OIII]1666", 1666.0, 1666.15, False),
    ("[SiIII]1883", 1883.0, 1882.71, False),
    ("SiIII]1888", 1888.0, 1892.03, False),
    ("CIII]1908", 1908.0, 1908.73, False),  # CIII] blend
)
_N_LINES: int = len(_LINES)

_WAVE_CONVENTION = (
    "vacuum; labels >= 2000 A converted from air with tengri.utils.air_vacuum (Cloudy convention)"
)
_CITATION = "Gutkin, Charlot & Bruzual 2016, MNRAS 462, 1757, doi:10.1093/mnras/stw1716"


def z_from_filename(path: Path) -> float:
    """Parse metallicity from a NEOGAL filename.

    Parameters
    ----------
    path : Path
        File named ``nebular_emission_Z<digits>.txt``. The digits are the decimal
        digits after ``0.``, so ``Z0001`` is 0.0001 and ``Z040`` is 0.040.

    Returns
    -------
    float
        Absolute metallicity Z.

    Raises
    ------
    ValueError
        If the filename does not match the NEOGAL pattern.
    """
    match = re.search(r"_Z(\d+)\.txt$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse Z from filename: {path.name}")
    return float("0." + match.group(1))


def load_neogal_sf_file(path: Path) -> np.ndarray:
    """Read one NEOGAL constant-SFR table into a float64 array.

    Parameters
    ----------
    path : Path
        Text file with ``#``-prefixed header lines and whitespace-separated rows.

    Returns
    -------
    ndarray, shape (n_rows, 23)
        Columns ``log(Us)``, ``xid``, ``nh``, ``(C/O)/(C/O)sol``, ``mup`` and 18
        line luminosities [L_sun / (M_sun/yr)].
    """
    rows = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rows.append([float(token) for token in stripped.split()])
    return np.array(rows, dtype=np.float64)


def fill_missing_nodes(values: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill absent nodes from the nearest tabulated logU row in the same column.

    Parameters
    ----------
    values : ndarray, shape (n_logU, n_xid, n_nH, n_CO, n_mup, n_lines)
        Line luminosities for one metallicity. Entries at absent nodes are ignored.
    present : ndarray, shape (n_logU, n_xid, n_nH, n_CO, n_mup), bool
        True where the text table tabulates the node.

    Returns
    -------
    filled : ndarray, same shape as ``values``
        Values with every absent node replaced by its nearest tabulated neighbor.
    filled_mask : ndarray, shape (n_logU, n_xid, n_nH, n_CO, n_mup), bool
        True at each node that was filled.

    Raises
    ------
    ValueError
        If a column has no tabulated logU row at all.

    Notes
    -----
    Ties between two equidistant logU neighbors resolve to the lower logU row,
    because :func:`numpy.argmin` returns the first minimum on the ascending axis.
    """
    filled = np.array(values, dtype=np.float64, copy=True)
    filled_mask = ~present
    for node in zip(*np.nonzero(filled_mask), strict=True):
        iu, rest = node[0], node[1:]
        candidates = np.flatnonzero(present[(slice(None), *rest)])
        if candidates.size == 0:
            raise ValueError(f"No tabulated logU row for column {rest}")
        nearest = candidates[np.argmin(np.abs(candidates - iu))]
        filled[(iu, *rest)] = filled[(nearest, *rest)]
    return filled, filled_mask


def qh_anchor_case_b(line_lum: np.ndarray) -> np.ndarray:
    """Anchor the ionizing photon rate to Case B H-beta at xi_d -> 0.

    Osterbrock & Ferland 2006, Table 4.4 (T = 1e4 K, Case B). The H-beta
    luminosity at logU = -4, n_H = 100, (C/O)/(C/O)_sun = 1 is taken at
    xi_d = 0.1, 0.3, 0.5, fit with a straight line in xi_d, and evaluated at
    xi_d = 0.

    Parameters
    ----------
    line_lum : ndarray, shape (n_Z, n_logU, n_xid, n_nH, n_CO, n_mup, n_lines)
        Filled line luminosities [L_sun / (M_sun/yr)].

    Returns
    -------
    ndarray, shape (n_Z, n_mup)
        log10 of the ionizing photon rate per unit SFR [photons/s / (M_sun/yr)].

    Raises
    ------
    ValueError
        If the extrapolated H-beta luminosity is not positive and finite.

    Notes
    -----
    Pure NumPy. The anchor is a table-internal calibration, not a stellar
    population output, so it does not depend on the SSP library.
    """
    xi = np.array([0.1, 0.3, 0.5])
    hbeta = line_lum[
        :,
        _CASE_B_LOGU_INDEX,
        :,
        _CASE_B_NH_INDEX,
        _CASE_B_CO_INDEX,
        :,
        _HBETA_LINE_INDEX,
    ]  # (n_Z, n_xid, n_mup)
    x_mean = xi.mean()
    slope_num = np.sum(
        (xi - x_mean)[None, :, None] * (hbeta - hbeta.mean(axis=1, keepdims=True)), axis=1
    )
    slope = slope_num / np.sum((xi - x_mean) ** 2)
    hb0 = hbeta.mean(axis=1) - slope * x_mean  # (n_Z, n_mup)
    if not np.all(np.isfinite(hb0) & (hb0 > 0)):
        raise ValueError("Case B H-beta anchor is not positive and finite")
    return np.log10(hb0 * _LSUN_ERG / _HB_PER_QH_ERG)


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qh_per_sfr_bc03(ssp_path: Path, logz_target: np.ndarray) -> np.ndarray:
    """Diagnostic Q_H per unit SFR from the BC03 SSP grid, at the NEOGAL metallicities.

    The 100 Myr constant-SFR photon rate is
    :math:`Q_H = \\int_0^{t_0} Q_H^{\\rm SSP}(t)\\,dt` with
    :math:`t_0 = 10^8` yr, evaluated by a trapezoid on the SSP age nodes at or
    below :math:`t_0` plus a linearly interpolated node at exactly :math:`t_0`.

    Parameters
    ----------
    ssp_path : Path
        DSPS-format SSP grid (``bc03_pdva_stelib_chabrier.h5``).
    logz_target : array_like, shape (n_Z,)
        Absolute log10(Z) at which to report the result.

    Returns
    -------
    ndarray, shape (n_Z,)
        log10(Q_H / SFR) [photons/s / (M_sun/yr)], linear in log Z between the
        SSP metallicity nodes.

    Notes
    -----
    Diagnostic only. The fitted grid uses the table's own Q_H, not this one.
    Each SSP age node is evaluated with :func:`tengri.components.nebular._shared.compute_qh_log10`.
    """
    ssp = load_ssp_data(str(ssp_path))
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    ages = 10.0 ** np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64) * 1e9  # yr
    lgmet = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)  # (n_met, n_age, n_wave)

    below = np.flatnonzero(ages <= _T_QH_AGE_YR * (1.0 + 1e-6))
    last = below[-1]
    on_node = abs(ages[last] - _T_QH_AGE_YR) <= 1e-6 * _T_QH_AGE_YR
    nodes = below if on_node else np.append(below, last + 1)
    ages_int = ages[nodes] if on_node else np.append(ages[below], _T_QH_AGE_YR)

    log_qh_met = np.empty(lgmet.size)
    for im in range(lgmet.size):
        q_nodes = np.array([10.0 ** float(compute_qh_log10(wave, flux[im, ia])) for ia in nodes])
        if not on_node:
            a_lo, a_hi = ages[last], ages[last + 1]
            q_lo = 10.0 ** float(compute_qh_log10(wave, flux[im, last]))
            q_hi = 10.0 ** float(compute_qh_log10(wave, flux[im, last + 1]))
            frac = (_T_QH_AGE_YR - a_lo) / (a_hi - a_lo)
            q_nodes = np.append(q_nodes[:-1], q_lo + frac * (q_hi - q_lo))
        log_qh_met[im] = np.log10(np.trapezoid(q_nodes, ages_int))
    return np.interp(np.asarray(logz_target, dtype=np.float64), lgmet, log_qh_met)


def _load_grid_arrays(
    files: list[Path],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[tuple]]:
    """Assemble the (n_Z, 7, 3, 2, 9, 2, 18) luminosity cube from the text files."""
    first = load_neogal_sf_file(files[0])
    logu_axis = np.unique(first[:, 0])
    xi_axis = np.unique(first[:, 1])
    co_axis = np.unique(first[:, 3])
    mup_axis = np.unique(first[:, 4])
    nh_axis = np.array(_NH_KEEP)
    shape = (logu_axis.size, xi_axis.size, nh_axis.size, co_axis.size, mup_axis.size)

    cube, masks, dropped = [], [], 0
    for path in files:
        table = load_neogal_sf_file(path)
        lum = np.zeros((*shape, _N_LINES))
        present = np.zeros(shape, dtype=bool)
        for row in table:
            if row[2] not in _NH_KEEP:
                dropped += 1
                continue
            idx = (
                _axis_index(logu_axis, row[0]),
                _axis_index(xi_axis, row[1]),
                _axis_index(nh_axis, row[2]),
                _axis_index(co_axis, row[3]),
                _axis_index(mup_axis, row[4]),
            )
            if present[idx]:
                raise ValueError(f"Duplicate node {idx} in {path.name}")
            lum[idx] = row[5:]
            present[idx] = True
        filled, mask = fill_missing_nodes(lum, present)
        cube.append(filled)
        masks.append(mask)
    axes = {
        "logU": logu_axis,
        "xi": xi_axis,
        "nH": nh_axis,
        "CO": co_axis,
        "mup": mup_axis,
    }
    return axes, np.stack(cube), np.stack(masks), dropped


def _axis_index(axis: np.ndarray, value: float) -> int:
    """Index of an exact axis value; raises if the value is not on the axis."""
    index = int(np.argmin(np.abs(axis - value)))
    if abs(axis[index] - value) > 1e-9:
        raise ValueError(f"Value {value} is not on the axis {axis}")
    return index


def build(
    input_dir: Path,
    output_path: Path,
    *,
    ssp_path: Path,
) -> None:
    """Build data/gutkin16_grid.h5 from the NEOGAL text tables.

    Parameters
    ----------
    input_dir : Path
        Directory containing ``nebular_emission_Z*.txt`` files.
    output_path : Path
        Destination HDF5 file.
    ssp_path : Path
        BC03 SSP grid for the diagnostic ``log_qh_per_sfr_bc03`` dataset. If the
        file is absent, that dataset is NaN and ``qh_reference_ssp`` is ``"absent"``.

    Raises
    ------
    SystemExit
        If ``input_dir`` holds no NEOGAL tables.
    ValueError
        If the number of filled nodes differs from the expected 19.
    """
    files = sorted(input_dir.glob("nebular_emission_Z*.txt"), key=z_from_filename)
    if not files:
        print(f"ERROR: No nebular_emission_Z*.txt files found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(files)} metallicity tables in {input_dir}")

    axes, lum_cube, mask_cube, dropped = _load_grid_arrays(files)
    z_values = np.array([z_from_filename(f) for f in files])
    logz_axis = np.log10(z_values)
    n_filled = int(mask_cube.sum())
    if n_filled != _N_EXPECTED_FILLED:
        raise ValueError(f"Expected {_N_EXPECTED_FILLED} filled nodes, got {n_filled}")
    _print_filled(files, mask_cube, axes)
    print(f"Dropped {dropped} rows at n_H in {_NH_DROPPED}")

    mup_duplicate = [
        float(logz_axis[iz])
        for iz in range(len(files))
        if np.array_equal(lum_cube[iz, :, :, :, :, 0, :], lum_cube[iz, :, :, :, :, 1, :])
    ]
    qh_g16 = qh_anchor_case_b(lum_cube)
    log_line = (
        np.log10(np.maximum(lum_cube, 1e-30) * _LSUN_ERG)
        - qh_g16[:, None, None, None, None, :, None]
    )

    ssp_present = ssp_path.exists()
    if ssp_present:
        qh_bc03 = qh_per_sfr_bc03(ssp_path, logz_axis)
        ssp_hash = _sha256(ssp_path)
    else:
        qh_bc03 = np.full(logz_axis.size, np.nan)
        ssp_hash = "absent"

    labels = np.array([line[1] for line in _LINES])
    vacuum = np.array([line[2] for line in _LINES])
    is_air = np.array([line[3] for line in _LINES])
    vacuum[is_air] = air_to_vac(vacuum[is_air])

    attrs = {
        "source": f"{_NEOGAL_TAG}; {_CITATION}",
        "neogal_url": "http://www.iap.fr/neogal/sf-models.html",
        "cloudy_version": "c13.03",
        "lsun_erg": _LSUN_ERG,
        "n_filled": n_filled,
        "dropped_nH": np.array(_NH_DROPPED),
        "mup300_equals_mup100_at_logZ": np.array(mup_duplicate),
        "builder": "scripts/build_gutkin16_grid.py",
        "citation": _CITATION,
        "wavelength_convention": _WAVE_CONVENTION,
        "hb_per_qh_erg": _HB_PER_QH_ERG,
        "qh_anchor_method": "case_b_hbeta_logU-4_nH100_CO1_xi_d_extrapolated_to_0",
        "qh_reference_ssp": "bc03_pdva_stelib_chabrier" if ssp_present else "absent",
        "qh_reference_ssp_sha256": ssp_hash,
    }
    _write_h5(
        output_path,
        attrs,
        {
            "logZ_axis": logz_axis,
            "logU_axis": axes["logU"],
            "xi_d_axis": axes["xi"],
            "log_nH_axis": np.log10(axes["nH"]),
            "co_ratio_axis": axes["CO"],
            "mup_axis": axes["mup"],
            "line_wavelengths_label_aa": labels,
            "line_wavelengths_aa": vacuum,
            "log_line_per_qh": log_line,
            "log_qh_per_sfr_g16": qh_g16,
            "log_qh_per_sfr_bc03": qh_bc03,
            "filled_mask": mask_cube,
        },
    )
    print(f"\nWrote {output_path} ({output_path.stat().st_size / 1e6:.2f} MB)")
    print(f"log_qh_per_sfr_g16 (m_up=100): {np.round(qh_g16[:, 0], 3)}")
    print(f"log_qh_per_sfr_g16 (m_up=300): {np.round(qh_g16[:, 1], 3)}")
    print(f"log_qh_per_sfr_bc03:           {np.round(qh_bc03, 3)}")


def _print_filled(files: list[Path], mask: np.ndarray, axes: dict[str, np.ndarray]) -> None:
    """Print every filled node with its Z, axis values and logU source."""
    print(f"Filled nodes: {int(mask.sum())}")
    for iz, *rest in zip(*np.nonzero(mask), strict=True):
        print(
            f"  Z={files[iz].stem.split('_Z')[1]} logU={axes['logU'][rest[0]]} "
            f"xi={axes['xi'][rest[1]]} nH={axes['nH'][rest[2]]} "
            f"CO={axes['CO'][rest[3]]} mup={axes['mup'][rest[4]]}"
        )


def _write_h5(output_path: Path, attrs: dict, datasets: dict[str, np.ndarray]) -> None:
    """Write the grid group and its provenance attributes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        group = handle.create_group("gutkin16")
        for target in (handle, group):
            for key, value in attrs.items():
                target.attrs[key] = value
        for name, data in datasets.items():
            group.create_dataset(name, data=data, compression="gzip")
        group.create_dataset(
            "line_names",
            data=np.array([line[0] for line in _LINES], dtype=object),
            dtype=str_dtype,
        )


def main() -> None:
    """Parse arguments and build the grid."""
    parser = argparse.ArgumentParser(
        description="Build data/gutkin16_grid.h5 from NEOGAL Gutkin+2016 text tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/neogal"),
        help="Directory containing nebular_emission_Z*.txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gutkin16_grid.h5"),
        help="Output path for gutkin16_grid.h5.",
    )
    parser.add_argument(
        "--ssp",
        type=Path,
        default=Path("data/bc03_pdva_stelib_chabrier.h5"),
        help="BC03 SSP grid for the diagnostic Q_H dataset.",
    )
    args = parser.parse_args()
    build(args.input, args.output, ssp_path=args.ssp)


if __name__ == "__main__":
    main()
