#!/usr/bin/env python3
"""Convert Draine & Li 2007 dust emission templates to tengri HDF5.

Reads the ASCII template files from the DL07spec tarball and builds a
single HDF5 grid for fast interpolation in tengri.

Download source:
    https://www.astro.princeton.edu/~draine/dust/irem4/DL07spec.tgz

Usage:
    python scripts/convert_dl07_templates.py [--input-dir data/dl07_raw]

Output: data/dl07_templates.h5
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import trapezoid

# DL07 dust models and their q_PAH values (percent)
# From Draine & Li 2007 Table 1
DUST_MODELS = {
    "MW3.1_00": 0.47,
    "MW3.1_10": 1.12,
    "MW3.1_20": 1.77,
    "MW3.1_30": 2.50,
    "MW3.1_40": 3.19,
    "MW3.1_50": 3.90,
    "MW3.1_60": 4.58,
    "LMC2_00": 0.75,
    "LMC2_05": 1.49,
    "LMC2_10": 2.37,
    "smc": 0.10,
}

# U_min values present in the grid
# U_min values that actually exist as directories in the DL07 tarball
UMIN_VALUES = [
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.70,
    0.80,
    1.00,
    1.20,
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
    5.00,
    7.00,
    8.00,
    12.0,
    15.0,
    20.0,
    25.0,
]

# U_max values (power-law upper bound)
UMAX_VALUES = ["1e2", "1e3", "1e4", "1e5", "1e6"]


def _umin_to_str(umin: float) -> str:
    """Convert U_min float to the DL07 naming convention.

    0.10 → 'U0.10', 1.00 → 'U1.00', 12.0 → 'U12.0', 25.0 → 'U25.0'
    """
    if umin >= 10:
        return f"U{umin:.1f}"
    return f"U{umin:.2f}"


def read_dl07_template(filepath: str) -> dict:
    """Read a single DL07 template file.

    Returns
    -------
    dict with keys:
        wavelength_um: (n_wave,) in microns
        j_nu: (n_wave,) in Jy cm^2 sr^-1 H^-1
        nu_pnu: (n_wave,) in erg s^-1 H^-1
        umin: float
        umax: float or str
        dust_model: str
        mean_u: float
        power_per_h: float
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Parse header
    umin = umax = beta = mean_u = power_h = None
    data_start = None
    for i, line in enumerate(lines):
        if "Umin , Umax, beta" in line:
            parts = line.split("=")[0].split()
            umin = float(parts[0])
            umax = float(parts[1])
            beta = float(parts[2])
        elif "<U>" in line:
            mean_u = float(line.split("=")[0].strip())
        elif "power/H" in line:
            power_h = float(line.split("=")[0].strip())
        elif line.strip().startswith("(um)"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Could not find data start in {filepath}")

    # Read data: wavelength(um), nu*P_nu, j_nu [, optional band name]
    # The file has broadband photometry first (with band names in col 4),
    # then the continuous spectrum (3 columns only, wavelength descending).
    # We skip the broadband lines and keep only the spectrum.
    wavelengths = []
    j_nu_values = []
    nu_pnu_values = []

    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        # Skip broadband photometry lines (have 4+ columns with band name)
        if (
            len(parts) >= 4
            and not parts[3]
            .replace(".", "")
            .replace("-", "")
            .replace("+", "")
            .replace("E", "")
            .replace("e", "")
            .isdigit()
        ):
            continue
        try:
            wl = float(parts[0])
            nupnu = float(parts[1])
            jnu = float(parts[2])
            wavelengths.append(wl)
            nu_pnu_values.append(nupnu)
            j_nu_values.append(jnu)
        except ValueError:
            continue

    # Sort by wavelength ascending
    wavelengths = np.array(wavelengths)
    j_nu_values = np.array(j_nu_values)
    nu_pnu_values = np.array(nu_pnu_values)
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    j_nu_values = j_nu_values[sort_idx]
    nu_pnu_values = nu_pnu_values[sort_idx]

    return {
        "wavelength_um": wavelengths,
        "j_nu": j_nu_values,
        "nu_pnu": nu_pnu_values,
        "umin": umin,
        "umax": umax,
        "beta": beta,
        "mean_u": mean_u,
        "power_per_h": power_h,
    }


def convert(input_dir: str, output_path: str) -> None:
    """Convert DL07 templates to tengri HDF5 grid.

    The output grid has shape (n_qpah, n_umin, n_umax, n_wave) for j_nu.
    For the standard DL07 usage with gamma, you need:
      - Single-U template: file U{umin}_{umin}_model.txt (umax = umin)
      - Power-law template: file U{umin}_{umax}_model.txt
    Then: j_nu_total = (1-gamma) * j_nu_single + gamma * j_nu_powerlaw
    """
    # Focus on MW models (most commonly used in SED fitting)
    mw_models = {k: v for k, v in DUST_MODELS.items() if k.startswith("MW")}
    qpah_values = sorted(set(mw_models.values()))
    model_by_qpah = {v: k for k, v in mw_models.items()}

    # Read one file to get wavelength grid
    test_file = None
    for d in sorted(os.listdir(input_dir)):
        dpath = os.path.join(input_dir, d)
        if os.path.isdir(dpath):
            for f in os.listdir(dpath):
                if f.endswith(".txt"):
                    test_file = os.path.join(dpath, f)
                    break
        if test_file:
            break

    if test_file is None:
        print(f"Error: no template files found in {input_dir}")
        sys.exit(1)

    test_data = read_dl07_template(test_file)
    wave_um_native = test_data["wavelength_um"]
    n_wave = len(wave_um_native)
    # Regrid onto a clean log-uniform micron axis spanning the native range.
    # The DL07spec continuous-spectrum rows are already log-uniform in
    # wavelength but printed to 4 significant figures, which is not exactly
    # reproducible; a canonical ``logspace`` axis is what tengri's shipped
    # grids actually carry as their ``wavelength`` dataset (verified against
    # the pre-#2535 file: bit-identical to
    # ``np.logspace(log10(wave_um_native[0]), log10(wave_um_native[-1]),
    # n_wave)``), so every node resamples onto the same reproducible grid.
    wave_um = np.logspace(np.log10(wave_um_native[0]), np.log10(wave_um_native[-1]), n_wave)
    print(f"Wavelength grid: {n_wave} points, {wave_um[0]:.3f} - {wave_um[-1]:.3f} um")

    # Build grids for single-U and power-law templates
    n_qpah = len(qpah_values)
    n_umin = len(UMIN_VALUES)

    # Single-U templates: L_lambda(qpah, umin, wave) — for the (1-gamma) component
    single_u = np.zeros((n_qpah, n_umin, n_wave))
    # Power-law templates: L_lambda(qpah, umin, wave) — for the gamma component
    # Using Umax=1e6 (standard DL07)
    powerlaw = np.zeros((n_qpah, n_umin, n_wave))

    c_um = 2.99792458e14  # c in um/s

    def _read_as_l_lambda_per_aa(path: str) -> np.ndarray:
        """Read a DL07 template and convert its ``j_nu`` column to L_lambda.

        The raw ``j_nu`` column is a frequency-space emissivity (Jy cm^2
        sr^-1 H^-1). ``emission_templates.dl07_tabulated`` treats the loaded
        grid as an L_lambda shape and applies its own L_lambda -> L_nu
        Jacobian at runtime, so the written grid must already be in that
        convention: multiply by ``|dnu/dlambda| = nu/lambda`` and resample
        onto the shared ``wave_um`` grid, per Angstrom.
        """
        data = read_dl07_template(path)
        wl_native = data["wavelength_um"]
        nu_native = c_um / wl_native
        l_lambda_per_um = data["j_nu"] * (nu_native / wl_native)
        l_lambda_per_aa = l_lambda_per_um / 1.0e4
        return np.interp(wave_um, wl_native, l_lambda_per_aa)

    found = 0
    missing = 0

    for iq, qpah in enumerate(qpah_values):
        model_name = model_by_qpah[qpah]
        for iu, umin in enumerate(UMIN_VALUES):
            u_str = _umin_to_str(umin)  # e.g., "U1.00" or "U12.0"

            # Single-U file: U{umin}/U{umin}_{umin_val}_{model}.txt
            # Note: second field has no "U" prefix
            u_val = u_str[1:]  # strip leading "U"
            single_fname = f"{u_str}_{u_val}_{model_name}.txt"
            single_path = os.path.join(input_dir, u_str, single_fname)

            if os.path.exists(single_path):
                single_u[iq, iu, :] = _read_as_l_lambda_per_aa(single_path)
                found += 1
            else:
                missing += 1
                if iu == 0:
                    print(f"  Missing single: {single_path}")

            # Power-law file: U{umin}/U{umin}_1e6_{model}.txt
            pl_fname = f"{u_str}_1e6_{model_name}.txt"
            pl_path = os.path.join(input_dir, u_str, pl_fname)

            if os.path.exists(pl_path):
                powerlaw[iq, iu, :] = _read_as_l_lambda_per_aa(pl_path)
                found += 1
            else:
                missing += 1
                if iu == 0:
                    print(f"  Missing powerlaw: {pl_path}")

    print(f"Templates read: {found} found, {missing} missing")

    # Convert j_nu from Jy cm^2 sr^-1 H^-1 to Lsun/Hz per Msun_dust
    # Following CIGALE/FSPS convention:
    # j_nu is emissivity per H atom. To get per unit dust mass:
    # j_nu_per_Mdust = j_nu / (m_H * (M_dust/M_gas)) * 4*pi
    # But for energy-balance normalization, we just need the SED *shape*
    # normalized so integral = 1. The absolute scaling comes from L_absorbed.
    #
    # NOTE: single_u and powerlaw are EACH normalized to unit integral, which
    # discards their *relative* DL07 power. The power-law (PDR) component emits
    # R = U_max ln(U_max/U_min)/(U_max - U_min) times more per unit dust mass
    # (Draine & Li 2007, Eq. 33). That factor is restored at runtime in
    # ``emission_templates.dl07_tabulated`` so that ``gamma`` is applied as a
    # mass fraction, not a luminosity fraction. Do NOT bake R in here as well.

    # Normalize each template to unit WAVELENGTH integral (shape only; forward
    # restores R). The grid is already L_lambda (per Angstrom), so the
    # consuming ``dl07_tabulated`` -- which applies its own L_lambda -> L_nu
    # Jacobian -- must find each template pre-normalized in the SAME variable
    # it was written in: integral over ``wave_aa``, not frequency. Normalizing
    # over frequency here (the pre-#2535 bug) silently hands the consumer a
    # template whose stored shape is off by a wavelength-dependent factor
    # once its Jacobian is applied.
    wave_aa = wave_um * 1.0e4  # Angstrom

    for iq in range(n_qpah):
        for iu in range(n_umin):
            for grid in [single_u, powerlaw]:
                total = trapezoid(grid[iq, iu, :], wave_aa)
                if total > 0:
                    grid[iq, iu, :] /= total

    # Write HDF5
    print(f"Writing: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("wavelength", data=wave_aa)
        f["wavelength"].attrs["units"] = "Angstrom"

        f.create_dataset("qpah_grid", data=np.array(qpah_values))
        f["qpah_grid"].attrs["units"] = "percent"
        f["qpah_grid"].attrs["description"] = "PAH mass fraction (%)"

        f.create_dataset("umin_grid", data=np.array(UMIN_VALUES))
        f["umin_grid"].attrs["description"] = (
            "Minimum radiation field intensity (Mathis ISRF units)"
        )

        f.create_dataset("single_u", data=single_u)
        f["single_u"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        f["single_u"].attrs["description"] = (
            "Single-U template: dust heated by U=U_min only. "
            "L_lambda convention, normalized to unit wavelength integral (shape only)."
        )

        f.create_dataset("powerlaw", data=powerlaw)
        f["powerlaw"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        f["powerlaw"].attrs["description"] = (
            "Power-law template: dust heated by U^{-2} from U_min to 1e6. "
            "L_lambda convention, normalized to unit wavelength integral (shape only)."
        )

        f.attrs["source"] = "Draine & Li 2007, ApJ 657, 810"
        f.attrs["url"] = "https://www.astro.princeton.edu/~draine/dust/irem4/"
        f.attrs["dust_models"] = "MW3.1 (Milky Way R_V=3.1)"
        f.attrs["n_qpah"] = n_qpah
        f.attrs["n_umin"] = n_umin
        f.attrs["n_wave"] = n_wave
        f.attrs["umax_powerlaw"] = 1e6
        f.attrs["description"] = (
            "DL07 IR emission templates for tengri. single_u and powerlaw are each "
            "in L_lambda convention, shape-normalized to unit wavelength integral. "
            "Usage: j_nu = (1-gamma)*single_u[iq,iu] + gamma*R*powerlaw[iq,iu] with "
            "R = U_max*ln(U_max/U_min)/(U_max-U_min) (DL07 Eq. 33 PDR luminosity "
            "weight), then multiply by L_absorbed for energy-balance normalization. "
            "Regenerated 2026-09-29 to correct the U_min axis (true 22-node Draine "
            "& Li 2007 grid ending 25.0, no spurious 10.0 node) and restrict q_PAH "
            "to the 7 genuine MW3.1 nodes (#2535, #2441). Source: "
            "https://www.astro.princeton.edu/~draine/dust/irem4/DL07spec.tgz"
        )

    # Verify the written grids match the published axes
    # This guard prevents a repeat of issue #2535 where the shipped files had
    # wrong axes (10.0 in U_min, SMC/LMC2 in q_PAH)
    with h5py.File(output_path, "r") as f:
        written_umin = np.asarray(f["umin_grid"])
        written_qpah = np.asarray(f["qpah_grid"])

    # U_min: verify it matches the DL07 published 22-node axis exactly
    assert len(written_umin) == 22, f"U_min has {len(written_umin)} nodes, expected 22"
    assert not np.any(written_umin == 10.0), "U_min contains spurious 10.0 node"
    assert 25.0 in written_umin, "U_min missing the top published node 25.0"
    np.testing.assert_array_almost_equal(
        written_umin,
        UMIN_VALUES,
        err_msg=f"Written U_min axis {written_umin} does not match declared {UMIN_VALUES}",
    )

    # q_PAH: verify it contains only MW3.1 nodes, not SMC/LMC2
    assert len(written_qpah) == 7, f"q_PAH has {len(written_qpah)} nodes, expected 7 MW3.1"
    mw_qpah_expected = sorted([v for k, v in DUST_MODELS.items() if k.startswith("MW")])
    np.testing.assert_array_almost_equal(
        written_qpah,
        mw_qpah_expected,
        err_msg=f"Written q_PAH {written_qpah} does not match MW3.1 nodes {mw_qpah_expected}",
    )
    assert 0.10 not in written_qpah, "q_PAH contains SMC (0.10)"
    assert 0.75 not in written_qpah, "q_PAH contains LMC2_00 (0.75)"
    assert 1.49 not in written_qpah, "q_PAH contains LMC2_05 (1.49)"
    assert 2.37 not in written_qpah, "q_PAH contains LMC2_10 (2.37)"

    # Summary
    print("\nDL07 template grid:")
    print(f"  q_PAH: {qpah_values} ({n_qpah} values)")
    print(f"  U_min: {UMIN_VALUES} ({n_umin} values)")
    print(f"  Wavelength: {n_wave} points ({wave_um[0]:.3f} - {wave_um[-1]:.3f} um)")
    print(
        f"  Grid size: {single_u.nbytes / 1e6:.1f} MB (single) + "
        f"{powerlaw.nbytes / 1e6:.1f} MB (power-law)"
    )
    print(f"\nWrote: {output_path}")


def convert_v2(input_dir: str, output_path: str) -> None:
    """Convert DL07 templates to v2 hierarchical HDF5 format.

    The v2 format uses a hierarchical structure:
    - /wavelength: wavelength grid in Angstrom
    - /grid/qpah: q_PAH values (MW3.1 only)
    - /grid/umin: U_min values (published 22-node DL07 grid)
    - /spectra/single_u: single-U templates
    - /spectra/pdr: power-law (PDR) templates
    """
    # Focus on MW models only
    mw_models = {k: v for k, v in DUST_MODELS.items() if k.startswith("MW")}
    qpah_values = sorted(set(mw_models.values()))
    model_by_qpah = {v: k for k, v in mw_models.items()}

    # Read one file to get wavelength grid
    test_file = None
    for d in sorted(os.listdir(input_dir)):
        dpath = os.path.join(input_dir, d)
        if os.path.isdir(dpath):
            for f in os.listdir(dpath):
                if f.endswith(".txt"):
                    test_file = os.path.join(dpath, f)
                    break
        if test_file:
            break

    if test_file is None:
        print(f"Error: no template files found in {input_dir}")
        sys.exit(1)

    test_data = read_dl07_template(test_file)
    wave_um_native = test_data["wavelength_um"]
    n_wave = len(wave_um_native)
    wave_um = np.logspace(np.log10(wave_um_native[0]), np.log10(wave_um_native[-1]), n_wave)
    print(f"Wavelength grid: {n_wave} points, {wave_um[0]:.3f} - {wave_um[-1]:.3f} um")

    # Build grids
    n_qpah = len(qpah_values)
    n_umin = len(UMIN_VALUES)

    single_u = np.zeros((n_qpah, n_umin, n_wave))
    powerlaw = np.zeros((n_qpah, n_umin, n_wave))

    c_um = 2.99792458e14

    def _read_as_l_lambda_per_aa(path: str) -> np.ndarray:
        """See ``convert._read_as_l_lambda_per_aa``: same L_nu -> L_lambda Jacobian."""
        data = read_dl07_template(path)
        wl_native = data["wavelength_um"]
        nu_native = c_um / wl_native
        l_lambda_per_um = data["j_nu"] * (nu_native / wl_native)
        l_lambda_per_aa = l_lambda_per_um / 1.0e4
        return np.interp(wave_um, wl_native, l_lambda_per_aa)

    found = 0
    missing = 0

    for iq, qpah in enumerate(qpah_values):
        model_name = model_by_qpah[qpah]
        for iu, umin in enumerate(UMIN_VALUES):
            u_str = _umin_to_str(umin)
            u_val = u_str[1:]
            single_fname = f"{u_str}_{u_val}_{model_name}.txt"
            single_path = os.path.join(input_dir, u_str, single_fname)

            if os.path.exists(single_path):
                single_u[iq, iu, :] = _read_as_l_lambda_per_aa(single_path)
                found += 1
            else:
                missing += 1

            pl_fname = f"{u_str}_1e6_{model_name}.txt"
            pl_path = os.path.join(input_dir, u_str, pl_fname)

            if os.path.exists(pl_path):
                powerlaw[iq, iu, :] = _read_as_l_lambda_per_aa(pl_path)
                found += 1
            else:
                missing += 1

    print(f"Templates read: {found} found, {missing} missing")

    # Normalize each template to unit WAVELENGTH integral (see ``convert``).
    wave_aa = wave_um * 1e4

    for iq in range(n_qpah):
        for iu in range(n_umin):
            for grid in [single_u, powerlaw]:
                total = trapezoid(grid[iq, iu, :], wave_aa)
                if total > 0:
                    grid[iq, iu, :] /= total

    # Write v2 HDF5
    print(f"Writing: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Top-level wavelength
        f.create_dataset("wavelength", data=wave_aa)
        f["wavelength"].attrs["unit"] = "Angstrom"
        f["wavelength"].attrs["description"] = "Rest-frame wavelength grid"

        # /grid group
        grid_group = f.create_group("grid")
        grid_group.create_dataset("qpah", data=np.array(qpah_values))
        grid_group["qpah"].attrs["unit"] = "percent"
        grid_group["qpah"].attrs["description"] = "PAH mass fraction (0.47-4.58%)"
        grid_group.create_dataset("umin", data=np.array(UMIN_VALUES))
        grid_group["umin"].attrs["unit"] = "dimensionless"
        grid_group["umin"].attrs["description"] = (
            "Minimum radiation field intensity (U_min in units of local ISRF)"
        )

        # /spectra group
        spectra_group = f.create_group("spectra")
        spectra_group.create_dataset("single_u", data=single_u)
        spectra_group["single_u"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        spectra_group["single_u"].attrs["description"] = (
            "Emission at single radiation field U=U_min (diffuse ISM component)"
        )
        spectra_group.create_dataset("pdr", data=powerlaw)
        spectra_group["pdr"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        spectra_group["pdr"].attrs["description"] = (
            "Power-law U distribution from U_min to U_max=1e6 with alpha=2 (PDR component)"
        )

        # /metadata group -- self-documenting provenance, read by nothing at
        # runtime (the loader only reads /wavelength, /grid, /spectra) but
        # kept so the file matches its pre-#2535 structure and is inspectable
        # with h5dump/h5py alone.
        metadata_group = f.create_group("metadata")
        metadata_group.attrs["model_name"] = "Draine & Li 2007"
        metadata_group.attrs["reference"] = "Draine, B. T. & Li, A. 2007, ApJ, 657, 810"
        metadata_group.attrs["wavelength_unit"] = "Angstrom"
        metadata_group.attrs["flux_unit"] = "Lsun_Hz_per_Msun (per solar mass of dust)"
        metadata_group.attrs["created_by"] = "tengri template converter"
        metadata_group.attrs["description"] = (
            "Silicate-graphite-PAH grain model. Two components: single radiation "
            "field (diffuse ISM) and power-law radiation field distribution (PDR "
            "regions). Regenerated 2026-09-29 to correct the U_min axis (true "
            "22-node Draine & Li 2007 grid ending 25.0, no spurious 10.0 node) and "
            "restrict q_PAH to the 7 genuine MW3.1 nodes (#2535, #2441). Source: "
            "https://www.astro.princeton.edu/~draine/dust/irem4/DL07spec.tgz"
        )

    # Verify the written grids
    with h5py.File(output_path, "r") as f:
        written_umin = np.asarray(f["grid"]["umin"])
        written_qpah = np.asarray(f["grid"]["qpah"])

    assert len(written_umin) == 22, f"U_min has {len(written_umin)} nodes, expected 22"
    assert not np.any(written_umin == 10.0), "U_min contains spurious 10.0 node"
    assert 25.0 in written_umin, "U_min missing the top published node 25.0"
    np.testing.assert_array_almost_equal(
        written_umin,
        UMIN_VALUES,
        err_msg=f"Written U_min axis {written_umin} does not match declared {UMIN_VALUES}",
    )

    assert len(written_qpah) == 7, f"q_PAH has {len(written_qpah)} nodes, expected 7 MW3.1"
    mw_qpah_expected = sorted([v for k, v in DUST_MODELS.items() if k.startswith("MW")])
    np.testing.assert_array_almost_equal(
        written_qpah,
        mw_qpah_expected,
        err_msg=f"Written q_PAH {written_qpah} does not match MW3.1 nodes {mw_qpah_expected}",
    )
    assert 0.10 not in written_qpah, "q_PAH contains SMC (0.10)"
    assert 0.75 not in written_qpah, "q_PAH contains LMC2_00 (0.75)"
    assert 1.49 not in written_qpah, "q_PAH contains LMC2_05 (1.49)"
    assert 2.37 not in written_qpah, "q_PAH contains LMC2_10 (2.37)"

    print("\nDL07 template grid (v2):")
    print(f"  q_PAH: {qpah_values} ({n_qpah} values)")
    print(f"  U_min: {UMIN_VALUES} ({n_umin} values)")
    print(f"  Wavelength: {n_wave} points ({wave_um[0]:.3f} - {wave_um[-1]:.3f} um)")
    print(
        f"  Grid size: {single_u.nbytes / 1e6:.1f} MB (single) + "
        f"{powerlaw.nbytes / 1e6:.1f} MB (power-law)"
    )
    print(f"\nWrote: {output_path}")


def relabel_shipped_grid(
    old_v1_path: str, old_v2_path: str, out_v1_path: str, out_v2_path: str
) -> None:
    """Correct the U_min/q_PAH axis labels of a previously-shipped DL07 grid.

    Issue #2535 found that the shipped ``dl07_templates.h5`` /
    ``dl07_templates_v2.h5`` carried a wrong 22-node ``U_min`` axis (a
    spurious ``10.0`` node, no ``25.0``) and #2441 found the ``q_PAH`` axis
    mixed in 4 non-MW3.1 models (SMC, LMC2 x3). Comparing the shipped
    ``single_u``/``powerlaw`` arrays at the four mislabeled top ``U_min``
    slots against a fresh read of the true DL07spec ``U12.0``/``U15.0``/
    ``U20.0``/``U25.0`` directories shows they already hold those exact
    physical templates (median relative difference ~2e-4, the print
    precision of the DL07spec ASCII release, at every one of the 7 MW3.1
    q_PAH nodes and both components) -- i.e. the DATA at those four slots was
    always correct; only the AXIS LABEL ARRAY was off by one, with a
    fabricated ``10.0`` label and a missing ``25.0`` label. This function
    therefore does not re-derive any spectra: it corrects the ``U_min``
    label array in place and drops the 4 non-MW3.1 ``q_PAH`` rows, leaving
    every retained node's numeric array byte-for-byte unchanged. See
    ``tests/components/dust/test_dl07_grid_axes.py`` for the equality proof.

    Parameters
    ----------
    old_v1_path, old_v2_path : str
        Paths to the previously-shipped (wrong-axis) v1/v2 grid files, e.g.
        checked out from an old commit (``git show <sha>:data/dl07_templates.h5``).
    out_v1_path, out_v2_path : str
        Paths to write the corrected v1/v2 grid files to.
    """
    mw_qpah = np.array(sorted(v for k, v in DUST_MODELS.items() if k.startswith("MW")))

    # -- v1 (legacy flat format) --
    with h5py.File(old_v1_path, "r") as f:
        wave = np.asarray(f["wavelength"])
        old_umin = np.asarray(f["umin_grid"])
        old_qpah = np.asarray(f["qpah_grid"])
        old_single_u = np.asarray(f["single_u"])
        old_powerlaw = np.asarray(f["powerlaw"])
        old_attrs = dict(f.attrs)

    assert list(old_umin[:18]) == UMIN_VALUES[:18], "low U_min nodes changed position"
    mw_idx = np.array([int(np.argmin(np.abs(old_qpah - v))) for v in mw_qpah])
    np.testing.assert_array_almost_equal(old_qpah[mw_idx], mw_qpah)

    new_single_u = old_single_u[mw_idx, :, :]
    new_powerlaw = old_powerlaw[mw_idx, :, :]

    os.makedirs(os.path.dirname(out_v1_path) or ".", exist_ok=True)
    with h5py.File(out_v1_path, "w") as f:
        f.create_dataset("wavelength", data=wave)
        f.create_dataset("qpah_grid", data=mw_qpah)
        f.create_dataset("umin_grid", data=np.array(UMIN_VALUES))
        f.create_dataset("single_u", data=new_single_u)
        f["single_u"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        f.create_dataset("powerlaw", data=new_powerlaw)
        f["powerlaw"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        for key in ("model", "reference", "spectra_unit", "wavelength_unit"):
            if key in old_attrs:
                f.attrs[key] = old_attrs[key]
        f.attrs["description"] = (
            "Regenerated 2026-09-29 to correct the U_min axis (true 22-node "
            "Draine & Li 2007 grid ending 25.0, no spurious 10.0 node) and "
            "restrict q_PAH to the 7 genuine MW3.1 nodes; numeric arrays for "
            "every retained node are unchanged from the prior release (#2535, "
            "#2441). Source: "
            "https://www.astro.princeton.edu/~draine/dust/irem4/DL07spec.tgz"
        )

    # -- v2 (hierarchical format) --
    with h5py.File(old_v2_path, "r") as f:
        wave2 = np.asarray(f["wavelength"])
        old_umin2 = np.asarray(f["grid"]["umin"])
        old_qpah2 = np.asarray(f["grid"]["qpah"])
        old_single_u2 = np.asarray(f["spectra"]["single_u"])
        old_pdr2 = np.asarray(f["spectra"]["pdr"])

    assert list(old_umin2[:18]) == UMIN_VALUES[:18], "low U_min nodes changed position"
    mw_idx2 = np.array([int(np.argmin(np.abs(old_qpah2 - v))) for v in mw_qpah])
    np.testing.assert_array_almost_equal(old_qpah2[mw_idx2], mw_qpah)

    new_single_u2 = old_single_u2[mw_idx2, :, :]
    new_pdr2 = old_pdr2[mw_idx2, :, :]

    os.makedirs(os.path.dirname(out_v2_path) or ".", exist_ok=True)
    with h5py.File(out_v2_path, "w") as f:
        f.create_dataset("wavelength", data=wave2)
        f["wavelength"].attrs["unit"] = "Angstrom"
        f["wavelength"].attrs["description"] = "Rest-frame wavelength grid"

        grid_group = f.create_group("grid")
        grid_group.create_dataset("qpah", data=mw_qpah)
        grid_group["qpah"].attrs["unit"] = "percent"
        grid_group["qpah"].attrs["description"] = "PAH mass fraction (0.47-4.58%)"
        grid_group.create_dataset("umin", data=np.array(UMIN_VALUES))
        grid_group["umin"].attrs["unit"] = "dimensionless"
        grid_group["umin"].attrs["description"] = (
            "Minimum radiation field intensity (U_min in units of local ISRF)"
        )

        spectra_group = f.create_group("spectra")
        spectra_group.create_dataset("single_u", data=new_single_u2)
        spectra_group["single_u"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        spectra_group["single_u"].attrs["description"] = (
            "Emission at single radiation field U=U_min (diffuse ISM component)"
        )
        spectra_group.create_dataset("pdr", data=new_pdr2)
        spectra_group["pdr"].attrs["shape"] = "(n_qpah, n_umin, n_wave)"
        spectra_group["pdr"].attrs["description"] = (
            "Power-law U distribution from U_min to U_max=1e6 with alpha=2 (PDR component)"
        )

        metadata_group = f.create_group("metadata")
        metadata_group.attrs["model_name"] = "Draine & Li 2007"
        metadata_group.attrs["reference"] = "Draine, B. T. & Li, A. 2007, ApJ, 657, 810"
        metadata_group.attrs["wavelength_unit"] = "Angstrom"
        metadata_group.attrs["flux_unit"] = "Lsun_Hz_per_Msun (per solar mass of dust)"
        metadata_group.attrs["created_by"] = "tengri template converter"
        metadata_group.attrs["description"] = (
            "Silicate-graphite-PAH grain model. Two components: single radiation "
            "field (diffuse ISM) and power-law radiation field distribution (PDR "
            "regions). Regenerated 2026-09-29 to correct the U_min axis (true "
            "22-node Draine & Li 2007 grid ending 25.0, no spurious 10.0 node) and "
            "restrict q_PAH to the 7 genuine MW3.1 nodes; numeric arrays for every "
            "retained node are unchanged from the prior release (#2535, #2441). "
            "Source: https://www.astro.princeton.edu/~draine/dust/irem4/DL07spec.tgz"
        )

    # Same axis guards as ``convert``/``convert_v2``.
    for path, umin_key, qpah_key in (
        (out_v1_path, ("umin_grid",), ("qpah_grid",)),
        (out_v2_path, ("grid", "umin"), ("grid", "qpah")),
    ):
        with h5py.File(path, "r") as f:
            node = f
            for k in umin_key:
                node = node[k]
            written_umin = np.asarray(node)
            node = f
            for k in qpah_key:
                node = node[k]
            written_qpah = np.asarray(node)
        assert len(written_umin) == 22
        assert not np.any(written_umin == 10.0)
        assert 25.0 in written_umin
        np.testing.assert_array_almost_equal(written_umin, UMIN_VALUES)
        assert len(written_qpah) == 7
        np.testing.assert_array_almost_equal(written_qpah, mw_qpah)

    print(f"Wrote: {out_v1_path}")
    print(f"Wrote: {out_v2_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert DL07 dust templates to tengri HDF5")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with DL07 template folders (default: data/dl07_raw/)",
    )
    parser.add_argument(
        "--output",
        default="data/dl07_templates.h5",
        help="Output HDF5 file (default: data/dl07_templates.h5)",
    )
    parser.add_argument(
        "--relabel-from",
        nargs=2,
        metavar=("OLD_V1", "OLD_V2"),
        default=None,
        help=(
            "Skip the ASCII-based conversion and instead correct the axis "
            "labels of a previously-shipped pair of grid files (see "
            "relabel_shipped_grid()); OLD_V1/OLD_V2 are paths to the old "
            "dl07_templates.h5 / dl07_templates_v2.h5."
        ),
    )
    args = parser.parse_args()

    if args.relabel_from is not None:
        old_v1, old_v2 = args.relabel_from
        output_v2 = args.output.replace(".h5", "_v2.h5")
        relabel_shipped_grid(old_v1, old_v2, args.output, output_v2)
        return

    if args.input_dir is None:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        args.input_dir = str(repo_root / "data" / "dl07_raw")

    # Generate v1 format (top-level keys)
    convert(args.input_dir, args.output)

    # Also generate v2 format (hierarchical structure)
    output_v2 = args.output.replace(".h5", "_v2.h5")
    convert_v2(args.input_dir, output_v2)


if __name__ == "__main__":
    main()
