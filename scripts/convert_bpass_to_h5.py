#!/usr/bin/env python3
"""Rebuild the ``bpss_stars_c3k_a_chabrier`` SSP grid from ProSpect's BPASS speclib.

The shipped grid ``data/bpss_stars_c3k_a_chabrier.h5`` is a repackaging of ProSpect's
``BPASS_St_Ch300.fits`` (Robotham et al. 2020, MNRAS 495, 905): BPASS v2.2.1 binary
stellar populations (``bin``), IMF ``imf_chab300`` (Chabrier, 0.1-300 Msun), 13
metallicities (Z = 1e-5 ... 0.04), 51 ages (log10 yr = 6.0 ... 11.0), 1221 wavelengths
(91 A ... 1.6e6 A). It is not built from the ``spectra-bin-imf135_300`` files (those are
a Salpeter-slope IMF with a different spectral grid) and it has 13 metallicities, not 11.

ProSpect stores each metallicity as extension ``Zspec<i>``, shape (n_wave, n_age), in
Lsun/Angstrom per Msun formed (BPASS's per-1e6-Msun values scaled by 1e-6). The grid
holds the same numbers as Lsun/Hz/Msun:

    L_nu = L_lambda * lambda**2 / c,        c = 2.998e18 A/s,

with ProSpect's rounded ``c`` (the exact 2.99792458e18 differs by the constant factor
1.0000251 and is not what the shipped file uses). Ages are ``log10(Age/yr) - 9``
[log10 Gyr]; metallicities are ``log10(Z)`` with Z the absolute metal mass fraction;
all stored as float32.

Usage::

    python scripts/convert_bpass_to_h5.py --input BPASS_St_Ch300.fits --output out.h5
    python scripts/convert_bpass_to_h5.py --input BPASS_St_Ch300.fits \\
        --verify data/bpss_stars_c3k_a_chabrier.h5

The surviving mass of this grid is not read from the FITS ``Zevo<i>`` tables (their
``SMstar``/``SMrem`` are BPASS's ``starmass`` columns 2 and 3, living stars and an
untested remnant column); see ``src/tengri/data/ssp_mass_remaining/PROVENANCE.md``.

References
----------
.. [1] Robotham, A. S. G., Bellstedt, S., Lagos, C. del P., et al. 2020, MNRAS, 495, 905,
       "ProSpect: Generating spectral energy distributions with complex star formation
       and metallicity histories", doi:10.1093/mnras/staa1116.
.. [2] Eldridge, J. J. et al. 2017, PASA, 34, e058, doi:10.1017/pasa.2017.51.
.. [3] Stanway, E. R. & Eldridge, J. J. 2018, MNRAS, 479, 75, doi:10.1093/mnras/sty1353.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

#: Speed of light as ProSpect writes it [A/s]; the shipped grid is built with it.
C_PROSPECT_A_PER_S = 2.998e18
N_MET = 13
N_AGE = 51
N_WAVE = 1221
#: Verification tolerance on ``ssp_flux`` relative to the rebuild, elementwise, for
#: values above ``FLUX_FLOOR``; float32 round-off is 6e-8.
VERIFY_RTOL = 5e-7
#: Axis nodes of the rebuild and of the shipped grid agree to one float32 ulp at |x| ~ 3
#: (1.19e-7): one metallicity and five age nodes round differently in the original build.
AXES_ATOL = 1.5e-7
#: Below this the grid stores float32 values at or near underflow, and are not compared.
FLUX_FLOOR = 1e-30


def read_prospect_bpass(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read ProSpect's BPASS speclib.

    Parameters
    ----------
    path : Path
        ``BPASS_St_Ch300.fits``.

    Returns
    -------
    wave : ndarray, shape (n_wave,)
        Wavelength [Angstrom].
    z_abs : ndarray, shape (n_met,)
        Absolute metal mass fraction [dimensionless].
    age_yr : ndarray, shape (n_age,)
        Age [yr].
    l_lambda : ndarray, shape (n_met, n_age, n_wave)
        Luminosity density per Msun formed [Lsun/Angstrom/Msun].
    """
    with fits.open(path, memmap=True) as hdul:
        wave = np.array(hdul["Wave"].data, dtype=np.float64).ravel()
        z_abs = np.array(hdul["Z"].data, dtype=np.float64).ravel()
        age_yr = np.array(hdul["Age"].data, dtype=np.float64).ravel()
        spec = np.stack(
            [np.array(hdul[f"Zspec{i}"].data, dtype=np.float64) for i in range(1, z_abs.size + 1)]
        )
    if (z_abs.size, age_yr.size, wave.size) != (N_MET, N_AGE, N_WAVE):
        raise ValueError(f"{path}: unexpected axes {(z_abs.size, age_yr.size, wave.size)}")
    if spec.shape != (N_MET, N_WAVE, N_AGE):
        raise ValueError(f"{path}: unexpected Zspec shape {spec.shape}")
    return wave, z_abs, age_yr, np.transpose(spec, (0, 2, 1))


def to_grid_arrays(
    wave: np.ndarray, z_abs: np.ndarray, age_yr: np.ndarray, l_lambda: np.ndarray
) -> dict[str, np.ndarray]:
    """Convert ProSpect arrays to the grid's datasets.

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Wavelength [Angstrom].
    z_abs : ndarray, shape (n_met,)
        Absolute metal mass fraction [dimensionless].
    age_yr : ndarray, shape (n_age,)
        Age [yr].
    l_lambda : ndarray, shape (n_met, n_age, n_wave)
        L_lambda [Lsun/Angstrom/Msun].

    Returns
    -------
    dict
        ``ssp_flux`` (n_met, n_age, n_wave) [Lsun/Hz/Msun], ``ssp_wave`` [Angstrom],
        ``ssp_lg_age_gyr`` [log10 Gyr], ``ssp_lgmet`` [log10 absolute Z]; all float32.
    """
    l_nu = l_lambda * (wave**2 / C_PROSPECT_A_PER_S)[None, None, :]
    return {
        "ssp_flux": l_nu.astype(np.float32),
        "ssp_wave": wave.astype(np.float32),
        "ssp_lg_age_gyr": (np.log10(age_yr) - 9.0).astype(np.float32),
        "ssp_lgmet": np.log10(z_abs).astype(np.float32),
    }


def write_grid(arrays: dict[str, np.ndarray], path: Path) -> None:
    """Write the grid file (datasets and attributes of the shipped grid)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=value)
        f.attrs["flux_units"] = "Lsun/Hz/Msun"
        f.attrs["wave_units"] = "Angstrom"
        f.attrs["n_met"] = N_MET
        f.attrs["n_ages"] = N_AGE
        f.attrs["n_wave"] = N_WAVE


def verify(arrays: dict[str, np.ndarray], grid: Path) -> dict[str, float]:
    """Compare a rebuild with an existing grid; raises ``AssertionError`` on a mismatch.

    Returns
    -------
    dict
        ``axes_max_abs_diff``, ``flux_median_rel``, ``flux_max_rel`` (above
        ``FLUX_FLOOR``) and ``n_compared``.
    """
    with h5py.File(grid, "r") as f:
        ref = {k: np.asarray(f[k][:]) for k in arrays}
    axes = max(
        float(np.max(np.abs(arrays[k].astype(np.float64) - ref[k].astype(np.float64))))
        for k in ("ssp_wave", "ssp_lg_age_gyr", "ssp_lgmet")
    )
    new = arrays["ssp_flux"].astype(np.float64)
    old = ref["ssp_flux"].astype(np.float64)
    mask = (new > FLUX_FLOOR) & (old > FLUX_FLOOR)
    rel = np.abs(old[mask] / new[mask] - 1.0)
    out = {
        "axes_max_abs_diff": axes,
        "flux_median_rel": float(np.median(rel)),
        "flux_max_rel": float(rel.max()),
        "n_compared": int(mask.sum()),
    }
    assert axes <= AXES_ATOL, f"axes differ: {axes}"
    assert out["flux_max_rel"] < VERIFY_RTOL, f"flux differs: {out}"
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="BPASS_St_Ch300.fits")
    parser.add_argument("--output", type=Path, help="write the rebuilt grid here")
    parser.add_argument("--verify", type=Path, help="compare the rebuild with this grid")
    args = parser.parse_args()
    if args.output is None and args.verify is None:
        parser.error("give --output and/or --verify")
    arrays = to_grid_arrays(*read_prospect_bpass(args.input))
    print(f"input sha256 {_sha256(args.input)}")
    if args.output is not None:
        write_grid(arrays, args.output)
        print(f"wrote {args.output}")
    if args.verify is not None:
        print("verify", verify(arrays, args.verify))
    return 0


if __name__ == "__main__":
    sys.exit(main())
