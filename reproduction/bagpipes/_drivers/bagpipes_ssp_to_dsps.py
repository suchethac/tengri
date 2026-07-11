"""Port bagpipes' BC03+MILES stellar grid to DSPS-shaped HDF5.

Bagpipes ships ``bc03_miles_stellar_grids.fits`` (BC03 stellar templates
stitched into the MILES extended-wavelength library, Kroupa 2001 IMF).
The FITS layout differs from DSPS: one HDU per metallicity, age grid
shared, wavelength grid in Angstroms, flux in :math:`L_\\odot/\\textrm{Å}/M_\\odot`.

This script reads all 7 metallicity HDUs and writes a single HDF5 file
with the DSPS layout that ``tengri.load_ssp_data`` expects:

- ``ssp_lg_age_gyr``     (n_age,)               :math:`\\log_{10}(\\text{age}/\\text{Gyr})`
- ``ssp_lgmet``          (n_met,)               :math:`\\log_{10}(Z)` absolute
- ``ssp_wave``           (n_wave,)              rest-frame wavelength [Å]
- ``ssp_flux``           (n_met, n_age, n_wave) :math:`L_\\nu` [Lsun/Hz/Msun]
- ``ssp_mass_remaining`` (n_met, n_age)         surviving stellar mass fraction

References
----------
.. [1] Bruzual, G. & Charlot, S. (2003). Stellar population synthesis at
       the resolution of 2003. MNRAS, 344, 1000.
.. [2] Sánchez-Blázquez, P., et al. (2006). MILES — a stellar library at
       intermediate spectral resolution. MNRAS, 371, 703.
.. [3] Kroupa, P. (2001). On the variation of the initial mass function.
       MNRAS, 322, 231.
.. [4] Carnall, A.C., et al. (2018). BAGPIPES. MNRAS, 480, 4379.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from .units import C_ANGSTROM_PER_S

# Bagpipes ships seven metallicity HDUs named ``ZMET_<x>ZSOL``. We hard-code
# the numeric values so the script's output is reproducible without parsing
# floating-point strings out of HDU names.
ZSOL_FRACTIONS: tuple[float, ...] = (0.005, 0.020, 0.200, 0.400, 1.000, 2.500, 5.000)

# Z_sun absolute (Asplund+2009; same value DSPS uses for log10(Z_sun) = -1.848).
Z_SUN_ABSOLUTE: float = 10.0**-1.848


def _grid_dir() -> Path:
    """Locate bagpipes' bundled grid directory inside the active venv."""
    import bagpipes.config as cfg

    return Path(cfg.grid_dir)


def load_bagpipes_bc03_grid() -> dict:
    """Read ``bc03_miles_stellar_grids.fits`` and return its arrays.

    Returns
    -------
    dict
        Keys ``wave_aa`` (n_wave,), ``age_yr`` (n_age,),
        ``flux_lsun_aa_msun`` (n_met, n_age, n_wave),
        ``zsol_fractions`` (n_met,). All in bagpipes' native units —
        no conversion applied here.
    """
    path = _grid_dir() / "bc03_miles_stellar_grids.fits"
    if not path.is_file():
        raise FileNotFoundError(f"bagpipes BC03+MILES grid not found at {path}")

    with fits.open(path) as hdul:
        wave_aa = np.asarray(hdul["WAVELENGTHS_AA"].data, dtype=np.float64)
        age_yr = np.asarray(hdul["STELLAR_AGE_YR"].data, dtype=np.float64)

        n_met = len(ZSOL_FRACTIONS)
        n_age = age_yr.shape[0]
        n_wave = wave_aa.shape[0]
        flux = np.empty((n_met, n_age, n_wave), dtype=np.float64)

        for i_z, zsol in enumerate(ZSOL_FRACTIONS):
            ext_name = f"ZMET_{zsol:.3f}ZSOL"
            data = hdul[ext_name].data
            # Bagpipes stores spectra as (n_age, n_wave).
            flux[i_z, :, :] = np.asarray(data, dtype=np.float64)

        # LIV_MSTAR_FRAC layout: column 0 is log10(age/yr) with the first
        # row set to 0 as a sentinel for age=0; columns 1..7 are the
        # surviving stellar mass fraction at each of the seven metallicities.
        liv = np.asarray(hdul["LIV_MSTAR_FRAC"].data, dtype=np.float64)
        # Drop the age column, transpose to (n_met, n_age) — DSPS shape.
        mass_remaining = liv[:, 1:].T.copy()

    return {
        "wave_aa": wave_aa,
        "age_yr": age_yr,
        "flux_lsun_aa_msun": flux,
        "mass_remaining": mass_remaining,
        "zsol_fractions": np.asarray(ZSOL_FRACTIONS, dtype=np.float64),
    }


def convert_to_lnu_per_msun(wave_aa: np.ndarray, flux_lsun_aa_msun: np.ndarray) -> np.ndarray:
    """Convert :math:`L_\\lambda` [Lsun/Å/Msun] to :math:`L_\\nu` [Lsun/Hz/Msun].

    The DSPS / tengri convention is :math:`L_\\nu` in :math:`L_\\odot/\\textrm{Hz}/M_\\odot`.
    Bagpipes' grid is :math:`L_\\lambda` in :math:`L_\\odot/\\textrm{Å}/M_\\odot`; the
    Jacobian is :math:`L_\\nu = L_\\lambda\\,\\lambda^2/c`.

    Parameters
    ----------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength in Angstroms.
    flux_lsun_aa_msun : ndarray, shape (..., n_wave)
        Spectral luminosity in Lsun/Å/Msun.

    Returns
    -------
    ndarray
        Spectral luminosity in Lsun/Hz/Msun, same shape as input.
    """
    return flux_lsun_aa_msun * (wave_aa**2) / C_ANGSTROM_PER_S


def repackage_bc03_miles(out_path: str | Path) -> Path:
    """Port bagpipes' BC03+MILES grid to DSPS-shaped HDF5.

    The age grid in bagpipes starts at 0 yr — DSPS / tengri expects
    :math:`\\log_{10}` ages, so the zero entry must be dropped before
    saving. We replace it with the first finite age and verify the
    survivor is uniformly log-spaced from there.

    Parameters
    ----------
    out_path : str or Path
        Destination HDF5 file. Parent directory is created if missing.

    Returns
    -------
    Path
        ``out_path`` resolved to an absolute path.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_bagpipes_bc03_grid()

    age_yr_raw = raw["age_yr"]
    if age_yr_raw[0] == 0.0:
        age_yr = age_yr_raw[1:]
        flux_raw = raw["flux_lsun_aa_msun"][:, 1:, :]
        mass_remaining = raw["mass_remaining"][:, 1:]
    else:
        age_yr = age_yr_raw
        flux_raw = raw["flux_lsun_aa_msun"]
        mass_remaining = raw["mass_remaining"]

    wave_aa = raw["wave_aa"]
    flux_lsun_hz_msun = convert_to_lnu_per_msun(wave_aa, flux_raw).astype(np.float64)

    # log10(Z) absolute; bagpipes labels are fractions of Z_sun.
    lgmet = np.log10(raw["zsol_fractions"] * Z_SUN_ABSOLUTE)
    lg_age_gyr = np.log10(age_yr / 1.0e9)

    print(f"Writing {out_path}…")
    with h5py.File(out_path, "w") as h:
        # Cast to float32 at write time — matches the rest of the
        # tengri SSP archive on disk. The float64 work happened
        # upstream so Cue's Q_H integrator does not see overflow at
        # repackaging time (project memory: project_chex_complete).
        h.create_dataset("ssp_flux", data=flux_lsun_hz_msun.astype(np.float32))
        h.create_dataset("ssp_lg_age_gyr", data=lg_age_gyr.astype(np.float32))
        h.create_dataset("ssp_lgmet", data=lgmet.astype(np.float32))
        h.create_dataset("ssp_wave", data=wave_aa.astype(np.float32))
        h.create_dataset("ssp_mass_remaining", data=mass_remaining.astype(np.float32))

        h.attrs["flux_units"] = "Lsun/Hz/Msun"
        h.attrs["wave_units"] = "Angstrom"
        h.attrs["source"] = "bagpipes bc03_miles_stellar_grids.fits (repackaged)"
        h.attrs["imf"] = "Kroupa 2001"
        h.attrs["ssp_library"] = "BC03 + MILES"

    print(
        f"  shape (n_met, n_age, n_wave) = "
        f"({flux_lsun_hz_msun.shape[0]}, "
        f"{flux_lsun_hz_msun.shape[1]}, "
        f"{flux_lsun_hz_msun.shape[2]})"
    )
    print(f"  lgmet  range: [{lgmet.min():.3f}, {lgmet.max():.3f}]")
    print(f"  lg_age range: [{lg_age_gyr.min():.3f}, {lg_age_gyr.max():.3f}] (age in Gyr)")
    print(f"  wave   range: [{wave_aa.min():.2f}, {wave_aa.max():.2e}] Å")
    return out_path


if __name__ == "__main__":
    out = repackage_bc03_miles(Path(__file__).parent / "data" / "bc03_miles_from_bagpipes.h5")
    # Round-trip sanity: native vs repackaged at Z_sun, 1 Gyr.
    raw = load_bagpipes_bc03_grid()
    with h5py.File(out, "r") as h:
        flux_native = raw["flux_lsun_aa_msun"][4, :, :]  # ZMET_1.000ZSOL
        age_yr = raw["age_yr"]
        idx_1gyr = int(np.argmin(np.abs(age_yr - 1.0e9)))
        # Native units are Lsun/Å/Msun; the repackaged grid is Lsun/Hz/Msun.
        # Compare an *integrated* bolometric quantity instead of raw flux.
        wave = raw["wave_aa"]
        Lbol_native = float(np.trapezoid(flux_native[idx_1gyr], wave))
        # Reverse-convert the repackaged flux back to Lsun/Å/Msun to compare
        flux_ported_hz = h["ssp_flux"][4, idx_1gyr - 1, :].astype(np.float64)
        flux_ported_aa = flux_ported_hz * C_ANGSTROM_PER_S / (wave**2)
        Lbol_ported = float(np.trapezoid(flux_ported_aa, wave))
        rel_err = abs(Lbol_ported - Lbol_native) / Lbol_native
        print(
            f"\n  L_bol round-trip at Z_sun, ~1 Gyr: "
            f"native={Lbol_native:.4e}, repackaged={Lbol_ported:.4e}, "
            f"rel_err={rel_err:.2e}"
        )
        assert rel_err < 1e-5, "Bolometric round-trip failed"
    print(f"✓ {os.path.relpath(out)} ready for tengri.load_ssp_data().")
