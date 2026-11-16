"""
Repackage CIGALE BC03 Chabrier SSP templates as DSPS-shaped HDF5 for the notebook.

CIGALE stores BC03 SSPs in pickle format (W/nm/Msun); this script loads all 6 metallicities,
converts units and shapes to match DSPS convention (Lsun/Hz/Msun, ages in Gyr), and writes
an HDF5 file matching the layout of tengri.load_ssp().

Unit conversion:
  CIGALE spec: [W/nm/Msun]
  1. Convert W/nm → erg/s/Å: multiply by 1e6 (1e7 erg/s/W × 0.1 nm/Å)
  2. Convert erg/s/nm → erg/s/Hz via λ²/c: multiply by λ² [Å²] / c [Å/s]
  3. Normalize to Lsun = 3.828e33 erg/s: divide by Lsun

  Formula: L_nu [Lsun/Hz/Msun] = L_lambda [W/nm/Msun] * 1e6 * lambda²_Angstrom
           / c_Angstrom_per_s / L_sun_erg_per_s
  where c and L_sun are tengri's ``C_AA`` and ``L_SUN``.

Two IMFs are repackaged: Chabrier (``repackage_bc03_chabrier``, float32) and
Salpeter (``repackage_bc03_salpeter``, float64). Run as a module with ``--imf
chab`` (default) or ``--imf salp`` to write the grid under ``_drivers/data/``.

The constants come from tengri rather than being written out here: the
notebook reads this grid back with tengri's own c, so a rounded c on the write
side would put a floor of |Δc|/c under §1's SSP residual for every wavelength
and every age.
"""

import pickle
from pathlib import Path

import h5py
import numpy as np

from tengri.utils.physics_constants import C_AA, L_SUN

# Physical constants
C_ANGSTROM_PER_S = C_AA  # speed of light in Angstrom/s
L_SUN_ERG_PER_S = L_SUN  # solar luminosity in erg/s


def bc03_data_dir() -> Path:
    """Directory holding the installed pcigale's BC03 pickles."""
    import pcigale.data

    return Path(pcigale.data.__file__).parent / "bc03"


def load_bc03_pickle(metallicity: float, imf: str = "chab") -> dict:
    """Load a BC03 pickle file from CIGALE's data directory.

    Parameters
    ----------
    metallicity : float
        Metallicity value (one of 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05).
    imf : str, optional
        IMF identifier; must be "chab" (Chabrier) or "salp" (Salpeter). Default: "chab".

    Returns
    -------
    dict
        Dictionary with keys: 'wl' (wavelength in nm), 'spec' (flux in W/nm/Msun),
        't' (age in Myr), 'Z' (metallicity), 'imf'.
    """
    filename = bc03_data_dir() / f"Z={metallicity}_imf={imf}.pickle"
    if not filename.exists():
        raise FileNotFoundError(f"BC03 pickle not found: {filename}")

    with open(filename, "rb") as f:
        ssp = pickle.load(f)

    # ssp.info rows: [m_star surviving, m_gas returned, n_ly ionizing photons s^-1]
    # m_star + m_gas = 1 exactly at every age (mass conservation).
    return {
        "wl": ssp.wl,  # nm
        "spec": ssp.spec,  # W/nm/Msun, shape (n_wave, n_age)
        "t": ssp.t,  # Myr
        "Z": ssp.Z,
        "m_star": ssp.info[0, :],  # surviving stellar mass fraction (n_age,)
        "m_gas": ssp.info[1, :],  # mass returned to ISM (n_age,)
        "n_ly": ssp.info[2, :],  # ionizing photon rate [s^-1 per Msun formed]
        "imf": ssp.imf,
    }


def convert_wavelength_nm_to_angstrom(wl_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength from nm to Angstrom."""
    return wl_nm * 10.0


def convert_age_myr_to_gyr(t_myr: np.ndarray) -> np.ndarray:
    """Convert age from Myr to Gyr."""
    return t_myr / 1000.0


def convert_flux_cigale_to_dsps(
    spec_cigale_w_nm: np.ndarray, wl_angstrom: np.ndarray
) -> np.ndarray:
    """
    Convert flux from CIGALE (W/nm/Msun) to DSPS (Lsun/Hz/Msun).

    Parameters
    ----------
    spec_cigale_w_nm : ndarray
        Flux in W/nm/Msun, shape (n_wave,) or (n_wave, n_age).
    wl_angstrom : ndarray
        Wavelength in Angstrom, shape (n_wave,).

    Returns
    -------
    ndarray
        Flux in Lsun/Hz/Msun, same shape as input spec_cigale_w_nm.
    """
    # Broadcast wavelength to match spec shape if needed
    shape_match = list(spec_cigale_w_nm.shape)
    shape_match[0] = len(wl_angstrom)
    wl_broadcasted = np.broadcast_to(wl_angstrom[:, np.newaxis], shape_match)

    # Unit conversion: W/nm/Msun → Lsun/Hz/Msun.
    #   spec [W/nm]          × 1e7 erg/s per W      → erg/s/nm
    #                        × 0.1 nm/Å             → erg/s/Å
    #   L_λ [erg/s/Å] × λ²[Å²] / c[Å/s]            → erg/s/Hz
    #                        / L_sun[erg/s]         → Lsun/Hz
    spec_dsps = spec_cigale_w_nm * 1e6 * (wl_broadcasted**2) / C_ANGSTROM_PER_S / L_SUN_ERG_PER_S

    return spec_dsps


BC03_METALLICITIES = (0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05)


def _write_ssp_grid(
    out_path: Path,
    data_by_z: dict,
    *,
    metallicities: tuple[float, ...],
    source: str,
    imf_name: str | None,
    dtype: type,
) -> None:
    """Convert loaded pcigale SSPs to Lsun/Hz/Msun and write the DSPS-layout HDF5 file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref_wl = data_by_z[metallicities[0]]["wl"]
    ref_t = data_by_z[metallicities[0]]["t"]
    for z in metallicities[1:]:
        assert np.allclose(data_by_z[z]["wl"], ref_wl), f"Wavelength grid mismatch at Z={z}"
        assert np.allclose(data_by_z[z]["t"], ref_t), f"Age grid mismatch at Z={z}"

    wl_angstrom = convert_wavelength_nm_to_angstrom(ref_wl)
    age_gyr = convert_age_myr_to_gyr(ref_t)
    lg_age_gyr = np.log10(age_gyr)

    n_met = len(metallicities)
    n_age = len(age_gyr)
    n_wave = len(wl_angstrom)

    flux_dsps = np.zeros((n_met, n_age, n_wave), dtype=dtype)
    # Surviving stellar mass fraction per (met, age), pcigale info[0].
    mass_remaining = np.zeros((n_met, n_age), dtype=dtype)

    print("Converting units...")
    for i_z, z in enumerate(metallicities):
        spec_cigale = data_by_z[z]["spec"]  # W/nm/Msun, (n_wave, n_age)
        spec_converted = convert_flux_cigale_to_dsps(spec_cigale, wl_angstrom)
        flux_dsps[i_z, :, :] = spec_converted.T.astype(dtype)
        mass_remaining[i_z, :] = data_by_z[z]["m_star"].astype(dtype)

    lgmet = np.log10(np.array(metallicities, dtype=np.float64))

    print(f"Writing to {out_path}...")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("ssp_flux", data=flux_dsps, dtype=dtype)
        f.create_dataset("ssp_lg_age_gyr", data=lg_age_gyr.astype(dtype), dtype=dtype)
        f.create_dataset("ssp_lgmet", data=lgmet.astype(dtype), dtype=dtype)
        f.create_dataset("ssp_wave", data=wl_angstrom.astype(dtype), dtype=dtype)
        # tengri's SSPData loader reads this under ssp_mass_remaining (n_met, n_age).
        f.create_dataset("ssp_mass_remaining", data=mass_remaining, dtype=dtype)

        f.attrs["flux_units"] = "Lsun/Hz/Msun"
        f.attrs["wave_units"] = "Angstrom"
        f.attrs["n_met"] = n_met
        f.attrs["n_age"] = n_age
        f.attrs["n_wave"] = n_wave
        f.attrs["source"] = source
        if imf_name is not None:
            f.attrs["imf"] = imf_name

    print(f"Wrote {out_path}")
    print(f"  Shape: ({n_met}, {n_age}, {n_wave}) [met, age, wave]")
    print(f"  Metallicities (log10 Z): {lgmet}")
    print(f"  Age range (Gyr): {age_gyr[0]:.6e} - {age_gyr[-1]:.6e}")
    print(f"  Wave range (Å): {wl_angstrom[0]:.1f} - {wl_angstrom[-1]:.1f}")


def repackage_bc03(
    out_path: str | Path,
    *,
    imf: str,
    source_label: str,
    imf_name: str | None,
    metallicities: tuple[float, ...] = BC03_METALLICITIES,
    dtype: type = np.float32,
) -> None:
    """
    Repackage CIGALE BC03 SSPs for one IMF as DSPS-shaped HDF5.

    Parameters
    ----------
    out_path : str or Path
        Output HDF5 file path.
    imf : str
        CIGALE IMF tag used in the pickle file name: ``"chab"`` or ``"salp"``.
    source_label : str
        IMF name written into the ``source`` attribute, e.g. ``"Chabrier"``.
    imf_name : str or None
        IMF label written as the ``imf`` HDF5 attribute. ``None`` writes no attribute.
    metallicities : tuple of float, optional
        Metallicities to write, a subset of ``BC03_METALLICITIES``. Default: all six.
    dtype : type, optional
        Storage type of every dataset. Default ``np.float32``.
    """
    print(f"Loading BC03 {source_label} templates from CIGALE...")
    data_by_z = {}
    for z in metallicities:
        print(f"  Loading Z={z}...")
        data_by_z[z] = load_bc03_pickle(z, imf=imf)
    _write_ssp_grid(
        Path(out_path),
        data_by_z,
        metallicities=metallicities,
        source=f"CIGALE BC03 {source_label} (repackaged)",
        imf_name=imf_name,
        dtype=dtype,
    )


def repackage_bc03_chabrier(out_path: str | Path) -> None:
    """
    Repackage CIGALE BC03 Chabrier SSPs as DSPS-shaped HDF5 (float32, all six Z).

    Parameters
    ----------
    out_path : str or Path
        Output HDF5 file path.
    """
    repackage_bc03(
        out_path,
        imf="chab",
        source_label="Chabrier",
        imf_name=None,
        metallicities=BC03_METALLICITIES,
        dtype=np.float32,
    )


def repackage_bc03_salpeter(
    out_path: str | Path, *, metallicities: tuple[float, ...] = BC03_METALLICITIES
) -> None:
    """
    Repackage CIGALE BC03 Salpeter SSPs as DSPS-shaped HDF5 (float64).

    The grid is stored in float64 so the repackaged values reproduce pcigale's pickles
    to round-off. The ``imf="salpeter"`` attribute labels the grid for
    ``tengri.load_ssp_data``.

    Parameters
    ----------
    out_path : str or Path
        Output HDF5 file path.
    metallicities : tuple of float, optional
        Subset of ``BC03_METALLICITIES`` to write. Default: all six.
    """
    repackage_bc03(
        out_path,
        imf="salp",
        source_label="Salpeter",
        imf_name="salpeter",
        metallicities=metallicities,
        dtype=np.float64,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Repackage CIGALE BC03 SSPs for tengri.")
    parser.add_argument(
        "--imf",
        choices=("chab", "salp"),
        default="chab",
        help="IMF to repackage: chab (Chabrier, float32) or salp (Salpeter, float64).",
    )
    args = parser.parse_args()
    data_dir = Path(__file__).parent / "data"
    if args.imf == "chab":
        repackage_bc03_chabrier(data_dir / "bc03_from_cigale.h5")
    else:
        repackage_bc03_salpeter(data_dir / "bc03_salpeter_from_cigale.h5")
