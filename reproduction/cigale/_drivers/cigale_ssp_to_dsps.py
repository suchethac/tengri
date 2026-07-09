"""
Port CIGALE BC03 Chabrier SSP templates to DSPS-shaped HDF5 for reproduction notebook.

CIGALE stores BC03 SSPs in pickle format (W/nm/Msun); this script loads all 6 metallicities,
converts units and shapes to match DSPS convention (Lsun/Hz/Msun, ages in Gyr), and writes
an HDF5 file matching the layout of tengri.load_ssp().

Unit conversion:
  CIGALE spec: [W/nm/Msun]
  1. Convert W/nm → erg/s/Å: multiply by 1e6 (1e7 erg/s/W × 0.1 nm/Å)
  2. Convert erg/s/nm → erg/s/Hz via λ²/c: multiply by λ² [Å²] / c [Å/s]
  3. Normalize to Lsun = 3.828e33 erg/s: divide by Lsun

  Formula: L_nu [Lsun/Hz/Msun] = L_lambda [W/nm/Msun] * 1e6 * lambda²_Angstrom / c_Angstrom_per_s / L_sun_erg_per_s
  where c = 2.998e18 Angstrom/s and L_sun = 3.828e33 erg/s.
"""

import pickle
from pathlib import Path
import numpy as np
import h5py


# Physical constants
C_ANGSTROM_PER_S = 2.998e18  # speed of light in Angstrom/s
L_SUN_ERG_PER_S = 3.828e33   # solar luminosity in erg/s


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
    import sys
    cigale_data_path = Path(sys.prefix) / "lib" / "python3.12" / "site-packages" / "pcigale" / "data" / "bc03"

    filename = cigale_data_path / f"Z={metallicity}_imf={imf}.pickle"
    if not filename.exists():
        raise FileNotFoundError(f"BC03 pickle not found: {filename}")

    with open(filename, "rb") as f:
        ssp = pickle.load(f)

    # ssp.info rows: [m_star surviving, m_gas returned, n_ly ionizing photons s^-1]
    # m_star + m_gas = 1 exactly at every age (mass conservation).
    return {
        'wl': ssp.wl,                 # nm
        'spec': ssp.spec,             # W/nm/Msun, shape (n_wave, n_age)
        't': ssp.t,                   # Myr
        'Z': ssp.Z,
        'm_star': ssp.info[0, :],     # surviving stellar mass fraction (n_age,)
        'm_gas':  ssp.info[1, :],     # mass returned to ISM (n_age,)
        'n_ly':   ssp.info[2, :],     # ionizing photon rate [s^-1 per Msun formed]
        'imf': ssp.imf,
    }


def convert_wavelength_nm_to_angstrom(wl_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength from nm to Angstrom."""
    return wl_nm * 10.0


def convert_age_myr_to_gyr(t_myr: np.ndarray) -> np.ndarray:
    """Convert age from Myr to Gyr."""
    return t_myr / 1000.0


def convert_flux_cigale_to_dsps(spec_cigale_w_nm: np.ndarray, wl_angstrom: np.ndarray) -> np.ndarray:
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
    spec_dsps = spec_cigale_w_nm * 1e6 * (wl_broadcasted ** 2) / C_ANGSTROM_PER_S / L_SUN_ERG_PER_S

    return spec_dsps


def port_bc03_chabrier(out_path: str | Path) -> None:
    """
    Port CIGALE BC03 Chabrier SSPs to DSPS-shaped HDF5.

    Loads all 6 metallicities, converts units and shapes, and writes to HDF5
    matching the layout expected by tengri.load_ssp().

    Parameters
    ----------
    out_path : str or Path
        Output HDF5 file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Metallicities available in CIGALE BC03
    metallicities = [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05]

    # Load all metallicities
    print("Loading BC03 Chabrier templates from CIGALE...")
    data_by_z = {}
    for z in metallicities:
        print(f"  Loading Z={z}...")
        data_by_z[z] = load_bc03_pickle(z, imf="chab")

    # Verify all have same wavelength and age grids
    ref_wl = data_by_z[metallicities[0]]['wl']
    ref_t = data_by_z[metallicities[0]]['t']
    for z in metallicities[1:]:
        assert np.allclose(data_by_z[z]['wl'], ref_wl), f"Wavelength grid mismatch at Z={z}"
        assert np.allclose(data_by_z[z]['t'], ref_t), f"Age grid mismatch at Z={z}"

    # Convert wavelength nm → Å and age Myr → Gyr
    wl_angstrom = convert_wavelength_nm_to_angstrom(ref_wl)
    age_gyr = convert_age_myr_to_gyr(ref_t)
    lg_age_gyr = np.log10(age_gyr)

    n_met = len(metallicities)
    n_age = len(age_gyr)
    n_wave = len(wl_angstrom)

    # Stack spectra: (n_met, n_age, n_wave)
    flux_dsps = np.zeros((n_met, n_age, n_wave), dtype=np.float32)
    # And the surviving-mass fraction per (met, age) — needed for
    # tengri's mass-loss / surviving-mass bookkeeping. CIGALE stores
    # this per-metallicity in ssp.info[0]; ssp.info[0] + ssp.info[1] = 1
    # at every age.
    mass_remaining = np.zeros((n_met, n_age), dtype=np.float32)

    print("Converting units...")
    for i_z, z in enumerate(metallicities):
        # spec shape from CIGALE: (n_wave, n_age)
        spec_cigale = data_by_z[z]['spec']  # W/nm/Msun

        # Convert to Lsun/Hz/Msun
        spec_converted = convert_flux_cigale_to_dsps(spec_cigale, wl_angstrom)

        # Transpose to (n_age, n_wave) and store
        flux_dsps[i_z, :, :] = spec_converted.T.astype(np.float32)
        mass_remaining[i_z, :] = data_by_z[z]['m_star'].astype(np.float32)

    # Compute log10(Z) absolute (not log10(Z/Zsun))
    lgmet = np.log10(np.array(metallicities, dtype=np.float32))

    # Write HDF5
    print(f"Writing to {out_path}...")
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('ssp_flux', data=flux_dsps, dtype=np.float32)
        f.create_dataset('ssp_lg_age_gyr', data=lg_age_gyr.astype(np.float32), dtype=np.float32)
        f.create_dataset('ssp_lgmet', data=lgmet, dtype=np.float32)
        f.create_dataset('ssp_wave', data=wl_angstrom.astype(np.float32), dtype=np.float32)
        # Surviving stellar mass fraction. tengri's SSPData loader looks
        # for this under ssp_mass_remaining (n_met, n_age).
        f.create_dataset('ssp_mass_remaining', data=mass_remaining, dtype=np.float32)

        # Attributes matching tengri convention
        f.attrs['flux_units'] = 'Lsun/Hz/Msun'
        f.attrs['wave_units'] = 'Angstrom'
        f.attrs['n_met'] = n_met
        f.attrs['n_age'] = n_age
        f.attrs['n_wave'] = n_wave
        f.attrs['source'] = 'CIGALE BC03 Chabrier (ported)'

    print(f"✓ Wrote {out_path}")
    print(f"  Shape: ({n_met}, {n_age}, {n_wave}) [met, age, wave]")
    print(f"  Metallicities (log10 Z): {lgmet}")
    print(f"  Age range (Gyr): {age_gyr[0]:.6e} - {age_gyr[-1]:.6e}")
    print(f"  Wave range (Å): {wl_angstrom[0]:.1f} - {wl_angstrom[-1]:.1f}")


if __name__ == "__main__":
    out_path = Path(__file__).parent / "data" / "bc03_from_cigale.h5"
    port_bc03_chabrier(out_path)
