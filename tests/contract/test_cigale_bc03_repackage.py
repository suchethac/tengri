# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the CIGALE BC03 repackager writes pcigale's spectra in the DSPS layout.

The reference is read straight from pcigale's pickles, not from the driver.
For a spectrum stored in W/nm per solar mass, the driver's Lsun/Hz/Msun value is

    L_nu [Lsun/Hz/Msun] = L_lambda [W/nm/Msun] * 1e6 * lambda[A]^2 / c[A/s] / L_sun[erg/s]

where 1e6 converts W/nm to erg/s/A (1 W = 1e7 erg/s and 1 nm = 10 A). The mass
column is pcigale's ``info[0]``, the surviving stellar mass per solar mass formed
(the same column the Chabrier grid has always used). The Salpeter grid is written
in float64, so the comparison can reach 1e-10. The Chabrier grid is float32 and is
pinned to values measured from the repackager before the Salpeter path was added.
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.constants

pytestmark = pytest.mark.contract

_RTOL_SALPETER = 1e-10
_RTOL_CHABRIER_DEFAULT = 1e-12
# Solar luminosity in erg/s (IAU 2015), the value tengri uses in utils/physics_constants.
_L_SUN_ERG_PER_S = 3.828e33
_C_ANGSTROM_PER_S = scipy.constants.c * 1e10
_W_NM_TO_ERG_S_ANGSTROM = 1e6
_SALPETER_Z = (0.0001, 0.02, 0.05)
_NODES = ((0, 0), (1, 99), (2, 13699))  # (metallicity index, age index) into _SALPETER_Z


@pytest.fixture(scope="module")
def driver():
    return importlib.import_module("reproduction.cigale._drivers.cigale_ssp_to_dsps")


@pytest.fixture(scope="module")
def bc03_root(driver) -> Path:
    """pcigale's BC03 pickle directory; skip where pcigale is installed without its database."""
    pytest.importorskip("pcigale")
    root = driver.bc03_data_dir()
    needed = [
        root / f"Z={z}_imf={imf}.pickle"
        for z in driver.BC03_METALLICITIES
        for imf in ("chab", "salp")
    ]
    missing = [p.name for p in needed if not p.is_file()]
    if missing:
        pytest.skip(f"pcigale BC03 database not built ({len(missing)} pickles absent from {root})")
    return root


@pytest.fixture(scope="module")
def pcigale_bc03(bc03_root):
    root = bc03_root

    def load(z: float, imf: str):
        with open(root / f"Z={z}_imf={imf}.pickle", "rb") as f:
            return pickle.load(f)

    return load


def _reference_lnu(entry, age_idx: int) -> np.ndarray:
    """pcigale spectrum at one age, converted to Lsun/Hz/Msun with lambda^2 / c."""
    wl_angstrom = np.asarray(entry.wl, dtype=np.float64) * 10.0
    spec_w_nm = np.asarray(entry.spec[:, age_idx], dtype=np.float64)
    return (
        spec_w_nm * _W_NM_TO_ERG_S_ANGSTROM * wl_angstrom**2 / _C_ANGSTROM_PER_S / _L_SUN_ERG_PER_S
    )


def test_salpeter_nodes_match_pcigale_pickle(driver, pcigale_bc03, tmp_path):
    out = tmp_path / "bc03_salp.h5"
    driver.repackage_bc03_salpeter(out, metallicities=_SALPETER_Z)

    with h5py.File(out, "r") as f:
        flux = f["ssp_flux"][...]
        mass = f["ssp_mass_remaining"][...]
        lg_age = f["ssp_lg_age_gyr"][...]
        wave = f["ssp_wave"][...]
        lgmet = f["ssp_lgmet"][...]
        assert flux.dtype == np.float64

    for i_z, age_idx in _NODES:
        entry = pcigale_bc03(_SALPETER_Z[i_z], "salp")
        assert entry.imf == "salp"
        np.testing.assert_allclose(
            flux[i_z, age_idx, :], _reference_lnu(entry, age_idx), rtol=_RTOL_SALPETER, atol=0.0
        )
        np.testing.assert_allclose(
            mass[i_z, age_idx], entry.info[0, age_idx], rtol=_RTOL_SALPETER, atol=0.0
        )

    entry = pcigale_bc03(_SALPETER_Z[0], "salp")
    np.testing.assert_allclose(wave, np.asarray(entry.wl) * 10.0, rtol=_RTOL_SALPETER, atol=0.0)
    np.testing.assert_allclose(
        lg_age, np.log10(np.asarray(entry.t, dtype=np.float64) / 1000.0), rtol=_RTOL_SALPETER
    )
    np.testing.assert_allclose(lgmet, np.log10(np.array(_SALPETER_Z)), rtol=_RTOL_SALPETER)


def test_chabrier_default_path_unchanged(driver, bc03_root, tmp_path):
    """Values measured from the Chabrier repackager before the Salpeter path was added."""
    out = tmp_path / "bc03_chab.h5"
    driver.repackage_bc03_chabrier(out)
    pinned = (
        # (met index, age index, wave index, flux, mass_remaining at (met, age))
        (5, 99, 1000, 3.993367746532974e-15, 0.7101234197616577),
        (0, 0, 10, 3.281198984681689e-20, 1.0),
        (3, 13699, 1261, 1.8514441146302033e-19, 0.4954628348350525),
    )
    with h5py.File(out, "r") as f:
        flux = f["ssp_flux"]
        mass = f["ssp_mass_remaining"]
        assert flux.dtype == np.float32
        for i_z, age_idx, wave_idx, flux_ref, mass_ref in pinned:
            np.testing.assert_allclose(
                float(flux[i_z, age_idx, wave_idx]), flux_ref, rtol=_RTOL_CHABRIER_DEFAULT
            )
            np.testing.assert_allclose(
                float(mass[i_z, age_idx]), mass_ref, rtol=_RTOL_CHABRIER_DEFAULT
            )


class _SyntheticEntry:
    """A pcigale-shaped SSP entry with spectra drawn here."""

    def __init__(self, wl_nm: np.ndarray, t_myr: np.ndarray, rng: np.random.Generator):
        self.wl = wl_nm
        self.t = t_myr
        self.spec = rng.uniform(1e-6, 1e-3, size=(wl_nm.size, t_myr.size))
        self.info = np.vstack([rng.uniform(0.5, 1.0, t_myr.size), np.zeros(t_myr.size)])


def test_writer_converts_synthetic_spectra_to_dsps_layout(driver, tmp_path):
    """The HDF5 writer on spectra built here, so it runs without pcigale's database."""
    metallicities = (0.004, 0.02)
    wl_nm = np.array([100.0, 550.0, 2200.0])
    t_myr = np.array([1.0, 100.0, 10000.0])
    rng = np.random.default_rng(3510)
    entries = {z: _SyntheticEntry(wl_nm, t_myr, rng) for z in metallicities}
    data_by_z = {
        z: {"wl": e.wl, "t": e.t, "spec": e.spec, "m_star": e.info[0]} for z, e in entries.items()
    }
    out = tmp_path / "synthetic.h5"
    driver._write_ssp_grid(
        out,
        data_by_z,
        metallicities=metallicities,
        source="synthetic",
        imf_name="salpeter",
        dtype=np.float64,
    )

    with h5py.File(out, "r") as f:
        flux = f["ssp_flux"][...]
        mass = f["ssp_mass_remaining"][...]
        assert flux.shape == (len(metallicities), t_myr.size, wl_nm.size)
        assert flux.dtype == np.float64
        assert f.attrs["imf"] == "salpeter"
        assert f.attrs["flux_units"] == "Lsun/Hz/Msun"
        wave = f["ssp_wave"][...]
        lg_age = f["ssp_lg_age_gyr"][...]
        lgmet = f["ssp_lgmet"][...]

    for i_z, z in enumerate(metallicities):
        for i_age in range(t_myr.size):
            np.testing.assert_allclose(
                flux[i_z, i_age], _reference_lnu(entries[z], i_age), rtol=_RTOL_SALPETER
            )
        np.testing.assert_allclose(mass[i_z], entries[z].info[0], rtol=_RTOL_SALPETER)
    np.testing.assert_allclose(wave, wl_nm * 10.0, rtol=_RTOL_SALPETER)
    np.testing.assert_allclose(lg_age, np.log10(t_myr / 1000.0), rtol=_RTOL_SALPETER)
    np.testing.assert_allclose(lgmet, np.log10(np.array(metallicities)), rtol=_RTOL_SALPETER)
