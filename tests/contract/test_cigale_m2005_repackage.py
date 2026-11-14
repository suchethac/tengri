# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the CIGALE Maraston 2005 repackager reproduces pcigale's pickles (issue #352).

The reference is read straight from pcigale's ``data/m2005`` pickles. The layout and
units are those of the BC03 grids: ``spec`` is W/nm per solar mass formed on
(wavelength, age) axes, and ``info[0]`` is the total stellar mass per solar mass formed
(the surviving-mass column the BC03 driver already uses). The M2005 solar metallicity
is Z = 0.02 (Maraston 2005, MNRAS 362, 799, Table 1: Z = 0.02 is labeled [Z/H] = +0.00).

The acceptance check of #352 is run through tengri's own SSP loader: the CIGALE
spectrum at matched (Z, age) nodes differs from the loaded grid by less than 1e-4.
"""

from __future__ import annotations

import importlib
import math
import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.constants

pytestmark = pytest.mark.contract

_RTOL = 1e-10
_RESIDUAL_MAX = 1e-4  # acceptance bound of #352
_L_SUN_ERG_PER_S = 3.828e33
_C_ANGSTROM_PER_S = scipy.constants.c * 1e10
_W_NM_TO_ERG_S_ANGSTROM = 1e6
_M2005_SOLAR_Z = 0.02
_SUBSET_Z = (0.001, 0.02, 0.04)
_NODES = ((0, 0), (1, 99), (2, 13699))  # (metallicity index, age index) into _SUBSET_Z


@pytest.fixture(scope="module")
def driver():
    return importlib.import_module("reproduction.cigale._drivers.cigale_ssp_to_dsps")


@pytest.fixture(scope="module")
def pcigale_m2005():
    pytest.importorskip("pcigale")
    import pcigale.data

    root = Path(pcigale.data.__file__).parent / "m2005"

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


@pytest.mark.parametrize("imf", ["salp", "krou"])
def test_m2005_nodes_match_pcigale_pickle(driver, pcigale_m2005, tmp_path, imf):
    out = tmp_path / f"m2005_{imf}.h5"
    driver.repackage_m2005(out, imf=imf, metallicities=_SUBSET_Z)

    with h5py.File(out, "r") as f:
        flux = f["ssp_flux"][...]
        mass = f["ssp_mass_remaining"][...]
        lg_age = f["ssp_lg_age_gyr"][...]
        wave = f["ssp_wave"][...]
        lgmet = f["ssp_lgmet"][...]
        assert flux.dtype == np.float64

    for i_z, age_idx in _NODES:
        entry = pcigale_m2005(_SUBSET_Z[i_z], imf)
        assert entry.imf == imf
        np.testing.assert_allclose(
            flux[i_z, age_idx, :], _reference_lnu(entry, age_idx), rtol=_RTOL, atol=0.0
        )
        np.testing.assert_allclose(
            mass[i_z, age_idx], entry.info[0, age_idx], rtol=_RTOL, atol=0.0
        )

    entry = pcigale_m2005(_SUBSET_Z[0], imf)
    np.testing.assert_allclose(wave, np.asarray(entry.wl) * 10.0, rtol=_RTOL, atol=0.0)
    np.testing.assert_allclose(
        lg_age, np.log10(np.asarray(entry.t, dtype=np.float64) / 1000.0), rtol=_RTOL
    )
    np.testing.assert_allclose(lgmet, np.log10(np.array(_SUBSET_Z)), rtol=_RTOL)


def test_m2005_solar_metallicity_label():
    """Maraston 2005 Table 1: Z = 0.02 is [Z/H] = +0.00, so Z_sun = 0.02 for M2005."""
    from tengri.parameters.translate import LOG10_ZSUN_BY_LIBRARY

    assert LOG10_ZSUN_BY_LIBRARY["m2005"] == pytest.approx(math.log10(_M2005_SOLAR_Z), abs=1e-4)


def test_m2005_residual_through_tengri_loader(driver, pcigale_m2005, tmp_path):
    """#352 acceptance: CIGALE vs tengri residual < 1e-4 at matched (Z, age)."""
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    out = tmp_path / "m2005_salp.h5"
    driver.repackage_m2005(out, imf="salp", metallicities=_SUBSET_Z)
    ssp = load_ssp_data(str(out))

    worst = 0.0
    for i_z, age_idx in _NODES:
        cigale = _reference_lnu(pcigale_m2005(_SUBSET_Z[i_z], "salp"), age_idx)
        tengri = np.asarray(ssp.ssp_flux[i_z, age_idx, :], dtype=np.float64)
        significant = cigale > 1e-8 * cigale.max()
        resid = np.abs(tengri[significant] - cigale[significant]) / cigale[significant]
        worst = max(worst, float(resid.max()))
    assert worst < _RESIDUAL_MAX
