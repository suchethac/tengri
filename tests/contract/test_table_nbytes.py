# SPDX-License-Identifier: BSD-3-Clause
"""``nbytes`` on each table type that #3560 gave a native-slope sibling (#3541, #3505).

The byte budget for precomputed tables reads ``nbytes``: the resident bytes of the stored
arrays only, the slope sibling included, a nested table counted once through its own
property. Each check builds the table from a small synthetic SSP through the real
precompute builders and sums the array fields independently here, so the property cannot
agree with itself by construction.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.energy_balance_precompute import (
    build_energy_balance_lut,
    build_energy_balance_lut_with_slopes,
)
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.components.stellar.sps.met_pchip import native_slope_flux
from tengri.components.stellar.sps.precompute import (
    precompute_photometry,
    precompute_photometry_ztable,
    precompute_restband_photometry,
    precompute_spectroscopy,
)
from tengri.utils.filter_convention import FilterConvention

pytestmark = pytest.mark.contract

#: Luminosity distance [cm] and redshift the photometric and spectroscopic builds use.
DL_CM = 1.0e26
REDSHIFT = 0.1


@pytest.fixture(autouse=True)
def _no_disk_cache(monkeypatch):
    """The z-table build is disk-cached; keep the test out of the user cache."""
    monkeypatch.setenv("TENGRI_DISABLE_PRECOMP_CACHE", "1")


@pytest.fixture(scope="module")
def ssp_data():
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=jnp.abs(_rng_cube((3, 20, 100))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-2.0, -1.0, 0.0]),
    )


#: The band convention the model uses by default (PhotometryConfig).
_CONV = FilterConvention.BESSELL


def _rng_cube(shape):
    return jnp.asarray(np.random.default_rng(3541).standard_normal(shape))


def _slope_ssp(ssp):
    """The SSP whose flux is the native PCHIP slopes across Z, as the model builds it."""
    return ssp._replace(ssp_flux=native_slope_flux(ssp.ssp_flux, ssp.ssp_lgmet))


def _independent_nbytes(table) -> int:
    """Sum of ``.nbytes`` over the array fields, recursing once into a nested sibling.

    Written from the NamedTuple fields directly, not through the property under test.
    """
    total = 0
    for name, value in table._asdict().items():
        if name == "slopes":
            if value is not None:
                total += _independent_nbytes(value)
        elif isinstance(value, (np.ndarray, jnp.ndarray)):
            total += int(value.nbytes)
    return total


def _filters():
    """Three synthetic top-hat bands, as per-band lists the precompute builders take."""
    waves = [
        jnp.linspace(3500.0, 4500.0, 30),
        jnp.linspace(5000.0, 6500.0, 30),
        jnp.linspace(7500.0, 9000.0, 30),
    ]
    trans = [jnp.ones(30) * 0.5 for _ in range(3)]
    return waves, trans


def _check(table, *, expect_sibling: bool):
    nbytes = table.nbytes
    assert isinstance(nbytes, int) and not isinstance(nbytes, bool)
    assert nbytes > 0
    assert nbytes == _independent_nbytes(table)
    if expect_sibling:
        assert table.slopes is not None
        without = table._replace(slopes=None)
        assert nbytes == without.nbytes + table.slopes.nbytes
    else:
        assert table.slopes is None
    with pytest.raises(AttributeError):
        table.nbytes = 0


def test_photometric_precomputation_counts_sibling(ssp_data):
    waves, trans = _filters()
    lut = precompute_photometry(ssp_data, waves, trans, REDSHIFT, DL_CM, convention=_CONV)
    lut = lut._replace(
        slopes=precompute_photometry(
            _slope_ssp(ssp_data), waves, trans, REDSHIFT, DL_CM, convention=_CONV
        )
    )
    _check(lut, expect_sibling=True)
    # Same filters, same shapes: the sibling carries exactly the values' bytes.
    assert lut.slopes.nbytes == lut._replace(slopes=None).nbytes
    assert lut.nbytes == 2 * lut._replace(slopes=None).nbytes


def test_photometric_precomputation_without_sibling(ssp_data):
    waves, trans = _filters()
    lut = precompute_photometry(ssp_data, waves, trans, REDSHIFT, DL_CM, convention=_CONV)
    _check(lut, expect_sibling=False)


def test_spectroscopic_precomputation_counts_sibling(ssp_data):
    wave_obs = np.linspace(5200.0, 7400.0, 200)
    lut = precompute_spectroscopy(ssp_data, wave_obs, REDSHIFT, DL_CM)
    slope = precompute_spectroscopy(_slope_ssp(ssp_data), wave_obs, REDSHIFT, DL_CM)
    lut = lut._replace(ssp_on_pixels_slope=slope.ssp_on_pixels)
    nbytes = lut.nbytes
    assert isinstance(nbytes, int) and nbytes > 0
    assert nbytes == _independent_nbytes(lut)
    # The sibling has the values' shape, so it adds exactly the values' bytes.
    assert nbytes == 2 * lut.ssp_on_pixels.nbytes + lut.wave_rest_pixels.nbytes + (
        lut.wave_obs_pixels.nbytes
    )
    with pytest.raises(AttributeError):
        lut.nbytes = 0


def test_spectroscopic_precomputation_without_sibling(ssp_data):
    wave_obs = np.linspace(5200.0, 7400.0, 200)
    lut = precompute_spectroscopy(ssp_data, wave_obs, REDSHIFT, DL_CM)
    assert lut.ssp_on_pixels_slope is None
    assert lut.nbytes == _independent_nbytes(lut) > 0


def test_restband_precomputation_counts_sibling(ssp_data):
    waves, trans = _filters()
    rb = precompute_restband_photometry(ssp_data, waves, trans, convention=_CONV)
    rb = rb._replace(
        slopes=precompute_restband_photometry(_slope_ssp(ssp_data), waves, trans, convention=_CONV)
    )
    _check(rb, expect_sibling=True)
    assert rb.slopes.nbytes == rb._replace(slopes=None).nbytes


def test_photometric_ztable_counts_sibling(ssp_data):
    waves, trans = _filters()
    zt = precompute_photometry_ztable(ssp_data, waves, trans, n_z=5, convention=_CONV)
    zt = zt._replace(
        slopes=precompute_photometry_ztable(
            _slope_ssp(ssp_data), waves, trans, n_z=5, convention=_CONV
        )
    )
    _check(zt, expect_sibling=True)
    assert zt.nbytes == 2 * zt._replace(slopes=None).nbytes


def test_energy_balance_lut_counts_sibling(ssp_data):
    tau_bc = jnp.asarray([0.0, 0.5])
    tau_diff = jnp.asarray([0.0, 0.3, 0.9])
    kwargs = dict(
        law_bc="calzetti", law_diff="calzetti", tau_bc_grid=tau_bc, tau_diff_grid=tau_diff
    )
    lut = build_energy_balance_lut_with_slopes(build_energy_balance_lut, ssp_data, **kwargs)
    _check(lut, expect_sibling=True)
    assert lut.slopes.nbytes == lut._replace(slopes=None).nbytes


def test_energy_balance_lut_without_sibling(ssp_data):
    tau_bc = jnp.asarray([0.0, 0.5])
    tau_diff = jnp.asarray([0.0, 0.3, 0.9])
    lut = build_energy_balance_lut(
        ssp_data.ssp_flux,
        ssp_data.ssp_wave,
        law_bc="calzetti",
        law_diff="calzetti",
        tau_bc_grid=tau_bc,
        tau_diff_grid=tau_diff,
    )
    _check(lut, expect_sibling=False)


def test_spectroscopic_ztable_counts_its_pixel_grid():
    from tengri.components.stellar.sps.precompute import SpectroscopicZTable

    wave_obs = jnp.linspace(5200.0, 7400.0, 200)
    zt = SpectroscopicZTable(wave_obs_pixels=wave_obs, z_min=0.01, z_max=3.0)
    assert zt.nbytes == int(wave_obs.nbytes)


def test_stellar_state_counts_each_stored_array_once(ssp_data):
    """One number per state: its tables' nbytes plus the arrays it holds directly."""
    from tengri.components.stellar.component import StellarSEDComponentState

    waves, trans = _filters()
    lut = precompute_photometry(ssp_data, waves, trans, REDSHIFT, DL_CM, convention=_CONV)
    lut = lut._replace(
        slopes=precompute_photometry(
            _slope_ssp(ssp_data), waves, trans, REDSHIFT, DL_CM, convention=_CONV
        )
    )
    cube = _rng_cube((3, 4, 50))
    cube_slope = _rng_cube((3, 4, 50))
    fw = _rng_cube((3, 30))
    state = StellarSEDComponentState(
        ssp_phot_lut=lut,
        spec_ssp_broadened=cube,
        spec_ssp_broadened_slope=cube_slope,
        phot_fw_padded=fw,
        phot_ft_padded=fw,
        spec_sigma_v_fixed=80.0,
    )
    expected = lut.nbytes + int(cube.nbytes) + int(cube_slope.nbytes) + 2 * int(fw.nbytes)
    assert state.nbytes == expected
    assert StellarSEDComponentState().nbytes == 0
