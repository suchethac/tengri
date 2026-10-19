# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2969: CB_19's neb_logZ_gas support is in the wrong units.

Public ``neb_logZ_gas`` is log10(Z/Zsun). The CB_19 lookup forms
``log_OH_total = neb_logZ_gas_absolute + _LOG_OH_OFFSET`` with
``_LOG_OH_OFFSET = _LOG_OH_SOLAR - LOG10_ZSUN``, so the public value maps to
``log_OH_total`` by adding ``_LOG_OH_SOLAR`` alone. The registered support was
instead ``log_OH_total - _LOG_OH_OFFSET`` (about [-3.84, -1.36]), which sits
1.22 dex low. Its midpoint lies below the grid, where ``_frac_idx`` clips and the
gradient is exactly zero, and a solar ``Fixed(0.0)`` is refused as overhang.

The correct support is the CB_19 log(O/H) axis ends minus ``_LOG_OH_SOLAR``,
about [-1.99, 0.49].
"""

from __future__ import annotations

from pathlib import Path

import h5py
import jax.numpy as jnp
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, Photometry
from tengri._data_setup import package_or_env_data_path
from tengri.components.grid_support import grid_support
from tengri.components.nebular._constants import _LOG_OH_SOLAR
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tests._cb19_grid import write_synthetic_cb19_grid

pytestmark = pytest.mark.regression_bug

#: Public log10(Z/Zsun) support implied by the shipped CB_19 log(O/H) axis.
_EXPECTED_PUBLIC_SUPPORT = (-1.99, 0.49)


def _shipped_log_oh_axis_ends() -> tuple[float, float]:
    """Ends of the CB_19 ``log_OH_total`` axis, read from the file the registry reads."""
    with h5py.File(package_or_env_data_path("cb19_templates.h5"), "r") as f:
        log_oh = f["axes/log_OH_total"][:]
    return float(log_oh.min()), float(log_oh.max())


@pytest.fixture(scope="module")
def ssp() -> SSPData:
    """Small synthetic SSP; only Q_H matters for the nebular lookup."""
    wave = jnp.logspace(2.0, 5.0, 400)
    ages_gyr = jnp.linspace(-3.0, 1.14, 15)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    flux = jnp.broadcast_to((5000.0 / wave) ** 2, (3, 15, wave.shape[0]))
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


def _cb19_model(ssp, tmp_path: Path, monkeypatch, logz_gas):
    """A cb19 build on the synthetic grid, with ``neb_logZ_gas`` set to ``logz_gas``."""
    grid = write_synthetic_cb19_grid(tmp_path / "cb19_templates.h5")
    monkeypatch.setattr("tengri.components.nebular.cloudy_cb19._DEFAULT_PATH", Path(grid))
    obs = Observation(photometry=Photometry.from_names(["sdss_g"]))
    return tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={"type": "cb19", "logZ_gas": logz_gas, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
    )


def test_cb19_logz_gas_support_is_log_oh_axis_in_public_units():
    """The registered support is the log(O/H) axis ends shifted by ``_LOG_OH_SOLAR``."""
    lo_oh, hi_oh = _shipped_log_oh_axis_ends()
    lo, hi = grid_support("neb", "cb19")["neb_logZ_gas"]

    assert lo == pytest.approx(lo_oh - _LOG_OH_SOLAR, abs=1e-6)
    assert hi == pytest.approx(hi_oh - _LOG_OH_SOLAR, abs=1e-6)
    assert lo == pytest.approx(_EXPECTED_PUBLIC_SUPPORT[0], abs=0.01)
    assert hi == pytest.approx(_EXPECTED_PUBLIC_SUPPORT[1], abs=0.01)


def test_cb19_solar_fixed_logz_gas_is_accepted(ssp, tmp_path, monkeypatch):
    """Solar metallicity (public 0.0) is inside the CB_19 support, so the build is not refused."""
    model = _cb19_model(ssp, tmp_path, monkeypatch, Fixed(0.0))
    assert model is not None


def test_cb19_free_logz_gas_prior_spans_the_log_oh_axis(ssp, tmp_path, monkeypatch):
    """A FREE ``neb_logZ_gas`` is narrowed to the public-unit support, not the shifted one."""
    model = _cb19_model(ssp, tmp_path, monkeypatch, FREE)
    lo, hi = model.spec.get_distribution("neb_logZ_gas").bounds

    lo_reg, hi_reg = grid_support("neb", "cb19")["neb_logZ_gas"]
    assert lo == pytest.approx(lo_reg, abs=1e-9)
    assert hi == pytest.approx(hi_reg, abs=1e-9)
    assert lo == pytest.approx(_EXPECTED_PUBLIC_SUPPORT[0], abs=0.01)
    assert hi == pytest.approx(_EXPECTED_PUBLIC_SUPPORT[1], abs=0.01)
