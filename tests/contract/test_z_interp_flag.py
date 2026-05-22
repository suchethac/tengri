# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the ``z_interp`` flag that selects triweight vs. linear z-table
interpolation inside the hybrid forward kernel.

Covers only the wiring — the numerical correctness of
``interpolate_ztable_smooth`` itself is exercised in
``test_ztable_precompute.py``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)

from tengri.forward.sed_model import SEDModel
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = [pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")]


def _make_spec(z_interp: str | None = None) -> Parameters:
    kwargs: dict = dict(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=Uniform(0.05, 0.15),
    )
    if z_interp is not None:
        kwargs["z_interp"] = z_interp
    return Parameters(**kwargs)


class TestZInterpFlag:
    def test_default_is_linear(self, ssp_data_wne, sdss_filters):
        model = SEDModel(_make_spec(), ssp_data_wne, filters=sdss_filters)
        assert model._z_interp == "linear"

    def test_smooth_opt_in(self, ssp_data_wne, sdss_filters):
        model = SEDModel(_make_spec(z_interp="smooth"), ssp_data_wne, filters=sdss_filters)
        assert model._z_interp == "smooth"

    def test_unknown_value_preserved(self, ssp_data_wne, sdss_filters):
        """Unknown strings pass through untouched — validation happens at the
        kernel-compile branch, not here, so callers still get a loud error."""
        model = SEDModel(_make_spec(z_interp="nonsense"), ssp_data_wne, filters=sdss_filters)
        assert model._z_interp == "nonsense"
