# SPDX-License-Identifier: BSD-3-Clause
"""Phase 4-D Category C tests: AGN template threading as JIT runtime inputs.

Extends Phase 4-C (nebular threading) to AGN components with large template
grids (SKIRTOR, KD18 disc, GRAHSP, CAT3D wind, Silva04, NTHCOMP). The contract:

* ``SEDModel._template_data_for_jit()`` returns a nested dict with "agn"
  sub-dict carrying SKIRTOR and other AGN template structures when the model
  has AGN components that use them.
* ``predict_observables_jit(params)`` threads it as the 4th JIT runtime
  input so AGN template arrays become JIT ``Parameter`` ops rather than
  baked Constants.
* JIT and non-JIT paths agree to floating-point precision.
"""

from __future__ import annotations

import pathlib
import warnings

import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed

pytestmark = pytest.mark.contract

_SSP_BARE = pathlib.Path("data/ssp_prsc_bc03_chabrier.h5").resolve()


@pytest.fixture(scope="module")
def ssp_bare():
    if not _SSP_BARE.exists():
        pytest.skip(f"bare-stellar SSP not available at {_SSP_BARE}")
    return load_ssp_data(str(_SSP_BARE))


@pytest.fixture(scope="module")
def obs():
    return Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )


def _no_agn_spec():
    """Spec without AGN (control case)."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        igm={"type": "none"},
    )


def _skirtor_spec():
    """Spec with SKIRTOR AGN."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        agn_log_lbol=Fixed(10.42),
        agn_torus_frac=Fixed(0.5),
        igm={"type": "none"},
    )


def _silent_build(spec, ssp, obs, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, **kwargs)


# ── _template_data_for_jit() contract ────────────────────────────────────────


def test_no_agn_returns_no_agn_templates(ssp_bare, obs):
    """When no AGN component is present, template_data has no 'agn' key."""
    model = _silent_build(_no_agn_spec(), ssp_bare, obs)
    td = model._template_data_for_jit()
    # Either None or a dict without 'agn' key is acceptable.
    if td is not None:
        assert "agn" not in td, "No-AGN model should not have 'agn' in template_data"


def test_skirtor_agn_publishes_templates_for_jit(ssp_bare, obs):
    """A model built with SKIRTOR AGN has template_data with 'agn.skirtor'."""
    try:
        model = _silent_build(_skirtor_spec(), ssp_bare, obs, agn_model="skirtor")
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("SKIRTOR templates not available")

    td = model._template_data_for_jit()
    assert td is not None, "SKIRTOR AGN model should publish template_data"
    assert "agn" in td, "SKIRTOR should populate 'agn' key in template_data"
    agn_data = td["agn"]
    assert isinstance(agn_data, dict), "agn value should be a dict"
    assert "skirtor" in agn_data, "SKIRTOR key should be in agn dict"

    # Verify the template is a callable (or similar).
    skirtor_template = agn_data["skirtor"]
    assert skirtor_template is not None, "SKIRTOR template should be non-None"


# ── JIT-path bit-exactness ──────────────────────────────────────────────────


def test_jit_and_non_jit_agree_no_agn(ssp_bare, obs):
    """JIT and non-JIT paths agree when no AGN (proves signature didn't break)."""
    model = _silent_build(_no_agn_spec(), ssp_bare, obs)
    params = {}

    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for no-AGN model after Phase 4-D wiring"
    )


def test_jit_and_non_jit_agree_with_skirtor(ssp_bare, obs):
    """JIT and non-JIT paths agree with SKIRTOR AGN after threading."""
    try:
        model = _silent_build(_skirtor_spec(), ssp_bare, obs, agn_model="skirtor")
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("SKIRTOR templates not available")

    params = {}

    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for SKIRTOR AGN after Phase 4-D threading"
    )
