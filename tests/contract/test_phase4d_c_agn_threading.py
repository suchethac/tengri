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

import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, SEDModel
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


def _build(ssp, obs, **groups):
    """Build through the group grammar, quietly.

    The two specs this replaced were `Parameters(...)` flat forms. The no-AGN
    one left `sfh_dpl_age_gyr` free while every test here calls with
    `params = {}`; the SKIRTOR one passed `agn_log_lbol` and `agn_torus_frac`
    without declaring an AGN model, which the grammar no longer accepts. Both
    raise the moment this file's SSP grid exists -- and nothing generates it,
    so the file has never run in CI (#2183).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
            **groups,
        )


def _no_agn(ssp, obs):
    """Control: no AGN group at all."""
    return _build(ssp, obs)


def _skirtor(ssp, obs):
    """SKIRTOR torus, every parameter pinned so `params = {}` is complete."""
    return _build(ssp, obs, agn={"torus": {"type": "skirtor"}, "all_params": Fixed(DEFAULT)})


# ── _template_data_for_jit() contract ────────────────────────────────────────


def test_no_agn_returns_no_agn_templates(ssp_bare, obs):
    """When no AGN component is present, template_data has no 'agn' key.

    Written as a disjunction rather than ``if td is not None:``. Both express
    the same claim -- ``td is None`` satisfies it, since nothing threaded means
    no ``agn`` key -- but the ``if`` form hides whether the body runs at all.
    Its sibling in ``test_phase4d_b_dust_ir_threading.py`` was measured always
    taking the ``None`` branch, so that form there was asserting nothing.
    """
    model = _no_agn(ssp_bare, obs)
    td = model._template_data_for_jit()
    assert td is None or "agn" not in td, (
        f"No-AGN model should not have 'agn' in template_data; got {td!r}"
    )


def test_skirtor_agn_publishes_templates_for_jit(ssp_bare, obs):
    """A model built with SKIRTOR AGN has template_data with 'agn.skirtor'."""
    try:
        model = _skirtor(ssp_bare, obs)
    except (ImportError, FileNotFoundError):
        pytest.skip("SKIRTOR templates not available")

    td = model._template_data_for_jit()
    assert td is not None, "SKIRTOR AGN model should publish template_data"
    assert "agn" in td, "SKIRTOR should populate 'agn' key in template_data"
    agn_data = td["agn"]
    assert isinstance(agn_data, dict), "agn value should be a dict"

    # The original asserted `"skirtor" in agn_data`. Measured, the blocks are
    # nested one level deeper and keyed by their composable slot, so that claim
    # was false as written -- invisible while the file skipped.
    blocks = agn_data["blocks"]
    assert "torus/skirtor" in blocks, (
        f"SKIRTOR torus block should be threaded; got {sorted(blocks)}"
    )
    leaves = [
        leaf for leaf in jax.tree.flatten(blocks["torus/skirtor"])[0] if hasattr(leaf, "shape")
    ]
    assert leaves, "the SKIRTOR block should carry array leaves, not an empty container"


# ── JIT-path bit-exactness ──────────────────────────────────────────────────


def test_jit_and_non_jit_agree_no_agn(ssp_bare, obs):
    """JIT and non-JIT paths agree when no AGN (proves signature didn't break)."""
    model = _no_agn(ssp_bare, obs)
    params = {}

    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for no-AGN model after Phase 4-D wiring"
    )


def test_jit_and_non_jit_agree_with_skirtor(ssp_bare, obs):
    """JIT and non-JIT paths agree with SKIRTOR AGN after threading."""
    try:
        model = _skirtor(ssp_bare, obs)
    except (ImportError, FileNotFoundError):
        pytest.skip("SKIRTOR templates not available")

    params = {}

    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for SKIRTOR AGN after Phase 4-D threading"
    )
