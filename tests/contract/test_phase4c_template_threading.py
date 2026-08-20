# SPDX-License-Identifier: BSD-3-Clause
"""Phase 4-C tests: nebular template data threading as JIT runtime inputs.

Extends Phase 4-B (SSP threading) to nebular backend grids (Cue weights,
CloudyGrid grids, etc.). The contract:

* ``SEDModel._template_data_for_jit()`` returns a dict-like (or ``None``)
  carrying the nebular backend's grid/weights when the model has a
  non-BakedIn nebular backend, ``None`` otherwise.
* ``predict_observables_jit(params)`` threads it as the 4th JIT runtime
  input (after params, fixed_values, ssp_data) so backend arrays become
  JIT ``Parameter`` ops rather than baked Constants.
* JIT and non-JIT paths agree to floating-point precision.
"""

from __future__ import annotations

import pathlib
import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.contract

# Cue requires a bare-stellar SSP (per project_ssp_cue_incompat memory).
_SSP_BARE = pathlib.Path("data/ssp_prsc_bc03_chabrier.h5").resolve()
_SSP_WNE = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp_bare():
    if not _SSP_BARE.exists():
        pytest.skip(f"bare-stellar SSP not available at {_SSP_BARE}")
    return load_ssp_data(str(_SSP_BARE))


@pytest.fixture(scope="module")
def ssp_wne():
    if not _SSP_WNE.exists():
        pytest.skip(f"wNE SSP not available at {_SSP_WNE}")
    return load_ssp_data(str(_SSP_WNE))


@pytest.fixture(scope="module")
def obs():
    return Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )


def _bakedin_spec():
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )


def _cue_spec():
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        nebular_backend="cue",
        neb_logU=Uniform(-4.0, -2.0),
        neb_xi_ion=Fixed(25.5),
        apply_igm=False,
    )


def _silent_build(spec, ssp, obs, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, **kwargs)


# ── _template_data_for_jit() contract ────────────────────────────────────────


def test_bakedin_nebular_returns_no_template_data(ssp_wne, obs):
    """When the nebular backend is BakedIn (or absent), there's no large
    grid to thread — ``_template_data_for_jit()`` returns ``None`` (or
    an empty mapping that JAX treats as no extra input).
    """
    model = _silent_build(_bakedin_spec(), ssp_wne, obs)
    td = model._template_data_for_jit()
    # Either None or an empty mapping is acceptable for "nothing to thread".
    assert td is None or (hasattr(td, "__len__") and len(td) == 0), (
        f"Expected None or empty mapping for BakedIn nebular; got {td!r}"
    )


def test_cue_backend_publishes_weights_for_jit(ssp_bare, obs):
    """A model built with the Cue backend has non-trivial template_data —
    so ``predict_observables_jit`` threads it.
    """
    try:
        model = _silent_build(_cue_spec(), ssp_bare, obs)
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("Cue backend not available")

    td = model._template_data_for_jit()
    assert td is not None, "Cue backend should publish non-None template_data"
    # The Cue backend carries multiple arrays (NN layer weights). The
    # structure varies per backend; we just check at least one array
    # is present.
    leaves, _ = jax.tree.flatten(td)
    array_leaves = [leaf for leaf in leaves if hasattr(leaf, "shape")]
    assert len(array_leaves) > 0, "template_data should contain at least one array leaf"


# ── JIT-path bit-exactness ──────────────────────────────────────────────────


def test_jit_and_non_jit_paths_agree_with_bakedin(ssp_wne, obs):
    """Sanity: even when there's no template_data, JIT and non-JIT paths
    agree (proves the 4-arg JIT signature didn't break the bakedin case).
    """
    model = _silent_build(_bakedin_spec(), ssp_wne, obs)
    params = {}

    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for BakedIn nebular after Phase 4-C wiring"
    )


def test_jit_and_non_jit_paths_agree_with_cue(ssp_bare, obs):
    """Same guarantee for Cue: the new ``template_data`` runtime input
    routes correctly through both paths.
    """
    try:
        model = _silent_build(_cue_spec(), ssp_bare, obs)
    except (ImportError, FileNotFoundError, KeyError):
        pytest.skip("Cue backend not available")

    params = {"neb_logU": jnp.asarray(-3.0)}
    via_jit = model.predict_observables_jit(params).phot_fnu
    via_nojit = model.predict_observables(params).phot_fnu

    assert jnp.allclose(via_jit, via_nojit, rtol=1e-12, atol=0), (
        "JIT and non-JIT paths diverged for Cue nebular after Phase 4-C wiring"
    )
