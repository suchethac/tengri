# SPDX-License-Identifier: BSD-3-Clause
"""The engine's NIFTy wrapper follows the forward topology, not an exception.

``build_jit_engine`` wraps the threaded signal response in a ``jft.Model`` whose
domain is one scalar per free name. That wrapper is valid only for the
single-galaxy topology the threaded forward supports; a ``PopulationSEDModel``
forward batches galaxies, and asking its vmap to run on scalar parameters
raises. The engine therefore builds the wrapper by topology and never by
catching that error (which used to hide it, and with it every other
``ValueError`` on the signal-response path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import Fixed, Parameters, SEDModel, Uniform
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population_sed_model import PopulationSEDModel
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.contract

_N_GAL = 2


def _spec() -> Parameters:
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(2.0),
        sfh_dpl_log_total_mass=Uniform(9.0, 11.0),
        redshift=Fixed(0.05),
        noise_frac_cal=Uniform(0.01, 0.2),
    )


def _single_fitter(synthetic_ssp, simple_observation) -> Fitter:
    model = SEDModel(_spec(), synthetic_ssp, observation=simple_observation)
    return Fitter(model, jnp.ones(3) * 1e-18, jnp.ones(3) * 1e-19, data_type="photometry")


def _population_fitter(synthetic_ssp, simple_observation) -> Fitter:
    template = SEDModel(_spec(), synthetic_ssp, observation=simple_observation)
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[
            {"flux_obs": jnp.ones(3) * 1e-18, "noise": jnp.ones(3) * 1e-19} for _ in range(_N_GAL)
        ],
    )
    forward = ForwardModel.build(population=pop, observation=simple_observation)
    return Fitter(forward)


def test_population_engine_builds_without_a_nifty_wrapper(synthetic_ssp, simple_observation):
    """A population fitter's engine builds, and carries no scalar-domain NIFTy model."""
    fitter = _population_fitter(synthetic_ssp, simple_observation)
    assert fitter.data.shape == (_N_GAL, 3)
    engine = fitter._build_jit_engine(fitter._initialize_unbounded(jax.random.PRNGKey(0)))
    assert engine["nifty_model"] is None


def test_single_galaxy_engine_carries_the_nifty_wrapper(synthetic_ssp, simple_observation):
    """A single-galaxy fitter's engine carries the wrapper whenever nifty is installed."""
    pytest.importorskip("nifty8.re")
    fitter = _single_fitter(synthetic_ssp, simple_observation)
    engine = fitter._build_jit_engine(fitter._initialize_unbounded(jax.random.PRNGKey(0)))
    assert engine["nifty_model"] is not None
