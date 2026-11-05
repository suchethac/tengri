# SPDX-License-Identifier: BSD-3-Clause
"""The catalog engine passes the shared SSP grid and templates as jit arguments.

``build_catalog_mcmc_engine`` serves every galaxy of a catalog from one compiled
program. Its shared inputs (the SSP grid, template data, fixed values, all under
``data_args["_jit_inputs"]``) are identical for every galaxy. Closure-capturing
them embeds the arrays in the executable: on a large SSP cube the executable
grows past the 2 GB protobuf limit, so the persistent compile cache cannot store
it and every process recompiles (#3192). These tests pin the seam:

1. The traced program holds no constant as large as the SSP grid. The shared
   pytree enters as an argument, and the per-galaxy arrays stay per galaxy.
2. Passing the shared arrays as arguments leaves the log-density bit-identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

pytestmark = pytest.mark.contract

_PROBE_SEED = 7


def _build_model(synthetic_ssp, simple_observation):
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, synthetic_ssp, observation=simple_observation)


@pytest.fixture
def catalog_engine_inputs(synthetic_ssp, simple_observation):
    """A template fitter, one galaxy's arrays, the shared pytree and the SSP grid."""
    from tengri import Fitter

    model = _build_model(synthetic_ssp, simple_observation)
    params = dict(model.spec.sample(jax.random.PRNGKey(_PROBE_SEED)))
    flux = model.predict_photometry(params)
    noise = jnp.abs(flux) * 0.02
    fitter = Fitter(model, flux, noise, data_type="photometry")
    init_flat = ravel_pytree(fitter._initialize_unbounded(jax.random.PRNGKey(0)))[0]
    galaxy = (
        init_flat,
        jax.random.PRNGKey(3),
        flux,
        noise,
        jnp.ones_like(flux),
        jnp.asarray(0.1),
        None,
        None,
    )
    return fitter, galaxy, fitter._data_args, synthetic_ssp


def test_engine_program_holds_no_shared_array_as_a_constant(catalog_engine_inputs):
    """No constant in the traced sampler is as large as the SSP flux cube.

    The SSP cube is the largest shared array; a constant that size is the 2 GB
    executable of #3192 in miniature.
    """
    from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

    fitter, galaxy, shared, ssp = catalog_engine_inputs
    run_one, _ = build_catalog_mcmc_engine(fitter, "nuts", n_warmup=5, n_burnin=2, n_samples=4)

    def program(*args):
        return run_one(*args[:8], args[8])

    jaxpr = jax.make_jaxpr(program)(*galaxy, shared)
    ssp_nbytes = np.asarray(ssp.ssp_flux).nbytes
    big = [int(np.asarray(c).nbytes) for c in jaxpr.consts if np.asarray(c).nbytes >= ssp_nbytes]
    assert big == [], f"constants at or above the SSP cube size ({ssp_nbytes} B): {big}"


def test_log_density_is_bit_identical_with_shared_arrays_as_arguments(catalog_engine_inputs):
    """Passing the shared tables as arguments leaves the log-density unchanged.

    The argument path is compared exactly with the eager evaluation, which takes the
    same tables as its argument. The closure path embeds the tables as constants, and
    XLA folds those differently: on the probe it lands one ulp (about 2e-16 relative)
    away, deterministically. That is a compiler rounding difference, not a change of
    value, so it is bounded at a few ulp rather than pinned exactly. No frozen constant
    is recorded: its value depends on the platform's libm and BLAS, and a pin recorded
    on one machine failed on the CI runners.
    """
    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity

    fitter, galaxy, _shared, _ssp = catalog_engine_inputs
    init_flat = galaxy[0]
    log_post, _unravel, _init, template = _get_flat_logdensity(
        fitter, fitter._initialize_unbounded(jax.random.PRNGKey(0))
    )
    eager_value = float(log_post(init_flat, template))
    argument_value = float(jax.jit(log_post)(init_flat, template))
    closure_value = float(jax.jit(lambda x: log_post(x, template))(init_flat))
    assert np.isfinite(argument_value)
    assert argument_value == eager_value
    np.testing.assert_allclose(closure_value, argument_value, rtol=4 * np.finfo(float).eps)


def _all_consts(jaxpr) -> list:
    """Every constant of ``jaxpr`` and of the jaxprs nested in its equations."""
    found = list(jaxpr.consts)
    for eqn in jaxpr.jaxpr.eqns:
        for value in eqn.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(item, "jaxpr") and hasattr(item, "consts"):
                    found.extend(_all_consts(item))
                elif hasattr(item, "eqns"):
                    found.extend(_all_consts(jax.core.ClosedJaxpr(item, ())))
    return found


def test_jitted_map_program_holds_no_shared_array_as_a_constant(
    catalog_engine_inputs, monkeypatch
):
    """The program ``map_with_shared`` hands to ``jax.jit`` embeds no SSP-sized constant.

    The wrapper's jitted function and its call arguments are recorded, then traced
    with ``make_jaxpr`` including the nested ``lax.map`` body. A wrapper that closes
    over ``shared`` instead of taking it as an input leaves the SSP cube in the
    constants.
    """
    from tengri.inference.backends.mcmc import catalog
    from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine, map_with_shared

    fitter, galaxy, shared, ssp = catalog_engine_inputs
    run_one, _ = build_catalog_mcmc_engine(fitter, "nuts", n_warmup=5, n_burnin=2, n_samples=4)
    xs = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf, leaf]), galaxy)
    recorded = {}
    real_jit = jax.jit

    def recording_jit(fn, *args, **kwargs):
        compiled = real_jit(fn, *args, **kwargs)

        def call(*call_args):
            # ``jax.jit`` is patched process-wide, so JAX's own jitted helpers run during
            # the map's trace are routed here too. Keep only the outermost call, the
            # first one: its arguments are concrete, while the inner calls receive tracers
            # that escape the trace, and which inner call comes last depends on JAX's
            # warm caches (so the test passed locally and failed on a cold CI worker).
            recorded.setdefault("fn", fn)
            recorded.setdefault("args", call_args)
            return compiled(*call_args)

        return call

    monkeypatch.setattr(catalog.jax, "jit", recording_jit)
    map_with_shared(run_one, xs, shared, batch_size=2)
    monkeypatch.undo()

    assert recorded["fn"].__qualname__ == "_mapped_program.<locals>.mapped"
    jaxpr = jax.make_jaxpr(recorded["fn"])(*recorded["args"])
    ssp_nbytes = np.asarray(ssp.ssp_flux).nbytes
    big = [
        int(np.asarray(c).nbytes) for c in _all_consts(jaxpr) if np.asarray(c).nbytes >= ssp_nbytes
    ]
    assert big == [], f"constants at or above the SSP cube size ({ssp_nbytes} B): {big}"
