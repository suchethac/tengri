# SPDX-License-Identifier: BSD-3-Clause
"""Catalog MCMC engine with the MEADS ("ghmc") sampler: seams, starts and wall safety.

The engine is driven through its ``flat_logdensity`` seam with a Gaussian target,
so no forward model is built. Each test pins one property of the seam: the
sampler is accepted, the return shapes hold, two galaxies share one compiled
program, the chain-count rules refuse the wrong counts, the superchains start
from distinct yet grouped positions, and a log-density with a hard wall keeps
the adapted step size and damping finite.
"""

from __future__ import annotations

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.backends.mcmc._shared import ghmc_catalog_scan
from tengri.inference.backends.mcmc.catalog import (
    build_catalog_mcmc_engine,
    ghmc_superchain_starts,
)

pytestmark = pytest.mark.contract

_N_WARMUP = 40
_N_BURNIN = 5
_N_SAMPLES = 12
_N_CHAINS = 16
_N_SUPERCHAINS = 4
_N_FOLDS = 4
_DIM = 2


def _gauss_logdensity(x, data_args):
    """Standard-normal-shaped log-density centred on the galaxy's ``mu``."""
    return -0.5 * jnp.sum((x - data_args["mu"]) ** 2)


def _substitute(data, noise, presence, redshift, line_flux_obs, line_flux_err):
    """Per-galaxy data_args: the galaxy's centre only."""
    return {"mu": data}


def _flat(logdensity=_gauss_logdensity):
    """The four-tuple ``flat_logdensity`` seam for a two-dimensional target."""
    init_flat = jnp.zeros(_DIM)
    return (logdensity, lambda x: x, init_flat, {"mu": jnp.zeros(_DIM)})


def _engine(**overrides):
    """Build the ghmc catalog engine with the test's small counts."""
    kwargs = dict(
        n_warmup=_N_WARMUP,
        n_burnin=_N_BURNIN,
        n_samples=_N_SAMPLES,
        n_chains=_N_CHAINS,
        n_superchains=_N_SUPERCHAINS,
        flat_logdensity=_flat(),
        substitute=_substitute,
    )
    kwargs.update(overrides)
    run_one, _unravel = build_catalog_mcmc_engine(None, "ghmc", **kwargs)
    return run_one


def test_engine_accepts_ghmc_and_returns_chain_stacked_draws():
    run_one = _engine()
    key = jax.random.PRNGKey(0)
    positions, divergent = run_one(
        jnp.zeros(_DIM), key, jnp.ones(_DIM), None, None, None, None, None
    )
    assert positions.shape == (_N_CHAINS * _N_SAMPLES, _DIM)
    assert divergent.shape == (_N_CHAINS * _N_SAMPLES,)
    assert bool(jnp.all(jnp.isfinite(positions)))


def test_two_galaxies_with_different_data_share_one_compiled_program():
    run_one = _engine()
    jitted = jax.jit(run_one)
    key = jax.random.PRNGKey(1)
    init = jnp.zeros(_DIM)
    first_data = jnp.array([1.0, 0.5])
    second_data = jnp.array([-2.0, 3.0])
    jitted(init, key, first_data, None, None, None, None, None)
    with jtu.count_jit_compilation_cache_miss() as misses:
        jitted(init, key, second_data, None, None, None, None, None)
    assert misses() == 0


def test_refuses_n_chains_not_a_multiple_of_n_superchains():
    with pytest.raises(ValueError, match=r"n_chains=10.*n_superchains=4"):
        _engine(n_chains=10, n_superchains=4)


def test_refuses_n_chains_not_a_multiple_of_n_folds():
    with pytest.raises(ValueError, match=r"n_chains=18.*n_folds=4"):
        _engine(n_chains=18, n_superchains=3)


def test_superchain_starts_distinct_across_and_shared_within():
    init = jnp.zeros(_DIM)
    starts = np.asarray(
        ghmc_superchain_starts(init, jax.random.PRNGKey(2), _N_CHAINS, _N_SUPERCHAINS, 0.5)
    )
    per = _N_CHAINS // _N_SUPERCHAINS
    assert starts.shape == (_N_CHAINS, _DIM)
    grouped = starts.reshape(_N_SUPERCHAINS, per, _DIM)
    assert np.allclose(grouped, grouped[:, :1, :])
    firsts = grouped[:, 0, :]
    for a in range(_N_SUPERCHAINS):
        for b in range(a + 1, _N_SUPERCHAINS):
            assert not np.allclose(firsts[a], firsts[b])
    assert not np.allclose(firsts, 0.0)


def test_hard_wall_keeps_adapted_step_size_and_damping_finite():
    def walled(x, data_args):
        """Log-density that is NaN outside the box |x| < 1 in every coordinate."""
        return jnp.sum(jnp.log(1.0 - x**2)) - 0.5 * jnp.sum(x**2)

    n_chain = 300
    chain_keys = jax.random.split(jax.random.PRNGKey(3), _N_CHAINS * n_chain).reshape(
        _N_CHAINS, n_chain, 2
    )
    starts = ghmc_superchain_starts(
        0.1 * jnp.ones(_DIM), jax.random.PRNGKey(5), _N_CHAINS, _N_SUPERCHAINS, 0.5
    )
    positions, divergent, step_size, momentum_inv_scale, alpha, delta = ghmc_catalog_scan(
        starts, jax.random.PRNGKey(4), chain_keys, walled, {}, 150, _N_FOLDS
    )
    assert bool(jnp.isfinite(step_size)) and float(step_size) > 0.0
    assert bool(jnp.all(jnp.isfinite(momentum_inv_scale)))
    assert bool(jnp.isfinite(alpha)) and bool(jnp.isfinite(delta))
    assert bool(jnp.all(jnp.isfinite(positions)))
    assert bool(jnp.all(jnp.abs(positions) < 1.0))
    assert divergent.dtype == jnp.bool_
    assert divergent.shape == (_N_CHAINS, n_chain)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MEADS in the catalog engine does not adapt to a sigma=1e-3 target: "
        "step 5e-7, sd 0.4 at 300 warmup (#3135)"
    ),
)
def test_default_recovers_a_sigma_1e3_gaussian():
    sigma = 1e-3

    def narrow_logdensity(x, data_args):
        """Gaussian of width ``sigma`` centred on the galaxy's ``mu``."""
        return -0.5 * jnp.sum((x - data_args["mu"]) ** 2) / sigma**2

    run_one, _unravel = build_catalog_mcmc_engine(
        None,
        "ghmc",
        n_warmup=300,
        n_burnin=20,
        n_samples=150,
        n_chains=_N_CHAINS,
        n_superchains=_N_SUPERCHAINS,
        flat_logdensity=_flat(narrow_logdensity),
        substitute=_substitute,
    )
    centre = jnp.array([2.0, 2.0])
    positions, _divergent = run_one(
        centre, jax.random.PRNGKey(6), centre, None, None, None, None, None
    )
    draws = np.asarray(positions)
    assert np.all(np.abs(draws.mean(axis=0) - 2.0) <= 0.05 * sigma)
    sd = draws.std(axis=0)
    assert np.all(np.abs(sd / sigma - 1.0) <= 0.2)


def test_driver_reports_finite_nested_rhat(synthetic_ssp):
    from tengri.inference.batched_spectro import fit_spectra_catalog_mcmc
    from tengri.observation.batched import build_spectro_batches
    from tests.contract.test_batched_spectro_loglik import _mock_catalog, _record
    from tests.contract.test_batched_spectrum_forward import _banded, _free_z_model

    mock = _mock_catalog(synthetic_ssp)
    template = _free_z_model(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))
    batches = build_spectro_batches(
        [_record(mock[i]) for i in (0, 2)], quantum=128, model=template
    )
    result = fit_spectra_catalog_mcmc(
        template,
        batches,
        sampler="ghmc",
        n_warmup=300,
        n_burnin=0,
        n_samples=30,
        init="map",
        map_steps=60,
        n_chains=32,
        n_superchains=8,
    )
    rhat = np.asarray(result["nested_rhat"])
    assert rhat.shape == (2,)
    assert np.all(np.isfinite(rhat))
