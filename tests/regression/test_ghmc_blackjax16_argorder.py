# SPDX-License-Identifier: BSD-3-Clause
"""GHMC call sites match the installed blackjax signature (argument-order regression).

``blackjax.mcmc.ghmc.init`` takes ``(position, logdensity_fn, rng_key)`` in
blackjax >= 1.5 and took ``(position, rng_key, logdensity_fn)`` before, which
handed the PRNG key to blackjax where the log-density callable was expected
(``TypeError: Expected a callable value, got JitTracer(uint32[2])``).

The claim is the blackjax call order and the kernel arguments, not posterior
quality, so the cached-adaptation test runs the broken-tier ``mcmc_ghmc``
backend with ``allow_unvalidated``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fitter, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.backends.mcmc._shared import _ghmc_full_scan
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

_CENTER = np.asarray([1.0, -2.0])
_N_DRAWS = 3000


def _logdensity(position, data_args):
    return -0.5 * jnp.sum((position - data_args) ** 2)


def test_ghmc_scan_reproduces_a_unit_gaussian():
    """tengri's GHMC scan recovers the mean and variance of a unit Gaussian at (1, -2).

    Measured autocorrelation time of a 3000-draw chain is at most 8 draws over
    four keys, so the effective sample size is at least 375: the standard error
    of the mean is 1/sqrt(375) = 0.052 and of the variance sqrt(2/375) = 0.073.
    Tolerances are 0.25 on the mean (4.8 standard errors) and 0.3 on the
    variance (4.1 standard errors). A key handed to blackjax in place of the
    log-density, or the step size and inverse mass matrix swapped in the kernel
    call, raises or gives a chain whose moments miss these bounds.
    """
    warmup_key, init_key, chain_key = jax.random.split(jax.random.PRNGKey(0), 3)
    positions, divergent, step_size, inv_mass = _ghmc_full_scan(
        jnp.zeros(2),
        warmup_key,
        init_key,
        jax.random.split(chain_key, _N_DRAWS),
        _logdensity,
        jnp.asarray(_CENTER),
        300,
        0.8,
        0.5,
        0.5,
    )
    draws = np.asarray(positions)
    assert np.isfinite(float(step_size))
    assert np.all(np.isfinite(np.asarray(inv_mass)))
    assert not np.any(np.asarray(divergent))
    assert draws.mean(axis=0) == pytest.approx(_CENTER, abs=0.25)
    assert draws.var(axis=0) == pytest.approx(np.ones(2), abs=0.3)


def _tiny_fitter():
    """One free parameter (log total mass), three bands."""
    ssp = tengri.load_ssp()
    curves = []
    for i, c in enumerate((4800.0, 6200.0, 7600.0)):
        wv = np.linspace(c - 500, c + 500, 16)
        curves.append(FilterCurve(wave=wv, trans=np.ones_like(wv), name=f"b{i}"))
    obs = Observation(photometry=Photometry(filters=tuple(curves)))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8.0, 12.0)},
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )
    pred = np.asarray(model.predict_photometry({"sfh_dpl_log_total_mass": 10.5}))
    return Fitter(model, data=pred, noise=0.05 * pred, data_type="photometry")


@pytest.mark.slow
@pytest.mark.parametrize("n_chains", [1, 2])
def test_ghmc_runs_from_cached_adaptation(n_chains):
    """A second ``mcmc_ghmc`` run reuses the cached adaptation and calls ``ghmc.init``.

    The first run tunes with MEADS and caches the step size and mass matrix; the
    second takes the cached branch, whose ``ghmc.init`` call sites are the single
    chain and the vmapped ``n_chains > 1`` paths.
    """
    fitter = _tiny_fitter()
    kw = {
        "n_warmup": 20,
        "n_burnin": 2,
        "n_samples": 5,
        "n_chains": n_chains,
        "allow_unvalidated": True,
        "verbose": False,
    }
    fitter.run("mcmc_ghmc", key=jax.random.PRNGKey(0), **kw)
    post = fitter.run("mcmc_ghmc", key=jax.random.PRNGKey(1), **kw)
    for arr in post.samples.values():
        assert np.all(np.isfinite(np.asarray(arr)))
