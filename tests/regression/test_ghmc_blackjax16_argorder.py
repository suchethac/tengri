# SPDX-License-Identifier: BSD-3-Clause
"""GHMC call sites match the installed blackjax signature (argument-order regression).

``blackjax.mcmc.ghmc.init`` takes ``(position, logdensity_fn, rng_key)`` in
blackjax >= 1.5 and took ``(position, rng_key, logdensity_fn)`` before, which
handed the PRNG key to blackjax where the log-density callable was expected
(``TypeError: Expected a callable value, got JitTracer(uint32[2])``). tengri
passes ``position``, ``logdensity_fn`` and ``rng_key`` by keyword. These tests
run tengri's GHMC scan on a two-dimensional Gaussian log-density, with no
model and no ``Fitter``, so they need neither a data file nor the
``allow_unvalidated`` escape hatch of the broken-tier ``mcmc_ghmc`` backend.
"""

from __future__ import annotations

import inspect

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.backends.mcmc._shared import _get_ghmc_kernel, _ghmc_full_scan

pytestmark = pytest.mark.regression_bug


def _logdensity(position, data_args):
    return -0.5 * jnp.sum((position - data_args) ** 2)


def test_ghmc_init_parameters_are_the_keywords_tengri_passes():
    """``ghmc.init`` accepts ``position``, ``logdensity_fn`` and ``rng_key`` by name."""
    names = set(inspect.signature(blackjax.mcmc.ghmc.init).parameters)
    assert {"position", "logdensity_fn", "rng_key"} <= names


def test_ghmc_scan_runs_and_returns_finite_samples_on_a_gaussian():
    """tengri's GHMC scan runs end to end on a unit Gaussian centred at (1, -2).

    The sample mean of 400 draws after 100 warmup steps lies within 0.5 of the
    centre per axis (the sample standard error is about 0.05 for independent
    draws; 0.5 admits the autocorrelation of a short GHMC chain and excludes a
    chain stuck at the origin start, 1 and 2 away).
    """
    key = jax.random.PRNGKey(0)
    warmup_key, init_key, chain_key = jax.random.split(key, 3)
    centre = jnp.asarray([1.0, -2.0])
    positions, divergent, step_size, inv_mass = _ghmc_full_scan(
        jnp.zeros(2),
        warmup_key,
        init_key,
        jax.random.split(chain_key, 400),
        _logdensity,
        centre,
        100,
        0.8,
        0.5,
        0.5,
    )
    assert np.all(np.isfinite(np.asarray(positions)))
    assert np.isfinite(float(step_size))
    assert np.all(np.isfinite(np.asarray(inv_mass)))
    assert divergent.shape == (400,)
    assert np.asarray(positions).mean(axis=0) == pytest.approx(np.asarray(centre), abs=0.5)


def test_ghmc_kernel_positional_order_matches_tengri_call():
    """The kernel's positional parameters follow ``(rng_key, state, logdensity_fn, ...)``."""
    names = list(inspect.signature(_get_ghmc_kernel()).parameters)
    assert names[:3] == ["rng_key", "state", "logdensity_fn"]
