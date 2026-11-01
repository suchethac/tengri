# SPDX-License-Identifier: BSD-3-Clause
"""Young-fraction coordinates for the continuity SFH: the map, the density, the pushforward.

The continuity SFH is sampled either in its Student-t ratio coordinates or in the
youngest-first stick-breaking fractions. The fractions carry the ratio prior as an
exact pullback, so both coordinate systems describe one posterior. These tests
check the claims that make that true:

* the map from fractions to ratios and its inverse agree to round-off;
* the fraction log-density is the Student-t ratio log-density plus log|det J|;
* the SFH built from fractions has the stick-breaking bin masses;
* sampling the prior alone in the fraction coordinates, on the standardized
  seam, reproduces the Student-t law of every ratio, truncated or not.

The prior-only sampling runs the catalog engine's NUTS with the flat log-density
set to minus :func:`standardized_neg_log_prior`. It needs no data and no model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from jax.scipy.special import ndtr, ndtri
from jax.scipy.stats import t as jax_student_t
from scipy import stats

from tengri.components.stellar.sfh.nonparametric import DEFAULT_BIN_EDGES_GYR
from tengri.components.stellar.sfh.young_fraction import (
    YOUNG_FRACTION_NAMES,
    continuity_young_fraction,
    flat_young_fractions,
    log_width_steps,
    ratios_from_young_fractions,
    young_fraction_prior_from_edges,
    young_fractions_from_ratios,
)
from tengri.inference.loss_functions import standardized_neg_log_prior

pytestmark = pytest.mark.conservation

_EDGES = np.asarray(DEFAULT_BIN_EDGES_GYR)
_DELTA = log_width_steps(_EDGES)


def _interior_fractions(seed: int, n: int) -> np.ndarray:
    """Random fractions in the open box (0.02, 0.98)^6."""
    rng = np.random.default_rng(seed)
    return 0.02 + 0.96 * rng.random((n, 6))


def test_round_trip_fractions_to_ratios_and_back():
    """Fractions -> ratios -> fractions returns the input to 1e-12."""
    for f in _interior_fractions(seed=1, n=200):
        f = jnp.asarray(f)
        back = young_fractions_from_ratios(ratios_from_young_fractions(f, _DELTA), _DELTA)
        np.testing.assert_allclose(np.asarray(back), np.asarray(f), rtol=0, atol=1e-12)


def test_round_trip_ratios_to_fractions_and_back():
    """Ratios -> fractions -> ratios returns the input to 1e-12, fractions stay in (0, 1)."""
    rng = np.random.default_rng(2)
    for r in rng.uniform(-4.0, 4.0, size=(200, 6)):
        r = jnp.asarray(r)
        f = young_fractions_from_ratios(r, _DELTA)
        assert bool(jnp.all((f > 0) & (f < 1)))
        back = ratios_from_young_fractions(f, _DELTA)
        np.testing.assert_allclose(np.asarray(back), np.asarray(r), rtol=0, atol=1e-12)


def test_flat_fractions_map_to_zero_ratios():
    """The flat SFH (equal SFR in every bin) is the default fraction vector."""
    r = ratios_from_young_fractions(flat_young_fractions(_EDGES), _DELTA)
    np.testing.assert_allclose(np.asarray(r), 0.0, atol=1e-12)


def test_sfh_from_fractions_has_stick_breaking_bin_masses():
    """Bin k of the SFH holds f_k R_k of the total mass, with R_k the mass not yet placed."""
    f = _interior_fractions(seed=3, n=1)[0]
    log_total = 10.0
    n_grid = 200_000
    dt = _EDGES[-1] * 1e9 / n_grid
    ages = (np.arange(n_grid) + 0.5) * dt
    sfr = np.asarray(
        continuity_young_fraction(
            jnp.asarray(ages),
            log_total,
            _EDGES,
            **{f"young_frac_{k}": float(f[k]) for k in range(6)},
        )
    )
    edges_yr = _EDGES * 1e9
    bin_masses = np.array(
        [np.sum(sfr[(ages >= edges_yr[k]) & (ages < edges_yr[k + 1])]) * dt for k in range(7)]
    )
    remaining = 10.0**log_total
    expected = []
    for k in range(6):
        expected.append(f[k] * remaining)
        remaining *= 1.0 - f[k]
    expected.append(remaining)
    np.testing.assert_allclose(bin_masses, np.asarray(expected), rtol=2e-3)


def test_log_density_is_student_t_ratio_density_plus_log_jacobian():
    """The fraction log-density equals the ratio Student-t log-density plus log|det J|."""
    prior = young_fraction_prior_from_edges(_EDGES)
    for f in _interior_fractions(seed=4, n=20):
        f = jnp.asarray(f)
        jac = jax.jacfwd(lambda x: ratios_from_young_fractions(x, _DELTA))(f)
        log_det = float(jnp.linalg.slogdet(jac)[1])
        r = ratios_from_young_fractions(f, _DELTA)
        direct = float(jnp.sum(jax_student_t.logpdf(r, 2.0, 0.0, 0.3)) + log_det)
        got = float(prior.log_density_from_latents(ndtri(f)))
        np.testing.assert_allclose(got, direct, rtol=0, atol=1e-9)


def test_truncated_density_is_refused_outside_the_box():
    """A ratio beyond the truncation makes the fraction density -inf, not a floor."""
    prior = young_fraction_prior_from_edges(_EDGES, lo=-3.0, hi=3.0)
    inside = ndtri(jnp.asarray(_interior_fractions(seed=5, n=1)[0]))
    assert np.isfinite(float(prior.log_density_from_latents(inside)))
    far = jnp.array([6.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert float(jnp.max(jnp.abs(ratios_from_young_fractions(ndtr(far), _DELTA)))) > 3.0
    assert float(prior.log_density_from_latents(far)) == -np.inf


def _flat_log_prior(names, unravel, prior):
    """Flat log-density: minus the standardized negative log-prior of the latents."""

    def log_post(position, _data_args):
        latents = unravel(position)
        return -standardized_neg_log_prior(latents, names, stochastic=False, young_fraction=prior)

    return log_post


def _sample_prior_nuts(prior, *, n_chains: int, n_draws: int, n_warmup: int, seed: int):
    """Prior-only NUTS on the standardized seam, through the catalog engine.

    Each chain is one galaxy of a vmapped batch, with no data. The latents are
    ``sfh_cont_log_total_mass`` and the six young fractions.

    Parameters
    ----------
    prior : YoungFractionPrior
        The fraction prior to sample.
    n_chains : int
        Independent chains, one per vmapped galaxy.
    n_draws : int
        Draws kept per chain.
    n_warmup : int
        Warmup steps per chain.
    seed : int
        Seed for the initial positions and the sampler keys.

    Returns
    -------
    fractions, ratios : ndarray, shape (n_chains * n_draws, 6)
        Young fractions :math:`f = \\Phi(\\xi)` and ratios :math:`r(f)` of the draws.
    """
    from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine

    names = ("sfh_cont_log_total_mass", *YOUNG_FRACTION_NAMES)
    init_flat, unravel = ravel_pytree({name: jnp.zeros(()) for name in names})
    run_one, _ = build_catalog_mcmc_engine(
        None,
        "nuts",
        n_warmup=n_warmup,
        n_burnin=0,
        n_samples=n_draws,
        flat_logdensity=(
            _flat_log_prior(names, unravel, prior),
            unravel,
            init_flat,
            {"obs": jnp.zeros(())},
        ),
        substitute=lambda data, noise, presence, redshift, lfo, lfe, shared: {"obs": data},
    )
    key = jax.random.PRNGKey(seed)
    inits = jax.random.normal(key, (n_chains, init_flat.shape[0]))
    keys = jax.random.split(jax.random.fold_in(key, 1), n_chains)
    zeros = jnp.zeros((n_chains,))
    positions, _divergent = jax.vmap(run_one, in_axes=(0,) * 8 + (None,))(
        inits, keys, zeros, zeros, zeros, zeros, zeros, zeros, {"obs": jnp.zeros(())}
    )
    xi_fraction = np.asarray(positions).reshape(-1, init_flat.shape[0])[:, 1:]
    fractions = np.asarray(ndtr(jnp.asarray(xi_fraction)))
    ratios = np.asarray(
        jax.vmap(ratios_from_young_fractions, in_axes=(0, None))(jnp.asarray(fractions), _DELTA)
    )
    return fractions, ratios


def _truncated_t_cdf(x, lo, hi):
    """CDF of the Student-t(df=2, scale=0.3) restricted to ``[lo, hi]``."""
    base_lo = stats.t.cdf(lo, 2.0, scale=0.3)
    base_hi = stats.t.cdf(hi, 2.0, scale=0.3)
    return (stats.t.cdf(x, 2.0, scale=0.3) - base_lo) / (base_hi - base_lo)


@pytest.fixture(scope="module")
def prior_draws_untruncated():
    prior = young_fraction_prior_from_edges(_EDGES)
    return _sample_prior_nuts(prior, n_chains=8, n_draws=4000, n_warmup=1000, seed=7)


@pytest.fixture(scope="module")
def prior_draws_truncated():
    prior = young_fraction_prior_from_edges(_EDGES, lo=-3.0, hi=3.0)
    return _sample_prior_nuts(prior, n_chains=8, n_draws=4000, n_warmup=1000, seed=8)


def test_prior_alone_pushes_forward_to_student_t_ratios(prior_draws_untruncated):
    """Each ratio of the NUTS prior draws follows the Student-t(0, 0.3, 2) law."""
    _fractions, ratios = prior_draws_untruncated
    for k in range(6):
        ks = stats.kstest(ratios[:, k], lambda x: stats.t.cdf(x, 2.0, scale=0.3))
        assert ks.statistic < 0.02, f"ratio {k}: KS {ks.statistic:.4f}"


def test_prior_alone_pushes_forward_to_truncated_student_t_ratios(prior_draws_truncated):
    """Under the |r| <= 3 truncation the ratio draws follow the truncated Student-t."""
    _fractions, ratios = prior_draws_truncated
    assert np.all(np.abs(ratios) <= 3.0 + 1e-9)
    for k in range(6):
        ks = stats.kstest(ratios[:, k], lambda x: _truncated_t_cdf(x, -3.0, 3.0))
        assert ks.statistic < 0.02, f"ratio {k}: KS {ks.statistic:.4f}"
