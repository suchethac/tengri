# SPDX-License-Identifier: BSD-3-Clause
"""Wiring proofs for the sweep: a correlated, non-isotropic target where the metric matters.

The stub targets of ``test_sampler_sweep.py`` are close to isotropic, so a dense metric that
is silently ignored, or a flat vector read back in the wrong order, would still pass them.
Here the posterior is ``N(0, Sigma)`` on two coordinates with sd (1, 0.25) and correlation
0.95, written on the PERMUTED vector ``(b, a)``:

* ``test_dense_metric_pays_for_itself``: for every chain sampler the dense layer applies to
  (diagonal metric adapts only per coordinate), ESS per draw with ``metric="dense"`` exceeds
  the diagonal one by at least ``_DENSE_GAIN``, in the vmapped catalog engine and the
  hierarchical engine alike, and both agree with the analytic posterior under the ESS bars of
  the main sweep.
* ``test_the_posterior_covariance_survives_the_flat_seam``: the nine samplers the flat seam
  used to refuse return all three entries of the covariance within their Monte Carlo error,
  so a ravel / unravel mix-up (which swaps the two sd's and flips nothing in the mean) fails.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.sampling.driver import get_sampler, sampler_names
from tests.contract.test_sampler_sweep import _ess, _split_rhat

pytestmark = pytest.mark.contract

_SD = np.array([1.0, 0.25])
_RHO = 0.95
_COV = np.array([[1.0, _RHO], [_RHO, 1.0]]) * np.outer(_SD, _SD)
#: Log density on the permuted vector (b, a): the sampled order differs from the natural one.
_PERM = np.array([1, 0])
_PREC = np.linalg.inv(_COV[np.ix_(_PERM, _PERM)])

_N_WARMUP, _N_BURNIN, _N_SAMPLES = 400, 100, 4000
#: Stated factor: dense ESS per draw over diagonal. Measured gains are 1.3 to 82 (NUTS 4.1, dynamic
#: HMC 6.7, Barker 38, MALA 33, ChEES 1.65; HMC 1.3 because its diagonal arm already reaches
#: 0.41 ESS per draw); the bar sits below the smallest unsaturated one, so a dense layer that
#: does nothing (factor 1) fails and noise does not.
_DENSE_GAIN = 1.4


def _ld(x, args):
    return -0.5 * x @ jnp.asarray(_PREC) @ x


def _catalog(sampler, n_chains, metric):
    from tengri.inference.backends.mcmc.catalog import (
        build_catalog_mcmc_engine,
        ghmc_ensemble_size,
    )

    run_one, _ = build_catalog_mcmc_engine(
        None,
        sampler,
        n_warmup=_N_WARMUP,
        n_burnin=_N_BURNIN,
        n_samples=_N_SAMPLES,
        n_chains=n_chains,
        metric=metric,
        flat_logdensity=(_ld, lambda x: x, jnp.zeros(2), {"y": jnp.zeros(1)}),
        substitute=lambda data, *rest: {"y": data},
    )
    positions, _ = jax.jit(run_one)(
        jnp.zeros(2), jax.random.PRNGKey(3), jnp.zeros(1), None, None, None, None, None, None
    )
    # ghmc returns every chain of its MEADS ensemble, a minimum of ``n_chains``.
    returned = ghmc_ensemble_size(2, n_chains) if sampler == "ghmc" else n_chains
    return np.asarray(positions).reshape(returned, -1, 2)[..., ::-1]


def _hierarchical(method, n_chains, metric, monkeypatch, **settings):
    from tengri.inference import _hierarchical_flat as hf
    from tengri.inference.sampling.target import LatentLayout

    layout = LatentLayout.vector(2)

    def log_prior(flat):
        return -0.5 * jnp.sum(flat**2)

    def log_lik_with_data(flat, args):
        return _ld(flat, args) - log_prior(flat)

    prob = hf.FlatProblem(
        init_flat=jnp.zeros(2),
        unravel=layout.unravel,
        n_dim=2,
        log_likelihood=lambda f: log_lik_with_data(f, None),
        log_prior=log_prior,
        log_prob=lambda f: _ld(f, None),
        prior_transform=lambda c: jax.scipy.special.ndtri(jnp.clip(c, 1e-12, 1 - 1e-12)),
        # The flat vector is (b, a); the seam reports (psd_sigma, psd_tau) = coordinates 0, 1.
        extract_shared=lambda f: f[:2],
        data_args={},
        log_prob_with_data=lambda f, a: _ld(f, a),
        log_likelihood_with_data=log_lik_with_data,
    )
    monkeypatch.setattr(hf, "build_flat_problem", lambda *a, **k: prob)

    class _Fitter:
        n_galaxies = 1

    post = hf.run_flat_sampler(
        _Fitter(),
        get_sampler(method).method,
        key=jax.random.PRNGKey(5),
        n_warmup=_N_WARMUP,
        n_burnin=_N_BURNIN,
        n_samples=_N_SAMPLES,
        n_chains=n_chains,
        metric=metric,
        allow_unvalidated=True,
        verbose=False,
        **settings,
    )
    shared = post.shared_samples
    draws = np.stack([np.asarray(shared["psd_sigma"]), np.asarray(shared["psd_tau_myr"])], axis=-1)
    return draws.reshape(n_chains, -1, 2)[..., ::-1]


def _ess_per_draw(chains):
    return float(_ess(chains).min() / (chains.shape[0] * chains.shape[1]))


def _applies(name):
    sampler = get_sampler(name)
    return sampler.kind == "chain" and sampler.metric_mode in ("layer", "native", "internal")


def _catalog_names():
    return sorted(
        {get_sampler(n).engine or n for n in sampler_names() if get_sampler(n).vmappable}
    )


def _hierarchical_names():
    skip = {"nuts_pathfinder", "ess", "nuts_fast"}
    return sorted(n for n in sampler_names() if _applies(n) and n not in skip)


def _agrees(chains):
    mean_bar = 4.0 * _SD / np.sqrt(_ess(chains))
    assert np.all(np.abs(chains.reshape(-1, 2).mean(0)) < mean_bar), chains.reshape(-1, 2).mean(0)
    ratio = chains.reshape(-1, 2).std(0) / _SD
    sq = np.stack(
        [_ess((chains[..., j] - 0.0)[..., None] ** 2).min(keepdims=False) for j in (0, 1)]
    )
    assert np.all(np.abs(ratio - 1.0) < 4.0 / np.sqrt(2.0 * sq)), (ratio, sq)
    assert np.all(_split_rhat(chains[..., :1]) < 1.1) if chains.shape[0] > 1 else True


#: ESS per draw at which a sampler has little headroom left to gain (antithetic draws exceed
#: it), so the gain factor is not asked of it.
_SATURATED = 0.5

#: Cells exempt from the gain factor, with the measurement: MEADS's per-fold scale and
#: partially refreshed momentum put the diagonal and the dense arm of the catalog GHMC at
#: 0.210 and 0.221 ESS per draw (gain 1.05), so whitening gives it nothing it can use. The
#: layer still runs: ``test_the_dense_layer_changes_the_catalog_ghmc_draws``. Low-rank HMC learns
#: a metric of rank 10 on D = 2, which already spans the posterior: both arms sit at 0.225.
_NO_HEADROOM = {("catalog", "ghmc"), ("hierarchical", "hmc_lowrank")}


def _gain_ok(engine, sampler, diag, dense):
    gain = _ess_per_draw(dense) / _ess_per_draw(diag)
    ok = gain >= _DENSE_GAIN or _ess_per_draw(dense) >= _SATURATED
    return ok or (engine, sampler) in _NO_HEADROOM, gain


@pytest.mark.slow
@pytest.mark.parametrize("sampler", _catalog_names())
def test_dense_metric_pays_for_itself_in_the_catalog_engine(sampler):
    diag = _catalog(sampler, 1, "diag")
    dense = _catalog(sampler, 1, "dense")
    _agrees(dense)
    if _applies(sampler):
        ok, gain = _gain_ok("catalog", sampler, diag, dense)
        assert ok, (sampler, gain, _ess_per_draw(diag), _ess_per_draw(dense))


def test_the_dense_layer_changes_the_catalog_ghmc_draws():
    assert not np.array_equal(_catalog("ghmc", 1, "diag"), _catalog("ghmc", 1, "dense"))


@pytest.mark.slow
@pytest.mark.parametrize("sampler", _hierarchical_names())
def test_dense_metric_pays_for_itself_in_the_hierarchical_engine(sampler, monkeypatch):
    diag = _hierarchical(sampler, 1, "diag", monkeypatch)
    dense = _hierarchical(sampler, 1, "dense", monkeypatch)
    _agrees(dense)
    ok, gain = _gain_ok("hierarchical", sampler, diag, dense)
    assert ok, (sampler, gain, _ess_per_draw(diag), _ess_per_draw(dense))


_NINE = (
    "mcmc_nuts_fast",
    "mcmc_chees",
    "mcmc_smc",
    "hmc_is",
    "vi_fullrank",
    "vi_meanfield",
    "mcmc_barker",
    "mcmc_mala",
    "mcmc_hmc_lowrank",
)


@pytest.mark.slow
@pytest.mark.parametrize("method", _NINE)
def test_the_posterior_covariance_survives_the_flat_seam(method, monkeypatch):
    """All three covariance entries, not only the mean: a swapped coordinate order fails."""
    settings = {"smc_n_particles": 2000, "vi_n_steps": 4000, "is_n_draws": 4000}
    chains = _hierarchical(method, 2, "diag", monkeypatch, **settings)
    flat = chains.reshape(-1, 2)
    cov = np.cov(flat.T)
    if method == "vi_meanfield":
        # A diagonal Gaussian has no off-diagonal; its variances are conditional ones.
        assert np.all(np.abs(flat.mean(0)) < 0.2 * _SD)
        assert cov[0, 0] > cov[1, 1], "the order of the coordinates is swapped"
        return
    ess = _ess(chains)
    bar = 4.0 * np.sqrt((np.outer(_SD**2, _SD**2) + _COV**2) / ess.min())
    assert np.all(np.abs(cov - _COV) < bar), (cov, _COV, bar)


@pytest.mark.slow
def test_nuts_fast_samples_the_correlated_posterior_with_its_dense_metric(monkeypatch):
    """``nuts_fast`` is dense by preset: there is no diagonal arm to gain over, only accuracy."""
    _agrees(_hierarchical("nuts_fast", 1, "diag", monkeypatch))
