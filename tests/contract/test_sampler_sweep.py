# SPDX-License-Identifier: BSD-3-Clause
"""Class sweep: every sampler works in every engine, with every setting.

One sampler driver (``tengri.inference.sampling``) serves every inference engine, so a
setting is either honored by every sampler in every engine or it is a bug of the whole
class. The sweep crosses engine x sampler x ``n_chains`` x ``metric`` on a linear-Gaussian
target whose posterior is known analytically. Samplers are enumerated from the registries,
never from a hand list, so a sampler added later enters the sweep without a test edit.

Per cell: no warning that a setting was ignored or is unsupported, finite draws, and the
accuracy bar for the sampler's declared fidelity. The mean-field warning (it drops
cross-level correlations) is expected and asserted.

The bar for a sampler that draws the posterior (``fidelity="exact"``) accounts for Monte
Carlo error instead of fixing a tolerance in posterior standard deviations. With the pooled
effective sample size ESS of each coordinate, measured from the draws (per-chain initial
positive sequence of the autocorrelation, summed over chains):

* ESS >= 100 for every coordinate, of the draw and of its squared deviation, so a chain
  that does not mix cannot pass on the width of its own error bar;
* |sample mean - analytic mean| <= 4 sd / sqrt(ESS), with the analytic sd;
* |sample sd / analytic sd - 1| <= 4 / sqrt(2 ESS_sq), the 4-sigma band of a Gaussian
  variance estimate from ESS_sq independent draws, ESS_sq the effective sample size of the
  squared deviations from the analytic mean (the quantity the variance averages);
* split R-hat < 1.1 with four chains.

A mean-field sampler (``fidelity="mean_only"``) must reach the mean within 0.2 sd, a bound
on the optimizer rather than on Monte Carlo error; a point estimate must be finite. Every
key is fixed, and ``test_a_cell_is_bitwise_reproducible`` pins that two runs of one cell
return identical draws.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

_DIM = 3
_NOISE = 0.5
_A = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0], [1.0, 1.0, 1.0]])
_PREC = _A.T @ _A / _NOISE**2 + np.eye(_DIM)
_COV = np.linalg.inv(_PREC)
_SD = np.sqrt(np.diag(_COV))
_Y = np.stack([_A @ np.array([0.4, -0.3, 0.2]), _A @ np.array([-0.5, 0.6, 0.1])])

_N_WARMUP, _N_BURNIN, _N_SAMPLES = 300, 50, 4000
_IGNORED = ("ignored", "not supported", "unsupported", "not available")


def _analytic_mean(y):
    return _COV @ (_A.T @ y / _NOISE**2)


def _ld(x, data_args):
    """Log posterior of the linear-Gaussian target: iid N(0, 1) prior, Gaussian data term."""
    resid = (data_args["y"] - jnp.asarray(_A) @ x) / _NOISE
    return -0.5 * jnp.sum(resid**2) - 0.5 * jnp.sum(x**2)


def _split_rhat(chains):
    """Split R-hat per coordinate of ``chains`` shape (n_chains, n_draws, D)."""
    half = chains.shape[1] // 2
    halves = np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)
    n = halves.shape[1]
    within = halves.var(axis=1, ddof=1).mean(axis=0)
    between = n * halves.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((n - 1) / n * within + between / n) / within)


def _ess(chains):
    """Pooled effective sample size per coordinate of ``chains`` (n_chains, n_draws, D).

    Per chain, the autocorrelation sum is truncated at the first non-positive pair of lags
    (Geyer's initial positive sequence); the chains' ESS add.
    """
    total = np.zeros(chains.shape[-1])
    for chain in chains:
        n = chain.shape[0]
        centered = chain - chain.mean(axis=0)
        spec = np.fft.rfft(centered, n=2 * n, axis=0)
        acov = np.fft.irfft(spec * np.conj(spec), axis=0)[:n] / n
        rho = acov / np.where(acov[0] > 0, acov[0], 1.0)
        for j in range(chain.shape[-1]):
            pairs = rho[: (n // 2) * 2, j].reshape(-1, 2).sum(axis=1)
            stop = np.argmax(pairs <= 0) if np.any(pairs <= 0) else len(pairs)
            tau = max(-1.0 + 2.0 * pairs[:stop].sum(), 1.0)
            total[j] += n / tau
    return total


def _catalog_samplers():
    from tengri.inference.sampling.driver import get_sampler, sampler_names

    return sorted(
        {get_sampler(n).engine or n for n in sampler_names() if get_sampler(n).vmappable}
    )


def _driver_samplers():
    from tengri.inference.sampling.driver import sampler_names

    return sorted(sampler_names())


def _run_catalog(sampler, n_chains, metric):
    """Two galaxies through the vmapped catalog engine; returns (n_gal, n_chains, n, D)."""
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
        flat_logdensity=(_ld, lambda x: x, jnp.zeros(_DIM), {"y": jnp.zeros(4)}),
        substitute=lambda data, *rest: {"y": data},
    )
    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    pos, _div = jax.vmap(run_one, in_axes=(None, 0, 0, None, None, None, None, None, None))(
        jnp.zeros(_DIM), keys, jnp.asarray(_Y), None, None, None, None, None, None
    )
    pos = np.asarray(pos)
    # ghmc returns every chain of its MEADS ensemble, a minimum of ``n_chains``.
    returned = ghmc_ensemble_size(_DIM, n_chains) if sampler == "ghmc" else n_chains
    return [pos[g].reshape(returned, -1, _DIM) for g in range(2)], _Y


def _run_hierarchical(method, n_chains, metric, monkeypatch):
    """The hierarchical engine on a stubbed flat problem; returns ([(n_chains, n, D)], y)."""
    from tengri.inference import _hierarchical_flat as hf
    from tengri.inference.sampling.target import LatentLayout

    y = jnp.asarray(_Y[0])
    layout = LatentLayout.vector(_DIM)

    def log_lik(flat, data_args):
        return -0.5 * jnp.sum(((data_args["y"] - jnp.asarray(_A) @ flat) / _NOISE) ** 2)

    def log_prior(flat):
        return -0.5 * jnp.sum(flat**2)

    data_args = {"y": y}
    prob = hf.FlatProblem(
        init_flat=jnp.zeros(_DIM),
        unravel=layout.unravel,
        n_dim=_DIM,
        log_likelihood=lambda f: log_lik(f, data_args),
        log_prior=log_prior,
        log_prob=lambda f: log_lik(f, data_args) + log_prior(f),
        prior_transform=lambda c: jax.scipy.special.ndtri(jnp.clip(c, 1e-12, 1 - 1e-12)),
        extract_shared=lambda f: f[:2],
        data_args=data_args,
        log_prob_with_data=lambda f, a: log_lik(f, a) + log_prior(f),
        log_likelihood_with_data=log_lik,
    )
    monkeypatch.setattr(hf, "build_flat_problem", lambda *a, **k: prob)

    class _Fitter:
        n_galaxies = 1

    from tengri.inference.sampling.driver import get_sampler

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
    )
    shared = post.shared_samples
    draws = np.stack([np.asarray(shared["psd_sigma"]), np.asarray(shared["psd_tau_myr"])], axis=-1)
    return [draws.reshape(n_chains, -1, 2)], _Y[:1]


_ESS_FLOOR = 100.0


def _check(chains, y, fidelity, n_chains, ncoord):
    flat = np.concatenate([c.reshape(-1, c.shape[-1]) for c in chains])
    assert np.all(np.isfinite(flat)), "non-finite draws"
    if fidelity == "point":
        return
    for chain, yy in zip(chains, y, strict=True):
        mean = _analytic_mean(yy)[:ncoord]
        sd = _SD[:ncoord]
        draws = chain.reshape(-1, ncoord)
        if fidelity == "mean_only":
            np.testing.assert_array_less(np.abs(draws.mean(0) - mean) / sd, 0.2)
            continue
        ess = _ess(chain)
        ess_sq = _ess((chain - mean) ** 2)
        floor = np.minimum(ess, ess_sq)
        assert np.all(floor >= _ESS_FLOOR), f"ESS {np.round(ess)}, {np.round(ess_sq)}"
        np.testing.assert_array_less(np.abs(draws.mean(0) - mean), 4.0 * sd / np.sqrt(ess))
        np.testing.assert_array_less(np.abs(draws.std(0) / sd - 1.0), 4.0 / np.sqrt(2.0 * ess_sq))
        if n_chains == 4 and chain.shape[1] > 20:
            assert np.all(_split_rhat(chain) < 1.1), _split_rhat(chain)


def _cells(engine, names):
    return [(engine, n, c, m) for n in names for c in (1, 4) for m in ("diag", "dense")]


def _marks(engine, cell=None):
    if engine in ("single", "batched"):
        return [
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "single-galaxy and batched-spectroscopy engines not yet on the shared driver"
                ),
                raises=NotImplementedError,
            )
        ]
    return [pytest.mark.slow]


def _params():
    cells = (
        _cells("catalog", _catalog_samplers())
        + _cells("hierarchical", _driver_samplers())
        + _cells("single", _driver_samplers())
        + _cells("batched", _driver_samplers())
    )
    return [
        pytest.param(*cell, id="-".join(map(str, cell)), marks=_marks(cell[0], cell))
        for cell in cells
    ]


@pytest.mark.parametrize(("engine", "sampler", "n_chains", "metric"), _params())
def test_every_sampler_runs_in_every_engine_with_every_setting(
    engine, sampler, n_chains, metric, monkeypatch
):
    from tengri.inference.sampling.driver import get_sampler

    if engine in ("single", "batched"):
        raise NotImplementedError(f"engine {engine!r} is not on the shared driver yet")
    fidelity = "exact" if engine == "catalog" else get_sampler(sampler).fidelity
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if engine == "catalog":
            chains, y = _run_catalog(sampler, n_chains, metric)
            ncoord = _DIM
        else:
            chains, y = _run_hierarchical(sampler, n_chains, metric, monkeypatch)
            ncoord = 2
    texts = [str(w.message) for w in caught]
    assert not [t for t in texts if any(k in t.lower() for k in _IGNORED)], texts
    meanfield = [t for t in texts if "mean-field drops cross-level correlations" in t]
    if engine == "hierarchical" and sampler == "vi_meanfield":
        assert len(meanfield) == 1, texts
    else:
        assert not meanfield, texts
    _check(chains, y, fidelity, n_chains, ncoord)


def test_the_per_engine_allow_lists_are_gone():
    from tengri.inference import _hierarchical_flat as hf
    from tengri.inference.backends.mcmc import catalog

    assert not hasattr(hf, "FLAT_UNSUPPORTED")
    assert not hasattr(catalog, "_SAMPLERS")


def test_every_flat_method_is_derived_from_the_sampler_registry():
    from tengri.inference._hierarchical_flat import FLAT_SAMPLERS
    from tengri.inference.sampling.driver import sampler_names

    assert set(FLAT_SAMPLERS.values()) == set(sampler_names())


@pytest.mark.parametrize("engine", ["catalog", "hierarchical"])
def test_a_cell_is_bitwise_reproducible(engine, monkeypatch):
    """Two runs of one cell, same keys, return identical draws: no unseeded source."""
    runs = []
    for _ in range(2):
        if engine == "catalog":
            chains, _y = _run_catalog("chees", 4, "dense")
        else:
            chains, _y = _run_hierarchical("mclmc", 4, "dense", monkeypatch)
        runs.append(np.concatenate([c.reshape(-1) for c in chains]))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_every_sampler_is_an_attribute_of_one_backend_entry():
    """The backend registry is the one registry: a sampler names the entry that carries it."""
    from tengri.inference._backend_registry import all_backends
    from tengri.inference.sampling.driver import method_map, sampler_names
    from tengri.inference.sampling.samplers import SAMPLERS

    carried = {e.sampler.name for e in all_backends() if e.sampler is not None}
    assert carried == set(SAMPLERS) == set(sampler_names())
    for entry in all_backends():
        if entry.sampler is not None and entry.name != "mcmc":
            assert entry.sampler.method == entry.name
    assert set(method_map()) == {e.name for e in all_backends() if e.sampler is not None}
