# SPDX-License-Identifier: BSD-3-Clause
"""Posterior agreement of the two continuity coordinate systems on one mock dwarf.

A continuity SFH sampled in its Student-t ratio coordinates and in the young-fraction
coordinates describes one posterior. An NUTS fit of the same mock photometry in either
should therefore give the same posterior distribution of ``log_total_mass``,
``mass_weighted_age_gyr`` and ``sfr_100myr`` (log10), so the medians of the two fits
must agree within Monte Carlo error.

The Monte Carlo error is ESS-based. The standard error of a median is about
``1.2533 * sd / sqrt(ESS)``, with ESS summed over the chains from Sokal's
autocorrelation time. The criterion is ``|median_a - median_b| / sqrt(se_a**2 + se_b**2) < 3``
for each quantity. The full run uses :data:`N_WARMUP`, :data:`N_SAMPLES` and
:data:`N_CHAINS`. A smoke run at a few dozen draws only checks that both fits run and move.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, ForwardModel, SEDModel
from tengri.analysis.diagnostics.autocorrelation import autocorrelation_time_combined
from tengri.components.stellar.sfh.young_fraction import (
    log_width_steps,
    young_fractions_from_ratios,
)
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.slow

#: Production draw counts for the full comparison.
N_WARMUP = 1000
N_SAMPLES = 3000
N_CHAINS = 4

#: Criterion: the two medians agree within this many ESS-based standard errors.
N_SIGMA = 3.0

#: Quantities compared, as posterior-draw names.
COMPARED = ("log_total_mass", "mass_weighted_age_gyr", "sfr_100myr")

_TRUTH_RATIOS = {
    "sfh_cont_log_total_mass": 9.8,
    "sfh_cont_ratio_0": 0.4,
    "sfh_cont_ratio_1": -0.2,
    "sfh_cont_ratio_2": 0.3,
    "sfh_cont_ratio_3": -0.1,
    "sfh_cont_ratio_4": 0.2,
    "sfh_cont_ratio_5": -0.3,
}


def _build(ssp, observation, coordinates):
    """Continuity model with all shape parameters free, in the given coordinates."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "continuity", "coordinates": coordinates, "all_params": FREE},
        met={"logzsol": Fixed(-0.5), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


def _truth_for(model):
    """The ratio truth, and the same SFH written as young fractions of the model's ladder."""
    edges = np.asarray(model.spec.bin_edges_gyr)
    r = jnp.asarray([_TRUTH_RATIOS[f"sfh_cont_ratio_{k}"] for k in range(6)])
    f = young_fractions_from_ratios(r, log_width_steps(edges))
    truth_yf = {"sfh_cont_log_total_mass": _TRUTH_RATIOS["sfh_cont_log_total_mass"]}
    truth_yf.update({f"sfh_cont_young_frac_{k}": float(f[k]) for k in range(6)})
    return dict(_TRUTH_RATIOS), truth_yf


def _values(posterior, name):
    """Pooled draws of a sampled parameter or a derived quantity, as the compared scale."""
    if name == "log_total_mass":
        return np.asarray(posterior.samples["sfh_cont_log_total_mass"])
    if name == "sfr_100myr":
        return np.log10(np.asarray(posterior.properties["sfr_100myr"]))
    return np.asarray(posterior.properties[name])


def _ess_and_se(draws, n_chains):
    """ESS summed over chains, and the standard error of the median.

    Parameters
    ----------
    draws : ndarray, shape (n_chains * n_samples,)
        Pooled draws, chain-major.
    n_chains : int
        Number of chains the draws came from.

    Returns
    -------
    ess : float
        Sum over chains of the per-chain Sokal effective sample size.
    se_median : float
        ``1.2533 * sd / sqrt(ess)``, the asymptotic standard error of the median.
    """
    chains = draws.reshape(n_chains, -1)
    ess = float(sum(autocorrelation_time_combined(chain)["ess"] for chain in chains))
    se_median = 1.2533 * float(np.std(draws)) / np.sqrt(max(ess, 1.0))
    return ess, se_median


def compare_medians(posterior_a, posterior_b, n_chains=N_CHAINS, names=COMPARED):
    """ESS-based comparison of the posterior medians of two fits.

    Parameters
    ----------
    posterior_a, posterior_b : Posterior
        Posteriors from the ratio and the young-fraction fits.
    n_chains : int, optional
        Chains each posterior was run with, chain-major in the draws.
    names : tuple of str, optional
        Quantities to compare.

    Returns
    -------
    dict
        ``name -> dict`` with the two medians, their ESS and standard errors, and
        ``ratio = |median_a - median_b| / sqrt(se_a**2 + se_b**2)``.
    """
    table = {}
    for name in names:
        a = _values(posterior_a, name)
        b = _values(posterior_b, name)
        ess_a, se_a = _ess_and_se(a, n_chains)
        ess_b, se_b = _ess_and_se(b, n_chains)
        diff = abs(float(np.median(a)) - float(np.median(b)))
        table[name] = {
            "median_ratios": float(np.median(a)),
            "median_young": float(np.median(b)),
            "ess_ratios": ess_a,
            "ess_young": ess_b,
            "se_ratios": se_a,
            "se_young": se_b,
            "ratio": diff / np.sqrt(se_a**2 + se_b**2),
        }
    return table


def fit_both(ssp, observation, *, n_warmup, n_samples, n_chains=N_CHAINS, seed=11):
    """NUTS fits of the same mock photometry in both coordinate systems.

    Parameters
    ----------
    ssp, observation
        Synthetic SSP and photometric observation.
    n_warmup, n_samples, n_chains : int
        NUTS settings, per chain.
    seed : int, optional
        Key seed.

    Returns
    -------
    dict
        ``{"ratios": Posterior, "young_fraction": Posterior}``.
    """
    model_r = _build(ssp, observation, "ratios")
    model_y = _build(ssp, observation, "young_fraction")
    truth_r, truth_y = _truth_for(model_r)
    data = jnp.asarray(model_r.predict(truth_r).photometry())
    np.testing.assert_allclose(
        np.asarray(model_y.predict(truth_y).photometry()), np.asarray(data), rtol=1e-9
    )
    noise = 0.1 * jnp.abs(data) + 1e-30
    key = jax.random.PRNGKey(seed)
    posteriors = {}
    for label, model in (("ratios", model_r), ("young_fraction", model_y)):
        fitter = Fitter(ForwardModel.build(sed=model), data=data, noise=noise)
        posteriors[label] = fitter.run(
            "mcmc_nuts", key=key, n_chains=n_chains, n_warmup=n_warmup, n_samples=n_samples
        )
    return posteriors


@pytest.fixture(scope="module")
def production_fits(synthetic_ssp_wide, synthetic_tophat_obs):
    """Both fits at the production draw counts."""
    return fit_both(
        synthetic_ssp_wide, synthetic_tophat_obs, n_warmup=N_WARMUP, n_samples=N_SAMPLES
    )


def test_both_coordinate_fits_run_and_move(production_fits):
    """Both fits return finite draws of every compared quantity that are not constant."""
    for label, posterior in production_fits.items():
        for name in COMPARED:
            values = _values(posterior, name)
            assert np.all(np.isfinite(values)), f"{label} {name} has non-finite draws"
            assert np.unique(values).size > 1, f"{label} {name} draws do not move"


def test_medians_agree_within_three_ess_sigma(production_fits):
    """The two coordinate systems give medians within 3 ESS-based standard errors."""
    table = compare_medians(production_fits["ratios"], production_fits["young_fraction"])
    for name, row in table.items():
        assert row["ratio"] < N_SIGMA, (name, row)
