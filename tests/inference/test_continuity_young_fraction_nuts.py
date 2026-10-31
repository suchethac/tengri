# SPDX-License-Identifier: BSD-3-Clause
"""Posterior agreement of the two continuity coordinate systems on one mock dwarf.

A continuity SFH sampled in its Student-t ratio coordinates and in the young-fraction
coordinates describes one posterior, so an NUTS fit of the same mock photometry in
either should give the same posterior medians of ``log_total_mass``,
``mass_weighted_age`` and ``log_sfr_young``, within Monte Carlo error.

The full comparison is slow (long chains on the wide synthetic SSP). The default
parameters here are a smoke run: a few dozen draws, which checks that both fits run,
return finite draws that move, and report the three medians. The orchestrator runs
the full comparison with :func:`compare_medians` at production draw counts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, ForwardModel, SEDModel
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.slow

_TRUTH_RATIOS = {
    "sfh_cont_log_total_mass": 9.8,
    "sfh_cont_ratio_0": 0.4,
    "sfh_cont_ratio_1": -0.2,
    "sfh_cont_ratio_2": 0.3,
    "sfh_cont_ratio_3": -0.1,
    "sfh_cont_ratio_4": 0.2,
    "sfh_cont_ratio_5": -0.3,
}
_PROPERTIES = ("mass_weighted_age_gyr", "sfr_100myr")


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
    from tengri.components.stellar.sfh.young_fraction import (
        log_width_steps,
        young_fractions_from_ratios,
    )

    edges = np.asarray(model.spec.bin_edges_gyr)
    r = jnp.asarray([_TRUTH_RATIOS[f"sfh_cont_ratio_{k}"] for k in range(6)])
    f = young_fractions_from_ratios(r, log_width_steps(edges))
    truth_yf = {"sfh_cont_log_total_mass": _TRUTH_RATIOS["sfh_cont_log_total_mass"]}
    truth_yf.update({f"sfh_cont_young_frac_{k}": float(f[k]) for k in range(6)})
    return dict(_TRUTH_RATIOS), truth_yf


def compare_medians(posterior_a, posterior_b, names=("log_total_mass", *_PROPERTIES)):
    """Medians and Monte Carlo errors of two posteriors, for the comparison table.

    Parameters
    ----------
    posterior_a, posterior_b : Posterior
        Posteriors from the ratio and the young-fraction fits.
    names : tuple of str
        Property names to compare. ``log_total_mass`` is read as ``log10`` of stellar mass.

    Returns
    -------
    dict
        ``name -> (median_a, median_b, mc_error)``, the error from the larger of the two
        standard errors of the median.
    """
    out = {}
    for name in names:
        a = np.asarray(_values(posterior_a, name))
        b = np.asarray(_values(posterior_b, name))
        se_a = np.std(a) / np.sqrt(max(a.size, 1))
        se_b = np.std(b) / np.sqrt(max(b.size, 1))
        out[name] = (float(np.median(a)), float(np.median(b)), float(max(se_a, se_b)))
    return out


def _values(posterior, name):
    """Draws of a sampled parameter (``log_total_mass``) or of a derived property."""
    if name == "log_total_mass":
        return np.asarray(posterior.samples["sfh_cont_log_total_mass"])
    if name == "sfr_100myr":
        return np.log10(np.asarray(posterior.properties[name]))
    return np.asarray(posterior.properties[name])


@pytest.fixture(scope="module")
def mock_fits(synthetic_ssp_wide, synthetic_tophat_obs):
    """Fit the same mock photometry in both coordinate systems (smoke draw counts)."""
    model_r = _build(synthetic_ssp_wide, synthetic_tophat_obs, "ratios")
    model_y = _build(synthetic_ssp_wide, synthetic_tophat_obs, "young_fraction")
    truth_r, truth_y = _truth_for(model_r)
    data = jnp.asarray(model_r.predict(truth_r).photometry())
    np.testing.assert_allclose(
        np.asarray(model_y.predict(truth_y).photometry()), np.asarray(data), rtol=1e-9
    )
    noise = 0.1 * jnp.abs(data) + 1e-30
    key = jax.random.PRNGKey(11)
    fits = {}
    for label, model in (("ratios", model_r), ("young_fraction", model_y)):
        fitter = Fitter(ForwardModel.build(sed=model), data=data, noise=noise)
        fits[label] = fitter.run("mcmc_nuts", key=key, n_warmup=60, n_samples=40)
    return fits


def test_both_coordinate_fits_run_and_move(mock_fits):
    """Both fits return finite property draws that are not constant."""
    for label, posterior in mock_fits.items():
        for name in _PROPERTIES:
            values = _values(posterior, name)
            assert np.all(np.isfinite(values)), f"{label} {name} has non-finite draws"
            assert np.unique(values).size > 1, f"{label} {name} draws do not move"


def test_medians_agree_within_monte_carlo_error(mock_fits):
    """Smoke check: the medians of both fits agree within a generous multiple of the error.

    At smoke draw counts the Monte Carlo error is large, so the bound is wide. The full
    comparison at production draw counts uses :func:`compare_medians` with a tight bound.
    """
    table = compare_medians(mock_fits["ratios"], mock_fits["young_fraction"])
    for name, (med_r, med_y, err) in table.items():
        assert abs(med_r - med_y) <= 6.0 * max(err, 1e-3) + 0.5, (name, med_r, med_y, err)
