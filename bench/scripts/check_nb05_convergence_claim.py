#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Does ``05_fitting_photometry`` still meet the convergence claim it ships?

The notebook builds a D=8 posterior, fits it with NUTS, and prints split-R-hat
under the bar every other sampler in ``bench/`` is measured against: **R-hat <
1.01 with 0 divergences**. That claim is a *rendered output*, not an assertion,
so nothing fails when it stops being true.

This script is that assertion, standalone: it reproduces the notebook's model,
mock (``PRNGKey(7)``, SNR 20) and fit call exactly -- same warmup, same
``n_burnin=0``, same short MAP seed -- and prints the diagnostics. It exists
because the GHMC/MEADS campaign of 2026-08-30 needed a NUTS baseline and found
one that does not reproduce
``bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md``'s published row (R-hat
1.0033, 0 divergences, min ESS 144.2 there). Whether that is model drift or a
sampler regression is not settled; see the "One thing this report could not
settle" section of ``bench/reports/2026-08-30_ghmc_meads_adaptation.md``.

Note the MAP seed matters and not in the obvious direction: the notebook's own
200-step MAP gives a *better*-mixing NUTS run than the benchmark harness's
``n_restarts=8, n_steps=800`` one.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/check_nb05_convergence_claim.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import jax
import numpy as np

import tengri
from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
    generate_mock,
)
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

FILTERS = (
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
    "wise_w1",
    "wise_w2",
    "wise_w3",
    "wise_w4",
)

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)
obs = Observation(photometry=Photometry.from_names(list(FILTERS)))
sed_model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    approx=WavePrecomp(),
    sfh=builders.sfh.tsnorm(all_params=FREE),
    dust_attenuation=builders.dust.two_component(
        all_params=Fixed(DEFAULT),
        law="calzetti",
        tau_bc=Uniform(0.0, 1.0),
        tau_diff=Uniform(0.0, 1.0),
    ),
    dust_emission=builders.dust.emission.modified_blackbody(all_params=Fixed(DEFAULT)),
    neb=builders.neb.none(),
    met={"logzsol": Uniform(-1.5, 0.3)},
    redshift=Fixed(0.05),
)
forward = ForwardModel.build(sed=sed_model)

key = jax.random.PRNGKey(7)
key_truth, key_mock, key_fit = jax.random.split(key, 3)
truth = sed_model.spec.sample(key_truth)
mock = generate_mock(sed_model, truth, key=key_mock, snr=20.0)
flux_obs = np.asarray(mock["flux_obs"])
noise = np.asarray(mock["noise"])

forward.fit(flux_obs, noise, method="map", key=key_fit, n_steps=200)
t = time.perf_counter()
posterior = forward.fit(
    flux_obs,
    noise,
    method="mcmc_nuts",
    key=key_fit,
    n_warmup=600,
    n_samples=600,
    n_chains=2,
    n_burnin=0,
)
wall = time.perf_counter() - t
rh = posterior.rhat()
ess = effective_sample_size({k: np.asarray(v) for k, v in posterior.samples.items()})
fin = [(k, v["ess"]) for k, v in ess.items() if np.isfinite(v["ess"])]
worst = min(fin, key=lambda p: p[1])
print(
    f"notebook-05-as-committed: wall {wall:.1f}s  maxRhat {max(rh.values()):.4f}  "
    f"div {posterior.diagnostics.get('n_divergent')}  minESS {worst[1]:.1f} ({worst[0]})"
)
print("per-parameter R-hat:", {k: round(float(v), 4) for k, v in rh.items()})
