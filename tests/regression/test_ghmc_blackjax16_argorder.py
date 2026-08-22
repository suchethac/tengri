# SPDX-License-Identifier: BSD-3-Clause
"""GHMC runs under blackjax >= 1.5 (argument-order regression).

Fresh-user audit (2026-07): ``fitter.run("mcmc_ghmc")`` crashed with
``TypeError: Expected a callable value, got JitTracer(uint32[2])``. blackjax
changed ``blackjax.mcmc.ghmc.init`` to ``init(position, logdensity_fn,
rng_key)``, but tengri still called it in the old ``(position, rng_key,
logdensity_fn)`` order, so the PRNG key was handed to blackjax where the
log-density callable was expected. This pins the fixed order end to end.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from tengri import (
    FIXED,
    FREE,
    Fitter,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    WavePrecomp,
    generate_mock,
    load_ssp_data,
)

pytestmark = [pytest.mark.regression_bug, pytest.mark.slow]

_SSP = Path("data/fsps_prsc_miles_chabrier.h5")


@pytest.mark.skipif(not _SSP.exists(), reason="needs data/fsps_prsc_miles_chabrier.h5")
def test_ghmc_runs_and_returns_finite_samples():
    ssp = load_ssp_data(str(_SSP))
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))
    key = jax.random.PRNGKey(0)
    sed = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(0.05),
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"law": "power_law", "type": "two_component", "all_params": FIXED},
        approx=WavePrecomp(),
    )
    mock = generate_mock(sed, sed.spec.sample(key), key=key, snr=30.0)
    fitter = Fitter(
        sed, np.asarray(mock["flux_obs"]), np.asarray(mock["noise"]), data_type="photometry"
    )
    post = fitter.run("mcmc_ghmc", key=key, n_samples=60)  # must not raise
    samples = post.samples
    assert samples, "GHMC returned no samples"
    for name, arr in samples.items():
        assert np.all(np.isfinite(np.asarray(arr))), f"non-finite GHMC samples for {name}"
