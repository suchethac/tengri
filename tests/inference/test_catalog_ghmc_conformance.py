# SPDX-License-Identifier: BSD-3-Clause
"""MEADS ("ghmc") catalog posteriors against the NUTS catalog run, on the batched mock.

Two galaxies of the synthetic catalog are fitted twice: once by the catalog NUTS sampler
and once by the MEADS sampler with 32 chains in 8 superchains. The posterior medians of
every free parameter must agree within one tenth of the NUTS 16-84 % width, and the
sampler's divergence counts must be finite non-negative integers.

The draw counts are the module constants below. The conformance run is the same helper
at smaller counts in the smoke tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.inference.batched_spectro import fit_spectra_catalog_mcmc
from tengri.observation.batched import build_spectro_batches
from tests.contract.test_batched_spectro_loglik import _mock_catalog, _record
from tests.contract.test_batched_spectrum_forward import _banded, _free_z_model

pytestmark = pytest.mark.contract

#: Galaxies of the synthetic catalog used here, in record order.
CONFORMANCE_RECORDS = (0, 2)

#: Full-run counts for the MEADS sampler (32 chains, 8 superchains).
GHMC_CHAINS = 32
GHMC_SUPERCHAINS = 8
GHMC_N_WARMUP = 300
GHMC_N_SAMPLES = 150

#: Full-run counts for the NUTS reference.
NUTS_N_WARMUP = 300
NUTS_N_SAMPLES = 150

#: MAP warm-start steps for both samplers.
MAP_STEPS = 150

#: Tolerance on the median gap, as a fraction of the NUTS 16-84 % width.
MEDIAN_TOLERANCE_WIDTHS = 0.1


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    return _free_z_model(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))


@pytest.fixture(scope="module")
def batches(mock, template):
    records = [_record(mock[i]) for i in CONFORMANCE_RECORDS]
    return build_spectro_batches(records, quantum=128, model=template)


def run_ghmc(template, batches, *, n_warmup, n_samples):
    """Fit the batches with the MEADS catalog sampler."""
    return fit_spectra_catalog_mcmc(
        template,
        batches,
        sampler="ghmc",
        n_warmup=n_warmup,
        n_burnin=0,
        n_samples=n_samples,
        init="map",
        map_steps=MAP_STEPS,
        n_chains=GHMC_CHAINS,
        n_superchains=GHMC_SUPERCHAINS,
    )


def run_nuts(template, batches, *, n_warmup, n_samples):
    """Fit the batches with the NUTS catalog sampler."""
    return fit_spectra_catalog_mcmc(
        template,
        batches,
        sampler="nuts",
        n_warmup=n_warmup,
        n_burnin=0,
        n_samples=n_samples,
        init="map",
        map_steps=MAP_STEPS,
    )


def assert_conforms(ghmc, nuts, *, n_warmup):
    """Medians within the tolerance of the NUTS width, with well-formed side outputs."""
    assert set(ghmc["draws"]) == set(nuts["draws"])
    for name in ghmc["draws"]:
        g = np.asarray(ghmc["draws"][name])
        n = np.asarray(nuts["draws"][name])
        lo, hi = np.percentile(n, [16.0, 84.0], axis=1)
        gap = np.abs(np.median(g, axis=1) - np.median(n, axis=1))
        assert np.all(gap <= MEDIAN_TOLERANCE_WIDTHS * (hi - lo)), (name, gap, hi - lo)
    divergences = np.asarray(ghmc["divergences"])
    assert np.issubdtype(divergences.dtype, np.integer)
    assert np.all(np.isfinite(divergences)) and np.all(divergences >= 0)
    assert np.all(np.asarray(ghmc["n_warmup_used"]) == n_warmup)


def test_ghmc_medians_agree_with_nuts_within_a_tenth_of_width(template, batches):
    ghmc = run_ghmc(template, batches, n_warmup=GHMC_N_WARMUP, n_samples=GHMC_N_SAMPLES)
    nuts = run_nuts(template, batches, n_warmup=NUTS_N_WARMUP, n_samples=NUTS_N_SAMPLES)
    assert_conforms(ghmc, nuts, n_warmup=GHMC_N_WARMUP)
