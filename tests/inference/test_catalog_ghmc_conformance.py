# SPDX-License-Identifier: BSD-3-Clause
"""MEADS ("ghmc") catalog posteriors against the NUTS catalog run at DESI-like S/N.

Two galaxies of the batched mock are fitted by the catalog NUTS sampler and by the MEADS
sampler with 32 chains in 8 superchains. Each galaxy's clean spectrum is the mock model at
its true mass, with Gaussian pixel noise of sigma = continuum(5000 A) / 5, a signal-to-noise
ratio of 5 per pixel. The posterior medians of every free parameter must agree with the
NUTS medians within a tenth of the NUTS full 16-84 % width, and the MEADS nested R-hat must
be at most 1.05.

The draw counts are the module constants below.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.inference.batched_spectro import fit_spectra_catalog_mcmc
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tests.contract.test_batched_spectro_loglik import _MASS, _mock_catalog
from tests.contract.test_batched_spectrum_forward import _banded, _free_z_model

pytestmark = pytest.mark.contract

#: Galaxies of the synthetic catalog used here, in record order.
CONFORMANCE_RECORDS = (0, 2)

#: Pixel signal-to-noise at 5000 Angstrom, the DESI-like regime.
CONTINUUM_SNR = 5.0

#: Wavelength at which the continuum level sets the pixel noise [Angstrom].
NOISE_REFERENCE_WAVE = 5000.0

#: Noise seed for the clean spectra.
NOISE_SEED = 20260930

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

#: Tolerance on the median gap, as a fraction of the NUTS full 16-84 % width.
MEDIAN_TOLERANCE_WIDTHS = 0.1

#: Largest nested R-hat accepted for the MEADS draws.
MAX_NESTED_RHAT = 1.05


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    return _free_z_model(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))


def desi_like_record(g, rng) -> GalaxySpectrum:
    """One galaxy's record: the clean mock spectrum plus pixel noise at CONTINUUM_SNR."""
    clean = np.asarray(g["ref"].predict_spectrum({_MASS: g["true_mass"], "redshift": g["z"]}))
    level = float(clean[np.argmin(np.abs(g["wave"] - NOISE_REFERENCE_WAVE))])
    sigma = level / CONTINUUM_SNR
    flux = clean + sigma * rng.standard_normal(clean.shape[0])
    return GalaxySpectrum(
        wave=g["wave"],
        flux=flux,
        ivar=np.full(clean.shape[0], 1.0 / sigma**2),
        resolution=g["bm"],
        z=g["z"],
    )


@pytest.fixture(scope="module")
def batches(mock, template):
    rng = np.random.default_rng(NOISE_SEED)
    records = [desi_like_record(mock[i], rng) for i in CONFORMANCE_RECORDS]
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
    """Medians within the tolerance of the NUTS full width, nested R-hat within bound."""
    assert set(ghmc["draws"]) == set(nuts["draws"])
    for name in ghmc["draws"]:
        g = np.asarray(ghmc["draws"][name])
        n = np.asarray(nuts["draws"][name])
        lo, hi = np.percentile(n, [16.0, 84.0], axis=1)
        gap = np.abs(np.median(g, axis=1) - np.median(n, axis=1))
        assert np.all(gap <= MEDIAN_TOLERANCE_WIDTHS * (hi - lo)), (name, gap, hi - lo)
    divergences = np.asarray(ghmc["divergences"])
    assert np.issubdtype(divergences.dtype, np.integer)
    assert np.all(divergences >= 0)
    assert np.all(np.asarray(ghmc["n_warmup_used"]) == n_warmup)
    assert np.all(np.asarray(ghmc["nested_rhat"]) <= MAX_NESTED_RHAT)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MEADS in the catalog engine reaches 0.48 NUTS widths at DESI-like S/N with 7x wider "
        "chains (smoke 40/20, #3135); the owner relabels the tier only when this passes"
    ),
)
def test_ghmc_medians_agree_with_nuts_at_desi_like_snr(template, batches):
    ghmc = run_ghmc(template, batches, n_warmup=GHMC_N_WARMUP, n_samples=GHMC_N_SAMPLES)
    nuts = run_nuts(template, batches, n_warmup=NUTS_N_WARMUP, n_samples=NUTS_N_SAMPLES)
    assert_conforms(ghmc, nuts, n_warmup=GHMC_N_WARMUP)
