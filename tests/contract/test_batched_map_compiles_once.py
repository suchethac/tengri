# SPDX-License-Identifier: BSD-3-Clause
"""The batched MAP driver compiles once per bucket and reuses the program (#3247).

``fit_spectra_map_vmap`` used to run its optimizer eagerly and to build a fresh
negative log-posterior on every call, so a repeated call re-traced and recompiled the
L-BFGS loop even for a batch of the same spec. The driver now caches one jitted chunk
program per (model, bucket spec, options) and passes the batch, the initial latents
and the template tables as arguments, so a second call with new data of the same
spec compiles nothing.

The reference values below were recorded from the unfixed (eager) driver before the
cache existed, at 10 L-BFGS iterations, seed 0, the second batch shifted by 0.3 sigma.
The jitted driver must reproduce them to rtol 1e-10.

Why 10 iterations: on this fixture the L-BFGS path is rounding-sensitive. The jitted
and unfixed drivers agree to 3e-13 (params) at 10 iterations, 1e-10 at 20, and drift
to 4e-3 at 60 because the MAP is not converged there. A reference at 60 iterations
would test XLA rounding, not the cache, so the comparison is taken where the
trajectory is still determined by the data.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from tengri import FREE, Observation, SEDModel, Spectroscopy, Uniform
from tengri.inference.batched_spectro import fit_spectra_map_vmap
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tests.contract.test_batched_spectro_loglik import _MASS, _mock_catalog, _record
from tests.contract.test_batched_spectrum_forward import _banded

pytestmark = pytest.mark.contract

_N_STEPS = 10
#: Flux offset of the second batch, in units of each galaxy's sigma.
_SHIFT = 0.3
_RTOL = 1e-10
_NLP_RTOL = 1e-9

#: Unfixed-driver reference at ``_N_STEPS``. Parameter order is the sorted free-latent
#: names; galaxies in catalog order.
_REF_A = {
    "sfh_dpl_age_frac": [0.9999920820333308, 0.7818258945229104, 0.9890415897512036],
    "sfh_dpl_alpha": [2.4675103886223506, 2.6020610661197763, 2.0690456920384737],
    "sfh_dpl_beta": [0.542637414241908, 1.1354911814638868, 0.9762608875366862],
    "sfh_dpl_log_total_mass": [9.997169713959169, 10.34422785132267, 9.699241596512879],
    "sfh_dpl_tau_gyr": [6.3202291942147975, 6.684015758691703, 7.225914426166645],
}
_REF_A_NLP = [114929.575122513, 3561470.721298251, 419817.24683666026]
_REF_B = {
    "sfh_dpl_age_frac": [0.9999993463934964, 0.7818875595766621, 0.9678902613283535],
    "sfh_dpl_alpha": [2.4314512669279753, 2.6020474239792493, 2.075660882211281],
    "sfh_dpl_beta": [0.46598835613596823, 1.1354784071406558, 1.1091483575717056],
    "sfh_dpl_log_total_mass": [9.995911815852713, 10.344304219308853, 9.697440951798354],
    "sfh_dpl_tau_gyr": [6.394946433682135, 6.684668546959996, 7.108188169842826],
}
_REF_B_NLP = [88812.85935235007, 3557303.36903124, 580734.1419278648]


def _dpl_template(ssp, wave, resolution):
    """Spectroscopic model with every DPL shape parameter and the mass free, z free."""
    spectroscopy = Spectroscopy(wave_obs=jnp.asarray(wave), resample="conserving", **resolution)
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": FREE, _MASS: Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.5),
    )


def _shifted(record: GalaxySpectrum, shift_in_sigma: float, sigma) -> GalaxySpectrum:
    """The same galaxy with its flux moved by ``shift_in_sigma`` noise standard deviations."""
    return GalaxySpectrum(
        wave=record.wave,
        flux=record.flux + shift_in_sigma * sigma,
        ivar=record.ivar,
        resolution=record.resolution,
        z=record.z,
    )


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    return _dpl_template(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))


@pytest.fixture(scope="module")
def batch_a(mock, template):
    records = [_record(g) for g in mock]
    return build_spectro_batches(records, quantum=420, model=template)


@pytest.fixture(scope="module")
def batch_b(mock, template):
    records = [_shifted(_record(g), _SHIFT, g["sigma"]) for g in mock]
    return build_spectro_batches(records, quantum=420, model=template)


def test_second_call_with_same_spec_compiles_nothing(template, batch_a, batch_b):
    """A repeated MAP call on a new batch of the same spec adds no compilation-cache miss."""
    [(spec_a, _, _)] = batch_a
    [(spec_b, _, _)] = batch_b
    assert spec_a == spec_b, "the two batches must share one bucket spec"

    fit_spectra_map_vmap(template, batch_a, n_steps=_N_STEPS)
    with jtu.count_jit_compilation_cache_miss() as misses:
        fit_spectra_map_vmap(template, batch_b, n_steps=_N_STEPS)
    assert misses() == 0, f"second call recompiled {misses()} programs"


def test_map_values_match_the_eager_driver(template, batch_a, batch_b):
    """The jitted driver reproduces the eager driver's MAP and final objective."""
    for batch, ref, ref_nlp in ((batch_a, _REF_A, _REF_A_NLP), (batch_b, _REF_B, _REF_B_NLP)):
        result = fit_spectra_map_vmap(template, batch, n_steps=_N_STEPS)
        assert set(result["params"]) == set(ref)
        for name, values in ref.items():
            np.testing.assert_allclose(
                np.asarray(result["params"][name]), np.asarray(values), rtol=_RTOL, atol=0.0
            )
        # The objective is a sum of order 1e5 to 1e6 nats accumulated along the same
        # trajectory; the largest measured gap at 10 steps is 2.8e-10 relative
        # (galaxy 2 of the second batch), so its bar is 1e-9, not the parameters' 1e-10.
        np.testing.assert_allclose(
            np.asarray(result["nlp"]), np.asarray(ref_nlp), rtol=_NLP_RTOL, atol=0.0
        )
