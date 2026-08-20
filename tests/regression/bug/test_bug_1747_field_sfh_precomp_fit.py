# SPDX-License-Identifier: BSD-3-Clause
"""A stochastic-field SFH must be *fittable* under a precompute LUT (#1747).

#1747 reported that ``WavePrecomp`` / ``SpectrumPrecomp`` / ``FeaturePrecomp``
all *build* against a field-SFH model cheaply, but that **fitting** one was
SIGKILLed -- exit 137 as a script, ``DeadKernelError`` under nbclient -- so
something on the fit path allocated without bound once a precompute was
attached. The cost was concrete: ``stochastic_sfh_recovery``, the slowest
published notebook, stayed on the exact wave grid while a measured 67x
per-forward-pass speedup sat unusable.

It no longer reproduces. Measured on the model of that page (``sfh={"type":
["dpl", "field"]}``, ``n_grid=16``, photometry, fixed redshift), a short
``mcmc_hmc`` fit completes both ways:

    exact          OK   95.3 s   peak RSS 3.51 GB
    WavePrecomp    OK   18.4 s   peak RSS 4.23 GB

and the full-scale page (300 warmup + 200 samples x 100 leapfrog) now executes
with ``WavePrecomp`` wired as its ``build()`` default.

**No commit claims this fix and nothing guarded it**, which is the reason for
this file. The repair was incidental -- most likely a side effect of the
precompute-state work -- so there is no other test standing between a future
refactor and a silent return to an unfittable fast path. A build-only check
would not do: #1747's whole point is that building always worked and only the
fit died.

Deliberately small (few HMC steps, ``n_grid`` reduced): the failure was an
unbounded allocation, which shows up during trace and compile, not a slow leak
that needs a long chain to accumulate. #1747 itself measured the reduced
``n_grid=8`` variant failing too, so the small case is representative.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import FREE, Fixed, ForwardModel, Observation, SEDModel, WavePrecomp
from tengri.observation import Photometry

pytestmark = pytest.mark.regression_bug

_BANDS = ["sdss_g", "sdss_r", "sdss_i", "sdss_z"]
_N_GRID = 8


def _build(ssp, obs, approx):
    """The stochastic_sfh_recovery model: a dpl modulated by the stochastic field."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": ["dpl", "field"], "all_params": FREE},
        met={"logzsol": Fixed(-0.3)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FREE,
        },
        neb={"type": "ssp"},
        redshift=Fixed(0.5),
        apply_igm=False,
        n_grid=_N_GRID,
        approx=approx,
    )


@pytest.fixture(scope="module")
def observation():
    return Observation(photometry=Photometry.from_names(_BANDS))


def test_field_sfh_fit_completes_under_wave_precomp(synthetic_ssp_wide, observation):
    """The regression: this fit used to be SIGKILLed, not merely slow."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build(synthetic_ssp_wide, observation, WavePrecomp())
        forward = ForwardModel.build(sed=model)
        params = model.spec.sample(jax.random.PRNGKey(0))
        flux = np.asarray(forward.predict_photometry(params), dtype=np.float64)
        noise = np.abs(flux) * 0.05 + 1e-30

        posterior = forward.fit(
            flux,
            noise,
            method="mcmc_hmc",
            key=jax.random.PRNGKey(1),
            n_warmup=2,
            n_samples=2,
            n_leapfrog_steps=2,
            approx=WavePrecomp(),
        )

    assert posterior.samples, "fit returned no samples; the LUT fit path is broken again"
    for name, draws in posterior.samples.items():
        arr = np.asarray(draws)
        assert arr.shape[0] == 2, f"{name}: expected 2 draws, got {arr.shape[0]}"
        assert np.all(np.isfinite(arr)), (
            f"{name}: non-finite draws from the precomputed field-SFH fit. #1747 was a "
            f"memory failure, but a NaN chain is the same defect one step later -- the "
            f"LUT and the field prior disagreeing about shape."
        )


def test_field_sfh_field_latents_are_actually_free(synthetic_ssp_wide, observation):
    """Control: the fit above must exercise the field, not a degenerate model.

    ``n_grid`` latents are what made this configuration distinctive -- every
    notebook where precompute worked fine used a parametric SFH. If a future
    change quietly stopped the field contributing free parameters, the test
    above would keep passing while no longer covering the thing #1747 broke.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build(synthetic_ssp_wide, observation, WavePrecomp())

    free = list(model.spec.free_params)
    field = [p for p in free if "field" in p]
    assert field, (
        f"no field parameters are free, so this model is not the #1747 configuration; "
        f"free params were {free}"
    )
