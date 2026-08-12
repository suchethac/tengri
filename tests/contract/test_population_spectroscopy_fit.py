# SPDX-License-Identifier: BSD-3-Clause
"""Canonical hierarchical fit on a stochastic-SFH spectroscopy population (#711).

Regression for the two gaps that broke the canonical
``Fitter(ForwardModel.build(population=...))`` path for a
``['tsnorm','field']`` spectroscopy population:

* **Gap 1** — ``PopulationSpecView`` lacked a public ``n_grid`` so
  ``Fitter.compile_signature`` raised ``AttributeError``.
* **Gap 2** — ``ForwardModel.predict_spectrum`` delegated to the *scalar* inner
  SED, bypassing the population vmap, so stacked per-galaxy SFH params collided
  with the age grid (``sub got incompatible shapes for broadcasting: (12,), (3,)``).

These run on the synthetic wide SSP (no ``data/ssp_*.h5`` needed) so they guard
the canonical spectroscopy-population path on CI. The slow-tier end-to-end
native-VI acceptance (``tests/inference/test_population_spectroscopy_vi.py``)
was removed 2026-07-10: its assertions were shape/topology-only (no truth
recovery at ``n_iterations=2``) at ~35 min of XLA compile per nightly run.
A truth-recovery version would need converged fits and belongs in a
deliberately-budgeted acceptance suite, not here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, FREE, Fixed, Observation, SEDModel
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population_sed_model import PopulationSEDModel
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.contract

_N_GAL = 3
_N_GRID = 12
_N_PIX = 80
_Z = 0.1


@pytest.fixture(scope="module")
def spec_obs():
    wave_obs = jnp.logspace(jnp.log10(3000.0 * (1.0 + _Z)), jnp.log10(7500.0 * (1.0 + _Z)), _N_PIX)
    return Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))


@pytest.fixture(scope="module")
def template(synthetic_ssp_wide, spec_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=spec_obs,
        sfh={"type": ["tsnorm", "field"], "*": FREE},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(_Z),
        n_grid=_N_GRID,
    )


@pytest.fixture(scope="module")
def population(template):
    key = jax.random.PRNGKey(0)
    galaxies = []
    for i in range(_N_GAL):
        params = template.spec.sample(jax.random.fold_in(key, i))
        flux = template.predict_spectrum(params)
        noise = jnp.abs(flux) * 0.05 + 1e-30
        galaxies.append({"flux_obs": flux, "noise": noise})
    return PopulationSEDModel(sed=template, galaxies=galaxies, data_type="spectroscopy")


def test_population_spec_view_exposes_public_n_grid(population):
    """Gap 1: the population spec mirrors the template's public ``n_grid``."""
    spec = population.spec
    assert spec.stochastic
    assert spec.n_grid == _N_GRID


def test_forward_predict_spectrum_is_batched_over_galaxies(population, spec_obs):
    """Gap 2: spectrum prediction returns ``(N_gal, n_pix)`` via the batched seam."""
    forward = ForwardModel.build(population=population, observation=spec_obs)
    params = forward.spec.sample(jax.random.PRNGKey(7))
    spectra = forward.predict_spectrum(params)
    assert spectra.shape == (_N_GAL, _N_PIX)
    assert jnp.all(jnp.isfinite(spectra))
