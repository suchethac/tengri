# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the hierarchical flat seam must honor ``field_centering < 1`` (#3161).

``_hierarchical_flat.build_flat_problem`` wrote ``log_prior`` as an iid N(0, 1)
on every latent. At ``field_centering=a < 1`` the per-galaxy GP-field latent
``psd_xi`` has prior N(0, sigma_s^(2-2a) I) that depends on the sampled
amplitude, so the flat seam sampled a different distribution from the rest of
the pipeline, with no refusal.

The reference is the standardized seam's prior, ``standardized_neg_log_prior``
(``tengri.inference.loss_functions``), which the single-galaxy and population
objectives both use. The flat seam must evaluate the same function at the same
latent point. The per-galaxy field latents are ``(n_gal, n_grid)`` on the flat
seam; the reference is handed them flattened, so the normalizer counts every
element once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    PopulationFitter,
    SEDModel,
    Uniform,
)
from tengri.inference.loss_functions import standardized_neg_log_prior
from tengri.utils.transforms import to_bounded

pytestmark = pytest.mark.regression_bug

_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
_CENTERING = 0.5


def _factory(ssp, obs, field_centering):
    def factory(psd_sigma, psd_tau_myr):
        sfh = {
            "type": ["dpl", "field"],
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "psd_sigma": Uniform(0.1, 4.0, default=float(psd_sigma)),
            "psd_tau_myr": Uniform(1.0, 300.0, default=float(psd_tau_myr)),
        }
        if field_centering is not None:
            sfh["field_centering"] = field_centering
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=sfh,
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )

    return factory


@pytest.fixture(scope="module")
def centered_population():
    """Two galaxies on a partially non-centered stochastic SFH (``field_centering=0.5``)."""
    ssp = tengri.load_ssp()
    obs = Observation(photometry=Photometry.from_names(_BANDS))
    factory = _factory(ssp, obs, _CENTERING)
    template = factory(1.0, 50.0)
    truth = {
        k: (
            10.0
            if "log_total_mass" in k
            else 1.0
            if k == "sfh_field_psd_sigma"
            else 50.0
            if k == "sfh_field_psd_tau_myr"
            else 0.0
        )
        for k in template.spec.free_params
    }
    flux = np.asarray(template.predict_photometry(truth))
    galaxies = [
        {"flux_obs": flux * (1.0 + 0.02 * i), "noise": np.abs(flux) * 0.05} for i in range(2)
    ]
    return factory, galaxies


def test_flat_log_prior_matches_reference_seam_at_field_centering_half(centered_population):
    """Flat ``log_prior`` at a latent point == the reference prior at the same point.

    On main the flat seam returns the iid -0.5 * sum(latent^2) here, which
    differs from the centered reference by the -n(1-a) log sigma_s normalizer
    and the 0.5 * (1 - a) field term, so this fails before the fix.
    """
    from tengri.inference._hierarchical_flat import build_flat_problem

    factory, galaxies = centered_population
    fitter = PopulationFitter(factory, galaxies)
    assert float(fitter._spec.field_centering) == _CENTERING

    prob = build_flat_problem(fitter, key=jax.random.PRNGKey(0), map_steps=0)

    x = 0.4 * jax.random.normal(jax.random.PRNGKey(7), (prob.n_dim,))
    p = prob.unravel(x)

    sigma_lo, sigma_hi = fitter.psd_sigma_bounds
    sigma_phys = to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi)

    latents = dict(p["gal"])
    latents["psd_xi"] = jnp.ravel(p["gal_xi"])
    reference_neg = standardized_neg_log_prior(
        latents,
        fitter._free_names,
        stochastic=True,
        centering=_CENTERING,
        psd_sigma_dex=sigma_phys,
    )
    # The PSD hyperparameters are population-level latents with an iid N(0, 1)
    # prior on both seams; only the per-galaxy field latents change with a.
    reference_log_prior = -float(reference_neg) - 0.5 * float(
        p["psd_sigma_u"] ** 2 + p["psd_tau_u"] ** 2
    )

    flat_log_prior = float(prob.log_prior(x))
    np.testing.assert_allclose(flat_log_prior, reference_log_prior, rtol=0.0, atol=1e-10)


def test_nested_sampler_refuses_partial_centering(centered_population):
    """``nss`` draws live points from the iid prior, so it refuses a < 1 by name."""
    from tengri.inference._hierarchical_flat import run_flat_sampler

    factory, galaxies = centered_population
    fitter = PopulationFitter(factory, galaxies)
    with pytest.raises(ValueError, match="field_centering"):
        run_flat_sampler(fitter, "nss", key=jax.random.PRNGKey(0), map_steps=0)
