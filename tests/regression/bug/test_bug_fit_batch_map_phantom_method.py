# SPDX-License-Identifier: BSD-3-Clause
"""``Fitter.fit_batch(method="map")`` called a method that never existed.

Both optimizer branches of ``_fit_batch_vmap_map`` unpacked their per-galaxy result
with ``self._bounded_from_unbounded(params_i)`` — a phantom: ``git log -S`` shows it
was **never defined** anywhere in the history. So the canonical batch-MAP API raised
``AttributeError: 'Fitter' object has no attribute '_bounded_from_unbounded'`` the
moment it took the vmap path (same-shape catalog + fixed-z photometry precompute —
exactly the fast path a catalog run wants). The correct method is ``_to_physical``,
which converts one unbounded param dict to physical space.

CI never caught it because the only batch-MAP test drives
``tengri.forward.convenience.fit_batch_map_vmap`` — a *separate* implementation that
does not route through ``Fitter._fit_batch_vmap_map``. Nothing exercised the
``Fitter.fit_batch(method="map")`` entry point itself.

Pins: the vmap batch-MAP path returns one finite, physical-space Posterior per galaxy,
with the free parameter present and inside its prior bounds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.inference.fitter import Fitter
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def batch_fitter(synthetic_ssp_wide):
    obs = Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))
    )
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8.0, 12.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    # The precompute is what routes fit_batch through the vmap path where the bug lived.
    assert model.has_fixedz_photometry_precompute, "must take the vmap batch-MAP path"
    # Synthetic flux must come from the model itself (#2296 follow-up): with
    # profile_mass engaged (auto, since this model's mass amplitude is linear
    # in the photometry -- see mass_profile.py), the batch-MAP path now
    # reinserts the ANALYTICALLY profiled mass instead of silently returning
    # the Fixed placeholder unconditionally (the pre-#2296 behavior this test
    # never actually exercised: ``_to_physical`` used to blanket-merge every
    # Fixed value, including the placeholder, into ``Posterior.params``,
    # which is why an arbitrary ``flux ~ N(1.0, 0.1)`` -- disconnected from
    # this model's own flux scale of order 1e-17 -- used to read back as a
    # trivially in-bounds 10.0 without ever exercising the real formula).
    # An analytically-profiled amplitude is only meaningful (and only
    # guaranteed to land near the declared prior) when the data is actually
    # consistent with the model at some mass inside that prior, so both
    # galaxies are drawn from ``model.predict_photometry`` at a true mass
    # inside [8, 12] plus realistic per-band noise.
    rng = np.random.default_rng(0)
    truth_1 = {"sfh_dpl_log_total_mass": 10.0, "dust_tau_bc": 0.3}
    truth_2 = {"sfh_dpl_log_total_mass": 9.5, "dust_tau_bc": 0.5}
    flux_true_1 = np.asarray(model.predict_photometry(truth_1))
    flux_true_2 = np.asarray(model.predict_photometry(truth_2))
    noise_1 = 0.05 * flux_true_1
    noise_2 = 0.05 * flux_true_2
    flux_1 = flux_true_1 + rng.normal(0.0, noise_1)
    flux_2 = flux_true_2 + rng.normal(0.0, noise_2)
    batch = [
        {"flux_obs": jnp.asarray(flux_1), "noise": jnp.asarray(noise_1)},
        {"flux_obs": jnp.asarray(flux_2), "noise": jnp.asarray(noise_2)},
    ]
    f = Fitter(model, batch[0]["flux_obs"], batch[0]["noise"], data_type="photometry")
    return f, batch


def test_fit_batch_map_returns_one_physical_posterior_per_galaxy(batch_fitter):
    """The vmap batch-MAP path must not raise, and must return physical params."""
    f, batch = batch_fitter

    results = f.fit_batch(
        batch, method="map", key=jax.random.PRNGKey(0), n_steps=40, verbose=False
    )

    assert len(results) == len(batch), "one Posterior per galaxy"
    for r in results:
        # MAP is a point estimate: it populates `params`, not `samples` — the same
        # convention the single-galaxy MAP backend uses (map_dispatch).
        assert r.params is not None, "batch-MAP must populate Posterior.params"
        assert "sfh_dpl_log_total_mass" in r.params, "free parameter must be present"
        val = float(np.asarray(r.params["sfh_dpl_log_total_mass"]).reshape(-1)[0])
        assert np.isfinite(val), "MAP estimate must be finite"
        # _to_physical unstandardizes into the prior's support — the phantom method
        # would have returned unbounded (standardized) values had it existed.
        assert 8.0 <= val <= 12.0, f"param must be in physical space / prior bounds, got {val}"
