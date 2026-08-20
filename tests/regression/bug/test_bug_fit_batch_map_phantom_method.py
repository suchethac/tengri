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

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
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
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(8.0, 12.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    # The precompute is what routes fit_batch through the vmap path where the bug lived.
    assert model.has_fixedz_photometry_precompute, "must take the vmap batch-MAP path"
    rng = np.random.default_rng(0)
    flux = np.abs(rng.normal(1.0, 0.1, size=3))
    batch = [
        {"flux_obs": jnp.asarray(flux), "noise": jnp.asarray(0.1 * flux)},
        {"flux_obs": jnp.asarray(1.3 * flux), "noise": jnp.asarray(0.1 * flux)},
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
