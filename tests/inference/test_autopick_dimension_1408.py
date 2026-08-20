# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #1408: MCMC auto-pick must count latent, not named, D.

``mcmc`` auto-dispatch chooses NUTS (low-D) vs ray tracing (high-D). Before
the fix the comparison used ``spec.n_free`` — the count of *named* free
parameters — so a field model with 3 named parameters but hundreds of latent
field coefficients was routed to NUTS as if it were 3-dimensional. The fix
adds ``spec.n_latent`` (flattened free-parameter size via ``jax.eval_shape``
over ``spec.sample``) and compares that at ``_registration.py`` (auto-pick)
and the ``fitter.py`` method-choice sites.

Two layers, neither of which re-implements the fix:

* ``test_mcmc_auto_pick_routes_by_latent_dimension`` calls the REAL
  ``_mcmc_auto_pick`` with the backends stubbed out, on a spec whose
  ``n_free`` sits below the threshold while ``n_latent`` sits above it — the
  exact configuration the bug mis-routed.
* ``test_n_latent_exceeds_n_free_for_field_sfh`` checks the property on a
  real field-SFH model.

Kill requirement (mutation-validated): changing ``context.spec.n_latent`` to
``context.spec.n_free`` at the auto-pick comparison in
``src/tengri/inference/_registration.py`` must turn
``test_mcmc_auto_pick_routes_by_latent_dimension`` red.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.slow


def test_mcmc_auto_pick_routes_by_latent_dimension(monkeypatch):
    """A low-n_free / high-n_latent spec must route to the high-D sampler.

    This drives the real ``_mcmc_auto_pick`` (including the real backend
    registry lookup for the high-D branch); only the terminal ``run_*``
    calls are recorded stubs. The mutant ``n_latent -> n_free`` routes the
    high-latent spec to NUTS and fails the first assertion.
    """
    from tengri.inference import _registration as reg
    from tengri.inference.fitter import _MCMC_AUTO_D_THRESHOLD

    calls: list[str] = []
    monkeypatch.setattr(reg, "_ctx_run_nuts", lambda context, **kw: calls.append("nuts"))
    monkeypatch.setattr(reg, "_ctx_run_raytrace", lambda context, **kw: calls.append("raytrace"))

    def _context(n_free, n_latent):
        return SimpleNamespace(spec=SimpleNamespace(n_free=n_free, n_latent=n_latent))

    # The #1408 configuration: 3 named parameters, but a latent field far
    # above the threshold. n_free says "low-D"; the truth is high-D.
    ctx_field = _context(n_free=3, n_latent=_MCMC_AUTO_D_THRESHOLD + 500)
    reg._mcmc_auto_pick(ctx_field, key=None)
    assert calls == ["raytrace"], (
        f"auto-pick routed a {_MCMC_AUTO_D_THRESHOLD + 500}-latent-dimension "
        f"model to {calls}; it must count n_latent, not n_free (#1408)"
    )

    # Control: a genuinely low-D spec still gets NUTS. This keeps the first
    # assertion honest — a dispatcher hardcoded to raytrace would fail here.
    calls.clear()
    ctx_low = _context(n_free=3, n_latent=_MCMC_AUTO_D_THRESHOLD)
    reg._mcmc_auto_pick(ctx_low, key=None)
    assert calls == ["nuts"], f"low-D control routed to {calls}, expected NUTS"


def test_n_latent_exceeds_n_free_for_field_sfh(ssp_data_fsps):
    """``spec.n_latent`` counts the flattened latents of a real field model.

    ``n_free`` counts named parameters only; the field's drawn coefficient
    vector must make ``n_latent`` strictly larger. Uses a real build so the
    ``eval_shape``-based property is exercised on an actual sampling path.
    """
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={
            "type": ["dpl", "field"],
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "psd_sigma": Fixed(1.0),
            "psd_tau_myr": Uniform(100, 500),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )

    n_latent = model.spec.n_latent
    n_free = model.spec.n_free

    assert isinstance(n_latent, int)
    assert n_latent > n_free, (
        f"n_latent ({n_latent}) must exceed n_free ({n_free}) for a field model: "
        "the field's coefficient vector is latent but not named"
    )
