# SPDX-License-Identifier: BSD-3-Clause
"""Regression: a tabulated SFH must not slip past the fast-path guard (#1395).

``StellarSEDComponent.compute_joint_weights`` is the FeaturePrecomp fast
line-path entry. Its docstring promises it "raises for anything else, so the
fast path can never silently diverge from the exact forward" — but the
guard is a membership test against an enumerated set of SFH *functions*, and
the tabulated SFH's registry entry (``_table_sfh_placeholder``) was never a
member. It therefore passed both guards, reached the CIC branch, and evaluated
the placeholder — whose entire body is ``jnp.zeros_like(t_lookback)``, because
the real table is wired in at ``apply``, which the fast path never reaches.

The zero was then laundered into a *finite* zero by the normalization clamp
``w / jnp.maximum(w.sum(), 1e-300)``: ``0 / 1e-300 == 0.0``. Every downstream
``isfinite`` check passed. Photometry does not route through this function, so
the observable symptom was correct photometry beside zero lines and zero
stellar mass — a mixed signal, far harder to notice than an all-zero result.

Measured on ``fb8d95470`` before the fix, on an identical age grid:

===================  ==================  ==============  =======
config               ``sum(weights)``    ``total_mass``  raised?
===================  ==================  ==============  =======
parametric (control) 1.0                 2.95964e+09     no
``table``            0.0                 0.0             no
===================  ==================  ==============  =======

The control is asserted in the same module so that "the table zeroes out"
cannot be an artifact of a dead helper.

``metallicity_model="table"`` is **not** affected — it trips the
delta-metallicity guard loudly and falls back to the exact forward, as designed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed
from tengri.components.stellar.sfh.registry import SFH_REGISTRY
from tengri.parameters.resolve import merge_fixed_params

pytestmark = pytest.mark.regression_bug


def test_table_sfh_with_its_arrays_is_served_and_matches_apply(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """With its runtime arrays, the tabulated SFH is served: fast path equals apply.

    The refusal map that used to guard the placeholder is gone (#3505 M7). The
    placeholder can no longer be reached: the table is routed to its runtime arrays
    before it is evaluated, on both paths.
    """
    model = _build({"type": "table"}, synthetic_ssp_wide, synthetic_tophat_obs)
    stellar = _stellar_of(model)
    params = merge_fixed_params(model.spec, dict(model.spec.sample(jax.random.PRNGKey(0))))
    params["sfh_t_gyr"] = jnp.linspace(1.0, 10.0, 19)
    params["sfh_sfr"] = 2.0 + 3.0 * jnp.exp(-0.5 * ((params["sfh_t_gyr"] - 4.0) / 1.5) ** 2)

    weights, mass, _ages = stellar.compute_joint_weights(params)
    assert jnp.all(jnp.isfinite(weights)) and abs(float(weights.sum()) - 1.0) < 1e-6
    assert float(mass) > 0.0, "a served table must form mass, not the placeholder's zero"

    from tengri.protocols.component import ForwardState

    state = stellar.apply(ForwardState(wave=jnp.logspace(3.0, 4.5, 64)), params, stellar.ssp_data)
    np.testing.assert_allclose(
        np.asarray(weights), np.asarray(state.derived["joint_weights"]), rtol=1e-12
    )


def test_binned_family_fast_path_is_served_with_normalized_weights(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """A binned family (continuity) is served by the fast path, not refused (#3505).

    The binned families used to be refused by the fast path's map. They now share
    evaluate_sfh with apply; the weights are finite and normalized.
    """
    model = _build({"type": "continuity", "all_params": Fixed(DEFAULT)}, synthetic_ssp_wide,
                   synthetic_tophat_obs)
    stellar = _stellar_of(model)
    params = merge_fixed_params(model.spec, dict(model.spec.sample(jax.random.PRNGKey(0))))
    weights, mass, _ages = stellar.compute_joint_weights(params)
    assert jnp.all(jnp.isfinite(weights))
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert float(mass) > 0.0


def test_table_sfh_placeholder_still_returns_zero():
    """Pin the reason the guard is load-bearing: the registry fn IS all-zero.

    If the placeholder ever grows a real implementation this test fails, which is
    the correct prompt to revisit the guard rather than leave it as dead weight.
    """
    ages = jnp.geomspace(1e6, 1.3e10, 64)
    assert float(jnp.abs(SFH_REGISTRY["table"].fn(ages)).sum()) == 0.0


def _stellar_of(model):
    """The StellarSEDComponent off the built chain (same accessor as the parity suite)."""
    from tengri.components.stellar.component import StellarSEDComponent

    chain = model._build_component_chain()
    return next(c for c in chain if isinstance(c, StellarSEDComponent))


def _build(sfh, ssp, obs):
    """Minimal model on the guard's supported axis (delta metallicity, no field)."""
    import warnings

    from tengri import Fixed, SEDModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=sfh,
            dust_attenuation=None,
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )


def test_compute_joint_weights_raises_for_table_sfh(synthetic_ssp_wide, synthetic_tophat_obs):
    """End-to-end: an unservable tabulated SFH raises, and never returns zeros.

    The *mechanism* changed after #1396 and this test changed with it, so the
    change is worth stating plainly. #1395 closed the hole by making the fast
    path **refuse** a tabulated SFH ("use the exact forward"); #1396 closed it
    the other way, by making the fast path **serve** it — which was always the
    stated end state ("either serves the table SFH or refuses loudly").

    What is asserted here is the part that did **not** change, and which is the
    actual #1395 invariant: a tabulated SFH the fast path *cannot* evaluate —
    here because the runtime arrays are absent, since they are records rather
    than sampled parameters — raises, naming what is missing. It must never
    reach the all-zero registry placeholder and launder it through the
    normalization clamp into a finite zero.

    The complementary half — a tabulated SFH *with* its arrays is served, and
    agrees with the exact forward — is gated in
    ``tests/contract/test_table_sfh_fast_line_parity.py``.
    """
    model = _build({"type": "table"}, synthetic_ssp_wide, synthetic_tophat_obs)
    stellar = _stellar_of(model)
    # ``compute_joint_weights`` is a direct component-level call, bypassing the
    # model's own merge boundary (``predict_state`` / the forward pipeline,
    # which merge ``{**fixed_values, **params}`` before any component runs) --
    # so this test must do that merge itself. ``spec.sample()`` is free-only
    # (#2296), so a bare ``dict(spec.sample(...))`` omits Fixed ``redshift``
    # here, same as it always legally omits it on the standard path; this
    # direct call needs it supplied explicitly via merge_fixed_params. Still
    # missing: the runtime history arrays -- so this is exactly the
    # unservable case.
    params = merge_fixed_params(model.spec, dict(model.spec.sample(jax.random.PRNGKey(0))))
    assert "sfh_sfr" not in params, "setup: the history must be absent for this case"

    with pytest.raises(ValueError, match="sfh_t_gyr"):
        stellar.compute_joint_weights(params)


def test_parametric_control_returns_live_weights(synthetic_ssp_wide, synthetic_tophat_obs):
    """Control: the same fast path on a parametric SFH is alive and normalized.

    Without this, the raise above would be satisfied by a wholly broken path.
    """
    model = _build({"type": "dpl"}, synthetic_ssp_wide, synthetic_tophat_obs)
    stellar = _stellar_of(model)
    # See the comment above: a direct component call needs the Fixed values
    # (e.g. redshift) merged in itself; the standard forward pipeline does
    # this merge for every ordinary predict_* caller.
    params = merge_fixed_params(model.spec, dict(model.spec.sample(jax.random.PRNGKey(0))))

    joint_weights, total_mass, _ages = stellar.compute_joint_weights(params)

    assert jnp.all(jnp.isfinite(joint_weights)), "control weights must be finite"
    assert abs(float(joint_weights.sum()) - 1.0) < 1e-6, "control weights must sum to 1"
    assert float(total_mass) > 0.0, "control total formed mass must be positive"
