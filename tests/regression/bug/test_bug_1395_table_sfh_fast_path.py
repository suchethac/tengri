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
import pytest

from tengri.components.stellar import component as C
from tengri.components.stellar.sfh.registry import SFH_REGISTRY

pytestmark = pytest.mark.regression_bug


def test_table_sfh_is_seen_by_the_fast_path_guard():
    """The tabulated SFH's registry fn must remain in the guard map as a backstop.

    The CI-visible lock: needs no SSP grid, so it runs in the fast tier where a
    data-gated test would silently skip. This is the single membership fact the
    whole silent-zero failure hung on.

    Since #1396 the tabulated SFH is normally **served** — routed to the runtime
    table before the registry placeholder is ever evaluated — so this entry is no
    longer the primary defense. It stays as the backstop for a tabulated SFH that
    reaches the CIC branch *unrouted*, which is the only way the all-zero
    placeholder could be evaluated again.
    """
    assert SFH_REGISTRY["table"].fn in C._FAST_PATH_UNSUPPORTED_SFH_FNS, (
        "the tabulated SFH is invisible to compute_joint_weights' guard — an "
        "unrouted table would reach the CIC branch and evaluate the all-zero "
        "placeholder (#1395)"
    )


def test_guard_still_covers_the_nonparametric_families():
    """The #1395 rename must not drop the four SFHs the guard already caught (#950)."""
    from tengri.components.stellar.sfh.nonparametric import (
        continuity,
        continuity_flex,
        dirichlet,
        psb_continuity,
    )

    for fn in (continuity, continuity_flex, dirichlet, psb_continuity):
        assert fn in C._FAST_PATH_UNSUPPORTED_SFH_FNS, f"{fn.__name__} fell out of the guard"


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
            dust=None,
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
    # sample() carries free and Fixed parameters, but not the runtime history
    # arrays — so this is exactly the unservable case.
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    assert "sfh_sfr" not in params, "setup: the history must be absent for this case"

    with pytest.raises(ValueError, match="sfh_t_gyr"):
        stellar.compute_joint_weights(params)


def test_parametric_control_returns_live_weights(synthetic_ssp_wide, synthetic_tophat_obs):
    """Control: the same fast path on a parametric SFH is alive and normalized.

    Without this, the raise above would be satisfied by a wholly broken path.
    """
    model = _build({"type": "dpl"}, synthetic_ssp_wide, synthetic_tophat_obs)
    stellar = _stellar_of(model)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))

    joint_weights, total_mass, _ages = stellar.compute_joint_weights(params)

    assert jnp.all(jnp.isfinite(joint_weights)), "control weights must be finite"
    assert abs(float(joint_weights.sum()) - 1.0) < 1e-6, "control weights must sum to 1"
    assert float(total_mass) > 0.0, "control total formed mass must be positive"
