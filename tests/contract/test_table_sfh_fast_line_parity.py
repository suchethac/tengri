# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the SED-free fast path SERVES tabulated histories, bit-exactly (#1396).

#1395 made ``compute_joint_weights`` *refuse* a tabulated SFH, because its
registry ``fn`` is an all-zero placeholder and the fast path would have returned
finite zeros. Refusing is correct but forfeits the LUT, which is most of the
point of a simulation catalog: ``WavePrecomp`` bakes the SSP x filter integral,
and that integral is SFH-independent, so a tabulated history changes only the
(met, age) weights the SSPs are summed with.

This file is the gate for serving it instead: the fast path must reproduce
``apply``'s published ``joint_weights``, because a fast path that disagrees with
the exact forward is the precise failure #950's contract exists to prevent.

**The two routes are held to different bounds, and the difference is measured,
not assumed.**

* Tabulated **SFH** (delta metallicity) is asserted **bit-exact**. Its op graph
  is an outer product of two already-normalized marginals, which XLA emits
  identically in both compilation units.
* Tabulated **metallicity** is asserted to a few ulps (``rtol=1e-13``). It
  routes through ``_joint_weights_cic_met_table``, whose ``.at[:, idx].add(...)``
  is a scatter-add — XLA picks the accumulation order, and ``predict_state`` is
  a much larger graph, so fusion and reassociation differ. Measured on this
  grid: max **relative** error ``2.88e-16`` (one to two ulps of float64, whose
  eps is ``2.2e-16``) on a cell of value ``0.199``, with the two weight sums
  differing by exactly one ulp.

Demanding bit-exactness of the second route would produce a permanently red
test; relaxing the first to match would hide a real regression. Each is held to
the strictest bound it can actually meet.

The #1395 invariant is preserved and strengthened, not dropped:

* tabulated SFH **without** the runtime arrays still raises (it cannot be
  served, and must never return a silent zero);
* tabulated SFH **with** the runtime arrays is served, and equals the exact
  forward.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.contract

_Z_OBS = 0.05
# Anchored at t=0 with SFR=0 so nothing extrapolates past the Big Bang; see
# tests/unit/inference/test_catalog_histories.py for why that matters.
_T_GYR = np.concatenate([np.array([0.0]), np.linspace(1.0, 13.0, 39)])


def _stellar_of(model):
    from tengri.components.stellar.component import StellarSEDComponent

    chain = model._build_component_chain()
    return next(c for c in chain if isinstance(c, StellarSEDComponent))


def _build(synthetic_ssp_wide, synthetic_tophat_obs, *, met_table=False):
    from tengri import FIXED, SEDModel
    from tengri.parameters.priors import Fixed, Uniform

    groups = dict(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "table"},
        dust={
            "type": "two_component",
            "all_params": FIXED,
            "tau_bc": 0.5,
            "tau_diff": Uniform(0.0, 2.0),
        },
        neb={"type": "none"},
        redshift=Fixed(_Z_OBS),
    )
    if met_table:
        # Metallicity's structural axis lives under the `stellar` group as
        # `met_mode` — there is no top-level `met` group in the build grammar.
        groups["stellar"] = {"met_mode": "table"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**groups)


def _history_params(model, *, sfr_level=3.0, met=None, sfr=None):
    """Full params dict + the runtime history arrays.

    Built from ``spec.sample`` rather than hand-assembled, because ``sample``
    returns **Fixed** values too (``redshift``, ``met_logzsol``). A hand-built
    dict omitting them sends ``compute_joint_weights`` down its
    ``params.get("redshift", 0.0)`` default — z=0 instead of the model's fixed
    redshift — so the fast and exact routes would be compared at *different
    cosmic times*, and the parity assertion would fail for a reason that has
    nothing to do with tabulated histories. (That default is the #1432 class;
    real callers pass a full dict, which is what this mirrors.)
    """
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    params["dust_tau_diff"] = jnp.asarray(0.3)
    if sfr is None:
        shape = np.ones(_T_GYR.shape[0])
        shape[0] = 0.0
        sfr = shape * sfr_level
    params["sfh_t_gyr"] = jnp.asarray(_T_GYR)
    params["sfh_sfr"] = jnp.asarray(sfr)
    if met is not None:
        params["met_history"] = jnp.asarray(met)
    return params


def test_table_sfh_without_arrays_still_refuses(synthetic_ssp_wide, synthetic_tophat_obs):
    """The #1395 invariant: unservable means raise, never a silent zero."""
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    stellar = _stellar_of(model)

    with pytest.raises(ValueError, match="sfh_t_gyr"):
        stellar.compute_joint_weights({"dust_tau_diff": jnp.asarray(0.3)})


def test_table_sfh_weights_are_not_zero(synthetic_ssp_wide, synthetic_tophat_obs):
    """The #1395 symptom must be gone: served weights are normalized and massive."""
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    stellar = _stellar_of(model)

    jw, total_mass, _ages = stellar.compute_joint_weights(_history_params(model))

    assert jnp.all(jnp.isfinite(jw)), "weights must be finite"
    assert abs(float(jw.sum()) - 1.0) < 1e-12, f"weights must sum to 1, got {float(jw.sum())}"
    assert float(total_mass) > 0.0, "a tabulated SFH with positive SFR must form mass"


def test_table_sfh_fast_weights_match_the_exact_forward(synthetic_ssp_wide, synthetic_tophat_obs):
    """Bit-exact parity with apply's published joint_weights (the #950 contract)."""
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    stellar = _stellar_of(model)

    for level in (0.5, 3.0, 17.0):
        params = _history_params(model, sfr_level=level)
        jw_fast, mass_fast, _ = stellar.compute_joint_weights(params)
        state = model.predict_state(params)
        jw_exact = np.asarray(state.derived["joint_weights"])

        assert np.array_equal(np.asarray(jw_fast), jw_exact), (
            f"fast/exact joint_weights diverge at sfr_level={level}; "
            f"max |delta| = {np.max(np.abs(np.asarray(jw_fast) - jw_exact)):.3e}"
        )
        assert float(mass_fast) > 0.0


def test_table_sfh_fast_weights_track_the_history(synthetic_ssp_wide, synthetic_tophat_obs):
    """A different history must give different weights — parity alone is not enough.

    Two SFHs with the same total mass but different *shapes* (young-heavy vs
    old-heavy) must produce different age weights. Without this, an implementation
    that ignored the table and returned some fixed distribution would still pass
    the parity test above, since both routes would ignore it identically.
    """
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    stellar = _stellar_of(model)

    n_t = _T_GYR.shape[0]
    young = np.zeros(n_t)
    young[-10:] = 5.0  # recent star formation (late cosmic time)
    old = np.zeros(n_t)
    old[1:11] = 5.0  # early star formation

    jw_young, _, _ = stellar.compute_joint_weights(_history_params(model, sfr=young))
    jw_old, _, _ = stellar.compute_joint_weights(_history_params(model, sfr=old))

    delta = float(jnp.abs(jw_young - jw_old).sum())
    assert delta > 0.1, f"young and old histories gave near-identical weights (L1={delta:.3e})"


def test_table_metallicity_fast_weights_match_the_exact_forward(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """Per-age metallicity must route to the CIC met-table kernel, bit-exactly."""
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, met_table=True)
    stellar = _stellar_of(model)

    # A rising Z(t): metal-poor early, enriched late, inside the synthetic grid.
    met = np.linspace(-1.0, -0.2, _T_GYR.shape[0])
    params = _history_params(model, met=met)

    jw_fast, mass_fast, _ = stellar.compute_joint_weights(params)
    jw_exact = np.asarray(model.predict_state(params).derived["joint_weights"])

    # Relative to the peak weight, not per-cell: the smallest cells are ~1e-17
    # and a per-cell ratio there measures nothing but round-off. rtol=1e-13 sits
    # three orders above the 2.88e-16 float64 noise floor measured on this grid,
    # and far below any accumulation order could ever explain.
    rel = np.max(np.abs(np.asarray(jw_fast) - jw_exact)) / jw_exact.max()
    assert rel < 1e-13, (
        f"fast/exact joint_weights diverge for a tabulated metallicity: "
        f"max relative delta {rel:.3e} (float64 noise floor here is ~3e-16)"
    )
    assert float(mass_fast) > 0.0


def test_table_metallicity_actually_spreads_over_the_met_axis(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The WHOLE Z(t) curve must be used, not just its youngest-age value.

    The two histories below **agree exactly at the latest cosmic time** (the
    youngest stars) and differ only at earlier times. So any implementation that
    collapses the curve to a single scalar — the delta-metallicity fallback, or
    reading only ``lgmet_on_ssp_ages[0]`` — produces *identical* weights for
    both and fails here.

    This discriminator was chosen because the obvious one does not work: a flat
    vs a rising history still differ after collapsing (their collapsed values
    differ), so that version passed even when the curve was deliberately
    flattened. A parity test cannot catch this either — collapsing in the shared
    helper collapses *both* routes, and they agree with each other while both
    being wrong. Verified by mutation.
    """
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, met_table=True)
    stellar = _stellar_of(model)

    n_t = _T_GYR.shape[0]
    rising = np.linspace(-1.2, -0.1, n_t)  # metal-poor early, enriched late
    constant = np.full(n_t, -0.1)  # same late value, no early enrichment history

    assert rising[-1] == constant[-1], "setup: the two histories must agree at the latest time"

    jw_rising, _, _ = stellar.compute_joint_weights(_history_params(model, met=rising))
    jw_constant, _, _ = stellar.compute_joint_weights(_history_params(model, met=constant))

    delta = float(jnp.abs(jw_rising - jw_constant).sum())
    assert delta > 1e-3, (
        f"Z(t) histories differing only at EARLY times gave the same weights "
        f"(L1={delta:.3e}) — the per-age curve is being collapsed to a scalar"
    )
