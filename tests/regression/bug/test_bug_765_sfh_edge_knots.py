# SPDX-License-Identifier: BSD-3-Clause
"""Regression: exact bin-edge knots in the DSPS SFH integrand (#765).

Follow-up to #758/#764. The dense log-spaced integrand (``_refine_sfh_table_ages``)
never lands exactly on a step SFH's bin edges, so DSPS interpolates across each
piecewise-constant transition and smears the per-SSP-age mass distribution — a
*resolution-insensitive* ~2 % optical residual vs Prospector (refining the log
grid 16->64 barely helps). The fix injects the SFH's exact bin edges as knots
(``sfh_bin_edges_yr`` + ``_inject_edge_knots``), making the step exact at any
resolution. This is gated to non-parametric SFHs, so the parametric path is
byte-unchanged.

continuity / dirichlet use ``DEFAULT_BIN_EDGES_GYR``, which already coincides
with the SSP log-age grid (so #764 alone closed them); continuity_flex derives
its edges from the SFR ratios and they fall *off* the grid — that is the case
the edge knots fix (verified: age-weight L1 error vs a converged reference drops
~100x).

The full <1 % residual-vs-Prospector check lives in
``reproduction/prospector/01_prospector.py`` (needs python-fsps + SPS_HOME);
here we guard the construction + the age-weight convergence that hold offline.

Issue: https://github.com/suchethac/tengri/issues/765
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.component import _inject_edge_knots, _refine_sfh_table_ages
from tengri.components.stellar.sfh.nonparametric import (
    CFLEX_DEFAULT_ANCHOR_GYR,
    DEFAULT_BIN_EDGES_GYR,
    PSB_FLEX_DEFAULT_MAX_AGE_GYR,
    PSB_FLEX_DEFAULT_N_FIXED,
    continuity,
    continuity_flex,
    dirichlet,
    psb_continuity,
    psb_continuity_flex,
    sfh_bin_edges_yr,
)
from tests._bounds import assert_non_negative

pytestmark = pytest.mark.conservation


def test_sfh_bin_edges_yr_continuity_and_dirichlet():
    """Fixed-bin families report ``DEFAULT_BIN_EDGES_GYR`` as their edges."""
    expect = np.asarray(DEFAULT_BIN_EDGES_GYR) * 1e9
    for fn in (continuity, dirichlet):
        edges = np.asarray(sfh_bin_edges_yr(fn, {}))
        assert np.allclose(edges, expect), fn.__name__


def test_sfh_bin_edges_yr_continuity_flex_matches_analytic():
    """continuity_flex edges follow the constant-mass flex-width derivation."""
    kw = dict(ratio_young=0.3, ratio_old=-0.4, flex_0=0.6, flex_1=-0.5, flex_2=0.4)
    edges = np.asarray(sfh_bin_edges_yr(continuity_flex, kw))
    # Reconstruct the expected edges independently.
    a = CFLEX_DEFAULT_ANCHOR_GYR * 1e9
    sr = 10.0 ** np.array([0.6, -0.5, 0.4])
    cpp = np.concatenate([[1.0], np.cumprod(sr)])
    dt = (a[1] - a[0]) * cpp / cpp.sum()
    expect = np.concatenate([[0.0, a[0]], a[0] + np.cumsum(dt), [a[2]]])
    assert edges.shape == expect.shape
    assert np.allclose(edges, expect, rtol=1e-9)
    assert_non_negative(np.diff(edges), name="output")


def test_sfh_bin_edges_yr_psb_continuity_flex():
    """The post-starburst ladder mirrors ``psb_continuity_flex``'s construction.

    ``[0, tlast]``, then the equal-width flexible edges out to ``tflex``, then
    the equal-width fixed edges out to the oldest age. This is the branch #2184
    added: both post-starburst registry entries name that callable, and until
    then neither was served any knots at all.
    """
    kw = dict(tlast_gyr=0.3, tflex_gyr=2.0, ratio_young=1.0)
    edges = np.asarray(sfh_bin_edges_yr(psb_continuity_flex, kw))
    fixed = np.linspace(2.0, PSB_FLEX_DEFAULT_MAX_AGE_GYR, PSB_FLEX_DEFAULT_N_FIXED + 1)
    expect = np.concatenate([[0.0, 0.3], fixed]) * 1e9
    assert np.allclose(edges, expect)
    assert_non_negative(np.diff(edges), name="output")


def test_sfh_bin_edges_yr_psb_continuity_flex_counts_the_flex_ratios():
    """``flex_*`` kwargs resolve the flexible zone, and the knots must follow it."""
    kw = dict(tlast_gyr=0.2, tflex_gyr=2.0, flex_0=0.1, flex_1=-0.2, flex_2=0.3, flex_3=0.0)
    edges = np.asarray(sfh_bin_edges_yr(psb_continuity_flex, kw)) / 1e9
    flex = np.linspace(0.2, 2.0, 6)  # four ratios describe five flexible bins
    fixed = np.linspace(2.0, PSB_FLEX_DEFAULT_MAX_AGE_GYR, PSB_FLEX_DEFAULT_N_FIXED + 1)
    expect = np.concatenate([[0.0], flex, fixed[1:]])
    assert np.allclose(edges, expect)


def test_sfh_bin_edges_yr_none_for_bare_psb_continuity():
    """``psb_continuity`` alone has no knowable ladder, so it gets none.

    Its fixed edges arrive as a separate argument the caller supplies, not in
    ``sfh_kwargs``, so a branch here could only guess at them; the branch that
    used to do exactly that returned the shipped default ladder for every
    caller. No registry entry names this callable since #2184 (both go through
    :func:`psb_continuity_flex`), so nothing on the model path is served by it.
    """
    assert sfh_bin_edges_yr(psb_continuity, dict(tlast_gyr=0.3, tflex_gyr=2.0)) is None


def test_sfh_bin_edges_yr_none_for_parametric():
    """A non-binned callable returns None (keeps the plain dense integrand)."""
    assert sfh_bin_edges_yr(lambda *a, **k: 0.0, {}) is None


def test_inject_edge_knots_monotonic_static_and_brackets_edges():
    """Knots are merged, sorted, static-sized, and bracket each edge."""
    fine = jnp.asarray(10.0 ** np.linspace(5.0, 10.13, 200))
    edges = jnp.asarray([3.16e7, 1.0e9, 5.0e9])
    lo, hi = float(fine[0]), float(fine[-1])
    merged = np.asarray(_inject_edge_knots(fine, edges, lo, hi))
    assert merged.shape[0] == 200 + 2 * 3  # static size
    assert_non_negative(np.diff(merged), name="output")  # sorted ascending
    # Each interior edge is bracketed by a knot just below and just above.
    for e in [1.0e9, 5.0e9]:
        assert np.any(merged < e) and np.any(merged > e)
        assert np.min(np.abs(merged - e)) < e * 1e-5


def test_edge_knot_sfh_conserves_mass_and_finite(synthetic_ssp_wide):
    """Edge-knot injection must not break mass conservation or finiteness."""
    sfh = {
        "type": "continuity_flex",
        "log_total_mass": Fixed(10.0),
        "ratio_young": Fixed(0.3),
        "flex_0": Fixed(0.6),
        "flex_1": Fixed(-0.5),
        "flex_2": Fixed(0.4),
        "ratio_old": Fixed(-0.4),
        "all_params": Fixed(DEFAULT),
    }
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh=sfh,
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.0),
    )
    state = model.predict_state({})
    formed = float(np.sum(np.asarray(state.derived["age_weights"])))
    assert np.isclose(np.log10(formed), 10.0, atol=1.5e-2), (
        f"formed log10={np.log10(formed):.5f} (expected 10.0) for {sfh['type']}"
    )
    sed = np.asarray(state.sed_intrinsic)
    assert np.all(np.isfinite(sed)) and np.all(sed >= 0.0)


@pytest.mark.parametrize("sfh_type", ["psb_suess2022", "psb_flex"])
def test_edge_knots_close_the_post_starburst_formed_mass(synthetic_ssp_wide, sfh_type):
    """Both post-starburst entries must form the mass they declare, on the default grid.

    Before #2184 wired ``psb_continuity_flex`` into :func:`sfh_bin_edges_yr`,
    neither entry got any knots, and the 256-node log-spaced integrand smeared
    every step of the ladder: measured 9.995266 against a declared 10.0, and a
    1.14 % offset on the whole intrinsic SED, which is a normalization error and
    not a shape one. With the knots both entries return 9.999999.

    Run at z = 0 so the lookback grid reaches the ladder's oldest edge; at any
    higher redshift the cosmic-age cap truncates the oldest bin and the formed
    mass is legitimately below the declared total.
    """
    log_total_mass = 10.0
    sfh = {
        "type": sfh_type,
        "log_total_mass": Fixed(log_total_mass),
        "tlast_gyr": Fixed(0.3),
        "tflex_gyr": Fixed(2.0),
        "ratio_young": Fixed(-1.5),
        "ratio_old_0": Fixed(0.2),
        "ratio_old_1": Fixed(-0.3),
        "all_params": Fixed(DEFAULT),
    }
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh=sfh,
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.0),
    )
    formed = float(model.predict_state({}).derived["log_mstar_formed"])
    assert abs(formed - log_total_mass) < 1e-4, (
        f"{sfh_type} formed log10={formed:.6f} against a declared {log_total_mass}: "
        "the ladder's bin edges are not reaching the integrand as knots (#765/#2184)"
    )


def test_edge_knot_sfh_is_jit_safe(synthetic_ssp_wide):
    """continuity_flex with edge knots stays jittable (static-size merge)."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={
            "type": "continuity_flex",
            "log_total_mass": Fixed(10.0),
            "flex_0": Fixed(0.4),
            "flex_1": Fixed(-0.3),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.0),
    )
    sed = jax.jit(lambda p: model.predict_state(p).sed_intrinsic)({})
    assert np.all(np.isfinite(np.asarray(sed)))


def test_edge_knots_converge_continuity_flex_age_weights(real_ssp_only):
    """Edge knots drive continuity_flex age weights to the converged reference.

    The decisive, FSPS-free check: for a step SFH the exact (FSPS-equivalent)
    age weights are the converged DSPS result on a dense edge-resolved grid.
    Injecting bin-edge knots into the operational log×16 integrand must close
    most of the gap that the plain log grid leaves.
    """
    from dsps.sed.stellar_sed import calc_ssp_weights_sfh_table_lognormal_mdf as ssp_w

    import tengri
    from tengri.components.stellar.component import _build_dsps_sfh_table

    ssp = tengri.load_ssp()
    ssp_lg_age_gyr = jnp.asarray(ssp.ssp_lg_age_gyr)
    ssp_lgmet = jnp.asarray(ssp.ssp_lgmet)
    ssp_ages_yr = 10.0 ** np.asarray(ssp_lg_age_gyr) * 1e9
    t_obs = 13.7
    kw = dict(
        log_total_mass=10.0, ratio_young=0.3, ratio_old=-0.4, flex_0=0.6, flex_1=-0.5, flex_2=0.4
    )

    def age_weights(age_yr):
        age_yr = np.sort(np.asarray(age_yr))
        sfr = np.asarray(continuity_flex(jnp.asarray(age_yr), **kw))
        gt, gs, _ = _build_dsps_sfh_table(jnp.asarray(age_yr), jnp.asarray(sfr), t_obs)
        return np.asarray(ssp_w(gt, gs, 0.0, 0.2, ssp_lgmet, ssp_lg_age_gyr, t_obs)[0])

    edges = sfh_bin_edges_yr(continuity_flex, kw)
    fine = _refine_sfh_table_ages(jnp.asarray(ssp_ages_yr))
    w_log = age_weights(fine)  # plain dense log grid (#764)
    w_new = age_weights(_inject_edge_knots(fine, edges, ssp_ages_yr[0], ssp_ages_yr[-1]))
    dense = 10.0 ** np.linspace(
        np.log10(ssp_ages_yr[0]), np.log10(ssp_ages_yr[-1]), (len(ssp_ages_yr) - 1) * 256 + 1
    )
    w_ref = age_weights(
        _inject_edge_knots(jnp.asarray(dense), edges, ssp_ages_yr[0], ssp_ages_yr[-1])
    )

    l1_log = float(np.sum(np.abs(w_log - w_ref)))
    l1_new = float(np.sum(np.abs(w_new - w_ref)))
    assert l1_new < l1_log / 20.0, f"edge knots gave only {l1_log / max(l1_new, 1e-12):.0f}x"
    assert np.isclose(w_new.sum(), w_ref.sum(), rtol=1e-4)  # mass preserved
