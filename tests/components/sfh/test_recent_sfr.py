# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the decoupled current star-formation episode (BEAGLE-style).

Grid used throughout: ``t = linspace(0, 5 Gyr, 50001)``, spacing 1e5 yr, with
``t[0] = 0``. The window length Delta t = 10 Myr is exactly 100 cells, so the
window edge falls on a cell edge and ``window_weight`` telescopes exactly.
The smooth history is ``delayed_tau`` with tau = 1 Gyr and age = 5 Gyr.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sfh import (
    check_specific_sfr_pole,
    current_episode,
    psi_from_specific_sfr,
    splice_current_episode,
)
from tengri.components.stellar.sfh.mean_sfh import delayed_tau, window_weight

DT_YR = 1e7  # Delta t = 10 Myr
TAU_YR = 1e9  # delayed-tau timescale, 1 Gyr
AGE_YR = 5e9  # age of the smooth history, 5 Gyr
M_SMOOTH = 1e10  # declared smooth formed mass [Msun]
PSI = 3.0  # episode SFR [Msun/yr]


def _grid():
    return jnp.linspace(0.0, AGE_YR, 50001)


def _smooth(t):
    return delayed_tau(t, tau=TAU_YR, norm=1.0, age=AGE_YR)


def _spliced(t, psi=PSI, dt=DT_YR):
    s = _smooth(t)
    return splice_current_episode(t, s, psi, dt)


@pytest.mark.limit
def test_episode_is_constant_psi_inside_window():
    """Inside the window the SFR is exactly psi (smooth history replaced by zero)."""
    t = _grid()
    inside = t < 0.9 * DT_YR
    sfr = _spliced(t)
    assert_allclose(np.asarray(sfr[inside]), PSI, rtol=0, atol=1e-12)
    ep = current_episode(t, jnp.log10(PSI), DT_YR)
    assert_allclose(np.asarray(ep), PSI * np.asarray(window_weight(t, 0.0, DT_YR)), atol=1e-12)


@pytest.mark.conservation
def test_splice_conserves_smooth_mass():
    """The smooth part of the splice integrates to the declared M_S (1e-10 rel).

    Conservation law: the smooth history keeps its formed mass M_S when the
    episode replaces it inside the window.
    """
    t = _grid()
    s = _smooth(t)
    m_s = jnp.trapezoid(s, t)
    smooth_part = _spliced(t) - PSI * window_weight(t, 0.0, DT_YR)
    got = jnp.trapezoid(smooth_part, t)
    assert_allclose(float(got), float(m_s), rtol=1e-10)


@pytest.mark.conservation
def test_total_mass_is_smooth_plus_psi_dt():
    """Total formed mass is M_S + psi * Delta t to 1e-6 relative.

    Grid: 50001 nodes on [0, 5 Gyr], spacing 1e5 yr, Delta t = 1e7 yr.
    """
    t = _grid()
    m_s = jnp.trapezoid(_smooth(t), t)
    total = jnp.trapezoid(_spliced(t), t)
    assert_allclose(float(total), float(m_s + PSI * DT_YR), rtol=1e-6)


@pytest.mark.limit
def test_mean_sfr_over_window_is_psi():
    """Window-averaged SFR equals psi to 1e-10.

    Evaluated on the episode-only term, whose cell-exact quadrature over the
    window telescopes to Delta t. The spliced SFR equals psi on every interior
    node of the window (see test_episode_is_constant_psi_inside_window); the
    straddling edge cell is not integrated by a node-based trapezoid, so the
    identity is checked on the cell-exact episode term.
    """
    t = _grid()
    ep = current_episode(t, jnp.log10(PSI), DT_YR)
    mean = jnp.trapezoid(ep, t) / DT_YR
    assert_allclose(float(mean), PSI, rtol=1e-10)


@pytest.mark.limit
@pytest.mark.parametrize("ssfr_dt", [1e-4, 0.1, 0.5])
def test_ssfr_inversion_round_trips(ssfr_dt):
    """psi -> psi_S -> psi round-trips to 1e-12 for psi_S * Delta t < 1."""
    psi_s = ssfr_dt / DT_YR
    psi_in = psi_s * M_SMOOTH / (1.0 - psi_s * DT_YR)
    # forward: psi_S relative to the TOTAL formed mass M_S + psi * Delta t
    psi_s_fwd = psi_in / (M_SMOOTH + psi_in * DT_YR)
    assert_allclose(psi_s_fwd, psi_s, rtol=1e-12)
    psi_out = psi_from_specific_sfr(jnp.log10(psi_s_fwd), M_SMOOTH, DT_YR)
    assert_allclose(float(psi_out), psi_in, rtol=1e-12)


@pytest.mark.limit
@pytest.mark.parametrize("ssfr_dt", [1.0, 1.5])
def test_ssfr_pole_refused(ssfr_dt):
    """A specific-SFR prior reaching the pole psi_S * Delta t = 1 is refused."""
    log_hi = float(np.log10(ssfr_dt / DT_YR))
    with pytest.raises(ValueError, match="pole"):
        check_specific_sfr_pole(log_hi, DT_YR)


@pytest.mark.limit
def test_pole_and_fully_windowed_history_propagate_non_finite():
    """No clamp hides a degenerate input: the pole and a zero splice norm give non-finite output.

    The build refuses a prior that reaches the pole (check_specific_sfr_pole is the
    only guard), so these are the kernel's own behavior when that guard is bypassed.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        at_pole = psi_from_specific_sfr(jnp.log10(1.0 / DT_YR), M_SMOOTH, DT_YR)
    assert not np.isfinite(float(at_pole))
    t = _grid()
    inside_only = jnp.where(t < DT_YR, 1.0, 0.0)  # all smooth mass inside the window
    with np.errstate(divide="ignore", invalid="ignore"):
        spliced = splice_current_episode(t, inside_only, PSI, DT_YR)
    assert not np.all(np.isfinite(np.asarray(spliced)))


@pytest.mark.gradient
def test_splice_gradient_wrt_psi_and_timescale_finite_nonzero():
    """d/d(log psi) and d/d(Delta t) of the total mass are finite and nonzero.

    Analytically the total is M_S + psi * Delta t, so d/dlog10(psi) =
    psi ln10 Delta t and d/dDelta t = psi.
    """
    t = _grid()

    def total_vs_log_psi(lp):
        return jnp.trapezoid(splice_current_episode(t, _smooth(t), 10.0**lp, DT_YR), t)

    def total_vs_dt(dt):
        return jnp.trapezoid(splice_current_episode(t, _smooth(t), PSI, dt), t)

    g_psi = float(jax.grad(total_vs_log_psi)(jnp.log10(PSI)))
    g_dt = float(jax.grad(total_vs_dt)(DT_YR))
    assert np.isfinite(g_psi) and g_psi != 0.0
    assert np.isfinite(g_dt) and g_dt != 0.0
    assert_allclose(g_psi, PSI * np.log(10.0) * DT_YR, rtol=1e-6)
    assert_allclose(g_dt, PSI, rtol=1e-6)


@pytest.mark.limit
def test_jit_matches_eager():
    """Jitted splice equals eager splice."""
    t = _grid()
    s = _smooth(t)
    eager = splice_current_episode(t, s, PSI, DT_YR)
    jitted = jax.jit(splice_current_episode)(t, s, PSI, DT_YR)
    assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-12, atol=1e-12)


# ── Registry types 'recent' and 'recent_ssfr' (#3177 part 2) ────────────
#
# The composer receives values already in internal units, exactly as
# ``get_internal_params`` delivers them: ``sfh_recent_timescale_myr`` arrives
# as Delta t in yr (scale 1e6), ``sfh_delayed_tau_gyr`` as tau in yr (scale 1e9).

M_DELAYED = 1e10  # declared smooth formed mass, 10**10 Msun
LOG_M_DELAYED = 10.0
TAU_GYR = 1.0
AGE_GYR = 5.0


def _delayed_recent_kw(**recent):
    base = {
        "sfh_delayed_log_total_mass": LOG_M_DELAYED,
        "sfh_delayed_tau_gyr": TAU_GYR * 1e9,
        "sfh_delayed_age_gyr": AGE_GYR * 1e9,
    }
    base.update(recent)
    return base


@pytest.mark.contract
def test_registry_lists_recent_and_recent_ssfr():
    """Both current-episode types are registered and appear in the SFH menu."""
    from tengri import list_sfh_models
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    assert "recent" in SFH_REGISTRY
    assert "recent_ssfr" in SFH_REGISTRY
    assert SFH_REGISTRY["recent"].composition_type == "current"
    assert SFH_REGISTRY["recent_ssfr"].composition_type == "current"
    menu = {row["name"]: row for row in list_sfh_models()}
    assert {"recent", "recent_ssfr"} <= set(menu)


@pytest.mark.contract
def test_resolve_sfh_delayed_recent_param_names():
    """resolve_sfh(['delayed', 'recent']) declares the episode and delayed names."""
    from tengri.components.stellar.sfh.registry import resolve_sfh

    _fn, params, pmap, _settings = resolve_sfh(["delayed", "recent"])
    for name in (
        "sfh_recent_log_sfr",
        "sfh_recent_timescale_myr",
        "sfh_delayed_log_total_mass",
        "sfh_delayed_tau_gyr",
        "sfh_delayed_age_gyr",
    ):
        assert name in params, name
    assert pmap["sfh_recent_timescale_myr"] == ("timescale_yr", 1e6, 0.0)
    assert pmap["sfh_recent_log_sfr"] == ("log_sfr", 1.0, 0.0)


@pytest.mark.parametrize(
    "types",
    [["delayed", "recent", "recent_ssfr"], ["delayed", "recent", "recent"]],
    ids=["recent_and_ssfr", "recent_twice"],
)
@pytest.mark.contract
def test_two_current_terms_refused(types):
    """At most one current episode per composition."""
    from tengri.components.stellar.sfh.registry import resolve_sfh

    with pytest.raises(ValueError, match="At most one current episode per composition"):
        resolve_sfh(types)


@pytest.mark.parametrize("types", [["recent"], ["recent_ssfr"]], ids=["recent", "recent_ssfr"])
@pytest.mark.contract
def test_recent_requires_a_smooth_member(types):
    """A current term alone, with no additive smooth member, is refused."""
    from tengri.components.stellar.sfh.registry import resolve_sfh

    with pytest.raises(ValueError, match="smooth"):
        resolve_sfh(types)


@pytest.mark.conservation
def test_composed_history_equals_psi_in_window_and_conserves_smooth_mass():
    """Composed SFR is psi in the window; smooth part keeps 10**log_total_mass."""
    from tengri.components.stellar.sfh.registry import resolve_sfh

    fn, _params, _pmap, _settings = resolve_sfh(["delayed", "recent"])
    t = _grid()
    kw = _delayed_recent_kw(
        sfh_recent_log_sfr=float(np.log10(PSI)),
        sfh_recent_timescale_myr=DT_YR,
    )
    sfr = np.asarray(fn(t, **kw))
    inside = np.asarray(t) < 0.9 * DT_YR
    assert_allclose(sfr[inside], PSI, rtol=0, atol=1e-12)

    smooth_part = sfr - PSI * np.asarray(window_weight(t, 0.0, DT_YR))
    got = np.trapezoid(smooth_part, np.asarray(t))
    assert_allclose(got, M_DELAYED, rtol=1e-6)
    total = np.trapezoid(sfr, np.asarray(t))
    assert_allclose(total, M_DELAYED + PSI * DT_YR, rtol=1e-6)


@pytest.mark.contract
def test_recent_ssfr_form_equals_recent_form_at_converted_psi():
    """recent_ssfr with log psi_S equals recent with psi = psi_from_specific_sfr(...)."""
    from tengri.components.stellar.sfh.registry import resolve_sfh

    log_ssfr = -8.0
    psi = float(psi_from_specific_sfr(log_ssfr, M_DELAYED, DT_YR))
    t = _grid()

    fn_ssfr, _p, _m, _s = resolve_sfh(["delayed", "recent_ssfr"])
    kw_ssfr = _delayed_recent_kw(
        sfh_recent_ssfr_log_ssfr=log_ssfr,
        sfh_recent_ssfr_timescale_myr=DT_YR,
    )
    fn_rec, _p2, _m2, _s2 = resolve_sfh(["delayed", "recent"])
    kw_rec = _delayed_recent_kw(
        sfh_recent_log_sfr=float(np.log10(psi)),
        sfh_recent_timescale_myr=DT_YR,
    )
    assert_allclose(
        np.asarray(fn_ssfr(t, **kw_ssfr)), np.asarray(fn_rec(t, **kw_rec)), rtol=1e-10, atol=0
    )


@pytest.mark.contract
def test_timescale_in_myr_reaches_kernel_in_years():
    """Delta t = 50 Myr, scaled by the registry map, puts the window edge at 5e7 yr."""
    from tengri.components.stellar.sfh.registry import resolve_sfh

    fn, _p, pmap, _s = resolve_sfh(["delayed", "recent"])
    scale = pmap["sfh_recent_timescale_myr"][1]
    dt_yr = 50.0 * scale
    assert_allclose(dt_yr, 5e7)
    t = _grid()
    sfr = np.asarray(
        fn(
            t,
            **_delayed_recent_kw(
                sfh_recent_log_sfr=float(np.log10(PSI)),
                sfh_recent_timescale_myr=dt_yr,
            ),
        )
    )
    tn = np.asarray(t)
    assert_allclose(sfr[tn < 0.99 * 5e7], PSI, rtol=0, atol=1e-12)
    assert np.all(sfr[tn > 1.01 * 5e7] != PSI)
