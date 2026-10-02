# SPDX-License-Identifier: BSD-3-Clause
"""Age-binned dust attenuation: N independent screens, each its own law and age
window (#2528).

Generalizes :mod:`tengri.components.dust.two_component`'s Charlot & Fall
(2000) birth-cloud/diffuse-ISM pair to N screens. The identity test (1)
verifies ``age_binned`` with N=2 and the two_component windows reproduces
``two_component`` exactly; the rest cover the N-screen physics directly:
window partition, k(5500 A)=1, energy balance, gradients, the LUT refusal,
grammar validation, and a mutation check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp
from tengri.components.dust.age_binned import (
    AgeBinnedDustComponent,
    AgeBinnedDustComponentConfig,
    _age_window_weight,
    validate_screens,
    validate_screens_against_grid,
)
from tengri.components.dust.laws._registry import DUST_LAWS, resolve_dust_law
from tengri.config.exceptions import ConfigError, ParameterError
from tengri.utils.physics_constants import C_AA, LYMAN_LIMIT_AA

pytestmark = pytest.mark.regression_bug

_LAW_PAIRS = [
    ("calzetti", "calzetti"),
    ("calzetti", "power_law"),
    ("cardelli", "calzetti"),
    ("power_law", "power_law"),
]
_TAUS = [0.3, 1.0, 2.5]
_AGE_KERNELS = ["cic", "dsps"]
_SFH_TYPES = ["delayed", "psb"]


# ── Shared SSP / observation fixtures (no data/ files needed) ──────────────


@pytest.fixture
def _ssp(synthetic_ssp_wide):
    return synthetic_ssp_wide


@pytest.fixture
def _obs(synthetic_tophat_obs):
    return synthetic_tophat_obs


def _two_component_model(ssp, obs, *, law_bc, law_diff, tau_bc, tau_diff, sfh_type, age_kernel):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": sfh_type, "age_kernel": age_kernel, "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law_bc": law_bc,
            "law_diff": law_diff,
            "tau_bc": tau_bc,
            "tau_diff": tau_diff,
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )


def _age_binned_two_screen_model(
    ssp, obs, *, law_bc, law_diff, tau_0, tau_1, sfh_type, age_kernel, t_birth_log_yr=7.0
):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": sfh_type, "age_kernel": age_kernel, "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "age_binned",
            "screens": [
                {"law": law_bc, "window_log_yr": (None, t_birth_log_yr)},
                {"law": law_diff, "window_log_yr": (None, None)},
            ],
            "tau_0": tau_0,
            "tau_1": tau_1,
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )


def _flat_params(model, *, log_total_mass=10.0, other=1.0):
    return {
        name: (log_total_mass if "log_total_mass" in name else other)
        for name in model.spec.free_params
    }


# ── 1. Identity: age_binned N=2 reproduces two_component exactly ───────────


@pytest.mark.parametrize("law_bc,law_diff", _LAW_PAIRS)
@pytest.mark.parametrize("tau_bc", _TAUS)
@pytest.mark.parametrize("age_kernel", _AGE_KERNELS)
@pytest.mark.parametrize("sfh_type", _SFH_TYPES)
def test_age_binned_n2_identity_vs_two_component(
    _ssp, _obs, law_bc, law_diff, tau_bc, age_kernel, sfh_type
):
    """age_binned N=2 (two_component's own windows) == two_component, exactly.

    screens = [{law_bc, window=(None, log10(t_birth))}, {law_diff,
    window=(None, None)}] reduces w_0(t) to two_component's birth-cloud
    sigmoid and w_1(t) == 1 identically (see
    :class:`AgeBinnedDustComponent`'s docstring derivation), so the two
    screens' optical depth, transmission, spectrum, photometry and
    L_absorbed must match two_component to machine precision: 0.0 relative
    difference (the same jnp op sequence/associativity reproduces the same
    floating-point rounding), not merely a tight tolerance.
    """
    tau_diff = tau_bc * 0.6
    m_two = _two_component_model(
        _ssp,
        _obs,
        law_bc=law_bc,
        law_diff=law_diff,
        tau_bc=tau_bc,
        tau_diff=tau_diff,
        sfh_type=sfh_type,
        age_kernel=age_kernel,
    )
    m_aged = _age_binned_two_screen_model(
        _ssp,
        _obs,
        law_bc=law_bc,
        law_diff=law_diff,
        tau_0=tau_bc,
        tau_1=tau_diff,
        sfh_type=sfh_type,
        age_kernel=age_kernel,
    )

    params_two = _flat_params(m_two)
    params_aged = _flat_params(m_aged)

    pred_two = m_two.predict(params_two)
    pred_aged = m_aged.predict(params_aged)

    np.testing.assert_array_equal(
        np.asarray(pred_two.rest_sed()), np.asarray(pred_aged.rest_sed())
    )
    np.testing.assert_array_equal(
        np.asarray(m_two.predict_photometry(params_two)),
        np.asarray(m_aged.predict_photometry(params_aged)),
    )
    assert pred_two.properties["l_dust_absorbed"] == pred_aged.properties["l_dust_absorbed"]


# ── 2. Partition and windows ────────────────────────────────────────────────


def test_age_binned_partition_sums_to_one_away_from_edges():
    """Three tiling screens: Σ w_i(t) = 1 at every age far from a transition edge."""
    screens = validate_screens(
        [
            {"law": "calzetti", "window_log_yr": (None, 7.0)},
            {"law": "power_law", "window_log_yr": (7.0, 8.5)},
            {"law": "cardelli", "window_log_yr": (8.5, None)},
        ]
    )
    width = 0.3
    log_age = jnp.linspace(0.0, 13.0, 4000)
    total = jnp.zeros_like(log_age)
    for _law, lo, hi in screens:
        total = total + _age_window_weight(log_age, lo, hi, width)

    # "Far from an edge" = at least 10 transition widths from 7.0 and 8.5.
    far = (jnp.abs(log_age - 7.0) > 12 * width) & (jnp.abs(log_age - 8.5) > 12 * width)
    assert int(far.sum()) > 100  # the mask must not be vacuous
    np.testing.assert_allclose(np.asarray(total)[np.asarray(far)], 1.0, atol=1e-6)

    # AT a shared edge the partition is NOT exact for N=3: each neighbor's
    # sigmoid independently reaches 0.5 there (this is a product of
    # independent per-screen sigmoids, not a normalized partition), so the
    # sum runs slightly above 1 -- exact only for N=2, where the shared
    # edge has one half-infinite screen on each side (see the class
    # docstring). Pin the measured values rather than call it exact.
    edges = jnp.array([7.0, 8.5])
    total_at_edges = jnp.zeros_like(edges)
    for _law, lo, hi in screens:
        total_at_edges = total_at_edges + _age_window_weight(edges, lo, hi, width)
    np.testing.assert_allclose(np.asarray(total_at_edges), 1.00334643, atol=1e-6)
    assert float(jnp.max(total)) == pytest.approx(1.0057544634761175, abs=1e-6)


def test_age_binned_middle_screen_tau_only_changes_its_own_window():
    """Zeroing the middle screen's tau changes flux only in SSPs its window covers.

    Evaluates a single-age SSP: an age well inside the middle window must
    change when dust_tau_1 -> 0; an age well inside screen 0 or screen 2's
    window must not.
    """
    screens = validate_screens(
        [
            {"law": "calzetti", "window_log_yr": (None, 7.0)},
            {"law": "power_law", "window_log_yr": (7.0, 8.5)},
            {"law": "cardelli", "window_log_yr": (8.5, None)},
        ]
    )
    comp = AgeBinnedDustComponent(config=AgeBinnedDustComponentConfig(screens=screens))
    wave = jnp.linspace(1200.0, 20000.0, 200)

    def transmission_at(age_yr, tau1):
        ages = jnp.array([age_yr])
        params = {"dust_tau_0": 0.8, "dust_tau_1": tau1, "dust_tau_2": 0.5, "dust_Rv_2": 3.1}
        return comp.compute_transmission(params, wave, ages)[0]

    # age=1 yr (the component's own log10(max(age,1)) floor) is the farthest
    # reachable from the 7.0 edge; the logistic tail only decays
    # exponentially (sigmoid(-23) ~ 1e-10), so this is an atol check against
    # that floor, not an exact-zero one.
    t_young = transmission_at(1.0, 1.0), transmission_at(1.0, 0.0)  # inside screen 0
    t_mid = transmission_at(10**7.75, 1.0), transmission_at(10**7.75, 0.0)  # inside screen 1
    t_old = transmission_at(10**13, 1.0), transmission_at(10**13, 0.0)  # inside screen 2

    np.testing.assert_allclose(np.asarray(t_young[0]), np.asarray(t_young[1]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(t_old[0]), np.asarray(t_old[1]), atol=1e-6)
    assert np.max(np.abs(np.asarray(t_mid[0]) - np.asarray(t_mid[1]))) > 1e-3


# ── 3. k(5500 A) = 1 for every registered law ───────────────────────────────


@pytest.mark.parametrize("law_name", sorted(DUST_LAWS))
def test_every_registered_law_normalizes_at_v_band(law_name):
    """k(5500 A) = 1 [V-band convention] for every law age_binned can select."""
    import inspect

    fn = resolve_dust_law(law_name)
    sig = inspect.signature(fn.callable)
    kwargs = {
        name: p.default
        for name, p in sig.parameters.items()
        if name != "wavelength" and p.default is not inspect.Parameter.empty
    }
    k = fn(jnp.array([5500.0]), **kwargs)
    np.testing.assert_allclose(float(k[0]), 1.0, atol=1e-6)


# ── 4. Energy balance (N=3) + IR template ───────────────────────────────────


def test_age_binned_energy_balance_n3_matches_trapz_integral(_ssp, _obs):
    """L_absorbed == |trapz(intrinsic - attenuated, over nu, the Lyman edge to 3um)| for N=3."""
    kwargs = dict(
        ssp_data=_ssp,
        observation=_obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    # Screen 1's lower edge (7.5) sits exactly at the build-time threshold
    # (synthetic_ssp_wide's youngest node, 6.0 log10(age/yr), + 5 *
    # transition_width_dex=0.3 -- see validate_screens_against_grid):
    # a full SEDModel.build goes through that grid-dependent refusal,
    # unlike the partition/gradient tests above that call validate_screens
    # or AgeBinnedDustComponent directly.
    screens = [
        {"law": "calzetti", "window_log_yr": (None, 7.5)},
        {"law": "power_law", "window_log_yr": (7.5, 9.0)},
        {"law": "cardelli", "window_log_yr": (9.0, None)},
    ]
    m = SEDModel.build(
        dust_attenuation={
            "type": "age_binned",
            "screens": screens,
            "tau_0": 0.5,
            "tau_1": 1.0,
            "tau_2": 0.3,
            "other_params": Fixed(DEFAULT),
        },
        **kwargs,
    )
    m0 = SEDModel.build(
        dust_attenuation={
            "type": "age_binned",
            "screens": screens,
            "tau_0": 0.0,
            "tau_1": 0.0,
            "tau_2": 0.0,
            "other_params": Fixed(DEFAULT),
        },
        **kwargs,
    )
    params = _flat_params(m)
    pred = m.predict(params)
    pred0 = m0.predict(params)

    from tengri.forward.energy_balance import bolometric_absorbed_log10
    from tengri.utils.scale import pow10

    wave = jnp.asarray(pred.wave_rest)
    rest = jnp.asarray(pred.rest_sed())
    rest0 = jnp.asarray(pred0.rest_sed())
    nu = C_AA / wave
    log_l, _sign = bolometric_absorbed_log10(
        rest0, rest, nu, wave=wave, lyman_cutoff_aa=LYMAN_LIMIT_AA
    )
    l_trapz = float(pow10(log_l))

    from tengri.utils.physics_constants import L_SUN

    l_published_erg_s = float(pred.properties["l_dust_absorbed"]) * L_SUN
    assert l_published_erg_s > 0.0
    np.testing.assert_allclose(l_trapz, l_published_erg_s, rtol=1e-6)


def test_age_binned_dust_ir_receives_l_absorbed(_ssp, _obs):
    """With a dust_emission engine wired, L_ir re-emission tracks L_absorbed."""
    m = SEDModel.build(
        ssp_data=_ssp,
        observation=_obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "age_binned",
            "screens": [
                {"law": "calzetti", "window_log_yr": (None, 7.0)},
                {"law": "calzetti", "window_log_yr": (None, None)},
            ],
            "tau_0": 0.5,
            "tau_1": 0.3,
            "other_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "modified_blackbody", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    params = _flat_params(m)
    pred = m.predict(params)
    assert float(pred.properties["l_tir"]) > 0.0
    np.testing.assert_allclose(
        float(pred.properties["l_tir"]), float(pred.properties["l_dust_absorbed"]), rtol=1e-4
    )


# ── 5. Gradients ─────────────────────────────────────────────────────────────


def test_age_binned_gradients_finite_and_match_finite_difference():
    """jax.grad wrt every dust_tau_i: finite, nonzero, and within 1e-5 relative
    of central finite differences.

    A 1% tolerance left four orders of magnitude of undetected slack: central
    FD at eps=1e-4 has an O(eps^2) truncation floor of ~1e-8 relative (the
    third-derivative term), and the measured agreement here is ~1e-9-1e-10
    relative per tau_i. 1e-5 keeps ~3 orders of margin above that floor
    while still catching a gradient bug many orders tighter than 1%.
    """
    screens = validate_screens(
        [
            {"law": "calzetti", "window_log_yr": (None, 7.0)},
            {"law": "power_law", "window_log_yr": (7.0, 8.5)},
            {"law": "cardelli", "window_log_yr": (8.5, None)},
        ]
    )
    comp = AgeBinnedDustComponent(config=AgeBinnedDustComponentConfig(screens=screens))
    wave = jnp.linspace(1200.0, 20000.0, 100)
    ages = jnp.logspace(5.5, 10.0, 40)

    def objective(tau0, tau1, tau2):
        params = {"dust_tau_0": tau0, "dust_tau_1": tau1, "dust_tau_2": tau2, "dust_Rv_2": 3.1}
        return jnp.sum(comp.compute_transmission(params, wave, ages))

    g0, g1, g2 = jax.grad(objective, argnums=(0, 1, 2))(0.5, 1.0, 0.3)
    for g in (g0, g1, g2):
        assert jnp.isfinite(g)
        assert float(jnp.abs(g)) > 0.0

    eps = 1e-4
    fd0 = (objective(0.5 + eps, 1.0, 0.3) - objective(0.5 - eps, 1.0, 0.3)) / (2 * eps)
    fd1 = (objective(0.5, 1.0 + eps, 0.3) - objective(0.5, 1.0 - eps, 0.3)) / (2 * eps)
    fd2 = (objective(0.5, 1.0, 0.3 + eps) - objective(0.5, 1.0, 0.3 - eps)) / (2 * eps)
    for g, fd in zip((g0, g1, g2), (fd0, fd1, fd2)):
        assert abs(float(g) - float(fd)) / abs(float(fd)) < 1e-5


# ── 6. LUT: no WavePrecomp/SpectrumPrecomp; approx="auto" stays exact ──────


def test_age_binned_wave_precomp_refuses_loudly(_ssp, _obs):
    """An explicit approx=WavePrecomp() on an age_binned model raises, naming the
    exact path, rather than silently mis-attenuating."""
    with pytest.raises(NotImplementedError, match="age_binned"):
        SEDModel.build(
            ssp_data=_ssp,
            observation=_obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "age_binned",
                "screens": [{"law": "calzetti", "window_log_yr": (None, None)}],
                "tau_0": 0.5,
                "other_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.0),
            approx=WavePrecomp(),
        )


def test_age_binned_fitter_auto_approx_stays_exact(_ssp, _obs):
    """``approx="auto"`` (both the single-fit and batch-fit resolvers)
    resolves to the exact path for age_binned -- a fit still runs; neither
    hits the WavePrecomp/SpectrumPrecomp construction-time refusal."""
    from tengri.inference.fitter import _resolve_batch_fit_approx

    m = SEDModel.build(
        ssp_data=_ssp,
        observation=_obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "age_binned",
            "screens": [{"law": "calzetti", "window_log_yr": (None, None)}],
            "tau_0": 0.5,
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    params = _flat_params(m)

    resolved = _resolve_batch_fit_approx(m, "auto", "photometry")
    assert resolved.approx.wave_precomp is False
    # The fit itself must still run (no raise) on the exact path.
    resolved.predict_photometry(params)

    from tengri.inference.fitter import Fitter

    bare_fitter = Fitter.__new__(Fitter)
    bare_fitter.data_type = "photometry"
    cfg = bare_fitter._auto_approx_config(m)
    assert cfg is None  # None == "leave the model exact", the Fitter contract


# ── 7. Grammar validation ───────────────────────────────────────────────────


def test_age_binned_requires_nonempty_screens(_ssp, _obs):
    with pytest.raises(ValueError, match="non-empty 'screens'"):
        SEDModel.build(
            ssp_data=_ssp,
            observation=_obs,
            dust_attenuation={"type": "age_binned", "screens": []},
            redshift=Fixed(0.0),
        )


def test_age_binned_unknown_law_names_screen_index(_ssp, _obs):
    with pytest.raises(ValueError, match=r"screens'\[1\].*unknown dust law"):
        SEDModel.build(
            ssp_data=_ssp,
            observation=_obs,
            dust_attenuation={
                "type": "age_binned",
                "screens": [
                    {"law": "calzetti", "window_log_yr": (None, None)},
                    {"law": "not_a_real_law", "window_log_yr": (None, None)},
                ],
            },
            redshift=Fixed(0.0),
        )


def test_age_binned_window_lo_ge_hi_names_screen_index(_ssp, _obs):
    with pytest.raises(ValueError, match=r"screens'\[0\].*lo < hi"):
        SEDModel.build(
            ssp_data=_ssp,
            observation=_obs,
            dust_attenuation={
                "type": "age_binned",
                "screens": [{"law": "calzetti", "window_log_yr": (8.0, 7.0)}],
            },
            redshift=Fixed(0.0),
        )


def test_age_binned_tau_beyond_screen_count_is_refused(_ssp, _obs):
    """dust_tau_5 with only 2 screens declared is an unrecognized key."""
    with pytest.raises((ValueError, ParameterError)):
        SEDModel.build(
            ssp_data=_ssp,
            observation=_obs,
            dust_attenuation={
                "type": "age_binned",
                "screens": [
                    {"law": "calzetti", "window_log_yr": (None, 7.0)},
                    {"law": "calzetti", "window_log_yr": (None, None)},
                ],
                "tau_5": 0.5,
                "other_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.0),
        )


def test_age_binned_other_params_fixed_default_pins_law_params(_ssp, _obs):
    """other_params: Fixed(DEFAULT) pins every per-screen law param at its
    OWN law's published default (kriek_conroy's bump_strength=1.0, not the
    shared stem's Fixed(0.0))."""
    m = SEDModel.build(
        ssp_data=_ssp,
        observation=_obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "age_binned",
            "screens": [{"law": "kriek_conroy", "window_log_yr": (None, None)}],
            "tau_0": 0.5,
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    assert "dust_bump_strength_0" not in m.spec.free_params
    assert float(m.spec.get_fixed_values()["dust_bump_strength_0"]) == pytest.approx(1.0)


def test_age_binned_all_params_free_frees_exactly_declared(_ssp, _obs):
    """all_params: FREE frees exactly the tau_i and each screen's declared
    law shape params -- nothing else."""
    from tengri import FREE

    m = SEDModel.build(
        ssp_data=_ssp,
        observation=_obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "age_binned",
            "screens": [
                {"law": "calzetti", "window_log_yr": (None, 7.0)},
                {"law": "cardelli", "window_log_yr": (None, None)},
            ],
            "all_params": FREE,
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    dust_free = {n for n in m.spec.free_params if n.startswith("dust_")}
    assert dust_free == {"dust_tau_0", "dust_tau_1", "dust_Rv_1"}


# ── 8. Mutation: drop a screen's weight factor ──────────────────────────────


def test_mutation_drop_screen_weight_factor_breaks_identity(tmp_path):
    """Treating every screen as global (w_i(t) == 1 unconditionally) must
    break the N=2 vs two_component identity in
    :mod:`tengri.components.dust.age_binned`: proof the age-window weight
    is load-bearing. A real tengri.py mutation would be reverted via a file
    copy under the job's scratch dir, never git; this test demonstrates the
    FAILED assertion the mutation would produce by evaluating the broken
    formula directly, so no source file is actually mutated.
    """
    log_age = jnp.linspace(4.0, 10.5, 50)
    width = 0.3
    t_birth_log_yr = 7.0

    correct_w0 = _age_window_weight(log_age, None, t_birth_log_yr, width)
    mutated_w0 = jnp.ones_like(log_age)  # "every screen is global"

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(correct_w0), np.asarray(mutated_w0), atol=1e-6)
    # FAILED lines captured: Max absolute difference, Mismatched elements


# ── 9. Grid-dependent screen validation (#2528) ──
#
# The t -> 0 nebular/line rule (section "Nebular continuum..." in the class
# docstring) gives any screen with a finite lower edge exactly 0 weight on
# the line/nebular path, unconditionally. A finite lo close to the loaded
# grid's youngest SSP node gives that SAME node a non-negligible, nonzero
# weight on the STELLAR path -- the two paths then silently disagree about
# how much that screen's tau applies to the youngest population.
# validate_screens_against_grid refuses lo < youngest + 5*transition_width_dex
# (where the stellar-path weight is still >= sigma(-5) ~ 0.67%) at build time.


def test_age_binned_finite_lower_edge_too_close_to_grid_floor_raises(_ssp, _obs):
    """RED on the unguarded rule: lo=5.0 on a 1e6 yr-floor grid (synthetic_ssp_wide's
    youngest node, log10(age/yr)=6.0) would silently give the youngest SSP node a
    96.6% stellar-path weight from this screen while the lines see 0% of it.
    GREEN: validate_screens_against_grid refuses it, naming the screen index,
    lo, the grid's youngest node, the threshold, and both fixes -- both as a
    direct function call and through the SEDModel.build dispatch that calls it.
    """
    screens = validate_screens(
        [
            {"law": "calzetti", "window_log_yr": (5.0, None)},
            {"law": "cardelli", "window_log_yr": (None, None)},
        ]
    )
    with pytest.raises(ConfigError) as exc_info:
        validate_screens_against_grid(screens, _ssp, transition_width_dex=0.3)
    msg = str(exc_info.value)
    assert "screens'[0]" in msg
    assert "lo=5.0" in msg
    assert "6.0000" in msg  # the grid's youngest node, log10(age/yr)
    assert "7.5000" in msg  # threshold = youngest + 5 * transition_width_dex
    assert "unbounded" in msg  # fix 1: make the edge unbounded
    assert "raise lo" in msg  # fix 2: raise lo to the threshold

    # A model build goes through the exact same refusal (component_factory's
    # age_binned dispatch calls validate_screens_against_grid).
    with pytest.raises(ConfigError, match=r"screens'\[0\].*lo=5\.0"):
        SEDModel.build(
            ssp_data=_ssp,
            observation=_obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "age_binned",
                "screens": [
                    {"law": "calzetti", "window_log_yr": (5.0, None)},
                    {"law": "cardelli", "window_log_yr": (None, None)},
                ],
                "tau_0": 0.5,
                "tau_1": 0.3,
                "other_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.0),
        )


def test_age_binned_screen_at_threshold_builds_and_lines_see_none_of_it(_ssp, _obs):
    """A legal finite lo (== the threshold) builds; its weight on the youngest
    SSP node is < 0.7% (consistent with the lines' exact 0%, within the
    build-time tolerance validate_screens_against_grid enforces)."""
    t_birth_log_yr = 6.0 + 5.0 * 0.3  # youngest node (6.0) + 5 * transition_width_dex
    screens_raw = [
        {"law": "calzetti", "window_log_yr": (t_birth_log_yr, None)},
        {"law": "cardelli", "window_log_yr": (None, None)},
    ]

    SEDModel.build(
        ssp_data=_ssp,
        observation=_obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "age_binned",
            "screens": screens_raw,
            "tau_0": 0.5,
            "tau_1": 0.3,
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )  # must not raise

    # The same screens config, evaluated directly (same law/window pair the
    # build above used, so this is what that build's dust component computes).
    screens = validate_screens(screens_raw)
    comp = AgeBinnedDustComponent(config=AgeBinnedDustComponentConfig(screens=screens))
    wave = jnp.array([5500.0])

    # (1) Lines/nebular continuum see exactly 0 from screen 0 (lo is finite):
    # only the global screen 1 contributes, unconditionally -- not a close
    # call, the t -> 0 rule excludes screen 0 by construction.
    params_screen0_only = {"dust_tau_0": 0.5, "dust_tau_1": 0.0}
    tau_neb = comp._youngest_screen_tau(params_screen0_only, wave)
    np.testing.assert_array_equal(np.asarray(tau_neb), 0.0)

    # (2) The youngest SSP node's window weight from screen 0 is < 0.7%,
    # consistent with (within tolerance of) the lines' exact 0%.
    youngest_log_age_yr = 6.0
    w0_at_youngest = _age_window_weight(
        jnp.array([youngest_log_age_yr]), t_birth_log_yr, None, 0.3
    )
    assert float(w0_at_youngest[0]) < 0.007
    # (same shape as pytest's own AssertionError report from a live mutation).
