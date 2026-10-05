# SPDX-License-Identifier: BSD-3-Clause
"""Age-binned dust attenuation: N independent screens, each its own law and age
window (#2528).

Generalizes :mod:`tengri.components.dust.two_component`'s Charlot & Fall
(2000) birth-cloud/diffuse-ISM pair to N screens. The identity test (1)
verifies ``age_binned`` with N=2 and the two_component windows reproduces
``two_component`` exactly; the rest cover the N-screen physics directly:
the exact window partition, k(5500 A)=1, energy balance, gradients, the LUT
refusal, grammar validation, the ionizing-weighted nebular screen and the
``lyc_`` family.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp
from tengri.components.dust._age_mixture import (
    interval_cover,
    interval_fractions,
    ionizing_interval_weights,
)
from tengri.components.dust.age_binned import (
    AgeBinnedDustComponent,
    AgeBinnedDustComponentConfig,
    validate_screens,
)
from tengri.components.dust.laws._registry import DUST_LAWS, resolve_dust_law
from tengri.config.exceptions import ParameterError
from tengri.protocols.component import ForwardState
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

_TILING = [
    {"law": "calzetti", "window_log_yr": (None, 7.0)},
    {"law": "power_law", "window_log_yr": (7.0, 8.5)},
    {"law": "cardelli", "window_log_yr": (8.5, None)},
]


def _tiling_component():
    return AgeBinnedDustComponent(
        config=AgeBinnedDustComponentConfig(screens=validate_screens(_TILING))
    )


def _window_weights(comp, younger):
    """Per-screen, per-node mass fraction inside the screen's window, ``(n_screen, n_age)``."""
    fractions = interval_fractions(jnp.asarray(younger))
    cover = interval_cover(comp.config.windows_yr, comp.config.age_boundaries_yr)
    return jnp.stack([sum(fractions[j] for j, c in enumerate(row) if c) for row in cover])


def test_age_binned_tiling_windows_partition_every_node_exactly(_ssp, _obs):
    """Three tiling screens: the windows' mass fractions sum to 1 at EVERY node (<= 1e-15).

    The stellar component publishes the exact per-node mass fraction younger than
    each edge, so the intervals telescope and the partition holds at the edges
    too, not only far from them (a product of independent logistic windows
    ran 0.3-0.6 % above 1 there).
    """
    m = SEDModel.build(
        ssp_data=_ssp,
        observation=_obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "age_binned",
            "screens": _TILING,
            "tau_0": 0.5,
            "tau_1": 1.0,
            "tau_2": 0.3,
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    younger = m.predict_state(_flat_params(m)).derived["age_boundary_younger_fraction"]
    assert younger.shape[0] == 2  # one row per finite edge (7.0, 8.5)
    weights = _window_weights(_tiling_component(), younger)
    np.testing.assert_allclose(np.asarray(weights.sum(axis=0)), 1.0, rtol=0.0, atol=1e-15)
    # the SSP node straddling an edge splits its mass between the two windows
    frac = np.asarray(younger)
    assert np.any((frac > 0.0) & (frac < 1.0)), "no node straddles an edge: the test is vacuous"


def test_age_binned_middle_screen_tau_only_changes_its_own_window():
    """Zeroing the middle screen's tau changes flux only in nodes its window covers.

    Evaluates single-population nodes: a node wholly inside the middle window
    must change when dust_tau_1 -> 0; a node wholly inside screen 0 or screen 2's
    window must not.
    """
    comp = _tiling_component()
    wave = jnp.linspace(1200.0, 20000.0, 200)
    ages = jnp.array([1.0, 10**7.75, 10.0**13])
    # fraction of each node's mass younger than 1e7 and than 10**8.5 yr
    younger = jnp.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    def transmission(tau1):
        params = {"dust_tau_0": 0.8, "dust_tau_1": tau1, "dust_tau_2": 0.5, "dust_Rv_2": 3.1}
        return np.asarray(comp.compute_transmission(params, wave, younger))

    with_tau, without = transmission(1.0), transmission(0.0)
    np.testing.assert_array_equal(with_tau[0], without[0])  # inside screen 0
    np.testing.assert_array_equal(with_tau[2], without[2])  # inside screen 2
    assert np.max(np.abs(with_tau[1] - without[1])) > 1e-3  # inside screen 1
    assert ages.shape[0] == younger.shape[1]


def test_age_binned_straddling_node_is_the_mass_weighted_mixture():
    """A node holding half its mass younger than the edge is the 50/50 mixture of
    the two populations' transmissions -- not the transmission of a mixed
    optical depth."""
    comp = _tiling_component()
    wave = jnp.linspace(1200.0, 20000.0, 50)
    params = {"dust_tau_0": 0.8, "dust_tau_1": 1.0, "dust_tau_2": 0.5, "dust_Rv_2": 3.1}
    both = np.asarray(comp.compute_transmission(params, wave, jnp.asarray([[0.5], [1.0]])))[
        0
    ]  # half younger than 1e7, none older than 10**8.5
    young = np.asarray(comp.compute_transmission(params, wave, jnp.asarray([[1.0], [1.0]])))[0]
    mid = np.asarray(comp.compute_transmission(params, wave, jnp.asarray([[0.0], [1.0]])))[0]
    np.testing.assert_allclose(both, 0.5 * young + 0.5 * mid, rtol=1e-14, atol=0.0)
    optical_depth_mix = np.exp(-0.5 * (-np.log(young) - np.log(mid)))
    assert np.max(np.abs(both - optical_depth_mix)) > 1e-4  # it is NOT the optical-depth blend


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
    # A finite-lower-edge middle screen builds: the nebular light weighs it by its
    # share of the ionizing luminosity.
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
    """jax.grad wrt every dust_tau_i and a law parameter: finite, nonzero, and within
    1e-5 relative of central finite differences, on nodes that straddle both edges.

    The window edges and the dispersal width are static build-time constants,
    so the optical depths and the law parameters are the differentiable
    inputs here; the gradient through the SFH (including a cell that straddles
    an edge) is pinned in ``tests/physics/gradients/test_age_boundary_gradient.py``.
    A 1% tolerance left four orders of magnitude of undetected slack: central
    FD at eps=1e-4 has an O(eps^2) truncation floor of ~1e-8 relative (the
    third-derivative term), and the measured agreement here is ~1e-9-1e-10
    relative per tau_i. 1e-5 keeps ~3 orders of margin above that floor.
    """
    comp = _tiling_component()
    wave = jnp.linspace(1200.0, 20000.0, 100)
    younger = jnp.asarray(
        [[1.0, 0.8, 0.3, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.9, 0.4, 0.0]]
    )  # nodes straddling both edges

    def objective(tau0, tau1, tau2, rv2):
        params = {"dust_tau_0": tau0, "dust_tau_1": tau1, "dust_tau_2": tau2, "dust_Rv_2": rv2}
        return jnp.sum(comp.compute_transmission(params, wave, younger))

    x0 = (0.5, 1.0, 0.3, 3.1)
    grads = jax.grad(objective, argnums=(0, 1, 2, 3))(*x0)
    for g in grads:
        assert jnp.isfinite(g)
        assert float(jnp.abs(g)) > 0.0

    eps = 1e-4
    for k, g in enumerate(grads):
        up = list(x0)
        down = list(x0)
        up[k] += eps
        down[k] -= eps
        fd = (objective(*up) - objective(*down)) / (2 * eps)
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


# ── 8. Nebular continuum and lines: the ionizing-weighted interval mixture ──


def _hand_state(younger, log_l_lyc, *, n_wave=40, with_lyc=False, fesc=0.3):
    """A hand-built stellar publication for a component driven directly."""
    wave = jnp.logspace(2.7, 4.3, n_wave)
    n_age = younger.shape[1]
    lnu_age = jnp.ones((n_age, n_wave)) * 1e29
    sed_neb = jnp.ones(n_wave) * 2e28
    derived = {
        "lnu_age": lnu_age,
        "ssp_ages_yr": jnp.logspace(6.0, 10.0, n_age),
        "age_boundary_younger_fraction": younger,
        "log_L_lyc_age": log_l_lyc,
        "sed_nebular": sed_neb,
        "line_waves": jnp.asarray([4861.0, 6563.0]),
        "log_line_lums": jnp.asarray([40.0, 41.0]),
    }
    if with_lyc:
        derived["lyc_transmission"] = jnp.where(wave < LYMAN_LIMIT_AA, fesc, 1.0)
        derived["lyc_fesc"] = jnp.asarray(fesc)
    return ForwardState(
        wave=wave, sed_intrinsic=jnp.sum(lnu_age, axis=0) + sed_neb, derived=derived
    )


_HAND_PARAMS = {"dust_tau_0": 0.8, "dust_tau_1": 1.0, "dust_tau_2": 0.5, "dust_Rv_2": 3.1}


def test_a_finite_lower_edge_screen_attenuates_lines_by_its_ionizing_weighted_share():
    """The middle screen has a finite lower edge. It is not refused: the lines see
    q_1 of its transmission, q_1 its share of the ionizing luminosity."""
    comp = _tiling_component()
    younger = jnp.asarray([[1.0, 0.8, 0.3, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.9, 0.4, 0.0]])
    log_l = jnp.asarray([38.0, 37.5, 37.0, 36.0, 35.0, 34.0])
    state = _hand_state(younger, log_l)
    out = comp.apply(state, _HAND_PARAMS)

    fractions = np.asarray(interval_fractions(younger))
    weights = 10.0 ** np.asarray(log_l)
    q = fractions @ weights / weights.sum()
    assert 0.0 < q[1] < 1.0 and abs(q.sum() - 1.0) < 1e-14

    def tau(i, law, wave, **kw):
        return _HAND_PARAMS[f"dust_tau_{i}"] * np.asarray(resolve_dust_law(law)(wave, **kw))

    line_wave = np.asarray(state.derived["line_waves"])
    t_int = [
        np.exp(-tau(0, "calzetti", line_wave)),
        np.exp(-tau(1, "power_law", line_wave)),
        np.exp(-tau(2, "cardelli", line_wave, dust_Rv=3.1)),
    ]
    want = sum(q[j] * t_int[j] for j in range(3))
    got = 10.0 ** (
        np.asarray(out.derived["log_line_lums_attenuated"])
        - np.asarray(state.derived["log_line_lums"])
    )
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)
    # and it differs from the old youngest-age rule (screens unbounded below only)
    assert np.max(np.abs(got - t_int[0])) > 1e-3


def test_no_ionizing_light_puts_the_nebular_screen_on_the_young_limit():
    comp = _tiling_component()
    younger = jnp.asarray([[1.0, 0.8, 0.3, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.9, 0.4, 0.0]])
    state = _hand_state(younger, jnp.full((6,), -jnp.inf))
    out = comp.apply(state, _HAND_PARAMS)
    line_wave = np.asarray(state.derived["line_waves"])
    want = np.exp(
        -_HAND_PARAMS["dust_tau_0"] * np.asarray(resolve_dust_law("calzetti")(line_wave))
    )
    got = 10.0 ** (
        np.asarray(out.derived["log_line_lums_attenuated"])
        - np.asarray(state.derived["log_line_lums"])
    )
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)


def test_default_fixture_young_share_of_the_ionizing_luminosity(_ssp, _obs):
    """q_young for the default two_component fixture: below 1 (old stars emit LyC)."""
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
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    derived = m.predict_state(_flat_params(m)).derived
    q = np.asarray(
        ionizing_interval_weights(
            interval_fractions(derived["age_boundary_younger_fraction"]),
            derived["log_L_lyc_age"],
        )
    )
    assert abs(q.sum() - 1.0) < 1e-14
    assert 0.0 < q[0] <= 1.0


# ── 9. The lyc_ family on age_binned, N = 2 identical to two_component ──────


@pytest.mark.parametrize(
    "mode,geometry",
    [
        ("young", "screened"),
        ("all", "screened"),
        ("young", "birth_cloud_holes"),
        ("young", "clear"),
    ],
)
@pytest.mark.parametrize("lyc_in_eb", [False, True])
def test_age_binned_n2_lyc_family_matches_two_component(mode, geometry, lyc_in_eb):
    """The same hand state through both attenuators: the sed, the absorbed energy and
    the HII-dust credit agree (bit-for-bit where no hole raw-curve product enters)."""
    from tengri.components.dust.two_component import DustSEDComponent, DustSEDComponentConfig

    younger = jnp.asarray([[1.0, 0.7, 0.2, 0.0, 0.0, 0.0]])
    log_l = jnp.asarray([38.0, 37.5, 37.0, 36.0, 35.0, 34.0])
    state = _hand_state(younger, log_l, with_lyc=True)
    state = state.with_(
        derived=state.derived.with_(
            lyc_fdust=jnp.asarray(0.2),
            log_L_lyc=jnp.asarray(38.2),
            lnu_age_ion=None,
            ssp_wave_ion=None,
            log_stellar_mass_scale=jnp.asarray(0.0),
        )
    )
    kw = dict(
        lyc_reprocessed_by=mode, lyc_escape_geometry=geometry, lyc_in_energy_balance=lyc_in_eb
    )
    two = DustSEDComponent(
        config=DustSEDComponentConfig(law_bc="calzetti", law_diff="cardelli", **kw)
    )
    ab = AgeBinnedDustComponent(
        config=AgeBinnedDustComponentConfig(
            screens=validate_screens(
                [
                    {"law": "calzetti", "window_log_yr": (None, 7.0)},
                    {"law": "cardelli", "window_log_yr": (None, None)},
                ]
            ),
            **kw,
        )
    )
    out2 = two.apply(
        state, {"dust_tau_bc": 1.0, "dust_tau_diff": 0.4, "dust_Rv": 3.1, "redshift": 0.0}
    )
    outa = ab.apply(state, {"dust_tau_0": 1.0, "dust_tau_1": 0.4, "dust_Rv_1": 3.1})
    rtol = 0.0 if geometry == "screened" else 1e-12
    for key in ("sed_dust_attenuated", "log_L_absorbed", "log_line_lums_attenuated"):
        np.testing.assert_allclose(
            np.asarray(out2.derived[key]), np.asarray(outa.derived[key]), rtol=rtol, atol=0.0
        )
    np.testing.assert_allclose(
        np.asarray(out2.sed_intrinsic), np.asarray(outa.sed_intrinsic), rtol=rtol, atol=0.0
    )
