# SPDX-License-Identifier: BSD-3-Clause
"""#2320: ``agn_norm='conserving'`` beside an active fracAGN sets only the covering factor.

Owner ruling (2026-10-31, option 1): under fracAGN (``agn_ir_frac = f > 0``) the
``'conserving'`` policy keeps the AGN power at ``agn_lum_ratio x 10**agn_log_lbol L_sun``
and fracAGN sets only ``agn_torus_frac``. The torus re-emits that share of the accretion
power and the disc keeps the rest, so the energy ledger closes on ``agn_log_lbol``. Before
the ruling the build was refused (R67, ``_validate_fracagn_requires_cigale_joint``).

The mapping (``AGNSEDComponent.apply`` Notes), total dust:

    t = min(t_max, L_absorbed f / ((1 - f) agn_lum_ratio L_bol)),

so the AGN dust is the fraction ``f`` of ``AGN dust + L_absorbed``; ``t_max = 1``, or
``1 - E_lines/E_disc`` with a line block. With ``ir_frac_window`` ``t`` solves
``W_agn(t) = H f/(1 - f)``: exactly by two passes on the affine window light, and by Newton
steps from that chord beside ``polar_dust``, where the window light is rational in ``t``.
Where ``f`` is out of reach ``t`` stays at the bound, ``agn_ir_frac_saturated`` is 1 and the
``agn_ir_frac_realized`` property reports the fraction the model reaches.

Checked here: admission and the screened refusal (both build paths); the realized fraction
(total dust, two windows, the polar-dust window at the review's points) against what the
model emitted; the saturation flag at both bounds, beside the line bound; the ledger closure;
gradients against central finite differences, inside and at a bound; the Fritz + polar-dust
exception (#3100); and ``'cigale_joint'`` against a reference frozen on main.

The synthetic SSP's flux scale is arbitrary: at ``log_total_mass = -7`` it absorbs
``L_absorbed ~ 1.3e43 erg/s``, the order of an ``agn_log_lbol = 10`` AGN, which puts ``t``
inside ``(0, 1)``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform
from tengri.components.agn._ir_frac_window import _window_q
from tengri.config.exceptions import AdvisoryWarning, ConfigError
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug

_LOG_LBOL = 10.0  # the registry default of agn_log_lbol
_LEDGER = (
    "log_L_agn_disc",
    "log_L_agn_torus",
    "log_L_agn_lines",
    "log_L_agn_polar_dust",
    "log_L_agn_disc_unemitted",
    "log_L_agn_torus_unemitted",
    "log_L_agn_lines_unemitted",
)
#: The ledger's C2 tolerance (tests/contract/test_agn_composable_ledger_closes.py).
_RTOL = 1.0e-6
#: The polar-dust window root is solved to this relative residual in the window fraction.
_ROOT_RTOL = 1.0e-9
_BAND = (5.0, 20.0)
_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "agn_2320_cigale_joint_reference"
    / "reference.npz"
)


def _build(
    ssp,
    *,
    norm="conserving",
    ir_frac=0.3,
    log_total_mass=-7.0,
    window=None,
    obs=None,
    log_lbol=None,
    screen="none",
    torus="skirtor",
    polar_ebv=None,
    lum_ratio=None,
    nlr=None,
    torus_frac=None,
):
    torus_group = {"type": torus, "all_params": Fixed(DEFAULT)}
    if torus_frac is not None:
        torus_group["agn_torus_frac"] = torus_frac
    agn = {
        "type": "composable",
        "norm": norm,
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": torus_group,
        "agn_ir_frac": ir_frac if not isinstance(ir_frac, float) else Fixed(ir_frac),
        "all_params": Fixed(DEFAULT),
    }
    if polar_ebv is not None:
        agn["atten"] = {
            "type": "polar_dust",
            "all_params": Fixed(DEFAULT),
            "agn_polar_ebv": Fixed(polar_ebv),
        }
    if nlr is not None:
        agn["nlr"] = {"type": nlr, "all_params": Fixed(DEFAULT)}
    if window is not None:
        agn["ir_frac_window"] = window
    if log_lbol is not None:
        agn["agn_log_lbol"] = log_lbol
    if lum_ratio is not None:
        agn["agn_lum_ratio"] = Fixed(lum_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp,
            observation=obs,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(log_total_mass),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
                "agn_screen": screen,
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            agn=agn,
            redshift=Fixed(0.05),
        )


def _ledger(model, params=None):
    props = model.predict_properties(params or {}, names=(*_LEDGER, "agn_ledger_closed"))
    return {k: float(v) for k, v in props.items()}


def _closure(ledger, log_lbol=_LOG_LBOL, lum_ratio=1.0):
    """C2: pieces plus unemitted over the accretion power, minus one."""
    total = sum(10.0 ** ledger[k] for k in _LEDGER)
    return total / (lum_ratio * 10.0**log_lbol * L_SUN) - 1.0


def _dust_fraction(model):
    """AGN dust (angle-integrated torus + polar dust) over AGN dust + L_absorbed."""
    state = model.predict_state({})
    ledger = _ledger(model)
    agn_dust = 10.0 ** ledger["log_L_agn_torus"] + 10.0 ** ledger["log_L_agn_polar_dust"]
    return agn_dust / (agn_dust + 10.0 ** float(state.derived["log_L_absorbed"]))


def _window_fraction(model, window, params=None):
    """AGN share of the light in the window, from the emitted SEDs [dimensionless]."""
    state = model.predict_state(params or {})
    wave = np.asarray(state.wave)
    agn = np.asarray(state.derived["sed_agn"])
    host = np.asarray(state.sed_intrinsic) - agn
    lo, hi = window[0] * 1.0e4, window[1] * 1.0e4
    if lo == hi:
        w_agn = float(np.interp(lo, wave, agn / wave**2))
        w_host = float(np.interp(lo, wave, host / wave**2))
    else:
        w_agn = float(_window_q(wave, agn, lo, hi))
        w_host = float(_window_q(wave, host, lo, hi))
    return w_agn / (w_agn + w_host)


def _derived(model, key, params=None):
    return float(model.predict_state(params or {}).derived[key])


def _realized(model, params=None):
    props = model.predict_properties(params or {}, names=("agn_ir_frac_realized",))
    return float(props["agn_ir_frac_realized"])


# ── admission ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ir_frac", [Fixed(0.3), FREE], ids=["fixed", "free"])
def test_conserving_with_active_fracagn_builds(synthetic_ssp_wide, ir_frac):
    """The refusal is gone: Fixed and FREE fracAGN both build under 'conserving'."""
    model = _build(synthetic_ssp_wide, ir_frac=ir_frac)
    assert model.spec.agn_norm == "conserving"


def test_screened_agn_with_fracagn_is_refused(synthetic_ssp_wide):
    """A screened AGN runs before the dust, so the absorbed power fracAGN reads does not exist
    yet: refused, instead of falling back to the user ``agn_torus_frac``."""
    with pytest.raises(ConfigError, match="dust_agn_screen='diffuse'"):
        _build(synthetic_ssp_wide, screen="diffuse")
    # Without the coupling the screened conserving AGN is legal.
    model = _build(synthetic_ssp_wide, ir_frac=Fixed(0.0), screen="diffuse")
    assert model.spec.agn_norm == "conserving"


def _flat(norm, screen, **extra):
    from tengri.parameters.parameters import Parameters

    return Parameters(
        agn_model="composable",
        agn_disc_block="schartmann2005",
        agn_torus_block="skirtor",
        agn_norm=norm,
        dust_agn_screen=screen,
        **extra,
    )


@pytest.mark.parametrize("norm", ["conserving", "independent"])
@pytest.mark.parametrize(
    "ir_frac", [Fixed(0.3), Uniform(0.0, 0.5)], ids=["fixed-0.3", "narrowed-prior"]
)
def test_flat_screened_explicit_fracagn_is_refused(norm, ir_frac):
    """Flat path: an agn_ir_frac the caller set beside a screened AGN raises (no fallback)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ConfigError, match="dust_agn_screen='diffuse'"):
            _flat(norm, "diffuse", agn_ir_frac=ir_frac)
        assert "agn_ir_frac" not in _flat(norm, "diffuse", agn_ir_frac=Fixed(0.0)).free_params


@pytest.mark.parametrize("norm", ["conserving", "independent"])
def test_flat_screened_default_fracagn_warns_and_is_dropped(norm):
    """Flat path: an agn_ir_frac freed only by its declared prior would be dead beside a screen;
    it is fixed at 0.0 with a warning naming it."""
    with pytest.warns(AdvisoryWarning, match="agn_ir_frac is free only through its declared"):
        spec = _flat(norm, "diffuse")
    assert "agn_ir_frac" not in spec.free_params
    assert float(spec.get_fixed_values()["agn_ir_frac"]) == 0.0


@pytest.mark.parametrize("norm", ["conserving", "independent"])
def test_flat_unscreened_fracagn_is_unchanged(norm):
    """Flat path without a screen: the default-prior agn_ir_frac stays free, and the screened
    settlement does not warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = _flat(norm, "none")
    assert "agn_ir_frac" in spec.free_params
    assert not [w for w in caught if "agn_ir_frac is free only through" in str(w.message)]


def test_build_screened_wildcard_fracagn_warns_and_is_dropped(synthetic_ssp_wide):
    """Build path: an ``all_params: FREE`` wildcard frees agn_ir_frac only by its declared prior;
    beside a screen it is fixed at 0.0 with a warning, and the build succeeds."""
    with pytest.warns(AdvisoryWarning, match="agn_ir_frac is free only through its declared"):
        model = SEDModel.build(
            synthetic_ssp_wide,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "agn_screen": "diffuse",
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            agn={
                "type": "composable",
                "norm": "conserving",
                "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                "all_params": FREE,
            },
            redshift=Fixed(0.05),
        )
    assert "agn_ir_frac" not in model.spec.free_params


def test_build_screened_named_free_fracagn_is_refused(synthetic_ssp_wide):
    """Build path: ``agn_ir_frac: FREE`` written by name is the user's choice, so beside a screen
    it raises like a Fixed value above 0 or a custom prior (only the wildcard is dropped)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ConfigError, match="dust_agn_screen='diffuse'"):
            SEDModel.build(
                synthetic_ssp_wide,
                sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                    "agn_screen": "diffuse",
                },
                dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
                agn={
                    "type": "composable",
                    "norm": "conserving",
                    "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
                    "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                    "agn_ir_frac": FREE,
                    "all_params": Fixed(DEFAULT),
                },
                redshift=Fixed(0.05),
            )


def test_flat_screened_named_declared_prior_is_refused():
    """Flat path: naming agn_ir_frac with its declared prior is still the caller's choice."""
    from tengri.parameters.parameters import Parameters

    declared = Parameters(agn_model="composable").get_distribution("agn_ir_frac")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ConfigError, match="dust_agn_screen='diffuse'"):
            _flat("conserving", "diffuse", agn_ir_frac=declared)


def test_build_unscreened_wildcard_fracagn_is_unchanged(synthetic_ssp_wide):
    """Build path without a screen: the wildcard leaves agn_ir_frac free."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            synthetic_ssp_wide,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            agn={
                "type": "composable",
                "norm": "conserving",
                "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                "all_params": FREE,
            },
            redshift=Fixed(0.05),
        )
    assert "agn_ir_frac" in model.spec.free_params


# ── the realized fraction ──────────────────────────────────────────────────


@pytest.mark.parametrize("frac", [0.1, 0.3])
def test_total_dust_fraction_is_realized(synthetic_ssp_wide, frac):
    """The AGN dust is the fraction ``f`` of the total dust (measured 2.6e-8 relative), and the
    published realized fraction says so."""
    model = _build(synthetic_ssp_wide, ir_frac=frac)
    assert 0.0 < _derived(model, "agn_torus_frac_effective") < 1.0
    assert _dust_fraction(model) == pytest.approx(frac, rel=_RTOL)
    assert _realized(model) == pytest.approx(frac, rel=_RTOL)
    assert _derived(model, "agn_ir_frac_saturated") == 0.0


@pytest.mark.parametrize("window", [(5.0, 20.0), (10.0, 10.0)], ids=["band", "mono"])
def test_window_fraction_is_realized(synthetic_ssp_wide, window):
    """With a window, ``f`` is the AGN share of the window light, from the emitted SED."""
    model = _build(synthetic_ssp_wide, ir_frac=0.3, log_total_mass=-6.0, window=window)
    assert 0.0 < _derived(model, "agn_torus_frac_effective") < 1.0
    assert _window_fraction(model, window) == pytest.approx(0.3, rel=_RTOL)
    assert _realized(model) == pytest.approx(0.3, rel=_RTOL)
    assert _derived(model, "agn_ir_frac_saturated") == 0.0


#: The review's polar-dust window points where ``f = 0.3`` is reachable (``t`` in (0, 1)):
#: (torus, log_lbol, polar_ebv, lum_ratio). The chord missed them by -1.1%, -0.4% and +6.7%;
#: three Newton steps reach 6e-15, 7e-16 and 3e-14.
_POLAR_INTERIOR = [
    ("skirtor", 10.0, 0.03, None),
    ("fritz", 10.0, 0.03, None),
    ("skirtor", 11.0, 0.03, 0.3),
]


@pytest.mark.parametrize(
    ("torus", "log_lbol", "polar_ebv", "lum_ratio"),
    _POLAR_INTERIOR,
    ids=["skirtor", "fritz", "skirtor-r0.3"],
)
def test_polar_dust_window_root_is_exact(
    synthetic_ssp_wide, torus, log_lbol, polar_ebv, lum_ratio
):
    """Beside polar dust the window light is rational in ``t``: the solve still lands on ``f``."""
    model = _build(
        synthetic_ssp_wide,
        log_total_mass=-6.0,
        window=_BAND,
        torus=torus,
        polar_ebv=polar_ebv,
        lum_ratio=lum_ratio,
        log_lbol=Fixed(log_lbol),
    )
    assert 0.0 < _derived(model, "agn_torus_frac_effective") < 1.0
    assert _window_fraction(model, _BAND) == pytest.approx(0.3, rel=_ROOT_RTOL)
    assert _derived(model, "agn_ir_frac_saturated") == 0.0


#: The review's polar-dust window points the chord reported +70% and +63% at. Both are out of
#: reach: the disc's own share of the window light at ``t = 0`` (0.4536) already exceeds ``f``.
_POLAR_FLOOR = [
    ("skirtor", 11.0, 0.3, None),
    ("fritz", 11.0, 0.3, None),
]


@pytest.mark.parametrize(
    ("torus", "log_lbol", "polar_ebv", "lum_ratio"),
    _POLAR_FLOOR,
    ids=["skirtor-ebv0.3", "fritz-ebv0.3"],
)
def test_polar_dust_window_below_the_floor_saturates(
    synthetic_ssp_wide, torus, log_lbol, polar_ebv, lum_ratio
):
    """An unreachable window fraction: ``t = 0``, the flag is 1, and the realized fraction is
    the one a bare ``agn_torus_frac = 0`` model emits."""
    kwargs = dict(
        log_total_mass=-6.0,
        torus=torus,
        polar_ebv=polar_ebv,
        lum_ratio=lum_ratio,
        log_lbol=Fixed(log_lbol),
    )
    model = _build(synthetic_ssp_wide, window=_BAND, **kwargs)
    floor = _window_fraction(
        _build(synthetic_ssp_wide, ir_frac=Fixed(0.0), torus_frac=Fixed(0.0), **kwargs), _BAND
    )
    assert floor > 0.3
    assert _derived(model, "agn_torus_frac_effective") == 0.0
    assert _derived(model, "agn_ir_frac_saturated") == 1.0
    assert _realized(model) == pytest.approx(floor, rel=_RTOL)
    assert _window_fraction(model, _BAND) == pytest.approx(floor, rel=_RTOL)


# ── saturation ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("log_total_mass", [-7.0, -6.0, -5.5, -5.0])
def test_saturation_flag_fires_exactly_when_the_bound_binds(synthetic_ssp_wide, log_total_mass):
    """The flag is ``t_raw >= 1`` with ``t_raw = L_abs f/((1 - f) r L_bol)``; where it is 0 the
    realized fraction is ``f``, where it is 1 it is ``f_max = r L_bol/(r L_bol + L_abs)``."""
    model = _build(synthetic_ssp_wide, ir_frac=0.3, log_total_mass=log_total_mass)
    l_abs = 10.0 ** _derived(model, "log_L_absorbed")
    l_bol = 10.0**_LOG_LBOL * L_SUN
    t_raw = l_abs * 0.3 / (0.7 * l_bol)
    saturated = _derived(model, "agn_ir_frac_saturated")
    assert saturated == float(t_raw >= 1.0)
    expected = 0.3 if t_raw < 1.0 else l_bol / (l_bol + l_abs)
    assert _realized(model) == pytest.approx(expected, rel=_RTOL)
    assert _derived(model, "agn_torus_frac_effective") == pytest.approx(min(t_raw, 1.0), rel=1e-12)


@pytest.mark.parametrize(("log_total_mass", "saturated"), [(-7.0, 0.0), (-5.0, 1.0)])
def test_saturation_outputs_are_properties(synthetic_ssp_wide, log_total_mass, saturated):
    """Both outputs are reachable on the property surfaces: ``predict_properties`` (the
    jit/vmap surface) and ``pred.properties``, with the same values."""
    model = _build(synthetic_ssp_wide, ir_frac=0.3, log_total_mass=log_total_mass)
    names = ("agn_ir_frac_saturated", "agn_ir_frac_realized")
    assert set(names) <= set(model.available_properties)
    props = model.predict_properties({}, names=names)
    pred = model.predict({})
    assert float(props["agn_ir_frac_saturated"]) == saturated
    assert float(pred.properties["agn_ir_frac_saturated"]) == saturated
    assert float(pred.properties["agn_ir_frac_realized"]) == pytest.approx(
        float(props["agn_ir_frac_realized"]), rel=1e-12
    )
    if saturated:
        assert float(props["agn_ir_frac_realized"]) < 0.3
    else:
        assert float(props["agn_ir_frac_realized"]) == pytest.approx(0.3, rel=_RTOL)


def test_covering_factor_saturates_and_the_ledger_still_closes(synthetic_ssp_wide):
    """A host that asks for more AGN dust than the accretion power has: ``t = 1``, no more.

    The torus then re-emits the whole power, the realized fraction falls below ``f``, and the
    pieces still add up to ``L_bol``.
    """
    model = _build(synthetic_ssp_wide, ir_frac=0.3, log_total_mass=-5.0)
    assert _derived(model, "agn_torus_frac_effective") == 1.0
    assert _dust_fraction(model) < 0.3
    ledger = _ledger(model)
    assert ledger["agn_ledger_closed"] == 1.0
    assert _closure(ledger) == pytest.approx(0.0, abs=_RTOL)


def test_line_block_bounds_the_covering_factor(synthetic_ssp_wide):
    """With an NLR the torus and the lines share the disc's power: ``t <= 1 - E_lines/E_disc``.

    At saturation the runner applies that bound, the flag is 1 and the ledger closes (before the
    bound C2 was +1.3e-2 here while ``agn_ledger_closed`` read 1).
    """
    model = _build(synthetic_ssp_wide, ir_frac=0.3, log_total_mass=-5.0, nlr="analytic")
    ledger = _ledger(model)
    lines = 10.0 ** ledger["log_L_agn_lines"] / (10.0**_LOG_LBOL * L_SUN)
    t = _derived(model, "agn_torus_frac_effective")
    assert t == pytest.approx(1.0 - lines, rel=1e-6)
    assert t < 1.0
    assert _derived(model, "agn_ir_frac_saturated") == 1.0
    assert ledger["agn_ledger_closed"] == 1.0
    assert _closure(ledger) == pytest.approx(0.0, abs=_RTOL)


def test_ledger_closes_on_agn_log_lbol(synthetic_ssp_wide):
    """``agn_ledger_closed == 1`` and the pieces plus unemitted are ``L_bol`` to 1e-6 (C2)."""
    ledger = _ledger(_build(synthetic_ssp_wide, ir_frac=0.3))
    assert ledger["agn_ledger_closed"] == 1.0
    assert _closure(ledger) == pytest.approx(0.0, abs=_RTOL)


def test_fritz_polar_dust_total_mapping_is_bounded(synthetic_ssp_wide):
    """Fritz + polar dust: the ledger does not close (#3100), and the realized total-dust
    fraction exceeds ``f`` by at most the measured 3.25%; the property reports it."""
    model = _build(synthetic_ssp_wide, ir_frac=0.3, torus="fritz", polar_ebv=0.1)
    assert _ledger(model)["agn_ledger_closed"] == 0.0
    realized = _realized(model)
    assert realized == pytest.approx(_dust_fraction(model), rel=1e-9)
    assert 0.0 < realized / 0.3 - 1.0 <= 0.0325


# ── gradients ──────────────────────────────────────────────────────────────

_FD_STEP = 1.0e-5


def _fd_check(model, params):
    """AD against central finite differences of ``sum(log10 photometry)``, per parameter."""

    def objective(p):
        return jnp.sum(jnp.log10(model.predict_photometry(p)))

    grads = jax.grad(objective)(params)
    for name, g in grads.items():
        up = {**params, name: params[name] + _FD_STEP}
        down = {**params, name: params[name] - _FD_STEP}
        fd = float(objective(up) - objective(down)) / (2.0 * _FD_STEP)
        # grad-assert: finite-only — a zero is correct at a bound; the FD comparison checks it
        assert np.isfinite(float(g)), name
        assert float(g) == pytest.approx(fd, rel=1e-5, abs=1e-8), name
    return {k: float(v) for k, v in grads.items()}


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("total", dict()),
        ("polar_window", dict(log_total_mass=-6.0, window=_BAND, polar_ebv=0.03)),
    ],
)
def test_gradients_match_finite_differences(
    synthetic_ssp_wide, synthetic_tophat_obs, case, kwargs
):
    """d(photometry)/d(agn_ir_frac, agn_log_lbol) equals central finite differences, and is
    nonzero inside the reachable range."""
    model = _build(
        synthetic_ssp_wide, ir_frac=FREE, obs=synthetic_tophat_obs, log_lbol=FREE, **kwargs
    )
    params = {"agn_ir_frac": jnp.asarray(0.3), "agn_log_lbol": jnp.asarray(_LOG_LBOL)}
    grads = _fd_check(model, params)
    # grad-assert: nonzero-only — _fd_check asserted both finite
    assert grads["agn_ir_frac"] != 0.0
    assert grads["agn_log_lbol"] != 0.0


def test_gradient_in_the_saturated_region(synthetic_ssp_wide, synthetic_tophat_obs):
    """At ``t = 1`` the photometry does not depend on ``agn_ir_frac``: AD = FD = 0."""
    model = _build(
        synthetic_ssp_wide,
        ir_frac=FREE,
        obs=synthetic_tophat_obs,
        log_lbol=FREE,
        log_total_mass=-5.0,
    )
    params = {"agn_ir_frac": jnp.asarray(0.3), "agn_log_lbol": jnp.asarray(_LOG_LBOL)}
    grads = _fd_check(model, params)
    assert grads["agn_ir_frac"] == 0.0


# ── 'cigale_joint' unchanged ──────────────────────────────────────────────

#: The reference is frozen on main (``generate.py`` beside it records the SHA). XLA's floating
#: point moves by ~1e-7 between JAX versions and CPU targets (#1763), so a frozen file is held
#: to the repository's golden tolerance rather than bitwise.
_GOLDEN_RTOL = 1.0e-6


@pytest.mark.parametrize(
    ("name", "window", "log_total_mass"), [("total", None, -5.0), ("window", _BAND, -6.0)]
)
def test_cigale_joint_matches_the_main_reference(synthetic_ssp_wide, name, window, log_total_mass):
    """'cigale_joint' + fracAGN reproduces the reference frozen on main: the SED and the
    unclipped ``t`` (above 1 for the total-dust case, which 'conserving' would clip)."""
    ref = np.load(_REFERENCE)
    state = _build(
        synthetic_ssp_wide, norm="cigale_joint", window=window, log_total_mass=log_total_mass
    ).predict_state({})
    t = float(state.derived["agn_torus_frac_effective"])
    assert t == pytest.approx(float(ref[f"{name}_torus_frac"]), rel=_GOLDEN_RTOL)
    np.testing.assert_allclose(
        np.asarray(state.sed_intrinsic), ref[f"{name}_sed"], rtol=_GOLDEN_RTOL, atol=0.0
    )
    if name == "total":
        assert t > 1.0
        assert (
            _derived(_build(synthetic_ssp_wide, log_total_mass=-5.0), "agn_torus_frac_effective")
            == 1.0
        )
