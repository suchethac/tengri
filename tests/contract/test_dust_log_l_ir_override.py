# SPDX-License-Identifier: BSD-3-Clause
"""``dust_log_L_ir``: the total dust IR budget override (#2187-series).

Declaring ``dust_log_L_ir`` -- Fixed or any free prior -- REPLACES the
energy-balance IR budget ``log_L_ir = log_L_absorbed + log10(dust_eta_balance)``
outright with the declared value; leaving it undeclared keeps strict/relaxed
energy balance exactly as before. Presence is a build-time provenance
question (``SEDModel._requested_dust_log_L_ir``), never a value sentinel.

Covers, per the implementation plan:

- the publish branch in all three attenuation publishers (two_component,
  single_component, wg00)
- the declared-prior-frees / undeclared-is-bit-identical parse contract
- radio following the override (FIRRC amplitudes read ``L_ir``)
- the ``dust_eta_balance`` inertness guard
- the fast-path allowlists (energy-balance LUT + band-response gate) staying
  live under a declared/free override, while BOSA's own (unrelated)
  non-homogeneity keeps its own fast path correctly disabled
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import ParameterError
from tengri.forward.sed_model import WavePrecomp
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.contract

LOG10_L_SUN = float(np.log10(L_SUN))
_C_AA_PER_S = 2.99792458e18


def _synthetic_ssp() -> SSPData:
    n_age = 25
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


@pytest.fixture(scope="module")
def ssp():
    return _synthetic_ssp()


def _build(
    ssp,
    *,
    dust_model="two_component",
    emission_type="dale2014",
    log_l_ir=None,
    eta=None,
    tau_bc=1.0,
    tau_diff=0.7,
    tau_v=0.5,
    law="calzetti",
    radio=None,
):
    emission = {"type": emission_type, "all_params": Fixed(DEFAULT)}
    if log_l_ir is not None:
        emission["log_L_ir"] = log_l_ir if hasattr(log_l_ir, "is_fixed") else Fixed(log_l_ir)
    if eta is not None:
        emission["eta_balance"] = eta if hasattr(eta, "is_fixed") else Fixed(eta)

    if dust_model == "two_component":
        attenuation = {
            "type": "two_component",
            "law_bc": law,
            "law_diff": law,
            "tau_bc": Fixed(tau_bc),
            "tau_diff": Fixed(tau_diff),
            "all_params": Fixed(DEFAULT),
        }
    elif dust_model == "single_component":
        attenuation = {
            "type": "single_component",
            "law": law,
            "tau_v": Fixed(tau_v),
            "all_params": Fixed(DEFAULT),
        }
    elif dust_model == "wg00":
        attenuation = {
            "type": "wg00",
            "tau_v": Fixed(tau_v),
            "all_params": Fixed(DEFAULT),
        }
    else:
        raise ValueError(dust_model)

    kwargs = dict(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=attenuation,
        dust_emission=emission,
        redshift=Fixed(0.0),
    )
    if radio is not None:
        kwargs["radio"] = radio
    return tengri.SEDModel.build(**kwargs)


def _sed_dust_ir_integral(state):
    wave = np.asarray(state.wave)
    sed = np.asarray(state.derived["sed_dust_ir"])
    nu = _C_AA_PER_S / wave
    return float(-np.trapezoid(sed, nu))


# ── 1. Positive: declared Fixed replaces the energy-balance budget ────────


def test_declared_fixed_replaces_two_component(ssp):
    """Two-component + dale2014: IR SED integral equals the declared budget.

    Mirrors the pattern of test_dust_emission_exact_energy_balance.py:112,
    which compares ``integral(sed_dust_ir)`` against the PUBLISHED budget
    (``L_ir``) -- that contract stays true, only what ``L_ir`` equals moves.
    """
    m = _build(ssp, log_l_ir=11.0)
    state = m.predict_state({})
    log_l_ir = float(np.asarray(state.derived["log_L_ir"]))
    expected_log = 11.0 + LOG10_L_SUN
    assert log_l_ir == pytest.approx(expected_log, abs=1e-9)
    integral = _sed_dust_ir_integral(state)
    np.testing.assert_allclose(integral, 10.0**expected_log, rtol=1e-6)


def test_declared_fixed_overrides_eta_relaxed_default_still_one(ssp):
    """Without an explicit eta, the guard leaves the implicit Fixed(1.0) untouched."""
    m = _build(ssp, log_l_ir=11.0)
    assert m.spec.get_fixed_values()["dust_eta_balance"] == 1.0


# ── 2. Declared prior frees, at the parse level ────────────────────────────


def test_declared_prior_frees_with_users_exact_prior():
    from tengri.parameters import parse_groups

    params = parse_groups(
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "log_L_ir": Uniform(9.0, 12.0)},
        ssp_data=None,
        redshift=Fixed(0.05),
    )
    assert "dust_log_L_ir" in params.free_params
    dist = params.get_distribution("dust_log_L_ir")
    assert dist.bounds == (9.0, 12.0)


def test_flat_parameters_escape_hatch_also_honors_the_override(ssp):
    """Requested-detection reads ``_flat_provenance`` for the flat ``Parameters(...)`` form too.

    ``SEDModel._requested_dust_log_L_ir`` falls back from
    ``spec._group_provenance`` (grammar builds) to ``spec._flat_provenance``
    (the flat escape hatch, #2231) -- the same two-source lookup
    ``_requested_law_shape_params`` already uses. Both the publish branch and
    the ``dust_eta_balance`` guard must work identically through this path.
    """
    from tengri import Parameters
    from tengri.config.exceptions import ParameterError as _ParameterError

    spec = Parameters(
        mean_sfh_type="delayed",
        sfh_delayed_tau_gyr=Fixed(1.0),
        sfh_delayed_age_gyr=Fixed(5.0),
        sfh_delayed_log_total_mass=Fixed(10.0),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(1.0),
        dust_tau_diff=Fixed(0.7),
        dust_emission="dale2014",
        dust_log_L_ir=Fixed(11.0),
        redshift=Fixed(0.0),
    )
    m = tengri.SEDModel(spec, ssp)
    log_l_ir = float(np.asarray(m.predict_state({}).derived["log_L_ir"]))
    np.testing.assert_allclose(log_l_ir, 11.0 + LOG10_L_SUN, atol=1e-9)

    bad_spec = Parameters(
        mean_sfh_type="delayed",
        sfh_delayed_tau_gyr=Fixed(1.0),
        sfh_delayed_age_gyr=Fixed(5.0),
        sfh_delayed_log_total_mass=Fixed(10.0),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(1.0),
        dust_tau_diff=Fixed(0.7),
        dust_emission="dale2014",
        dust_log_L_ir=Fixed(11.0),
        dust_eta_balance=Fixed(2.0),
        redshift=Fixed(0.0),
    )
    with pytest.raises(_ParameterError, match="dust_eta_balance"):
        tengri.SEDModel(bad_spec, ssp)


# ── 3. Undeclared: bit-identical to pre-change energy balance ─────────────


def test_undeclared_is_bit_identical_to_energy_balance(ssp):
    """Omitting ``log_L_ir`` must reproduce the eta formula exactly."""
    m_plain = _build(ssp)
    m_via_eta = _build(ssp, eta=1.0)
    state_plain = m_plain.predict_state({})
    state_eta = m_via_eta.predict_state({})
    log_ir_plain = float(np.asarray(state_plain.derived["log_L_ir"]))
    log_ir_eta = float(np.asarray(state_eta.derived["log_L_ir"]))
    log_abs = float(np.asarray(state_plain.derived["log_L_absorbed"]))
    assert log_ir_plain == pytest.approx(log_abs, abs=1e-12)
    assert log_ir_plain == pytest.approx(log_ir_eta, abs=1e-12)


def test_undeclared_relaxed_eta_still_the_eta_formula(ssp):
    m = _build(ssp, eta=2.0)
    state = m.predict_state({})
    log_ir = float(np.asarray(state.derived["log_L_ir"]))
    log_abs = float(np.asarray(state.derived["log_L_absorbed"]))
    assert log_ir == pytest.approx(log_abs + math.log10(2.0), abs=1e-9)


# ── 4. Per-publisher coverage: single_component and wg00 ──────────────────


def test_declared_fixed_replaces_single_component(ssp):
    m = _build(ssp, dust_model="single_component", log_l_ir=10.5)
    state = m.predict_state({})
    log_l_ir = float(np.asarray(state.derived["log_L_ir"]))
    np.testing.assert_allclose(log_l_ir, 10.5 + LOG10_L_SUN, atol=1e-9)
    integral = _sed_dust_ir_integral(state)
    np.testing.assert_allclose(integral, 10.0 ** (10.5 + LOG10_L_SUN), rtol=1e-6)


def test_declared_fixed_replaces_wg00(ssp):
    """WG00's own published ``L_ir``/``log_L_ir`` must carry the override.

    ``component_factory.build_components`` never pairs a WG00 screen with a
    separate dust-emission component (``atten_type != "wg00"`` guards that
    wiring, a pre-existing structural choice, not something this feature
    touches) -- so there is no ``sed_dust_ir`` to integrate here. What is
    testable, and what this pins, is the WG00 publisher's own ``L_ir`` /
    ``log_L_ir`` derived keys.
    """
    try:
        m = _build(ssp, dust_model="wg00", log_l_ir=10.5)
        state = m.predict_state({})
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"WG00 grid not on disk: {exc}")
    log_l_ir = float(np.asarray(state.derived["log_L_ir"]))
    l_ir = float(np.asarray(state.derived["L_ir"]))
    np.testing.assert_allclose(log_l_ir, 10.5 + LOG10_L_SUN, atol=1e-9)
    np.testing.assert_allclose(l_ir, 10.0 ** (10.5 + LOG10_L_SUN), rtol=1e-6)


def test_l_absorbed_unaffected_by_override_single_component(ssp):
    m_plain = _build(ssp, dust_model="single_component")
    m_over = _build(ssp, dust_model="single_component", log_l_ir=10.5)
    log_abs_plain = float(np.asarray(m_plain.predict_state({}).derived["log_L_absorbed"]))
    log_abs_over = float(np.asarray(m_over.predict_state({}).derived["log_L_absorbed"]))
    np.testing.assert_allclose(log_abs_over, log_abs_plain, rtol=1e-10)


def test_l_absorbed_unaffected_by_override_wg00(ssp):
    try:
        m_plain = _build(ssp, dust_model="wg00")
        m_over = _build(ssp, dust_model="wg00", log_l_ir=10.5)
        log_abs_plain = float(np.asarray(m_plain.predict_state({}).derived["log_L_absorbed"]))
        log_abs_over = float(np.asarray(m_over.predict_state({}).derived["log_L_absorbed"]))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"WG00 grid not on disk: {exc}")
    np.testing.assert_allclose(log_abs_over, log_abs_plain, rtol=1e-10)


# ── 5. Radio follows the override ──────────────────────────────────────────


def test_radio_follows_the_override(ssp):
    """A radio-enabled build's SED must move when the declared budget moves."""
    radio = {"sf": {"type": "bell2003"}, "all_params": Fixed(DEFAULT)}
    m_plain = _build(ssp, emission_type="dale2014_cigale", radio=radio)
    m_over = _build(ssp, emission_type="dale2014_cigale", log_l_ir=13.0, radio=radio)
    sed_radio_plain = np.asarray(m_plain.predict_state({}).derived["sed_radio"])
    sed_radio_over = np.asarray(m_over.predict_state({}).derived["sed_radio"])
    assert np.any(sed_radio_plain != 0.0), "setup: expected a nonzero radio SED"
    assert not np.allclose(sed_radio_plain, sed_radio_over), (
        "radio SED did not move when dust_log_L_ir moved -- FIRRC amplitude "
        "should track the declared IR budget"
    )


# ── 6. Guard: dust_eta_balance inertness ───────────────────────────────────


def test_guard_freed_eta_raises(ssp):
    with pytest.raises(ParameterError, match="dust_eta_balance"):
        _build(ssp, log_l_ir=11.0, eta=Uniform(0.5, 1.5))


def test_guard_fixed_nonunity_eta_raises(ssp):
    with pytest.raises(ParameterError, match="dust_eta_balance"):
        _build(ssp, log_l_ir=11.0, eta=2.0)


def test_guard_fixed_unity_eta_passes(ssp):
    m = _build(ssp, log_l_ir=11.0, eta=1.0)
    assert m is not None


def test_guard_no_eta_mentioned_passes(ssp):
    """No explicit eta key at all -- the implicit Fixed(1.0) default agrees."""
    m = _build(ssp, log_l_ir=11.0)
    assert m is not None


# ── 7. Fast-path allowlists ─────────────────────────────────────────────────


def _tophat_observation():
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    def _tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    curves = tuple(_tophat(c) for c in (3500.0, 8.0e5, 1.5e6))
    return Observation(photometry=Photometry(filters=curves))


@pytest.mark.parametrize("declaration", ["fixed", "free"])
def test_energy_balance_lut_survives_declared_override(ssp, declaration):
    """The absorbed-energy LUT must stay live: it never reads dust_log_L_ir."""
    obs = _tophat_observation()
    log_l_ir = Fixed(11.0) if declaration == "fixed" else Uniform(9.0, 12.0)
    m = tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=WavePrecomp(),
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(0.7),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "log_L_ir": log_l_ir, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
    )
    chain = m._build_component_chain()
    lut = m._energy_balance_lut(chain)
    assert lut is not None, (
        f"energy-balance LUT disabled under a {declaration} dust_log_L_ir override "
        "-- it should be unaffected (it caches log_L_absorbed only)"
    )


@pytest.mark.parametrize("declaration", ["fixed", "free"])
@pytest.mark.parametrize("emission_type", ["dale2014", "dh02_ce01"])
def test_band_response_fast_path_survives_free_override(ssp_real, declaration, emission_type):
    """Linear (non-budget-shaped) engines keep the band-response fast path.

    ``dust_log_L_ir`` only rescales the overall L_ir amplitude fed to a
    fixed-shape emission template; it must not be treated as a "shape" free
    parameter the way ``dust_T``/``dust_umin`` genuinely are.
    """
    obs = _tophat_observation()
    log_l_ir = Fixed(11.0) if declaration == "fixed" else Uniform(9.0, 12.0)
    m = tengri.SEDModel.build(
        ssp_data=ssp_real,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"all_params": Fixed(DEFAULT)},
        dust_attenuation={"law": "calzetti", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": emission_type, "log_L_ir": log_l_ir, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
    )
    chain = m._build_component_chain()
    resp = m._dust_emission_band_response(chain)
    assert resp is not None, (
        f"{emission_type} band-response fast path disabled under a {declaration} "
        "dust_log_L_ir override, though its emission shape does not depend on "
        "dust_log_L_ir at all"
    )


@pytest.mark.parametrize("declaration", ["fixed", "free"])
def test_bosa_band_response_stays_disabled_regardless_of_override(ssp_real, declaration):
    """BOSA's OWN (pre-existing, unrelated) shape/L_ir coupling still disables the fast path.

    Not a claim that dust_log_L_ir causes this: the homogeneity probe
    (``SEDModel._dust_emission_band_response``) measures BOSA's raw
    ``predict()`` at two L_ir values and finds it non-proportional
    independent of anything this feature touches. Pinned here so a future
    change to either mechanism is caught by name.
    """
    obs = _tophat_observation()
    log_l_ir = Fixed(11.0) if declaration == "fixed" else Uniform(9.0, 12.0)
    m = tengri.SEDModel.build(
        ssp_data=ssp_real,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"all_params": Fixed(DEFAULT)},
        dust_attenuation={"law": "calzetti", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "bosa", "log_L_ir": log_l_ir, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
    )
    chain = m._build_component_chain()
    resp = m._dust_emission_band_response(chain)
    assert resp is None, "BOSA's band-response fast path unexpectedly active"


@pytest.fixture(scope="module")
def ssp_real():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")
