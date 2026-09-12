# SPDX-License-Identifier: BSD-3-Clause
"""AGN dust screen functional tests (#2260, PR-D2).

``agn_screen`` selects which of the two-component screens attenuates AGN light
(``"birth_cloud"``, ``"diffuse"``, or ``"none"``). The screen is applied only
when ``agn_norm != "cigale_joint"``, since the joint norm reads the dust budget
that a screen would then depend on. This file tests the build-time grammar and
flat-surface validation, the runtime attenuation, and the precompute equivalence.
"""

from __future__ import annotations

import chex
import numpy as np
import pytest
from jax import numpy as jnp

from tengri import DEFAULT, SEDModel, WavePrecomp
from tengri.components.dust.attenuation import resolve_bc_diff_law_params
from tengri.components.dust.laws._registry import resolve_dust_law, select_law_kwargs
from tengri.components.dust.two_component import (
    DustSEDComponent,
    DustSEDComponentConfig,
)
from tengri.config.exceptions import AGNDustDoubleCountWarning, ParameterError
from tengri.forward.energy_balance import bolometric_absorbed
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed
from tengri.protocols.component import ForwardState
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.contract

_WAVE = jnp.array([1500.0, 2700.0, 3550.0, 5500.0, 9000.0])
_AGES = jnp.array([1.0e6, 1.0e8, 1.0e10])
_LAW_BC = "calzetti"
_LAW_DIFF = "power_law"
_PARAMS = {
    "dust_tau_bc": 0.8,
    "dust_tau_diff": 0.4,
    "dust_f_obscuration": 0.2,
    "dust_slope": -1.2,
    "agn_log_lbol": 12.0,
}


def _config_dust(**screens) -> DustSEDComponentConfig:
    """Dust config with specified screens."""
    return DustSEDComponentConfig(law_bc=_LAW_BC, law_diff=_LAW_DIFF, **screens)


def _oracle_agn_transmission(
    comp: DustSEDComponent, params: dict, wave, choice: str
) -> np.ndarray:
    """Independent re-derivation of AGN screen transmission from law curves.

    Mirrors the oracle in test_source_screen_choice.py: resolves the BC/diffuse
    law kwargs and evaluates the registry law functions directly.
    """
    config = comp.config
    bc_params, diff_params = resolve_bc_diff_law_params(
        params,
        dict(config.bc_law_overrides),
        dict(config.diff_law_overrides),
        config.live_shape_params,
        bc_law=config.law_bc,
        diff_law=config.law_diff,
        redshift=params.get("redshift"),
    )
    # For AGN (unlike nebular), use the BC and diffuse laws directly
    # (not the nebular override law).
    neb_law = config.law_bc
    neb_bc_kw = select_law_kwargs(neb_law, {**bc_params, **dict(config.neb_law_overrides)})
    k_bc = np.asarray(resolve_dust_law(neb_law)(wave, **neb_bc_kw))
    k_diff = np.asarray(resolve_dust_law(config.law_diff)(wave, **diff_params))

    tau_bc = float(params["dust_tau_bc"])
    tau_diff = float(params["dust_tau_diff"])
    f_obsc = float(params.get("dust_f_obscuration", 0.0))

    if choice == "none":
        return np.ones_like(k_diff)
    if choice == "birth_cloud":
        tau = tau_bc * k_bc + tau_diff * k_diff
    elif choice == "diffuse":
        tau = tau_diff * k_diff
    else:
        raise ValueError(choice)
    return f_obsc + (1.0 - f_obsc) * np.exp(-tau)


def _state_with_agn(sed_agn) -> ForwardState:
    """Create a ForwardState with AGN light plus minimal stellar data."""
    # Dust component requires lnu_age and ssp_ages_yr
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    stellar = jnp.sum(lnu_age, axis=0)
    return ForwardState(
        wave=_WAVE,
        sed_intrinsic=stellar + sed_agn,
        derived={
            "lnu_age": lnu_age,
            "ssp_ages_yr": _AGES,
            "sed_agn": sed_agn,
        },
    )


def _build_agn_dust_model(
    ssp,
    *,
    screen: str,
    approx=None,
    agn_norm: str = "independent",
    observation=None,
    log_total_mass: float = -15.0,
    agn_log_lbol: float = 15.0,
    with_dust_emission: bool = True,
):
    """Minimal AGN + two-component dust model, on committed-only blocks.

    Uses the ``multicolor`` disc block (no external grid dependency, unlike
    ``grahsp``/``skirtor`` template families) so it builds on the
    session-scoped synthetic SSP fixtures with no ``data/`` files needed
    (mirrors ``tests/contract/test_spectrum_precomp_includes_agn.py``).

    ``log_total_mass`` defaults far sub-solar and ``agn_log_lbol`` defaults
    bright: the synthetic SSP's UV-rising continuum (#613 fixture) would
    otherwise swamp the AGN in the optical, making any dust-screen effect on
    AGN light invisible at float64 precision. This combination makes the AGN
    the dominant source in the fixture's filters/wave range, which is what
    makes tests (e)/(f) non-vacuous.
    """
    kwargs = dict(
        ssp_data=ssp,
        observation=observation,
        redshift=Fixed(0.3),
        igm={"type": "none"},
        approx=approx,
        sfh={
            "type": "tsnorm",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Fixed(log_total_mass),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "agn_screen": screen,
        },
        neb={"type": "none"},
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
            "log_lbol": Fixed(agn_log_lbol),
            "norm": agn_norm,
        },
    )
    if with_dust_emission:
        kwargs["dust_emission"] = {"type": "modified_blackbody", "all_params": Fixed(DEFAULT)}
    return SEDModel.build(**kwargs)


def _agn_oracle_transmission(wave, params, choice: str) -> np.ndarray:
    """Independent re-derivation of the AGN screen transmission (model-level).

    Same formula as :func:`_oracle_agn_transmission` above, but resolved from
    a full :class:`~tengri.forward.sed_model.SEDModel`'s fixed params rather
    than a bare :class:`DustSEDComponent`, for the built-model tests below.
    """
    comp = DustSEDComponent(
        config=DustSEDComponentConfig(law_bc="calzetti", law_diff="calzetti", agn_screen=choice)
    )
    return _oracle_agn_transmission(comp, params, wave, choice)


# ── (a) agn_screen="none" is bit-identical to the tree before this change ──


def test_agn_screen_none_unattenuated():
    """With agn_screen='none', AGN SED difference is zero (not screened)."""
    comp = DustSEDComponent(config=_config_dust(agn_screen="none"))
    sed_agn = jnp.full(_WAVE.shape, 1.0e28)
    state = _state_with_agn(sed_agn)

    out = comp.apply(state, _PARAMS)

    # The SED should include the unattenuated AGN. Verify that zeroing AGN
    # gives the SED without any AGN contribution.
    stellar_only = np.asarray(state.sed_intrinsic) - np.asarray(sed_agn)
    state_no_agn = ForwardState(
        wave=state.wave,
        sed_intrinsic=stellar_only,
        derived={**state.derived, "sed_agn": None},
    )
    out_no_agn = comp.apply(state_no_agn, _PARAMS)

    # The difference should be the unattenuated AGN contribution.
    sed_agn_contribution = np.asarray(out.sed_intrinsic) - np.asarray(out_no_agn.sed_intrinsic)
    np.testing.assert_allclose(sed_agn_contribution, sed_agn, rtol=1e-10, atol=0.0)


# ── (b) diffuse and birth_cloud multiply AGN by the hand-computed factor ──


@pytest.mark.parametrize("choice", ["birth_cloud", "diffuse"])
def test_agn_screen_attenuated(choice):
    """AGN attenuation multiplies by hand-computed oracle for each screen."""
    comp = DustSEDComponent(config=_config_dust(agn_screen=choice))
    comp_none = DustSEDComponent(config=_config_dust(agn_screen="none"))
    sed_agn = jnp.full(_WAVE.shape, 1.0e28)
    state = _state_with_agn(sed_agn)

    out = comp.apply(state, _PARAMS)
    out_none = comp_none.apply(state, _PARAMS)

    # The difference between screened and unscreened is the attenuation effect.
    # sed_none - sed_screened = sed_agn * (1 - T_screen)
    sed_agn_contribution_screened = np.asarray(out_none.sed_intrinsic) - np.asarray(
        out.sed_intrinsic
    )
    transmission = _oracle_agn_transmission(comp, _PARAMS, _WAVE, choice)
    expected_difference = np.asarray(sed_agn) * (1.0 - transmission)
    np.testing.assert_allclose(
        sed_agn_contribution_screened, expected_difference, rtol=1e-10, atol=0.0
    )


@pytest.mark.parametrize("screen_choice", ["birth_cloud", "diffuse", "none"])
def test_agn_screen_does_not_affect_stellar(screen_choice):
    """AGN screen attenuation does not change stellar attenuation."""
    comp_screened = DustSEDComponent(config=_config_dust(agn_screen=screen_choice))
    comp_unscreened = DustSEDComponent(config=_config_dust(agn_screen="none"))

    # State with only stellar (no AGN).
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    stellar = jnp.sum(lnu_age, axis=0)
    state = ForwardState(
        wave=_WAVE,
        sed_intrinsic=stellar,
        derived={"lnu_age": lnu_age, "ssp_ages_yr": _AGES},
    )

    out_screened = comp_screened.apply(state, _PARAMS)
    out_unscreened = comp_unscreened.apply(state, _PARAMS)

    # Stellar attenuation should be identical.
    np.testing.assert_allclose(
        out_screened.derived["sed_dust_attenuated"],
        out_unscreened.derived["sed_dust_attenuated"],
        rtol=1e-10,
    )


# ── (c) Dust component optional_inputs declaration ──


def test_optional_inputs_declares_sed_agn_when_screened():
    """DustSEDComponent.optional_inputs() declares sed_agn when agn_screen != 'none'."""
    comp_screened = DustSEDComponent(config=_config_dust(agn_screen="diffuse"))
    comp_unscreened = DustSEDComponent(config=_config_dust(agn_screen="none"))

    optional_screened = comp_screened.optional_inputs()
    optional_unscreened = comp_unscreened.optional_inputs()

    screened_names = [k.name for k in optional_screened]
    unscreened_names = [k.name for k in optional_unscreened]

    assert "sed_agn" in screened_names, "sed_agn should be in optional_inputs when screened"
    assert "sed_agn" not in unscreened_names, (
        "sed_agn should not be in optional_inputs when unscreened"
    )


def test_agn_component_optional_inputs_when_cigale_joint():
    """AGNSEDComponent.optional_inputs() declares dust deps only for cigale_joint norm."""
    from tengri.components.agn.component import AGNSEDComponent, AGNSEDComponentConfig

    config_cigale = AGNSEDComponentConfig(agn_norm="cigale_joint")
    config_independent = AGNSEDComponentConfig(agn_norm="independent")

    comp_cigale = AGNSEDComponent(config=config_cigale)
    comp_independent = AGNSEDComponent(config=config_independent)

    optional_cigale = comp_cigale.optional_inputs()
    optional_independent = comp_independent.optional_inputs()

    cigale_names = [k.name for k in optional_cigale]
    independent_names = [k.name for k in optional_independent]

    # CIGALE joint reads L_ir and L_absorbed
    for key in ["log_L_ir", "L_absorbed"]:
        assert key in cigale_names, f"{key} should be in optional_inputs for cigale_joint"

    # Independent does not
    for key in ["log_L_ir", "L_absorbed"]:
        assert key not in independent_names, (
            f"{key} should not be in optional_inputs for independent"
        )


# ── (d) cigale_joint + screened AGN raises on flat surface only ──


def test_cigale_joint_plus_screened_agn_raises_flat():
    """Flat Parameters raises when agn_screen != 'none' with agn_norm='cigale_joint'."""
    with pytest.raises(ParameterError):
        params = Parameters(
            agn_model="composable",
            dust_agn_screen="diffuse",
            agn_norm="cigale_joint",
            redshift=Fixed(0.1),
        )


@pytest.mark.parametrize(
    "agn_screen,should_raise",
    [
        ("none", False),
        ("birth_cloud", True),
        ("diffuse", True),
    ],
)
def test_agn_screen_flat_cigale_joint_validation(agn_screen, should_raise):
    """Flat Parameters validates agn_screen + agn_norm=cigale_joint cycle rule."""
    if should_raise:
        with pytest.raises(ParameterError, match=r"agn_screen.*cigale_joint"):
            Parameters(
                agn_model="composable",
                dust_agn_screen=agn_screen,
                agn_norm="cigale_joint",
                redshift=Fixed(0.1),
            )
    else:
        # Should not raise with agn_screen='none' or without agn_model
        params = Parameters(
            dust_agn_screen=agn_screen,
            agn_norm="cigale_joint",
            redshift=Fixed(0.1),
        )
        assert params is not None


@pytest.mark.parametrize(
    "agn_norm,agn_screen",
    [
        ("independent", "none"),
        ("independent", "birth_cloud"),
        ("independent", "diffuse"),
        ("conserving", "none"),
        ("conserving", "birth_cloud"),
        ("conserving", "diffuse"),
    ],
)
def test_agn_screen_flat_other_norms_allowed(agn_norm, agn_screen):
    """Flat Parameters allows screened AGN with independent/conserving norms."""
    params = Parameters(
        agn_norm=agn_norm,
        dust_agn_screen=agn_screen,
        redshift=Fixed(0.1),
    )
    assert params is not None


# ── (e) Energy balance: absorbed AGN power enters L_absorbed ──


@pytest.mark.parametrize("branch,approx", [("exact", None), ("lut", WavePrecomp())])
def test_energy_balance_screened_agn(synthetic_ssp_wide, branch, approx):
    """Energy balance: L_absorbed(screened) - L_absorbed(none) equals AGN absorbed power.

    Pins that the absorbed AGN power (#2260) joins the dust energy balance on
    BOTH computation branches inside :meth:`DustSEDComponent.apply`: the exact
    integral (``eb_lut is None``, ``approx=None``) and the ``build_energy_balance_lut``
    fast path (``approx=WavePrecomp()`` + a dust-emission consumer, so
    ``_energy_balance_lut`` bakes a LUT). ``dust_emission='modified_blackbody'``
    (analytic, no template grid) makes both branches reachable on the same
    model family.

    The two models (``agn_screen='diffuse'`` vs ``'none'``) share every other
    parameter, so the only difference in ``L_absorbed`` is the AGN term: with
    ``'none'`` the AGN SED is invisible to dust (it runs after dust), with
    ``'diffuse'`` it is attenuated by the diffuse screen before dust's energy
    balance integral runs. That delta must equal the independent
    :func:`tengri.forward.energy_balance.bolometric_absorbed` integral of
    (intrinsic AGN, screened AGN), LyC-masked the same way (912 Angstrom,
    the ``dust_eb_include_lyc=False`` default).
    """
    m_screen = _build_agn_dust_model(synthetic_ssp_wide, screen="diffuse", approx=approx)
    m_none = _build_agn_dust_model(synthetic_ssp_wide, screen="none", approx=approx)

    # eb_lut branch sanity: confirm this build actually reaches the branch
    # this parametrization claims (informative, not the test's real assertion).
    chain = m_screen._build_component_chain()
    eb_lut = m_screen._energy_balance_lut(chain)
    if branch == "lut":
        assert eb_lut is not None, "approx=WavePrecomp() build did not reach the eb_lut branch"
    else:
        assert eb_lut is None, "approx=None build unexpectedly reached the eb_lut branch"

    p = dict(m_screen.spec.get_fixed_values())
    state_screen = m_screen.predict_state(p)
    state_none = m_none.predict_state(p)

    def _l_absorbed(state) -> float:
        derived = state.derived
        value = derived.get("L_absorbed")
        if value is not None:
            return float(value)
        # dust_eta_balance is Fixed at its default (1.0) in this model, so
        # log_L_ir == log_L_absorbed exactly; fallback per the L_absorbed
        # contract if a future build omits the direct key.
        return float(10.0 ** derived["log_L_ir"])

    delta = _l_absorbed(state_screen) - _l_absorbed(state_none)

    sed_agn = jnp.asarray(state_screen.derived["sed_agn"])
    wave = jnp.asarray(state_screen.wave)
    nu = C_AA / wave
    transmission = jnp.asarray(_agn_oracle_transmission(np.asarray(wave), p, "diffuse"))
    chex.assert_tree_all_finite((sed_agn, transmission))
    expected = abs(
        float(
            bolometric_absorbed(
                sed_agn, sed_agn * transmission, nu, wave=wave, lyman_cutoff_aa=912.0
            )
        )
    )

    rel_err = abs(delta - expected) / expected
    assert rel_err < 1e-6, (
        f"branch={branch}: L_absorbed delta {delta:.6e} vs oracle {expected:.6e}, "
        f"rel_err={rel_err:.3e}"
    )


# ── (f) Exact vs WavePrecomp photometry agreement ──


@pytest.mark.parametrize("screen", ["none", "birth_cloud", "diffuse"])
def test_exact_vs_precomp_photometry_agn(synthetic_ssp_wide, synthetic_tophat_obs, screen):
    """predict_photometry agrees between the exact and WavePrecomp paths (#2260).

    Regression shape of #1434 (shock) applied to AGN: before this PR's fix to
    ``predict_via_precomp`` (``tengri/observation/observation.py``), the
    membership check meant to substitute ``agn_phot_lnu_attenuated_precomp``
    for the intrinsic ``agn_phot_lnu_precomp`` in the summed photometry could
    never fire (the attenuated key's name does not end in
    ``_phot_lnu_precomp``, the suffix the substitution list is filtered on),
    and the attenuated value was never added to the total either -- so a
    screened AGN would have been silently unattenuated under
    ``approx=WavePrecomp()`` while correctly attenuated under the exact path.

    Uses a bright AGN (``agn_log_lbol=15``) against a near-massless host
    (``log_total_mass=-15``) so the AGN dominates the synthetic SSP's
    UV-rising optical continuum in the tophat filters: with a normal stellar
    mass the AGN's contribution is ~1e-9 of the total flux, and any screening
    discrepancy would be lost in float64 roundoff on the total (measured
    while developing this test) rather than caught.
    """
    m_exact = _build_agn_dust_model(
        synthetic_ssp_wide,
        screen=screen,
        approx=None,
        observation=synthetic_tophat_obs,
        with_dust_emission=False,
    )
    m_lut = _build_agn_dust_model(
        synthetic_ssp_wide,
        screen=screen,
        approx=WavePrecomp(),
        observation=synthetic_tophat_obs,
        with_dust_emission=False,
    )
    p = dict(m_exact.spec.get_fixed_values())
    phot_exact = np.asarray(m_exact.predict_photometry(p))
    phot_lut = np.asarray(m_lut.predict_photometry(p))

    chex.assert_tree_all_finite(phot_exact)
    assert np.all(phot_exact > 0.0)
    max_rel = float(np.max(np.abs((phot_lut - phot_exact) / phot_exact)))
    print(f"agn_screen={screen!r}: max relative difference (exact vs WavePrecomp) = {max_rel:.3e}")
    # Tolerance from tests/contract/test_shock_attenuation_equivalence.py's
    # exact-vs-precomp shock bound; measured here at ~1e-14 (float64 roundoff)
    # for all three screens once the observation.py substitution bug (above)
    # is fixed, so 5% is generous headroom, not a tight measurement.
    chex.assert_trees_all_close(phot_lut, phot_exact, rtol=0.05, atol=0.0)


# ── (g) to_groups() round-trip ──


@pytest.mark.parametrize("screen", ["birth_cloud", "diffuse"])
def test_agn_screen_roundtrip_to_groups(synthetic_ssp_wide, screen):
    """agn_screen round-trips through Parameters.spec.to_groups() (#2260).

    ``"none"`` is the default and, per ``to_groups()``'s general contract
    (default-valued keys are omitted, #2234's siblings have the same
    property), does not round-trip -- only a non-default explicit choice
    does, which is what this test pins.
    """
    m = _build_agn_dust_model(synthetic_ssp_wide, screen=screen, with_dust_emission=False)
    roundtrip_groups = m.spec.to_groups()
    assert roundtrip_groups["dust_attenuation"]["agn_screen"] == screen


# ── (h) warn_agn_dust_double_count still fires ──


def test_warn_agn_dust_double_count_fires_with_screened_agn(synthetic_ssp_wide):
    """_warn_agn_dust_double_count is untouched by PR-D2, and the two features
    are correctly, not silently, mutually exclusive.

    ``_warn_agn_dust_double_count`` (``src/tengri/forward/sed_model.py``)
    fires only when ``spec.dust_emission == "dale2014"`` and BOTH
    ``dust_frac_agn`` and ``agn_ir_frac`` (fracAGN) are positive-active.
    fracAGN is itself refused outside ``agn_norm="cigale_joint"``
    (``_validate_fracagn_requires_cigale_joint``, R65/R67: 'independent' and
    'conserving' both raise ``ConfigError`` for an active fracAGN, since the
    disc and torus would sit on two unrelated luminosity scales). PR-D2's own
    cycle rule refuses ``agn_screen != "none"`` under exactly
    ``agn_norm="cigale_joint"``. So the double-count trigger and a screened
    AGN cannot coexist in any buildable model -- verified here as two halves:
    (a) the warning still fires under its documented (unscreened,
    cigale_joint) trigger, unaffected by this PR; (b) trying to pair that
    same trigger with a screened AGN raises the cycle-rule ``ParameterError``
    before the warning is ever reached, rather than silently dropping either
    check.
    """

    def _build(*, screen, norm):
        return SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            redshift=Fixed(0.3),
            igm={"type": "none"},
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(8.0)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "agn_screen": screen,
            },
            dust_emission={
                "type": "dale2014",
                "all_params": Fixed(DEFAULT),
                "frac_agn": Fixed(0.3),
            },
            neb={"type": "none"},
            agn={
                "type": "composable",
                "all_params": Fixed(DEFAULT),
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
                "log_lbol": Fixed(12.0),
                "norm": norm,
                "ir_frac": Fixed(0.3),
            },
        )

    # (a) Unscreened + cigale_joint: the documented trigger, unaffected by PR-D2.
    with pytest.warns(AGNDustDoubleCountWarning):
        _build(screen="none", norm="cigale_joint")

    # (b) Screened + cigale_joint: refused by the PR-D2 cycle rule before the
    # warning check in SEDModel.build is ever reached.
    with pytest.raises(ParameterError, match="incompatible with agn_norm"):
        _build(screen="diffuse", norm="cigale_joint")


# ── (c) Component chain order on a built model ──────────────────────────


@pytest.mark.parametrize(
    "screen,agn_before_dust",
    [("none", False), ("birth_cloud", True), ("diffuse", True)],
)
def test_chain_order_agn_relative_to_dust(synthetic_ssp_wide, screen, agn_before_dust):
    """AGNSEDComponent's chain index is before DustSEDComponent's iff screened.

    Built via :meth:`SEDModel.build`, not the bare component-level
    ``optional_inputs()`` check in test (c)'s sibling above: this pins that
    :func:`~tengri.forward.orchestrator.topological_sort` actually moves AGN
    in the real, built component chain, not merely that the declaration
    exists.
    """
    m = _build_agn_dust_model(synthetic_ssp_wide, screen=screen, with_dust_emission=False)
    chain = m._build_component_chain()
    names = [type(c).__name__ for c in chain]
    agn_idx = names.index("AGNSEDComponent")
    dust_idx = names.index("DustSEDComponent")
    if agn_before_dust:
        assert agn_idx < dust_idx, f"screen={screen!r}: expected AGN before dust, chain={names}"
    else:
        assert agn_idx > dust_idx, f"screen={screen!r}: expected AGN after dust, chain={names}"


# ── Mutation tests ──


def test_mutation_agn_screen_affects_attenuation():
    """Mutation: removing agn_screen from apply() makes the test fail."""
    comp = DustSEDComponent(config=_config_dust(agn_screen="diffuse"))
    comp_none = DustSEDComponent(config=_config_dust(agn_screen="none"))
    sed_agn = jnp.full(_WAVE.shape, 1.0e28)
    state = _state_with_agn(sed_agn)

    out = comp.apply(state, _PARAMS)
    out_none = comp_none.apply(state, _PARAMS)

    # The oracle transmission should differ from identity (attenuation is real).
    transmission = _oracle_agn_transmission(comp, _PARAMS, _WAVE, "diffuse")
    assert not np.allclose(transmission, 1.0, rtol=1e-10), (
        "Oracle transmission should differ from identity (attenuation exists)"
    )

    # The difference in SEDs should show the attenuation effect.
    sed_agn_contribution_screened = np.asarray(out_none.sed_intrinsic) - np.asarray(
        out.sed_intrinsic
    )
    expected_difference = np.asarray(sed_agn) * (1.0 - transmission)
    np.testing.assert_allclose(
        sed_agn_contribution_screened, expected_difference, rtol=1e-10, atol=0.0
    )


def test_mutation_optional_inputs_present():
    """Mutation: optional_inputs must declare sed_agn when screened."""
    comp = DustSEDComponent(config=_config_dust(agn_screen="diffuse"))

    # The optional_inputs should declare sed_agn as a dependency.
    optional = comp.optional_inputs()
    opt_names = [k.name for k in optional]

    assert "sed_agn" in opt_names, "sed_agn should be in optional_inputs when agn_screen != 'none'"


# Removing the AGN term from the energy-balance integral is exercised by hand
# (PR-D2 item 4), not as a persisted test: it is exactly the mutation that
# would make test_energy_balance_screened_agn above (e) fail, so a third
# static copy of the same assertion here would guard nothing new -- see
# the PR report for the by-hand mutation run and its restore diff-stat.
