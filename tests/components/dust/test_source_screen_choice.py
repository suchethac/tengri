# SPDX-License-Identifier: BSD-3-Clause
"""Per-source dust-screen choice: nebular_screen / shock_screen / agn_screen (#2234).

Each non-stellar emission source (nebular continuum + line catalog, shock,
AGN) picks which of the two-component screens attenuates it: the young-star
transmission (``"birth_cloud"``), the old-star transmission (``"diffuse"``),
or none at all (``"none"``/``"off"``). This file is the contract for that
grammar: every attenuation site routes through
:func:`tengri.components.dust.two_component._screen_transmission`, one
validator (:func:`tengri.parameters._dust_keys.resolve_screen_choices`) backs
both the grammar and the flat surface, and the choice round-trips through
``to_groups()``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import (
    apply_lyman_cutoff,
    resolve_bc_diff_law_params,
)
from tengri.components.dust.laws._registry import resolve_dust_law, select_law_kwargs
from tengri.components.dust.two_component import (
    DustSEDComponent,
    DustSEDComponentConfig,
)
from tengri.config.exceptions import ParameterError
from tengri.parameters._dust_keys import SCREEN_CHOICES
from tengri.parameters.groups import parameters_to_groups, parse_groups
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.contract

_WAVE = jnp.array([1500.0, 2700.0, 3550.0, 5500.0, 9000.0])
_AGES = jnp.array([1.0e6, 1.0e8, 1.0e10])
#: bc=calzetti, diff=power_law(dust_slope=-1.2): distinct curves so a
#: test that accidentally reads the wrong screen's law is caught.
_LAW_BC = "calzetti"
_LAW_DIFF = "power_law"
_PARAMS = {
    "dust_tau_bc": 0.8,
    "dust_tau_diff": 0.4,
    "dust_f_obscuration": 0.2,
    "dust_slope": -1.2,
}


def _config(**screens) -> DustSEDComponentConfig:
    return DustSEDComponentConfig(law_bc=_LAW_BC, law_diff=_LAW_DIFF, **screens)


def _oracle_transmission(comp: DustSEDComponent, params: dict, wave, choice: str) -> np.ndarray:
    """Independent re-derivation of the screen transmission from the law curves.

    Deliberately a SEPARATE spelling from ``_screen_transmission`` (never
    calls it): resolves the birth-cloud/diffuse law kwargs and evaluates the
    registry law functions directly, the same way
    ``DustSEDComponent.attenuate_line_catalog`` / ``apply()`` do, so a defect
    in the shared helper cannot hide behind a shared oracle.
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
    neb_law = config.law_neb or config.law_bc
    neb_bc_kw = select_law_kwargs(neb_law, {**bc_params, **dict(config.neb_law_overrides)})
    k_bc = np.asarray(
        apply_lyman_cutoff(
            resolve_dust_law(neb_law)(wave, **neb_bc_kw), wave, config.lyman_cutoff_aa
        )
    )
    k_diff = np.asarray(
        apply_lyman_cutoff(
            resolve_dust_law(config.law_diff)(wave, **diff_params), wave, config.lyman_cutoff_aa
        )
    )
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


def _state_nebular_only(sed_neb) -> ForwardState:
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    stellar = jnp.sum(lnu_age, axis=0)
    return ForwardState(
        wave=_WAVE,
        sed_intrinsic=stellar + sed_neb,
        derived={"lnu_age": lnu_age, "ssp_ages_yr": _AGES, "sed_nebular": sed_neb},
    )


def _state_shock_only(sed_shock) -> ForwardState:
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    stellar = jnp.sum(lnu_age, axis=0)
    return ForwardState(
        wave=_WAVE,
        sed_intrinsic=stellar + sed_shock,
        derived={"lnu_age": lnu_age, "ssp_ages_yr": _AGES, "sed_shock": sed_shock},
    )


# ── (1) Hand-computed factor for each source x choice ──────────────────────


@pytest.mark.parametrize("choice", SCREEN_CHOICES)
def test_nebular_continuum_matches_hand_computed_factor(choice):
    """The nebular continuum's attenuation equals the oracle for its choice."""
    comp = DustSEDComponent(config=_config(nebular_screen=choice))
    sed_neb = jnp.full(_WAVE.shape, 1.0e28)
    state = _state_nebular_only(sed_neb)

    out = comp.apply(state, _PARAMS)
    stellar_att = np.asarray(out.derived["sed_dust_attenuated"])
    neb_observed = np.asarray(out.sed_intrinsic) - stellar_att

    expected = np.asarray(sed_neb) * _oracle_transmission(comp, _PARAMS, _WAVE, choice)
    np.testing.assert_allclose(neb_observed, expected, rtol=1e-10, atol=0.0)
    # Published sed_nebular is the SAME observed (attenuated) continuum.
    np.testing.assert_allclose(
        np.asarray(out.derived["sed_nebular"]), expected, rtol=1e-10, atol=0.0
    )


@pytest.mark.parametrize("choice", SCREEN_CHOICES)
def test_line_catalog_matches_hand_computed_factor(choice):
    """attenuate_line_catalog's transmission equals the oracle for its choice."""
    comp = DustSEDComponent(config=_config(nebular_screen=choice))
    line_wave = jnp.array([1216.0, 4862.71, 6564.61])
    log_line_lums = jnp.log10(jnp.array([1.0e41, 5.0e40, 1.5e41]))

    log_attenuated = comp.attenuate_line_catalog(_PARAMS, line_wave, log_line_lums)
    transmission = np.asarray(10.0 ** (np.asarray(log_attenuated) - np.asarray(log_line_lums)))

    expected = _oracle_transmission(comp, _PARAMS, line_wave, choice)
    np.testing.assert_allclose(transmission, expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("choice", SCREEN_CHOICES)
def test_shock_matches_hand_computed_factor(choice):
    """The shock SED's attenuation equals the oracle for its choice."""
    comp = DustSEDComponent(config=_config(shock_screen=choice))
    sed_shock = jnp.full(_WAVE.shape, 2.0e27)
    state = _state_shock_only(sed_shock)

    out = comp.apply(state, _PARAMS)
    stellar_att = np.asarray(out.derived["sed_dust_attenuated"])
    shock_observed = np.asarray(out.sed_intrinsic) - stellar_att

    expected = np.asarray(sed_shock) * _oracle_transmission(comp, _PARAMS, _WAVE, choice)
    np.testing.assert_allclose(shock_observed, expected, rtol=1e-10, atol=0.0)


# ── (2) Defaults: nebular=birth_cloud, shock=diffuse ────────────────────────


def test_defaults_are_nebular_birth_cloud_shock_diffuse():
    """Default config: nebular gets birth_cloud, shock gets diffuse.

    Mutation check: a config that swapped the two defaults would produce a
    DIFFERENT number here (the two oracle curves disagree on this fixture,
    since bc=calzetti and diff=power_law are distinct laws), so this test
    fails under that swap.
    """
    default_config = DustSEDComponentConfig(law_bc=_LAW_BC, law_diff=_LAW_DIFF)
    assert default_config.nebular_screen == "birth_cloud"
    assert default_config.shock_screen == "diffuse"

    neb_comp = DustSEDComponent(config=default_config)
    sed_neb = jnp.full(_WAVE.shape, 1.0e28)
    state = _state_nebular_only(sed_neb)
    out = neb_comp.apply(state, _PARAMS)
    stellar_att = np.asarray(out.derived["sed_dust_attenuated"])
    neb_observed = np.asarray(out.sed_intrinsic) - stellar_att

    bc_oracle = np.asarray(sed_neb) * _oracle_transmission(neb_comp, _PARAMS, _WAVE, "birth_cloud")
    diff_oracle = np.asarray(sed_neb) * _oracle_transmission(neb_comp, _PARAMS, _WAVE, "diffuse")
    np.testing.assert_allclose(neb_observed, bc_oracle, rtol=1e-10, atol=0.0)
    assert not np.allclose(neb_observed, diff_oracle, rtol=1e-6), (
        "nebular default must be distinguishable from the diffuse choice on "
        "this fixture, or the swap-detection below is vacuous."
    )

    shock_comp = DustSEDComponent(config=default_config)
    sed_shock = jnp.full(_WAVE.shape, 2.0e27)
    state = _state_shock_only(sed_shock)
    out = shock_comp.apply(state, _PARAMS)
    shock_observed = np.asarray(out.sed_intrinsic) - np.asarray(out.derived["sed_dust_attenuated"])
    shock_diff_oracle = np.asarray(sed_shock) * _oracle_transmission(
        shock_comp, _PARAMS, _WAVE, "diffuse"
    )
    shock_bc_oracle = np.asarray(sed_shock) * _oracle_transmission(
        shock_comp, _PARAMS, _WAVE, "birth_cloud"
    )
    np.testing.assert_allclose(shock_observed, shock_diff_oracle, rtol=1e-10, atol=0.0)
    assert not np.allclose(shock_observed, shock_bc_oracle, rtol=1e-6), (
        "shock default must be distinguishable from the birth_cloud choice on "
        "this fixture, or the swap-detection above is vacuous."
    )


# ── (3) 'none' is bit-identical to the unattenuated source ─────────────────


def test_none_is_bit_identical_to_unattenuated():
    comp = DustSEDComponent(config=_config(nebular_screen="none", shock_screen="none"))
    sed_neb = jnp.full(_WAVE.shape, 1.0e28)
    sed_shock = jnp.full(_WAVE.shape, 2.0e27)
    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    stellar = jnp.sum(lnu_age, axis=0)
    state = ForwardState(
        wave=_WAVE,
        sed_intrinsic=stellar + sed_neb + sed_shock,
        derived={
            "lnu_age": lnu_age,
            "ssp_ages_yr": _AGES,
            "sed_nebular": sed_neb,
            "sed_shock": sed_shock,
        },
    )
    out = comp.apply(state, _PARAMS)
    non_stellar_observed = np.asarray(out.sed_intrinsic) - np.asarray(
        out.derived["sed_dust_attenuated"]
    )
    # The 'none' screen multiplies by exactly 1.0 (see _screen_transmission),
    # so it is bit-identical at the point of attenuation; the tiny (~1e-16
    # relative) residual below comes from recovering the non-stellar term via
    # float64 subtraction of a ~1e29 stellar term (state.sed_intrinsic =
    # stellar + sed_neb + sed_shock), not from any inexactness in the screen.
    np.testing.assert_allclose(
        non_stellar_observed, np.asarray(sed_neb + sed_shock), rtol=1e-12, atol=0.0
    )
    np.testing.assert_array_equal(np.asarray(out.derived["sed_nebular"]), np.asarray(sed_neb))


# ── (4)/(5) Grammar + flat spellings agree; round-trip; 'off' -> 'none' ────


@pytest.mark.parametrize("raw,resolved", [("birth_cloud", "birth_cloud"), ("off", "none")])
def test_grammar_and_flat_spellings_build_the_same_config(raw, resolved):
    spec_grammar = parse_groups(
        dust_attenuation={"type": "two_component", "law": "calzetti", "nebular_screen": raw},
        redshift=Fixed(0.1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec_flat = Parameters(
            dust_model="two_component",
            dust_law_bc="calzetti",
            dust_law_diff="calzetti",
            dust_nebular_screen=raw,
            redshift=Fixed(0.1),
        )
    assert spec_grammar.dust_nebular_screen == resolved
    assert spec_flat.dust_nebular_screen == resolved


def test_screen_choices_round_trip_through_to_groups():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "nebular_screen": "diffuse",
                "shock_screen": "off",
            },
            redshift=Fixed(0.1),
        )
        groups = parameters_to_groups(spec)
    dust_group = groups["dust_attenuation"]
    assert dust_group["nebular_screen"] == "diffuse"
    # 'off' normalizes to (and round-trips as) its canonical spelling 'none'.
    assert dust_group["shock_screen"] == "none"
    # agn_screen was never touched and sits at its default -> not re-emitted.
    assert "agn_screen" not in dust_group


# ── (6) single_component refuses birth_cloud/diffuse on every source ──────

#: The non-default, non-'none' choice for each source: nebular's own default
#: is birth_cloud (so 'diffuse' is the one single_component must refuse),
#: shock's own default is diffuse (so 'birth_cloud' is refused), and agn has
#: no default other than 'none' (either non-'none' choice is refused, by the
#: separate agn-deferral rule -- 'diffuse' exercises that here).
_NON_DEFAULT_CHOICE = {"nebular": "diffuse", "shock": "birth_cloud", "agn": "diffuse"}


@pytest.mark.parametrize("source", ["nebular", "shock", "agn"])
def test_single_component_refuses_non_default_screen(source):
    choice = _NON_DEFAULT_CHOICE[source]
    kwargs = {f"dust_{source}_screen": choice}
    with pytest.raises(ParameterError):
        Parameters(
            dust_model="single_component",
            dust_law_bc="calzetti",
            redshift=Fixed(0.1),
            **kwargs,
        )
    with pytest.raises(ParameterError):
        parse_groups(
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                f"{source}_screen": choice,
            },
            redshift=Fixed(0.1),
        )


# ── (7) Unknown value raises naming the three choices ──────────────────────


def test_unknown_value_names_the_three_choices():
    with pytest.raises(ParameterError) as excinfo:
        Parameters(dust_nebular_screen="bogus", redshift=Fixed(0.1))
    message = str(excinfo.value)
    for choice in SCREEN_CHOICES:
        assert choice in message, f"{choice!r} missing from error message: {message}"


# ── (8) agn_screen != 'none' raises the deferral message ───────────────────


def test_agn_screen_non_none_raises_deferral_message():
    with pytest.raises(ParameterError, match="polar-dust screen"):
        Parameters(dust_agn_screen="diffuse", redshift=Fixed(0.1))
    with pytest.raises(ParameterError, match="polar-dust screen"):
        parse_groups(
            dust_attenuation={"type": "two_component", "law": "calzetti", "agn_screen": "diffuse"},
            redshift=Fixed(0.1),
        )


# ── (9) Fast-nebular fallback honors nebular_screen (after the #2235 snapping fix) ──


def test_fast_nebular_fallback_honors_nebular_screen():
    """enable_fast_nebular's reconstruction respects a non-default nebular_screen.

    Builds two Cue models differing ONLY in ``nebular_screen`` (diffuse vs the
    birth_cloud default), enables the fast grid on each, and checks that the
    fast path's line transmission matches the EXACT (state-based) path for
    its own screen choice at float precision -- proving the fallback reads
    ``self.config.nebular_screen`` rather than a hardcoded formula.
    """
    import tengri
    from tengri import DEFAULT, Fixed as F, Observation, Photometry, SEDModel, Uniform

    try:
        ssp_bare = tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"default bare-stellar SSP not available: {exc}")

    def band(center, n=24):
        from tengri.observation.photometry import FilterCurve

        wave = np.linspace(center * 0.85, center * 1.15, n)
        trans = np.sin(np.linspace(0.0, np.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")

    observation = Observation(
        photometry=Photometry(filters=tuple(band(c) for c in (1500.0, 2200.0, 6200.0)))
    )
    target_wavelengths = jnp.asarray([4862.71, 6564.72])

    def build(nebular_screen):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SEDModel.build(
                ssp_data=ssp_bare,
                observation=observation,
                sfh={
                    "type": "const",
                    "all_params": F(DEFAULT),
                    "log_total_mass": 10.0,
                    "start_gyr": 10.0,
                    "end_gyr": 0.0,
                },
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "calzetti",
                    "law_diff": "calzetti",
                    "tau_bc": Uniform(0.1, 1.0),
                    "tau_diff": Uniform(0.1, 3.0),
                    "nebular_screen": nebular_screen,
                },
                neb={"type": "cue", "all_params": F(DEFAULT)},
                redshift=F(0.05),
            )

    model = build("diffuse")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.enable_fast_nebular(target_wavelengths, n_grid=2)
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))

        fast_atten = np.asarray(model.predict_line_fluxes(params, redden=True))
        fast_intr = np.asarray(model.predict_line_fluxes(params, redden=False))
        fast_transmission = fast_atten / fast_intr

        state = model.predict_state(params)
        exact_atten = np.asarray(
            model.predict_line_fluxes(
                params, target_wavelengths=target_wavelengths, state=state, redden=True
            )
        )
        exact_intr = np.asarray(
            model.predict_line_fluxes(
                params, target_wavelengths=target_wavelengths, state=state, redden=False
            )
        )
    exact_transmission = exact_atten / exact_intr

    np.testing.assert_allclose(fast_transmission, exact_transmission, rtol=1e-10, atol=0.0)

    # Non-vacuity: the diffuse screen must actually attenuate (not a no-op),
    # otherwise the parity check above would pass trivially at 1.0 everywhere.
    assert np.all(fast_transmission < 1.0)
    assert np.all(fast_transmission > 0.0)


# ── (10) Energy balance: L_absorbed rises by exactly the shock power removed ─


def test_energy_balance_gains_exactly_the_shock_absorbed_power():
    """L_absorbed with shock minus without equals the shock screen's absorbed power.

    Conservation check (#2234): the shock SED's absorbed power
    (intrinsic - attenuated, integrated over frequency) must enter
    ``L_absorbed`` under its OWN ``shock_screen`` choice -- the pre-existing
    gap #2234 closes. Computed both sides purely from the component's own
    published state: no independent re-implementation of the integral.
    """
    from tengri.forward.energy_balance import bolometric_absorbed_log10
    from tengri.utils.physics_constants import C_AA
    from tengri.utils.scale import pow10

    config = _config(shock_screen="diffuse")
    comp = DustSEDComponent(config=config)
    sed_shock = jnp.full(_WAVE.shape, 3.0e27)

    lnu_age = jnp.zeros((_AGES.shape[0], _WAVE.shape[0])).at[2, :].set(1.0e29)
    stellar = jnp.sum(lnu_age, axis=0)

    state_with_shock = ForwardState(
        wave=_WAVE,
        sed_intrinsic=stellar + sed_shock,
        derived={"lnu_age": lnu_age, "ssp_ages_yr": _AGES, "sed_shock": sed_shock},
    )
    state_without_shock = ForwardState(
        wave=_WAVE,
        sed_intrinsic=stellar,
        derived={"lnu_age": lnu_age, "ssp_ages_yr": _AGES},
    )

    out_with = comp.apply(state_with_shock, _PARAMS)
    out_without = comp.apply(state_without_shock, _PARAMS)

    L_absorbed_with = float(np.asarray(out_with.derived["L_absorbed"]))
    L_absorbed_without = float(np.asarray(out_without.derived["L_absorbed"]))

    # The shock power removed under its OWN screen (diffuse): sed_shock is
    # unattenuated intrinsic; recover the attenuated form the SAME way the
    # hand-computed tests above do (out.sed_intrinsic - stellar_attenuated).
    stellar_att = np.asarray(out_with.derived["sed_dust_attenuated"])
    sed_shock_attenuated = np.asarray(out_with.sed_intrinsic) - stellar_att
    nu = np.asarray(C_AA / _WAVE)
    log_shock_absorbed, _ = bolometric_absorbed_log10(
        jnp.asarray(sed_shock), jnp.asarray(sed_shock_attenuated), jnp.asarray(nu), wave=_WAVE
    )
    shock_absorbed = float(pow10(log_shock_absorbed))

    assert shock_absorbed > 0.0, (
        "shock must be genuinely absorbed on this fixture, or the "
        "conservation check below is vacuous."
    )
    np.testing.assert_allclose(
        L_absorbed_with - L_absorbed_without, shock_absorbed, rtol=1e-6, atol=0.0
    )
