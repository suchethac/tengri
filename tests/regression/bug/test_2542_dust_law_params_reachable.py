# SPDX-License-Identifier: BSD-3-Clause
"""Every dust-law signature parameter is reachable through the grammar (#2542).

The li08, noll09, salim_sbl18, and tea attenuation laws read parameters that
were not declared in the parameter grammar and were masked by _KNOWN_UNDECLARED.

- li08 reads: dust_c1, dust_c2, dust_c3, dust_c4
- noll09 reads: dust_bump_x0, dust_bump_gamma
- salim_sbl18 reads: dust_bump_x0, dust_bump_gamma
- tea reads: dust_tea_scatter

Part A adds 7 ParamDeclaration entries to ATTENUATION_PARAMS, each Fixed at the
law's own default with a sensible free_prior. Part A also extends
two_component's resolve_bc_diff_law_params() and merge_neb_screen_live_overrides()
to forward all law parameters to all screens (bc/diff/neb identically).

This test sweeps every registered law and every parameter in its signature,
asserting that:
1. Every parameter is declared in ATTENUATION_PARAMS
2. Setting a parameter to a non-default value CHANGES the SED (single_component)
3. Setting a parameter to a non-default value CHANGES the SED (two_component on bc/diff/neb)
4. For li08 and noll09: the curve equals the law function evaluated directly (rtol 1e-10)

Taxonomy: regression_bug (#2542)
"""

from __future__ import annotations

import inspect
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, SSPData
from tengri.components.dust._params import ATTENUATION_PARAMS
from tengri.components.dust.laws._registry import DUST_LAWS, law_kwarg_names
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

_INERT_TOL = 1e-9


@pytest.fixture(scope="module")
def uv_ssp() -> SSPData:
    """SSP grid spanning ages and metallicities for UV testing."""
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 1200)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-30, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


@pytest.fixture(scope="module")
def uv_obs() -> Observation:
    """Bands bracketing the UV slope and 2175 A bump."""

    def _tophat(center: float, frac: float = 0.10, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (1500.0, 2175.0, 2800.0, 4400.0, 6200.0)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum relative difference between two arrays."""
    return float(np.max(np.abs(a - b) / np.where(np.abs(b) > 0, np.abs(b), 1.0)))


def _build(uv_ssp, uv_obs, dust: dict, screen: str = "single_component"):
    """Build a model with the given dust attenuation config and screen."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=uv_ssp,
            observation=uv_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={**dust, "type": screen},
            redshift=Fixed(0.5),
        )


def _sed(model) -> np.ndarray:
    """Sample and predict SED from a model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = model.spec.sample(jax.random.PRNGKey(0))
        return np.asarray(model.predict(params).rest_sed())


def _law_default(law_name: str, param_name: str) -> float | None:
    """Extract the default value for a law parameter from its signature."""
    law_func = DUST_LAWS.get(law_name)
    if law_func is None:
        return None
    sig = inspect.signature(law_func)
    if param_name not in sig.parameters:
        return None
    return sig.parameters[param_name].default


def _is_declared(param_name: str) -> bool:
    """Check whether a parameter is declared in ATTENUATION_PARAMS."""
    return any(p.name == param_name for p in ATTENUATION_PARAMS)


def _get_param_declaration(param_name: str):
    """Fetch the ParamDeclaration for a parameter name."""
    for p in ATTENUATION_PARAMS:
        if p.name == param_name:
            return p
    return None


# The four ``OVERRIDE_STEMS`` table entries (full flat names): the only
# parameters that support per-screen spelling (``dust_slope_bc``, ...). Every
# other declared attenuation parameter -- including the law-specific ones
# #2542 added (``dust_c1``-``c4``, ``dust_bump_x0``, ``dust_bump_gamma``,
# ``dust_tea_scatter``) -- only accepts the shared spelling (A3).
_PER_SCREEN_TABLED_NAMES = frozenset({"dust_slope", "dust_bump_strength", "dust_delta", "dust_Rv"})


def _enabling_overrides(param_name: str) -> dict[str, float]:
    """Extra params that must be nonzero for ``param_name``'s effect to show.

    ``noll09``/``salim_sbl18`` multiply their UV-bump Drude profile by
    ``dust_bump_strength`` (default 0.0, "no bump"), so sweeping
    ``dust_bump_x0`` or ``dust_bump_gamma`` alone -- holding
    ``dust_bump_strength`` at its default -- changes a term that is
    structurally zero either way, and the SED cannot move. This is not a
    #2542 forwarding defect; it is the same law reading its OWN declared
    default for a gating parameter. Both the baseline and the perturbed
    build must carry the SAME nonzero ``dust_bump_strength`` so the
    isolated effect is the swept parameter alone.
    """
    if param_name in ("dust_bump_x0", "dust_bump_gamma"):
        return {"dust_bump_strength": 0.5}
    return {}


# ── Test 1: Every law parameter is declared in ATTENUATION_PARAMS ─────────


@pytest.mark.parametrize("law_name", list(DUST_LAWS.keys()), ids=lambda x: x)
def test_every_law_parameter_is_declared(law_name: str):
    """Every parameter in a law's signature must be in ATTENUATION_PARAMS (except wavelength)."""
    kw_set = law_kwarg_names(law_name)
    subset = kw_set - {"wavelength", "redshift"}
    declared_names = {param.name for param in ATTENUATION_PARAMS}

    missing = subset - declared_names
    assert not missing, (
        f"Law {law_name!r} has undeclared parameters: {missing}. Declared: {declared_names}"
    )


# ── Test 2: Single component — setting a parameter changes the SED ─────────


@pytest.mark.parametrize("law_name", list(DUST_LAWS.keys()), ids=lambda x: x)
def test_single_component_param_reaches_curve(law_name: str, uv_ssp, uv_obs):
    """For single_component, setting a law parameter to a non-default value changes the SED.

    This test iterates over every parameter of every law and verifies it is reachable
    through the grammar by checking that the SED changes when the parameter changes.
    """
    kw_set = law_kwarg_names(law_name)
    params_to_test = kw_set - {"wavelength", "redshift"}
    if not params_to_test:
        # Enumerated from the registry, not hand-listed: a law with a bare
        # ``(wavelength)`` signature (calzetti, smc, lmc, ...) has nothing to
        # sweep. Skip explicitly rather than let the for-loop below execute
        # zero assertions and report a silent PASS (#2542 review finding).
        pytest.skip(f"{law_name} declares no shape parameters beyond wavelength")

    # For each parameter, test that a non-default value changes the SED
    for param_name in params_to_test:
        default_val = _law_default(law_name, param_name)
        if default_val is None or default_val == inspect.Parameter.empty:
            pytest.skip(f"{law_name}.{param_name}: no default found")

        # Choose a perturbed value (roughly ±20% or ±0.1 in absolute units)
        if isinstance(default_val, (int, float)) and default_val != 0:
            perturbed_val = default_val * 1.2
        else:
            perturbed_val = default_val + 0.1

        enabling = _enabling_overrides(param_name)
        base_model = _build(uv_ssp, uv_obs, {"law": law_name, **enabling})
        base_sed = _sed(base_model)

        # Build a model with the perturbed value
        perturbed_model = _build(
            uv_ssp, uv_obs, {"law": law_name, **enabling, param_name: perturbed_val}
        )
        perturbed_sed = _sed(perturbed_model)

        rel_diff = _max_rel(perturbed_sed, base_sed)
        assert rel_diff > _INERT_TOL, (
            f"single_component: {law_name}.{param_name}={perturbed_val} did not change "
            f"the SED (max rel diff: {rel_diff}). The grammar accepted the value but "
            f"the law never saw it."
        )


# ── Test 3: Two component (bc/diff) — parameters reach all screens ─────


@pytest.mark.parametrize("law_name", list(DUST_LAWS.keys()), ids=lambda x: x)
@pytest.mark.parametrize("screen", ["bc", "diff"])
def test_two_component_param_reaches_bc_diff_screens(law_name: str, screen: str, uv_ssp, uv_obs):
    """For two_component on bc/diff screens, setting a law parameter changes the SED.

    This verifies that resolve_bc_diff_law_params() correctly forwards every law
    parameter to the law on all screens (bc, diff).

    Before Part A, bc and diff screens silently dropped law-specific parameters.
    """
    kw_set = law_kwarg_names(law_name)
    params_to_test = kw_set - {"wavelength", "redshift"}
    if not params_to_test:
        # Same reasoning as test_single_component_param_reaches_curve: a
        # parameter-less law has nothing to sweep on this screen either.
        pytest.skip(f"{law_name} declares no shape parameters beyond wavelength")

    # For each parameter, test that a non-default value changes the SED
    for param_name in params_to_test:
        default_val = _law_default(law_name, param_name)
        if default_val is None or default_val == inspect.Parameter.empty:
            pytest.skip(f"{law_name}.{param_name}: no default found")

        # Choose a perturbed value
        if isinstance(default_val, (int, float)) and default_val != 0:
            perturbed_val = default_val * 1.2
        else:
            perturbed_val = default_val + 0.1

        enabling = _enabling_overrides(param_name)
        # Build a base two_component model with both screens using the same
        # law (law_bc and law_diff set the law on each screen).
        base_config = {
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "law_bc": law_name,
            "law_diff": law_name,
            **enabling,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model = SEDModel.build(
                ssp_data=uv_ssp,
                observation=uv_obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation=base_config,
                redshift=Fixed(0.5),
            )
        base_sed = _sed(base_model)

        # Build a model with the perturbed value on the chosen screen. The
        # four OVERRIDE_STEMS table entries (slope/bump_strength/delta/Rv)
        # accept a per-screen spelling (dust_slope_bc); every other declared
        # attenuation parameter -- including every law-specific one #2542
        # added -- accepts only the shared spelling (A3), which moves BOTH
        # screens identically. Either way, the value must reach the law.
        perturbed_config = dict(base_config)
        if param_name in _PER_SCREEN_TABLED_NAMES:
            perturbed_config[f"{param_name}_{screen}"] = perturbed_val
        else:
            perturbed_config[param_name] = perturbed_val

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perturbed_model = SEDModel.build(
                ssp_data=uv_ssp,
                observation=uv_obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation=perturbed_config,
                redshift=Fixed(0.5),
            )
        perturbed_sed = _sed(perturbed_model)

        rel_diff = _max_rel(perturbed_sed, base_sed)
        assert rel_diff > _INERT_TOL, (
            f"two_component/{screen}: {law_name}.{param_name}={perturbed_val} did not "
            f"change the SED (max rel diff: {rel_diff}). The grammar accepted the value "
            f"but the law never saw it."
        )


# ── Test 4: li08, noll09, salim_sbl18, tea — law-specific parameters ────


@pytest.mark.parametrize(
    ("law_name", "param_names"),
    [
        ("li08", ["dust_c1", "dust_c2", "dust_c3", "dust_c4"]),
        ("noll09", ["dust_bump_x0", "dust_bump_gamma"]),
        ("salim_sbl18", ["dust_bump_x0", "dust_bump_gamma"]),
        ("tea", ["dust_tea_scatter"]),
    ],
)
def test_law_specific_params_single_component(
    law_name: str, param_names: list[str], uv_ssp, uv_obs
):
    """Law-specific parameters reach the law in single_component.

    li08 (c1-c4), noll09/salim_sbl18 (bump x0/gamma), and tea (scatter)
    are law-specific and must be reachable through the grammar.
    """
    # Verify each parameter is declared and settable
    for param_name in param_names:
        default_val = _law_default(law_name, param_name)
        if default_val is None or default_val == inspect.Parameter.empty:
            pytest.skip(f"{law_name}.{param_name}: no default found")

        # Choose a perturbed value
        if isinstance(default_val, (int, float)) and default_val != 0:
            perturbed_val = default_val * 1.2
        else:
            perturbed_val = default_val + 0.1

        # Build base and perturbed models. dust_bump_x0/dust_bump_gamma need
        # a nonzero dust_bump_strength to have any effect (see
        # _enabling_overrides); both builds carry the same one.
        enabling = _enabling_overrides(param_name)
        base_model = _build(uv_ssp, uv_obs, {"law": law_name, **enabling})
        base_sed = _sed(base_model)

        perturbed_model = _build(
            uv_ssp, uv_obs, {"law": law_name, **enabling, param_name: perturbed_val}
        )
        perturbed_sed = _sed(perturbed_model)

        rel_diff = _max_rel(perturbed_sed, base_sed)
        assert rel_diff > _INERT_TOL, (
            f"single_component: {law_name}.{param_name}={perturbed_val} did not "
            f"change the SED (max rel diff: {rel_diff}). The grammar accepted the "
            f"value but the law never saw it."
        )


@pytest.mark.parametrize(
    ("law_name", "param_names"),
    [
        ("li08", ["dust_c1", "dust_c2", "dust_c3", "dust_c4"]),
        ("noll09", ["dust_bump_x0", "dust_bump_gamma"]),
        ("salim_sbl18", ["dust_bump_x0", "dust_bump_gamma"]),
        ("tea", ["dust_tea_scatter"]),
    ],
)
def test_law_specific_params_two_component(law_name: str, param_names: list[str], uv_ssp, uv_obs):
    """Law-specific parameters reach the law in two_component on all screens.

    Before Part A, bc/diff screens silently dropped law-specific parameters like
    dust_c1-c4, dust_bump_x0/gamma, and dust_tea_scatter. These parameters do NOT
    support per-screen spelling (only shared), so we test that the shared spelling
    reaches both screens correctly.
    """
    # Build base and perturbed models for each law-specific parameter
    for param_name in param_names:
        default_val = _law_default(law_name, param_name)
        if default_val is None or default_val == inspect.Parameter.empty:
            pytest.skip(f"{law_name}.{param_name}: no default found")

        # Choose a perturbed value
        if isinstance(default_val, (int, float)) and default_val != 0:
            perturbed_val = default_val * 1.2
        else:
            perturbed_val = default_val + 0.1

        # Build base model. dust_bump_x0/dust_bump_gamma need a nonzero
        # dust_bump_strength to have any effect (see _enabling_overrides).
        enabling = _enabling_overrides(param_name)
        base_config = {
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "law_bc": law_name,
            "law_diff": law_name,
            **enabling,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model = SEDModel.build(
                ssp_data=uv_ssp,
                observation=uv_obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation=base_config,
                redshift=Fixed(0.5),
            )
        base_sed = _sed(base_model)

        # Build perturbed model with shared parameter (no per-screen variants)
        # Law-specific parameters only support shared spelling, not bc/diff variants
        perturbed_config = base_config.copy()
        perturbed_config[param_name] = perturbed_val

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perturbed_model = SEDModel.build(
                ssp_data=uv_ssp,
                observation=uv_obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                dust_attenuation=perturbed_config,
                redshift=Fixed(0.5),
            )
        perturbed_sed = _sed(perturbed_model)

        rel_diff = _max_rel(perturbed_sed, base_sed)
        assert rel_diff > _INERT_TOL, (
            f"two_component: {law_name}.{param_name}={perturbed_val} did not "
            f"change the SED (max rel diff: {rel_diff}). The grammar accepted the value "
            f"but the law never saw it."
        )


# ── Test 5: li08 cross-code check against Synthesizer's independent Li08 ───
#
# tengri's li08 and Synthesizer's Li08 (Wilkins et al. 2025) implement the
# same Li et al. (2008) Eq. 1 functional form independently. Units differ:
# Synthesizer takes unyt Angstrom and returns the raw A_lambda/A_V (NOT
# pre-normalized to 1 at 5500 A); tengri returns k(lambda) normalized so
# k(5500 A) = 1 by construction. Both sides are normalized by their OWN value
# at 5500 A before comparing, per the brief.

_LI08_TABLE1_TEMPLATES = {
    # name: (c1, c2, c3, c4) == (UV_slope, OPT_NIR_slope, FUV_slope, bump)
    # in Synthesizer's naming. Li et al. (2008) Table 1.
    "Calzetti": (44.9, 7.56, 61.2, 0.0),
    "SMC": (38.7, 3.83, 6.34, 0.0),
    "MW": (14.4, 6.52, 2.04, 0.0519),
    "LMC": (4.47, 2.39, -0.988, 0.0221),
}


@pytest.mark.parametrize("template_name", list(_LI08_TABLE1_TEMPLATES.keys()), ids=lambda x: x)
def test_li08_matches_synthesizer_table1_templates(template_name: str):
    """tengri li08(wave, *template) == Synthesizer Li08(wave, ..., model=None).

    rtol 1e-10 over 0.1-3 um, for each of the four Li et al. (2008) Table 1
    templates, after normalizing both curves by their own value at 5500 A.
    """
    unyt = pytest.importorskip("unyt", reason="unyt not installed")
    synth_dust_attenuation = pytest.importorskip(
        "synthesizer.emission_models.transformers.dust_attenuation",
        reason="synthesizer not installed",
    )

    from tengri.components.dust.attenuation import li08 as tengri_li08

    c1, c2, c3, c4 = _LI08_TABLE1_TEMPLATES[template_name]
    wave_aa = np.logspace(np.log10(1000.0), np.log10(30000.0), 3000)  # 0.1-3 um

    tengri_k = np.asarray(
        tengri_li08(jnp.asarray(wave_aa), dust_c1=c1, dust_c2=c2, dust_c3=c3, dust_c4=c4)
    )

    lam_unyt = wave_aa * unyt.angstrom
    synth_raw = np.asarray(synth_dust_attenuation.Li08(lam_unyt, c1, c2, c3, c4, model=None))
    lam_v = np.array([5500.0]) * unyt.angstrom
    synth_v = np.asarray(synth_dust_attenuation.Li08(lam_v, c1, c2, c3, c4, model=None))[0]
    synth_k = synth_raw / synth_v

    np.testing.assert_allclose(
        tengri_k,
        synth_k,
        rtol=1e-10,
        err_msg=f"li08 {template_name} template disagrees with Synthesizer's Li08",
    )


# ── Test 6: exact-value check -- grammar-built curve == direct law call ────
#
# The brief's missing check: it is not enough that the SED "changes" (Tests
# 2-4 above); the VALUE that reaches the law through the grammar's forwarding
# machinery must equal the law evaluated directly with the same parameter,
# for li08 and noll09, on single_component AND on every two_component screen
# (bc, diff, neb). This exercises the exact functions #2542 touches:
# ``DustAttenuationSEDComponent._curve`` (single_component),
# ``resolve_bc_diff_law_params`` (bc/diff), and
# ``merge_neb_screen_live_overrides`` (neb, which inherits ``law_bc`` and the
# shared spelling when no ``*_neb`` override is given).

_EXACT_VALUE_CASES = [
    ("li08", "dust_c1", 25.0),
    ("li08", "dust_c3", 10.0),
    ("noll09", "dust_bump_gamma", 0.05),
    ("noll09", "dust_bump_x0", 0.22),
]

_EXACT_VALUE_WAVE = jnp.logspace(3.0, 4.3, 300)  # ~1000-20000 A


@pytest.mark.parametrize(
    ("law_name", "param_name", "value"), _EXACT_VALUE_CASES, ids=lambda v: str(v)
)
def test_exact_value_single_component(law_name: str, param_name: str, value: float):
    """single_component: curve built through ``_curve`` == direct law call."""
    from tengri.components.dust.attenuation import resolve_dust_law
    from tengri.components.dust.component import (
        DustAttenuationSEDComponent,
        DustAttenuationSEDComponentConfig,
    )

    comp = DustAttenuationSEDComponent(
        config=DustAttenuationSEDComponentConfig(
            law=law_name, live_shape_params=frozenset({param_name})
        )
    )
    curve_fn = comp._curve({param_name: value})
    grammar_k = np.asarray(curve_fn(_EXACT_VALUE_WAVE))
    direct_k = np.asarray(resolve_dust_law(law_name)(_EXACT_VALUE_WAVE, **{param_name: value}))

    np.testing.assert_allclose(
        grammar_k,
        direct_k,
        rtol=1e-10,
        err_msg=f"single_component: {law_name}.{param_name}={value} curve != direct law call",
    )


@pytest.mark.parametrize("screen", ["bc", "diff", "neb"])
@pytest.mark.parametrize(
    ("law_name", "param_name", "value"), _EXACT_VALUE_CASES, ids=lambda v: str(v)
)
def test_exact_value_two_component(law_name: str, param_name: str, value: float, screen: str):
    """two_component, every screen: forwarded curve == direct law call.

    ``resolve_bc_diff_law_params``/``merge_neb_screen_live_overrides`` are the
    two functions #2542 fixed to stop bc/diff from silently dropping
    law-specific parameters. Calling them directly (with the SAME
    ``live_shape_params`` the grammar computes for an explicitly-set
    parameter) is the minimal, exact reproduction of that forwarding path.
    """
    from tengri.components.dust._apply import (
        merge_neb_screen_live_overrides,
        resolve_bc_diff_law_params,
    )
    from tengri.components.dust.attenuation import resolve_dust_law
    from tengri.components.dust.laws._registry import select_law_kwargs

    live = frozenset({param_name})
    params = {param_name: value}
    bc_kwargs, diff_kwargs = resolve_bc_diff_law_params(
        params, {}, {}, live, bc_law=law_name, diff_law=law_name, redshift=None
    )
    if screen == "bc":
        kwargs = bc_kwargs
    elif screen == "diff":
        kwargs = diff_kwargs
    else:  # neb inherits law_bc and the shared spelling with no *_neb override
        neb_overrides = merge_neb_screen_live_overrides(params, (), live)
        kwargs = select_law_kwargs(law_name, {**bc_kwargs, **neb_overrides})

    assert kwargs.get(param_name) == value, (
        f"two_component/{screen}: {law_name}.{param_name} was dropped by the "
        f"forwarding resolver (got {kwargs}) -- the #2542 defect."
    )

    grammar_k = np.asarray(resolve_dust_law(law_name)(_EXACT_VALUE_WAVE, **kwargs))
    direct_k = np.asarray(resolve_dust_law(law_name)(_EXACT_VALUE_WAVE, **{param_name: value}))

    np.testing.assert_allclose(
        grammar_k,
        direct_k,
        rtol=1e-10,
        err_msg=(
            f"two_component/{screen}: {law_name}.{param_name}={value} curve != direct law call"
        ),
    )


# ── Test 7: A3 -- law-specific parameters refuse per-screen spelling ───────


@pytest.mark.parametrize(
    "key",
    ["c1_bc", "dust_c1_bc", "dust_bump_x0_bc", "dust_tea_scatter_neb"],
)
def test_law_specific_param_per_screen_spelling_raises(key: str):
    """A3: dust_c1/dust_bump_x0/dust_bump_gamma/dust_tea_scatter are NOT
    ``OVERRIDE_STEMS`` table entries, so unlike ``slope``/``bump_strength``/
    ``delta``/``Rv`` they do not get a per-screen spelling
    (``dust_c1_bc``). The grammar already refuses the per-screen spelling of
    these names with a clear "Unknown key" error -- verified directly here
    so a future change cannot silently start accepting (and discarding) it.
    """
    from tengri.parameters.groups import parse_groups

    with pytest.raises(ValueError, match=r"[Uu]nknown key"):
        parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law_bc": "li08",
                "law_diff": "li08",
                key: 20.0,
            },
            redshift=Fixed(0.1),
        )
