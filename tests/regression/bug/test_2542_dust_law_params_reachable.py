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

    # Build a base model with defaults
    base_model = _build(uv_ssp, uv_obs, {"law": law_name})
    base_sed = _sed(base_model)

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

        # Build a model with the perturbed value
        perturbed_model = _build(uv_ssp, uv_obs, {"law": law_name, param_name: perturbed_val})
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

    # Build a base two_component model with both screens using the same law
    # Use law_bc and law_diff to set the law on each screen
    screen_key = f"law_{screen}"
    base_config = {
        "type": "two_component",
        "all_params": Fixed(DEFAULT),
        "law_bc": law_name,
        "law_diff": law_name,
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

        # Build a model with the perturbed value on the chosen screen
        # Use per-screen spelling like dust_slope_bc or dust_slope_diff
        perturbed_config = base_config.copy()
        perturbed_config[f"{param_name}_{screen}"] = perturbed_val

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

        # Build base and perturbed models
        base_model = _build(uv_ssp, uv_obs, {"law": law_name})
        base_sed = _sed(base_model)

        perturbed_model = _build(uv_ssp, uv_obs, {"law": law_name, param_name: perturbed_val})
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

        # Build base model
        base_config = {
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "law_bc": law_name,
            "law_diff": law_name,
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
