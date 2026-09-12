# SPDX-License-Identifier: BSD-3-Clause
"""Test that radio parameter defaults are declared once and reused everywhere.

Ensure that:

1. Every ``alpha_sf``/``nu_ref`` function signature default is a named module
   constant (never a bare numeric literal) and matches ITS OWN declared
   source -- Delvecchio+2021 and McCheyne+2022 cite their own 0.7 (Novak+2017
   / SEMPER), distinct from the Bell-2003-family/registry default 0.8
   (Condon 1992); a blanket "everyone shares one value" assumption is wrong
   (ruling R8).
2. ``nu_ref`` wiring in ``radio_agn``/``radio_agn_dpl`` actually affects the
   output and defaults to 5 GHz.
3. ``RadioPowerLawSEDComponent.alpha_sf`` and ``RadioDPL.alpha_sf`` derive
   from the canonical ``radio_alpha_sf`` declaration with the correct
   (positive) sign convention, so the default SF radio spectrum FALLS with
   rising frequency (round-2 physics fix).
"""

import ast
import inspect

import jax.numpy as jnp

from tengri.components.radio import radio
from tengri.components.radio._params import PARAMS
from tengri.components.radio.radio_dpl_model import RadioDPL
from tengri.components.radio.radio_model import RadioPowerLawSEDComponent
from tengri.protocols.component import declared_default
from tengri.utils.physics_constants import C_AA

#: Each function's alpha_sf default must equal ITS OWN owning constant, not
#: a single value assumed to apply to every function (round-1's hand-listed
#: subset and round-2's blanket-0.8 reading both missed this). Explicit dict
#: rather than an inferred rule, so a function that legitimately needs a
#: different source is a deliberate addition here, not a silent pass.
_ALPHA_SF_EXPECTED_SOURCE: dict[str, float] = {
    "radio_sfr_bell2003": radio._ALPHA_SF_DEFAULT,
    "radio_star_forming": radio._ALPHA_SF_DEFAULT,  # backward-compat alias of bell2003
    "radio_sfr_delvecchio2021": radio._ALPHA_SF_DELVECCHIO2021,
    "radio_sfr_mccheyne2022": radio._ALPHA_SF_MCCHEYNE2022,
    "radio_total_terms": radio._ALPHA_SF_DEFAULT,
    "radio_total": radio._ALPHA_SF_DEFAULT,
    "radio_total_dpl_terms": radio._ALPHA_SF_DEFAULT,
    "radio_total_dpl": radio._ALPHA_SF_DEFAULT,
    "compute_radio_components": radio._ALPHA_SF_DEFAULT,
}

#: Signature-default parameter names this module's AST guard tracks.
_TRACKED_PARAM_NAMES = frozenset({"alpha_sf", "nu_ref"})


def _iter_arg_defaults(func_def: ast.FunctionDef):
    """Yield ``(arg_name, default_node)`` for every parameter with a default."""
    args = func_def.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    offset = len(positional) - len(defaults)
    for arg, default in zip(positional[offset:], defaults, strict=True):
        yield arg.arg, default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if default is not None:
            yield arg.arg, default


def _find_literal_default_violations(module) -> list[tuple[str, str, object]]:
    """Return ``(function, param, literal_value)`` for every tracked param
    whose signature default is a bare numeric literal (``ast.Constant``)
    rather than a named module constant (``ast.Name``/``ast.Attribute``).
    """
    tree = ast.parse(inspect.getsource(module))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for name, default in _iter_arg_defaults(node):
            if name not in _TRACKED_PARAM_NAMES:
                continue
            if isinstance(default, ast.Constant) and isinstance(default.value, int | float):
                violations.append((node.name, name, default.value))
    return violations


def test_alpha_sf_signature_defaults_match_declared():
    """Every alpha_sf function signature default equals ITS OWN declared
    constant (not a single value assumed for every function).
    """
    discovered = set()
    for name, obj in inspect.getmembers(radio, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(obj)
        if "alpha_sf" not in sig.parameters:
            continue
        if sig.parameters["alpha_sf"].default is inspect.Parameter.empty:
            continue
        discovered.add(name)

        assert name in _ALPHA_SF_EXPECTED_SOURCE, (
            f"{name} has an alpha_sf default but is not mapped in "
            f"_ALPHA_SF_EXPECTED_SOURCE; add its owning constant"
        )
        expected = _ALPHA_SF_EXPECTED_SOURCE[name]
        actual = sig.parameters["alpha_sf"].default
        assert actual == expected, f"{name}.alpha_sf has default {actual}, expected {expected}"

    assert discovered, "No public functions with alpha_sf parameter and default found"
    # Guard the map itself against silently going stale (function
    # removed/renamed/added without updating the mapping above).
    assert discovered == set(_ALPHA_SF_EXPECTED_SOURCE), (
        "_ALPHA_SF_EXPECTED_SOURCE is out of sync with discovered functions: "
        f"missing={set(_ALPHA_SF_EXPECTED_SOURCE) - discovered}, "
        f"extra={discovered - set(_ALPHA_SF_EXPECTED_SOURCE)}"
    )


def test_alpha_sf_and_nu_ref_defaults_are_named_constants_not_literals():
    """AST-level guard: no ``alpha_sf``/``nu_ref`` signature default may be a
    bare numeric literal. The value-equality check above alone cannot
    distinguish a literal ``0.8`` from the declared constant (both compare
    equal) -- this is why three literal ``alpha_sf: float = 0.8`` defaults
    survived round 1's hand-listed subset undetected.
    """
    violations = _find_literal_default_violations(radio)
    assert not violations, (
        f"Found numeric literal defaults for alpha_sf/nu_ref (must be named "
        f"module constants instead): {violations}"
    )


def test_nu_ref_default_is_five_ghz():
    """nu_ref default in radio_agn must be 5.0e9 Hz (5 GHz)."""
    sig = inspect.signature(radio.radio_agn)
    actual_default = sig.parameters["nu_ref"].default
    assert actual_default == 5.0e9, (
        f"radio_agn nu_ref default is {actual_default}, expected 5.0e9 (5 GHz)"
    )


def test_radio_agn_dpl_nu_ref_default_is_five_ghz():
    """radio_agn_dpl's nu_ref must also default to 5 GHz.

    It previously restated ``nu_ref: float = 5.0e9`` as its own literal
    (round-2 finding item 8) instead of sharing ``_NU_REF_AGN_HZ`` with
    ``radio_agn``.
    """
    sig = inspect.signature(radio.radio_agn_dpl)
    actual_default = sig.parameters["nu_ref"].default
    assert actual_default == radio._NU_REF_AGN_HZ
    assert actual_default == 5.0e9


def test_nu_ref_parameter_affects_radio_agn_output():
    """nu_ref parameter in radio_agn should affect output, not be dead.

    Also verifies that default call (5 GHz) differs from alternative (1 GHz).
    """
    L_agn_bol = 1e46  # erg/s
    radio_loudness = 0.0
    alpha_agn = 0.7
    wavelength = jnp.logspace(2, 5, 100)  # Angstrom

    # Compute with default nu_ref (5 GHz)
    output_default = radio.radio_agn(
        wavelength,
        L_agn_bol=L_agn_bol,
        radio_loudness=radio_loudness,
        alpha_agn=alpha_agn,
    )

    # Compute with explicit 5 GHz (should be bit-exact with default)
    output_explicit_5ghz = radio.radio_agn(
        wavelength,
        L_agn_bol=L_agn_bol,
        radio_loudness=radio_loudness,
        alpha_agn=alpha_agn,
        nu_ref=5e9,
    )

    # Default and explicit 5 GHz should be bit-exact
    assert jnp.allclose(output_default, output_explicit_5ghz, rtol=0, atol=0), (
        "radio_agn default nu_ref should produce bit-exact results with 5e9"
    )

    # Compute with different nu_ref (1e9 Hz instead of default 5e9)
    output_alt_nu_ref = radio.radio_agn(
        wavelength,
        L_agn_bol=L_agn_bol,
        radio_loudness=radio_loudness,
        alpha_agn=alpha_agn,
        nu_ref=1e9,
    )

    # Outputs should differ (nu_ref parameter must be used)
    assert not jnp.allclose(output_default, output_alt_nu_ref), (
        "nu_ref parameter should affect radio_agn output, "
        "but default and alternative nu_ref gave the same result"
    )


def test_radio_dpl_alpha_sf_matches_declared_default():
    """RadioDPL.alpha_sf must derive from the canonical declaration, not
    restate its value as a literal.
    """
    comp = RadioDPL()
    assert comp.alpha_sf.default == declared_default(PARAMS, "radio_alpha_sf")


def test_radio_powerlaw_alpha_sf_matches_declared_and_falls_with_frequency():
    """RadioPowerLawSEDComponent.alpha_sf must be the canonical positive-slope
    declaration (Uniform(0.5, 1.2, default=0.8)), and its default SED must
    FALL with rising frequency (L_nu ∝ nu^(-alpha_sf)).

    Round-2 physics fix: the class previously derived
    ``default=-declared_default(...)`` (= −0.8), fed unchanged by
    ``predict()`` into ``radio_total`` -> ``radio_sfr_bell2003``
    (``L_nu = L_ref * (nu/nu_ref)**(-alpha_sf)``), producing a spectrum that
    RISES with frequency. The test that existed at that time only ever
    evaluated ``radio_agn`` (the AGN jet term), never this component's own
    SF term, so the defect shipped undetected.
    """
    declared = next(d.free_prior for d in PARAMS if d.name == "radio_alpha_sf")
    comp = RadioPowerLawSEDComponent()

    assert comp.alpha_sf.lo == declared.lo
    assert comp.alpha_sf.hi == declared.hi
    assert comp.alpha_sf.default == declared.default

    wave_14ghz = jnp.array([C_AA / 1.4e9])
    wave_5ghz = jnp.array([C_AA / 5.0e9])
    p = {
        "q_ir": comp.q_ir.default,
        "alpha_sf": comp.alpha_sf.default,
        "loudness": comp.loudness.value,
        "alpha_agn": comp.alpha_agn.default,
        "T_e": comp.T_e.value,
        "alpha_ff": comp.alpha_ff.value,
        "redshift": 0.0,
    }
    _, published_14ghz = comp.predict(p, jnp.zeros_like(wave_14ghz), wave_14ghz, L_ir=1e43)
    _, published_5ghz = comp.predict(p, jnp.zeros_like(wave_5ghz), wave_5ghz, L_ir=1e43)
    l_14ghz = float(published_14ghz["sed_radio"][0])
    l_5ghz = float(published_5ghz["sed_radio"][0])
    assert l_5ghz < l_14ghz, (
        f"Default radio_powerlaw spectrum must fall with frequency; "
        f"got L(1.4 GHz)={l_14ghz:.3e}, L(5 GHz)={l_5ghz:.3e}"
    )
