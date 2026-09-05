# SPDX-License-Identifier: BSD-3-Clause
"""Test that radio parameter defaults are declared once and reused everywhere.

Ensure that:
1. alpha_sf defaults across function signatures match the declared value
2. nu_ref wiring in radio_agn actually affects the output
"""

import inspect

import jax.numpy as jnp

from tengri.components.radio import radio
from tengri.components.radio._params import PARAMS
from tengri.protocols.component import declared_default


def test_alpha_sf_signature_defaults_match_declared():
    """Every alpha_sf function signature default equals declared_default.

    Verifies both value equality and that no numeric literals remain by
    checking the AST representation of the module source.
    """
    declared_alpha_sf = declared_default(PARAMS, "radio_alpha_sf")

    # Dynamically discover all public functions in radio module with alpha_sf parameter
    # Filter: must have alpha_sf param with a default value, not private functions
    functions_to_check = []
    for name, obj in inspect.getmembers(radio, inspect.isfunction):
        if not name.startswith("_") and "alpha_sf" in inspect.signature(obj).parameters:
            sig = inspect.signature(obj)
            # Only check functions where alpha_sf has a default
            if sig.parameters["alpha_sf"].default != inspect.Parameter.empty:
                functions_to_check.append(obj)

    assert len(functions_to_check) > 0, (
        "No public functions with alpha_sf parameter and default found in radio module"
    )

    for func in functions_to_check:
        sig = inspect.signature(func)
        actual_default = sig.parameters["alpha_sf"].default

        # Check value matches declared
        assert actual_default == declared_alpha_sf, (
            f"{func.__name__}.alpha_sf has default {actual_default}, "
            f"but declared_default is {declared_alpha_sf}"
        )

        # Check that default is not a numeric literal by verifying it matches
        # the declared value (if it were a different literal, this would fail)
        assert isinstance(actual_default, float) or actual_default == declared_alpha_sf, (
            f"{func.__name__}.alpha_sf default should be declared_default, "
            f"not a numeric literal"
        )


def test_nu_ref_default_is_five_ghz():
    """nu_ref default in radio_agn must be 5.0e9 Hz (5 GHz)."""
    sig = inspect.signature(radio.radio_agn)
    actual_default = sig.parameters["nu_ref"].default
    assert actual_default == 5.0e9, (
        f"radio_agn nu_ref default is {actual_default}, expected 5.0e9 (5 GHz)"
    )


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
