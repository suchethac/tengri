# SPDX-License-Identifier: BSD-3-Clause
"""Test that radio parameter defaults are declared once and reused everywhere.

Ensure that:
1. alpha_sf defaults across function signatures match the declared value
2. nu_ref wiring in radio_agn actually affects the output
"""

import inspect

import jax.numpy as jnp
import pytest

from tengri.components.radio import radio
from tengri.components.radio._params import PARAMS
from tengri.protocols.component import declared_default


def test_alpha_sf_signature_defaults_match_declared():
    """Every alpha_sf function signature default equals declared_default."""
    declared_alpha_sf = declared_default(PARAMS, "radio_alpha_sf")

    # Functions with alpha_sf signatures
    functions_to_check = [
        radio.radio_sfr_bell2003,
        radio.radio_sfr_delvecchio2021,
        radio.radio_sfr_mccheyne2022,
        radio.radio_total_terms,
        radio.radio_total,
    ]

    for func in functions_to_check:
        sig = inspect.signature(func)
        if "alpha_sf" in sig.parameters:
            actual_default = sig.parameters["alpha_sf"].default
            assert actual_default == declared_alpha_sf, (
                f"{func.__name__}.alpha_sf has default {actual_default}, "
                f"but declared_default is {declared_alpha_sf}"
            )


def test_nu_ref_parameter_affects_radio_agn_output():
    """nu_ref parameter in radio_agn should affect output, not be dead."""
    L_agn_bol = 1e46  # erg/s
    radio_loudness = 0.0
    alpha_agn = 0.7
    wavelength = jnp.logspace(2, 5, 100)  # Angstrom

    # Compute with default nu_ref
    output_default = radio.radio_agn(
        wavelength,
        L_agn_bol=L_agn_bol,
        radio_loudness=radio_loudness,
        alpha_agn=alpha_agn,
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
