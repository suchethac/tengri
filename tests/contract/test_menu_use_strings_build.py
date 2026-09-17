# SPDX-License-Identifier: BSD-3-Clause
"""Every production menu row's `use` string must build.

The `use` field is a copy-pasteable hint for "how do I actually use this?"
rendered from the registry. A `use` string that SEDModel.build rejects is
broken, because a user who copies it into their notebook gets a failure,
not the model they asked for. This sweep evaluates every production row's
`use` as a dict-grammar snippet and verifies it builds.
"""

from __future__ import annotations

import ast

import pytest

import tengri
from tengri import Fixed, Observation, Photometry, SEDModel

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def bare_stellar_ssp():
    """A bare-stellar SSP for building models."""
    from pathlib import Path

    if not (
        Path(__file__).resolve().parents[2] / "data" / "fsps_prsc_miles_chabrier.h5"
    ).is_file():
        pytest.skip("bare-stellar SSP grid not available")
    return tengri.load_ssp("fsps_prsc_miles_chabrier")


@pytest.fixture(scope="module")
def observation() -> Observation:
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))


def _parse_use_string(use_str: str) -> dict | None:
    """Extract all kwargs from a use string like 'SEDModel.build(..., sfh={...}, dust=...)'."""
    # Find the start of SEDModel.build(...,
    start_idx = use_str.find("SEDModel.build(...,")
    if start_idx == -1:
        return None

    # Extract everything after "SEDModel.build(..., " and before the closing )
    remainder = use_str[start_idx + len("SEDModel.build(..., "):]
    remainder = remainder.rstrip(")")

    # Parse kwargs using a simple state machine to handle nested dicts
    result = {}
    pos = 0
    while pos < len(remainder):
        # Find the next '=' to get the kwarg name
        eq_pos = remainder.find("=", pos)
        if eq_pos == -1:
            break

        kwarg_name = remainder[pos:eq_pos].strip()
        pos = eq_pos + 1

        # Now find the value (a dict)
        # Skip whitespace
        while pos < len(remainder) and remainder[pos].isspace():
            pos += 1

        # Find matching closing brace for the dict
        if remainder[pos] != "{":
            return None

        brace_count = 0
        start_pos = pos
        while pos < len(remainder):
            if remainder[pos] == "{":
                brace_count += 1
            elif remainder[pos] == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            pos += 1

        dict_str = remainder[start_pos : pos + 1]
        try:
            dict_value = ast.literal_eval(dict_str)
            result[kwarg_name] = dict_value
        except (ValueError, SyntaxError):
            return None

        # Move past the closing brace and find the next kwarg
        pos += 1
        # Skip comma and whitespace
        while pos < len(remainder) and (remainder[pos] in ", " or remainder[pos].isspace()):
            pos += 1

    return result if result else None


def _get_production_menu_rows():
    """Collect all production status menu rows."""
    rows = []

    menus = [
        ("radio_models", tengri.list_radio_models()),
        ("dust_laws", tengri.list_dust_laws()),
        ("dust_models", tengri.list_dust_models()),
        ("dust_emission_models", tengri.list_dust_emission_models()),
        ("sfh_models", tengri.list_sfh_models()),
        ("nebular_backends", tengri.list_nebular_backends()),
        ("metallicity_modes", tengri.list_metallicity_modes()),
        ("igm_models", tengri.list_igm_models()),
        ("xray_models", tengri.list_xray_models()),
        ("shock_models", tengri.list_shock_models()),
        ("age_kernels", tengri.list_age_kernels()),
        ("agn_blocks", tengri.list_agn_blocks()),
    ]

    for menu_name, menu_rows in menus:
        for row in menu_rows:
            if row.get("status") == "production":
                rows.append((menu_name, row["name"], row))

    return rows


@pytest.mark.parametrize(
    ("menu_name", "entry_name", "row"),
    [(m, n, r) for m, n, r in _get_production_menu_rows()],
    ids=[f"{m}:{n}" for m, n, _ in _get_production_menu_rows()],
)
def test_production_row_use_string_builds(
    menu_name, entry_name, row, bare_stellar_ssp, observation
):
    """Every production menu row's use string must build or be marked not-buildable."""
    use_str = row.get("use", "")

    # Rows marked as "not builder-available" are correct even if they don't build
    if "not builder-available" in use_str:
        pytest.skip(f"{entry_name} is marked not builder-available")

    # Parse the use string to extract the kwargs
    kwargs = _parse_use_string(use_str)
    if kwargs is None:
        pytest.fail(
            f"{menu_name}/{entry_name}: could not parse use string: {use_str}"
        )

    # Try to build with the parsed kwargs
    try:
        SEDModel.build(
            ssp_data=bare_stellar_ssp,
            observation=observation,
            redshift=Fixed(0.1),
            **kwargs,
        )
    except Exception as exc:
        pytest.fail(
            f"{menu_name}/{entry_name}: use string '{use_str}' failed to build: {exc}"
        )
