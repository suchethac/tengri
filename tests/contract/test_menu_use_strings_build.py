# SPDX-License-Identifier: BSD-3-Clause
"""Every production menu row's `use` string must build or carry not-builder-available.

The `use` field is a copy-pasteable hint for "how do I actually use this?"
rendered from the registry. A `use` string that SEDModel.build rejects is
broken, because a user who copies it into their notebook gets a failure,
not the model they asked for. This sweep evaluates every production row's
`use` as a dict-grammar snippet and verifies it builds.
"""

from __future__ import annotations

import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tengri.config.exceptions import TengriIOError

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def bare_stellar_ssp():
    """A bare-stellar SSP for building models."""
    from pathlib import Path

    ssp_path = Path(__file__).resolve().parents[2] / "data" / "fsps_prsc_miles_chabrier.h5"
    if not ssp_path.is_file():
        pytest.skip("bare-stellar SSP grid not available")
    return tengri.load_ssp("fsps_prsc_miles_chabrier")


@pytest.fixture(scope="module")
def observation() -> Observation:
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))


def _get_production_menu_rows() -> list[tuple[str, str, dict]]:
    """Collect all production status menu rows.

    Returns (menu_name, entry_name, row) tuples.
    """
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
    marker_prefix = "[not builder-available:"

    # Rows marked as "not builder-available" carry a composable form in `use`
    # and must build. Verify they are marked unvalidated (status != production).
    if marker_prefix in use_str:
        assert row.get("status") != "production", (
            f"{entry_name} has marker but status={row.get('status')}"
        )
        assert marker_prefix in row.get("short_doc", ""), (
            f"{entry_name} marker in use but not in short_doc"
        )
        # These rows should build; fall through to verify.

    # Require the use string to start with SEDModel.build
    assert use_str.startswith("SEDModel.build(..., "), (
        f"{entry_name} use string does not start with SEDModel.build(..., : {use_str[:60]}"
    )

    # Extract everything between SEDModel.build(..., and the closing )
    start_idx = use_str.find("SEDModel.build(..., ")
    if start_idx == -1:
        pytest.fail(f"{entry_name}: could not parse use string: {use_str}")

    remainder = use_str[start_idx + len("SEDModel.build(..., ") :]
    remainder = remainder.rstrip(" )")  # strip trailing spaces and closing parens

    # Parse the kwargs
    namespace = {
        "SEDModel": SEDModel,
        "Fixed": Fixed,
        "DEFAULT": DEFAULT,
        "FREE": tengri.FREE,
        "Uniform": tengri.Uniform,
    }

    try:
        build_kwargs = eval(f"dict({remainder})", namespace)
    except Exception as e:
        pytest.fail(f"{menu_name}/{entry_name}: could not parse use string: {use_str}\nError: {e}")

    # Add base dust emission for radio models (FIRRC requires it)
    base_kwargs = dict(build_kwargs)
    if menu_name == "radio_models" and "radio" in base_kwargs:
        base_kwargs["dust_emission"] = {
            "type": "dale2014",
            "all_params": Fixed(DEFAULT),
        }

    # Try to build. Some rows need external data and will raise FileNotFoundError.
    # For those, use pytest.raises to catch the expected error.
    external_data_rows = {
        ("nebular_backends", "cloudy"),
        ("agn_blocks", "synthesizer"),
        ("agn_blocks", "synthesizer_spectra"),
    }

    if (menu_name, entry_name) in external_data_rows:
        # These require external grid files
        with pytest.raises((FileNotFoundError, TengriIOError)):
            SEDModel.build(
                ssp_data=bare_stellar_ssp,
                observation=observation,
                redshift=Fixed(0.1),
                **base_kwargs,
            )
    else:
        # All other production rows must build successfully
        try:
            SEDModel.build(
                ssp_data=bare_stellar_ssp,
                observation=observation,
                redshift=Fixed(0.1),
                **base_kwargs,
            )
        except Exception as exc:
            pytest.fail(
                f"{menu_name}/{entry_name}: use string failed to build: {use_str}\nError: {exc}"
            )
