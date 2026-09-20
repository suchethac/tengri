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


def test_menu_use_strings_have_no_trailing_whitespace():
    """Menu use/short_doc/citation strings must be stripped and contain no RST literal markers."""
    menus_to_check = [
        ("age_kernels", tengri.list_age_kernels()),
        ("agn_blocks", tengri.list_agn_blocks()),
        ("agn_models", tengri.list_agn_models()),
        ("components", tengri.list_components()),
        ("dust_emission_models", tengri.list_dust_emission_models()),
        ("dust_laws", tengri.list_dust_laws()),
        ("dust_models", tengri.list_dust_models()),
        ("filters", tengri.list_filters()),
        ("igm_models", tengri.list_igm_models()),
        ("inference_methods", tengri.list_inference_methods()),
        ("instruments", tengri.list_instruments()),
        ("known_ssps", tengri.list_known_ssps()),
        ("metallicity_modes", tengri.list_metallicity_modes()),
        ("nebular_backends", tengri.list_nebular_backends()),
        ("plots", tengri.list_plots()),
        ("radio_blocks", tengri.list_radio_blocks()),
        ("radio_models", tengri.list_radio_models()),
        ("recipes", tengri.list_recipes()),
        ("shock_models", tengri.list_shock_models()),
        ("sfh_models", tengri.list_sfh_models()),
        ("xray_models", tengri.list_xray_models()),
    ]

    failures = []
    for menu_name, rows in menus_to_check:
        for row in rows:
            entry_name = row.get("name", "")
            for field_name in ["use", "short_doc", "citation"]:
                field_value = row.get(field_name, "")
                if not field_value:
                    continue

                # Check for trailing whitespace
                if field_value != field_value.rstrip():
                    failures.append(
                        f"{menu_name}/{entry_name}/{field_name}: "
                        f"has trailing whitespace: {field_value[-20:]!r}"
                    )

                # Check for inline literal markers ("``") in use and short_doc
                if field_name in ["use", "short_doc"] and "``" in field_value:
                    failures.append(
                        f"{menu_name}/{entry_name}/{field_name}: "
                        f"contains inline literal markers (``): {field_value[:80]}"
                    )

    if failures:
        pytest.fail("\n".join(failures))


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
    short_doc = row.get("short_doc", "")
    is_marked = marker_prefix in short_doc

    # Rows marked as "not builder-available" in short_doc carry a composable form in `use`
    # and must not be production status.
    if is_marked:
        assert row.get("status") != "production", (
            f"{entry_name} has marker in short_doc but status={row.get('status')}"
        )
        # Marked rows can have either a SEDModel.build(..., ) string OR a dotted path
        if use_str.startswith("SEDModel.build(..., "):
            # Will be validated below as a build form
            pass
        elif "." in use_str and not use_str.startswith("SEDModel.build"):
            # Dotted path form: validate it can be imported
            # Extract the function path (everything before the first parenthesis)
            func_path = use_str.split("(")[0].strip()
            module_path, func_name = func_path.rsplit(".", 1)
            try:
                import importlib

                mod = importlib.import_module(module_path)
                assert hasattr(mod, func_name), (
                    f"{entry_name}: function {func_name} not found in {module_path}"
                )
            except (ImportError, ValueError) as e:
                pytest.fail(
                    f"{entry_name}: marked row with dotted path failed to import:"
                    f" {use_str}\nError: {e}"
                )
            # Dotted path forms are considered valid for marked rows; skip build verification
            return
        else:
            pytest.fail(f"{entry_name}: marked row has invalid use format: {use_str}")

    # Production rows must require the use string to start with SEDModel.build
    if not is_marked:
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
