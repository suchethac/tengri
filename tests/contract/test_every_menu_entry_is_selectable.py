# SPDX-License-Identifier: BSD-3-Clause
"""Every name a ``list_*`` menu offers must build, or refuse for a stated reason.

The menus are the public answer to "what can I choose?" -- ``list_dust_laws``,
``list_sfh_models``, ``list_xray_models`` and the rest, which the docs tell
readers to prefer over any hand-maintained list. A name can therefore be
registered, appear in the menu, be copied into a config, and only then turn out
not to reach the forward model. Nothing else checks that: the physics behind a
block is tested directly by its own module's tests, which pass whether or not
the *selector* is wired up.

This sweeps the whole surface. Each entry gets one minimal ``SEDModel.build``
through the public grammar, and must do one of two things:

* build, or
* raise ``ValueError`` with a reason from :data:`ALLOWED_REFUSALS`.

Both outcomes are correct. What is not correct is a name that raises something
else, or raises with no explanation -- that is a menu advertising a choice the
grammar will not honor.

Measured when this was written, across eleven menus plus the AGN blocks: 158
entries, of which 144 build. All ten refusals are deliberate -- ``burst`` and
``field`` are additive-only and need a smooth component beside them, and the
other eight are held behind an explicit "registered but not yet validated
against the DSPS forward path" guard, which is the right way to ship a model
that is not ready: loudly, rather than silently returning something wrong. The
remaining four are the Synthesizer AGN NLR/BLR blocks, which build once
``TENGRI_SYNTHESIZER_AGN_GRID_DIR`` points at the downloaded grids and skip
otherwise. Nothing in the surface is broken.

Guarding the refusal *reasons*, not just the count, is what makes this useful:
when one of those eight is validated and starts building, this test keeps
passing; when a wired model quietly regresses into refusing, it fails.

A third outcome is a skip: blocks backed by a grid that is downloaded rather
than committed (Synthesizer's AGN NLR/BLR tables) cannot be built without it,
which is missing data rather than a defect. Those skips name the missing path.
"""

from __future__ import annotations

import re

import pytest

import tengri
from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
from tengri.config.exceptions import TengriIOError

pytestmark = pytest.mark.contract

# Substrings that mark a refusal as designed rather than broken.
ALLOWED_REFUSALS = (
    "not yet validated against the DSPS forward path",
    "At least one additive (smooth) SFH component required",
)

#: Wordings that mark a ``ValueError`` as *absent data* rather than a refusal.
#: The CLOUDY nebular backend reports a missing grid this way instead of with
#: ``FileNotFoundError``, so the branch below cannot catch it by type -- it
#: builds fine on a machine that has the grid and fails in CI, which ships none.
#:
#: Both substrings are required, and the second is the point: "Searched <dirs>"
#: is the backend reporting where it looked and came up empty. Matching on
#: "needs a grid file" alone would also swallow a build that *has* the grid and
#: rejects it for some other reason, which is a real defect and must still fail.
MISSING_DATA_MARKERS = ("needs a grid file", "Searched ")


def _names(menu) -> list[str]:
    if isinstance(menu, dict):
        return sorted(menu)
    return sorted({(m.get("name") if isinstance(m, dict) else str(m)) for m in menu})


def _radio_model_to_composable(n: str) -> dict:
    """Convert legacy radio model name to composable form."""
    from tengri.parameters.groups import _legacy_radio_type_to_blocks

    if n == "none":
        return {"radio": {"sf": {"type": "none"}, "agn": {"type": "none"}, "*": FIXED}}
    elif n == "condon92":
        return {
            "radio": {
                "sf": {"type": "bell2003"},
                "agn": {"type": "powerlaw"},
                "*": FIXED,
            }
        }
    else:
        sf_variant, agn_variant = _legacy_radio_type_to_blocks(n)
        return {
            "radio": {
                "sf": {"type": sf_variant},
                "agn": {"type": agn_variant},
                "*": FIXED,
            }
        }


def _cases() -> list[tuple[str, str, dict]]:
    """``(menu, entry, build kwargs)`` for every entry in every live menu."""
    specs = [
        (
            "dust law",
            tengri.list_dust_laws,
            lambda n: {
                "dust_attenuation": {"type": "two_component", "law": n, "all_params": FIXED}
            },
        ),
        (
            "dust model",
            tengri.list_dust_models,
            lambda n: {
                "dust_attenuation": (
                    {"type": n, "law": "calzetti", "all_params": FIXED}
                    if n in ("two_component", "single_component")
                    else {"type": n, "all_params": FIXED}
                )
            },
        ),
        (
            "dust emission",
            tengri.list_dust_emission_models,
            lambda n: {
                "dust_attenuation": {
                    "law": "calzetti",
                    "type": "two_component",
                    "all_params": FIXED,
                },
                "dust_emission": {"type": n, "all_params": FIXED},
            },
        ),
        ("sfh model", tengri.list_sfh_models, lambda n: {"sfh": {"type": n, "all_params": FIXED}}),
        (
            "nebular backend",
            tengri.list_nebular_backends,
            lambda n: {"neb": {"type": n, "all_params": FIXED}},
        ),
        (
            "metallicity mode",
            tengri.list_metallicity_modes,
            lambda n: {"met": {"type": n, "all_params": FIXED}},
        ),
        ("igm model", tengri.list_igm_models, lambda n: {"igm": {"type": n, "all_params": FIXED}}),
        (
            "radio model",
            tengri.list_radio_models,
            lambda n: _radio_model_to_composable(n),
        ),
        (
            "xray model",
            tengri.list_xray_models,
            lambda n: {"xray": {"type": n, "all_params": FIXED}},
        ),
        (
            "shock model",
            tengri.list_shock_models,
            lambda n: {"shock": {"type": n, "all_params": FIXED}},
        ),
        (
            "age kernel",
            tengri.list_age_kernels,
            lambda n: {"sfh": {"type": "delayed", "age_kernel": n, "all_params": FIXED}},
        ),
    ]
    out = []
    for menu, lister, mk in specs:
        for name in _names(lister()):
            out.append((menu, name, mk(name)))

    # The AGN blocks are nested selectors (agn.disc, agn.torus, ...), and the
    # registry already states how to reach each one in its `use` column. Parse
    # that rather than keep a private category -> group map here: a new block
    # category then arrives covered instead of silently unswept.
    for row in tengri.list_agn_blocks():
        name = row["name"]
        if name == "none":
            continue
        m = re.search(r"agn=\{'(\w+)':", row.get("use", ""))
        if m is None:
            continue
        # Special handling for atten/smc_prevot: use law key instead of type
        axis = m.group(1)
        if axis == "atten" and name == "smc_prevot":
            sub_block_spec = {"law": "prevot_smc", "all_params": FIXED}
        else:
            sub_block_spec = {"type": name, "all_params": FIXED}
        out.append(
            (
                f"agn.{axis}",
                name,
                {"agn": {"type": "composable", axis: sub_block_spec}},
            )
        )
    return out


CASES = _cases()


@pytest.fixture(scope="module")
def bare_stellar_ssp():
    """A *bare-stellar* SSP, which the photoionization backends require.

    ``cue`` and ``cloudy`` refuse a wNE grid by design -- ``CueWNESSPError``,
    a dedicated exception -- because nebular emission is already baked into
    those templates and running a photoionization model on top double-counts
    it. Handing them the grid they need tests that they build, which is the
    stronger claim; accepting the refusal would only have tested the guard.
    """
    from pathlib import Path

    import tengri

    if not (
        Path(__file__).resolve().parents[2] / "data" / "fsps_prsc_miles_chabrier.h5"
    ).is_file():
        pytest.skip("bare-stellar SSP grid not available")
    return tengri.load_ssp("fsps_prsc_miles_chabrier")


@pytest.fixture(scope="module")
def observation() -> Observation:
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))


@pytest.mark.contract
@pytest.mark.parametrize(
    ("menu", "entry", "kwargs"),
    CASES,
    ids=[f"{m.replace(' ', '-')}:{n}" for m, n, _ in CASES],
)
def test_menu_entry_builds_or_refuses_clearly(menu, entry, kwargs, bare_stellar_ssp, observation):
    """A menu entry either reaches the forward model or says why it cannot."""
    try:
        SEDModel.build(
            ssp_data=bare_stellar_ssp,
            observation=observation,
            **{"redshift": Fixed(0.1), **kwargs},
        )
    except ValueError as exc:
        if all(marker in str(exc) for marker in MISSING_DATA_MARKERS):
            pytest.skip(f"{menu} '{entry}' needs an external grid that is not installed: {exc}")
        if not any(reason in str(exc) for reason in ALLOWED_REFUSALS):
            pytest.fail(
                f"{menu} '{entry}' is offered by its list_* menu but refuses to build "
                f"for an undocumented reason: {exc}\n"
                f"Either wire it up, or make the refusal explicit and add its wording "
                f"to ALLOWED_REFUSALS in this file."
            )
    except (FileNotFoundError, TengriIOError) as exc:
        # Blocks backed by a third-party grid that is downloaded rather than
        # committed -- Synthesizer's AGN NLR/BLR tables, for one. Absent data is
        # not a broken selector, so skip rather than fail, and name the path so
        # the reader can see exactly what to fetch. TengriIOError is the
        # tengri-native absent-grid signal since #1952 (was FileNotFoundError).
        pytest.skip(f"{menu} '{entry}' needs an external grid that is not installed: {exc}")
    except Exception as exc:  # the point is that nothing else is acceptable
        pytest.fail(
            f"{menu} '{entry}' raised {type(exc).__name__} rather than building or "
            f"refusing with a ValueError: {exc}"
        )


@pytest.mark.contract
def test_menus_are_not_empty():
    """A menu that silently empties would make the sweep above vacuous."""
    assert len(CASES) > 100, f"only {len(CASES)} menu entries discovered; menus may have broken"
