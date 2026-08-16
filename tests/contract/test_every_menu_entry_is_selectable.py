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

Measured when this was written: 104 of 114 entries build. All ten refusals are
deliberate. ``burst`` and ``field`` are additive-only and need a smooth
component beside them; the other eight are held behind an explicit
"registered but not yet validated against the DSPS forward path" guard, which
is the right way to ship a model that is not ready -- loudly, rather than
silently returning something wrong.

Guarding the refusal *reasons*, not just the count, is what makes this useful:
when one of those eight is validated and starts building, this test keeps
passing; when a wired model quietly regresses into refusing, it fails.
"""

from __future__ import annotations

import pytest

import tengri
from tengri import FIXED, Observation, Photometry, SEDModel

pytestmark = pytest.mark.contract

# Substrings that mark a refusal as designed rather than broken.
ALLOWED_REFUSALS = (
    "not yet validated against the DSPS forward path",
    "At least one additive (smooth) SFH component required",
)


def _names(menu) -> list[str]:
    if isinstance(menu, dict):
        return sorted(menu)
    return sorted({(m.get("name") if isinstance(m, dict) else str(m)) for m in menu})


def _cases() -> list[tuple[str, str, dict]]:
    """``(menu, entry, build kwargs)`` for every entry in every live menu."""
    specs = [
        (
            "dust law",
            tengri.list_dust_laws,
            lambda n: {"dust": {"type": "two_component", "law_diff": n, "all_params": FIXED}},
        ),
        (
            "dust model",
            tengri.list_dust_models,
            lambda n: {"dust": {"type": n, "all_params": FIXED}},
        ),
        (
            "dust emission",
            tengri.list_dust_emission_models,
            lambda n: {
                "dust": {
                    "type": "two_component",
                    "all_params": FIXED,
                    "emission": {"type": n, "all_params": FIXED},
                }
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
            lambda n: {"radio": {"type": n, "all_params": FIXED}},
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
        SEDModel.build(ssp_data=bare_stellar_ssp, observation=observation, **kwargs)
    except ValueError as exc:
        if not any(reason in str(exc) for reason in ALLOWED_REFUSALS):
            pytest.fail(
                f"{menu} '{entry}' is offered by its list_* menu but refuses to build "
                f"for an undocumented reason: {exc}\n"
                f"Either wire it up, or make the refusal explicit and add its wording "
                f"to ALLOWED_REFUSALS in this file."
            )
    except Exception as exc:  # the point is that nothing else is acceptable
        pytest.fail(
            f"{menu} '{entry}' raised {type(exc).__name__} rather than building or "
            f"refusing with a ValueError: {exc}"
        )


@pytest.mark.contract
def test_menus_are_not_empty():
    """A menu that silently empties would make the sweep above vacuous."""
    assert len(CASES) > 100, f"only {len(CASES)} menu entries discovered; menus may have broken"
