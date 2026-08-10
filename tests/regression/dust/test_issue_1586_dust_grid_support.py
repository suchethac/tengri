# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the #1586 grid-support check must cover more than AGN discs.

#1586 landed a real mechanism -- a ``(component, parameter)`` registry of grid
extents, checked at composition time -- but registered exactly one component,
``('agn.disc', 'slone_netzer')``, in a module scoped to AGN blocks. The same
defect exists wherever a template-backed component clips a parameter onto a
grid axis, and the dust IR libraries do it in ~30 places.

Measured on the shipped grids before this test was written:

===================  ================  ===================  ==============
component            parameter         grid extent          declared
===================  ================  ===================  ==============
astrodust            ``dust_lgU``      ``[-3, 6]``          ``U(0, 7)``
dl07                 ``dust_umin``     ``[0.1, 20]``        no upper bound
dl14                 ``dust_umin``     ``[0.1, 50]``        no upper bound
themis               ``dust_umin``     ``[0.1, 80]``        no upper bound
schreiber2018        ``dust_T``        ``[14.24, 60.21]``   no upper bound
===================  ================  ===================  ==============

``dust_lgU`` is the one that bites without any user action: its own declared
``free_prior`` overhangs, so ``all_params: FREE`` hands a sampler a range whose
top 14.3% is bit-identical with an exactly-zero gradient. The three ``umin``
rows show why the constraint cannot live on the declaration -- one parameter,
three different caps depending on which backend is selected.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.components import grid_support as grid_support_mod
from tengri.components.grid_support import (
    GRID_SUPPORT,
    check_grid_support,
    describe_clipping,
    grid_support,
)
from tengri.config.exceptions import AdvisoryWarning, GridSupportWarning
from tengri.parameters import FREE, parse_groups
from tengri.parameters.priors import Uniform

pytestmark = pytest.mark.regression_bug


def _grid_warnings(**group_kwargs) -> list[str]:
    """Build a spec and return only the grid-overhang warning messages."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_groups(sfh={"type": "dpl"}, redshift=0.1, **group_kwargs)
    return [str(w.message) for w in caught if issubclass(w.category, GridSupportWarning)]


# --------------------------------------------------------------------------
# The census -- the thing that was actually wrong.
# --------------------------------------------------------------------------


def test_the_registry_covers_more_than_one_component():
    """#1586's mechanism shipped with a one-row census; that was the defect."""
    selectors = {selector for selector, _ in GRID_SUPPORT}
    assert "agn.disc" in selectors, "the original AGN entry must survive"
    assert "dust.emission" in selectors, (
        "dust IR libraries clip parameters onto grid axes exactly as the SN12 "
        "disc does, so they belong in the same registry"
    )
    assert len(GRID_SUPPORT) > 1


#: Menu entries that interpolate no template grid, so nothing can be clipped.
_GRID_FREE = frozenset(
    {
        "casey2012",
        "energy_balance_split",
        "mbb",
        "modified_blackbody",
        "pah_drude",
        # Only grid axis is L_TIR, derived from L_absorbed rather than set by
        # the user, so no prior can overhang it.
        "dh02_ce01",
        # Template-backed, but its grid (pahspec_draine2021.h5) is not shipped,
        # so the component raises before it could ever clip silently. Its HDF5
        # axis key is therefore unverifiable here; registering a guessed key
        # would be an unchecked assertion.
        "draine2021_pah",
    }
)


def test_every_selectable_variant_is_registered_or_declared_grid_free():
    """The census guard: this is how the one-row registry happened.

    #1586 shipped a correct mechanism whose table listed a single component.
    A registry is only as wide as its census, so enumerate the *menu* and
    require each entry to be either registered or explicitly declared
    grid-free. Adding a template-backed variant then fails here until it is
    registered, instead of silently going unchecked.
    """
    from tengri.builders.dust import emission as builder

    registered = {name for selector, name in GRID_SUPPORT if selector == "dust.emission"}
    unaccounted = sorted(set(builder._FACTORIES) - registered - _GRID_FREE)
    assert unaccounted == [], (
        f"dust emission variants neither registered in GRID_SUPPORT nor listed "
        f"as grid-free: {unaccounted}"
    )


@pytest.mark.parametrize("alias", ["dl07_tabulated", "draine_li2007", "draine_li2014"])
def test_menu_aliases_resolve_to_the_same_support(alias):
    """An alias must be checked as thoroughly as the canonical spelling."""
    support = grid_support("dust.emission", alias)
    if not support:
        pytest.skip(f"{alias} grid not installed")
    assert "dust_umin" in support


@pytest.mark.parametrize(
    ("name", "param", "expected"),
    [
        ("astrodust", "dust_lgU", (-3.0, 6.0)),
        ("dl07", "dust_umin", (0.1, 20.0)),
        ("dl14", "dust_umin", (0.1, 50.0)),
        ("themis", "dust_umin", (0.1, 80.0)),
    ],
)
def test_grid_extents_are_read_from_the_shipped_grids(name, param, expected):
    """Extents come from the data files, so a rebuilt grid cannot go stale."""
    support = grid_support("dust.emission", name)
    if not support:
        pytest.skip(f"{name} grid not installed")
    assert support[param] == pytest.approx(expected, rel=1e-6)


def test_one_parameter_has_three_different_caps():
    """Why the constraint cannot live on the declaration.

    ``dust_umin`` is capped at 20, 50 or 80 depending only on which emission
    backend is selected. No single ``bound_check`` can express that, which is
    precisely the argument :mod:`tengri.components.grid_support` makes.
    """
    caps = {}
    for backend in ("dl07", "dl14", "themis"):
        support = grid_support("dust.emission", backend)
        if support:
            caps[backend] = support["dust_umin"][1]
    if len(caps) < 3:
        pytest.skip("not all dust emission grids installed")
    assert len(set(caps.values())) == 3, caps


def test_themis_qhac_extent_is_the_relabeled_axis_not_the_raw_dataset():
    """The raw file axis is not the grid the model interpolates on.

    ``create_themis_from_grid`` relabels an FSPS-scaled qhac axis
    (``[0.909, 18.18]``) to the CIGALE convention (``[0.02, 0.40]``). An
    accessor reading the raw dataset would report a support the model never
    sees -- and would then call the perfectly-fine 0.17 default off-grid.
    """
    support = grid_support("dust.emission", "themis")
    if not support:
        pytest.skip("themis grid not installed")
    lo, hi = support["dust_qhac"]
    assert (lo, hi) == pytest.approx((0.02, 0.40), rel=1e-6)
    # The declared default must sit inside it -- this is the assertion that
    # would have caught a raw-axis accessor.
    assert lo < 0.17 < hi


# --------------------------------------------------------------------------
# The live bug: a declared free_prior that overhangs its own grid.
# --------------------------------------------------------------------------


def test_astrodust_lgu_free_prior_overhangs_its_grid():
    """``all_params: FREE`` alone is enough to trigger it -- no user prior."""
    if not grid_support("dust.emission", "astrodust"):
        pytest.skip("astrodust grid not installed")
    messages = _grid_warnings(
        dust={"type": "two_component", "emission": {"type": "astrodust", "all_params": FREE}}
    )
    assert len(messages) == 1, messages
    message = messages[0]
    assert "dust_lgU" in message
    assert "[-3, 6]" in message  # the extent to narrow to
    assert "14%" in message  # (7 - 6) / 7


def test_a_prior_inside_the_grid_is_silent():
    """The control. A guard that always fires carries no information."""
    if not grid_support("dust.emission", "dl14"):
        pytest.skip("dl14 grid not installed")
    assert (
        _grid_warnings(
            dust={
                "type": "two_component",
                "emission": {"type": "dl14", "umin": Uniform(0.1, 50.0)},
            }
        )
        == []
    )


def test_the_same_prior_warns_on_a_narrower_backend():
    """Identical prior, different backend: dl07 caps at 20 where dl14 caps at 50.

    Paired with the control above this isolates the backend as the only
    difference, so the warning cannot be an artifact of the prior itself.
    """
    if not grid_support("dust.emission", "dl07"):
        pytest.skip("dl07 grid not installed")
    messages = _grid_warnings(
        dust={
            "type": "two_component",
            "emission": {"type": "dl07", "umin": Uniform(0.1, 50.0)},
        }
    )
    assert len(messages) == 1, messages
    assert "[0.1, 20]" in messages[0]


# --------------------------------------------------------------------------
# Introspection must stay quiet.
# --------------------------------------------------------------------------


def test_importing_tengri_emits_no_advisory():
    """The builder factory discovery frees every variant at import time.

    Without the ``AdvisoryWarning`` suppression on the discovery path, the
    astrodust overhang above fires during ``import tengri`` -- before the user
    has built anything. An advisory that fires on import is noise, not signal.
    """
    import importlib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(importlib.import_module("tengri.builders.dust.emission"))
    advisories = [w for w in caught if issubclass(w.category, AdvisoryWarning)]
    assert advisories == [], [str(w.message) for w in advisories]


def test_grid_support_warning_is_an_advisory():
    """Subclassing is what keeps a new advisory quiet on the discovery path."""
    assert issubclass(GridSupportWarning, AdvisoryWarning)


# --------------------------------------------------------------------------
# Shared helpers behave the same for every component.
# --------------------------------------------------------------------------


def test_an_unbounded_prior_is_not_called_inert():
    """``live_fraction`` returns 0.0 for an unbounded support by definition.

    Reporting that as "100% outside" would be a self-contradiction: most of the
    mass may sit on the grid and only the tails clip.
    """
    detail = describe_clipping((float("-inf"), float("inf")), (0.0, 1.0))
    assert detail is not None
    assert "unbounded" in detail
    assert "%" not in detail


def test_check_grid_support_skips_unregistered_components():
    """A component with no template grid is unconstrained, not suspect."""
    assert check_grid_support([("dust.emission", "casey2012")], {"dust_T": (1.0, 1e6)}) == []


def test_registering_a_component_is_all_it_takes(monkeypatch):
    """The extension point: one registry entry, and the check covers it."""
    monkeypatch.setitem(
        grid_support_mod.GRID_SUPPORT,
        ("dust.emission", "casey2012"),
        lambda: {"dust_T": (20.0, 60.0)},
    )
    findings = check_grid_support([("dust.emission", "casey2012")], {"dust_T": (20.0, 100.0)})
    assert len(findings) == 1
    _, _, pname, detail, extent = findings[0]
    assert pname == "dust_T"
    assert extent == (20.0, 60.0)
    assert "50%" in detail
