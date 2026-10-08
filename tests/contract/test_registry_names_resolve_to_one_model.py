# SPDX-License-Identifier: BSD-3-Clause
"""One published name resolves to one model, on every path (#2765, #2425 class).

A name can be reached two ways: the runtime registry dict (``DUST_EMISSION_MODELS``)
and the build grammar's alias table (``_EMISSION_TYPE_ALIASES``, which maps a
grammar name to a component). When both hold a name, they must evaluate the same
physics. The #2765 defect was ``draine2021_pah``: the dict held the Smith+2007 Drude
closure while the build path built the tabulated Draine+2021 component.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.dust.emission.emission import (
    DUST_EMISSION_MODELS,
    draine_li2007,
    draine_li2014,
)
from tengri.components.dust.emission_templates import _DUST_EMISSION_ALIASES
from tengri.forward.component_factory import _EMISSION_TYPE_ALIASES

pytestmark = pytest.mark.contract

_WAVE_AA = np.geomspace(2.0e3, 1.0e6, 256)
_SHARED_NAMES = sorted(set(DUST_EMISSION_MODELS) & set(_EMISSION_TYPE_ALIASES))


def test_shared_names_exist():
    """The check below is only meaningful if some names are shared."""
    assert _SHARED_NAMES, "no name is in both the registry and the alias table"


@pytest.mark.parametrize("name", _SHARED_NAMES)
def test_shared_name_is_the_alias_target_callable(name):
    """A shared name must be the very callable its alias target is registered as."""
    target = _EMISSION_TYPE_ALIASES[name]
    assert target in DUST_EMISSION_MODELS, (
        f"{name!r} is in the registry and aliases to {target!r}, which is not "
        "registered: the two resolutions cannot be the same model"
    )
    assert DUST_EMISSION_MODELS[name] is DUST_EMISSION_MODELS[target], (
        f"{name!r} and its alias target {target!r} are different callables"
    )


def test_draine2021_pah_is_not_a_registry_closure():
    """The tabulated Draine+2021 name must not carry the Smith+2007 Drude closure."""
    assert "draine2021_pah" not in DUST_EMISSION_MODELS
    assert _EMISSION_TYPE_ALIASES["draine2021_pah"] == "draine2021_pah_ir"


@pytest.mark.parametrize(("name", "target"), sorted(_DUST_EMISSION_ALIASES.items()))
def test_emission_alias_pairs_evaluate_the_same_model(name, target):
    """Each alias in the emission-templates table evaluates the model it names.

    The DL07 and DL14 loaders swap their registry slots on first call, so both
    slots are resolved before the comparison.
    """
    draine_li2007(_WAVE_AA, 1.0)
    draine_li2014(_WAVE_AA, 1.0)
    assert name in DUST_EMISSION_MODELS and target in DUST_EMISSION_MODELS
    ref = np.asarray(DUST_EMISSION_MODELS[target](_WAVE_AA, 1.0))
    out = np.asarray(DUST_EMISSION_MODELS[name](_WAVE_AA, 1.0))
    assert np.all(np.isfinite(ref)) and np.any(ref > 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=0.0)


def test_pah_drude_stays_reachable_under_its_own_name():
    from tengri.components.dust.emission.analytic._closures import pah_drude

    assert DUST_EMISSION_MODELS["pah_drude"] is pah_drude


def test_dl07_slots_agree_after_first_call():
    """draine_li2007 swaps its registry slots on first call; every name must still agree.

    Any output difference after the first resolution would mean one name serves a
    stale or different template set.
    """
    first = np.asarray(DUST_EMISSION_MODELS["dl07"](_WAVE_AA, 1.0))
    via_function = np.asarray(draine_li2007(_WAVE_AA, 1.0))
    via_tabulated_slot = np.asarray(DUST_EMISSION_MODELS["dl07_tabulated"](_WAVE_AA, 1.0))
    via_canonical_slot = np.asarray(DUST_EMISSION_MODELS["draine_li2007"](_WAVE_AA, 1.0))
    assert np.all(np.isfinite(first)) and np.any(first > 0.0)
    for other in (via_function, via_tabulated_slot, via_canonical_slot):
        np.testing.assert_allclose(other, first, rtol=1e-6, atol=0.0)


def test_draine2021_native_grid_is_the_tabulated_template_grid():
    """The draine2021_pah master grid is the template grid, not the analytic placeholder."""
    from tengri.forward import wavelength_extension as wx

    grid = wx.native_wave_dust_emission("draine2021_pah")
    template = wx._read_wavelength("pahspec_draine2021.h5", "wavelength_um", 1.0e4)
    assert grid is not None and template is not None
    np.testing.assert_allclose(np.asarray(grid), np.asarray(template), rtol=1e-12)
    assert not np.array_equal(np.asarray(grid), np.asarray(wx._ANALYTIC_DUST_WAVE_AA))
