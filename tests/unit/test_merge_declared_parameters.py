# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :func:`tengri.forward.orchestrator.merge_declared_parameters`.

Validates that the helper correctly flattens per-component
:meth:`SEDComponent.declared_parameters` lists into a single prior
dict, and rejects every contract violation it is supposed to catch:

- Wrong prefix on a declared name.
- Two components declaring the same parameter name.
- A component returning a non-:class:`ParamDeclaration` entry.

Closes the loop on the seam: each adapter declares parameters; the
orchestrator merges them; the resulting dict is suitable for spreading
into :class:`tengri.Parameters` once Phase II-6 lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tengri.components.dust.component import DustAttenuationSEDComponent
from tengri.components.igm.component import IGMSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.xray.component import XRaySEDComponent
from tengri.protocols import (
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.forward.orchestrator import merge_declared_parameters
from tengri.parameters.priors import Fixed


@dataclass(frozen=True)
class _BadPrefixComponent:
    """Adapter that declares a parameter without the right prefix."""

    name: str = "bad_prefix"
    parameter_prefix: str = "good_"
    config: SEDComponentConfig = field(default_factory=SEDComponentConfig)

    def declared_parameters(self) -> list[ParamDeclaration]:
        return [ParamDeclaration("wrong_name", Fixed(0.0), "")]

    def precompute(self, ssp_data=None, wave_grid=None) -> SEDComponentState:
        return SEDComponentState()

    def apply(self, state: PipelineState, params) -> PipelineState:
        return state


@dataclass(frozen=True)
class _DuplicateRadio:
    """Adapter that re-declares ``radio_q_ir`` (collides with RadioSEDComponent)."""

    name: str = "duplicate_radio"
    parameter_prefix: str = "radio_"
    config: SEDComponentConfig = field(default_factory=SEDComponentConfig)

    def declared_parameters(self) -> list[ParamDeclaration]:
        return [ParamDeclaration("radio_q_ir", Fixed(0.0), "")]

    def precompute(self, ssp_data=None, wave_grid=None) -> SEDComponentState:
        return SEDComponentState()

    def apply(self, state: PipelineState, params) -> PipelineState:
        return state


@dataclass(frozen=True)
class _NonParamDeclEntry:
    """Adapter that returns a tuple instead of ParamDeclaration."""

    name: str = "non_param_decl"
    parameter_prefix: str = "x_"
    config: SEDComponentConfig = field(default_factory=SEDComponentConfig)

    def declared_parameters(self) -> list[Any]:
        return [("x_oops", Fixed(0.0), "")]  # tuple, not ParamDeclaration

    def precompute(self, ssp_data=None, wave_grid=None) -> SEDComponentState:
        return SEDComponentState()

    def apply(self, state: PipelineState, params) -> PipelineState:
        return state


# ─────────────────────────────────────────────────────────────────────
# Happy-path tests
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_single_component_round_trip():
    """One adapter → all its declared params appear in the output."""
    radio = RadioSEDComponent()
    merged = merge_declared_parameters([radio])

    expected_names = {decl.name for decl in radio.declared_parameters()}
    assert set(merged.keys()) == expected_names


@pytest.mark.unit
def test_four_adapters_merge_disjoint_namespaces():
    """The full first cohort merges into a single dict with no collisions."""
    adapters = [
        RadioSEDComponent(),
        IGMSEDComponent(),
        XRaySEDComponent(),
        DustAttenuationSEDComponent(),
    ]
    merged = merge_declared_parameters(adapters)

    # The union of every adapter's declared names equals the merged keys.
    expected = set()
    for a in adapters:
        for decl in a.declared_parameters():
            expected.add(decl.name)
    assert set(merged.keys()) == expected


@pytest.mark.unit
def test_priors_are_passed_through_unchanged():
    """``merged[name]`` must equal the Distribution the adapter declared.

    Identity is too strict — :meth:`declared_parameters` returns fresh
    Distribution instances on each call, so we check structural equality
    via ``==`` (Distribution dataclasses implement value-based equality).
    """
    radio = RadioSEDComponent()
    merged = merge_declared_parameters([radio])
    for decl in radio.declared_parameters():
        assert merged[decl.name] == decl.prior


@pytest.mark.unit
def test_iteration_order_follows_input_order():
    """Component order is preserved (radio params come before xray params)."""
    radio = RadioSEDComponent()
    xray = XRaySEDComponent()
    merged = merge_declared_parameters([radio, xray])

    keys = list(merged.keys())
    radio_idx = max(i for i, k in enumerate(keys) if k.startswith("radio_"))
    xray_idx = min(i for i, k in enumerate(keys) if k.startswith("xray_"))
    assert radio_idx < xray_idx


# ─────────────────────────────────────────────────────────────────────
# Negative-path tests
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_wrong_prefix_raises():
    """A declared name that doesn't start with parameter_prefix is a hard error."""
    with pytest.raises(ValueError, match="violates the prefix rule"):
        merge_declared_parameters([_BadPrefixComponent()])


@pytest.mark.unit
def test_duplicate_name_raises_with_useful_message():
    """Two components claiming the same name -> ValueError naming both."""
    with pytest.raises(ValueError) as excinfo:
        merge_declared_parameters([RadioSEDComponent(), _DuplicateRadio()])
    msg = str(excinfo.value)
    assert "radio_q_ir" in msg
    assert "radio" in msg
    assert "duplicate_radio" in msg


@pytest.mark.unit
def test_non_param_declaration_entry_raises_typeerror():
    """A component must return ParamDeclaration instances, not tuples."""
    with pytest.raises(TypeError, match="non-ParamDeclaration"):
        merge_declared_parameters([_NonParamDeclEntry()])


@pytest.mark.unit
def test_empty_component_list_returns_empty_dict():
    assert merge_declared_parameters([]) == {}


@pytest.mark.unit
def test_redshift_in_bare_allowlist_is_accepted():
    """A bare 'redshift' declaration is allowed even with non-empty prefix."""

    @dataclass(frozen=True)
    class _RedshiftDeclarer:
        name: str = "redshift_owner"
        parameter_prefix: str = "obs_"
        config: SEDComponentConfig = field(default_factory=SEDComponentConfig)

        def declared_parameters(self):
            return [ParamDeclaration("redshift", Fixed(0.0), "object redshift")]

        def precompute(self, ssp_data=None, wave_grid=None):
            return SEDComponentState()

        def apply(self, state, params):
            return state

    merged = merge_declared_parameters([_RedshiftDeclarer()])
    assert "redshift" in merged
