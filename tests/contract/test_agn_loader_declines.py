# SPDX-License-Identifier: BSD-3-Clause
"""AGN template-loader declines are recorded, not silently skipped (#3505 X1)."""

from __future__ import annotations

import pytest

from tengri.components.agn.blocks._protocol import (
    AGN_BLOCK_TEMPLATE_LOADERS,
    BlockTemplates,
    collect_block_templates,
)
from tests.contract._band_table_cases import build_composable_agn
from tests.contract._engagement import assert_declined

pytestmark = pytest.mark.contract

#: A block with no registered loader, given one that fails for the test only.
CATEGORY, NAME = "torus", "simple"
KEY = f"{CATEGORY}/{NAME}"
MISSING_MESSAGE = "AGN test template not found: /nowhere/simple.h5"


def _missing_loader():
    raise FileNotFoundError(MISSING_MESSAGE)


def _broken_loader():
    raise RuntimeError("corrupt grid")


def test_missing_file_is_returned_as_a_decline_with_its_reason(monkeypatch):
    monkeypatch.setitem(AGN_BLOCK_TEMPLATE_LOADERS, (CATEGORY, NAME), _missing_loader)
    result = collect_block_templates({CATEGORY: NAME})
    assert isinstance(result, BlockTemplates)
    assert KEY not in result.templates
    assert len(result.declines) == 1
    key, reason = result.declines[0]
    assert key == f"template_loader:{KEY}"
    assert KEY in reason and MISSING_MESSAGE in reason


def test_unreadable_file_oserror_is_declined(monkeypatch):
    def unreadable():
        raise OSError("file signature not found")

    monkeypatch.setitem(AGN_BLOCK_TEMPLATE_LOADERS, (CATEGORY, NAME), unreadable)
    result = collect_block_templates({CATEGORY: NAME})
    assert [key for key, _ in result.declines] == [f"template_loader:{KEY}"]
    assert "file signature not found" in result.declines[0][1]


def test_a_loader_that_loads_declines_nothing(monkeypatch):
    monkeypatch.setitem(AGN_BLOCK_TEMPLATE_LOADERS, (CATEGORY, NAME), lambda: {"grid": 1})
    result = collect_block_templates({CATEGORY: NAME})
    assert result.templates == {KEY: {"grid": 1}}
    assert result.declines == ()


def test_a_non_file_error_propagates(monkeypatch):
    monkeypatch.setitem(AGN_BLOCK_TEMPLATE_LOADERS, (CATEGORY, NAME), _broken_loader)
    with pytest.raises(RuntimeError, match="corrupt grid"):
        collect_block_templates({CATEGORY: NAME})


def test_no_loader_means_no_templates_and_no_declines():
    result = collect_block_templates({CATEGORY: "not_a_block"})
    assert result.templates == {}
    assert result.declines == ()


def test_model_records_the_decline_at_the_threading_call_site(monkeypatch):
    monkeypatch.setitem(AGN_BLOCK_TEMPLATE_LOADERS, (CATEGORY, NAME), _missing_loader)
    model = build_composable_agn()
    model.threaded_template_inputs()
    decline = model.band_table_decline("agn", f"template_loader:{KEY}")
    assert decline is not None
    assert MISSING_MESSAGE in decline.reason
    assert_declined(model, f"agn.template_loader:{KEY}", MISSING_MESSAGE)
