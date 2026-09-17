# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the check_literal_param_defaults guard.

This module tests the guard's ability to detect literal-copy sites,
especially edge cases like negative defaults that parse as UnaryOp.
"""

from __future__ import annotations

import ast

import pytest


@pytest.mark.unit
def test_negative_literal_defaults_are_detected() -> None:
    """Negative literals like -3.0 parse as UnaryOp(USub, Constant) and must be caught.

    Before this fix, the guard's isinstance(default, ast.Constant) check
    could not see negative defaults because ast.parse renders them as
    UnaryOp(USub, Constant(3.0)), not ast.Constant(-3.0). This test pins
    that the guard now treats UnaryOp(USub, Constant) as a literal.
    """
    from tools.check_literal_param_defaults import _signature_default_sites

    code = """
def foo(x: float = -3.0) -> None:
    pass
"""
    tree = ast.parse(code)
    declared = {"x"}

    sites = list(_signature_default_sites(tree, declared, prefix=None))
    assert len(sites) == 1, f"Expected 1 site, got {len(sites)}: {sites}"

    source_name, matched, _lineno, func, is_literal, repr_val = sites[0]
    assert source_name == "x"
    assert matched == "x"
    assert func == "foo"
    assert is_literal is True, f"Negative literal was not detected as literal: {repr_val}"
    assert repr_val == "-3.0"


@pytest.mark.unit
def test_positive_literal_defaults_still_detected() -> None:
    """Positive literals like 30.0 continue to be detected."""
    from tools.check_literal_param_defaults import _signature_default_sites

    code = """
def foo(x: float = 30.0) -> None:
    pass
"""
    tree = ast.parse(code)
    declared = {"x"}

    sites = list(_signature_default_sites(tree, declared, prefix=None))
    assert len(sites) == 1
    _, _, _, _, is_literal, repr_val = sites[0]
    assert is_literal is True
    assert repr_val == "30.0"


@pytest.mark.unit
def test_unary_plus_negative_literals() -> None:
    """Unary plus (UAdd) wrapping a literal should also be detected."""
    from tools.check_literal_param_defaults import _signature_default_sites

    code = """
def foo(x: float = +3.0) -> None:
    pass
"""
    tree = ast.parse(code)
    declared = {"x"}

    sites = list(_signature_default_sites(tree, declared, prefix=None))
    assert len(sites) == 1
    _, _, _, _, is_literal, repr_val = sites[0]
    assert is_literal is True, f"Unary plus literal was not detected: {repr_val}"
    assert repr_val == "+3.0"


@pytest.mark.unit
def test_get_fallback_negative_literals_detected() -> None:
    """Negative literals in .get(name, value) fallbacks must be detected."""
    from tools.check_literal_param_defaults import _get_fallback_sites

    code = """
p = {"key": None}
val = p.get("x", -2.0)
"""
    tree = ast.parse(code)
    declared = {"x"}

    sites = list(_get_fallback_sites(tree, declared, prefix=None))
    assert len(sites) == 1
    source_name, _matched, _lineno, is_literal, repr_val = sites[0]
    assert source_name == "x"
    assert is_literal is True, f"Negative literal in .get() was not detected: {repr_val}"
    assert repr_val == "-2.0"
