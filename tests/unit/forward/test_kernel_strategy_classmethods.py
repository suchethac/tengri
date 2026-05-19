"""Tests for the scientist-facing ``KernelStrategy`` classmethods.

The classmethods (``fast()``, ``bit_exact()``, ``low_memory()``,
``reference_only()``) are aliases for the engineering-facing module
singletons (``DEFAULT``, ``COMPOSITIONAL_ONLY``, ``LOW_MEMORY``,
``EXACT_ONLY``). These tests pin the identity guarantee so existing
``is``-comparisons in production code and tests keep working whichever
surface the caller used.
"""

from __future__ import annotations

import pytest

from tengri.forward._kernels import (
    COMPOSITIONAL_ONLY,
    DEFAULT,
    EXACT_ONLY,
    LOW_MEMORY,
    KernelStrategy,
)


@pytest.mark.parametrize(
    "classmethod_name, expected_singleton",
    [
        ("fast", DEFAULT),
        ("bit_exact", COMPOSITIONAL_ONLY),
        ("low_memory", LOW_MEMORY),
        ("reference_only", EXACT_ONLY),
    ],
)
def test_classmethod_is_aliased_to_singleton(classmethod_name, expected_singleton):
    """Each classmethod returns the *same instance* as the engineering name."""
    method = getattr(KernelStrategy, classmethod_name)
    assert method() is expected_singleton


def test_classmethods_are_idempotent():
    """Calling the same classmethod twice returns the same instance."""
    for name in ("fast", "bit_exact", "low_memory", "reference_only"):
        method = getattr(KernelStrategy, name)
        assert method() is method()


def test_classmethod_docstrings_describe_physics_impact():
    """Each classmethod's docstring names the physics tradeoff (not just
    the engineering term). Light contract — keeps drift visible."""
    expectations = {
        "fast": "tolerance",
        "bit_exact": "bit-identical",
        "low_memory": "hybrid",
        "reference_only": "reference",
    }
    for name, marker in expectations.items():
        doc = getattr(KernelStrategy, name).__doc__ or ""
        assert marker in doc.lower(), f"KernelStrategy.{name}.__doc__ should mention '{marker}'"


def test_classmethods_callable_on_instance_too():
    """Python classmethods are accessible from both class and instance."""
    s = KernelStrategy()
    assert s.fast() is DEFAULT
    assert s.bit_exact() is COMPOSITIONAL_ONLY
