# SPDX-License-Identifier: BSD-3-Clause
"""Every SEDModelComponent that reads cross-component inputs must declare them.

Issues: #1846, #1755, #1706.

The problem: ``predict`` methods read from ``**inputs`` dict with ``.get()``
fallbacks, but the component does not declare these keys in ``inputs()`` or
``optional_inputs()``. The base ``apply()`` method only populates ``input_kwargs``
from declared keys, so undeclared reads always take their fallback values —
silently bypassing upstream publishers. This is #1706, #1755, #1846 recurrence.

This census:
- Parses every component class's ``predict`` method via AST for ``.get()`` calls
- Lists every key read (including ``**inputs``-style unpacking)
- Asserts every key is declared in ``inputs()`` or ``optional_inputs()``
- Reports coverage per component (a census covering 3 of 33 classes is the disease)
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

pytestmark = pytest.mark.contract


def _extract_input_reads(cls: type) -> set[str]:
    """Extract all keys read from inputs.get(...) and **inputs in predict method.

    Parameters
    ----------
    cls : type
        A component class with a predict method.

    Returns
    -------
    set[str]
        Set of key names read from the inputs dict in predict.
    """
    try:
        source = inspect.getsource(cls.predict)
        source = textwrap.dedent(source)
    except (OSError, TypeError):
        # Can't get source (compiled module, C extension, etc.)
        return set()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Can't parse the source
        return set()

    reads = set()

    class InputVisitor(ast.NodeVisitor):
        """Extract keys from inputs.get("key", ...) and **inputs usage."""

        def visit_Call(self, node: ast.Call) -> None:
            """Find inputs.get("key", ...) calls and **inputs unpacking."""
            # Check for inputs.get(...) pattern
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "inputs"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                reads.add(node.args[0].value)

            # Check for **inputs unpacking in calls
            for keyword in node.keywords:
                if (
                    keyword.arg is None  # **inputs
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "inputs"
                ):
                    reads.add("**inputs")

            self.generic_visit(node)

    visitor = InputVisitor()
    visitor.visit(tree)
    return reads


def _get_declared_inputs(cls: type) -> set[str]:
    """Get all declared input names from inputs() and optional_inputs() methods.

    Parameters
    ----------
    cls : type
        A component class.

    Returns
    -------
    set[str]
        Set of all declared input key names.
    """
    declared = set()

    # Try inputs() method
    if hasattr(cls, "inputs") and callable(cls.inputs):
        try:
            result = cls.inputs(cls())
            if isinstance(result, (list, tuple)):
                for item in result:
                    if hasattr(item, "name"):
                        declared.add(item.name)
                    elif isinstance(item, dict):
                        declared.update(item.keys())
        except Exception:
            pass

    # Try inputs dict attribute (SEDModelComponent style)
    if hasattr(cls, "inputs") and isinstance(getattr(cls, "inputs", None), dict):
        declared.update(cls.inputs.keys())

    # Try optional_inputs() method
    if hasattr(cls, "optional_inputs") and callable(getattr(cls, "optional_inputs", None)):
        try:
            result = cls.optional_inputs(cls())
            if isinstance(result, (list, tuple)):
                for item in result:
                    if hasattr(item, "name"):
                        declared.add(item.name)
                    elif isinstance(item, dict):
                        declared.update(item.keys())
        except Exception:
            pass

    # Try optional_inputs dict attribute (SEDModelComponent style)
    if hasattr(cls, "optional_inputs") and isinstance(getattr(cls, "optional_inputs", None), dict):
        declared.update(cls.optional_inputs.keys())

    return declared


# Per-component allowlist: keys that legitimately are not declared because they
# are handled by special fallback logic (e.g., checked for zero-sum in yang20).
# SHOULD BE EMPTY OR NEAR-EMPTY AFTER #1846 FIX.
LEGITIMATE_UNDECLARED: dict[str, frozenset[str]] = {
    # templates are provided via accepts_threaded_templates pattern, not declared
    "astrodust": frozenset({"templates"}),
}


def test_component_declared_inputs_census() -> None:
    """Verify every component declares all inputs it reads.

    Fails before #1846 fix on XRayAirdSEDComponent with undeclared:
    sfr, log_mstar, stellar_age_gyr, L_2500_30deg, age_weights, ssp_ages_yr.
    """
    import tengri  # noqa: F401 - populate registry
    from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent

    undeclared_by_component: dict[str, set[str]] = {}
    components_checked = []

    for name, cls in sorted(_REGISTRY.items()):
        # Only check SEDModelComponent subclasses (not bare SEDComponent)
        if not issubclass(cls, SEDModelComponent):
            continue

        components_checked.append(name)

        # Extract reads from predict method
        reads = _extract_input_reads(cls)

        # Skip if no reads found
        if not reads or reads == {"**inputs"}:
            continue

        # Get declared inputs
        declared = _get_declared_inputs(cls)

        # Remove **inputs from comparison
        reads_concrete = reads - {"**inputs"}

        # Find undeclared
        undeclared = reads_concrete - declared

        # Allow legitimate exceptions
        allowed = LEGITIMATE_UNDECLARED.get(name, frozenset())
        undeclared -= allowed

        if undeclared:
            undeclared_by_component[name] = undeclared

    # Report coverage
    assert len(components_checked) > 0, (
        "No SEDModelComponent classes found in registry. "
        "Census must cover at least the main component types."
    )

    # Fail with detailed message
    if undeclared_by_component:
        msg = "Components with undeclared input reads (see #1846, #1755, #1706):\n"
        for name in sorted(undeclared_by_component.keys()):
            undeclared = sorted(undeclared_by_component[name])
            msg += f"  {name}: {undeclared}\n"
        msg += f"\nComponents checked: {len(components_checked)}"
        pytest.fail(msg)

    # Report success
    print(f"✓ Census passed: {len(components_checked)} components, all declared")
