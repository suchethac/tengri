# SPDX-License-Identifier: BSD-3-Clause
"""Contract: literal signature defaults and .get() fallbacks in dust tree match declarations."""

import ast
from pathlib import Path

import pytest

from tengri.components.dust._params import ATTENUATION_PARAMS, PARAMS
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parent.parent.parent
DUST_TREE = ROOT / "src" / "tengri" / "components" / "dust"


def _extract_numeric_defaults_from_file(path: Path) -> dict[str, tuple[int, float]]:
    """Extract all (lineno, literal_value) for numeric defaults in a file."""
    tree = ast.parse(path.read_text())
    defaults = {}

    for node in ast.walk(tree):
        # Handle function signature defaults
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for i, arg in enumerate(node.args.args + node.args.kwonlyargs):
                # Find the default for this argument
                defaults_list = node.args.defaults + node.args.kw_defaults
                idx = i - (len(node.args.args) - len(node.args.defaults))
                if 0 <= idx < len(defaults_list) and defaults_list[idx] is not None:
                    default_node = defaults_list[idx]
                    if isinstance(default_node, ast.Constant) and isinstance(
                        default_node.value, (int, float)
                    ):
                        key = f"{node.name}:{arg.arg}"
                        defaults[key] = (default_node.lineno, default_node.value)
                    elif isinstance(default_node, ast.UnaryOp) and isinstance(
                        default_node.op, (ast.UAdd, ast.USub)
                    ):
                        if isinstance(default_node.operand, ast.Constant) and isinstance(
                            default_node.operand.value, (int, float)
                        ):
                            value = default_node.operand.value
                            if isinstance(default_node.op, ast.USub):
                                value = -value
                            key = f"{node.name}:{arg.arg}"
                            defaults[key] = (default_node.lineno, value)

        # Handle .get() calls
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) == 2
        ):
            first_arg = node.args[0]
            second_arg = node.args[1]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                if isinstance(second_arg, ast.Constant) and isinstance(
                    second_arg.value, (int, float)
                ):
                    key = f".get({first_arg.value})"
                    defaults[key] = (node.lineno, second_arg.value)
                elif isinstance(second_arg, ast.UnaryOp) and isinstance(
                    second_arg.op, (ast.UAdd, ast.USub)
                ):
                    if isinstance(second_arg.operand, ast.Constant) and isinstance(
                        second_arg.operand.value, (int, float)
                    ):
                        value = second_arg.operand.value
                        if isinstance(second_arg.op, ast.USub):
                            value = -value
                        key = f".get({first_arg.value})"
                        defaults[key] = (node.lineno, value)

    return defaults


@pytest.mark.parametrize(
    "filename",
    [
        "attenuation.py",
        "emission_templates.py",
        "_apply.py",
        "two_component.py",
        "energy_balance_precompute.py",
        "component.py",
        "wg00_model.py",
    ],
)
def test_dust_closure_literals_match_declarations(filename):
    """Every numeric literal for a declared dust param matches its declaration."""
    file_path = DUST_TREE / filename
    if not file_path.exists():
        pytest.skip(f"{filename} not found in dust tree")

    defaults = _extract_numeric_defaults_from_file(file_path)

    # Build the expected defaults from PARAMS and ATTENUATION_PARAMS
    expected = {}
    for pd in PARAMS:
        expected[pd.name] = declared_default(PARAMS, pd.name)
    for pd in ATTENUATION_PARAMS:
        expected[pd.name] = declared_default(ATTENUATION_PARAMS, pd.name)

    # Check each signature default
    failures = []
    for key, (lineno, literal_value) in defaults.items():
        if ":" in key:
            # Signature default
            func_name, param_name = key.rsplit(":", 1)
            if param_name in expected:
                expected_value = expected[param_name]
                if not (expected_value is None or abs(literal_value - expected_value) < 1e-10):
                    failures.append(
                        f"{file_path.name}:{lineno} {func_name}() {param_name}={literal_value} "
                        f"(expected {expected_value})"
                    )
        elif ".get(" in key:
            # Extract the key name from ".get(name)"
            key_name = key.split("(")[1].rstrip(")")
            if key_name in expected:
                expected_value = expected[key_name]
                if not (expected_value is None or abs(literal_value - expected_value) < 1e-10):
                    failures.append(
                        f"{file_path.name}:{lineno} {key} literal={literal_value} "
                        f"(expected {expected_value})"
                    )

    if failures:
        pytest.fail("\n".join(failures))


@pytest.mark.parametrize(
    ("declaration_tuple", "param_name"),
    [
        (PARAMS, "dust_T"),
        (PARAMS, "dust_beta_ir"),
        (ATTENUATION_PARAMS, "dust_bump_strength"),
        (ATTENUATION_PARAMS, "dust_delta"),
        (ATTENUATION_PARAMS, "dust_Rv"),
        (ATTENUATION_PARAMS, "dust_slope"),
    ],
)
def test_dust_params_table_consistency(declaration_tuple, param_name):
    """The _params.py table entries are reasonable."""
    # This test just verifies that the declarations exist and have reasonable values
    default_val = declared_default(declaration_tuple, param_name)
    assert default_val is not None, f"{param_name} has no default in the table"
    assert isinstance(default_val, (int, float)), f"{param_name} default is not numeric"
