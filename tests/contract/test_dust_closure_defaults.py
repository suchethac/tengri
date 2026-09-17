# SPDX-License-Identifier: BSD-3-Clause
"""Contract: literal signature defaults and .get() fallbacks in dust tree match declarations.

Two independent checks, deliberately kept separate because they read different
things:

- :func:`test_dust_closure_literals_match_declarations` is an **AST check**: it
  parses each named source file by path (``ast.parse(path.read_text())``) and
  looks for a bare numeral standing in for a name ``PARAMS``/``ATTENUATION_PARAMS``
  declares. It never imports ``tengri``, so it can be pointed at any checkout by
  changing ``ROOT``/``DUST_TREE`` rather than the Python import path. It
  resolves the bare spelling of a declared name (e.g. ``f_obscuration`` for
  ``dust_f_obscuration``) the same way ``tools/check_literal_param_defaults.py``
  does -- see :func:`_resolve_bare_name`.
- :func:`test_dust_params_table_consistency` **imports tengri objects**
  (``_REGISTRY``, the component classes, ``PARAMS``/``ATTENUATION_PARAMS``) and
  compares each registered dust emission component's own class-level prior
  against the shared table, parametrized one case per (class, declared name)
  pair, with a documented exception list for the pairs left as a tracked,
  live discrepancy rather than reconciled.
"""

import ast
from pathlib import Path

import pytest

from tengri.components.dust._params import ATTENUATION_PARAMS, PARAMS
from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.components.sed_model_component import _REGISTRY
from tengri.parameters.priors import Distribution
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parent.parent.parent
DUST_TREE = ROOT / "src" / "tengri" / "components" / "dust"

#: The only bare-name fallback this suite knows, mirroring
#: ``tools/check_literal_param_defaults.py``'s ``_file_fallback_prefix``: every
#: file under ``components/dust/`` may spell a declared name bare (e.g.
#: ``f_obscuration`` for ``dust_f_obscuration``), because ``EmissionComponent``
#: and the two-component/single-screen closures all set
#: ``parameter_prefix = "dust_"`` without every call site restating it. All the
#: files this test parses live under ``DUST_TREE``, so this prefix always
#: applies -- unlike the CLI guard, this test never widens beyond that tree.
_BARE_NAME_PREFIX = "dust_"


def _resolve_bare_name(name: str, expected: dict[str, float]) -> str | None:
    """Match ``name`` against ``expected``, trying the bare and prefixed spelling.

    Before this fix, ``f_obscuration`` (the ``_apply.py`` / ``two_component.py``
    / ``energy_balance_precompute.py`` spelling of ``dust_f_obscuration``) was
    invisible to this test: ``"f_obscuration" not in expected`` and the site was
    silently skipped, so a literal-copy regression there would not have failed
    here even though ``check_literal_param_defaults.py`` catches it. Checking
    the prefixed spelling as a fallback closes that gap.
    """
    if name in expected:
        return name
    prefixed = f"{_BARE_NAME_PREFIX}{name}"
    if prefixed in expected:
        return prefixed
    return None


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
    """Every numeric literal for a declared dust param matches its declaration.

    AST check: parses ``filename`` by path and never imports ``tengri``.
    """
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
            matched = _resolve_bare_name(param_name, expected)
            if matched is not None:
                expected_value = expected[matched]
                if not (expected_value is None or abs(literal_value - expected_value) < 1e-10):
                    failures.append(
                        f"{file_path.name}:{lineno} {func_name}() {param_name}={literal_value} "
                        f"(expected {expected_value} for {matched})"
                    )
        elif ".get(" in key:
            # Extract the key name from ".get(name)"
            key_name = key.split("(")[1].rstrip(")")
            matched = _resolve_bare_name(key_name, expected)
            if matched is not None:
                expected_value = expected[matched]
                if not (expected_value is None or abs(literal_value - expected_value) < 1e-10):
                    failures.append(
                        f"{file_path.name}:{lineno} {key} literal={literal_value} "
                        f"(expected {expected_value} for {matched})"
                    )

    if failures:
        pytest.fail("\n".join(failures))


# ── Registered-class vs shared-table consistency ───────────────────────────
#
# The component's own class-level prior is the declaration for a name that
# some class declares; a shared ``components/dust/_params.py`` table entry
# that disagrees is a stale copy. ``dust_T`` is the one case where multiple
# classes disagree with EACH OTHER, not just the table: MBB/schreiber2016
# read 30.0, graybody/casey2012 read 35.0, schreiber2018 reads 25.0, and the
# table keeps 35.0 as the majority (see ``_params.py``'s ``dust_T``
# docstring).
#
# Every OTHER class-vs-table disagreement this test finds must either match
# (the common case) or be a reasoned, named exception below -- never silently
# pass. ``dust_lgU`` is the one other case: AstrodustIRSEDComponent and
# Draine2021PAHIRSEDComponent agree with EACH OTHER (default=1.0) but not
# with the table's ``Fixed(0.0)``; see the matching note on ``_params.py``'s
# ``dust_lgU`` entry for why that stays a tracked discrepancy (#2261) rather
# than corrected here.
_TABLE_CLASS_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("ModifiedBlackbodyIRSEDComponent", "dust_T"): (
        "declares Fixed(MBB_T_K_DEFAULT)=30.0; table keeps Fixed(35.0), the "
        "majority across graybody/casey2012/MBB/schreiber2016/schreiber2018 "
        "-- see _params.py's dust_T docstring, #2261."
    ),
    ("Schreiber2016AnalyticIRSEDComponent", "dust_T"): (
        "declares Fixed(SCHREIBER_T_K_DEFAULT)=30.0; see dust_T docstring, #2261."
    ),
    ("Schreiber2018IRSEDComponent", "dust_T"): (
        "declares Fixed(SCHREIBER2018_T_K_DEFAULT)=25.0; see dust_T docstring, #2261."
    ),
    ("AstrodustIRSEDComponent", "dust_lgU"): (
        "declares default=1.0; the table's Fixed(0.0) disagrees with both "
        "lgU-declaring classes even though they agree with each other -- "
        "see dust_lgU docstring, #2265."
    ),
    ("Draine2021PAHIRSEDComponent", "dust_lgU"): (
        "declares default=1.0; see dust_lgU docstring, #2265."
    ),
}


def _registered_emission_class_param_cases():
    """Every (class, prefixed_name, class_default) the shared table also names.

    Walks ``_REGISTRY`` (populated by ``SEDModelComponent.__init_subclass__``
    for every concrete component with its own ``name``), keeps subclasses of
    ``EmissionComponent`` (the dust IR emission authoring base -- excludes the
    attenuation/two-component/single-screen classes, which read
    ``ATTENUATION_PARAMS`` structurally rather than via class-level
    ``Fixed``/``Uniform`` attributes), and yields one case per class-level
    ``Distribution`` attribute whose prefixed name is a ``PARAMS`` or
    ``ATTENUATION_PARAMS`` entry. A class attribute whose name is NOT a table
    entry (e.g. a genuinely component-private knob) is not a case here -- this
    mirrors the CLI guard's declared-name-set construction, which only ever
    flags a name a declaration already owns.
    """
    expected = {}
    for pd in PARAMS:
        expected[pd.name] = declared_default(PARAMS, pd.name)
    for pd in ATTENUATION_PARAMS:
        expected[pd.name] = declared_default(ATTENUATION_PARAMS, pd.name)

    seen_classes = set()
    cases = []
    for _name, cls in sorted(_REGISTRY.items(), key=lambda item: item[0]):
        if cls in seen_classes or not issubclass(cls, EmissionComponent):
            continue
        seen_classes.add(cls)
        prefix = getattr(cls, "parameter_prefix", None) or "dust_"
        for attr_name, value in sorted(vars(cls).items()):
            if attr_name.startswith("_") or not isinstance(value, Distribution):
                continue
            full_name = f"{prefix}{attr_name}"
            if full_name not in expected:
                continue
            class_default = getattr(value, "default", None)
            cases.append((cls, full_name, class_default, expected[full_name]))
    return cases


_TABLE_CLASS_CASES = _registered_emission_class_param_cases()
_TABLE_CLASS_IDS = [f"{cls.__name__}:{name}" for cls, name, _cd, _td in _TABLE_CLASS_CASES]


@pytest.mark.parametrize(
    ("cls", "full_name", "class_default", "table_default"),
    _TABLE_CLASS_CASES,
    ids=_TABLE_CLASS_IDS,
)
def test_dust_params_table_consistency(cls, full_name, class_default, table_default):
    """A registered class's own prior matches the shared table, or is excepted.

    Imports tengri objects (``_REGISTRY``, the component classes, the two
    param tables) and is parametrized one case per (class, declared name)
    pair (see :func:`_registered_emission_class_param_cases`).
    """
    exception_key = (cls.__name__, full_name)
    if exception_key in _TABLE_CLASS_EXCEPTIONS:
        pytest.skip(f"documented exception: {_TABLE_CLASS_EXCEPTIONS[exception_key]}")

    assert table_default is not None, f"{full_name} has no default in the table"
    assert class_default is not None, f"{cls.__name__}.{full_name} has no class default"
    assert abs(class_default - table_default) < 1e-10, (
        f"{cls.__name__} declares {full_name}={class_default}, but the shared "
        f"table declares {table_default}. Either correct the table entry to "
        "match the class or add a reasoned entry to _TABLE_CLASS_EXCEPTIONS."
    )
