# SPDX-License-Identifier: BSD-3-Clause
"""Contract: literal signature defaults and .get() fallbacks in dust tree match declarations.

Two independent checks, deliberately kept separate because they read different
things:

- :func:`test_dust_closure_literals_match_declarations` is a value-drift check
  **complementary to** ``tools/check_literal_param_defaults.py``'s structural
  guard, not a restatement of it: it parses each named source file by path
  (``ast.parse(path.read_text())``) and compares a bare numeral standing in
  for a name ``PARAMS``/``ATTENUATION_PARAMS`` declares against that
  declaration's current value, failing only on a mismatch. A literal that
  happens to match the current declared value stays green here while the CLI
  guard flags it regardless -- that guard asks the structural question ("is a
  declared name spelled as a literal at all"), this test asks the value
  question ("does the literal still agree with the declaration"). It never
  imports ``tengri``, so it can be pointed at any checkout by changing
  ``ROOT``/``DUST_TREE`` rather than the Python import path. It resolves the
  bare spelling of a declared name (e.g. ``f_obscuration`` for
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
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
    assert file_path.exists(), f"{filename} not found in dust tree -- this is a repo error"

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
# table stays at 35.0, left unchanged pending #2261 (see ``_params.py``'s
# ``dust_T`` docstring).
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
        "declares Fixed(MBB_T_K_DEFAULT)=30.0; table keeps Fixed(35.0), left "
        "unchanged pending #2261 -- see _params.py's dust_T docstring."
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
        # A documented exception records a live disagreement, not a permanent
        # exemption: if the table and the class have since been reconciled
        # (e.g. a future correction like #2265's dust_beta_ir fix), this
        # exception is stale and must fail loudly rather than keep skipping.
        assert (
            class_default is not None
            and table_default is not None
            and abs(class_default - table_default) > 1e-10
        ), (
            f"{cls.__name__} declares {full_name}={class_default}, matching the "
            f"table's {table_default} -- the documented exception "
            f"({_TABLE_CLASS_EXCEPTIONS[exception_key]}) is stale and must be "
            "removed from _TABLE_CLASS_EXCEPTIONS."
        )
        pytest.skip(f"documented exception: {_TABLE_CLASS_EXCEPTIONS[exception_key]}")

    assert table_default is not None, f"{full_name} has no default in the table"
    assert class_default is not None, f"{cls.__name__}.{full_name} has no class default"
    assert abs(class_default - table_default) < 1e-10, (
        f"{cls.__name__} declares {full_name}={class_default}, but the shared "
        f"table declares {table_default}. Either correct the table entry to "
        "match the class or add a reasoned entry to _TABLE_CLASS_EXCEPTIONS."
    )


# ── Grammar-path pin for a law's own wildcard default (#2265 gap) ──────────
#
# ``KRIEK_CONROY_BUMP_STRENGTH_DEFAULT`` / ``TEA_DELTA_DEFAULT``
# (``_params.py``) were pinned only on the direct function call
# (``TestKriekConroyMatchesFSPS`` in
# ``tests/components/dust/test_dust_attenuation_laws.py``), never on the
# ``SEDModel.build`` grammar path a wildcard caller actually exercises. These
# two cases close that gap: a ``single_component`` build with
# ``all_params: Fixed(DEFAULT)`` must predict the same photometry as one that
# states the law's own published value explicitly, and different photometry
# from one that states the shared ``ATTENUATION_PARAMS`` table's
# structural-off zero.


def _bump_ssp():
    """UV-sensitive synthetic SSP, so the 2175 A bump reaches photometry."""
    ages = jnp.linspace(-3.0, 1.14, 20)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 700)
    base = ((5000.0 / wave) ** 2 * jnp.where(wave < 912.0, 1e-6, 1.0))[None, None, :]
    flux = (
        base
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    from tengri import SSPData

    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


def _bump_obs():
    """Bands straddling the 2175 A bump, where kriek_conroy/tea differ."""
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    def _tophat(center, n=24):
        wave = jnp.linspace(center * 0.8, center * 1.2, n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")

    return Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (1500.0, 2800.0, 3500.0, 6200.0)))
    )


def _wildcard_build(ssp, obs, dust_attenuation):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_attenuation,
        redshift=Fixed(0.5),
    )


def _wildcard_photometry(model):
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    return np.asarray(model.predict_photometry(params))


def _wildcard_rel(a, b):
    return float(np.max(np.abs(a - b) / np.where(np.abs(b) > 0, np.abs(b), 1.0)))


@pytest.fixture(scope="module")
def _wildcard_ssp():
    return _bump_ssp()


@pytest.fixture(scope="module")
def _wildcard_obs():
    return _bump_obs()


@pytest.mark.parametrize(
    ("law", "key", "own_default", "table_off"),
    [
        ("kriek_conroy", "bump_strength", 1.0, 0.0),
        ("tea", "delta", -0.2, 0.0),
    ],
)
def test_wildcard_single_component_reads_the_laws_own_default(
    _wildcard_ssp, _wildcard_obs, law, key, own_default, table_off
):
    """A wildcard ``single_component`` build reads the LAW's own default.

    ``dust_bump_strength`` / ``dust_delta`` are both declared ``Fixed(0.0)``
    in the shared ``ATTENUATION_PARAMS`` table (the structural "off" value for
    a law like ``calzetti`` that discards them) -- ``resolve_bc_diff_law_params``
    hands a caller that never asked (``all_params: Fixed(DEFAULT)`` included)
    the selected law's own published constant instead (#1833). Only the direct
    function call was pinned before this test; here the grammar path is
    exercised end to end.
    """
    wildcard = _wildcard_photometry(
        _wildcard_build(
            _wildcard_ssp,
            _wildcard_obs,
            {"type": "single_component", "law": law, "all_params": Fixed(DEFAULT)},
        )
    )
    stated_own = _wildcard_photometry(
        _wildcard_build(
            _wildcard_ssp,
            _wildcard_obs,
            {
                "type": "single_component",
                "law": law,
                "all_params": Fixed(DEFAULT),
                key: Fixed(own_default),
            },
        )
    )
    stated_off = _wildcard_photometry(
        _wildcard_build(
            _wildcard_ssp,
            _wildcard_obs,
            {
                "type": "single_component",
                "law": law,
                "all_params": Fixed(DEFAULT),
                key: Fixed(table_off),
            },
        )
    )
    assert _wildcard_rel(wildcard, stated_own) <= 1e-9, (
        f"{law}: a wildcard build differs from stating its own published "
        f"default ({key}={own_default}) explicitly -- the grammar path is not "
        "reading the law's own constant."
    )
    moved = _wildcard_rel(wildcard, stated_off)
    assert moved > 1e-3, (
        f"{law}: the wildcard default is indistinguishable from the shared "
        f"table's structural-off {key}={table_off} (rel {moved:.3e}) -- the "
        "law's own default is not reaching the grammar path."
    )
