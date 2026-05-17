"""Smoke tests for the per-component ``_params.py`` skeletons (PR1).

PR1 of the parameter-registry consolidation introduces empty
``PARAMS`` tuples under each component directory. PR2+ will populate
them; until then this test asserts the scaffolding shape so the import
graph stays healthy and the contract is unambiguous.
"""

from __future__ import annotations

import importlib

import pytest

from tengri.core.component import ParamDeclaration

SKELETON_MODULES = (
    "tengri.components.radio._params",
    "tengri.components.agn._params",
    "tengri.components.dust._params",
    "tengri.components.nebular._params",
    "tengri.components.igm._params",
    "tengri.components.xray._params",
    "tengri.components.stellar._params",
    "tengri.components.stellar.sfh._params",
    "tengri.components.stellar.sps._params",
)

# Skeletons whose ``PARAMS`` have been populated by later PRs in the
# consolidation. Excluded from the empty-sentinel assertion.
POPULATED_MODULES = frozenset({"tengri.components.radio._params"})


@pytest.mark.parametrize("module_name", SKELETON_MODULES)
def test_skeleton_imports_and_shape(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    params = mod.PARAMS
    assert isinstance(params, tuple), f"{module_name}.PARAMS must be a tuple"
    assert all(isinstance(p, ParamDeclaration) for p in params), (
        f"{module_name}.PARAMS must contain only ParamDeclaration instances"
    )


@pytest.mark.parametrize("module_name", sorted(set(SKELETON_MODULES) - POPULATED_MODULES))
def test_skeleton_currently_empty(module_name: str) -> None:
    # Sentinel — each later PR removes its module from POPULATED_MODULES.
    mod = importlib.import_module(module_name)
    assert mod.PARAMS == (), (
        f"{module_name}.PARAMS is no longer empty — add it to "
        "POPULATED_MODULES at the top of this test file."
    )


def test_param_declaration_three_arg_construction() -> None:
    # Backwards-compat: every existing call site uses ≤3 positional args.
    decl = ParamDeclaration("radio_q_ir", None, "FIR-radio correlation")
    assert decl.name == "radio_q_ir"
    assert decl.description == "FIR-radio correlation"
    assert decl.bound_check is None
    assert decl.bound_error == ""


def test_radio_legacy_bucket_matches_canonical_tuple() -> None:
    # PR2 contract: the legacy 4-tuple bucket in _param_defs must be a
    # pure derived view of the canonical RADIO PARAMS tuple. Names,
    # priors, descriptions, and bound_error must agree byte-for-byte.
    from tengri.components.radio._params import PARAMS as canonical
    from tengri.parameters._param_defs import _RADIO_PARAMS as bucket

    assert set(bucket) == {d.name for d in canonical}
    for decl in canonical:
        desc, _check, err, default = bucket[decl.name]
        assert desc == decl.description
        assert err == decl.bound_error
        assert default is decl.prior


def test_param_declaration_five_arg_construction() -> None:
    # New surface added in PR1: components can now own bound metadata.
    decl = ParamDeclaration(
        "dust_tau_bc",
        None,
        "Birth cloud optical depth",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
    )
    assert decl.bound_check(0.0, 4.0) is True
    assert decl.bound_check(-0.1, 4.0) is False
    assert decl.bound_error == "must have lo >= 0"
