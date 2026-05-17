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


@pytest.mark.parametrize("module_name", SKELETON_MODULES)
def test_skeleton_imports_and_shape(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    params = mod.PARAMS
    assert isinstance(params, tuple), f"{module_name}.PARAMS must be a tuple"
    assert all(isinstance(p, ParamDeclaration) for p in params), (
        f"{module_name}.PARAMS must contain only ParamDeclaration instances"
    )


@pytest.mark.parametrize("module_name", SKELETON_MODULES)
def test_skeleton_currently_empty(module_name: str) -> None:
    # Sentinel — PR2 deletes this assertion as it populates the first
    # skeleton.
    mod = importlib.import_module(module_name)
    assert mod.PARAMS == (), (
        f"{module_name}.PARAMS is no longer empty — remove this assertion "
        "when populating the skeleton in PR2."
    )


def test_param_declaration_three_arg_construction() -> None:
    # Backwards-compat: every existing call site uses ≤3 positional args.
    decl = ParamDeclaration("radio_q_ir", None, "FIR-radio correlation")
    assert decl.name == "radio_q_ir"
    assert decl.description == "FIR-radio correlation"
    assert decl.bound_check is None
    assert decl.bound_error == ""


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
