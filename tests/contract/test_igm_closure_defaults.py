# SPDX-License-Identifier: BSD-3-Clause
"""Contract test: IGM closure defaults match declared parameters.

Compares every closure (pure-function) default in the IGM module against
the corresponding declared parameter from the registry, ensuring that no
signature default silently drifts from its declaration (#2241).
"""

from __future__ import annotations

import inspect

import pytest

from tengri.components.igm import igm
from tengri.components.igm._params import DLA_PARAMS, PARAMS as _IGM_PARAMS
from tengri.protocols.component import declared_default


@pytest.mark.contract
def test_igm_closure_function_defaults_match_declarations() -> None:
    """Every signature default in IGM module functions matches its declaration.

    Compares the resolved default value of each declared-name parameter in
    each IGM function against the corresponding entry in the parameter tables,
    ensuring they are bit-identical. This test is RED at HEAD on dla_log_n_hi
    (declared 20.3, function default was 20.0) due to #2265.
    """
    # Build a map of declared-name -> default value from all IGM param tables
    declared_defaults = {}
    for decl in _IGM_PARAMS:
        declared_defaults[decl.name] = declared_default(_IGM_PARAMS, decl.name)
    for decl in DLA_PARAMS:
        declared_defaults[decl.name] = declared_default(DLA_PARAMS, decl.name)

    # Check all functions in the igm module for signature defaults
    # that match declared names
    mismatches = []
    for name, obj in inspect.getmembers(igm, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(obj)
        for param_name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                continue
            # Check if this parameter name is in the declared set
            if param_name not in declared_defaults:
                continue
            # If there's a default, it should match the declaration
            declared_val = declared_defaults[param_name]
            actual_val = param.default
            if actual_val != declared_val:
                mismatches.append(
                    f"{name}(): {param_name} = {actual_val} (declared {declared_val})"
                )

    assert not mismatches, "IGM function defaults do not match declarations:\n" + "\n".join(
        mismatches
    )
