# SPDX-License-Identifier: BSD-3-Clause
"""Contract test: stellar closure defaults match declared parameters.

Compares every closure (pure-function) default in the stellar module against
the corresponding declared parameter from the registry, ensuring that no
signature default silently drifts from its declaration (#2241).
"""

from __future__ import annotations

import inspect

import pytest

from tengri.components.stellar import component
from tengri.components.stellar._params import ALPHA_FE_PARAMS
from tengri.protocols.component import declared_default


@pytest.mark.contract
def test_stellar_closure_function_defaults_match_declarations() -> None:
    """Every signature default in stellar module functions matches its declaration.

    Compares the resolved default value of each declared-name parameter in
    each stellar function against the corresponding entry in ALPHA_FE_PARAMS,
    ensuring they are bit-identical.
    """
    # Build a map of declared-name -> default value from ALPHA_FE_PARAMS
    declared_defaults = {
        decl.name: declared_default(ALPHA_FE_PARAMS, decl.name) for decl in ALPHA_FE_PARAMS
    }

    # Check all functions in the stellar module for signature defaults
    # that match declared names
    mismatches = []
    for name, obj in inspect.getmembers(component, inspect.isfunction):
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

    assert not mismatches, "Stellar function defaults do not match declarations:\n" + "\n".join(
        mismatches
    )
