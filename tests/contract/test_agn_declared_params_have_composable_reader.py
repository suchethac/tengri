# SPDX-License-Identifier: BSD-3-Clause
"""Guard G1: Every declared agn_* parameter must have a composable reader.

Detects "accepted-but-inert" parameters: names declared in the parameters registry
but never read by any composable block (they are legal at the top level because
ownership is "agn" = shared, but they have no effect on the composable output).

Related issues: #985 (GRAHSP Balmer inert), #2175 (qsogen_balmer inert), #941 (#1168).
"""

from __future__ import annotations

import inspect
from inspect import Parameter

import pytest

from tengri.components.agn._params import PARAMS
from tengri.components.agn.blocks import AGN_BLOCKS
from tengri.components.agn.blocks._consumes import AGN_BLOCK_CONSUMES, AGN_SHARED_PARAMS
from tengri.parameters.agn_ownership import (
    _AGN_CATEGORY_WIDE_COMPANION_PARAMS,
    _agn_subblock_companion_params,
)
from tengri.parameters.groups import (
    _AGN_CONSUMES_CATEGORY,
    _agn_subblock_declared_params,
)

pytestmark = pytest.mark.contract

# Allowlist of declared parameters that are intentionally NOT read by any
# composable block, with documented reasons. Each entry MUST carry a reason
# string. This test also fails if an entry becomes stale (now read by a block).
_PENDING = {
    "agn_grahsp_netzer_log_mbh": "T4: read by disc:grahsp_netzer",
    "agn_grahsp_netzer_spin": "T4: read by disc:grahsp_netzer",
    "agn_grahsp_netzer_log_mdot": "T4: read by disc:grahsp_netzer",
}

# Measured non-response: these parameters exist but are documented to be
# monolithic-class-only or have been verified not to move sed_agn when varied
# on all composable blocks.
_MEASURED_INERT = {
    "agn_delta": (
        "Monolithic SKIRTORTorus class only (components/agn/skirtor_model.py:67); "
        "composable design puts disc shape in a disc block. "
        "Measurement: all composable torus blocks tested with agn_delta perturbed — no response."
    ),
    "agn_tau_torus": (
        "Measured non-response on all composable torus blocks "
        "(grahsp, cat3d_wind, relagn, skirtor_torus_only) "
        "with agn_tau_torus 1 -> 10: max relative change 0.0. "
        "Monolithic torus functions read it (components/agn/torus.py:65,151) "
        "but composable blocks do not."
    ),
}


def _read_declared_params() -> set[str]:
    """All declared parameter names from the parameter registry."""
    return {p.name for p in PARAMS}


def _read_composable_params() -> set[str]:
    """All parameters read by composable blocks (any category, any type)."""
    read_params = set(AGN_SHARED_PARAMS)

    # Add all CONSUMES entries
    for v in AGN_BLOCK_CONSUMES.values():
        read_params |= v

    # Add category-wide companion params (e.g., wildcard-freed params)
    for v in _AGN_CATEGORY_WIDE_COMPANION_PARAMS.values():
        read_params |= v

    # Build grammar category mapping
    grammar_of = {v: k for k, v in _AGN_CONSUMES_CATEGORY.items()}

    # Add params from registered block signatures and companion params
    for category in AGN_BLOCKS:
        for block_type, block_fn in AGN_BLOCKS[category].items():
            if block_type == "none":
                continue
            sig = inspect.signature(block_fn)
            for param_name, param in sig.parameters.items():
                if (
                    param_name.startswith("agn_")
                    and param.kind not in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
                ):
                    read_params.add(param_name)

            # Add sub-block companion params (e.g., fixed params that freeable wildcards co-declare)
            try:
                grammar_category = grammar_of.get(category, category)
                declared = _agn_subblock_declared_params(
                    grammar_category, block_type, selection={grammar_category: block_type}
                )
                read_params |= declared
                companion = _agn_subblock_companion_params(grammar_category, block_type)
                read_params |= companion
            except (KeyError, ValueError):
                pass

    return read_params


def test_every_declared_agn_param_has_a_composable_reader():
    """Every declared agn_* parameter must be read by at least one composable block."""
    declared = _read_declared_params()
    read = _read_composable_params()
    inert = declared - read - set(_PENDING) - set(_MEASURED_INERT)

    assert (
        not inert
    ), f"Declared agn_* parameters with no composable reader (accepted-but-inert): {sorted(inert)}. These are wired into the grammar but have zero effect on composable output. Check _PENDING and _MEASURED_INERT allowlists."


def test_not_composable_allowlist_is_not_stale():
    """Entries in _PENDING and _MEASURED_INERT must still be inert (not stale)."""
    declared = _read_declared_params()
    read = _read_composable_params()

    # _PENDING entries should still be declared but not read
    for name in _PENDING:
        assert name in declared, f"_PENDING entry {name!r} is no longer declared (stale)."
        assert name not in read, (
            f"_PENDING entry {name!r} is now read by a composable block "
            f"— remove it from _PENDING."
        )

    # _MEASURED_INERT entries should still be declared but not read
    for name in _MEASURED_INERT:
        assert name in declared, f"_MEASURED_INERT entry {name!r} is no longer declared (stale)."
        assert name not in read, (
            f"_MEASURED_INERT entry {name!r} is now read by a composable block "
            f"— remove it from _MEASURED_INERT."
        )
