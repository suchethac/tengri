# SPDX-License-Identifier: BSD-3-Clause
"""Guard G1: every declared agn_* parameter has a composable reader."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

# Worktree root is 2 levels up from tests/contract/
PROJ_ROOT = Path(__file__).resolve().parents[2]
SRC_AGN_PARAMS = PROJ_ROOT / "src" / "tengri" / "components" / "agn" / "_params.py"
SRC_BLOCKS_CONSUMES = PROJ_ROOT / "src" / "tengri" / "components" / "agn" / "blocks" / "_consumes.py"


def test_every_declared_agn_param_has_a_composable_reader():
    """Assert every declared agn_* param is read by composable blocks."""
    import sys

    # Parse PARAMS
    exec_dict = {}
    with open(SRC_AGN_PARAMS) as f:
        exec(f.read(), exec_dict)
    PARAMS = exec_dict["PARAMS"]

    # Parse AGN_BLOCK_CONSUMES
    exec_dict = {}
    with open(SRC_BLOCKS_CONSUMES) as f:
        exec(f.read(), exec_dict)
    AGN_BLOCK_CONSUMES = exec_dict["AGN_BLOCK_CONSUMES"]

    # Import blocks
    sys.path.insert(0, str(PROJ_ROOT / "src"))
    from tengri.components.agn.blocks import AGN_BLOCKS

    DECLARED = {p.name for p in PARAMS}
    READ = set()
    
    # Read from CONSUMES
    for v in AGN_BLOCK_CONSUMES.values():
        READ |= v
    
    # Runner-level reads: agn_ebv_disc is applied at runner disc stage,
    # agn_lum_ratio used for cross-block normalization
    READ |= {"agn_ebv_disc", "agn_lum_ratio"}
    
    # Read from block signatures (AGN_BLOCKS is nested: {cat: {type: fn}})
    for category, types_dict in AGN_BLOCKS.items():
        for block_type, block_fn in types_dict.items():
            if block_type == "none":
                continue
            sig = inspect.signature(block_fn)
            for param in sig.parameters.values():
                if param.name.startswith("agn_") and param.kind not in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    READ.add(param.name)

    # Exceptions
    NOT_COMPOSABLE = {
        "agn_delta": "monolithic SKIRTORTorus only",
        "agn_tau_torus": "monolithic torus class only (measure/refusal pending T1)",
    }

    _PENDING = {
        "agn_grahsp_netzer_log_mbh": "T4: read by disc:grahsp_netzer",
        "agn_grahsp_netzer_spin": "T4: read by disc:grahsp_netzer",
        "agn_grahsp_netzer_log_mdot": "T4: read by disc:grahsp_netzer",
    }

    unreached = DECLARED - READ - set(NOT_COMPOSABLE) - set(_PENDING)
    assert not unreached, f"Unreached: {sorted(unreached)}"


def test_pending_allowlist_not_stale():
    """Assert _PENDING exceptions are still needed."""
    import sys

    # Parse data
    exec_dict = {}
    with open(SRC_AGN_PARAMS) as f:
        exec(f.read(), exec_dict)
    PARAMS = exec_dict["PARAMS"]

    exec_dict = {}
    with open(SRC_BLOCKS_CONSUMES) as f:
        exec(f.read(), exec_dict)
    AGN_BLOCK_CONSUMES = exec_dict["AGN_BLOCK_CONSUMES"]

    sys.path.insert(0, str(PROJ_ROOT / "src"))
    from tengri.components.agn.blocks import AGN_BLOCKS

    DECLARED = {p.name for p in PARAMS}
    READ = set()
    
    for v in AGN_BLOCK_CONSUMES.values():
        READ |= v
    
    READ |= {"agn_ebv_disc", "agn_lum_ratio"}
    
    for category, types_dict in AGN_BLOCKS.items():
        for block_type, block_fn in types_dict.items():
            if block_type == "none":
                continue
            sig = inspect.signature(block_fn)
            for param in sig.parameters.values():
                if param.name.startswith("agn_") and param.kind not in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    READ.add(param.name)

    _PENDING = {
        "agn_grahsp_netzer_log_mbh": "T4: read by disc:grahsp_netzer",
        "agn_grahsp_netzer_spin": "T4: read by disc:grahsp_netzer",
        "agn_grahsp_netzer_log_mdot": "T4: read by disc:grahsp_netzer",
    }

    not_read = DECLARED - READ
    stale = set(_PENDING) - not_read
    assert not stale, f"Stale _PENDING: {sorted(stale)}"
