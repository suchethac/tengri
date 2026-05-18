# SPDX-License-Identifier: BSD-3-Clause
"""Regression snapshots for the ADR-0004 cross-component contract.

Two layers of regression protection, both keyed on a deterministic
fixture file at ``tests/integration/_snapshots/derived_contract_v1.json``:

1. **Contract-graph snapshot** (always runs). For each canonical
   pipeline configuration, build the component list via
   :func:`tengri.forward.component_factory.build_components`, then
   collect the full ``(component_class, role, key_name, units)`` tuple
   set across every component's ``publishes`` / ``requires`` /
   ``requires_optional`` annotation. The set is canonicalised and
   hashed. Catches a future PR that silently changes an annotation key
   string, units string, or which component owns which key — even when
   the runtime behaviour cannot be exercised locally (no SSP data).

2. **SED-output snapshot** (skips when SSP not available). For each
   ``(recipe-equivalent configuration, fixed PRNG key)`` pair, build
   the components, sample a parameter draw, run the orchestrator's
   ``run_components``, and hash the resulting ``sed_intrinsic`` byte
   buffer. Catches a future PR that silently changes runtime values
   despite the contract metadata staying unchanged.

Fixture regeneration: when an intentional change shifts a snapshot,
run

::

    TENGRI_REGENERATE_SNAPSHOTS=1 pytest tests/integration/test_derived_contract_snapshots.py

once locally (with SSP data on disk) and commit the updated JSON.

Closes issue #22.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.core import PipelineState
from tengri.forward.component_factory import build_components
from tengri.forward.orchestrator import run_components, sample_params_dict

_SNAPSHOT_FILE = Path(__file__).parent / "_snapshots" / "derived_contract_v1.json"
_REGENERATE = os.environ.get("TENGRI_REGENERATE_SNAPSHOTS") == "1"


# Canonical pipeline configurations that exercise the full contract
# surface: stellar + nebular always; dust two-component on; AGN /
# radio / xray / IGM toggled to vary which alternates fire.
_CONFIGS: dict[str, dict] = {
    "minimal_phot": {
        "sfh_model": "tsnorm",
        "nebular_backend": "baked_in",
        "use_dust": True,
        "use_radio": False,
        "use_xray": False,
        "use_igm": False,
    },
    "full_panchromatic": {
        "sfh_model": "tsnorm",
        "nebular_backend": "baked_in",
        "use_dust": True,
        "use_radio": True,
        "use_xray": True,
        "use_igm": True,
    },
    "single_component_dust": {
        "sfh_model": "tsnorm",
        "nebular_backend": "baked_in",
        "use_dust": True,
        "dust_model": "one_component",
        "use_radio": True,
        "use_xray": False,
        "use_igm": False,
    },
}


def _load_snapshot() -> dict:
    if not _SNAPSHOT_FILE.is_file():
        return {}
    return json.loads(_SNAPSHOT_FILE.read_text())


def _save_snapshot(data: dict) -> None:
    _SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_FILE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _contract_graph(components) -> list[tuple[str, str, str, str]]:
    """Canonicalised tuple list of every declared cross-component edge.

    Each entry is ``(component_class, role, key_name, units)`` where
    ``role`` is one of ``"publishes"``, ``"requires"``,
    ``"requires_optional"``. Sorted for deterministic hashing.
    """
    rows: list[tuple[str, str, str, str]] = []
    for c in components:
        cls = type(c).__name__
        for role in ("publishes", "requires", "requires_optional"):
            fn = getattr(c, role, None)
            if not callable(fn):
                continue
            for k in fn():
                rows.append((cls, role, k.name, k.units))
    return sorted(rows)


def _hash_graph(graph: list[tuple[str, str, str, str]]) -> str:
    payload = json.dumps(graph, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_array(arr) -> str:
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


# ── Part 1: contract-graph snapshot (no SSP required) ─────────────


@pytest.mark.parametrize("config_name", list(_CONFIGS.keys()))
def test_contract_graph_snapshot(config_name):
    """Hash the publishes/requires/requires_optional edges per config.

    Catches a silent annotation change (key rename, units drift,
    ownership move) without needing real SSP data — the contract is
    metadata, so the regression target is also metadata.
    """
    components = build_components(ssp_data=None, **_CONFIGS[config_name])
    graph = _contract_graph(components)
    digest = _hash_graph(graph)

    snapshot = _load_snapshot()
    key = f"contract_graph::{config_name}"

    if _REGENERATE or key not in snapshot:
        snapshot.setdefault("_meta", {})["regenerated_by"] = "TENGRI_REGENERATE_SNAPSHOTS=1"
        snapshot[key] = {
            "digest": digest,
            "rows": [list(r) for r in graph],
        }
        _save_snapshot(snapshot)
        if not _REGENERATE:
            pytest.fail(
                f"No baseline for {key!r} — wrote one. "
                f"Re-run after committing {_SNAPSHOT_FILE.name}."
            )
        return

    expected = snapshot[key]["digest"]
    assert digest == expected, (
        f"Contract graph for {config_name!r} drifted from baseline.\n"
        f"  Expected: {expected}\n"
        f"  Got:      {digest}\n"
        f"  If this change is intentional, regenerate the baseline:\n"
        f"    TENGRI_REGENERATE_SNAPSHOTS=1 pytest {Path(__file__).name}"
    )


# ── Part 2: SED-output snapshot (skipped when SSP absent) ─────────


@pytest.mark.parametrize("config_name", list(_CONFIGS.keys()))
def test_sed_output_snapshot(config_name, ssp_data_wne):
    """Hash sed_intrinsic from a forward pass on a fixed PRNG key.

    Catches a future PR that silently changes runtime values despite
    contract metadata staying unchanged — the bit-equivalence claim
    the ADR-0004 contract is supposed to preserve.

    Skips automatically when the SSP fixture file is absent (the
    fixture itself calls ``pytest.skip``).
    """
    components = build_components(ssp_data=ssp_data_wne, **_CONFIGS[config_name])

    # Deterministic param draw — independent of build_components' own
    # ordering so reshuffles don't invalidate the snapshot.
    key = jax.random.PRNGKey(0)
    params = sample_params_dict(components, key=key, overrides={"redshift": 0.05})

    wave = jnp.asarray(ssp_data_wne.ssp_wave)
    initial = PipelineState(wave=wave)
    final = run_components(components, initial, params)

    if final.sed_intrinsic is None:
        pytest.skip(f"Pipeline {config_name!r} did not produce sed_intrinsic")
    digest = _hash_array(final.sed_intrinsic)

    snapshot = _load_snapshot()
    sk = f"sed_intrinsic::{config_name}"

    if _REGENERATE or sk not in snapshot:
        snapshot.setdefault("_meta", {})["regenerated_by"] = "TENGRI_REGENERATE_SNAPSHOTS=1"
        snapshot[sk] = {"digest": digest}
        _save_snapshot(snapshot)
        if not _REGENERATE:
            pytest.fail(
                f"No baseline for {sk!r} — wrote one. "
                f"Re-run after committing {_SNAPSHOT_FILE.name}."
            )
        return

    expected = snapshot[sk]["digest"]
    assert digest == expected, (
        f"SED output for {config_name!r} drifted from baseline.\n"
        f"  Expected: {expected}\n"
        f"  Got:      {digest}\n"
        f"  If intentional, regenerate the baseline:\n"
        f"    TENGRI_REGENERATE_SNAPSHOTS=1 pytest {Path(__file__).name}"
    )
