# SPDX-License-Identifier: BSD-3-Clause
"""Capture and load the band-table parity goldens for the six cases in ``_band_table_cases``.

A golden is two files per case under ``tests/fixtures/band_table_parity/``: ``<case>.json``
(the engagement summary, the SHA-256 of ``compile_signature()``, the structure of
``_template_data_for_jit()`` with a SHA-256 of the raw bytes of every array leaf, and the sampled
parameters, and the ``platform`` that captured it) and ``<case>.npz`` (the ``predict_photometry``
output at the sampled point, and every array leaf of ``_template_data_for_jit()`` under
``leaf_NNN``, indexed as the ``tree`` lists them). The record's digests are exact, dtype- and
shape-aware equalities on the capturing platform. Other platforms compare the stored values
within a tolerance, because XLA on x86-64 rounds the same program in the last bit.
"""

from __future__ import annotations

import hashlib
import json
import platform as _platform
import sys
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "band_table_parity"


def current_platform() -> str:
    """The platform string a golden records, such as ``darwin-arm64`` or ``linux-x86_64``."""
    return f"{sys.platform}-{_platform.machine()}"


def leaf_key(index: int) -> str:
    """The ``.npz`` key of the array leaf at position ``index`` of the ``tree`` list."""
    return f"leaf_{index:03d}"


def signature_sha256(model) -> str:
    """SHA-256 of the repr of ``model.compile_signature()``."""
    return hashlib.sha256(repr(model.compile_signature()).encode()).hexdigest()


def _flatten(tree, prefix: str = ""):
    """Yield ``(path, leaf)`` pairs for every leaf of a nested dict/list/tuple, in key order."""
    if isinstance(tree, dict):
        for key in tree:
            yield from _flatten(tree[key], f"{prefix}/{key}")
    elif isinstance(tree, (list, tuple)):
        for i, item in enumerate(tree):
            yield from _flatten(item, f"{prefix}/{i}")
    else:
        yield prefix, tree


def capture(model) -> tuple[dict, dict]:
    """Return the JSON-able record and the array payload that pin one model's outputs.

    Parameters
    ----------
    model : SEDModel
        A built model from one of the six cases.

    Returns
    -------
    record : dict
        ``platform``, ``summary``, ``signature_sha256``, ``tree`` (path, kind, dtype, shape or
        repr per leaf, with a digest of the bytes of each array leaf) and ``params`` (the
        sampled point).
    arrays : dict of ndarray
        ``photometry``, and each array leaf under ``leaf_NNN`` (its position in ``tree``).
    """
    import jax

    from tengri.forward.precompute_report import precompute_engagement_report

    params = model.spec.sample(jax.random.PRNGKey(1))
    photometry = np.asarray(model.predict_photometry(params))
    template = model._template_data_for_jit() or {}
    tree, arrays = [], {"photometry": photometry}
    for path, leaf in _flatten(template):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            value = np.ascontiguousarray(np.asarray(leaf))
            arrays[leaf_key(len(tree))] = value
            tree.append(
                {
                    "path": path,
                    "kind": "array",
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
            )
        else:
            tree.append({"path": path, "kind": "scalar", "repr": repr(leaf)})
    record = {
        "platform": current_platform(),
        "summary": precompute_engagement_report(model).summary_text(),
        "signature_sha256": signature_sha256(model),
        "tree": tree,
        "params": {k: np.asarray(v).tolist() for k, v in params.items()},
    }
    return record, arrays


def write_golden(name: str, model) -> None:
    """Write the ``<name>.json`` and ``<name>.npz`` golden for one case."""
    record, arrays = capture(model)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_DIR / f"{name}.npz", **arrays)
    with open(FIXTURE_DIR / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
        fh.write("\n")


def load_golden(name: str) -> tuple[dict, dict]:
    """Read the ``(record, arrays)`` golden pair written by :func:`write_golden`."""
    with open(FIXTURE_DIR / f"{name}.json", encoding="utf-8") as fh:
        record = json.load(fh)
    with np.load(FIXTURE_DIR / f"{name}.npz") as npz:
        arrays = {key: npz[key] for key in npz.files}
    return record, arrays
