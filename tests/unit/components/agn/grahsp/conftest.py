"""Skip all GRAHSP tests in this directory when the template bundle is absent.

The full GRAHSP suite needs ``data/grahsp/grahsp_templates.h5`` which is not
shipped in CI (~MB-scale HDF5 generated from upstream Polletta+ templates by
``tools/build_grahsp_hdf5.py``). Locally, build the bundle once and the
tests light up automatically.
"""

from __future__ import annotations

import pytest
from tests._data_skip import GRAHSP_TEMPLATE

if not GRAHSP_TEMPLATE.is_file():
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(config, items):  # pragma: no cover - guard
    if GRAHSP_TEMPLATE.is_file():
        return
    skip_marker = pytest.mark.skip(reason=f"GRAHSP template bundle not found at {GRAHSP_TEMPLATE}")
    for item in items:
        item.add_marker(skip_marker)
