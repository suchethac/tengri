# SPDX-License-Identifier: BSD-3-Clause
"""Test the jax-dependent half of the BMA model space (``bma_space``).

Pinned invariants:
- Every factorial sfh/attenuation value has a builder, and the builders cover
  exactly the axes ``_bma_keys`` enumerates.
- A factorial model builds (construction only, no fit).

Keys, axes, enumeration and weight-set membership are jax-free and pinned in
``test_bma_keys.py``; this module imports ``bma_space`` and so needs a JAX
backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER1 = Path(__file__).resolve().parent.parent
ANALYSIS = PAPER1.parent
for entry in [str(ANALYSIS), str(PAPER1)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pytest
from paper1 import _bma_keys as bk
from paper1.bma_space import _ATTENUATION_BUILDERS, _SFH_BUILDERS, build_model
from paper1.configs import load_ssp_for

from tengri import Observation, Photometry

pytestmark = pytest.mark.unit


def test_builders_cover_exactly_the_key_axes():
    """Builders and _bma_keys enumerate the same sfh and attenuation values."""
    assert tuple(_SFH_BUILDERS) == bk.SFH_TYPES
    assert tuple(_ATTENUATION_BUILDERS) == bk.ATTENUATION_TYPES


def test_build_model_factorial():
    """Build one factorial model (construction only, no JAX fit)."""
    model_dict = bk.enumerate_factorial()[0]
    ssp_data = load_ssp_for(bk.ssp_entry(model_dict["ssp"]).config)
    obs = Observation(photometry=Photometry.from_names(["hst_f814w"]))

    model = build_model(model_dict, ssp_data, obs, 1.0)
    assert model is not None
    assert len(model.spec.free_params) > 0


def test_xlike_import_error_raises():
    """A broken xlike_configs import raises instead of returning an empty dict."""
    from unittest.mock import patch

    from paper1.bma_evidence import _load_xlike_builders

    if not (PAPER1 / "xlike_configs.py").is_file():
        pytest.skip("xlike_configs.py not present")

    with (
        patch.dict(sys.modules, {"paper1.xlike_configs": None}),
        pytest.raises(ImportError, match="Failed to import xlike_configs"),
    ):
        _load_xlike_builders()
