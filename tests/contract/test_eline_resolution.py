# SPDX-License-Identifier: BSD-3-Clause
"""The e-line design matrix uses the instrument's declared resolution.

``_build_eline_G_eff`` probed ``model._spectral_resolution`` — an attribute
nothing in the codebase has ever assigned — so its ``or 2000.0`` fallback
always won and ``Spectroscopy(resolution=...)`` never reached the emission
line profiles: an R = 1000 grating got lines built at R = 2000, half their
true instrumental width. Same failure class as ``ForwardModel._approx``
(#1222): a defensive default silently converting "attribute never existed"
into wrong physics.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.likelihood import _build_eline_G_eff, _eline_scalar_resolution
from tengri.observation.eline_marginalization import build_eline_design_matrix

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def _stub_model(resolution):
    return SimpleNamespace(
        wave_obs=jnp.linspace(4000.0, 7000.0, 500),
        observation=SimpleNamespace(spectroscopy=SimpleNamespace(resolution=resolution)),
    )


def test_scalar_resolution_passes_through():
    assert _eline_scalar_resolution(_stub_model(1000.0)) == 1000.0


def test_array_resolution_reduces_to_median():
    model = _stub_model(np.array([800.0, 1000.0, 1200.0]))
    assert _eline_scalar_resolution(model) == 1000.0


def test_missing_resolution_falls_back_to_2000():
    assert _eline_scalar_resolution(_stub_model(None)) == 2000.0
    assert _eline_scalar_resolution(SimpleNamespace()) == 2000.0


def test_design_matrix_is_built_at_the_declared_resolution():
    """End-to-end through ``_build_eline_G_eff``: R = 500 lines are R = 500 wide."""
    model = _stub_model(500.0)
    lines = jnp.array([4862.68, 6564.61])  # Hbeta, Halpha (vacuum)
    eye = jnp.eye(lines.shape[0])
    params = {"redshift": 0.0}

    G = _build_eline_G_eff(params, {}, model, lines, eye)
    expected = build_eline_design_matrix(model.wave_obs, lines, 500.0, 0.0)
    at_2000 = build_eline_design_matrix(model.wave_obs, lines, 2000.0, 0.0)

    assert jnp.allclose(G, expected), "design matrix ignored the declared resolution"
    assert not jnp.allclose(G, at_2000), (
        "R=500 and R=2000 matrices are indistinguishable — the test lost its teeth"
    )
