# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the ``agn_ebv_disc`` free parameter.

The Prévot (1984) SMC law is applied to the disc-component of every
registered disc+torus AGN model via :func:`tengri.components.agn.unified._redden_disc`.
These tests verify:

* ``agn_ebv_disc=0.0`` is a no-op (previous behavior preserved).
* Positive ``agn_ebv_disc`` reduces UV flux relative to IR flux (the
  Prévot law vanishes in the IR).
* Gradients flow through the new parameter.

Exercised via the Silva+04 model (simplest disc+torus path); the
underlying helper is shared by every other disc+torus registered model.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from tests._grad_parity import assert_grad_matches_fd

_GRID_PATH = Path(__file__).resolve().parents[4] / "data" / "silva04_torus_grid.h5"
_has_grid = _GRID_PATH.is_file()

pytestmark = [
    pytest.mark.bounds,
    pytest.mark.skipif(
        not _has_grid,
        reason="Silva+04 grid not built; AGN disc reddening test needs a disc+torus model.",
    ),
]


@pytest.fixture(scope="module")
def wavelength() -> jnp.ndarray:
    return jnp.geomspace(1e3, 1e6, 512)


@pytest.fixture(scope="module")
def model():
    from tengri.components.agn.unified import resolve_agn_model

    return resolve_agn_model("silva04")


def test_zero_ebv_is_noop(model, wavelength) -> None:
    base = model(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.1)
    with_zero = model(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.1, agn_ebv_disc=0.0)
    assert jnp.allclose(base, with_zero, rtol=1e-12, atol=0.0)


def test_positive_ebv_reduces_uv_disc_flux(model, wavelength) -> None:
    """The Prévot law has k(λ) > 0 in the UV; reddening must suppress L_ν there."""
    unreddened = model(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.1, agn_ebv_disc=0.0)
    reddened = model(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.1, agn_ebv_disc=0.3)
    uv_mask = wavelength < 3000.0
    assert bool(jnp.all(reddened[uv_mask] <= unreddened[uv_mask] + 1e-30))
    # Require a meaningful (not numerical-noise) reduction somewhere in the UV.
    rel_drop = float(1.0 - (reddened[uv_mask].sum() / unreddened[uv_mask].sum()))
    assert rel_drop > 0.01


def test_ir_flux_change_is_small_vs_uv(model, wavelength) -> None:
    """Reddening must move the UV much more than the IR.

    The torus dominates in the IR, so the disc reddening's fractional
    effect on the total SED must be small there relative to the UV,
    where the disc dominates.
    """
    unreddened = model(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.1, agn_ebv_disc=0.0)
    reddened = model(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.1, agn_ebv_disc=0.3)
    rel_change = jnp.abs(reddened - unreddened) / (unreddened + 1e-300)
    uv_mask = wavelength < 3000.0
    ir_mask = wavelength > 5e4
    uv_rel = float(rel_change[uv_mask].mean())
    ir_rel = float(rel_change[ir_mask].mean()) if bool(jnp.any(ir_mask)) else 0.0
    assert uv_rel > 10.0 * ir_rel + 1e-6, (
        f"UV fractional change ({uv_rel:.2e}) must dominate IR ({ir_rel:.2e})"
    )


def test_grad_flows_through_ebv(model, wavelength) -> None:
    def scalar_loss(ebv: float) -> float:
        sed = model(wavelength, agn_log_lbol=44.0, agn_lum_ratio=0.1, agn_ebv_disc=ebv)
        return jnp.log1p(jnp.sum(sed))

    g = assert_grad_matches_fd(scalar_loss, 0.1)
    assert jnp.isfinite(g)
    assert float(g) != 0.0
