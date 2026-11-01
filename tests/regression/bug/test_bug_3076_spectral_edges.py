# SPDX-License-Identifier: BSD-3-Clause
"""The energy ledger's grid has a node pair at every spectral edge of the selected blocks (#3076).

A fixed-grid trapezoid converges at first order across a step, with an error set by where the
step falls between two nodes: the round-1 ledger matched a dense integral of the multicolor
disc only to 6.6e-5 (its EUV tail ends at 30 A) and of a disc behind the SKIRTOR or Fritz
screen only to 1e-4 (the screen steps at 10 A and at each Morrison & McCammon absorption edge
below it). ``register_spectral_edges`` declares those steps per block, and ``ledger_wave`` puts
a node on each side of every one, so the integrals converge at second order.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks._ledger import ledger_wave
from tengri.components.agn.blocks._protocol import SPECTRAL_EDGE_BLOCKS
from tengri.components.agn.blocks.spectral_edges import block_edges
from tengri.components.xray.xray import _MM83_E_EDGES
from tengri.utils.physics_constants import HC_KEV_ANGSTROM

pytestmark = pytest.mark.regression_bug


def _selectors(disc="none", torus="none"):
    return {
        "agn_disc_block": disc,
        "agn_nlr_block": "none",
        "agn_blr_block": "none",
        "agn_feii_block": "none",
        "agn_torus_block": torus,
        "agn_attenuation_block": "none",
    }


@pytest.mark.parametrize(("category", "name"), sorted(SPECTRAL_EDGE_BLOCKS), ids=lambda v: str(v))
def test_every_registered_edge_is_a_finite_positive_wavelength(category, name):
    """Each registration returns host floats [A], or reports its library absent."""
    try:
        edges = SPECTRAL_EDGE_BLOCKS[(category, name)]()
    except FileNotFoundError as exc:
        pytest.skip(f"library data absent: {exc}")
    assert len(edges) > 0
    assert all(np.isfinite(float(e)) and float(e) > 0.0 for e in edges)


def test_screened_torus_carries_the_absorption_edges_below_its_library():
    """The SKIRTOR and Fritz screens step at every MM83 edge shortward of 10 A."""
    expected = HC_KEV_ANGSTROM / np.asarray(_MM83_E_EDGES, dtype=np.float64)
    expected = expected[expected < 10.0]
    for torus in ("skirtor", "fritz"):
        try:
            edges = np.asarray(block_edges(_selectors(torus=torus)))
        except FileNotFoundError as exc:
            pytest.skip(f"library data absent: {exc}")
        for e in expected:
            assert np.any(np.isclose(edges, e, rtol=1e-12)), (torus, e)


def test_edges_are_concrete_inside_a_trace():
    """A CIGALE disc builds its band with ``jnp``; the edges stay host values under ``jit``."""
    sel = _selectors(disc="skirtor")

    @jax.jit
    def grid(x):
        return ledger_wave(x.dtype, block_edges(sel)).shape[0] + 0 * x

    assert int(grid(jnp.asarray(0.0))) == ledger_wave(jnp.float64, block_edges(sel)).shape[0]
    assert 80.0 in block_edges(sel)


def test_ledger_grid_brackets_every_edge():
    """``ledger_wave`` has a node just below and just above each edge in its range."""
    sel = _selectors(disc="multicolor", torus="fritz")
    edges = np.asarray(block_edges(sel))
    wave = np.asarray(ledger_wave(jnp.float64, edges))
    assert np.all(np.diff(wave) > 0.0)
    for e in edges[(edges > wave[0]) & (edges < wave[-1])]:
        i = np.searchsorted(wave, e)
        assert wave[i - 1] == pytest.approx(e, rel=2e-9)
        assert wave[i] == pytest.approx(e, rel=2e-9)
