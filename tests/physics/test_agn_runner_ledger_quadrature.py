# SPDX-License-Identifier: BSD-3-Clause
"""The runner's fixed budget integrals equal dense integrals of the same functions.

The composable AGN runner measures its bolometric budgets on fixed grids so they do not
move with the caller's wavelength array. Each shortcut that keeps that cheap is pinned to
a dense reference here: the SKIRTOR torus power as the exact log-log integral on the
library's nodes, the compact Kubota & Done budget grid, the static switches that skip a
branch, and the traced selection of the polar reference.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import resolve_agn_block
from tengri.components.agn.blocks._protocol import collect_block_templates
from tengri.components.agn.blocks.runner import _KUBOTA_LEDGER_WAVE, compose_l_nu
from tengri.utils.grid_interp import loglog_integral

pytestmark = pytest.mark.limit

_W = jnp.asarray(np.geomspace(912.0, 1.0e8, 300))


def test_skirtor_torus_power_on_library_nodes_equals_the_dense_integral():
    tmpl = collect_block_templates({"torus": "skirtor"})["torus/skirtor"]
    axis = jnp.asarray(tmpl.torus.wave_grid)
    torus = resolve_agn_block("torus", "skirtor")
    kw = dict(agn_log_lbol=11.0, l5100_disc=1e43, templates=tmpl)
    exact = float(loglog_integral(axis, torus(axis, **kw)))
    dense_w = jnp.asarray(np.geomspace(float(axis[0]), float(axis[-1]), 400001))
    dense = float(jnp.trapezoid(torus(dense_w, **kw), dense_w))
    assert exact == pytest.approx(dense, rel=1e-7)


@pytest.mark.parametrize(
    "log_mbh, log_lbol, spin, f_hard",
    list(itertools.product((6.0, 8.0, 10.0), (11.0,), (0.0, 0.9), (0.02, 0.1))),
)
def test_compact_kubota_ledger_matches_a_dense_integral(log_mbh, log_lbol, spin, f_hard):
    disc = resolve_agn_block("disc", "kubota_done")
    kw = dict(
        agn_log_lbol=log_lbol,
        agn_log_mbh=log_mbh,
        agn_a_spin=spin,
        agn_f_hard=f_hard,
        templates=None,
    )
    w = jnp.asarray(_KUBOTA_LEDGER_WAVE)
    dense_w = jnp.asarray(np.geomspace(1.0e-3, 1.0e10, 400001))
    got = float(jnp.trapezoid(disc(w, **kw), w))
    ref = float(jnp.trapezoid(disc(dense_w, **kw), dense_w))
    assert got == pytest.approx(ref, rel=2e-6)


def _compose(**kw):
    base = dict(
        agn_disc_block="powerlaw",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="none",
        agn_attenuation_block="none",
        agn_norm="conserving",
        agn_torus_frac=0.3,
        return_components=True,
    )
    base.update(kw)
    return compose_l_nu(_W, 11.0, **base)[1]


def test_no_line_block_leaves_the_disc_at_one_minus_the_torus_fraction():
    """With every line stage off the debit is zero: the disc is exactly ``(1 - f)`` of itself."""
    ts = collect_block_templates({"torus": "skirtor"})
    debited = np.asarray(_compose(agn_torus_block="skirtor", template_state=ts)["disc"])
    full = np.asarray(
        _compose(agn_torus_block="skirtor", agn_norm="independent", template_state=ts)["disc"]
    )
    np.testing.assert_allclose(debited, 0.7 * full, rtol=1e-13)


def test_concrete_zero_fe2_strength_equals_a_traced_zero():
    from tengri.components.agn.blocks.blr import blr_analytic_block

    w = _W
    static = blr_analytic_block(w, 11.0, jnp.asarray(1.0e43), agn_fe2_strength=0.0)
    traced = jax.jit(
        lambda s: blr_analytic_block(w, 11.0, jnp.asarray(1.0e43), agn_fe2_strength=s)
    )(jnp.asarray(0.0))
    np.testing.assert_allclose(np.asarray(static), np.asarray(traced), rtol=1e-12)


def test_traced_polar_reference_selection_matches_the_concrete_one():
    """``lax.cond`` on a traced ``agn_ir_frac`` returns what the Python branch returns."""
    ts = collect_block_templates({"disc": "powerlaw", "torus": "skirtor"})

    def total(frac):
        comps = _compose(
            agn_torus_block="skirtor",
            agn_attenuation_block="polar_dust",
            agn_polar_ebv=0.2,
            agn_norm="cigale_joint",
            agn_ir_frac=frac,
            template_state=ts,
        )
        return jnp.stack([jnp.sum(comps["polar"]), jnp.sum(comps["disc"])])

    traced = jax.jit(total)
    for frac in (0.0, 0.1):
        np.testing.assert_allclose(
            np.asarray(traced(jnp.asarray(frac))), np.asarray(total(frac)), rtol=1e-12
        )
    assert float(total(0.1)[0]) != pytest.approx(float(total(0.0)[0]), rel=1e-3)
