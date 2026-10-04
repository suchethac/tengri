# SPDX-License-Identifier: BSD-3-Clause
"""The closed-form line power of the analytic NLR/BLR/FeII blocks equals their grid integral.

The conserving ledger debits the line power from the disc. The analytic blocks sum
unit-integral Gaussians (and, for the BLR and FeII, a flux-conserving template), so the
power follows from the normalization alone; evaluating the blocks on a 30 001-node line
grid to integrate it cost 9.6e7 FLOP per gradient. These tests pin the closed forms to
the dense grid integral of the very blocks they replace.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import resolve_agn_block
from tengri.components.agn.blocks._protocol import LINE_ENERGY_BLOCKS
from tengri.components.agn.blocks.masking import split_lines_result

pytestmark = pytest.mark.limit

_WAVE = np.geomspace(900.0, 3.0e5, 120001)
_LOG_LBOL = 12.0
_L5100 = 3.0e44


def _grid_power(kind, name, **params):
    fn = resolve_agn_block(kind, name)
    out = fn(jnp.asarray(_WAVE), agn_log_lbol=_LOG_LBOL, l5100_disc=_L5100, **params)
    if kind != "feii":
        out = sum(split_lines_result(out))
    return float(np.trapezoid(np.asarray(out, float), _WAVE))


def _closed_power(kind, name, **params):
    return float(LINE_ENERGY_BLOCKS[(kind, name)](_LOG_LBOL, _L5100, **params))


@pytest.mark.parametrize("fwhm", [300.0, 500.0, 1500.0])
@pytest.mark.parametrize("cf, eff", [(0.1, 0.10), (0.3, 0.02)])
def test_nlr_analytic_power(fwhm, cf, eff):
    kw = dict(agn_nlr_fwhm_kms=fwhm, agn_nlr_cf=cf, agn_nlr_line_efficiency=eff)
    assert _closed_power("nlr", "analytic", **kw) == pytest.approx(
        _grid_power("nlr", "analytic", **kw), rel=1e-6
    )


@pytest.mark.parametrize("fwhm", [2500.0, 5000.0, 9000.0])
@pytest.mark.parametrize("fe2", [0.0, 1.0, 2.5])
def test_blr_analytic_power(fwhm, fe2):
    kw = dict(agn_blr_fwhm_kms=fwhm, agn_fe2_strength=fe2, agn_blr_cf=0.2)
    assert _closed_power("blr", "analytic", **kw) == pytest.approx(
        _grid_power("blr", "analytic", **kw), rel=1e-6
    )


@pytest.mark.parametrize("fwhm", [2500.0, 5000.0, 9000.0])
def test_feii_boroson_green_power(fwhm):
    kw = dict(agn_blr_fwhm_kms=fwhm, agn_fe2_strength=1.7, agn_blr_cf=0.15)
    assert _closed_power("feii", "boroson_green", **kw) == pytest.approx(
        _grid_power("feii", "boroson_green", **kw), rel=1e-6
    )


def test_power_gradient_is_finite():
    import jax

    g = jax.grad(
        lambda s: LINE_ENERGY_BLOCKS[("blr", "analytic")](_LOG_LBOL, _L5100, agn_fe2_strength=s)
    )(jnp.asarray(1.0))
    assert np.isfinite(float(g)) and float(g) > 0.0
