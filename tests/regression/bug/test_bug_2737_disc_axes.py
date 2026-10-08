# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #2737: the powerlaw_disc precompute table spans the parameter's reach.

A widened prior on ``agn_alpha`` must reach a table whose nodes cover it, and the table must
reproduce the exact closure there. The literal alpha axis held the edge value beyond [-2, 0]
with zero gradient.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Uniform
from tengri.components.agn import disc_precompute as adapter
from tengri.components.agn.disc import powerlaw_disc
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.regression_bug

_WAVE_AA = np.logspace(1, 5, 20000)
_BANDS_AA = ((1400.0, 1600.0), (4000.0, 5000.0), (8000.0, 9000.0))
_ACCURACY = 1e-2  # fractional, lookup/exact - 1 (measured <= 6.4e-3 in these bands)


def _tophat(lo, hi):
    inner = np.geomspace(lo, hi, 400)
    return inner, np.ones_like(inner)


_FILTERS = tuple(_tophat(*band) for band in _BANDS_AA)


def _band_average(l_nu):
    """Band-averaged L_nu over each filter, the convention the adapter's tables use."""
    out = []
    for fw, ft in _FILTERS:
        num = np.trapezoid(np.interp(fw, _WAVE_AA, l_nu) * ft / fw, fw)
        out.append(num / np.trapezoid(ft / fw, fw))
    return np.asarray(out)


def _powerlaw_parameters(lo, hi):
    return Parameters(
        agn_model="composable",
        agn_disc_block="powerlaw",
        agn_torus_block="none",
        agn_alpha=Uniform(lo, hi),
    )


def test_powerlaw_widened_alpha_table_equals_exact_closure_at_alpha_minus_2_7():
    """agn_alpha ~ Uniform(-3, 0.5): the table at alpha = -2.7 is the exact closure."""
    fw = [f[0] for f in _FILTERS]
    ft = [f[1] for f in _FILTERS]
    result = adapter.precompute(
        fw, ft, 0.0, _powerlaw_parameters(-3.0, 0.5), model="powerlaw_disc"
    )
    lookup = adapter.build_lookup(result, model="powerlaw_disc")
    log_lbol = 10.0
    exact = _band_average(
        np.asarray(powerlaw_disc(jnp.asarray(_WAVE_AA), agn_log_lbol=log_lbol, agn_alpha=-2.7))
    )
    got = np.asarray(lookup(log_lbol, -2.7))
    ratio = got / exact
    assert np.allclose(ratio, 1.0, rtol=_ACCURACY, atol=0.0), (
        f"powerlaw_disc at alpha=-2.7: lookup/exact = {ratio} (the table holds the edge value "
        f"beyond its alpha nodes)"
    )
