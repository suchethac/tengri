# SPDX-License-Identifier: BSD-3-Clause
"""Regression: SKIRTOR torus grid-axis params are differentiable (#892).

``skirtor_disc_dust_ratio`` (the CIGALE-joint disc/dust normalization, active
for ``agn_norm='cigale_joint'`` — the default — with the SKIRTOR torus)
interpolates the raw SKIRTOR disk/dust grid with PCHIP. The dust template is
non-monotonic in wavelength, which tripped a ``0 * inf`` VJP trap in
``_pchip_slopes`` (fixed there): the returned ``R`` / ``R_faceon`` ratios had
NaN gradients w.r.t. ``agn_p/q/tau_skirtor``, so any gradient-based fit
(MAP/NUTS/VI) that freed the SKIRTOR torus geometry broke — even though the
forward SED was finite.

Data-gated: needs the raw SKIRTOR disk/dust grid (gitignored); skips in CI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def _skirtor_grid():
    from tengri.components.agn.skirtor import _load_raw_disk_dust_grid

    if _load_raw_disk_dust_grid() is None:
        pytest.skip("raw SKIRTOR disk/dust grid not available")


@pytest.mark.usefixtures("_skirtor_grid")
@pytest.mark.parametrize(
    "param", ["agn_p_skirtor", "agn_q_skirtor", "agn_tau_skirtor", "agn_oa_skirtor"]
)
@pytest.mark.parametrize("at_node", [True, False])
def test_skirtor_disc_dust_ratio_grad_finite(param, at_node):
    """R and R_faceon are differentiable w.r.t. each SKIRTOR grid-axis param,
    both at grid nodes and off-node."""
    from tengri.components.agn.blocks import resolve_agn_block
    from tengri.components.agn.skirtor import skirtor_disc_dust_ratio

    wave = jnp.logspace(2.5, 6.5, 300)
    disc = resolve_agn_block("disc", "multicolor")(wave, agn_log_lbol=12.0)
    ext = jnp.ones_like(wave)

    nodes = {
        "agn_p_skirtor": 1.0,
        "agn_q_skirtor": 1.0,
        "agn_tau_skirtor": 7.0,
        "agn_oa_skirtor": 40.0,
    }
    off = {
        "agn_p_skirtor": 0.7,
        "agn_q_skirtor": 0.7,
        "agn_tau_skirtor": 6.3,
        "agn_oa_skirtor": 42.5,
    }
    base = dict(nodes if at_node else off)

    def loss(v):
        kw = dict(base)
        kw[param] = v
        R, incl, R_faceon = skirtor_disc_dust_ratio(wave, disc, ext, agn_cos_inc=0.6, **kw)
        return R + jnp.sum(incl) + R_faceon

    g = float(jax.grad(loss)(base[param]))
    assert np.isfinite(g), (
        f"NaN/Inf gradient of skirtor_disc_dust_ratio w.r.t. {param} (at_node={at_node})"
    )
    assert np.any(g != 0.0), (
        "`g` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
