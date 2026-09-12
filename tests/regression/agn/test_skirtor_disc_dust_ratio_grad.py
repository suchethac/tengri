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
    """Every field of the returned tie is differentiable w.r.t. each SKIRTOR
    grid-axis param, both at grid nodes and off-node.

    ``faceon_shape_native`` joined the return value when the polar-dust
    face-on reference moved onto the native grid (R60), and it is
    ``disc_n / trapezoid(disc_n, wave_grid)`` with ``disc_n`` resampled from
    the caller's grid -- a second path through the same PCHIP interpolant, so
    it is summed into the loss here rather than left unguarded.
    """
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
        tie = skirtor_disc_dust_ratio(wave, disc, ext, agn_cos_inc=0.6, **kw)
        return (
            tie.R
            + jnp.sum(tie.incl_ratio)
            + tie.R_faceon
            + jnp.sum(tie.faceon_shape_native)
            + jnp.sum(tie.wave_native)
        )

    g = float(jax.grad(loss)(base[param]))
    assert np.isfinite(g), (
        f"NaN/Inf gradient of skirtor_disc_dust_ratio w.r.t. {param} (at_node={at_node})"
    )
    assert np.any(g != 0.0), (
        "`g` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


class TestNoGridFallbackReturnsAUnitShape:
    """The no-grid fallback normalizes by the integral it measured, not a floor.

    When the raw SKIRTOR disk/dust grid is absent, ``skirtor_disc_dust_ratio``
    degrades to the caller's own grid and returns the disc renormalized to
    unit area as ``faceon_shape_native`` -- the array the polar dust's
    absorbed-power proxy integrates against. That renormalization used to
    divide by ``jnp.maximum(int, 1e-30)``. The floor is not inert: a disc
    whose integral falls below ``1e-30`` (a faint AGN, or simply a disc whose
    support barely overlaps the caller's grid) came back scaled by
    ``int / 1e-30`` instead of to unit area, so the polar reference silently
    lost the same factor. The denominator is now SELECTED -- zero integral
    gives a zero shape, anything positive is divided by itself -- which is
    also what keeps division's ``-num/den**2`` VJP off a squared floor.
    """

    @staticmethod
    def _tie(monkeypatch, disc_scale):
        import tengri.components.agn.skirtor as sk

        monkeypatch.setattr(sk, "_load_raw_disk_dust_grid", lambda *a, **k: None)
        wave = jnp.linspace(1000.0, 100000.0, 512)
        disc = jnp.full_like(wave, disc_scale)
        ext = jnp.ones_like(wave)
        return wave, sk.skirtor_disc_dust_ratio(wave, disc, ext, agn_cos_inc=0.6)

    def test_a_faint_disc_still_gets_a_unit_area_shape(self, monkeypatch):
        """Integral ~1e-35, well under the retired 1e-30 floor."""
        wave, tie = self._tie(monkeypatch, 1.0e-40)
        area = float(jnp.trapezoid(tie.faceon_shape_native, wave))
        assert area == pytest.approx(1.0, rel=1e-10, abs=0.0), (
            "the no-grid fallback must renormalize by the integral it measured; a "
            f"1e-30 floor leaves the shape scaled by int/1e-30 instead. area={area:.6e}"
        )

    def test_an_ordinary_disc_is_unchanged(self, monkeypatch):
        """The control: above the retired floor nothing about this moves."""
        wave, tie = self._tie(monkeypatch, 1.0)
        area = float(jnp.trapezoid(tie.faceon_shape_native, wave))
        assert area == pytest.approx(1.0, rel=1e-10, abs=0.0)

    def test_a_dead_disc_gives_a_zero_shape_not_a_spike(self, monkeypatch):
        """The documented degenerate value: nothing to normalize, so nothing."""
        _wave, tie = self._tie(monkeypatch, 0.0)
        shape = np.asarray(tie.faceon_shape_native)
        assert np.all(np.isfinite(shape)) and np.all(shape == 0.0), (
            f"a disc that integrates to zero has no shape to normalize; got {shape[:4]}"
        )
