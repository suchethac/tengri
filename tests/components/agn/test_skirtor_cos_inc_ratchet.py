# SPDX-License-Identifier: BSD-3-Clause
"""Ratchet tests for the SKIRTOR ``cos_inclination`` axis (#1911, class of #1851).

The v3 templates store ``cos_inclination`` **descending** (1 → 0), because the
inclination nodes ascend in angle.  Every interpolation helper requires strictly
ascending nodes, and ``compute_grid_weights`` derives its kernel bandwidth from
the *first* spacing -- which on the descending axis is the axis's **smallest**
spacing, and negative.  The result was a near-nearest-neighbor lookup on a
fittable parameter: 23/40 distinct outputs and 60% exactly-zero gradients
through the production SED path.

Two things fix it together, and the tests below pin both because neither is
sufficient on its own (measured against CIGALE's linear interpolation of the
same grid, RMS dex over the native template wavelengths, 27 between-node
samples):

===========================================  ========  =========  ==========
path                                          mean      worst      distinct
===========================================  ========  =========  ==========
descending + physical space (pre-#1911)       0.0399    0.5070     23/40
ascending  + physical space (reversal only)   0.0518    0.3926     40/40
descending + index space                      --        --          2/40
ascending  + index space (shipped)            0.0300    0.3232     40/40
===========================================  ========  =========  ==========

Reversal alone is *worse* than the status quo on parity (the physical-space
bandwidth is then the widest spacing, which smears the finely-spaced face-on
end); index space alone on a descending axis is catastrophic.

These tests drive the **production** entry points.  They must not reach into
``compute_grid_weights`` directly with their own flags: an earlier version of
this file did, and it stayed green while production ran the defective path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gradient]

_WAVE = jnp.geomspace(1e3, 1e6, 128)
# Interior grid point: every non-inclination coordinate sits on a node so the
# sweep isolates cos_inc.
_FIXED = dict(
    agn_log_lbol=11.0,
    agn_tau_skirtor=7.0,
    agn_p_skirtor=1.0,
    agn_q_skirtor=1.0,
    agn_oa_skirtor=40.0,
    agn_radius_ratio=20.0,
    agn_torus_frac=0.5,
)


def _skirtor_available() -> bool:
    from tengri.components.agn.skirtor import _find_skirtor_grid

    try:
        return _find_skirtor_grid() is not None
    except Exception:
        return False


requires_skirtor = pytest.mark.skipif(
    not _skirtor_available(), reason="SKIRTOR template grid not installed"
)


def _sum_sed(cos_inc):
    """Production SED path, reduced to a scalar objective."""
    from tengri.components.agn.skirtor import skirtor_sed

    return jnp.sum(skirtor_sed(_WAVE, agn_cos_inc=cos_inc, **_FIXED))


@requires_skirtor
def test_skirtor_cos_inc_sweep_is_interpolated_not_nearest_neighbor():
    """A 40-point cos_inc sweep through ``skirtor_sed`` yields 40 distinct SEDs.

    Pre-#1911 this gave 23 distinct values, with the repeats forming plateaus
    over the widely-spaced (edge-on) end of the axis.
    """
    sweep = np.linspace(0.0, 1.0, 40)
    values = np.array([float(_sum_sed(float(x))) for x in sweep])

    # Raw float64 comparison: the values span a factor ~1.7, so genuinely
    # distinct interpolated points differ far above rounding noise. Do NOT
    # round to a fixed number of decimals here -- the templates are normalized
    # to values below 1e-10, and rounding collapses every entry to 0.0.
    distinct = len(np.unique(values))
    assert distinct == 40, (
        f"cos_inc sweep gave {distinct}/40 distinct SEDs; repeats mean the axis is "
        "being read as a nearest-neighbor lookup again (#1911)."
    )


@requires_skirtor
def test_skirtor_cos_inc_gradient_is_never_exactly_zero():
    """d(SED)/d(cos_inc) is nonzero at every sweep point.

    Pre-#1911, 60% of the sweep returned an exactly-zero gradient, so a fit
    initialized on one of those plateaus received no signal at all.
    """
    grad_fn = jax.jit(jax.grad(_sum_sed))
    sweep = np.linspace(0.0, 1.0, 40)
    grads = np.array([float(grad_fn(float(x))) for x in sweep])

    zero_fraction = float(np.mean(grads == 0.0))
    assert zero_fraction == 0.0, (
        f"{zero_fraction:.1%} of cos_inc sweep points have an exactly-zero gradient; "
        "the inclination parameter is unfittable there (#1911)."
    )


@requires_skirtor
def test_every_skirtor_grid_consumer_sees_ascending_axes():
    """All SKIRTOR loaders must agree on axis orientation.

    The defect was fixed once, in ``_load_grid_arrays``, precisely so these
    cannot drift apart.  Reversing in only some consumers silently
    desynchronizes the runtime and precompute paths, which is what made the
    first migration attempt look like a physics change.
    """
    from tengri.components.agn import skirtor as sk

    def ascending(axes) -> bool:
        return all(
            np.asarray(a).size < 2 or bool(np.all(np.diff(np.asarray(a, dtype=float)) > 0))
            for a in axes
        )

    raw = sk._load_grid_arrays(sk._find_skirtor_grid())
    assert ascending(raw["axes"]), "_load_grid_arrays returned a descending axis"
    assert ascending(sk.load_skirtor_grid().axes), "load_skirtor_grid returned a descending axis"

    disc_dust = sk._load_raw_disk_dust_grid()
    assert disc_dust is not None, "probe setup failed: disc_dust was not published"
    assert ascending(disc_dust[3]), "_load_raw_disk_dust_grid returned a descending axis"

    atten = sk.load_skirtor_disc_atten_grid()
    assert atten is not None, "probe setup failed: atten was not published"
    assert ascending(atten.axes), "disc-attenuation bundle returned a descending axis"


@requires_skirtor
def test_skirtor_cos_inc_is_wired_to_index_space_interpolation():
    """The production call must pass ``index_space_interp=True``.

    This is the ratchet proper.  The two sweep tests above would still pass on
    a reversed-but-physical-space path (it also gives 40/40 distinct), so
    without this test the weaker half of the fix could be dropped unnoticed --
    and reversal alone measures *worse* against CIGALE than the pre-fix code.
    """
    from tengri.components.agn import skirtor as sk

    seen = []
    real = sk.interp_nd_triweight

    def spy(grid, axes, edges, point, *args, **kwargs):
        seen.append(kwargs.get("index_space_interp"))
        return real(grid, axes, edges, point, *args, **kwargs)

    sk.interp_nd_triweight = spy
    try:
        _sum_sed(0.42)
    finally:
        sk.interp_nd_triweight = real

    assert seen, "the production SED path made no triweight call"
    assert all(flag is True for flag in seen), (
        f"skirtor_sed reached interp_nd_triweight with index_space_interp={seen}; "
        "the non-uniform cos_inclination axis needs index-space interpolation (#1911)."
    )
