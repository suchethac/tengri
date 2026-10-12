# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR conserves energy: the sphere carries the accretion power, polar dust included (#2851).

SKIRTOR's radiative transfer conserves energy. The disc light that escapes (direct plus
scattered) and the dust re-emission, integrated over the sphere, are the intrinsic power of the
central source, which is ``10**agn_log_lbol L_sun`` (the accretion power, #2678). The component
therefore scales the records ONCE per parameter point so that

.. math:: \\bar P_{\\rm disc} + \\bar P_{\\rm torus} = f\\,10^{\\log L_{\\rm bol}} L_\\odot,
          \\qquad \\bar P = \\int_0^1 P(\\mu)\\,d\\mu,\\quad \\mu = \\cos i,

with :math:`f` = ``torus_frac`` (the AGN share, which multiplies every piece). The polar dust
absorbs disc light emitted into the cone and re-emits it, so the escaping disc published beside
it is net of what the polar dust took and the three published powers add up to the same budget:

.. math:: \\bar P_{\\rm disc}^{\\rm esc} + \\bar P_{\\rm torus} + P_{\\rm polar}
          = f\\,10^{\\log L_{\\rm bol}} L_\\odot.

The references are independent of the component: ``RawSkirtor`` reads the file with ``h5py``,
interpolates with SciPy's PCHIP and integrates the power laws in closed form; the sphere sums
are Gauss-Legendre of order 48 per inclination interval.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data
from tengri.utils.physics_constants import C_AA, L_SUN
from tests.physics._agn_publication import PUB_COARSE, ln_nu_integral
from tests.physics._agn_published import build_component, declared_box
from tests.physics._skirtor_reference import (
    DISC_SUPPORT_AA,
    RawSkirtor,
    emitted_pieces,
)

pytestmark = pytest.mark.regression_bug

_RTOL = 1.0e-6
#: Node parameters of the library (tau, p, q, opening angle, radius ratio).
_NODE_POINT = (7.0, 1.0, 1.0, 40.0, 20.0)
_OFF_NODE_POINT = (6.1, 0.8, 1.2, 43.0, 17.0)
_PARAM_KEYS = ("tau_skirtor", "p_skirtor", "q_skirtor", "oa_skirtor", "radius_ratio")
_WIDE = np.unique(
    np.concatenate(
        [
            np.geomspace(10.0, 1.0e8, 30_000),
            [x for e in DISC_SUPPORT_AA for x in (e * (1 - 1e-9), e, e * (1 + 1e-9))],
        ]
    )
)


@pytest.fixture(scope="module")
def skirtor():
    comp = build_component("skirtor")
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in declared_box(comp).items()}
    return comp, mid


@pytest.fixture(scope="module")
def raw_library():
    path = find_data("skirtor_templates_v3.h5")
    if path is None:
        pytest.skip("SKIRTOR v3 templates unavailable")
    return RawSkirtor(str(path))


def _published(comp, params, wave=PUB_COARSE):
    w = jnp.asarray(wave)
    _, pub = comp.predict(
        {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}, jnp.zeros_like(w), w
    )
    return {k: float(v) for k, v in pub.items()}


def _ledger(pub):
    """Escaping disc, torus and polar re-emission [erg/s] from the published keys."""
    return np.array([10.0 ** pub[f"log_L_agn_{k}"] for k in ("disc", "torus", "polar_dust")])


@pytest.mark.parametrize("polar_ebv", [0.0, 0.2])
@pytest.mark.parametrize("delta", [0.0, 0.6])
@pytest.mark.parametrize("cos_inc", [0.05, 0.4, 0.9])
@pytest.mark.parametrize("frac", [1.0, 0.4])
def test_published_sphere_powers_add_up_to_the_accretion_power(
    skirtor, frac, cos_inc, delta, polar_ebv
):
    """Escaping disc + torus + polar dust = ``torus_frac 10**agn_log_lbol L_sun`` (1e-6)."""
    comp, mid = skirtor
    pub = _published(
        comp,
        {**mid, "torus_frac": frac, "cos_inc": cos_inc, "delta": delta, "polar_ebv": polar_ebv},
    )
    expected = frac * 10.0 ** mid["log_lbol"] * L_SUN
    err = abs(_ledger(pub).sum() / expected - 1.0)
    assert err < _RTOL, f"sum of the sphere powers / (f L) - 1 = {err:.2e}"


def test_polar_dust_takes_its_power_from_the_disc(skirtor):
    """Switching the polar dust on moves power from the disc to the polar term, nothing else.

    The torus power is the same, the polar power is what the escaping disc lost, and the loss
    is a fraction of the disc that the dust opacity bounds.
    """
    comp, mid = skirtor
    base = {**mid, "cos_inc": 0.4, "delta": 0.0}
    clear = _ledger(_published(comp, {**base, "polar_ebv": 0.0}))
    dusty = _ledger(_published(comp, {**base, "polar_ebv": 0.3}))
    assert clear[2] == 0.0
    assert dusty[2] > 0.0
    assert abs(dusty[1] / clear[1] - 1.0) < 1e-12
    assert abs((clear[0] - dusty[0]) / dusty[2] - 1.0) < 1e-9
    assert dusty[0] > 0.0


@pytest.mark.parametrize("delta", [0.0, 0.6])
def test_sphere_integral_of_the_emitted_pieces_is_the_accretion_power(skirtor, delta):
    """The dense-grid power of the emitted SEDs, summed over the sphere, is ``f L`` (1e-6).

    Independent of the published keys and of the template scale: the disc and dust SEDs are
    rebuilt from the component's template function on a wide grid at Gauss-Legendre points of
    order 12 on every interval between the library's inclination nodes, and integrated with
    the ln(nu) trapezoid.
    """
    comp, mid = skirtor
    params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in {**mid, "delta": delta}.items()}
    nodes = np.asarray(comp.data.native_cos_inc)
    edges = np.concatenate([[0.0], nodes[1:]])
    x, w = np.polynomial.legendre.leggauss(12)
    total = 0.0
    for a, b in itertools.pairwise(edges):
        for xi, wi in zip(x, w, strict=True):
            mu = 0.5 * (b - a) * xi + 0.5 * (a + b)
            piece = emitted_pieces(comp, {**params, "cos_inc": mu}, _WIDE)
            power = ln_nu_integral(piece["disc"], _WIDE) + ln_nu_integral(piece["dust"], _WIDE)
            total += 0.5 * (b - a) * wi * power
    expected = float(params["torus_frac"]) * 10.0 ** mid["log_lbol"] * L_SUN
    assert abs(total / expected - 1.0) < _RTOL


def _node_params(point, extra=None):
    return {**dict(zip(_PARAM_KEYS, point, strict=True)), **(extra or {})}


@pytest.mark.parametrize("point", [_NODE_POINT, _OFF_NODE_POINT], ids=["on_nodes", "off_nodes"])
def test_templates_at_inclination_nodes_are_the_raw_records_times_one_scale(
    skirtor, raw_library, point
):
    """At each inclination node the disc and dust equal the raw records times ONE factor (1e-12).

    The factor is the same at every wavelength, for the disc and the dust, and at every node:
    the interpolation is node-exact (PCHIP) and no inclination is rescaled on its own. Off the
    parameter nodes the reference interpolates the five parameter axes with SciPy's PCHIP in the
    same order. The factor equals ``f L`` over the raw sphere mean of the escaping disc plus
    the dust (order 48), so the library's energy is the accretion power.
    """
    comp, mid = skirtor
    nodes = np.asarray(comp.data.native_cos_inc)
    wave = np.asarray(comp.data.native_wave)
    params = {**mid, **_node_params(point), "torus_frac": 1.0}
    template = jax.jit(comp.data)
    scales = []
    for key in ("disk", "dust"):
        ref_lam = raw_library.record(key, *point)  # (n_inc, n_wave)
        for k, mu in enumerate(nodes):
            out = template(
                wavelength=jnp.asarray(wave),
                agn_log_lbol=params["log_lbol"],
                agn_tau_skirtor=point[0],
                agn_p_skirtor=point[1],
                agn_q_skirtor=point[2],
                agn_oa_skirtor=point[3],
                agn_radius_ratio=point[4],
                agn_cos_inc=float(mu),
                frac_agn=1.0,
            )
            lnu = np.asarray(out.disk if key == "disk" else out.dust)
            ref_lnu = ref_lam[k] * wave**2 / C_AA
            keep = ref_lnu > 1e-30 * ref_lnu.max()
            ratio = lnu[keep] / ref_lnu[keep]
            scales.append(ratio)
    flat = np.concatenate(scales)
    assert np.max(np.abs(flat / flat[0] - 1.0)) < 1e-12
    mean_total = raw_library.sphere_mean(lambda m: raw_library.total_power(point, m))
    expected = 10.0 ** params["log_lbol"] * L_SUN / mean_total
    assert abs(flat[0] / expected - 1.0) < 1e-9
