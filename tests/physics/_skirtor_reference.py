# SPDX-License-Identifier: BSD-3-Clause
"""Independent references for the SKIRTOR published luminosities (#2851).

Two kinds of reference live here.

* ``emitted_pieces`` and ``cone_absorbed`` rebuild the pieces of the SED the component emits
  along a sightline from the component's own template function, on a wide dense grid, with the
  re-tilt written out again. They check the published powers against the SED that is emitted.
* ``RawSkirtor`` reads ``data/skirtor_templates_v3.h5`` with ``h5py`` alone: no tengri import, no
  tengri interpolation. It multiplies the stored spectra by their ``norm`` records (the stored
  spectra are divided by their model's dust integral), smooths them with the index-space
  triweight kernel the component interpolates with (re-derived here from the kernel's closed
  form), and integrates the resulting power laws analytically. It checks the relative flux
  between inclinations that the component publishes.
"""

from __future__ import annotations

import itertools

import h5py
import jax.numpy as jnp
import numpy as np

from tests.physics._agn_publication import ln_nu_integral

_C_AA = 2.99792458e18

#: Triweight kernel: density ``(35/96)(1 - z^2/9)^3`` on ``|z| < 3``; the bandwidth in index space
#: is half a cell.
_KERNEL_HALF_WIDTH = 3.0
_INDEX_BANDWIDTH = 0.5


def _kernel_cdf(z):
    """CDF of the triweight kernel, the integral of ``(35/96)(1 - z^2/9)^3`` from ``-3``."""
    zc = np.clip(z, -_KERNEL_HALF_WIDTH, _KERNEL_HALF_WIDTH)
    return 0.5 + (35.0 / 96.0) * (zc - zc**3 / 9.0 + zc**5 / 135.0 - zc**7 / 5103.0)


def triweight_weights(value, axis):
    """Index-space triweight weights of ``value`` on the ascending node array ``axis``."""
    axis = np.asarray(axis, dtype=float)
    n = axis.size
    lower = int(np.clip(np.searchsorted(axis, value) - 1, 0, n - 2))
    frac = (value - axis[lower]) / (axis[lower + 1] - axis[lower])
    u = lower + frac
    k = np.arange(n)
    # Cell k spans [k - 1/2, k + 1/2] in index space; the weight is the kernel mass inside it.
    w = _kernel_cdf((u - (k - 0.5)) / _INDEX_BANDWIDTH) - _kernel_cdf(
        (u - (k + 0.5)) / _INDEX_BANDWIDTH
    )
    return w / w.sum()


def _segment_integral(x0, x1, y0, y1):
    """Integral of the power law through ``(x0, y0)`` and ``(x1, y1)`` over ``[x0, x1]``."""
    s = np.log(y1 / y0) / np.log(x1 / x0)
    ratio = x1 / x0
    if abs(s + 1.0) < 1e-9:
        return y0 * x0 * np.log(ratio)
    return y0 * x0 / (s + 1.0) * (ratio ** (s + 1.0) - 1.0)


def power_law_integral(x, y, lower=None):
    """Integral of a log-log interpolated ``y(x)``; trapezoid across a segment ending in zero.

    ``lower`` cuts the integral at that abscissa (inside a segment, along its power law).
    """
    total = 0.0
    for i in range(x.size - 1):
        x0, x1, y0, y1 = x[i], x[i + 1], y[i], y[i + 1]
        if lower is not None:
            if x1 <= lower:
                continue
            if x0 < lower:
                y0 = (
                    y0 * (lower / x0) ** (np.log(y1 / y0) / np.log(x1 / x0))
                    if y0 > 0 < y1
                    else 0.0
                )
                x0 = lower
        if y0 > 0.0 and y1 > 0.0:
            total += _segment_integral(x0, x1, y0, y1)
        else:
            total += 0.5 * (y0 + y1) * (x1 - x0)
    return total


class RawSkirtor:
    """The SKIRTOR v3 library read from the file, on the radiative-transfer run's own scale."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.wave = f["wavelength"][:]
            axes = [
                f["grid/tau_97"][:],
                f["grid/p"][:],
                f["grid/q"][:],
                f["grid/opening_angle"][:],
                f["grid/radius_ratio"][:],
                f["grid/cos_inclination"][:],
            ]
            norm = f["spectra/norm"][:].astype(float)
            cubes = {
                "disk": f["spectra/disk_emission"][:].astype(float),
                "dust": f["spectra/dust_emission"][:].astype(float),
            }
        # The file lists cos(i) descending; the analysis wants it ascending.
        axes[-1] = axes[-1][::-1]
        norm = norm[..., ::-1]
        self.axes = axes
        self.norm = norm
        self.cubes = {k: v[..., ::-1, :] * norm[..., None] for k, v in cubes.items()}

    def record(self, key, tau, p, q, oa, radius, *, per_inclination_norm=False):
        """``(n_inc, n_wave)``: the library smoothed over the five parameter axes."""
        cube = self.cubes[key]
        if per_inclination_norm:
            cube = cube / self.norm[..., None]
        for axis, value in zip(self.axes[:5], (tau, p, q, oa, radius), strict=True):
            cube = np.tensordot(triweight_weights(value, axis), cube, axes=(0, 0))
        return cube

    def power(self, key, params, cos_inc, *, per_inclination_norm=False, lower=None):
        """Power [library units] emitted along ``cos_inc`` (integral of ``L_lambda`` d lambda)."""
        by_inc = self.record(key, *params, per_inclination_norm=per_inclination_norm)
        l_lambda = triweight_weights(cos_inc, self.axes[-1]) @ by_inc
        return power_law_integral(self.wave, l_lambda, lower=lower)


def emitted_pieces(comp, params, wave, disk_type=0):
    """Emitted SKIRTOR disc and dust SEDs and the polar-absorbed SED along ``cos_inc``.

    The disc is re-tilted by the CIGALE shape ratio and renormalized by the ratio of two
    integrals over the analytic disc's support, on a wide grid, so it matches the component's
    publication-grid factor to the quadrature error of that grid.

    Returns
    -------
    dict
        ``disc``, ``dust`` (SEDs on ``wave``), ``absorbed`` (the disc along this sightline times
        the polar dust's absorbed fraction) and ``absorbed_power`` (its integral [erg/s]).
    """
    from tengri.components.agn.disc_cigale import (
        adaf_disk_spectrum,
        schartmann2005_disk_spectrum,
        skirtor_disk_spectrum,
    )
    from tengri.components.agn.polar_dust import polar_dust_extinction

    p = {k: float(v) for k, v in params.items()}
    w = jnp.asarray(wave)
    comps = comp.data(
        wavelength=w,
        agn_log_lbol=p["log_lbol"],
        agn_tau_skirtor=p["tau_skirtor"],
        agn_p_skirtor=p["p_skirtor"],
        agn_q_skirtor=p["q_skirtor"],
        agn_oa_skirtor=p["oa_skirtor"],
        agn_radius_ratio=p["radius_ratio"],
        agn_cos_inc=p["cos_inc"],
        frac_agn=1.0,
    )
    wave_nm = w / 10.0
    shape_fn = (skirtor_disk_spectrum, schartmann2005_disk_spectrum, adaf_disk_spectrum)[disk_type]
    shape_sel = shape_fn(wave_nm, delta=p["delta"])
    shape_ref = skirtor_disk_spectrum(wave_nm, delta=0.0)
    retilt = shape_sel / jnp.maximum(shape_ref, 1e-100)
    support = (shape_ref > 0.0).astype(w.dtype)
    raw = comps.disk
    k = ln_nu_integral(np.asarray(raw * support), wave) / ln_nu_integral(
        np.asarray(raw * retilt * support), wave
    )
    disc = raw * retilt * k
    _, l_abs = polar_dust_extinction(
        disc, w, p["cos_inc"], p["oa_skirtor"], p["polar_ebv"], law="smc"
    )
    return {
        "disc": np.asarray(disc) * p["torus_frac"],
        "dust": np.asarray(comps.dust) * p["torus_frac"],
        "absorbed": np.asarray(l_abs) * p["torus_frac"],
        "absorbed_power": ln_nu_integral(np.asarray(l_abs), wave) * p["torus_frac"],
    }


def cone_intervals(cos_inc_nodes, oa_deg, order):
    """Gauss-Legendre ``(mu, weight)`` for ``int_{sin(oa)}^1 d(mu)``, a panel per node interval."""
    nodes = np.asarray(cos_inc_nodes, dtype=float)
    edges = np.concatenate([[0.0], nodes[1:]]) if nodes[0] < 1e-9 else nodes
    x, w = np.polynomial.legendre.leggauss(order)
    edge = np.sin(np.radians(oa_deg))
    mu, weight = [], []
    for a, b in itertools.pairwise(edges):
        lo = min(max(edge, a), b)
        half = 0.5 * (b - lo)
        mu.extend(half * x + lo + half)
        weight.extend(half * w)
    return np.array(mu), np.array(weight)


def cone_absorbed(comp, params, wave, disk_type=0, order=20):
    """Power [erg/s] the polar dust absorbs from the disc emitted into the cone.

    ``int_{sin(oa)}^1 A(mu) d(mu)`` with ``A(mu)`` the absorbed power of the disc emitted along
    ``mu``: the share of the sphere mean that falls inside the polar cone (both lobes), by
    Gauss-Legendre of ``order`` points on every interval between the template's inclination
    nodes, the lower limit raised to the cone edge.
    """
    mu, weight = cone_intervals(comp.data.native_cos_inc, float(params["oa_skirtor"]), order)
    total = 0.0
    for m, wgt in zip(mu, weight, strict=True):
        if wgt == 0.0:
            continue
        piece = emitted_pieces(comp, {**params, "cos_inc": m}, wave, disk_type)
        total += wgt * piece["absorbed_power"]
    return total
