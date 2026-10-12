# SPDX-License-Identifier: BSD-3-Clause
"""Independent references for the SKIRTOR published luminosities (#2851).

Two kinds of reference live here.

* ``emitted_pieces`` and ``cone_absorbed`` rebuild the pieces of the SED the component emits
  along a sightline from the component's own template function, on a wide dense grid, with the
  re-tilt written out again. They check the published powers against the SED that is emitted.
* ``RawSkirtor`` reads ``data/skirtor_templates_v3.h5`` with ``h5py`` alone: no tengri import, no
  tengri interpolation. It multiplies the stored spectra by their ``norm`` records (the stored
  spectra are divided by their model's dust integral), interpolates them with SciPy's
  ``PchipInterpolator`` one axis at a time in the order of the file (tau, p, q, opening angle,
  radius ratio, then cos i), and integrates the resulting power laws analytically. It checks the
  relative flux between inclinations that the component publishes, and the energy the sphere
  carries.
"""

from __future__ import annotations

import itertools

import h5py
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator

from tests.physics._agn_publication import ln_nu_integral

_C_AA = 2.99792458e18


def _segment_integral(x0, x1, y0, y1):
    """Integral of the power law through ``(x0, y0)`` and ``(x1, y1)`` over ``[x0, x1]``."""
    s = np.log(y1 / y0) / np.log(x1 / x0)
    ratio = x1 / x0
    if abs(s + 1.0) < 1e-9:
        return y0 * x0 * np.log(ratio)
    return y0 * x0 / (s + 1.0) * (ratio ** (s + 1.0) - 1.0)


def _cut(x0, x1, y0, y1, at):
    """Value at ``at`` of the power law through two nodes (zero if either node is zero)."""
    if y0 > 0 < y1:
        return y0 * (at / x0) ** (np.log(y1 / y0) / np.log(x1 / x0))
    return 0.0


def power_law_integral(x, y, lower=None, upper=None):
    """Integral of a log-log interpolated ``y(x)``; trapezoid across a segment ending in zero.

    ``lower`` and ``upper`` cut the integral at those abscissae (inside a segment, along its
    power law).
    """
    total = 0.0
    for i in range(x.size - 1):
        x0, x1, y0, y1 = x[i], x[i + 1], y[i], y[i + 1]
        if lower is not None:
            if x1 <= lower:
                continue
            if x0 < lower:
                y0, x0 = _cut(x0, x1, y0, y1, lower), lower
        if upper is not None:
            if x0 >= upper:
                continue
            if x1 > upper:
                y1, x1 = _cut(x0, x1, y0, y1, upper), upper
        if y0 > 0.0 and y1 > 0.0:
            total += _segment_integral(x0, x1, y0, y1)
        else:
            total += 0.5 * (y0 + y1) * (x1 - x0)
    return total


#: Support of the emitted SKIRTOR disc [Angstrom]: 8 nm to 1e6 nm.
DISC_SUPPORT_AA = (80.0, 1.0e7)


def pchip_axis0(x, y, xq):
    """SciPy PCHIP along axis 0 of ``y`` at ``xq`` (the reference interpolant)."""
    return PchipInterpolator(x, y, axis=0)(xq)


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
        """``(n_inc, n_wave)``: the library interpolated over the five parameter axes.

        One SciPy PCHIP per axis, in the order tau, p, q, opening angle, radius ratio.
        """
        cube = self.cubes[key]
        if per_inclination_norm:
            cube = cube / self.norm[..., None]
        for axis, value in zip(self.axes[:5], (tau, p, q, oa, radius), strict=True):
            cube = pchip_axis0(axis, cube, value)
        return cube

    def l_lambda(self, key, params, cos_inc, *, per_inclination_norm=False):
        """``L_lambda`` [library units] along ``cos_inc``: PCHIP in cos i of the record."""
        by_inc = self.record(key, *params, per_inclination_norm=per_inclination_norm)
        return pchip_axis0(self.axes[-1], by_inc, cos_inc)

    def power(self, key, params, cos_inc, *, per_inclination_norm=False, lower=None, upper=None):
        """Power [library units] emitted along ``cos_inc`` (integral of ``L_lambda`` d lambda)."""
        lam = self.l_lambda(key, params, cos_inc, per_inclination_norm=per_inclination_norm)
        return power_law_integral(self.wave, lam, lower=lower, upper=upper)

    def total_power(self, params, cos_inc):
        """Escaping disc (inside the disc support) plus dust power [library units]."""
        lo, hi = DISC_SUPPORT_AA
        return self.power("disk", params, cos_inc, lower=lo, upper=hi) + self.power(
            "dust", params, cos_inc
        )

    def sphere_mean(self, power_fn, order=48):
        """``int_0^1 power_fn(mu) d(mu)``: Gauss-Legendre of ``order`` on each node interval."""
        nodes = self.axes[-1]
        edges = np.concatenate([[0.0], nodes[1:]]) if nodes[0] < 1e-9 else nodes
        x, w = np.polynomial.legendre.leggauss(order)
        total = 0.0
        for a, b in itertools.pairwise(edges):
            for xi, wi in zip(x, w, strict=True):
                total += 0.5 * (b - a) * wi * power_fn(0.5 * (b - a) * xi + 0.5 * (a + b))
        return total


def emitted_pieces(comp, params, wave, disk_type=0):
    """Emitted SKIRTOR disc and dust SEDs and the polar-absorbed SED along ``cos_inc``.

    The disc is re-tilted by the CIGALE shape ratio and renormalized by the ratio of two
    integrals over the analytic disc's support, on a wide grid, so it matches the component's
    publication-grid factor to the quadrature error of that grid.

    Returns
    -------
    dict
        ``disc``, ``dust`` (SEDs on ``wave``), ``disc_attenuated`` (the disc after the polar
        dust's Type-1-masked attenuation along this sightline), ``absorbed`` (the disc along
        this sightline times the polar dust's absorbed fraction) and ``absorbed_power`` (its
        integral [erg/s]).
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
    attenuated, l_abs = polar_dust_extinction(
        disc, w, p["cos_inc"], p["oa_skirtor"], p["polar_ebv"], law="smc"
    )
    return {
        "disc": np.asarray(disc) * p["torus_frac"],
        "disc_attenuated": np.asarray(attenuated) * p["torus_frac"],
        "dust": np.asarray(comps.dust) * p["torus_frac"],
        "absorbed": np.asarray(l_abs) * p["torus_frac"],
        "absorbed_power": ln_nu_integral(np.asarray(l_abs), wave) * p["torus_frac"],
    }


def interval_quadrature(cos_inc_nodes, order):
    """Gauss-Legendre ``(mu, weight)`` for ``int_0^1 d(mu)``, a panel per node interval."""
    nodes = np.asarray(cos_inc_nodes, dtype=float)
    edges = np.concatenate([[0.0], nodes[1:]]) if nodes[0] < 1e-9 else nodes
    x, w = np.polynomial.legendre.leggauss(order)
    mu, weight = [], []
    for a, b in itertools.pairwise(edges):
        mu.extend(0.5 * (b - a) * x + 0.5 * (a + b))
        weight.extend(0.5 * (b - a) * w)
    return np.array(mu), np.array(weight)


def type1_logistic(mu, oa_deg, width=0.025):
    """The Type-1 weight: the logistic in cos i of ``width`` about ``sin(oa)``."""
    return 1.0 / (1.0 + np.exp(-(mu - np.sin(np.radians(oa_deg))) / width))


def cone_absorbed(comp, params, wave, disk_type=0, order=20):
    """Power [erg/s] the polar dust absorbs from the disc light emitted into the polar cone.

    ``int_0^1 w(mu) A(mu) d(mu)`` with ``A(mu)`` the absorbed power of the disc emitted along
    ``mu`` and ``w`` the Type-1 logistic of the SED (``type1_logistic``): the share of the
    sphere mean that falls inside the polar cone (both lobes), by Gauss-Legendre of ``order``
    points on every interval between the template's inclination nodes.
    """
    mu, weight = interval_quadrature(comp.data.native_cos_inc, order)
    oa = float(params["oa_skirtor"])
    total = 0.0
    for m, wgt in zip(mu, weight, strict=True):
        piece = emitted_pieces(comp, {**params, "cos_inc": m}, wave, disk_type)
        total += wgt * type1_logistic(m, oa) * piece["absorbed_power"]
    return total
