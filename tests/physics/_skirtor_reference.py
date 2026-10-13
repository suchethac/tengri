# SPDX-License-Identifier: BSD-3-Clause
"""Independent references for the SKIRTOR published luminosities (#2851).

Two kinds of reference live here.

* ``emitted_pieces`` and ``cone_absorbed`` rebuild the pieces of the SED the component emits
  along a sightline on a wide dense grid from the raw library records (``RawSkirtor``), the one
  normalization scale written out from the native sphere mean, the CIGALE piecewise power-law
  disc shapes written out again, and the polar dust's closed forms (the Type-1 logistic, the
  absorbed fraction ``1 - exp(-tau)``). The only pieces taken from the implementation are the
  SMC extinction curve ``k(lambda)`` (a tabulated law, Pei 1992) and, for the ADAF disc
  (``disk_type`` 2), its shape.
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
#: IAU 2015 nominal solar luminosity [erg/s], the value of ``physics_constants.L_SUN``.
L_SUN_CGS = 3.828e33
#: Order of the library coordinates in a ``RawSkirtor`` point (the order of the file).
_POINT_KEYS = ("tau_skirtor", "p_skirtor", "q_skirtor", "oa_skirtor", "radius_ratio")
#: Breakpoints [nm], power-law indices and the index that ``delta`` shifts, for the CIGALE
#: piecewise discs (``disk_type`` 0 and 1).
_DISC_SHAPES = {
    0: ((8.0, 10.0, 100.0, 5000.0, 1e6), (0.2, -1.0, -1.5, -4.0), 2),
    1: ((8.0, 50.0, 125.0, 10000.0, 1e6), (1.0, -0.2, -1.5, -4.0), 2),
}


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


def loglog_power(wave, y, lower=None, upper=None):
    """Vectorized ``int y d(wave)`` of log-log interpolated ``y`` of shape ``(..., n_wave)``.

    A segment with a zero end is a trapezoid; ``lower`` and ``upper`` cut the integral along the
    power law inside a segment (the same rule as ``power_law_integral``).
    """
    y = np.asarray(y, dtype=float)
    lo = wave[0] if lower is None else lower
    hi = wave[-1] if upper is None else upper
    cuts = [c for c in (lo, hi) if wave[0] < c < wave[-1]]
    xs = np.unique(np.concatenate([wave, cuts]))
    idx = np.clip(np.searchsorted(wave, xs, side="right") - 1, 0, wave.size - 2)
    t = (np.log(xs) - np.log(wave[idx])) / (np.log(wave[idx + 1]) - np.log(wave[idx]))
    y0, y1 = y[..., idx], y[..., idx + 1]
    both = (y0 > 0) & (y1 > 0)
    ll = np.exp(np.log(np.where(both, y0, 1.0)) * (1 - t) + np.log(np.where(both, y1, 1.0)) * t)
    ys = np.where(both, ll, y0 + (y1 - y0) * (xs - wave[idx]) / (wave[idx + 1] - wave[idx]))
    keep = (xs >= lo) & (xs <= hi)
    xs, ys = xs[keep], ys[..., keep]
    x0, x1, a, b = xs[:-1], xs[1:], ys[..., :-1], ys[..., 1:]
    pos = (a > 0) & (b > 0)
    log_ratio = np.log(x1 / x0)
    u = log_ratio + np.log(np.where(pos, b, 1.0)) - np.log(np.where(pos, a, 1.0))
    small = np.abs(u) < 1e-8
    r = np.where(small, 1 + 0.5 * u, np.expm1(u) / np.where(small, 1.0, u))
    law = x0 * np.where(pos, a, 1.0) * log_ratio * r
    trap = 0.5 * (a + b) * (x1 - x0)
    return np.sum(np.where(pos, law, trap), axis=-1)


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
        self._mean_cache = {}
        self._record_cache = {}

    def record(self, key, tau, p, q, oa, radius, *, per_inclination_norm=False):
        """``(n_inc, n_wave)``: the library interpolated over the five parameter axes.

        One SciPy PCHIP per axis, in the order tau, p, q, opening angle, radius ratio.
        """
        cache_key = (key, per_inclination_norm, tau, p, q, oa, radius)
        if cache_key not in self._record_cache:
            cube = self.cubes[key]
            if per_inclination_norm:
                cube = cube / self.norm[..., None]
            for axis, value in zip(self.axes[:5], (tau, p, q, oa, radius), strict=True):
                cube = pchip_axis0(axis, cube, value)
            self._record_cache[cache_key] = cube
        return self._record_cache[cache_key]

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

    def native_total_power(self, params, cos_inc):
        """Escaping disc over the template's whole native range plus dust power [library units]."""
        return self.power("disk", params, cos_inc) + self.power("dust", params, cos_inc)

    def unemitted_power(self, params, cos_inc):
        """Escaping disc power outside the support of the emitted disc [library units]."""
        lo, hi = DISC_SUPPORT_AA
        return self.power("disk", params, cos_inc, upper=lo) + self.power(
            "disk", params, cos_inc, lower=hi
        )

    def scale(self, params):
        """The one factor [erg/s per library power] that sets the sphere-mean native budget.

        ``params`` holds ``log_lbol``, ``torus_frac`` and the five library coordinates; the
        factor makes the sphere mean of the native escaping disc plus dust equal
        ``torus_frac 10**log_lbol L_sun``.
        """
        point = tuple(float(params[k]) for k in _POINT_KEYS)
        if point not in self._mean_cache:
            by_inc = {k: self.record(k, *point) for k in ("disk", "dust")}

            def native_power(mu):
                return sum(
                    power_law_integral(self.wave, pchip_axis0(self.axes[-1], rec, mu))
                    for rec in by_inc.values()
                )

            self._mean_cache[point] = self.sphere_mean(native_power)
        return (
            float(params["torus_frac"]) * 10.0 ** float(params["log_lbol"]) * L_SUN_CGS
        ) / self._mean_cache[point]

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


def _loglog_resample(xq, x, y):
    """Power law between adjacent nodes (linear if a neighbor is zero); zero outside ``x``."""
    xq = np.asarray(xq, dtype=float)
    inside = (xq >= x[0]) & (xq <= x[-1])
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
    t = (np.log(xq) - np.log(x[i])) / (np.log(x[i + 1]) - np.log(x[i]))
    y0, y1 = y[i], y[i + 1]
    both = (y0 > 0.0) & (y1 > 0.0)
    ll = np.exp(np.log(np.where(both, y0, 1.0)) * (1 - t) + np.log(np.where(both, y1, 1.0)) * t)
    return np.where(inside, np.where(both, ll, y0 + (y1 - y0) * t), 0.0)


def _piecewise_shape(wave_aa, disk_type, delta):
    """Un-normalized CIGALE piecewise power-law disc (continuous), zero outside its limits.

    Any constant factor cancels in the re-tilt: the disc is renormalized to the template
    disc's power inside the support. ``delta`` shifts the third index as in CIGALE.
    """
    limits, coefs, delta_slot = _DISC_SHAPES[disk_type]
    coefs = list(coefs)
    coefs[delta_slot] = coefs[delta_slot] + delta
    lam = np.asarray(wave_aa, dtype=float) / 10.0
    seg = np.clip(np.searchsorted(limits, lam, side="right") - 1, 0, len(coefs) - 1)
    steps = [(coefs[k - 1] - coefs[k]) * np.log(limits[k]) for k in range(1, len(coefs))]
    log_c = np.concatenate([[0.0], np.cumsum(steps)])
    value = np.exp(np.array(coefs)[seg] * np.log(lam) + log_c[seg])
    return np.where((lam >= limits[0]) & (lam < limits[-1]), value, 0.0)


def _retilt(wave_aa, disk_type, delta):
    """CIGALE re-tilt (selected disc over the SKIRTOR disc at delta = 0) and the support mask."""
    ref = _piecewise_shape(wave_aa, 0, 0.0)
    if disk_type in _DISC_SHAPES:
        sel = _piecewise_shape(wave_aa, disk_type, delta)
    else:
        # The ADAF blend (disk_type 2) is the implementation's own function.
        from tengri.components.agn.disc_cigale import adaf_disk_spectrum

        sel = np.asarray(adaf_disk_spectrum(jnp.asarray(wave_aa) / 10.0, delta=delta))
    support = ref > 0.0
    return np.where(support, sel / np.where(support, ref, 1.0), 0.0), support


def emitted_pieces(raw, params, wave, disk_type=0):
    """Emitted SKIRTOR disc and dust SEDs and the polar-absorbed SED along ``cos_inc``.

    Built from the raw library records: the template disc and dust at ``cos_inc`` (PCHIP of the
    records, power law between nodes) times the one normalization scale, the disc re-tilted by
    the CIGALE shape ratio and renormalized to the template disc's power inside the support of
    the analytic disc, on ``wave``.

    Parameters
    ----------
    raw : RawSkirtor
        The library read from the file.
    params : mapping
        ``log_lbol``, ``torus_frac``, the five library coordinates, ``cos_inc``, ``delta`` and
        ``polar_ebv`` (floats).
    wave : array_like, shape (n_wave,)
        Dense wavelength grid [Angstrom].
    disk_type : int
        0 (SKIRTOR), 1 (Schartmann) or 2 (ADAF) disc.

    Returns
    -------
    dict
        ``disc``, ``dust`` (SEDs on ``wave``), ``disc_attenuated`` (the disc after the polar
        dust's Type-1-masked attenuation along this sightline), ``absorbed`` (the disc along
        this sightline times the polar dust's absorbed fraction) and ``absorbed_power`` (its
        integral [erg/s]).
    """
    from tengri.components.dust.attenuation import smc as smc_curve

    p = {k: float(v) for k, v in params.items()}
    point = tuple(p[k] for k in _POINT_KEYS)
    wave = np.asarray(wave, dtype=float)
    factor = raw.scale(p) * wave**2 / _C_AA
    lam = {
        k: _loglog_resample(wave, raw.wave, raw.l_lambda(k, point, p["cos_inc"]))
        for k in ("disk", "dust")
    }
    template_disc = lam["disk"] * factor
    retilt, support = _retilt(wave, disk_type, p["delta"])
    k = ln_nu_integral(template_disc * support, wave) / ln_nu_integral(
        template_disc * retilt * support, wave
    )
    disc = template_disc * retilt * k
    # Polar dust: tau = 0.921 E(B-V) R_V k(lambda), Pei's SMC curve (the implementation's
    # tabulated law) with R_V = 2.93; the Type-1 weight is the logistic of width 0.025 in cos i
    # about sin(oa).
    tau = 0.921 * p["polar_ebv"] * 2.93 * np.asarray(smc_curve(jnp.asarray(wave)))
    weight = type1_logistic(p["cos_inc"], p["oa_skirtor"])
    absorbed = disc * (-np.expm1(-tau))
    return {
        "disc": disc,
        "disc_attenuated": disc - weight * absorbed,
        "dust": lam["dust"] * factor,
        "absorbed": absorbed,
        "absorbed_power": ln_nu_integral(absorbed, wave),
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


def cone_absorbed(raw, params, wave, disk_type=0, order=20):
    """Power [erg/s] the polar dust absorbs from the disc light emitted into the polar cone.

    ``int_0^1 w(mu) A(mu) d(mu)`` with ``A(mu)`` the absorbed power of the disc emitted along
    ``mu`` and ``w`` the Type-1 logistic of the SED (``type1_logistic``): the share of the
    sphere mean that falls inside the polar cone (both lobes), by Gauss-Legendre of ``order``
    points on every interval between the template's inclination nodes.
    """
    mu, weight = interval_quadrature(raw.axes[-1], order)
    oa = float(params["oa_skirtor"])
    total = 0.0
    for m, wgt in zip(mu, weight, strict=True):
        piece = emitted_pieces(raw, {**params, "cos_inc": m}, wave, disk_type)
        total += wgt * type1_logistic(m, oa) * piece["absorbed_power"]
    return total
