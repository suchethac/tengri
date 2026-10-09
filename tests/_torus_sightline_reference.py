# SPDX-License-Identifier: BSD-3-Clause
r"""Reference line-of-sight transmission of a library torus, written from the library files.

The transmission of a SKIRTOR or Fritz (2006) torus is the library's normalized inclination
ratio :math:`R_n = {\rm disk}_i\,{\rm norm}_i/({\rm disk}_0\,{\rm norm}_0)` over the disc
anisotropy :math:`\eta` (:math:`\cos i(1 + 2\cos i)/3` for SKIRTOR, 1 for Fritz). The library
tables are read from the shipped files: node-exact PCHIP in their logarithms, a power law in
wavelength between library nodes, the edge value held beyond them. :math:`\eta` takes
:math:`\cos i` softly floored at :math:`\cos 85^\circ` (softplus width 0.01) so an isotropic
source stays finite edge-on. Below the library's 10 A edge the transmission is
:math:`\exp(-\sigma(E)N_H)` with the Morrison & McCammon (1983) cross-section and
:math:`N_H = 2.21\times10^{21}A_V` cm^-2 (Guver & Ozel 2009), with
:math:`A_V = -2.5\log_{10}T(5500\,\AA)` floored at zero.

The Morrison & McCammon (1983, ApJ 270, 119, Table 2) coefficients are transcribed from the two
public implementations that agree on all 42 coefficients and 15 edges: SOXS
(lynx-x-ray-observatory/soxs, ``soxs/spectra/foreground_absorption.py``) and cgm_toolkit
(ethlau/cgm_toolkit, ``cgm_toolkit/xray_emissivity.py``). jracusin/Judy-code ``wabs.pro`` prints
``c2 = -476.0`` for 0.1 to 0.284 keV, against -476.1 in both of those.
"""

from __future__ import annotations

import functools

import h5py
import jax.numpy as jnp
import numpy as np

from tengri.utils.grid_interp import interp_nd_pchip
from tests._data_skip import DATA_DIR

MM83_EDGES = np.array(
    [
        0.03,
        0.1,
        0.284,
        0.4,
        0.532,
        0.707,
        0.867,
        1.303,
        1.84,
        2.471,
        3.21,
        4.038,
        7.111,
        8.331,
        10.0,
    ]
)
MM83_C0 = np.array(
    [17.3, 34.6, 78.1, 71.4, 95.5, 308.9, 120.6, 141.3, 202.7, 342.7, 352.2, 433.9, 629.0, 701.2]
)
MM83_C1 = np.array(
    [608.1, 267.9, 18.8, 66.8, 145.8, -380.6, 169.3, 146.8, 104.7, 18.7, 18.7, -2.4, 30.9, 25.2]
)
MM83_C2 = np.array(
    [-2150.0, -476.1, 4.3, -51.4, -61.1, 294.0, -47.7, -31.5, -17.0, 0.0, 0.0, 0.75, 0.0, 0.0]
)
HC_KEV_AA = 12.39841984  # [keV A]
NH_PER_AV = 2.21e21  # [cm^-2 mag^-1]
V_WAVE = 5500.0  # [A]
_COS_FLOOR = 0.0871557427476582
_COS_FLOOR_WIDTH = 0.01
_TABLE_FLOOR = 1.0e-35
_RATIO_CLIP = 1.5


def sigma_mm83(energy_kev):
    """Photoelectric cross-section [cm^2]; the last segment extends beyond 10 keV."""
    e = np.asarray(energy_kev, dtype=float)
    k = np.clip(np.searchsorted(MM83_EDGES, e, side="right") - 1, 0, 13)
    return (MM83_C0[k] + MM83_C1[k] * e + MM83_C2[k] * e**2) / e**3 * 1e-24


@functools.cache
def _skirtor_tables():
    from tengri.components.agn.skirtor import _find_skirtor_grid, _load_grid_arrays

    raw = _load_grid_arrays(_find_skirtor_grid())
    return (
        tuple(jnp.asarray(a) for a in raw["axes"]),
        np.asarray(raw["wave"], dtype=float),
        jnp.asarray(np.log(np.maximum(np.asarray(raw["disk"], dtype=float), _TABLE_FLOOR))),
        jnp.asarray(np.log(np.maximum(np.asarray(raw["norm"], dtype=float), _TABLE_FLOOR))),
    )


@functools.cache
def _fritz_tables():
    with h5py.File(DATA_DIR / "fritz2006_torus_grid.h5", "r") as f:
        g = f["fritz2006"]
        axes = tuple(
            jnp.asarray(np.asarray(g[k + "_axis"][:], dtype=float))
            for k in ("r_ratio", "tau", "beta", "gamma", "opening_angle", "psy")
        )
        wave = np.asarray(g["wavelength_aa"][:], dtype=float)
        disk = np.asarray(g["disk"][:], dtype=float)
        norm = np.asarray(g["norm"][:], dtype=float)
    return (
        axes,
        wave,
        jnp.asarray(np.log(np.maximum(disk, _TABLE_FLOOR))),
        jnp.asarray(np.log(np.maximum(norm, _TABLE_FLOOR))),
    )


def _at(table, axes, point):
    return np.asarray(interp_nd_pchip(table, axes, tuple(jnp.asarray(v) for v in point)))


def _library_ratio(tables, point, point_face, wave_aa):
    """``(R_n, live)`` on ``wave_aa``, held at the library edges."""
    axes, wave_native, log_disk, log_norm = tables
    drift = (
        _at(log_disk, axes, point)
        - _at(log_disk, axes, point_face)
        + (_at(log_norm, axes, point) - _at(log_norm, axes, point_face))
    )
    live = (_at(log_disk, axes, point_face) > np.log(_TABLE_FLOOR) + 1.0).astype(float)
    held = np.log(np.clip(np.asarray(wave_aa, float), wave_native[0], wave_native[-1]))
    lx = np.log(wave_native)
    return (
        np.minimum(np.exp(np.interp(held, lx, drift)), _RATIO_CLIP),
        np.interp(held, lx, live) > 0.5,
    )


def _floored_cos(cos):
    return _COS_FLOOR + _COS_FLOOR_WIDTH * np.logaddexp(0.0, (cos - _COS_FLOOR) / _COS_FLOOR_WIDTH)


def _below_edge(wave_aa, edge, t_lib, t_v):
    w = np.asarray(wave_aa, float)
    below = w < edge
    if not below.any():
        return t_lib
    a_v = max(-2.5 * np.log10(max(t_v, 1e-30)), 0.0)
    t_x = np.exp(-sigma_mm83(HC_KEV_AA / np.where(below, w, 1.0)) * NH_PER_AV * a_v)
    return np.where(below, t_x, t_lib)


def skirtor_transmission(wave_aa, cos_inc, *, tau=7.0, p=1.0, q=1.0, oa=40.0, radius=20.0):
    """SKIRTOR ``T = R_n/eta`` (X-ray absorbed below the library edge) on ``wave_aa``."""
    tables = _skirtor_tables()

    def library(wave):
        geometry = (tau, p, q, oa, radius)
        ratio, live = _library_ratio(tables, (*geometry, cos_inc), (*geometry, 1.0), wave)
        c = _floored_cos(cos_inc)
        return np.where(live, ratio / (c * (1.0 + 2.0 * c) / 3.0), 1.0)

    t_lib = library(wave_aa)
    t_v = float(library(np.array([V_WAVE]))[0])
    return _below_edge(wave_aa, tables[1][0], t_lib, t_v)


def fritz_transmission(wave_aa, psi_deg, *, r_ratio=60.0, tau=1.0, beta=-0.5, gamma=4.0, oa=60.0):
    """Fritz ``T = R_n`` (X-ray absorbed below the library edge) on ``wave_aa``."""
    tables = _fritz_tables()
    psi_face = float(np.asarray(tables[0][5])[-1])

    def library(wave):
        geometry = (r_ratio, tau, beta, gamma, oa)
        ratio, live = _library_ratio(tables, (*geometry, psi_deg), (*geometry, psi_face), wave)
        return np.where(live, ratio, 1.0)

    t_lib = library(wave_aa)
    t_v = float(library(np.array([V_WAVE]))[0])
    return _below_edge(wave_aa, tables[1][0], t_lib, t_v)
