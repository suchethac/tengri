# SPDX-License-Identifier: BSD-3-Clause
r"""Independent reader of the template-torus libraries for the energy-ledger tests (#2851).

``RawTorus`` reads one library file with ``h5py`` alone (no tengri import, no tengri
interpolation): the stored records, their native wavelength nodes and the parameter axes. Each
record's power is its log-log interpolant integrated in closed form (``loglog_power``); for the
Fritz (2006) grid, whose stored ``dust`` records have unit integral, it is their ``norm`` (the
divisor CIGALE took out of each model). The powers are divided by
their model's trapezoid mean in :math:`\mu = \cos i` over the inclination nodes, then
interpolated with SciPy's ``PchipInterpolator`` one axis at a time, the inclination axis last;
the sphere mean of that profile is a dense trapezoid in :math:`\mu`, not the Gauss-Legendre
quadrature the component uses. The component emits ``f L_bol`` times this profile over its
sphere mean along each sightline.
"""

from __future__ import annotations

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator

from tests.physics._skirtor_reference import loglog_power

_C_AA = 2.99792458e18
#: Points of the reference trapezoid in ``mu`` over [0, 1].
MU_POINTS = 4001

#: Library -> (file, group, axis datasets in file order, inclination axis name, variable).
LIBRARIES = {
    "nenkova_agnfitter": (
        "nenkova_agnfitter_torus_grid.h5",
        "nenkova_agnfitter",
        ("incl_axis",),
        "incl_axis",
    ),
    "nenkova_agnfitter_2p": (
        "nenkova_agnfitter_2p_torus_grid.h5",
        "nenkova_agnfitter_2p",
        ("incl_axis", "oa_axis"),
        "incl_axis",
    ),
    "nenkova_agnfitter_3p": (
        "nenkova_agnfitter_3p_torus_grid.h5",
        "nenkova_agnfitter_3p",
        ("incl_axis", "oa_axis", "tv_axis"),
        "incl_axis",
    ),
    "skirtor_agnfitter_1p": (
        "skirtor_mean1p_torus_grid.h5",
        "skirtor_mean1p",
        ("incl_axis",),
        "incl_axis",
    ),
    "fritz": (
        "fritz2006_torus_grid.h5",
        "fritz2006",
        (
            "r_ratio_axis",
            "tau_axis",
            "beta_axis",
            "gamma_axis",
            "opening_angle_axis",
            "psy_axis",
        ),
        "psy_axis",
    ),
}


def power_of_lnu(wave, lnu):
    """``int L_nu dnu`` of a log-log interpolated ``L_nu(lambda)`` record [record units]."""
    return loglog_power(wave, np.asarray(lnu) * _C_AA / wave**2)


class RawTorus:
    """One template-torus library read from its file: its records' powers per sightline."""

    def __init__(self, name, path):
        _, group, axis_names, inc_name = LIBRARIES[name]
        self.name = name
        with h5py.File(path, "r") as f:
            g = f[group]
            if name == "fritz":
                self.wave = np.asarray(g["wavelength_aa"][()], dtype=float)
                dust = np.asarray(g["dust"][()], dtype=float)
                norm = np.asarray(g["norm"][()], dtype=float)
                records = dust * norm[..., None] * self.wave**2 / _C_AA
                # The record's power on the common scale is ``norm`` itself (CIGALE's divisor;
                # the stored ``dust`` has unit integral), the field the component reads.
                powers = norm
            else:
                self.wave = np.asarray(g["wavelength"][()], dtype=float)
                records = np.asarray(g["template"][()], dtype=float)
                powers = None
            axes = [np.asarray(g[a][()], dtype=float) for a in axis_names]
        self.inc_index = axis_names.index(inc_name)
        self.inc_nodes = axes[self.inc_index]
        self.other_axes = [a for i, a in enumerate(axes) if i != self.inc_index]
        # The inclination axis is moved last; it is interpolated after the others. Records
        # are scaled by one constant so the powers sit near 1 (they are stored at 1e-55 to
        # 1e44 depending on the library); only ratios within a model enter.
        records = np.moveaxis(records, self.inc_index, len(axes) - 1)
        self.records = records / np.max(records)
        if powers is None:
            powers = power_of_lnu(self.wave, self.records)  # (..., n_inc)
        else:
            powers = np.moveaxis(powers, self.inc_index, len(axes) - 1)
        mu_nodes = self.node_mu()
        mean = np.trapezoid(powers, mu_nodes, axis=-1) / (mu_nodes[-1] - mu_nodes[0])
        self.anisotropy = powers / mean[..., None]
        self._profiles = {}

    def node_mu(self):
        """``cos i`` at the stored inclination nodes."""
        if self.name == "fritz":
            return np.sin(np.radians(self.inc_nodes))
        return np.cos(np.radians(self.inc_nodes))

    def mu_to_x(self, mu):
        """The interpolation coordinate at ``mu = cos i``, with the matching node axis.

        The CLUMPY reductions are interpolated in ``cos i`` (nodes reordered ascending),
        ``SKIRTOR_mean_1p`` in the inclination [deg], Fritz in the elevation [deg].
        """
        mu = np.asarray(mu, dtype=float)
        if self.name == "fritz":
            return np.degrees(np.arcsin(mu)), self.inc_nodes
        if self.name.startswith("nenkova"):
            return mu, np.cos(np.radians(self.inc_nodes))
        return np.degrees(np.arccos(mu)), self.inc_nodes

    def profile(self, point):
        """The anisotropy at every inclination node, other axes at ``point`` (SciPy PCHIP)."""
        key = tuple(float(v) for v in point)
        if key not in self._profiles:
            table = self.anisotropy
            for axis, value in zip(self.other_axes, key, strict=True):
                table = PchipInterpolator(axis, table, axis=0)(np.clip(value, axis[0], axis[-1]))
            self._profiles[key] = table  # (n_inc,)
        return self._profiles[key]

    def power(self, point, mu):
        """The anisotropy along ``mu``: SciPy PCHIP in the library's inclination coordinate.

        The AGNfitter-rX libraries clamp at their end nodes; the Fritz lookup continues its
        end cubic (SciPy's ``extrapolate=True``), as the component does.
        """
        x, nodes = self.mu_to_x(mu)
        prof = self.profile(point)
        if self.name == "fritz":
            return PchipInterpolator(nodes, prof, extrapolate=True)(x)
        if nodes[0] > nodes[-1]:
            nodes, prof = nodes[::-1], prof[::-1]
        return PchipInterpolator(nodes, prof)(np.clip(x, nodes[0], nodes[-1]))

    def sphere_mean(self, point):
        """``int_0^1 a(mu) dmu`` by a dense trapezoid in ``mu``."""
        mu = np.linspace(0.0, 1.0, MU_POINTS)
        return float(np.trapezoid(self.power(point, mu), mu))

    def node_power(self, idx_other, inc_index):
        """Power of one stored record (indices into the other axes, then inclination)."""
        return power_of_lnu(self.wave, self.records[(*idx_other, inc_index)])
