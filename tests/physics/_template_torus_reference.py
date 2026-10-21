# SPDX-License-Identifier: BSD-3-Clause
"""Independent reader of the template-torus libraries for the energy-ledger tests (#2851).

``RawTorus`` reads one library file with ``h5py`` alone (no tengri import, no tengri
interpolation): the stored records, their native wavelength nodes and the parameter axes. For
the Fritz (2006) grid it multiplies the ``dust`` records by their ``norm`` (the divisor CIGALE
took out of each model) and converts :math:`L_\\lambda` to :math:`L_\\nu`. It interpolates with
SciPy's ``PchipInterpolator`` one axis at a time, the inclination axis last, integrates the
log-log interpolant in closed form (``loglog_power``), and forms the sphere mean of the power
as a dense trapezoid in :math:`\\mu = \\cos i`, not the Gauss-Legendre quadrature the
component uses.
"""

from __future__ import annotations

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator

from tests.physics._skirtor_reference import loglog_power

_C_AA = 2.99792458e18
#: IAU 2015 nominal solar luminosity [erg/s], the value of ``physics_constants.L_SUN``.
L_SUN_CGS = 3.828e33
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
    """One template-torus library read from its file, on its stored common scale."""

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
            else:
                self.wave = np.asarray(g["wavelength"][()], dtype=float)
                records = np.asarray(g["template"][()], dtype=float)
            axes = [np.asarray(g[a][()], dtype=float) for a in axis_names]
        self.inc_index = axis_names.index(inc_name)
        # The inclination axis is moved last; it is interpolated after the others.
        self.records = np.moveaxis(records, self.inc_index, len(axes) - 1)
        self.inc_nodes = axes[self.inc_index]
        self.other_axes = [a for i, a in enumerate(axes) if i != self.inc_index]
        # Records are scaled by a common constant so the powers sit near 1 (they are stored
        # at 1e-55 to 1e44 depending on the library); the ratios are what the tests read.
        self.records = self.records / np.max(self.records)
        self._rows = {}

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

    def rows(self, point):
        """Records interpolated over the non-inclination axes at ``point`` (SciPy PCHIP)."""
        key = tuple(float(v) for v in point)
        if key not in self._rows:
            self._rows[key] = self._interpolate_rows(key)
        return self._rows[key]

    def _interpolate_rows(self, point):
        table = self.records
        for axis, value in zip(self.other_axes, point, strict=True):
            table = PchipInterpolator(axis, table, axis=0)(np.clip(value, axis[0], axis[-1]))
        return table  # (n_inc, n_wave)

    def row_at(self, point, mu):
        """The record at ``mu``: SciPy PCHIP in the library's inclination coordinate.

        The AGNfitter-rX libraries clamp at their end nodes; the Fritz lookup continues its
        end cubic (SciPy's ``extrapolate=True``), as the component does.
        """
        x, nodes = self.mu_to_x(mu)
        rows = self.rows(point)
        if self.name == "fritz":
            return PchipInterpolator(nodes, rows, axis=0, extrapolate=True)(x)
        if nodes[0] > nodes[-1]:
            nodes, rows = nodes[::-1], rows[::-1]
        return PchipInterpolator(nodes, rows, axis=0)(np.clip(x, nodes[0], nodes[-1]))

    def power(self, point, mu):
        """Power of the record along ``mu`` [common relative units]."""
        return power_of_lnu(self.wave, self.row_at(point, mu))

    def sphere_mean(self, point):
        """``int_0^1 P(mu) dmu`` by a dense trapezoid in ``mu``."""
        mu = np.linspace(0.0, 1.0, MU_POINTS)
        rows = self.row_at(point, mu)
        powers = power_of_lnu(self.wave, rows)
        return float(np.trapezoid(powers, mu))

    def node_power(self, idx_other, inc_index):
        """Power of one stored record (indices into the other axes, then inclination)."""
        return power_of_lnu(self.wave, self.records[(*idx_other, inc_index)])
