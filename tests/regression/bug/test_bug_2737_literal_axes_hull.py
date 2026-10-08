# SPDX-License-Identifier: BSD-3-Clause
"""Default precompute axes span the literal range and the parameter's reach (#2737).

A literal default axis that does not reach a parameter's Fixed value or free prior is
clipped by the lookup, with zero gradient beyond its nodes. Each default axis is now the
hull of its literal range and the parameter's active support, at the literal's node density.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


class _Spec:
    """Minimal stand-in for ``Parameters``: the Fixed values and free priors only."""

    def __init__(self, fixed: dict | None = None, free: dict | None = None):
        self._fixed = dict(fixed or {})
        self._free = dict(free or {})

    @property
    def free_params(self):
        return list(self._free)

    def get_fixed_values(self):
        return dict(self._fixed)

    def get_distribution(self, name):
        lo, hi = self._free[name]
        return type("_Dist", (), {"bounds": (lo, hi)})()


# (module, parameter, literal default axis, Fixed value outside the literal range)
_ADAPTER_CASES = [
    (
        "tengri.components.agn.grahsp.precompute",
        "agn_grahsp_plslope",
        np.array([-2.5, -1.7, -1.0]),
        -2.9,
    ),
    (
        "tengri.components.agn.grahsp.precompute",
        "agn_grahsp_ebv",
        np.array([0.0, 0.05, 0.1, 0.3, 1.0]),
        1.2,
    ),
    (
        "tengri.components.agn.qsogen_precompute",
        "agn_ebv",
        np.array([0.0, 0.05, 0.1, 0.2, 0.3]),
        0.5,
    ),
    (
        "tengri.components.agn.kd_precompute",
        "agn_gamma_hard",
        np.linspace(1.4, 3.0, 20),
        3.3,
    ),
    (
        "tengri.components.agn.kd_precompute",
        "agn_kt_hot",
        np.geomspace(10.0, 500.0, 15),
        650.0,
    ),
    (
        "tengri.components.radio.radio_precompute",
        "radio_alpha_sf",
        np.linspace(0.5, 1.0, 8),
        1.4,
    ),
    (
        "tengri.components.radio.radio_precompute",
        "radio_alpha_ff",
        np.linspace(-0.2, 0.0, 6),
        -0.35,
    ),
    (
        "tengri.components.xray.xray_precompute",
        "xray_gamma_hmxb",
        np.linspace(1.7, 2.3, 5),
        2.5,
    ),
    (
        "tengri.components.xray.xray_precompute",
        "xray_gamma_lmxb",
        np.linspace(1.4, 1.9, 5),
        2.0,
    ),
    (
        "tengri.components.xray.xray_precompute",
        "xray_gamma_agn",
        np.linspace(1.5, 2.3, 6),
        2.5,
    ),
    (
        "tengri.components.xray.xray_precompute",
        "xray_alpha_irx",
        np.linspace(0.0, 0.6, 6),
        0.8,
    ),
]


@pytest.mark.parametrize(("module", "param", "literal", "fixed_value"), _ADAPTER_CASES)
def test_fixed_value_outside_literal_is_inside_default_axis(module, param, literal, fixed_value):
    mod = importlib.import_module(module)
    axis = mod._axis(param, None, literal, _Spec(fixed={param: fixed_value}))
    assert axis.min() <= fixed_value <= axis.max()
    assert axis.min() <= literal.min() and axis.max() >= literal.max()


@pytest.mark.parametrize(("module", "param", "literal", "fixed_value"), _ADAPTER_CASES)
def test_no_parameters_keeps_literal_axis(module, param, literal, fixed_value):
    mod = importlib.import_module(module)
    np.testing.assert_array_equal(mod._axis(param, None, literal, None), literal)


def test_hull_axis_without_support_is_the_literal():
    from tengri.forward.precompute.reach_axes import hull_axis

    literal = np.linspace(0.0, 1.0, 5)
    axis = hull_axis(literal, None)
    np.testing.assert_array_equal(axis, literal)
    assert axis is not literal


def test_hull_axis_inside_literal_is_the_literal_unchanged():
    from tengri.forward.precompute.reach_axes import hull_axis

    literal = np.array([-2.5, -1.7, -1.0])
    np.testing.assert_array_equal(hull_axis(literal, (-2.0, -1.2)), literal)


def test_hull_axis_widens_linearly_at_literal_density():
    from tengri.forward.precompute.reach_axes import hull_axis

    literal = np.linspace(0.0, 1.0, 5)
    axis = hull_axis(literal, (0.0, 2.0))
    assert axis[0] == pytest.approx(0.0)
    assert axis[-1] == pytest.approx(2.0)
    assert axis.size == 10
    assert np.allclose(np.diff(axis), np.diff(axis)[0])


def test_hull_axis_widens_below_the_literal():
    from tengri.forward.precompute.reach_axes import hull_axis

    literal = np.linspace(1.4, 3.0, 20)
    axis = hull_axis(literal, (1.0, 3.0))
    assert axis[0] == pytest.approx(1.0)
    assert axis[-1] == pytest.approx(3.0)
    assert axis.size == int(np.ceil(20 * 2.0 / 1.6))


def test_hull_axis_logarithmic_is_geometric_with_scaled_count():
    from tengri.forward.precompute.reach_axes import hull_axis

    literal = np.geomspace(1.0, 10.0, 4)
    axis = hull_axis(literal, (0.5, 10.0), log_axis=True)
    assert axis[0] == pytest.approx(0.5)
    assert axis[-1] == pytest.approx(10.0)
    assert axis.size == int(np.ceil(4 * np.log(20.0) / np.log(10.0)))
    ratios = axis[1:] / axis[:-1]
    assert np.allclose(ratios, ratios[0])


def test_hull_axis_logarithmic_rejects_non_positive_lower_bound():
    from tengri.forward.precompute.reach_axes import hull_axis

    with pytest.raises(ValueError, match="lo > 0"):
        hull_axis(np.geomspace(1.0, 10.0, 4), (0.0, 10.0), log_axis=True)


def test_hull_axis_rejects_degenerate_literal():
    from tengri.forward.precompute.reach_axes import hull_axis

    with pytest.raises(ValueError):
        hull_axis(np.array([1.0, 1.0]), (0.0, 2.0))
