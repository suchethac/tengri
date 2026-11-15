# SPDX-License-Identifier: BSD-3-Clause
"""Regression #3431: template-backed dust emission refuses out-of-grid values.

The DL07 kernel clips ``dust_qpah`` and ``dust_umin`` onto their grid edges
and returns a bit-identical SED for every excess value. The build boundary is
where that must be refused: a declared prior or a ``Fixed`` value outside the
template grid raises at build, naming the grid extent. ``dust_gamma_dl`` must
lie in [0, 1]; a negative value gives an all-zero SED through ``safe_ratio``.

These tests pin that boundary for DL07 and for every tabulated dust-emission
family registered in ``GRID_SUPPORT``. The in-kernel clip is kept only as
numerical protection and is not the thing under test.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.components.grid_support import GRID_SUPPORT, grid_support
from tengri.config.exceptions import ParameterError
from tengri.parameters import parse_groups
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.regression_bug

_ATTENUATION = {"law": "power_law", "type": "two_component"}


def _build(dust_emission: dict):
    """Parse a spec with the given dust-emission group, silencing advisories."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_groups(
            sfh={"type": "dpl"},
            redshift=0.1,
            dust_attenuation=_ATTENUATION,
            dust_emission=dust_emission,
        )


def test_fixed_qpah_above_dl07_grid_is_refused_at_build():
    with pytest.raises(ParameterError) as info:
        _build({"type": "dl07", "qpah": Fixed(6.0)})
    message = str(info.value)
    assert "dust_qpah" in message
    assert "[0.47, 4.58]" in message


def test_uniform_qpah_overhanging_dl07_grid_is_refused_at_build():
    with pytest.raises(ParameterError) as info:
        _build({"type": "dl07", "qpah": Uniform(0.0, 5.0)})
    assert "dust_qpah" in str(info.value)
    assert "[0.47, 4.58]" in str(info.value)


def test_fixed_umin_above_dl07_grid_is_refused_at_build():
    with pytest.raises(ParameterError) as info:
        _build({"type": "dl07", "umin": Fixed(50.0)})
    assert "dust_umin" in str(info.value)
    assert "[0.1, 25]" in str(info.value)


@pytest.mark.parametrize("gamma", [-0.5, 1.5])
def test_gamma_dl_outside_unit_interval_is_refused_at_build(gamma):
    with pytest.raises((ValueError, ParameterError), match="dust_gamma_dl"):
        _build({"type": "dl07", "gamma_dl": Fixed(gamma)})


def test_uniform_gamma_dl_overhanging_unit_interval_is_refused_at_build():
    with pytest.raises((ValueError, ParameterError), match="dust_gamma_dl"):
        _build({"type": "dl07", "gamma_dl": Uniform(-0.5, 0.5)})


def test_in_grid_dl07_values_build_unchanged():
    """The control: in-grid values are accepted and kept verbatim."""
    spec = _build(
        {"type": "dl07", "qpah": Fixed(2.5), "umin": Fixed(1.0), "gamma_dl": Fixed(0.01)}
    )
    assert spec._distributions["dust_qpah"].bounds == pytest.approx((2.5, 2.5))
    assert spec._distributions["dust_umin"].bounds == pytest.approx((1.0, 1.0))
    assert spec._distributions["dust_gamma_dl"].bounds == pytest.approx((0.01, 0.01))


def _family_axes():
    """``(name, param, grid_hi)`` for every tabulated dust-emission axis present."""
    rows = []
    for selector, name in sorted(GRID_SUPPORT):
        if selector != "dust.emission":
            continue
        for pname, (_lo, hi) in grid_support(selector, name).items():
            rows.append((name, pname, hi))
    return rows


_FAMILY_AXES = _family_axes()


@pytest.mark.skipif(not _FAMILY_AXES, reason="no tabulated dust-emission grids installed")
@pytest.mark.parametrize(("name", "pname", "hi"), _FAMILY_AXES)
def test_fixed_value_past_any_tabulated_family_grid_is_refused(name, pname, hi):
    """Every registered template family refuses an out-of-grid Fixed value."""
    key = pname[len("dust_") :]
    with pytest.raises(ParameterError) as info:
        _build({"type": name, key: Fixed(hi * 1.5)})
    assert pname in str(info.value)
