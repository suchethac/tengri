# SPDX-License-Identifier: BSD-3-Clause
r"""``sfh_burst_log_fburst`` is a fraction of the formed mass, not of the peak SFR.

The composed closure used to mix the burst in as
``(1 - f) * smooth + f * burst_shape * max(smooth)``: the burst amplitude was
set by the smooth history's *peak* SFR, a quantity with no bearing on how much
mass the burst forms. Two things followed, both measured through
``SEDModel.build`` on the FSPS grid with a const(10.0, 5 Gyr) history and
``log_fburst = -0.5``:

* the formed mass moved, ``log_mstar_formed`` 10.000000 -> 9.956891 for a
  broad burst and -> 9.835262 for a compact one, so a burst quietly rewrote
  the stellar mass the fit was reporting;
* the burst carried the wrong mass, 0.0008 of the total for the compact burst
  against the 0.316228 its ``log_fburst = -0.5`` names.

The parameter's own declaration reads "log10 burst mass fraction", so the
mixture is

.. math::

    \mathrm{SFR}(t) = (1 - f)\,\mathrm{SFR}_{\rm smooth}(t)
                    + f\,M\,\frac{B(t)}{\int B\,\mathrm{d}t},
    \qquad M = \int \mathrm{SFR}_{\rm smooth}\,\mathrm{d}t

with :math:`f = 10^{\log f_{\rm burst}}`. The three invariants below follow
exactly and are what this file pins: the formed mass is :math:`M` with or
without the burst, the burst carries the fraction :math:`f`, and wherever the
kernel is zero the history is exactly :math:`1 - f` times the smooth one.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.sfh.mean_sfh import triweight_burst

pytestmark = pytest.mark.contract

_W = Fixed(DEFAULT)
_Z = Fixed(0.05)
_LOG_FBURST = -0.5
_F = 10.0**_LOG_FBURST

_CONST = {
    "sfh_const_log_total_mass": Fixed(10.0),
    "sfh_const_start_gyr": Fixed(5.0),
    "sfh_const_end_gyr": Fixed(0.0),
}
# The triweight kernel's support is |log10(t/Myr) - log_tpeak| < 3 * log_tmax
# *dex*, so the registry default log_tmax_myr = 2.5 spans 15 dex and covers
# every node of any SFH grid. log_tmax_myr = 0.3 confines this burst to
# t in (1.26, 79.4) Myr, which leaves the grid's old half exactly zero.
_BURST_COMPACT = {
    "sfh_burst_log_fburst": Fixed(_LOG_FBURST),
    "sfh_burst_log_tpeak_myr": Fixed(1.0),
    "sfh_burst_log_tmax_myr": Fixed(0.3),
}
_BURST_BROAD = {
    "sfh_burst_log_fburst": Fixed(_LOG_FBURST),
    "sfh_burst_log_tpeak_myr": Fixed(1.5),
    "sfh_burst_log_tmax_myr": Fixed(2.5),
}


def _state(ssp, sfh):
    return SEDModel.build(ssp_data=ssp, sfh={"all_params": _W, **sfh}, redshift=_Z).predict_state(
        {}
    )


@pytest.mark.parametrize(
    "burst", [_BURST_COMPACT, _BURST_BROAD], ids=["compact_kernel", "broad_kernel"]
)
def test_adding_a_burst_does_not_move_the_formed_mass(ssp_data_fsps, burst):
    """log_mstar_formed of const+burst equals const alone (measured: identical)."""
    s_const = _state(ssp_data_fsps, {"type": "const", **_CONST})
    s_mix = _state(ssp_data_fsps, {"type": ["const", "burst"], **_CONST, **burst})

    log_m_const = float(s_const.derived["log_mstar_formed"])
    log_m_mix = float(s_mix.derived["log_mstar_formed"])
    assert abs(log_m_mix - log_m_const) < 1e-6, (
        f"the burst moved the formed mass: {log_m_const:.6f} -> {log_m_mix:.6f}"
    )


def test_smooth_history_is_scaled_by_one_minus_f_where_the_burst_is_zero(ssp_data_fsps):
    """Off the compact kernel's support the ratio is exactly 1 - 10**log_fburst."""
    s_const = _state(ssp_data_fsps, {"type": "const", **_CONST})
    s_mix = _state(ssp_data_fsps, {"type": ["const", "burst"], **_CONST, **_BURST_COMPACT})

    lbt = np.asarray(s_const.derived["sfh_grid_lbt_yr"])
    sfr_const = np.asarray(s_const.derived["sfr_history"])
    sfr_mix = np.asarray(s_mix.derived["sfr_history"])

    shape = np.asarray(triweight_burst(lbt, log_tpeak_myr=1.0, log_tmax_myr=0.3))
    outside = (shape == 0.0) & (sfr_const > 0.0)
    assert outside.sum() > 50, "the compact kernel must leave most of the grid untouched"

    np.testing.assert_allclose(
        sfr_mix[outside] / sfr_const[outside], 1.0 - _F, rtol=1e-6, atol=0.0
    )

    # And at 3 Gyr specifically, the epoch the old peak-scaled mixture put at
    # 0.683772 only by coincidence of the compact kernel and at 0.776718 for
    # the broad one.
    i_old = int(np.argmin(np.abs(lbt - 3.0e9)))
    assert shape[i_old] == 0.0
    assert abs(sfr_mix[i_old] / sfr_const[i_old] - (1.0 - _F)) < 1e-6


@pytest.mark.parametrize(
    "burst", [_BURST_COMPACT, _BURST_BROAD], ids=["compact_kernel", "broad_kernel"]
)
def test_the_burst_carries_fraction_f_of_the_formed_mass(ssp_data_fsps, burst):
    """The mass above the (1 - f)-scaled smooth history is f of the total."""
    s_const = _state(ssp_data_fsps, {"type": "const", **_CONST})
    s_mix = _state(ssp_data_fsps, {"type": ["const", "burst"], **_CONST, **burst})

    lbt = np.asarray(s_const.derived["sfh_grid_lbt_yr"])
    sfr_const = np.asarray(s_const.derived["sfr_history"])
    sfr_mix = np.asarray(s_mix.derived["sfr_history"])

    m_total = np.trapezoid(sfr_mix, lbt)
    m_burst = np.trapezoid(sfr_mix - (1.0 - _F) * sfr_const, lbt)
    assert abs(m_burst / m_total - _F) < 1e-6, (
        f"burst carries {m_burst / m_total:.6f} of the mass, not {_F:.6f}"
    )


def test_the_burst_mass_is_independent_of_the_smooth_history_peak(ssp_data_fsps):
    """Two smooth histories of equal mass but 5x different peak SFR give the same burst mass.

    This is the defect named directly: the old mixture scaled the burst by
    ``max(smooth)``, so shortening the const window from 5 Gyr to 1 Gyr --
    which raises the peak SFR from 2.00 to 10.01 Msun/yr at fixed formed mass
    -- multiplied the burst's mass by 5. Measured under the old code:
    5.5211e+06 Msun against 2.7628e+07, a ratio of 5.004. Under the
    mass-fraction mixture both are exactly f * 1e10 = 3.1623e+09 Msun.
    """
    wide = {**_CONST, "sfh_const_start_gyr": Fixed(5.0)}
    narrow = {**_CONST, "sfh_const_start_gyr": Fixed(1.0)}

    masses = []
    for base in (wide, narrow):
        s_const = _state(ssp_data_fsps, {"type": "const", **base})
        s_mix = _state(ssp_data_fsps, {"type": ["const", "burst"], **base, **_BURST_COMPACT})
        lbt = np.asarray(s_const.derived["sfh_grid_lbt_yr"])
        sfr_const = np.asarray(s_const.derived["sfr_history"])
        sfr_mix = np.asarray(s_mix.derived["sfr_history"])
        masses.append(np.trapezoid(sfr_mix - (1.0 - _F) * sfr_const, lbt))

    np.testing.assert_allclose(masses[0], masses[1], rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(masses[0], _F * 1e10, rtol=1e-3, atol=0.0)


def test_the_burst_kernel_integral_is_resolved_by_the_evaluation_grid(ssp_data_fsps):
    """Trapezoid of the kernel on the SFH grid matches a 10**6-point reference.

    The mixture divides by ``int B dt`` on whatever grid the composed closure is
    handed, so a grid too coarse for the kernel would mis-scale the burst.
    Measured relative error against a 1e6-point reference on the same interval:
    7.7e-06 for the registry default (log_tmax_myr = 2.5, log_tpeak_myr = 2.0)
    and 2.3e-04 for the compact kernel used above -- both far inside 1e-3, so
    no extra knots are injected for the burst.
    """
    s_const = _state(ssp_data_fsps, {"type": "const", **_CONST})
    lbt = np.asarray(s_const.derived["sfh_grid_lbt_yr"])
    ref = np.linspace(float(lbt[0]), float(lbt[-1]), 1_000_001)

    for log_tpeak, log_tmax in ((2.0, 2.5), (1.5, 2.5), (1.0, 0.3)):
        on_grid = np.trapezoid(
            np.asarray(triweight_burst(lbt, log_tpeak_myr=log_tpeak, log_tmax_myr=log_tmax)), lbt
        )
        exact = np.trapezoid(
            np.asarray(triweight_burst(ref, log_tpeak_myr=log_tpeak, log_tmax_myr=log_tmax)), ref
        )
        rel = abs(on_grid - exact) / exact
        assert rel < 1e-3, f"kernel ({log_tpeak}, {log_tmax}) unresolved: rel err {rel:.3e}"
