# SPDX-License-Identifier: BSD-3-Clause
"""A composite SFH list is composed, not truncated to its first member.

``sfh={"type": ["const", "norm"]}`` must evaluate to the const history plus
the norm history, with the formed mass equal to the sum of the two, and
``["const", "burst"]`` must apply the burst mixture. The forward model used
to hand the stellar component only the first non-field type, so the second
additive member and any burst were silently dropped: the composite SED was
bit-identical to the const-only SED.

The composite kwargs must also be dispatched per member by public name:
two parametric families both take the internal ``log_total_mass``, so a
composite keyed by internal name lets the second member overwrite the first.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.contract

_W = Fixed(DEFAULT)
_Z = Fixed(0.05)
_CONST = {
    "sfh_const_log_total_mass": Fixed(10.0),
    "sfh_const_start_gyr": Fixed(1.0),
    "sfh_const_end_gyr": Fixed(0.0),
}
_NORM = {
    "sfh_norm_log_total_mass": Fixed(10.3),
    "sfh_norm_peak_lbt_gyr": Fixed(0.5),
    "sfh_norm_width_gyr": Fixed(0.0707),
}


def _state(ssp, sfh):
    return SEDModel.build(ssp_data=ssp, sfh={"all_params": _W, **sfh}, redshift=_Z).predict_state(
        {}
    )


@pytest.mark.parametrize(
    "order", [["const", "norm"], ["norm", "const"]], ids=lambda o: "+".join(o)
)
def test_additive_list_is_the_sum_of_its_members(ssp_data_fsps, order):
    """SFR, formed mass, and intrinsic SED of ["const", "norm"] equal const + norm."""
    s_const = _state(ssp_data_fsps, {"type": "const", **_CONST})
    s_norm = _state(ssp_data_fsps, {"type": "norm", **_NORM})
    s_both = _state(ssp_data_fsps, {"type": order, **_CONST, **_NORM})

    sfr_sum = np.asarray(s_const.derived["sfr_history"]) + np.asarray(
        s_norm.derived["sfr_history"]
    )
    np.testing.assert_allclose(
        np.asarray(s_both.derived["sfr_history"]), sfr_sum, rtol=1e-10, atol=0.0
    )

    expected_log_mass = np.log10(10.0**10.0 + 10.0**10.3)
    assert abs(float(s_both.derived["log_mstar_formed"]) - expected_log_mass) < 1e-6

    # The age weights are cloud-in-cell integrals of the summed history, so the
    # composite SED matches the sum of the member SEDs to the kernel's
    # discretization (measured 3e-5); the dropped-member defect gave ~100%.
    sed_sum = np.asarray(s_const.sed_intrinsic) + np.asarray(s_norm.sed_intrinsic)
    np.testing.assert_allclose(np.asarray(s_both.sed_intrinsic), sed_sum, rtol=1e-3)


def test_members_sharing_an_internal_kwarg_keep_their_own_values(ssp_data_fsps):
    """const(10.0)+norm(10.3) and const(10.3)+norm(10.0) form the same mass but differ in shape."""
    swapped_const = {**_CONST, "sfh_const_log_total_mass": Fixed(10.3)}
    swapped_norm = {**_NORM, "sfh_norm_log_total_mass": Fixed(10.0)}
    s_a = _state(ssp_data_fsps, {"type": ["const", "norm"], **_CONST, **_NORM})
    s_b = _state(ssp_data_fsps, {"type": ["const", "norm"], **swapped_const, **swapped_norm})

    assert (
        abs(float(s_a.derived["log_mstar_formed"]) - float(s_b.derived["log_mstar_formed"])) < 1e-6
    )
    lbt = np.asarray(s_a.derived["sfh_grid_lbt_yr"])
    i_plateau = np.argmin(np.abs(lbt - 0.9e9))  # inside the const window, outside the norm peak
    sfr_a = float(np.asarray(s_a.derived["sfr_history"])[i_plateau])
    sfr_b = float(np.asarray(s_b.derived["sfr_history"])[i_plateau])
    assert sfr_b > 1.5 * sfr_a, "the const member did not receive its own log_total_mass"


def test_burst_mixture_is_applied(ssp_data_fsps):
    """["const", "burst"] with log_fburst=-0.5 gives the burst 10^-0.5 of the formed mass.

    The mixture is a mass fraction, not a peak fraction: the smooth history is
    scaled by ``1 - f`` and the burst kernel is normalized by its own time
    integral and given ``f * M``, so the formed mass is untouched. The
    peak-scaled mixture this file first pinned put the burst amplitude at
    ``f * max(smooth)``, which moved ``log_mstar_formed`` 10.000000 ->
    9.956891 and left the burst carrying 0.260371 of the mass instead of
    0.316228. See ``test_sfh_burst_is_a_mass_fraction.py`` for the full set of
    invariants; this test keeps the composition-level check that the burst
    reaches the SFH at all (dropping it left every ratio at exactly 1).
    """
    base = {
        "sfh_const_log_total_mass": Fixed(10.0),
        "sfh_const_start_gyr": Fixed(5.0),
        "sfh_const_end_gyr": Fixed(0.0),
    }
    burst = {
        "sfh_burst_log_fburst": Fixed(-0.5),
        "sfh_burst_log_tpeak_myr": Fixed(1.5),
        "sfh_burst_log_tmax_myr": Fixed(2.5),
    }
    f = 10.0**-0.5
    s_const = _state(ssp_data_fsps, {"type": "const", **base})
    s_mix = _state(ssp_data_fsps, {"type": ["const", "burst"], **base, **burst})

    lbt = np.asarray(s_const.derived["sfh_grid_lbt_yr"])
    sfr_const = np.asarray(s_const.derived["sfr_history"])
    sfr_mix = np.asarray(s_mix.derived["sfr_history"])

    # The burst is present: at 3 Gyr the mixture is 0.805878 of the const
    # history (below 1 because of the 1 - f rescaling, above 1 - f = 0.683772
    # because this broad kernel still has support there).
    i_old = np.argmin(np.abs(lbt - 3.0e9))
    ratio_old = sfr_mix[i_old] / sfr_const[i_old]
    assert abs(ratio_old - 1.0) > 0.05, (
        f"burst mixture not applied at 3 Gyr (ratio={ratio_old:.4f})"
    )
    assert ratio_old > 1.0 - f, f"the burst adds nothing at 3 Gyr (ratio={ratio_old:.4f})"

    # And it carries exactly f of the formed mass, which is itself unchanged.
    assert abs(float(s_mix.derived["log_mstar_formed"]) - 10.0) < 1e-6
    m_total = np.trapezoid(sfr_mix, lbt)
    m_burst = np.trapezoid(sfr_mix - (1.0 - f) * sfr_const, lbt)
    assert abs(m_burst / m_total - f) < 1e-6, (
        f"burst carries {m_burst / m_total:.6f} of the mass, not {f:.6f}"
    )


def test_single_type_and_default_builds_are_unchanged(ssp_data_fsps):
    """The default group and a single explicit type still take the single-type path."""
    s_default = _state(ssp_data_fsps, {})
    assert np.isfinite(float(s_default.derived["log_mstar_formed"]))
    s_delayed = _state(ssp_data_fsps, {"type": "delayed", "log_total_mass": Fixed(10.0)})
    assert abs(float(s_delayed.derived["log_mstar_formed"]) - 10.0) < 1e-6
