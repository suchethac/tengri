# SPDX-License-Identifier: BSD-3-Clause
"""Property + parity tests for the ``delayed`` SFH (#406).

Matches CIGALE ``sfh_delayed`` / Bagpipes ``delayed``:

.. math::

   \\mathrm{SFR}(T) \\propto T \\, \\exp(-T/\\tau),
   \\quad T = \\mathrm{age} - t_{\\mathrm{lb}} \\geq 0

Distinct from tengri's ``tau`` SFH (``declining_exponential``), which
peaks at galaxy formation (T = 0) rather than at T = τ.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper


def test_delayed_peaks_at_age_minus_tau():
    """SFR maximum is at lookback time ``age − τ``."""
    from tengri.components.stellar.sfh.mean_sfh import sfhdelayed

    t = np.linspace(0.0, 5e9, 5001)
    sfr = np.asarray(sfhdelayed(t, log_total_mass=0.0, tau=1e9, age=5e9))

    peak_idx = int(np.argmax(sfr))
    peak_t_yr = t[peak_idx]
    expected_t_yr = 5e9 - 1e9  # age - tau
    # Grid resolution is ~1 Myr; allow 2× that.
    assert abs(peak_t_yr - expected_t_yr) < 2e6, (
        f"Peak at t_lb = {peak_t_yr:.3e} yr, expected {expected_t_yr:.3e}"
    )


def test_delayed_mass_conserves_to_one_msun_at_log_total_mass_zero():
    """``log_total_mass=0`` ⇒ integrated mass = 1 Msun exactly."""
    from tengri.components.stellar.sfh.mean_sfh import sfhdelayed

    t = np.linspace(0.0, 5e9, 5001)
    sfr = np.asarray(sfhdelayed(t, log_total_mass=0.0, tau=1e9, age=5e9))

    total = np.trapezoid(sfr, t)
    assert np.isclose(total, 1.0, rtol=1e-3), f"Total mass = {total:.6e} Msun, expected 1.0"


def test_delayed_zero_at_formation_epoch_and_before():
    """SFR(t_lb = age) = 0 (no SFR at formation); SFR(t_lb > age) = 0."""
    from tengri.components.stellar.sfh.mean_sfh import sfhdelayed

    age = 5e9
    t = np.array([age, age + 1e8])
    sfr = np.asarray(sfhdelayed(t, log_total_mass=0.0, tau=1e9, age=age))

    assert sfr[0] == 0.0, f"SFR at formation (t_lb=age={age}) = {sfr[0]}, expected 0"
    assert sfr[1] == 0.0, "SFR before formation (t_lb > age) leaked"


def test_delayed_distinct_from_tau_at_formation():
    """The ``delayed`` shape is zero at t_lb=age; the ``tau`` shape is at its peak.

    Pins the conceptual distinction — confusing the two was the root of #406.
    """
    from tengri.components.stellar.sfh.mean_sfh import declining_exponential, sfhdelayed

    age = 5e9
    tau = 1e9
    # A grid point at galaxy formation (lookback time = age).
    t = np.array([age])

    sfr_delayed = float(sfhdelayed(t, log_total_mass=0.0, tau=tau, age=age)[0])
    sfr_tau = float(declining_exponential(t, log_total_mass=0.0, tau=tau, age=age)[0])

    # delayed: zero at formation (T = 0 → T·e^(-T/τ) = 0).
    assert sfr_delayed == 0.0
    # tau: peaks at formation (T = 0 → e^(-T/τ) = 1, the maximum).
    assert sfr_tau > 0.0


def test_delayed_registered_in_sfh_registry():
    """``tengri.list_sfh_models()`` returns ``delayed`` with the expected param prefix."""
    import tengri

    rows = tengri.list_sfh_models()
    names = {r["name"] for r in rows}
    assert "delayed" in names, "delayed SFH not registered"

    delayed_row = next(r for r in rows if r["name"] == "delayed")
    param_names = {p["name"] for p in delayed_row["param_details"]}
    assert "sfh_delayed_log_total_mass" in param_names
    assert "sfh_delayed_tau_gyr" in param_names
    assert "sfh_delayed_age_gyr" in param_names


def test_delayed_buildable_via_sedmodel_build():
    """``SEDModel.build(sfh={'type': 'delayed', ...})`` resolves cleanly."""
    import tengri

    try:
        ssp = tengri.load_ssp()
    except Exception:
        pytest.skip("SSP fixture not present")

    model = tengri.SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "all_params": tengri.FIXED,
            "log_total_mass": 10.0,
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={"type": "single_component", "law": "calzetti", "all_params": tengri.FIXED},
        neb={"type": "ssp", "all_params": tengri.FIXED},
        redshift=tengri.Fixed(0.05),
    )
    # If we got here the registry round-tripped — that's the contract this test pins.
    assert model is not None
