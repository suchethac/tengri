# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the CIGALE interop helpers (#357).

These helpers exist so cross-code comparison runs (e.g.
``reproduction/cigale/01_cigale.py``) aren't booby-trapped by silent
convention mismatches. We test:

1. The SFH mass-formed inverter actually recovers the target mass
   by numerically integrating the resulting SFR over [0, age].
2. The dust mapping reproduces CIGALE's invariant that
   ``tau_bc + tau_diff`` equals the full ``E(B-V)_lines`` attenuation
   (and that ``tau_diff`` alone equals the stellar continuum
   attenuation).
3. Argument validation rejects bad inputs with clear messages.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh import (
    declining_exponential,
    delayed_exponential,
    delayed_tau,
)
from tengri.interop.cigale import (
    CIGALE_CALZETTI_AV_OVER_EBV,
    CIGALE_MODIFIED_STARBURST_EBV_RATIO,
    cigale_ebv_lines_to_tau,
    log_peak_sfr_for_mass_formed,
)


def _mass_formed(sfr_fn, age_yr: float, *, n_points: int = 100_000) -> float:
    """Trapezoidal integral of an SFR function on [0, age_yr]."""
    t = jnp.linspace(0.0, age_yr, n_points)
    sfr = sfr_fn(t)
    return float(jnp.trapezoid(sfr, t))


class TestSFHMassFormedInverter:
    """Round-trip: invert log_peak_sfr from target mass, integrate, compare."""

    @pytest.mark.parametrize(
        "model,sfh_fn",
        [
            ("dexp", lambda t, lp, tau, age: delayed_exponential(t, lp, tau, start=0.0)),
            ("delayed_exponential", lambda t, lp, tau, age: delayed_exponential(t, lp, tau, 0.0)),
            ("tau", lambda t, lp, tau, age: declining_exponential(t, lp, tau, age)),
            (
                "declining_exponential",
                lambda t, lp, tau, age: declining_exponential(t, lp, tau, age),
            ),
        ],
    )
    @pytest.mark.parametrize("mass_msun", [1.0, 1e9, 1e11])
    @pytest.mark.parametrize("tau_gyr,age_gyr", [(1.0, 5.0), (0.5, 13.0), (3.0, 10.0)])
    def test_round_trip_recovers_target_mass(self, model, sfh_fn, mass_msun, tau_gyr, age_gyr):
        log_peak = log_peak_sfr_for_mass_formed(model, mass_msun, tau_gyr=tau_gyr, age_gyr=age_gyr)
        actual_mass = _mass_formed(
            lambda t: sfh_fn(t, log_peak, tau_gyr * 1e9, age_gyr * 1e9),
            age_yr=age_gyr * 1e9,
        )
        np.testing.assert_allclose(actual_mass, mass_msun, rtol=2e-3)

    def test_delayed_tau_round_trip(self):
        """`delayed_tau` is parameterised by linear `norm`, not peak SFR."""
        log_norm = log_peak_sfr_for_mass_formed(
            "delayed_tau", mass_formed_msun=1.0, tau_gyr=1.0, age_gyr=5.0
        )
        norm = 10.0**log_norm
        actual_mass = _mass_formed(
            lambda t: delayed_tau(t, tau=1e9, norm=norm),
            age_yr=5e9,
        )
        np.testing.assert_allclose(actual_mass, 1.0, rtol=2e-3)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown SFH model"):
            log_peak_sfr_for_mass_formed("not-a-real-sfh", 1.0, tau_gyr=1.0, age_gyr=5.0)

    @pytest.mark.parametrize("bad_kwarg", ["mass", "tau", "age"])
    def test_non_positive_input_raises(self, bad_kwarg):
        kwargs = {"mass_formed_msun": 1.0, "tau_gyr": 1.0, "age_gyr": 5.0}
        kwargs[{"mass": "mass_formed_msun", "tau": "tau_gyr", "age": "age_gyr"}[bad_kwarg]] = 0.0
        with pytest.raises(ValueError):
            log_peak_sfr_for_mass_formed("dexp", **kwargs)


class TestCigaleDustMapping:
    def test_ebv_lines_zero_gives_zero_tau(self):
        out = cigale_ebv_lines_to_tau(0.0)
        assert out == {"tau_bc": 0.0, "tau_diff": 0.0}

    def test_sum_matches_full_lines_attenuation(self):
        """The lines see ``tau_bc + tau_diff`` total — equal to the
        CIGALE optical depth at V derived from ``E(B-V)_lines``."""
        ebv = 0.3
        out = cigale_ebv_lines_to_tau(ebv)
        expected_total_tau_v = math.log(10) / 2.5 * CIGALE_CALZETTI_AV_OVER_EBV * ebv
        np.testing.assert_allclose(
            out["tau_bc"] + out["tau_diff"], expected_total_tau_v, rtol=1e-12
        )

    def test_tau_diff_matches_continuum_attenuation(self):
        """``tau_diff`` alone reproduces the stellar continuum attenuation
        (``ebv_ratio * tau_v_lines``)."""
        ebv = 0.3
        out = cigale_ebv_lines_to_tau(ebv)
        expected_tau_diff = (
            math.log(10)
            / 2.5
            * CIGALE_CALZETTI_AV_OVER_EBV
            * CIGALE_MODIFIED_STARBURST_EBV_RATIO
            * ebv
        )
        np.testing.assert_allclose(out["tau_diff"], expected_tau_diff, rtol=1e-12)

    def test_linear_scaling(self):
        """Doubling E(B-V) doubles both tau components."""
        a = cigale_ebv_lines_to_tau(0.2)
        b = cigale_ebv_lines_to_tau(0.4)
        np.testing.assert_allclose(b["tau_bc"], 2.0 * a["tau_bc"], rtol=1e-12)
        np.testing.assert_allclose(b["tau_diff"], 2.0 * a["tau_diff"], rtol=1e-12)

    def test_custom_ebv_ratio_redistributes(self):
        """If the CIGALE config flips the BC/continuum split, the
        returned ``tau_bc``/``tau_diff`` follow."""
        ebv = 0.3
        std = cigale_ebv_lines_to_tau(ebv)
        equal = cigale_ebv_lines_to_tau(ebv, ebv_ratio=0.5)
        # Same total
        np.testing.assert_allclose(
            std["tau_bc"] + std["tau_diff"],
            equal["tau_bc"] + equal["tau_diff"],
            rtol=1e-12,
        )
        # With ebv_ratio=0.5 the two halves are equal.
        np.testing.assert_allclose(equal["tau_bc"], equal["tau_diff"], rtol=1e-12)

    @pytest.mark.parametrize(
        "bad_input,kwargs",
        [
            ("ebv_lines", {"ebv_lines": -0.1}),
            ("av_over_ebv", {"ebv_lines": 0.3, "av_over_ebv": 0.0}),
            ("ebv_ratio<0", {"ebv_lines": 0.3, "ebv_ratio": -0.1}),
            ("ebv_ratio>1", {"ebv_lines": 0.3, "ebv_ratio": 1.5}),
        ],
    )
    def test_argument_validation(self, bad_input, kwargs):
        with pytest.raises(ValueError):
            cigale_ebv_lines_to_tau(**kwargs)


class TestModuleExportSurface:
    def test_top_level_interop_namespace(self):
        import tengri.interop

        assert hasattr(tengri.interop, "cigale")
        from tengri.interop.cigale import log_peak_sfr_for_mass_formed as fn

        assert callable(fn)
