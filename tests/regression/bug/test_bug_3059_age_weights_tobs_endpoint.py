# SPDX-License-Identifier: BSD-3-Clause
"""
The closed t_obs endpoint of the age kernel is included on every platform (#3059).

Mechanism: the dense integrand's tail ends at ``top = t_obs * 1e9`` (the age of
the universe), but its last element came from a ``10 ** log10(top)`` round trip,
which differs from ``top`` by up to 28 ulp between x86_64 and arm64. The parcel
mask ``age < t_obs_gyr * 1e9`` then decided, by rounding alone, whether the
closed endpoint's trapezoid half-panel was kept. On macOS arm64 it was dropped
(about 1.2e7 Msun, 0.12 % of the total, which the formed-mass pin spread into a
1.2e-2 jump on the oldest SSP node). The trapezoid's closed endpoint at
``age = t_obs`` belongs to ``int_0^{t_obs} SFR dt``, so the mask must keep it.

These tests pin the fix:

1. The tail's last node is exactly ``top``, with no log round trip.
2. The age weights are continuous under a 10-ulp move of ``t_obs`` (the grid
   and the mask are built from the same value, so the endpoint never splits).
3. A constant SFH spanning to ``t_obs`` integrates to ``SFR * t_obs``: the
   endpoint half-panel is included.
4. Ages strictly beyond the age of the universe are still excluded.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.component import (
    _cic_parcels,
    _extend_integrand_to_history,
    _refine_sfh_table_ages,
)

pytestmark = pytest.mark.regression_bug

_Z = 0.05
_ULP_STEPS = 10


def _t_obs_yr_from_z(z):
    """Age of the universe at ``z`` in yr, through the same product the kernel uses."""
    from tengri.cosmology import age_at_z

    return float(age_at_z(z)) * 1e9


def _shift_ulps(value, steps):
    """Move a float64 ``value`` by ``steps`` ulp (negative moves down)."""
    out = float(value)
    direction = np.inf if steps > 0 else -np.inf
    for _ in range(abs(steps)):
        out = float(np.nextafter(out, direction))
    return out


class TestTailEndpointIsExact:
    """The integrand's tail ends at ``top`` exactly, not at a log round trip (#3059)."""

    @pytest.mark.parametrize("t_obs_gyr", [13.09002957780028, 13.5, 12.25])
    def test_last_node_equals_top_bitwise(self, t_obs_gyr):
        """The last node is ``max(t_obs, oldest template)`` exactly; t_obs when it is older."""
        ssp_ages_yr = jnp.asarray(np.geomspace(1e6, 1.2589e10, 93))
        fine = _refine_sfh_table_ages(ssp_ages_yr, factor=16)
        extent = jnp.reshape(jnp.asarray(t_obs_gyr) * 1e9, (1,))
        age_yr, top_yr = _extend_integrand_to_history(fine, extent, ssp_ages_yr, factor=16)

        t_obs_yr = float(jnp.asarray(t_obs_gyr) * 1e9)
        expected = max(t_obs_yr, float(ssp_ages_yr[-1]))
        assert float(age_yr[-1]) == expected
        assert float(top_yr) == expected


class TestClosedEndpointMask:
    """The parcel mask keeps the closed endpoint and drops only ages beyond t_obs."""

    @staticmethod
    def _ssp_ages_yr():
        return jnp.asarray(np.geomspace(1e6, 1.2589e10, 93))

    def test_constant_sfh_to_t_obs_integrates_to_sfr_times_t_obs(self):
        """Unnormalized trapezoid mass of a constant SFH spanning to t_obs is SFR * t_obs."""
        t_obs_gyr = 13.0
        t_end_yr = t_obs_gyr * 1e9
        # Grid from lookback 0 to t_obs, closed endpoint included exactly.
        age_yr = jnp.asarray(np.linspace(0.0, t_end_yr, 2001))
        sfr = jnp.ones_like(age_yr)
        _, _, _, total_mass, _ = _cic_parcels(age_yr, sfr, self._ssp_ages_yr(), t_obs_gyr)

        np.testing.assert_allclose(float(total_mass), t_end_yr, rtol=1e-12, atol=0.0)

    def test_ages_strictly_beyond_t_obs_are_excluded(self):
        """SFR beyond t_obs does not change the mass: those ages are masked out."""
        t_obs_gyr = 13.0
        t_end_yr = t_obs_gyr * 1e9
        ages_in = np.linspace(0.0, t_end_yr, 2001)
        age_yr = jnp.asarray(np.concatenate([ages_in, [t_end_yr * 1.001, t_end_yr * 1.002]]))
        inside = np.asarray(age_yr) <= t_end_yr
        sfr_zero_beyond = jnp.asarray(np.where(inside, 1.0, 0.0))
        sfr_hot_beyond = jnp.asarray(np.where(inside, 1.0, 1000.0))
        ssp_ages_yr = self._ssp_ages_yr()

        _, _, _, total_ref, _ = _cic_parcels(age_yr, sfr_zero_beyond, ssp_ages_yr, t_obs_gyr)
        _, _, _, total_hot, _ = _cic_parcels(age_yr, sfr_hot_beyond, ssp_ages_yr, t_obs_gyr)

        np.testing.assert_allclose(float(total_hot), float(total_ref), rtol=1e-12, atol=0.0)


class TestAgeWeightsContinuousInTObs:
    """Age weights of the #2578 model do not move under a 10-ulp move of t_obs (#3059)."""

    @staticmethod
    def _build(ssp):
        from tengri import Observation, Photometry
        from tengri.observation.photometry import FilterCurve

        def tophat(lo_rest, hi_rest, name, n=200):
            lo, hi = lo_rest * (1.0 + _Z), hi_rest * (1.0 + _Z)
            wave = np.linspace(lo, hi, n)
            trans = np.sin(np.linspace(0.0, np.pi, n)) * 0.6
            return FilterCurve(wave=wave, trans=trans, name=name)

        obs = Observation(
            photometry=Photometry(
                filters=(
                    tophat(5.0e3, 6.0e3, "opt"),
                    tophat(1.01e8, 1.05e8, "cm1"),
                    tophat(1.06e8, 1.10e8, "cm2"),
                )
            )
        )
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "ssp"},
            radio={
                "sf": {"type": "bell2003"},
                "agn": {"type": "powerlaw"},
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(_Z),
            approx=None,
        )

    @staticmethod
    def _age_weights_at(model, monkeypatch, t_obs_yr):
        import tengri.cosmology as cosmology

        real = cosmology.age_at_z

        def patched(z, *args, **kwargs):
            if np.ndim(z) == 0 and float(z) == _Z:
                return jnp.asarray(t_obs_yr / 1e9)
            return real(z, *args, **kwargs)

        monkeypatch.setattr(cosmology, "age_at_z", patched)
        state = model.predict_state({})
        return np.asarray(state.derived["age_weights"])

    def test_age_weights_invariant_to_ulp_move_of_t_obs(self, ssp_data_wne, monkeypatch):
        model = self._build(ssp_data_wne)
        t_obs_yr = _t_obs_yr_from_z(_Z)
        reference = self._age_weights_at(model, monkeypatch, t_obs_yr)

        for steps in (-_ULP_STEPS, _ULP_STEPS):
            moved = self._age_weights_at(model, monkeypatch, _shift_ulps(t_obs_yr, steps))
            np.testing.assert_allclose(moved, reference, rtol=1e-12, atol=0.0)
