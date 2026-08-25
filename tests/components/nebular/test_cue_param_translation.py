# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for Cue param plumbing through translate.py.

Regression test for MISSING_FEATURES.md #16: ``gas_logno``, ``gas_logco``,
``gas_logn`` and the seven ``ionspec_*`` params are validated by
``Parameters`` but were silently stripped by ``translate.get_internal_params``
because they were missing from ``_param_map``. This test pins the fix.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.parameters.translate import get_internal_params

# One assignment, not two: Python rebinds the name, so a second
# `pytestmark = ...` silently discarded the taxonomy marker.
pytestmark = [pytest.mark.bounds, pytest.mark.unit]


def _identity_pm(names):
    """Build a minimal identity param_map dict — `{n: (n, 1.0, 0.0)}`.

    Inline replacement for the removed `identity_param_map` helper
    (deleted in `77661f8f`, ADR-deepening Step B). The full registry
    coverage of Cue identity params is tested by
    `TestSEDModelCueParamMap` below; these tests only need a tiny
    param_map to exercise `get_internal_params` pass-through.
    """
    return {n: (n, 1.0, 0.0) for n in names}


class TestIdentityMapPropagation:
    """get_internal_params should pass gas_logno / gas_logco / ionspec_* through."""

    def _make_spec_stub(self):
        """Minimal spec stub with get_distribution always raising KeyError so the
        free-param fallback isn't exercised; we only test recognized pass-through."""

        class _Stub:
            def get_distribution(self, name):
                raise KeyError(name)

        return _Stub()

    def test_gas_logno_passes_through(self):
        pm = _identity_pm(["gas_logno"])
        params = {"gas_logno": 0.7}
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # unrecognized-name warning would fail here
            internal = get_internal_params(params, pm, self._make_spec_stub(), has_field=False)
        assert internal["gas_logno"] == pytest.approx(0.7)

    def test_gas_logco_passes_through(self):
        pm = _identity_pm(["gas_logco"])
        params = {"gas_logco": -0.36}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            internal = get_internal_params(params, pm, self._make_spec_stub(), has_field=False)
        assert internal["gas_logco"] == pytest.approx(-0.36)

    def test_ionspec_index1_passes_through(self):
        pm = _identity_pm(["ionspec_index1"])
        params = {"ionspec_index1": 12.5}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            internal = get_internal_params(params, pm, self._make_spec_stub(), has_field=False)
        assert internal["ionspec_index1"] == pytest.approx(12.5)

    def test_unknown_param_raises_by_default(self):
        """A truly bogus key raises ValueError under the default strict mode."""
        pm = _identity_pm(["gas_logno"])
        params = {"gas_logno": 0.0, "definitely_not_a_param": 1.0}
        with pytest.raises(ValueError, match="Unrecognized parameter"):
            get_internal_params(params, pm, self._make_spec_stub(), has_field=False)

    def test_unknown_param_warns_when_strict_false(self):
        """Legacy soft mode: passing ``strict_unknown_params=False`` downgrades to a warning."""
        pm = _identity_pm(["gas_logno"])
        params = {"gas_logno": 0.0, "definitely_not_a_param": 1.0}
        with pytest.warns(UserWarning, match="Unrecognized parameter"):
            get_internal_params(
                params,
                pm,
                self._make_spec_stub(),
                has_field=False,
                strict_unknown_params=False,
            )


class TestSEDModelCueParamMap:
    """If SSP fixture is available, build a Cue SEDModel and check its _param_map."""

    def test_cue_model_param_map_contains_gas_logno(self, ssp_data_fsps):
        # Use bare-stellar SSP (fsps_*); Cue + wNE now raises CueWNESSPError.
        # This test only checks _param_map structure, so SSP physics is irrelevant.
        from tengri import Fixed, Parameters, SEDModel

        spec = Parameters(
            nebular_cue=True,
            sfh_tsnorm_log_total_mass=Fixed(1.0),
            sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),
            sfh_tsnorm_width_gyr=Fixed(0.3),
            sfh_tsnorm_skew=Fixed(0.0),
            sfh_tsnorm_trunc=Fixed(3.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.0),
            neb_logU=Fixed(-3.0),
            neb_logZ_gas=Fixed(-0.3),
            gas_logn=Fixed(2.0),
            gas_logno=Fixed(0.0),
            gas_logco=Fixed(-0.36),
        )
        model = SEDModel(spec, ssp_data_fsps)
        pm = model._param_map
        for name in ("gas_logn", "gas_logno", "gas_logco"):
            assert name in pm, f"{name} missing from SEDModel _param_map"
