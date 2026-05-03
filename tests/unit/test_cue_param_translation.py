"""Unit tests for Cue param plumbing through translate.py.

Regression test for MISSING_FEATURES.md #16: ``gas_logno``, ``gas_logco``,
``gas_logn`` and the seven ``ionspec_*`` params are validated by
``Parameters`` but were silently stripped by ``translate.get_internal_params``
because they were missing from ``_param_map``. This test pins the fix.
"""

from __future__ import annotations

import warnings

import pytest

from tengri.parameters.translate import (
    _CUE_GAS_IDENTITY_PARAMS,
    _CUE_IONSPEC_IDENTITY_PARAMS,
    get_internal_params,
    identity_param_map,
)

pytestmark = pytest.mark.unit


_GAS_NAMES = ("gas_logn", "gas_logno", "gas_logco")
_IONSPEC_NAMES = (
    "ionspec_index1",
    "ionspec_index2",
    "ionspec_index3",
    "ionspec_index4",
    "ionspec_logLratio1",
    "ionspec_logLratio2",
    "ionspec_logLratio3",
)


class TestCueIdentityParamLists:
    def test_gas_extras_listed(self):
        for name in _GAS_NAMES:
            assert name in _CUE_GAS_IDENTITY_PARAMS, f"{name} missing"

    def test_ionspec_listed(self):
        for name in _IONSPEC_NAMES:
            assert name in _CUE_IONSPEC_IDENTITY_PARAMS, f"{name} missing"


class TestIdentityMapPropagation:
    """get_internal_params should pass gas_logno / gas_logco / ionspec_* through."""

    def _make_spec_stub(self):
        """Minimal spec stub with get_distribution always raising KeyError so the
        free-param fallback isn't exercised; we only test recognised pass-through."""

        class _Stub:
            def get_distribution(self, name):
                raise KeyError(name)

        return _Stub()

    def test_gas_logno_passes_through(self):
        pm = identity_param_map(["gas_logno"])
        params = {"gas_logno": 0.7}
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # unrecognised-name warning would fail here
            internal = get_internal_params(params, pm, self._make_spec_stub(), has_field=False)
        assert internal["gas_logno"] == pytest.approx(0.7)

    def test_gas_logco_passes_through(self):
        pm = identity_param_map(["gas_logco"])
        params = {"gas_logco": -0.36}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            internal = get_internal_params(params, pm, self._make_spec_stub(), has_field=False)
        assert internal["gas_logco"] == pytest.approx(-0.36)

    def test_ionspec_index1_passes_through(self):
        pm = identity_param_map(["ionspec_index1"])
        params = {"ionspec_index1": 12.5}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            internal = get_internal_params(params, pm, self._make_spec_stub(), has_field=False)
        assert internal["ionspec_index1"] == pytest.approx(12.5)

    def test_unknown_param_still_warns(self):
        """Sanity check: a truly bogus key still triggers the unrecognised warning."""
        pm = identity_param_map(["gas_logno"])
        params = {"gas_logno": 0.0, "definitely_not_a_param": 1.0}
        with pytest.warns(UserWarning, match="Unrecognized parameter"):
            get_internal_params(params, pm, self._make_spec_stub(), has_field=False)


class TestSEDModelCueParamMap:
    """If SSP fixture is available, build a Cue SEDModel and check its _param_map."""

    def test_cue_model_param_map_contains_gas_logno(self, ssp_data_wne):
        from tengri import Fixed, Parameters, SEDModel

        spec = Parameters(
            nebular_cue=True,
            sfh_tsnorm_log_peak_sfr=Fixed(1.0),
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
        model = SEDModel(spec, ssp_data_wne)
        pm = model._param_map
        for name in ("gas_logn", "gas_logno", "gas_logco"):
            assert name in pm, f"{name} missing from SEDModel _param_map"
