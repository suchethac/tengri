# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for _param_translate module."""

import warnings

import pytest

from tengri.parameters.translate import (
    _NON_SFH_PARAM_MAP,
    LOG10_ZSUN,
    PARAM_MAP,
    _build_param_map,
    find_short_param,
    get_internal_params,
)

pytestmark = pytest.mark.contract

# ── Tests for LOG10_ZSUN ──────────────────────────────────────────


class TestLog10Zsun:
    """Tests for the solar metallicity constant."""

    def test_value_asplund2009(self):
        # log10(0.0142) ≈ -1.848 (Asplund 2009)
        assert abs(LOG10_ZSUN - (-1.8477116556169435)) < 1e-12

    def test_used_in_non_sfh_map(self):
        _, _, offset = _NON_SFH_PARAM_MAP["met_logzsol"]
        assert offset == LOG10_ZSUN


# ── Tests for _build_param_map ────────────────────────────────────


class TestBuildParamMap:
    """Tests for the dynamic _build_param_map factory."""

    def test_tsnorm_has_sfh_params(self):
        pm = _build_param_map(["tsnorm"])
        assert "sfh_tsnorm_log_total_mass" in pm
        assert "sfh_tsnorm_peak_lbt_gyr" in pm

    def test_tsnorm_has_non_sfh_params(self):
        pm = _build_param_map(["tsnorm"])
        assert "met_logzsol" in pm
        assert "dust_tau_bc" in pm
        assert "dust_tau_diff" in pm

    def test_tsnorm_field_has_psd_params(self):
        pm = _build_param_map(["tsnorm", "field"])
        assert "sfh_field_psd_sigma" in pm
        assert "sfh_field_psd_tau_myr" in pm

    def test_returns_new_dict_each_call(self):
        pm1 = _build_param_map(["tsnorm"])
        pm2 = _build_param_map(["tsnorm"])
        assert pm1 is not pm2

    def test_met_unit_conversion_present(self):
        pm = _build_param_map(["tsnorm"])
        int_name, scale, offset = pm["met_logzsol"]
        assert int_name == "log_z_abs"
        assert scale == 1.0
        assert offset == LOG10_ZSUN


# ── Tests for find_short_param ───────────────────────────────────


class TestFindLegacyParam:
    """Tests for the legacy alias lookup."""

    def test_finds_psd_sigma(self):
        params = {"psd_sigma": 0.5}
        result = find_short_param(params, "sfh_field_psd_sigma")
        assert result == 0.5

    def test_finds_psd_tau_myr(self):
        params = {"psd_tau_myr": 200.0}
        result = find_short_param(params, "sfh_field_psd_tau_myr")
        assert result == 200.0

    def test_finds_sfh_alpha(self):
        params = {"sfh_alpha": 1.5}
        result = find_short_param(params, "sfh_dpl_alpha")
        assert result == 1.5

    def test_returns_none_for_unknown(self):
        params = {"some_other": 1.0}
        result = find_short_param(params, "sfh_field_psd_sigma")
        assert result is None

    def test_returns_none_when_alias_absent(self):
        params = {}
        result = find_short_param(params, "sfh_dpl_beta")
        assert result is None

    def test_does_not_mutate_params(self):
        params = {"psd_sigma": 0.5}
        original = dict(params)
        find_short_param(params, "sfh_field_psd_sigma")
        assert params == original


# ── Tests for get_internal_params ─────────────────────────────────


class _DummyDist:
    """Minimal distribution stub for testing."""

    def __init__(self, value, is_fixed=True):
        self.bounds = (value, value)
        self.is_fixed = is_fixed


class _DummySpec:
    """Minimal Parameters stub for testing."""

    def __init__(self, fixed_params=None):
        self._fixed = fixed_params or {}

    def get_distribution(self, name):
        if name in self._fixed:
            return _DummyDist(self._fixed[name], is_fixed=True)
        raise KeyError(name)


class TestGetInternalParams:
    """Tests for the main translation function."""

    def _simple_map(self):
        return {
            "met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN),
            "dust_tau_bc": ("tau_bc", 1.0, 0.0),
        }

    def test_scale_and_offset_applied(self):
        param_map = self._simple_map()
        spec = _DummySpec()
        params = {"met_logzsol": 0.0, "dust_tau_bc": 0.3}
        result = get_internal_params(params, param_map, spec, has_field=False)
        assert abs(result["log_z_abs"] - LOG10_ZSUN) < 1e-12  # 0.0 * 1.0 + LOG10_ZSUN
        assert abs(result["tau_bc"] - 0.3) < 1e-12

    def test_met_logzsol_solar_is_log10zsun(self):
        """met_logzsol=0 (solar) should map to log_z_abs = LOG10_ZSUN."""
        param_map = {"met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN)}
        spec = _DummySpec()
        params = {"met_logzsol": 0.0}
        result = get_internal_params(params, param_map, spec, has_field=False)
        assert abs(result["log_z_abs"] - LOG10_ZSUN) < 1e-12

    def test_met_logzsol_above_solar(self):
        """met_logzsol=0.2 should map to log_z_abs = 0.2 + LOG10_ZSUN."""
        param_map = {"met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN)}
        spec = _DummySpec()
        params = {"met_logzsol": 0.2}
        result = get_internal_params(params, param_map, spec, has_field=False)
        assert abs(result["log_z_abs"] - (0.2 + LOG10_ZSUN)) < 1e-12

    def test_fixed_param_from_spec(self):
        """When a param is absent from params but fixed in spec, use spec value."""
        param_map = {"dust_tau_bc": ("tau_bc", 1.0, 0.0)}
        spec = _DummySpec(fixed_params={"dust_tau_bc": 0.5})
        params = {}
        result = get_internal_params(params, param_map, spec, has_field=False)
        assert abs(result["tau_bc"] - 0.5) < 1e-12

    def test_legacy_alias_fallback(self):
        """psd_sigma in params should be accepted for sfh_field_psd_sigma."""
        param_map = {"sfh_field_psd_sigma": ("psd_sigma", 1.0, 0.0)}
        spec = _DummySpec()
        params = {"psd_sigma": 0.7}
        result = get_internal_params(params, param_map, spec, has_field=False)
        assert abs(result["psd_sigma"] - 0.7) < 1e-12

    def test_has_field_passes_xi(self):
        param_map = {}
        spec = _DummySpec()
        xi_val = [1.0, 2.0, 3.0]
        params = {"sfh_field_xi": xi_val}
        result = get_internal_params(params, param_map, spec, has_field=True)
        assert result["xi"] is xi_val

    def test_has_field_legacy_psd_xi(self):
        param_map = {}
        spec = _DummySpec()
        xi_val = [0.1, 0.2]
        params = {"psd_xi": xi_val}
        result = get_internal_params(params, param_map, spec, has_field=True)
        assert result["xi"] is xi_val

    def test_no_field_xi_not_in_result(self):
        param_map = {}
        spec = _DummySpec()
        params = {"sfh_field_xi": [1.0]}
        result = get_internal_params(params, param_map, spec, has_field=False)
        assert "xi" not in result

    def test_unrecognized_keys_raise(self):
        """Strict-by-default: unknown keys raise ValueError with a helpful message."""
        param_map = {"met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN)}
        spec = _DummySpec()
        params = {"met_logzsol": 0.0, "totally_unknown_param": 99.0}
        with pytest.raises(ValueError, match="totally_unknown_param"):
            get_internal_params(params, param_map, spec, has_field=False)

    def test_unrecognized_keys_warn_when_lenient(self):
        """Opt-out path: strict_unknown_params=False keeps the legacy warning behavior."""
        param_map = {"met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN)}
        spec = _DummySpec()
        params = {"met_logzsol": 0.0, "totally_unknown_param": 99.0}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_internal_params(
                params, param_map, spec, has_field=False, strict_unknown_params=False
            )
        assert len(w) == 1
        assert "totally_unknown_param" in str(w[0].message)
        assert issubclass(w[0].category, UserWarning)

    def test_free_param_missing_raises(self):
        """A free param absent from both params and spec should raise KeyError."""
        param_map = {"dust_tau_bc": ("tau_bc", 1.0, 0.0)}

        class _FreeSpec:
            def get_distribution(self, name):
                return _DummyDist(0.0, is_fixed=False)

        with pytest.raises(KeyError, match="dust_tau_bc"):
            get_internal_params({}, param_map, _FreeSpec(), has_field=False)

    def test_does_not_mutate_params(self):
        param_map = {"met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN)}
        spec = _DummySpec()
        params = {"met_logzsol": 0.0}
        original = dict(params)
        get_internal_params(params, param_map, spec, has_field=False)
        assert params == original


# ── Tests for legacy PARAM_MAP ────────────────────────────────────


class TestLegacyParamMap:
    """Tests for the legacy module-level PARAM_MAP (backward compat)."""

    def test_contains_sfh_params(self):
        assert "sfh_alpha" in PARAM_MAP
        assert "sfh_beta" in PARAM_MAP

    def test_unit_conversion_tau_peak(self):
        _, scale, offset = PARAM_MAP["sfh_tau_peak_gyr"]
        assert scale == 1e9  # Gyr -> yr
        assert offset == 0.0

    def test_unit_conversion_psd_tau(self):
        _, scale, offset = PARAM_MAP["psd_tau_myr"]
        assert scale == 1e6  # Myr -> yr
        assert offset == 0.0

    def test_metallicity_solar_offset(self):
        _, scale, offset = PARAM_MAP["met_logzsol"]
        assert scale == 1.0
        assert abs(offset - LOG10_ZSUN) < 1e-12
