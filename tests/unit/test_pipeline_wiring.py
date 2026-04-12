"""Regression tests: all physics parameters reach the pipeline call sites.

Each test verifies that a parameter wired in parameters.py + model.py is also
forwarded to the underlying physics function in sed_pipeline.py. Tests use
source inspection where possible (no SSP data needed) and mock-patching for
call-site kwarg verification.

These tests are designed to catch silent parameter drops — the bug pattern
where a user sets e.g. agn_a_spin=0.7 but the pipeline ignores it and uses
the function's own default.
"""

import inspect

import pytest

# ---------------------------------------------------------------------------
# Source-inspection helpers
# ---------------------------------------------------------------------------


def _pipeline_src() -> str:
    from tengri.core import sed_pipeline

    return inspect.getsource(sed_pipeline)


def _model_src() -> str:
    from tengri.core import model as model_mod

    return inspect.getsource(model_mod)


def _emission_helpers_src() -> str:
    from tengri.core import emission_helpers

    return inspect.getsource(emission_helpers)


def _params_src() -> str:
    from tengri.core import parameters

    return inspect.getsource(parameters)


# ---------------------------------------------------------------------------
# AGN spin + inclination forwarding
# ---------------------------------------------------------------------------
class TestAGNSpinCosInc:
    """agn_a_spin and agn_cos_inc must be forwarded to the AGN model call."""

    def test_agn_a_spin_in_pipeline_call(self):
        src = _pipeline_src()
        assert 'agn_a_spin=p.get("agn_a_spin"' in src, (
            "sed_pipeline.py must forward agn_a_spin to the AGN model call"
        )

    def test_agn_cos_inc_in_pipeline_call(self):
        src = _pipeline_src()
        assert 'agn_cos_inc=p.get("agn_cos_inc"' in src, (
            "sed_pipeline.py must forward agn_cos_inc to the AGN model call"
        )

    def test_agn_a_spin_declared_in_params(self):
        src = _params_src()
        assert '"agn_a_spin"' in src, "agn_a_spin must be declared in _AGN_PARAMS in parameters.py"

    def test_agn_a_spin_in_param_map(self):
        src = _model_src()
        assert '"agn_a_spin"' in src, "agn_a_spin must be registered in model.py _param_map loop"


# ---------------------------------------------------------------------------
# SKIRTOR parameter forwarding
# ---------------------------------------------------------------------------
class TestSKIRTORParams:
    """SKIRTOR clumpy torus params must be forwarded."""

    @pytest.mark.parametrize(
        "param",
        [
            "agn_tau_skirtor",
            "agn_p_skirtor",
            "agn_q_skirtor",
            "agn_oa_skirtor",
        ],
    )
    def test_skirtor_param_in_pipeline(self, param):
        src = _pipeline_src()
        assert f'{param}=p.get("{param}"' in src, (
            f"sed_pipeline.py must forward {param} to the AGN model call"
        )

    @pytest.mark.parametrize(
        "param",
        [
            "agn_tau_skirtor",
            "agn_p_skirtor",
            "agn_q_skirtor",
            "agn_oa_skirtor",
        ],
    )
    def test_skirtor_param_in_param_map(self, param):
        src = _model_src()
        assert f'"{param}"' in src, f"{param} must be registered in model.py _param_map loop"


# ---------------------------------------------------------------------------
# K&D full 3-zone disc parameters
# ---------------------------------------------------------------------------
class TestKDFullParams:
    """Kubota & Done (2018) 3-zone disc params must be forwarded."""

    @pytest.mark.parametrize(
        "param",
        [
            "agn_f_hard",
            "agn_gamma_warm",
            "agn_kt_warm",
            "agn_gamma_hard",
            "agn_kt_hot",
            "agn_r_warm_ratio",
        ],
    )
    def test_kd_param_in_pipeline(self, param):
        src = _pipeline_src()
        assert f'{param}=p.get("{param}"' in src, (
            f"sed_pipeline.py must forward {param} to the AGN model call"
        )

    @pytest.mark.parametrize(
        "param",
        [
            "agn_f_hard",
            "agn_gamma_warm",
            "agn_kt_warm",
            "agn_gamma_hard",
            "agn_kt_hot",
            "agn_r_warm_ratio",
        ],
    )
    def test_kd_param_declared_in_params(self, param):
        src = _params_src()
        assert f'"{param}"' in src, f"{param} must be declared in _AGN_PARAMS in parameters.py"


# ---------------------------------------------------------------------------
# Polar dust forwarding
# ---------------------------------------------------------------------------
class TestPolarDustForwarding:
    """Polar dust (agn_polar_ebv, agn_polar_oa) must gate and forward correctly."""

    def test_polar_ebv_declared_in_params(self):
        src = _params_src()
        assert '"agn_polar_ebv"' in src

    def test_polar_oa_declared_in_params(self):
        src = _params_src()
        assert '"agn_polar_oa"' in src

    def test_polar_dust_guard_present(self):
        """Pipeline must guard polar dust application on agn_polar_ebv > 0.

        The guard lives in emission_helpers.agn_emission() which is called by
        sed_pipeline.py.  We check the helpers module because the refactor
        moved the polar-dust block there (sed_pipeline.py only forwards the
        parameter value via agn_polar_ebv=p.get(...)).
        """
        src = _emission_helpers_src()
        assert "agn_polar_ebv) > 0.0" in src, (
            "emission_helpers.py must skip polar dust when agn_polar_ebv == 0"
        )

    def test_polar_dust_block_uses_cos_inc(self):
        src = _pipeline_src()
        assert 'cos_inc=p.get("agn_cos_inc"' in src, (
            "polar_dust_total call must pass cos_inc from agn_cos_inc param"
        )

    def test_polar_dust_block_uses_opening_angle(self):
        """opening_angle_deg must be forwarded from agn_polar_oa.

        sed_pipeline.py passes agn_polar_oa=p.get("agn_polar_oa", ...) to
        agn_emission(), which translates it to opening_angle_deg=agn_polar_oa
        inside emission_helpers.py.  We verify both sides of the indirection.
        """
        # sed_pipeline.py must forward the parameter to agn_emission()
        pipeline_src = _pipeline_src()
        assert 'agn_polar_oa=p.get("agn_polar_oa"' in pipeline_src, (
            "sed_pipeline.py must forward agn_polar_oa to agn_emission()"
        )
        # emission_helpers.py must pass it through as opening_angle_deg
        helpers_src = _emission_helpers_src()
        assert "opening_angle_deg=agn_polar_oa" in helpers_src, (
            "emission_helpers.agn_emission must pass opening_angle_deg=agn_polar_oa"
        )


# ---------------------------------------------------------------------------
# X-ray extra parameters
# ---------------------------------------------------------------------------
class TestXrayExtraParams:
    """gamma_hmxb, gamma_lmxb, E_cut must be forwarded to xray_total."""

    @pytest.mark.parametrize(
        "param,kwarg",
        [
            ("xray_gamma_hmxb", "gamma_hmxb"),
            ("xray_gamma_lmxb", "gamma_lmxb"),
            ("xray_E_cut", "E_cut"),
        ],
    )
    def test_xray_param_forwarded(self, param, kwarg):
        src = _pipeline_src()
        assert f'{kwarg}=p.get("{param}"' in src, (
            f"sed_pipeline.py must forward {param} as {kwarg} to xray_total"
        )

    @pytest.mark.parametrize("param", ["xray_gamma_hmxb", "xray_gamma_lmxb", "xray_E_cut"])
    def test_xray_param_declared(self, param):
        src = _params_src()
        assert f'"{param}"' in src, f"{param} must be declared in _XRAY_PARAMS in parameters.py"


# ---------------------------------------------------------------------------
# Radio free-free parameters
# ---------------------------------------------------------------------------
class TestRadioFreeFreeParams:
    """radio_T_e, radio_alpha_ff, sfr_mode, include_freefree must reach radio_total."""

    @pytest.mark.parametrize(
        "param,kwarg",
        [
            ("radio_T_e", "T_e"),
            ("radio_alpha_ff", "alpha_ff"),
        ],
    )
    def test_radio_float_param_forwarded(self, param, kwarg):
        src = _pipeline_src()
        assert f'{kwarg}=p.get("{param}"' in src, (
            f"sed_pipeline.py must forward {param} as {kwarg} to radio_total"
        )

    def test_radio_sfr_mode_forwarded_from_model_attr(self):
        src = _pipeline_src()
        assert "sfr_mode=model._radio_sfr_mode" in src, (
            "sed_pipeline.py must forward model._radio_sfr_mode to radio_total"
        )

    def test_radio_include_freefree_forwarded_from_model_attr(self):
        src = _pipeline_src()
        assert "include_freefree=model._radio_include_freefree" in src

    def test_radio_sfr_mode_attr_set_in_model(self):
        src = _model_src()
        assert "_radio_sfr_mode" in src, "model.py must set self._radio_sfr_mode on SEDModel"

    def test_radio_include_freefree_attr_set_in_model(self):
        src = _model_src()
        assert "_radio_include_freefree" in src

    @pytest.mark.parametrize("param", ["radio_T_e", "radio_alpha_ff"])
    def test_radio_param_declared(self, param):
        src = _params_src()
        assert f'"{param}"' in src, f"{param} must be declared in _RADIO_PARAMS in parameters.py"


# ---------------------------------------------------------------------------
# Dust emission alpha_dl14
# ---------------------------------------------------------------------------
class TestDustAlphaDL14:
    """dust_alpha_dl14 must be forwarded to the dust emission call."""

    def test_dust_alpha_dl14_in_pipeline(self):
        src = _pipeline_src()
        assert 'dust_alpha_dl14=p.get("dust_alpha_dl14"' in src, (
            "sed_pipeline.py must forward dust_alpha_dl14 to resolve_emission_model call"
        )

    def test_dust_alpha_dl14_in_param_map(self):
        src = _model_src()
        assert '"dust_alpha_dl14"' in src, (
            "dust_alpha_dl14 must be registered in model.py _param_map"
        )


# ---------------------------------------------------------------------------
# Shock b_over_sqrt_n registration
# ---------------------------------------------------------------------------
class TestShockBOverSqrtN:
    """shock_b_over_sqrt_n must be in _param_map when shock is enabled."""

    def test_shock_b_over_sqrt_n_declared_in_params(self):
        src = _params_src()
        assert '"shock_b_over_sqrt_n"' in src, (
            "shock_b_over_sqrt_n must be declared in _SHOCK_PARAMS in parameters.py"
        )

    def test_shock_b_over_sqrt_n_in_param_map(self):
        src = _model_src()
        assert '"shock_b_over_sqrt_n"' in src, (
            "shock_b_over_sqrt_n must be registered in model.py _param_map loop"
        )
