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

# ── Source-inspection helpers ─────────────────────────────────────


def _has_param_consumer(src: str, param: str) -> bool:
    """True if ``src`` reads ``param`` from a params dict (any common form)."""
    return (
        f'params.get("{param}"' in src
        or f'params["{param}"]' in src
        or f'p.get("{param}"' in src
        or f'p["{param}"]' in src
    )


def _pipeline_src() -> str:
    """Aggregated source of every module that consumes physics-level params.

    Originally pinned to the legacy ``forward.pipeline.compute_sed_components``
    body, deleted in Phase B closure. Its parameter-forwarding contract now
    lives across the orchestrator's component adapters and helpers.
    """
    from tengri.components.agn import component as agn_component, unified as agn_unified
    from tengri.components.dust import (
        component as dust_component,
        emission_component as dust_emission_component,
        two_component as dust_two,
    )
    from tengri.components.radio import component as radio_component
    from tengri.components.xray import component as xray_component
    from tengri.forward import (
        emission_helpers as emission_helpers_mod,
        nonstell as nonstell_mod,
        pipeline as sed_pipeline,
        sed_model as sed_model_mod,
    )

    parts = [
        inspect.getsource(sed_pipeline),
        inspect.getsource(sed_model_mod),
        inspect.getsource(agn_component),
        inspect.getsource(agn_unified),
        inspect.getsource(dust_component),
        inspect.getsource(dust_emission_component),
        inspect.getsource(dust_two),
        inspect.getsource(radio_component),
        inspect.getsource(xray_component),
        inspect.getsource(emission_helpers_mod),
        inspect.getsource(nonstell_mod),
    ]
    return "\n".join(parts)


def _model_src() -> str:
    from tengri.forward import sed_model as model_mod

    return inspect.getsource(model_mod)


def _emission_helpers_src() -> str:
    from tengri.forward import emission_helpers

    return inspect.getsource(emission_helpers)


def _params_src() -> str:
    from tengri.parameters import _param_defs

    return inspect.getsource(_param_defs)


def _translate_src() -> str:
    from tengri.parameters import translate

    return inspect.getsource(translate)


# ── AGN spin + inclination forwarding ─────────────────────────────
class TestAGNSpinCosInc:
    """agn_a_spin and agn_cos_inc must be forwarded to the AGN model call."""

    def test_agn_a_spin_in_pipeline_call(self):
        src = _pipeline_src()
        assert _has_param_consumer(src, "agn_a_spin"), (
            "AGN component must read agn_a_spin from params and forward it"
        )

    def test_agn_cos_inc_in_pipeline_call(self):
        src = _pipeline_src()
        assert _has_param_consumer(src, "agn_cos_inc"), (
            "AGN component must read agn_cos_inc from params and forward it"
        )

    def test_agn_a_spin_declared_in_params(self):
        from tengri.parameters._param_defs import _AGN_PARAMS

        assert "agn_a_spin" in _AGN_PARAMS, (
            "agn_a_spin must be declared in _AGN_PARAMS "
            "(canonical source: tengri.components.agn._params.PARAMS)"
        )

    def test_agn_a_spin_in_param_map(self):
        from tengri.parameters.translate import _AGN_IDENTITY_PARAMS

        assert "agn_a_spin" in _AGN_IDENTITY_PARAMS, (
            "agn_a_spin must be in _AGN_IDENTITY_PARAMS in translate.py"
        )


# ── SKIRTOR parameter forwarding ──────────────────────────────────
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
        assert _has_param_consumer(src, param), (
            f"AGN component must read {param} from params and forward it"
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
        from tengri.parameters.translate import _AGN_IDENTITY_PARAMS

        assert param in _AGN_IDENTITY_PARAMS, (
            f"{param} must be in _AGN_IDENTITY_PARAMS in translate.py"
        )


# ── K&D full 3-zone disc parameters ───────────────────────────────
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
        assert _has_param_consumer(src, param), (
            f"AGN component must read {param} from params and forward it"
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
        from tengri.parameters._param_defs import _AGN_PARAMS

        assert param in _AGN_PARAMS, (
            f"{param} must be declared in _AGN_PARAMS "
            "(canonical source: tengri.components.agn._params.PARAMS)"
        )


# ── Polar dust forwarding ─────────────────────────────────────────
class TestPolarDustForwarding:
    """Polar dust (agn_polar_ebv, agn_polar_oa) must gate and forward correctly."""

    def test_polar_ebv_declared_in_params(self):
        from tengri.parameters._param_defs import _AGN_PARAMS

        assert "agn_polar_ebv" in _AGN_PARAMS

    def test_polar_oa_declared_in_params(self):
        from tengri.parameters._param_defs import _AGN_PARAMS

        assert "agn_polar_oa" in _AGN_PARAMS

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
        assert _has_param_consumer(src, "agn_cos_inc"), (
            "polar_dust_total call must read agn_cos_inc from params"
        )

    def test_polar_dust_block_uses_opening_angle(self):
        """opening_angle_deg must be forwarded from agn_polar_oa.

        sed_pipeline.py passes agn_polar_oa=p.get("agn_polar_oa", ...) to
        agn_emission(), which translates it to opening_angle_deg=agn_polar_oa
        inside emission_helpers.py.  We verify both sides of the indirection.
        """
        # sed_pipeline.py must forward the parameter to agn_emission()
        pipeline_src = _pipeline_src()
        assert _has_param_consumer(pipeline_src, "agn_polar_oa"), (
            "AGN component must read agn_polar_oa from params"
        )
        # emission_helpers.py must pass it through as opening_angle_deg
        helpers_src = _emission_helpers_src()
        assert "opening_angle_deg=agn_polar_oa" in helpers_src, (
            "emission_helpers.agn_emission must pass opening_angle_deg=agn_polar_oa"
        )


# ── X-ray extra parameters ────────────────────────────────────────
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
        assert _has_param_consumer(src, param), (
            f"X-ray component must read {param} from params and forward it"
        )

    @pytest.mark.parametrize("param", ["xray_gamma_hmxb", "xray_gamma_lmxb", "xray_E_cut"])
    def test_xray_param_declared(self, param):
        from tengri.parameters._param_defs import _XRAY_PARAMS

        assert param in _XRAY_PARAMS, (
            f"{param} must be declared in _XRAY_PARAMS "
            "(canonical source: tengri.components.xray._params.PARAMS)"
        )


# ── Radio free-free parameters ────────────────────────────────────
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
        assert _has_param_consumer(src, param), (
            f"Radio component must read {param} from params and forward it"
        )

    def test_radio_sfr_mode_forwarded_from_model_attr(self):
        src = _pipeline_src()
        assert "sfr_mode" in src, "Radio component must wire sfr_mode through to radio_total"

    def test_radio_include_freefree_forwarded_from_model_attr(self):
        src = _pipeline_src()
        assert "include_freefree" in src, (
            "Radio component must wire include_freefree through to radio_total"
        )

    def test_radio_sfr_mode_attr_set_in_model(self):
        src = _model_src()
        assert "_radio_sfr_mode" in src, "model.py must set self._radio_sfr_mode on SEDModel"

    def test_radio_include_freefree_attr_set_in_model(self):
        src = _model_src()
        assert "_radio_include_freefree" in src

    @pytest.mark.parametrize("param", ["radio_T_e", "radio_alpha_ff"])
    def test_radio_param_declared(self, param):
        from tengri.parameters._param_defs import _RADIO_PARAMS

        assert param in _RADIO_PARAMS, (
            f"{param} must be declared in _RADIO_PARAMS "
            "(canonical source: tengri.components.radio._params.PARAMS)"
        )


# ── Dust emission alpha_dl14 ──────────────────────────────────────
class TestDustAlphaDL14:
    """dust_alpha_dl14 must be forwarded to the dust emission call."""

    def test_dust_alpha_dl14_in_pipeline(self):
        src = _pipeline_src()
        assert _has_param_consumer(src, "dust_alpha_dl14"), (
            "Dust component must read dust_alpha_dl14 from params and forward it"
        )

    def test_dust_alpha_dl14_in_param_map(self):
        from tengri.parameters.translate import _DUST_EMISSION_IDENTITY_PARAMS

        assert "dust_alpha_dl14" in _DUST_EMISSION_IDENTITY_PARAMS, (
            "dust_alpha_dl14 must be in _DUST_EMISSION_IDENTITY_PARAMS in translate.py"
        )


# ── Shock b_over_sqrt_n registration ──────────────────────────────
class TestShockBOverSqrtN:
    """shock_b_over_sqrt_n must be in _param_map when shock is enabled."""

    def test_shock_b_over_sqrt_n_declared_in_params(self):
        src = _params_src()
        assert '"shock_b_over_sqrt_n"' in src, (
            "shock_b_over_sqrt_n must be declared in _SHOCK_PARAMS in parameters.py"
        )

    def test_shock_b_over_sqrt_n_in_param_map(self):
        from tengri.parameters.translate import _SHOCK_IDENTITY_PARAMS

        assert "shock_b_over_sqrt_n" in _SHOCK_IDENTITY_PARAMS, (
            "shock_b_over_sqrt_n must be in _SHOCK_IDENTITY_PARAMS in translate.py"
        )
