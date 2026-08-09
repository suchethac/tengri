# SPDX-License-Identifier: BSD-3-Clause
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

pytestmark = pytest.mark.contract


#: Domain prefixes that ``SEDModelComponent.slice_params`` strips before
#: :meth:`predict` sees the dict. A modern component reads ``p["tau_skirtor"]``,
#: never ``p["agn_tau_skirtor"]``, so a scan for the fully-qualified spelling
#: alone cannot see live wiring (#1403).
_DOMAIN_PREFIXES = ("agn_", "dust_", "neb_", "shock_", "radio_", "xray_", "igm_", "sfh_", "met_")


def _has_param_consumer(src: str, param: str) -> bool:
    """True if ``src`` reads ``param`` from a params dict, in any live spelling.

    Accepts the prefix-stripped name as well as the fully-qualified one.
    ``SEDModelComponent`` strips its ``parameter_prefix`` in ``slice_params``
    before calling ``predict``, so ``skirtor_model.py`` legitimately reads
    ``p["tau_skirtor"]`` for the parameter declared as ``agn_tau_skirtor``.

    Checking only the qualified spelling used to pass for the wrong reason: the
    sole match was ``SEDModel._get_non_stellar_kwargs``, a method that nothing
    called. Deleting it as dead code (#1403) turned these assertions red while
    the wiring they describe was, and still is, entirely intact — so the test
    had been reporting on a string in a dead method rather than on the pipeline.
    Had the real forwarding in ``skirtor_model.py`` broken, this would have
    stayed green.
    """
    names = [param]
    for prefix in _DOMAIN_PREFIXES:
        if param.startswith(prefix):
            names.append(param[len(prefix) :])
            break
    return any(
        f'params.get("{name}"' in src
        or f'params["{name}"]' in src
        or f'p.get("{name}"' in src
        or f'p["{name}"]' in src
        for name in names
    )


def _pipeline_src() -> str:
    """Aggregated source of every module that consumes physics-level params.

    Originally pinned to the legacy ``forward.pipeline.compute_sed_components``
    body, deleted in Phase B closure. Its parameter-forwarding contract now
    lives across the orchestrator's component adapters and helpers.

    ``forward.pipeline`` itself was still listed here until 2026-08, long after
    it stopped holding anything this scan cares about: by then the module was
    two metallicity dispatchers, and it contributed no AGN, X-ray, or radio
    parameter text to the aggregate. The module has since been removed
    entirely — every symbol in it was unreferenced — so the import went with it.
    """
    from tengri.components.agn import (
        component as agn_component,
        kd18_disc_model as agn_kd18_disc,
        skirtor_model as agn_skirtor,
        unified as agn_unified,
    )
    from tengri.components.dust import (
        component as dust_component,
        two_component as dust_two,
    )
    from tengri.components.radio import component as radio_component
    from tengri.components.xray import component as xray_component
    from tengri.forward import sed_model as sed_model_mod

    parts = [
        inspect.getsource(sed_model_mod),
        inspect.getsource(agn_component),
        inspect.getsource(agn_unified),
        # The SEDModelComponent torus/disc blocks — where the SKIRTOR and KD18
        # parameters are actually forwarded (#1403). Without these the scan can
        # only see the legacy fully-qualified spelling, which no live code uses.
        inspect.getsource(agn_skirtor),
        inspect.getsource(agn_kd18_disc),
        inspect.getsource(dust_component),
        inspect.getsource(dust_two),
        inspect.getsource(radio_component),
        inspect.getsource(xray_component),
    ]
    return "\n".join(parts)


def _model_src() -> str:
    from tengri.forward import sed_model as model_mod

    return inspect.getsource(model_mod)


def _emission_helpers_src() -> str:
    from tengri.forward import emission_helpers

    return inspect.getsource(emission_helpers)


def _params_src() -> str:
    from tengri.parameters import _builders

    return inspect.getsource(_builders)


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
        from tengri.parameters._builders import _resolve_lazy_bucket

        _AGN_PARAMS = _resolve_lazy_bucket("_AGN_PARAMS")

        assert "agn_a_spin" in _AGN_PARAMS, (
            "agn_a_spin must be declared in _AGN_PARAMS "
            "(canonical source: tengri.components.agn._params.PARAMS)"
        )

    def test_agn_a_spin_in_param_map(self):
        from tengri.parameters.registry import registry

        agn_a_spin_record = registry().get("agn_a_spin")
        assert agn_a_spin_record is not None, "agn_a_spin must be in the parameter registry"


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
        from tengri.parameters.registry import registry

        param_record = registry().get(param)
        assert param_record is not None, f"{param} must be in the parameter registry"


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
        from tengri.parameters._builders import _resolve_lazy_bucket

        _AGN_PARAMS = _resolve_lazy_bucket("_AGN_PARAMS")

        assert param in _AGN_PARAMS, (
            f"{param} must be declared in _AGN_PARAMS "
            "(canonical source: tengri.components.agn._params.PARAMS)"
        )


# ── Polar dust forwarding ─────────────────────────────────────────
class TestPolarDustForwarding:
    """Polar dust (agn_polar_ebv, agn_polar_oa) must gate and forward correctly."""

    def test_polar_ebv_declared_in_params(self):
        from tengri.parameters._builders import _resolve_lazy_bucket

        _AGN_PARAMS = _resolve_lazy_bucket("_AGN_PARAMS")

        assert "agn_polar_ebv" in _AGN_PARAMS

    def test_polar_oa_declared_in_params(self):
        from tengri.parameters._builders import _resolve_lazy_bucket

        _AGN_PARAMS = _resolve_lazy_bucket("_AGN_PARAMS")

        assert "agn_polar_oa" in _AGN_PARAMS

    def test_polar_dust_is_a_noop_at_zero_ebv(self):
        """Polar dust must do nothing when ``agn_polar_ebv == 0``.

        Previously asserted by grepping ``emission_helpers.agn_emission`` for an
        ``agn_polar_ebv > 0.0`` branch. That helper was a dead duplicate, so the
        test passed on code nothing ran, and it pinned a design the live path had
        abandoned: ``polar_dust_total`` is branchless because
        ``exp(-0.921 * ebv * ...)`` is already the identity at ``ebv = 0`` (a
        Python-level branch would not survive ``jax.jit``). Assert the invariant
        the guard existed to protect instead of any particular implementation.
        """
        import jax.numpy as jnp
        import numpy as np

        from tengri.components.agn.polar_dust import polar_dust_total

        wave = jnp.linspace(1000.0, 30000.0, 128)
        l_nu_disc = jnp.ones_like(wave)
        atten, emis = polar_dust_total(
            l_nu_disc, wave, cos_inc=0.9, opening_angle_deg=40.0, ebv=0.0
        )
        np.testing.assert_allclose(np.asarray(atten), np.asarray(l_nu_disc), rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(emis), 0.0, atol=0)

    def test_polar_dust_block_uses_cos_inc(self):
        src = _pipeline_src()
        assert _has_param_consumer(src, "agn_cos_inc"), (
            "polar_dust_total call must read agn_cos_inc from params"
        )

    def test_polar_dust_block_uses_opening_angle(self):
        """``agn_polar_oa`` must be read from params and actually change the result.

        The AGN component reads ``agn_polar_oa`` from params and passes it to
        ``polar_dust_total(..., opening_angle_deg=...)`` in
        ``components/agn/polar_dust.py``. (The old docstring described a
        ``sed_pipeline.py`` -> ``emission_helpers.agn_emission`` chain; neither
        exists any more — that module was deleted and that helper was a dead
        duplicate.) Both sides are checked: the parameter is consumed, and
        varying it moves the output.
        """
        pipeline_src = _pipeline_src()
        assert _has_param_consumer(pipeline_src, "agn_polar_oa"), (
            "AGN component must read agn_polar_oa from params"
        )
        # ...and the live polar-dust model must actually consume it. Asserting the
        # text "opening_angle_deg=agn_polar_oa" only ever matched the dead
        # emission_helpers copy; vary the angle and require the output to move.
        import jax.numpy as jnp
        import numpy as np

        from tengri.components.agn.polar_dust import polar_dust_total

        wave = jnp.linspace(1000.0, 30000.0, 128)
        disc = jnp.ones_like(wave)
        narrow, _ = polar_dust_total(disc, wave, cos_inc=0.5, opening_angle_deg=10.0, ebv=0.3)
        wide, _ = polar_dust_total(disc, wave, cos_inc=0.5, opening_angle_deg=80.0, ebv=0.3)
        assert not np.allclose(np.asarray(narrow), np.asarray(wide)), (
            "opening_angle_deg must change the polar-dust result — it is being ignored"
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
        from tengri.parameters._builders import _resolve_lazy_bucket

        _XRAY_PARAMS = _resolve_lazy_bucket("_XRAY_PARAMS")

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
        from tengri.parameters._builders import _resolve_lazy_bucket

        _RADIO_PARAMS = _resolve_lazy_bucket("_RADIO_PARAMS")

        assert param in _RADIO_PARAMS, (
            f"{param} must be declared in _RADIO_PARAMS "
            "(canonical source: tengri.components.radio._params.PARAMS)"
        )


# ── Dust emission alpha_dl14 ──────────────────────────────────────
class TestDustAlphaDL14:
    """dust_alpha_dl14 must be forwarded to the dust emission call."""

    def test_dust_alpha_dl14_in_pipeline(self):
        """The live DL14 template must receive ``dust_alpha_dl14``.

        This used to check ``_pipeline_src()`` for a ``params["dust_alpha_dl14"]``
        style read, which only matched the dead ``emission_helpers.dust_ir_emission``
        copy. The live path forwards it prefix-stripped
        (``dust_alpha_dl14=p["alpha_dl14"]`` in ``dust/emission/templates/draine_li.py``),
        so the old pattern never described live code at all.
        """
        import inspect

        from tengri.components.dust.emission.templates import draine_li

        src = inspect.getsource(draine_li)
        assert "dust_alpha_dl14=" in src, (
            "DL14 template must forward dust_alpha_dl14 to the emission call"
        )

    def test_dust_alpha_dl14_in_param_map(self):
        from tengri.parameters.registry import registry

        param_record = registry().get("dust_alpha_dl14")
        assert param_record is not None, "dust_alpha_dl14 must be in the parameter registry"


# ── Shock b_over_sqrt_n registration ──────────────────────────────
class TestShockBOverSqrtN:
    """shock_b_over_sqrt_n must be in _param_map when shock is enabled."""

    def test_shock_b_over_sqrt_n_declared_in_params(self):
        from tengri.parameters._builders import _resolve_lazy_bucket

        _SHOCK_PARAMS = _resolve_lazy_bucket("_SHOCK_PARAMS")

        assert "shock_b_over_sqrt_n" in _SHOCK_PARAMS, (
            "shock_b_over_sqrt_n must be declared in _SHOCK_PARAMS "
            "(canonical source: tengri.components.nebular._params.SHOCK_PARAMS)"
        )

    def test_shock_b_over_sqrt_n_in_param_map(self):
        from tengri.parameters.registry import registry

        param_record = registry().get("shock_b_over_sqrt_n")
        assert param_record is not None, "shock_b_over_sqrt_n must be in the parameter registry"
