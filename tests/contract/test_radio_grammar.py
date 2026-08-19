# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for composable radio SF + AGN grammar.

Tests the additive radio grammar:
- radio={'sf':{'type':'delvecchio2021'}, 'agn':{'type':'dpl'}}
- back-compat: radio={'type':'bell2003'}
- 'none' mode disables individual sub-models
- grid-based forward-model builds + predicts finite

Marker: @pytest.mark.contract
"""

from __future__ import annotations

import pytest

from tengri import (
    FIXED,
    Fixed,
    SEDModel,
    Uniform,
)
from tengri.parameters import parse_groups

pytest.importorskip("dsps")  # Need DSPS for StellarSEDComponent

#: Zero-dust two-component block for predict_state({}) builds.
_DUST0 = {"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED}


@pytest.mark.contract
class TestRadioGrammarParsing:
    """Test radio grammar parsing and parameter routing."""

    def test_legacy_type_form_back_compat(self):
        """Legacy radio={'type': X} should work (back-compat)."""
        # Legacy 'type' (a RADIO_MODEL like condon92) predates the SF/AGN split;
        # it turns radio on with the default sf=bell2003 + agn=powerlaw models.
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"type": "condon92"},
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "bell2003"
        assert params.radio_agn_model == "powerlaw"  # default

    def test_legacy_type_none_disables(self):
        """Legacy radio={'type': 'none'} disables radio."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"type": "none"},
        )
        assert params.radio is False

    def test_composable_sf_only(self):
        """radio={'sf': {'type': 'delvecchio2021'}} enables SF only."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"sf": {"type": "delvecchio2021"}},
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "delvecchio2021"
        assert params.radio_agn_model == "powerlaw"  # default

    def test_composable_agn_only(self):
        """radio={'agn': {'type': 'dpl'}} enables AGN only."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"agn": {"type": "dpl"}},
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "bell2003"  # default
        assert params.radio_agn_model == "dpl"

    def test_composable_both_axes(self):
        """radio={'sf':{...}, 'agn':{...}} specifies both axes."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={
                "sf": {"type": "mccheyne2022"},
                "agn": {"type": "dpl"},
            },
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "mccheyne2022"
        assert params.radio_agn_model == "dpl"

    def test_sf_none_disables_sf_only(self):
        """radio={'sf': {'type': 'none'}} disables SF, keeps AGN."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={
                "sf": {"type": "none"},
                "agn": {"type": "powerlaw"},
            },
        )
        assert params.radio is True  # AGN is enabled
        assert params.radio_sfr_mode == "none"
        assert params.radio_agn_model == "powerlaw"

    def test_agn_none_disables_agn_only(self):
        """radio={'agn': {'type': 'none'}} disables AGN, keeps SF."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={
                "sf": {"type": "bell2003"},
                "agn": {"type": "none"},
            },
        )
        assert params.radio is True  # SF is enabled
        assert params.radio_sfr_mode == "bell2003"
        assert params.radio_agn_model == "none"

    def test_both_none_disables_radio(self):
        """radio={'sf': {'type': 'none'}, 'agn': {'type': 'none'}} disables radio."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={
                "sf": {"type": "none"},
                "agn": {"type": "none"},
            },
        )
        # Both sides 'none' → radio off (the sf/agn mode values are moot once off).
        assert params.radio is False

    def test_mixed_legacy_and_new_raises(self):
        """Mixing legacy 'type' with 'sf'/'agn' raises ValueError."""
        with pytest.raises(ValueError, match="cannot mix legacy"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={
                    "type": "bell2003",
                    "sf": {"type": "delvecchio2021"},
                },
            )

    def test_invalid_sf_variant_raises(self):
        """Invalid SF variant raises with helpful error."""
        with pytest.raises(ValueError, match="Unknown radio sf type"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"sf": {"type": "invalid_sf"}},
            )

    def test_invalid_agn_variant_raises(self):
        """Invalid AGN variant raises with helpful error."""
        with pytest.raises(ValueError, match="Unknown radio agn type"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"agn": {"type": "invalid_agn"}},
            )

    def test_sf_dict_not_dict_raises(self):
        """radio['sf'] must be a dict."""
        with pytest.raises(TypeError, match="radio\\['sf'\\] must be a dict"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"sf": "bell2003"},  # string, not dict
            )

    def test_agn_dict_not_dict_raises(self):
        """radio['agn'] must be a dict."""
        with pytest.raises(TypeError, match="radio\\['agn'\\] must be a dict"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"agn": "powerlaw"},  # string, not dict
            )


@pytest.mark.contract
class TestRadioBuilders:
    """Test the new builders.radio.sf and builders.radio.agn interface."""

    def test_builders_radio_sf_available(self):
        """builders.radio.sf.available() lists all SF variants."""
        from tengri import builders

        available_sf = builders.radio.sf.available()
        assert "none" in available_sf
        assert "bell2003" in available_sf
        assert "delvecchio2021" in available_sf
        assert "mccheyne2022" in available_sf

    def test_builders_radio_agn_available(self):
        """builders.radio.agn.available() lists all AGN variants."""
        from tengri import builders

        available_agn = builders.radio.agn.available()
        assert "none" in available_agn
        assert "powerlaw" in available_agn
        assert "dpl" in available_agn

    def test_builders_radio_axes_dict(self):
        """builders.radio.axes() returns the composable {sf, agn} axes dict."""
        from tengri import builders

        axes = builders.radio.axes()
        assert isinstance(axes, dict)
        assert "sf" in axes
        assert "agn" in axes
        assert isinstance(axes["sf"], list)
        assert isinstance(axes["agn"], list)
        # available() stays the legacy flat list (parallel with igm/xray).
        assert isinstance(builders.radio.available(), list)

    def test_builder_sf_bell2003_factory(self):
        """builders.radio.sf.bell2003() returns dict with type."""
        from tengri import builders

        result = builders.radio.sf.bell2003()
        assert isinstance(result, dict)
        assert result["type"] == "bell2003"

    def test_builder_sf_delvecchio_with_params(self):
        """builders.radio.sf.delvecchio2021(q_ir=...) sets a radio parameter."""
        from tengri import builders

        result = builders.radio.sf.delvecchio2021(q_ir=Uniform(2.4, 3.1))
        assert result["type"] == "delvecchio2021"
        assert "q_ir" in result
        assert isinstance(result["q_ir"], Uniform)

    def test_builder_agn_powerlaw_factory(self):
        """builders.radio.agn.powerlaw() returns dict with type."""
        from tengri import builders

        result = builders.radio.agn.powerlaw()
        assert isinstance(result, dict)
        assert result["type"] == "powerlaw"

    def test_builder_agn_dpl_with_params(self):
        """builders.radio.agn.dpl(...) sets DPL parameters."""
        from tengri import builders

        result = builders.radio.agn.dpl(
            alpha_thin=Uniform(-2, 0),
            alpha_thick=Uniform(-1, 1),
        )
        assert result["type"] == "dpl"
        assert "alpha_thin" in result
        assert "alpha_thick" in result


@pytest.mark.contract
class TestRadioComponentPhysics:
    """Test radio component physics with different grammar forms."""

    def test_model_build_sf_only(self, synthetic_ssp_wide):
        """SEDModel.build with radio SF only — selectors reach the spec."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "dpl", "*": FIXED},
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "none"}},
            redshift=Fixed(0.1),
        )
        assert model.spec.radio is True
        assert model.spec.radio_sfr_mode == "bell2003"
        assert model.spec.radio_agn_model == "none"

    def test_model_build_agn_only(self, synthetic_ssp_wide):
        """SEDModel.build with radio AGN only — dpl now reachable via grammar."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "dpl", "*": FIXED},
            radio={"sf": {"type": "none"}, "agn": {"type": "dpl"}},
            redshift=Fixed(0.1),
        )
        assert model.spec.radio is True
        assert model.spec.radio_sfr_mode == "none"
        assert model.spec.radio_agn_model == "dpl"

    def test_model_build_both_axes(self, synthetic_ssp_wide):
        """SEDModel.build with both SF and AGN radio."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "dpl", "*": FIXED},
            radio={"sf": {"type": "delvecchio2021"}, "agn": {"type": "dpl"}},
            redshift=Fixed(0.1),
        )
        assert model.spec.radio is True
        assert model.spec.radio_sfr_mode == "delvecchio2021"
        assert model.spec.radio_agn_model == "dpl"

    def test_model_predict_finite_sf_none(self, synthetic_ssp_wide):
        """Predict with SF='none' (AGN-only radio) produces a finite SED."""
        import jax.numpy as jnp

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust=_DUST0,
            radio={"sf": {"type": "none"}, "agn": {"type": "powerlaw"}},
            redshift=Fixed(0.1),
        )
        assert jnp.all(jnp.isfinite(model.predict_state({}).sed_intrinsic))

    def test_model_predict_finite_agn_none(self, synthetic_ssp_wide):
        """Predict with AGN='none' (SF-only radio) produces a finite SED."""
        import jax.numpy as jnp

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust=_DUST0,
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "none"}},
            redshift=Fixed(0.1),
        )
        assert jnp.all(jnp.isfinite(model.predict_state({}).sed_intrinsic))

    def test_model_predict_finite_both_axes(self, synthetic_ssp_wide):
        """Predict with both SF (mccheyne2022) and AGN (dpl) is finite."""
        import jax.numpy as jnp

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust=_DUST0,
            radio={"sf": {"type": "mccheyne2022"}, "agn": {"type": "dpl"}},
            redshift=Fixed(0.1),
        )
        assert jnp.all(jnp.isfinite(model.predict_state({}).sed_intrinsic))


@pytest.mark.contract
class TestRadioLegacyTypeRetirement:
    """PR6: Legacy radio={'type': X} form is retired.

    Users who relied on the flat legacy form must use the composable surface.
    The error message preserves the mapping so they can mechanically convert.
    """

    def test_legacy_condon92_type_raises_with_composable_equivalent(self):
        """radio={'type': 'condon92'} raises, showing composable form."""
        with pytest.raises(ValueError, match="legacy.*retired"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"type": "condon92"},
            )

    def test_legacy_none_type_raises(self):
        """radio={'type': 'none'} raises (use radio={'sf': None} instead)."""
        with pytest.raises(ValueError, match="legacy.*retired"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"type": "none"},
            )

    def test_legacy_radio_dpl_type_raises(self):
        """radio={'type': 'radio_dpl'} raises."""
        with pytest.raises(ValueError, match="legacy.*retired"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"type": "radio_dpl"},
            )

    def test_legacy_error_message_shows_mapping_for_condon92(self):
        """Error message for condon92 includes the composable equivalent."""
        with pytest.raises(ValueError) as excinfo:
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"type": "condon92"},
            )
        message = str(excinfo.value)
        # Should show the mapping: condon92 -> sf=bell2003, agn=powerlaw
        assert "radio=" in message
        assert "sf" in message
        assert "agn" in message or "bell2003" in message

    def test_composable_radio_sf_still_works(self):
        """radio={'sf': {'type': 'bell2003'}} still works (non-legacy form)."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"sf": {"type": "bell2003"}},
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "bell2003"

    def test_composable_radio_agn_still_works(self):
        """radio={'agn': {'type': 'powerlaw'}} still works (non-legacy form)."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"agn": {"type": "powerlaw"}},
        )
        assert params.radio is True
        assert params.radio_agn_model == "powerlaw"

    def test_composable_radio_both_axes_still_works(self):
        """radio={'sf': {...}, 'agn': {...}} still works (non-legacy form)."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={
                "sf": {"type": "bell2003"},
                "agn": {"type": "powerlaw"},
            },
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "bell2003"
        assert params.radio_agn_model == "powerlaw"
