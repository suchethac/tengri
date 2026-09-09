# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for composable radio SF + AGN grammar.

Tests the additive radio grammar:
- radio={'sf':{'type':'delvecchio2021'}, 'agn':{'type':'dpl'}}
- the legacy radio={'type': X} spelling is RETIRED (#1980) and raises
  with the composable equivalent in the message
- 'none' mode disables individual sub-models
- grid-based forward-model builds + predicts finite

Marker: @pytest.mark.contract
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.parameters import parse_groups

pytest.importorskip("dsps")  # Need DSPS for StellarSEDComponent

#: Zero-dust two-component block for predict_state({}) builds.
_DUST0 = {
    "type": "two_component",
    "law": "power_law",
    "tau_bc": Fixed(0.0),
    "tau_diff": Fixed(0.0),
    "all_params": Fixed(DEFAULT),
}


@pytest.mark.contract
class TestRadioGrammarParsing:
    """Test radio grammar parsing and parameter routing."""

    def test_legacy_type_form_retired(self):
        """Legacy radio={'type': X} form is retired (PR6)."""
        # Legacy 'type' form predates the SF/AGN split and is no longer accepted.
        # Users must use the composable surface with sf/agn axes.
        with pytest.raises(ValueError, match=r"legacy.*retired"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"type": "condon92"},
                redshift=Fixed(0.1),
            )

    def test_legacy_type_none_form_retired(self):
        """Legacy radio={'type': 'none'} form is retired (PR6)."""
        with pytest.raises(ValueError, match=r"legacy.*retired"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"type": "none"},
                redshift=Fixed(0.1),
            )

    def test_composable_sf_only(self):
        """radio={'sf': {'type': 'delvecchio2021'}} enables SF only."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={"sf": {"type": "delvecchio2021"}},
            redshift=Fixed(0.1),
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "delvecchio2021"
        assert params.radio_agn_model == "powerlaw"  # default

    def test_composable_agn_only(self):
        """radio={'agn': {'type': 'dpl'}} enables AGN only."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={"agn": {"type": "dpl"}},
            redshift=Fixed(0.1),
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "bell2003"  # default
        assert params.radio_agn_model == "dpl"

    def test_composable_both_axes(self):
        """radio={'sf':{...}, 'agn':{...}} specifies both axes."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={
                "sf": {"type": "mccheyne2022"},
                "agn": {"type": "dpl"},
            },
            redshift=Fixed(0.1),
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "mccheyne2022"
        assert params.radio_agn_model == "dpl"

    def test_sf_none_disables_sf_only(self):
        """radio={'sf': {'type': 'none'}} disables SF, keeps AGN."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={
                "sf": {"type": "none"},
                "agn": {"type": "powerlaw"},
            },
            redshift=Fixed(0.1),
        )
        assert params.radio is True  # AGN is enabled
        assert params.radio_sfr_mode == "none"
        assert params.radio_agn_model == "powerlaw"

    def test_agn_none_disables_agn_only(self):
        """radio={'agn': {'type': 'none'}} disables AGN, keeps SF."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={
                "sf": {"type": "bell2003"},
                "agn": {"type": "none"},
            },
            redshift=Fixed(0.1),
        )
        assert params.radio is True  # SF is enabled
        assert params.radio_sfr_mode == "bell2003"
        assert params.radio_agn_model == "none"

    def test_both_none_disables_radio(self):
        """radio={'sf': {'type': 'none'}, 'agn': {'type': 'none'}} disables radio."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={
                "sf": {"type": "none"},
                "agn": {"type": "none"},
            },
            redshift=Fixed(0.1),
        )
        # Both sides 'none' → radio off (the sf/agn mode values are moot once off).
        assert params.radio is False

    def test_mixed_legacy_and_new_raises(self):
        """Mixing legacy 'type' with 'sf'/'agn' raises, advising composable only.

        #1980: the message must NOT offer the retired legacy spelling as a
        valid alternative — the match pins the retirement wording.
        """
        with pytest.raises(ValueError, match=r"retired and cannot be mixed"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={
                    "type": "bell2003",
                    "sf": {"type": "delvecchio2021"},
                },
                redshift=Fixed(0.1),
            )

    def test_invalid_sf_variant_raises(self):
        """Invalid SF variant raises with helpful error."""
        with pytest.raises(ValueError, match="Unknown radio sf type"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"sf": {"type": "invalid_sf"}},
                redshift=Fixed(0.1),
            )

    def test_invalid_agn_variant_raises(self):
        """Invalid AGN variant raises with helpful error."""
        with pytest.raises(ValueError, match="Unknown radio agn type"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"agn": {"type": "invalid_agn"}},
                redshift=Fixed(0.1),
            )

    def test_sf_dict_not_dict_raises(self):
        """radio['sf'] must be a dict."""
        with pytest.raises(TypeError, match="radio\\['sf'\\] must be a dict"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"sf": "bell2003"},
                redshift=Fixed(0.1),  # string, not dict
            )

    def test_agn_dict_not_dict_raises(self):
        """radio['agn'] must be a dict."""
        with pytest.raises(TypeError, match="radio\\['agn'\\] must be a dict"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"agn": "powerlaw"},
                redshift=Fixed(0.1),  # string, not dict
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
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "none"}},
            # FIRRC normalizes against L_ir: dust is required at build (#2106).
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )
        assert model.spec.radio is True
        assert model.spec.radio_sfr_mode == "bell2003"
        assert model.spec.radio_agn_model == "none"

    def test_model_build_agn_only(self, synthetic_ssp_wide):
        """SEDModel.build with radio AGN only — dpl now reachable via grammar."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
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
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={"sf": {"type": "delvecchio2021"}, "agn": {"type": "dpl"}},
            # FIRRC normalizes against L_ir: dust is required at build (#2106).
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
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
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation=_DUST0,
            radio={"sf": {"type": "none"}, "agn": {"type": "powerlaw"}},
            redshift=Fixed(0.1),
        )
        assert jnp.all(jnp.isfinite(model.predict_state({}).sed_intrinsic))

    def test_model_predict_finite_agn_none(self, synthetic_ssp_wide):
        """Predict with AGN='none' (SF-only radio) produces a finite SED."""
        import jax.numpy as jnp

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation=_DUST0,
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "none"}},
            redshift=Fixed(0.1),
        )
        assert jnp.all(jnp.isfinite(model.predict_state({}).sed_intrinsic))

    def test_model_predict_finite_both_axes(self, synthetic_ssp_wide):
        """Predict with both SF (mccheyne2022) and AGN (dpl) is finite."""
        import jax.numpy as jnp

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation=_DUST0,
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
        with pytest.raises(ValueError, match=r"legacy.*retired"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"type": "condon92"},
                redshift=Fixed(0.1),
            )

    def test_legacy_none_type_raises(self):
        """radio={'type': 'none'} raises (use radio={'sf': None} instead)."""
        with pytest.raises(ValueError, match=r"legacy.*retired"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"type": "none"},
                redshift=Fixed(0.1),
            )

    def test_legacy_radio_dpl_type_raises(self):
        """radio={'type': 'radio_dpl'} raises."""
        with pytest.raises(ValueError, match=r"legacy.*retired"):
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"type": "radio_dpl"},
                redshift=Fixed(0.1),
            )

    def test_legacy_error_message_shows_mapping_for_condon92(self):
        """Error message for condon92 includes the composable equivalent."""
        with pytest.raises(ValueError) as excinfo:
            parse_groups(
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                radio={"type": "condon92"},
                redshift=Fixed(0.1),
            )
        message = str(excinfo.value)
        # Should show the mapping: condon92 -> sf=bell2003, agn=powerlaw
        assert "radio=" in message
        assert "sf" in message
        assert "agn" in message or "bell2003" in message

    def test_composable_radio_sf_still_works(self):
        """radio={'sf': {'type': 'bell2003'}} still works (non-legacy form)."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={"sf": {"type": "bell2003"}},
            redshift=Fixed(0.1),
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "bell2003"

    def test_composable_radio_agn_still_works(self):
        """radio={'agn': {'type': 'powerlaw'}} still works (non-legacy form)."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={"agn": {"type": "powerlaw"}},
            redshift=Fixed(0.1),
        )
        assert params.radio is True
        assert params.radio_agn_model == "powerlaw"

    def test_composable_radio_both_axes_still_works(self):
        """radio={'sf': {...}, 'agn': {...}} still works (non-legacy form)."""
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={
                "sf": {"type": "bell2003"},
                "agn": {"type": "powerlaw"},
            },
            redshift=Fixed(0.1),
        )
        assert params.radio is True
        assert params.radio_sfr_mode == "bell2003"
        assert params.radio_agn_model == "powerlaw"


@pytest.mark.contract
class TestDaleRadioGuardMeasuresTheTemplate:
    """The #1970 refusal must key on the grid's red edge, not on its name (R58).

    Dale+2014's published templates embed a star-forming radio synchrotron
    continuum, so pairing them with an active SF radio block double-counts the
    radio. The guard refused the *name* ``'dale2014'``, which is neither
    sufficient nor necessary:

    * a tail-free grid registered under the name ``dale2014`` -- exactly what
      ``register_dale2014_tabulated(cigale_grid, name='dale2014')`` produces --
      was refused although it carries no radio to double-count;
    * a tail-bearing grid registered under any other name was accepted.

    Measured red edges (reddest wavelength with non-zero flux, union over
    every alpha row): ``data/dale2014_templates.h5`` reaches **2.2459e9 A**
    (1.335 GHz) and ``data/dale2014_templates_cigale.h5`` stops at
    **7.727e7 A**, the strip edge ``Dale2014CigaleIRSEDComponent`` documents.
    The 1e8 A threshold (1 cm, 30 GHz) sits between them -- 22x below the
    tail-bearing one, 1.29x above the stripped one -- and blueward of the
    whole 1.34-10 GHz double-count window.

    The union matters: a single row is not the grid. The ``alpha=2.0`` row
    alone stops at 6.026e7 A, 1.28x blueward of the 64-row union, and a
    build-time refusal has to hold for every alpha the model can reach.

    Reach alone is not enough, either: the tail must be **rising** in L_nu,
    which is what synchrotron does and cold dust does not.
    ``data/astrodust_templates.h5`` emits out to 3.0e8 A on its spinning-dust
    component -- past the threshold -- and double-counts nothing. Measured
    red-end slopes: dale2014 **+0.665** against -3.111 (bosa), -3.326
    (astrodust), -4.810 (schreiber2016), -5.510 (dale2014_cigale).
    """

    _RADIO_SF: ClassVar[dict] = {"sf": {"type": "bell2003"}, "agn": {"type": "none"}}

    def _build(self, ssp, emission_type, *, radio=True):
        kw = {}
        if radio:
            kw["radio"] = self._RADIO_SF
        return SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": emission_type, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
            **kw,
        )

    #: ``registry name -> (grid file, red edge [A], red-end dlogLnu/dloglam)``,
    #: every template-backed emission model whose grid ships here. Only the
    #: first is a radio tail: the others either stop blueward of 1e8 A or fall
    #: steeply in L_nu where a synchrotron tail rises.
    _MEASURED_RED_ENDS: ClassVar[dict[str, tuple[str, float, float]]] = {
        "dale2014": ("data/dale2014_templates.h5", 2.245912e9, +0.665),
        "dale2014_cigale": ("data/dale2014_templates_cigale.h5", 7.727e7, -5.510),
        "astrodust": ("data/astrodust_templates.h5", 3.0e8, -3.326),
        "bosa": ("data/bosa_templates.h5", 1.0e8, -3.111),
        "schreiber2016": ("data/schreiber2016_templates.h5", 3.001310e7, -4.810),
    }

    @pytest.mark.parametrize("name", sorted(_MEASURED_RED_ENDS))
    def test_measured_red_end_of_every_shipped_grid(self, name):
        """Pin the edge AND the slope for every grid the guard can see."""
        import os

        from tengri.components.dust.emission_templates import _red_end_from_grid_file

        path, edge, slope = self._MEASURED_RED_ENDS[name]
        if not os.path.exists(path):
            pytest.skip(f"{path} not available")
        got_edge, got_slope = _red_end_from_grid_file(path)
        assert got_edge == pytest.approx(edge, rel=1e-3, abs=0.0)
        assert got_slope == pytest.approx(slope, rel=1e-2, abs=0.0)

    @pytest.mark.parametrize("name", sorted(_MEASURED_RED_ENDS))
    def test_only_dale2014_reads_as_a_radio_tail(self, name):
        """The rising-slope condition is what keeps astrodust out.

        astrodust reaches 3.0e8 A -- past the 1e8 A threshold -- so an
        edge-only test would newly refuse ``astrodust`` + SF radio, which
        double-counts nothing.
        """
        import os

        from tengri.components.dust.emission_templates import dust_emission_radio_tail_aa

        path, edge, _slope = self._MEASURED_RED_ENDS[name]
        if not os.path.exists(path):
            pytest.skip(f"{path} not available")
        tail = dust_emission_radio_tail_aa(name)
        if name == "dale2014":
            assert tail == pytest.approx(edge, rel=1e-3, abs=0.0)
        else:
            assert tail is None, f"{name} must not read as a radio tail"

    def test_builtin_dale2014_still_refused_with_sf_radio(self, synthetic_ssp_wide):
        """#1970's own case keeps raising: the tail-bearing grid is unsafe."""
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError, match=r"radio"):
            self._build(synthetic_ssp_wide, "dale2014")

    def test_refusal_names_the_offending_red_edge(self, synthetic_ssp_wide):
        """The message must carry the measured edge, not just the name."""
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError) as exc:
            self._build(synthetic_ssp_wide, "dale2014")
        msg = str(exc.value)
        assert "2.2459e+09" in msg or "2.246e+09" in msg, (
            f"the refusal must name the template's measured red edge; got: {msg}"
        )

    def test_builtin_dale2014_builds_without_sf_radio(self, synthetic_ssp_wide):
        """No SF radio, nothing to double-count."""
        model = self._build(synthetic_ssp_wide, "dale2014", radio=False)
        assert model.spec.dust_emission == "dale2014"

    def test_tail_free_grid_registered_as_dale2014_builds_with_radio(
        self, synthetic_ssp_wide, monkeypatch
    ):
        """R58: the CIGALE grid under the name ``dale2014`` carries no radio.

        This is the case ``eae23ba02`` had to work around in the reproduction
        notebook, and the name test refused it.
        """
        import os

        from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS
        from tengri.components.dust.emission_templates import register_dale2014_tabulated

        path = "data/dale2014_templates_cigale.h5"
        if not os.path.exists(path):
            pytest.skip(f"{path} not available")
        saved = DUST_EMISSION_MODELS.get("dale2014")
        register_dale2014_tabulated(path, name="dale2014")
        try:
            model = self._build(synthetic_ssp_wide, "dale2014")
            assert model.spec.dust_emission == "dale2014"
        finally:
            if saved is not None:
                DUST_EMISSION_MODELS["dale2014"] = saved

    def test_tail_bearing_grid_under_the_tail_free_name_is_refused(self, synthetic_ssp_wide):
        """R58's mirror: the tail follows the data, so the refusal must too.

        Registering the radio-bearing grid under ``dale2014_cigale`` -- the
        name whose whole point is that its tail is stripped -- must be
        refused. The name test accepted it. (An arbitrary new registry name
        is not reachable here: ``dust_emission={'type': ...}`` validates
        against the component registry and raises ``ValueError`` for a name
        it does not know, so ``dale2014_cigale`` is the reachable mirror.)
        """
        import os

        from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS
        from tengri.components.dust.emission_templates import register_dale2014_tabulated
        from tengri.config.exceptions import ConfigError

        path = "data/dale2014_templates.h5"
        if not os.path.exists(path):
            pytest.skip(f"{path} not available")
        saved = DUST_EMISSION_MODELS.get("dale2014_cigale")
        register_dale2014_tabulated(path, name="dale2014_cigale")
        try:
            with pytest.raises(ConfigError, match=r"radio"):
                self._build(synthetic_ssp_wide, "dale2014_cigale")
        finally:
            if saved is not None:
                DUST_EMISSION_MODELS["dale2014_cigale"] = saved
            else:
                DUST_EMISSION_MODELS.pop("dale2014_cigale", None)

    def test_astrodust_builds_with_sf_radio(self, synthetic_ssp_wide):
        """The control the rising-slope condition exists for.

        astrodust's emitting span reaches 3.0e8 A, past the 1e8 A threshold,
        but falls at -3.326 in L_nu: spinning dust, not synchrotron.
        """
        import os

        if not os.path.exists("data/astrodust_templates.h5"):
            pytest.skip("astrodust grid not available")
        model = self._build(synthetic_ssp_wide, "astrodust")
        assert model.spec.dust_emission == "astrodust"

    def test_tail_free_builtin_variant_builds_with_radio(self, synthetic_ssp_wide):
        """``dale2014_cigale`` keeps working, by measurement now not by name."""
        model = self._build(synthetic_ssp_wide, "dale2014_cigale")
        assert model.spec.dust_emission == "dale2014_cigale"
