# SPDX-License-Identifier: BSD-3-Clause
"""FIR-radio correlation (FIRRC) evolution coefficients as free parameters.

The evolving SF-radio models (Delvecchio+2021 at 1.4 GHz, McCheyne+2022 at
150 MHz) carry a mass- and redshift-dependent ``q_IR(M*, z)`` governed by
three coefficients ``(q0, mass_slope, z_slope)``. These are now surfaced as
fittable, model-specific ``radio_delv_*`` / ``radio_mcch_*`` parameters,
addressable through the composable ``radio={'sf': {...}}`` sub-block grammar.

Naming is model-specific because the two calibrations have genuinely
different literature defaults and mass-slope sign conventions, so a single
shared name would silently apply the wrong default to the inactive model.
Only the active ``sfr_mode``'s triplet is consumed by the component.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import FIXED, Fixed, SEDModel, Uniform
from tengri.components.radio.component import RadioSEDComponent, RadioSEDComponentConfig
from tengri.parameters import parse_groups
from tengri.parameters._builders import _resolve_lazy_bucket
from tengri.protocols import ForwardState

pytestmark = pytest.mark.contract

_FIRRC_PARAMS = (
    "radio_delv_q0",
    "radio_delv_mass_slope",
    "radio_delv_z_slope",
    "radio_mcch_q0",
    "radio_mcch_mass_slope",
    "radio_mcch_z_slope",
)

#: Zero-dust two-component block for predict_state({}) builds.
_DUST0 = {
    "type": "two_component",
    "law": "power_law",
    "tau_bc": Fixed(0.0),
    "tau_diff": Fixed(0.0),
    "*": FIXED,
}


@pytest.mark.unit
class TestFirrcParamRegistry:
    """The six FIRRC coefficients are declared under the radio_ prefix."""

    @pytest.mark.parametrize("name", _FIRRC_PARAMS)
    def test_declared_in_param_defs(self, name):
        radio_params = _resolve_lazy_bucket("_RADIO_PARAMS")
        assert name in radio_params, f"{name!r} missing from _RADIO_PARAMS"

    @pytest.mark.parametrize("name", _FIRRC_PARAMS)
    def test_declared_by_component(self, name):
        decls = {d.name for d in RadioSEDComponent().declared_parameters()}
        assert name in decls

    def test_literature_defaults(self):
        priors = {d.name: d.prior for d in RadioSEDComponent().declared_parameters()}
        # Delvecchio+2021 best-fit (SEMPER Eq. 4).
        assert float(priors["radio_delv_q0"].value) == pytest.approx(2.743)
        assert float(priors["radio_delv_mass_slope"].value) == pytest.approx(0.234)
        assert float(priors["radio_delv_z_slope"].value) == pytest.approx(-0.025)
        # McCheyne+2022 best-fit (SEMPER Eq. 5) — distinct values + sign.
        assert float(priors["radio_mcch_q0"].value) == pytest.approx(1.98)
        assert float(priors["radio_mcch_mass_slope"].value) == pytest.approx(-0.22)
        assert float(priors["radio_mcch_z_slope"].value) == pytest.approx(0.02)


@pytest.mark.unit
class TestFirrcGrammar:
    """Free the coefficients through the radio={'sf': {...}} sub-block."""

    def test_delv_per_param_free(self):
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"sf": {"type": "delvecchio2021", "delv_q0": Uniform(2.4, 3.1)}},
        )
        assert "radio_delv_q0" in params.free_params

    def test_mcch_per_param_free(self):
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"sf": {"type": "mccheyne2022", "mcch_q0": Uniform(1.5, 2.5)}},
        )
        assert "radio_mcch_q0" in params.free_params

    def test_full_name_also_accepted(self):
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"sf": {"type": "delvecchio2021", "radio_delv_q0": Uniform(2.4, 3.1)}},
        )
        assert "radio_delv_q0" in params.free_params

    def test_dpl_agn_per_param_free(self):
        """The symmetric radio={'agn': {...}} sub-block frees DPL knobs."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"agn": {"type": "dpl", "alpha_thin": Uniform(-1.5, 0.0)}},
        )
        assert "radio_alpha_thin" in params.free_params

    def test_typo_in_sf_block_raises(self):
        with pytest.raises(ValueError, match="Unknown key 'delv_q00'"):
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"sf": {"type": "delvecchio2021", "delv_q00": Uniform(2.0, 3.0)}},
            )

    def test_cross_model_param_not_freed_by_default(self):
        """Freeing a delv coeff leaves the mcch coeffs fixed (and vice versa)."""
        params = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            radio={"sf": {"type": "delvecchio2021", "delv_q0": Uniform(2.4, 3.1)}},
        )
        freed = [n for n in params.free_params if n.startswith("radio_")]
        assert freed == ["radio_delv_q0"]


@pytest.mark.unit
class TestFirrcSlopeDegeneracyGuard:
    """Freeing a FIRRC *slope* per-galaxy is degenerate — warn, don't crash.

    The slopes vary q_IR across a sample; at one galaxy's fixed (M*, z) they
    collapse to a single scalar (degenerate with the q0 normalization). They
    are identifiable only as PopulationFitter hyperparameters.
    """

    @pytest.mark.parametrize(
        "radio_sf",
        [
            {"type": "delvecchio2021", "delv_mass_slope": Uniform(0.0, 0.5)},
            {"type": "delvecchio2021", "delv_z_slope": Uniform(-0.2, 0.05)},
            {"type": "mccheyne2022", "mcch_mass_slope": Uniform(-0.5, 0.0)},
            {"type": "mccheyne2022", "mcch_z_slope": Uniform(-0.1, 0.2)},
        ],
    )
    def test_free_slope_warns(self, radio_sf):
        from tengri.components.radio._params import RadioFIRRCDegeneracyWarning

        with pytest.warns(RadioFIRRCDegeneracyWarning, match="degenerate"):
            parse_groups(sfh={"type": "dpl", "*": FIXED}, radio={"sf": radio_sf})

    def test_free_q0_does_not_warn(self):
        """The q0 normalization is the legitimate per-galaxy radio-excess knob."""
        import warnings

        from tengri.components.radio._params import RadioFIRRCDegeneracyWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", RadioFIRRCDegeneracyWarning)
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"sf": {"type": "delvecchio2021", "delv_q0": Uniform(2.4, 3.1)}},
            )

    def test_all_fixed_does_not_warn(self):
        import warnings

        from tengri.components.radio._params import RadioFIRRCDegeneracyWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", RadioFIRRCDegeneracyWarning)
            parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"sf": {"type": "delvecchio2021"}},
            )

    def test_warning_is_filterable(self):
        """A deliberate hierarchical fit can silence the category."""
        import warnings

        from tengri.components.radio._params import RadioFIRRCDegeneracyWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RadioFIRRCDegeneracyWarning)
            params = parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                radio={"sf": {"type": "mccheyne2022", "mcch_mass_slope": Uniform(-0.5, 0.0)}},
            )
        # The param is still freed — the guard only warns, never blocks.
        assert "radio_mcch_mass_slope" in params.free_params


@pytest.mark.unit
class TestFirrcBuilderGrammar:
    """The builder-factory surface is the canonical way to free the coeffs.

    ``builders.radio.sf.<variant>(...)`` / ``builders.radio.agn.<variant>(...)``
    are the discoverable, autocomplete-friendly entry points; the factory dict
    must round-trip through the grammar to a freed ``radio_*`` parameter.
    """

    def test_sf_factory_round_trips_to_free_param(self):
        from tengri import builders

        sf = builders.radio.sf.delvecchio2021(delv_q0=Uniform(2.4, 3.1))
        assert sf["type"] == "delvecchio2021"
        params = parse_groups(sfh={"type": "dpl", "*": FIXED}, radio={"sf": sf})
        assert "radio_delv_q0" in params.free_params

    def test_agn_factory_round_trips_to_free_param(self):
        from tengri import builders

        agn = builders.radio.agn.dpl(alpha_thin=Uniform(-1.5, 0.0))
        assert agn["type"] == "dpl"
        params = parse_groups(sfh={"type": "dpl", "*": FIXED}, radio={"agn": agn})
        assert "radio_alpha_thin" in params.free_params

    def test_build_via_factory_threads_and_predicts(self, synthetic_ssp_wide):
        """Full canonical surface: factory → SEDModel.build → predict finite."""
        from tengri import builders

        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust={
                "type": "two_component",
                "law": "power_law",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "*": FIXED,
            },
            radio={"sf": builders.radio.sf.mccheyne2022(mcch_q0=Uniform(1.5, 2.5))},
            redshift=Fixed(0.3),
        )
        assert "radio_mcch_q0" in model.spec.free_params
        assert model.spec.radio_sfr_mode == "mccheyne2022"
        sed = model.predict_state({"radio_mcch_q0": 2.0}).sed_intrinsic
        assert jnp.all(jnp.isfinite(sed))


@pytest.mark.unit
class TestFirrcComponentThreading:
    """The freed coefficient actually drives the radio SED (no silent drop)."""

    @pytest.fixture
    def state(self):
        wave = jnp.linspace(1e3, 1e9, 400)
        return ForwardState(
            wave=wave,
            sed_intrinsic=jnp.zeros_like(wave),
            sed_observed=jnp.ones_like(wave),
            derived={"L_ir": 1e44, "L_agn_bol": 1e45, "log_mstar": 10.8},
        )

    def _params(self):
        return {
            "redshift": 1.2,
            "radio_q_ir": 2.64,
            "radio_alpha_sf": 0.7,
            "radio_loudness": 1.0,
            "radio_alpha_agn": 0.7,
            "radio_T_e": 1e4,
            "radio_alpha_ff": -0.1,
            "radio_alpha_thin": -0.75,
            "radio_alpha_thick": -0.1,
            "radio_log_nu_t": 10.0,
            "radio_log_nu_cut": 13.0,
            "radio_delv_q0": 2.743,
            "radio_delv_mass_slope": 0.234,
            "radio_delv_z_slope": -0.025,
            "radio_mcch_q0": 1.98,
            "radio_mcch_mass_slope": -0.22,
            "radio_mcch_z_slope": 0.02,
        }

    def test_delv_q0_changes_sed_and_direction(self, state):
        """Higher q0 → fainter radio (q ≡ log10(L_IR / L_radio))."""
        cfg = RadioSEDComponentConfig(sfr_mode="delvecchio2021", agn_radio_model="none")
        comp = RadioSEDComponent(config=cfg)
        sed_lo = comp.apply(state, self._params()).derived["sed_radio"]
        p_hi = self._params() | {"radio_delv_q0": 3.2}
        sed_hi = comp.apply(state, p_hi).derived["sed_radio"]
        assert not jnp.allclose(sed_lo, sed_hi)
        radio = state.wave > 1e7
        assert float(jnp.trapezoid(sed_hi[radio], state.wave[radio])) < float(
            jnp.trapezoid(sed_lo[radio], state.wave[radio])
        )

    def test_mcch_coeff_threads_under_mccheyne_mode(self, state):
        cfg = RadioSEDComponentConfig(sfr_mode="mccheyne2022", agn_radio_model="none")
        comp = RadioSEDComponent(config=cfg)
        sed = comp.apply(state, self._params()).derived["sed_radio"]
        p2 = self._params() | {"radio_mcch_q0": 2.5}
        assert not jnp.allclose(sed, comp.apply(state, p2).derived["sed_radio"])

    def test_inactive_model_coeff_is_noop(self, state):
        """A delv coefficient must not perturb the SED under mccheyne mode."""
        cfg = RadioSEDComponentConfig(sfr_mode="mccheyne2022", agn_radio_model="none")
        comp = RadioSEDComponent(config=cfg)
        sed = comp.apply(state, self._params()).derived["sed_radio"]
        p2 = self._params() | {"radio_delv_q0": 9.9}
        assert jnp.allclose(sed, comp.apply(state, p2).derived["sed_radio"])

    def test_bell2003_ignores_firrc_coeffs(self, state):
        """The fixed-q bell2003 mode is unaffected by the evolution coeffs."""
        cfg = RadioSEDComponentConfig(sfr_mode="bell2003", agn_radio_model="none")
        comp = RadioSEDComponent(config=cfg)
        sed = comp.apply(state, self._params()).derived["sed_radio"]
        p2 = self._params() | {"radio_delv_q0": 9.9, "radio_mcch_q0": 9.9}
        assert jnp.allclose(sed, comp.apply(state, p2).derived["sed_radio"])


@pytest.mark.unit
class TestFirrcEndToEnd:
    """SEDModel.build with a freed FIRRC coefficient predicts finite."""

    def test_build_fixed_delvecchio_predicts_finite(self, synthetic_ssp_wide):
        """A delvecchio2021 build with default (fixed) coeffs predicts finite
        through ``predict_state({})`` — exercises the component threading."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust=_DUST0,
            radio={"sf": {"type": "delvecchio2021"}},
            redshift=Fixed(0.5),
        )
        assert model.spec.radio_sfr_mode == "delvecchio2021"
        assert jnp.all(jnp.isfinite(model.predict_state({}).sed_intrinsic))

    def test_build_free_delv_q0_predicts_finite(self, synthetic_ssp_wide):
        """Freeing delv_q0 surfaces it in the free params; predict with a
        supplied value is finite (``predict_state({})`` would omit free params)."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "*": FIXED},
            dust=_DUST0,
            radio={"sf": {"type": "delvecchio2021", "delv_q0": Uniform(2.4, 3.1)}},
            redshift=Fixed(0.5),
        )
        assert "radio_delv_q0" in model.spec.free_params
        assert model.spec.radio_sfr_mode == "delvecchio2021"
        sed = model.predict_state({"radio_delv_q0": 2.9}).sed_intrinsic
        assert jnp.all(jnp.isfinite(sed))
