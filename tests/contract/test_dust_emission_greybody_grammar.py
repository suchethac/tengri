# SPDX-License-Identifier: BSD-3-Clause
"""Tests for greybody dust emission model registration and grammar.

Covers component registration, parameter declarations, and public API.
"""

import pytest

import tengri

pytestmark = pytest.mark.contract


class TestGreybodyRegistration:
    """Greybody component should be registered alongside other analytic models."""

    def test_greybody_in_list_dust_emission_models(self):
        """greybody should appear in the public dust emission models list."""
        models = tengri.list_dust_emission_models()
        model_names = [m["name"] for m in models]
        assert "greybody" in model_names

    def test_greybody_describe(self):
        """tengri.describe('greybody') should return a use line."""
        doc = tengri.describe("greybody")
        assert doc is not None
        # describe() returns a _DescribeRecord, which has a string representation
        assert "greybody" in str(doc).lower()

    def test_greybody_grammar_fixed_defaults(self):
        """SEDModel.build with greybody and all params fixed should work."""
        ssp = tengri.load_ssp()
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_emission={
                "type": "greybody",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            redshift=tengri.Fixed(0.0),
        )
        assert model is not None

    def test_greybody_grammar_custom_params(self):
        """SEDModel.build with greybody and custom params should work."""
        ssp = tengri.load_ssp()
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_emission={
                "type": "greybody",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "T": tengri.Fixed(30.0),
                "beta_ir": tengri.Fixed(2.0),
                "lambda_0_um": tengri.Fixed(100.0),
            },
            redshift=tengri.Fixed(0.05),
        )
        assert model is not None

    def test_greybody_sed_dust_ir_published(self):
        """Greybody should publish sed_dust_ir in predict_state output."""
        ssp = tengri.load_ssp()
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_emission={
                "type": "greybody",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "T": tengri.Fixed(30.0),
                "beta_ir": tengri.Fixed(2.0),
            },
            redshift=tengri.Fixed(0.0),
        )
        state = model.predict_state({})
        assert "sed_dust_ir" in state.derived


class TestLambda0UmParameter:
    """dust_lambda_0_um parameter should exist and work with greybody and casey2012."""

    def test_lambda_0_um_in_greybody(self):
        """greybody should accept lambda_0_um parameter."""
        ssp = tengri.load_ssp()
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_emission={
                "type": "greybody",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "lambda_0_um": tengri.Fixed(50.0),
            },
            redshift=tengri.Fixed(0.0),
        )
        assert model is not None

    @pytest.mark.parametrize("dust_type", ["greybody", "casey2012"])
    def test_lambda_0_um_reaches_the_closure(self, dust_type):
        """A different pivot must change the emitted spectrum through the public build.

        Accepting the key while never passing it to the closure is the silent
        failure this test guards: the casey2012 component once declared no
        pivot at all, so ``lambda_0_um`` was accepted (the group wildcard admits
        every dust parameter) and ignored.
        """
        import numpy as np

        ssp = tengri.load_ssp()
        seds = []
        for lambda_0_um in (50.0, 400.0):
            model = tengri.SEDModel.build(
                ssp_data=ssp,
                sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
                dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
                dust_emission={
                    "type": dust_type,
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                    "T": tengri.Fixed(35.0),
                    "beta_ir": tengri.Fixed(2.0),
                    "lambda_0_um": tengri.Fixed(lambda_0_um),
                },
                redshift=tengri.Fixed(0.0),
            )
            state = model.predict_state({})
            seds.append(np.asarray(state.derived["sed_dust_ir"]) / float(state.derived["L_ir"]))
        # Measured through the closure: the 50 vs 400 micron pivots differ by
        # 93% of the peak for casey2012 and 38% for greybody at 25-35 K.
        rel = np.max(np.abs(seds[0] - seds[1])) / np.max(seds[0])
        assert rel > 0.1, (
            f"{dust_type}: lambda_0_um is inert through the public build (rel={rel:.2e})"
        )


class TestGreybodyWavelengthExtension:
    """Greybody needs FIR/submm support on the master grid (#1005)."""

    def test_greybody_master_grid_extends_to_far_ir(self):
        """Greybody build should extend master grid to at least 1 cm (1e8 A)."""
        ssp = tengri.load_ssp()
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_emission={"type": "greybody", "all_params": tengri.Fixed(tengri.DEFAULT)},
            redshift=tengri.Fixed(0.0),
        )
        state = model.predict_state({})
        # Master grid should extend to at least 1 cm (1e8 Angstrom)
        assert state.wave[-1] >= 1e8, (
            f"Master grid max wavelength {state.wave[-1]} A is less than 1 cm "
            "(1e8 A); greybody emission will truncate"
        )

    def test_greybody_150um_peak_in_grid(self):
        """For 20 K greybody, FIR peak (~150 um) should be inside master grid."""
        ssp = tengri.load_ssp()
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_emission={
                "type": "greybody",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "T": tengri.Fixed(20.0),
            },
            redshift=tengri.Fixed(0.0),
        )
        state = model.predict_state({})
        ir_emission = state.derived["sed_dust_ir"]

        # Find peak wavelength in IR (should be around 150 um = 1.5e6 A for 20 K)
        peak_idx = ir_emission.argmax()
        peak_wavelength_aa = state.wave[peak_idx]
        peak_wavelength_um = float(peak_wavelength_aa) / 1e4

        # Peak should be in the FIR range (100-200 um for 20 K blackbody)
        assert 80 < peak_wavelength_um < 250, (
            f"Peak at {peak_wavelength_um} um is outside expected FIR range [80-250] for 20 K dust"
        )


class TestBetaIrRelaxation:
    """dust_beta_ir should now allow >= 0 (was > 0)."""

    def test_beta_ir_zero_in_modified_blackbody(self):
        """beta_ir = 0 (pure blackbody) should be accepted in modified_blackbody."""
        ssp = tengri.load_ssp()
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_emission={
                "type": "modified_blackbody",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "beta_ir": tengri.Fixed(0.0),
            },
            redshift=tengri.Fixed(0.0),
        )
        assert model is not None

    def test_beta_ir_zero_in_greybody(self):
        """beta_ir = 0 should be accepted in greybody."""
        ssp = tengri.load_ssp()
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_emission={
                "type": "greybody",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "beta_ir": tengri.Fixed(0.0),
            },
            redshift=tengri.Fixed(0.0),
        )
        assert model is not None

    def test_beta_ir_zero_in_casey2012(self):
        """beta_ir = 0 should be accepted in casey2012."""
        ssp = tengri.load_ssp()
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "delayed", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={"law": "calzetti", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_emission={
                "type": "casey2012",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "beta_ir": tengri.Fixed(0.0),
            },
            redshift=tengri.Fixed(0.0),
        )
        assert model is not None
