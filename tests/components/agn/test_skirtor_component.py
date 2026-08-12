# SPDX-License-Identifier: BSD-3-Clause
"""Tests for SKIRTOR on SEDModelComponent.

Validates:
- isinstance(SKIRTORTorus, SEDModelComponent)
- Registry entry under name "skirtor"
- Free parameter discovery and units preservation
- Parity vs existing skirtor_analytic for same parameters
- Cross-component output (L_agn_torus) publishing
- Missing template data: zero emission when no grid was requested, a loud
  FileNotFoundError when a named grid cannot be loaded
"""

from __future__ import annotations

import re
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig
from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent
from tests._bounds import assert_non_negative

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SKIRTOR_CANDIDATES = [
    _DATA_DIR / "skirtor_templates_v3.h5",
    _DATA_DIR / "skirtor_templates_v2.h5",
]
_SKIRTOR_PATH = next((p for p in _SKIRTOR_CANDIDATES if p.is_file()), None)
_has_skirtor_data = _SKIRTOR_PATH is not None


class TestSKIRTORComponentBasics:
    """Test component class structure and discovery."""

    def test_skirtor_is_sed_model_component(self):
        """SKIRTORTorus subclasses SEDModelComponent."""
        assert issubclass(SKIRTORTorus, SEDModelComponent)

    def test_skirtor_in_registry(self):
        """Component name is registered in _REGISTRY."""
        assert "skirtor" in _REGISTRY
        assert _REGISTRY["skirtor"] is SKIRTORTorus

    def test_skirtor_attributes(self):
        """Required class attributes are set."""
        comp = SKIRTORTorus()
        assert comp.name == "skirtor"
        assert comp.parameter_prefix == "agn_"

    def test_skirtor_outputs(self):
        """outputs() method returns the six CIGALE-faithful cross-component keys.

        Post-#329 (CIGALE-faithful BLR/NLR/X-ray/SKIRTOR), SKIRTORTorus publishes
        a richer set: separate disc/torus/polar-dust bolometric luminosities, the
        AGN-only 2500 A monochromatic luminosity (used for α_OX), and the mid-IR
        diagnostics at 6 and 12 micron.
        """
        comp = SKIRTORTorus()
        outs = comp.outputs()
        names = {o.name for o in outs}
        expected = {
            "L_agn_disc",
            "L_agn_torus",
            "L_agn_polar_dust",
            "L_2500_30deg",
            "L_6um",
            "L_12um",
        }
        assert names == expected
        units = {o.name: o.units for o in outs}
        assert units["L_agn_torus"] == "erg/s"
        assert units["L_6um"] == "erg/s/Hz"


class TestSKIRTORParameterDiscovery:
    """Test auto-discovery of free parameters."""

    def test_declared_parameters_count(self):
        """SKIRTORTorus declares eleven free parameters.

        Seven core + polar-dust trio (10) plus the CIGALE disc-shape
        modulation ``agn_delta`` (disk_type is a static config choice, not a
        free parameter).
        """
        comp = SKIRTORTorus()
        decls = comp.declared_parameters()
        assert len(decls) == 11

    def test_declared_parameter_names(self):
        """Parameter names have agn_ prefix as per naming contract.

        Post-#329 ``agn_torus_frac`` was renamed to ``agn_band_frac`` to align
        with CIGALE's nomenclature. The polar-dust trio (``agn_polar_ebv``,
        ``agn_polar_temperature``, ``agn_polar_beta``) was promoted to free
        params when the bi-conical re-emission pipeline was wired through
        (Yang+2020 §2.2.2).
        """
        comp = SKIRTORTorus()
        decls = comp.declared_parameters()
        names = {d.name for d in decls}
        expected = {
            "agn_log_lbol",
            "agn_tau_skirtor",
            "agn_p_skirtor",
            "agn_q_skirtor",
            "agn_oa_skirtor",
            "agn_cos_inc",
            "agn_band_frac",
            "agn_polar_ebv",
            "agn_polar_temperature",
            "agn_polar_beta",
            # CIGALE disc-shape modulation (skirtor2016 ``delta``).
            "agn_delta",
        }
        assert names == expected

    def test_parameter_units_preserved(self):
        """Parameter units are preserved from Uniform priors."""
        comp = SKIRTORTorus()
        decls = {d.name: d for d in comp.declared_parameters()}

        assert decls["agn_log_lbol"].units == "dex (L_sun)"
        assert decls["agn_oa_skirtor"].units == "deg"
        # Others dimensionless
        assert decls["agn_tau_skirtor"].units == "dimensionless"

    def test_parameter_descriptions_nonempty(self):
        """All parameters have descriptions."""
        comp = SKIRTORTorus()
        decls = comp.declared_parameters()
        for d in decls:
            assert isinstance(d.description, str)
            assert len(d.description) > 0


@pytest.mark.skipif(not _has_skirtor_data, reason="SKIRTOR template data not available")
class TestSKIRTORPredictParity:
    """Test numerical equivalence to original skirtor_analytic."""

    @pytest.fixture
    def wave(self):
        """IR wavelength grid (1 um to 1000 um)."""
        return jnp.logspace(4, 7, 200)

    @pytest.fixture
    def skirtor_component(self):
        """SKIRTORTorus instance with templates loaded."""
        config = SKIRTORTorusConfig(grid_path=str(_SKIRTOR_PATH))
        return SKIRTORTorus(config=config)

    @pytest.fixture
    def precomputed_state(self, skirtor_component, wave):
        """Pre-computed component state."""
        comp_state = skirtor_component.precompute(wave_grid=wave)
        # Manually attach the loaded data
        skirtor_component.data = comp_state.skirtor_fn
        return comp_state

    def test_predict_returns_tuple(self, skirtor_component, wave):
        """predict() returns (sed_out, published) tuple."""
        sed_in = jnp.zeros_like(wave)
        params = {
            "agn_log_lbol": jnp.array(10.0),
            "agn_tau_skirtor": jnp.array(7.0),
            "agn_p_skirtor": jnp.array(1.0),
            "agn_q_skirtor": jnp.array(1.0),
            "agn_oa_skirtor": jnp.array(40.0),
            "agn_cos_inc": jnp.array(0.5),
            "agn_torus_frac": jnp.array(0.5),
        }

        # Load templates first
        skirtor_component.data = skirtor_component.load(wave)
        if skirtor_component.data is None:
            pytest.skip("Could not load SKIRTOR templates")

        sed_out, published = skirtor_component.predict(params, sed_in, wave)

        chex.assert_equal_shape([sed_out, wave])
        assert "L_agn_torus" in published
        assert published["L_agn_torus"].shape == ()

    def test_output_shape(self, skirtor_component, wave):
        """Output SED has same shape as input."""
        sed_in = jnp.zeros_like(wave)
        params = {
            "agn_log_lbol": jnp.array(10.0),
            "agn_tau_skirtor": jnp.array(7.0),
            "agn_p_skirtor": jnp.array(1.0),
            "agn_q_skirtor": jnp.array(1.0),
            "agn_oa_skirtor": jnp.array(40.0),
            "agn_cos_inc": jnp.array(0.5),
            "agn_torus_frac": jnp.array(0.5),
        }

        skirtor_component.data = skirtor_component.load(wave)
        if skirtor_component.data is None:
            pytest.skip("Could not load SKIRTOR templates")

        sed_out, _ = skirtor_component.predict(params, sed_in, wave)
        chex.assert_equal_shape([sed_out, sed_in])

    def test_output_positive(self, skirtor_component, wave):
        """Output SED is non-negative."""
        sed_in = jnp.zeros_like(wave)
        params = {
            "agn_log_lbol": jnp.array(10.0),
            "agn_tau_skirtor": jnp.array(7.0),
            "agn_p_skirtor": jnp.array(1.0),
            "agn_q_skirtor": jnp.array(1.0),
            "agn_oa_skirtor": jnp.array(40.0),
            "agn_cos_inc": jnp.array(0.5),
            "agn_torus_frac": jnp.array(0.5),
        }

        skirtor_component.data = skirtor_component.load(wave)
        if skirtor_component.data is None:
            pytest.skip("Could not load SKIRTOR templates")

        sed_out, _ = skirtor_component.predict(params, sed_in, wave)
        assert_non_negative(sed_out, name="sed_out", msg="SKIRTOR SED should be non-negative")

    def test_parity_vs_skirtor_analytic(self, skirtor_component, wave):
        """Output matches create_skirtor_from_grid result."""
        from tengri.components.agn.skirtor import create_skirtor_from_grid

        # Manually call the old path
        old_fn = create_skirtor_from_grid(str(_SKIRTOR_PATH))
        old_sed = old_fn(
            wavelength=wave,
            agn_log_lbol=10.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_cos_inc=0.5,
            agn_torus_frac=0.5,
        )

        # New component path
        sed_in = jnp.zeros_like(wave)
        params = {
            "agn_log_lbol": jnp.array(10.0),
            "agn_tau_skirtor": jnp.array(7.0),
            "agn_p_skirtor": jnp.array(1.0),
            "agn_q_skirtor": jnp.array(1.0),
            "agn_oa_skirtor": jnp.array(40.0),
            "agn_cos_inc": jnp.array(0.5),
            "agn_torus_frac": jnp.array(0.5),
        }

        skirtor_component.data = skirtor_component.load(wave)
        if skirtor_component.data is None:
            pytest.skip("Could not load SKIRTOR templates")

        sed_out, _ = skirtor_component.predict(params, sed_in, wave)
        new_sed = sed_out  # Since sed_in was zeros

        # Check parity to rtol=1e-12 (machine precision for C²-continuous interpolation)
        np.testing.assert_allclose(new_sed, old_sed, rtol=1e-12)


class TestSKIRTORMissingData:
    """Test graceful degradation when templates are unavailable."""

    def test_no_templates_no_crash(self):
        """predict() returns zero emission when data is None."""
        comp = SKIRTORTorus(config=SKIRTORTorusConfig(grid_path=None))
        # Don't load anything
        assert not hasattr(comp, "data") or comp.data is None

        sed_in = jnp.zeros(100)
        params = {
            "agn_log_lbol": jnp.array(10.0),
            "agn_tau_skirtor": jnp.array(7.0),
            "agn_p_skirtor": jnp.array(1.0),
            "agn_q_skirtor": jnp.array(1.0),
            "agn_oa_skirtor": jnp.array(40.0),
            "agn_cos_inc": jnp.array(0.5),
            "agn_torus_frac": jnp.array(0.5),
        }
        wave = jnp.logspace(4, 7, 100)

        sed_out, published = comp.predict(params, sed_in, wave)

        assert jnp.allclose(sed_out, sed_in), "Should return input SED unchanged"
        assert jnp.allclose(published["L_agn_torus"], 0.0), "Should publish zero luminosity"

    def test_load_nonexistent_path_raises(self):
        """load() raises for a grid_path that is set but cannot be loaded.

        Degrading to ``None`` here would be graceful in the wrong direction: the
        user named a grid file, so a typo would silently become a fit in which
        the torus contributes zero and every ``agn_*_skirtor`` parameter is a
        no-op. Only an *unset* grid_path means "no torus wanted" -- see
        :func:`test_no_templates_no_crash` above, which still holds.
        """
        comp = SKIRTORTorus(config=SKIRTORTorusConfig(grid_path="/nonexistent/path.h5"))
        with pytest.raises(FileNotFoundError, match=re.escape("/nonexistent/path.h5")):
            comp.load(wave=jnp.logspace(4, 7, 100))
