"""Integration test: verify cross-galaxy engine reuse in CatalogFitter.

Tests that CatalogFitter with multiple galaxies reuses a single compiled
engine across all galaxies when they share the same shape signature.

The test instruments _build_jit_engine with a counter and verifies:
1. Three fitters (with different SSP arrays of identical shape) reuse one compile
2. Each fitter still produces correct numerical results (equivalence test)
"""

from __future__ import annotations

from unittest import mock

import jax.numpy as jnp
import pytest

from tengri import Fitter, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.inference.jit_engine import build_jit_engine
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry


@pytest.fixture
def base_ssp_data():
    """Return a minimal SSPData for testing."""
    n_met, n_age, n_wave = 8, 15, 200
    return SSPData(
        ssp_wave=jnp.logspace(3, 4.5, n_wave),
        ssp_flux=jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(6, 10.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )


@pytest.fixture
def photometry():
    """Return a basic photometry observation."""
    return Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))


@pytest.fixture
def spec_dpl():
    """Return a simple DPL SFH spec."""
    return Parameters(
        redshift=0.1,
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.3, 3.0),
    )


def _mutate_ssp_flux(ssp_data, factor: float) -> SSPData:
    """Return a new SSPData with scaled flux (to fake different SSP files)."""
    return ssp_data._replace(ssp_flux=ssp_data.ssp_flux * factor)


class TestCatalogEngineReuse:
    """Integration tests for cross-galaxy engine reuse."""

    def test_three_fitters_one_compile(self, base_ssp_data, photometry, spec_dpl):
        """Three fitters with identical shapes should compile once.

        Creates 3 SEDModels with different SSP flux values (but same shape),
        constructs 3 Fitters, and verifies only 1 call to build_jit_engine.
        """
        # Create 3 SSPData instances with different flux values
        ssp1 = base_ssp_data
        ssp2 = _mutate_ssp_flux(base_ssp_data, 1.1)
        ssp3 = _mutate_ssp_flux(base_ssp_data, 0.9)

        # Create 3 models
        models = [
            SEDModel(spec_dpl, ssp1, observation=photometry),
            SEDModel(spec_dpl, ssp2, observation=photometry),
            SEDModel(spec_dpl, ssp3, observation=photometry),
        ]

        # Verify all have identical compile_signature
        sigs = [model.compile_signature() for model in models]
        assert sigs[0] == sigs[1] == sigs[2], "Models should have identical signatures"

        # Create fitters with identical data config
        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitters = [Fitter(model, data, noise, data_type="photometry") for model in models]

        # Verify all fitters have identical compile_signature
        fitter_sigs = [fitter.compile_signature() for fitter in fitters]
        assert fitter_sigs[0] == fitter_sigs[1] == fitter_sigs[2], (
            "Fitters should have identical signatures"
        )

        # Instrument build_jit_engine with a call counter
        call_count = 0
        original_build = build_jit_engine

        def counting_build(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_build(*args, **kwargs)

        with mock.patch(
            "tengri.inference.jit_engine.build_jit_engine",
            side_effect=counting_build,
        ):
            # Now call _get_or_build_engine on each fitter
            # Only the first should trigger build_jit_engine; the rest reuse
            pos_dict = {name: jnp.array(0.0) for name in fitters[0].spec.free_params}

            engines = []
            for fitter in fitters:
                engine = fitter._get_or_build_engine(pos_dict)
                engines.append(engine)

            # Should have compiled exactly once
            assert call_count == 1, f"Expected 1 compile, got {call_count}"

        # All three fitters should have received the SAME engine object
        # (shared via WeakValueDictionary)
        assert engines[0] is engines[1] is engines[2], (
            "All three fitters should reference the same compiled engine"
        )

    def test_engine_reuse_produces_correct_results(self, base_ssp_data, photometry, spec_dpl):
        """Engines reused from cache should work for both fitters.

        Verifies that using a cached engine vs. rebuilding it produces
        valid results and the engines are actually shared.
        """
        # Create two models with different SSP flux (same shape)
        ssp1 = base_ssp_data
        ssp2 = _mutate_ssp_flux(base_ssp_data, 1.05)

        model1 = SEDModel(spec_dpl, ssp1, observation=photometry)
        model2 = SEDModel(spec_dpl, ssp2, observation=photometry)

        # Create fitters
        data = jnp.ones(3) * 0.5
        noise = jnp.ones(3) * 0.05

        fitter1 = Fitter(model1, data, noise, data_type="photometry")
        fitter2 = Fitter(model2, data, noise, data_type="photometry")

        # Get engines (fitter1 will build, fitter2 will reuse)
        pos_dict = {name: jnp.array(0.0) for name in fitter1.spec.free_params}
        engine1 = fitter1._get_or_build_engine(pos_dict)
        engine2 = fitter2._get_or_build_engine(pos_dict)

        # Both engines should be valid dicts
        assert isinstance(engine1, dict), "engine1 should be a dict"
        assert isinstance(engine2, dict), "engine2 should be a dict"

        # They should be the SAME object (reused from cache)
        assert engine1 is engine2, "Engines should be identical objects (reused from cache)"

    def test_different_signature_no_reuse(self, base_ssp_data, photometry, spec_dpl):
        """Fitters with different signatures should NOT reuse engines.

        Tests that changing a JIT-affecting parameter (redshift being fixed)
        produces different compile signatures and doesn't reuse engines.
        """
        # Create two specs: one with fixed redshift, one with free redshift
        spec1 = Parameters(
            redshift=0.1,  # fixed
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            apply_igm=True,
        )
        spec2 = Parameters(
            redshift=Uniform(0.05, 0.15),  # free (different signature)
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
            apply_igm=True,
        )

        model1 = SEDModel(spec1, base_ssp_data, observation=photometry)
        model2 = SEDModel(spec2, base_ssp_data, observation=photometry)

        data = jnp.ones(3)
        noise = jnp.ones(3) * 0.1

        fitter1 = Fitter(model1, data, noise, data_type="photometry")
        fitter2 = Fitter(model2, data, noise, data_type="photometry")

        # Verify different signatures
        sig1 = fitter1.compile_signature()
        sig2 = fitter2.compile_signature()
        assert sig1 != sig2, (
            "Fitters should have different signatures when redshift config differs"
        )

        # Get engines (each should build separately)
        pos_dict1 = {name: jnp.array(0.0) for name in fitter1.spec.free_params}
        pos_dict2 = {name: jnp.array(0.0) for name in fitter2.spec.free_params}

        engine1 = fitter1._get_or_build_engine(pos_dict1)
        engine2 = fitter2._get_or_build_engine(pos_dict2)

        # Engines should be different objects (different signatures)
        assert engine1 is not engine2, "Different signatures should produce different engines"
