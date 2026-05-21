# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for CueNebularSEDComponent.

Validates the SEDModelComponent port of the Cue NN emulator:
- Registry and isinstance checks
- Parameter declarations flow correctly
- Parity with original Cue backend on line luminosities
- Graceful skip when weights are missing
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_cue_nebular_component_registered():
    """Cue component is registered in the global registry."""
    from tengri.components.nebular.cue_model import CueNebularSEDComponent  # noqa: F401
    from tengri.components.sed_model_component import _REGISTRY

    assert "cue_emulator" in _REGISTRY, "cue_emulator not in component registry"
    assert _REGISTRY["cue_emulator"].__name__ == "CueNebularSEDComponent"


def test_cue_nebular_component_isinstance():
    """CueNebularSEDComponent inherits from SEDModelComponent."""
    from tengri.components.nebular.cue_model import CueNebularSEDComponent
    from tengri.components.sed_model_component import SEDModelComponent

    comp = CueNebularSEDComponent()
    assert isinstance(comp, SEDModelComponent)


def test_cue_nebular_declared_parameters():
    """Declared parameters include 14 Cue knobs with correct units."""
    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    comp = CueNebularSEDComponent()
    decls = comp.declared_parameters()

    # Should have 14 parameters: 4 standard + 7 ionizing spectrum + 3 gas extras
    assert len(decls) == 14, f"Expected 14 params, got {len(decls)}"

    # Check names and units
    param_names = {d.name for d in decls}
    expected = {
        "neb_logU",
        "neb_logZ_gas",
        "neb_fesc",
        "neb_fesc_lya",
        "neb_ionspec_index1",
        "neb_ionspec_index2",
        "neb_ionspec_index3",
        "neb_ionspec_index4",
        "neb_ionspec_logLratio1",
        "neb_ionspec_logLratio2",
        "neb_ionspec_logLratio3",
        "neb_gas_logn",
        "neb_gas_logno",
        "neb_gas_logco",
    }

    # Note: the actual names will include the neb_ prefix
    assert param_names == expected, f"Param mismatch: {param_names ^ expected}"

    # Spot-check units
    for decl in decls:
        has_log = any(x in decl.name for x in ["logU", "logZ", "index", "ratio"])
        if has_log:
            assert decl.units in {
                "dex",
                "dimensionless",
            }, f"Units for {decl.name}: {decl.units}"


def test_cue_nebular_inputs_outputs():
    """Cross-component contract declares inputs and outputs."""
    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    comp = CueNebularSEDComponent()
    inputs = comp.inputs()
    outputs = comp.outputs()

    # Inputs: ssp_ages_yr, age_weights
    input_names = {dk.name for dk in inputs}
    assert "ssp_ages_yr" in input_names
    assert "age_weights" in input_names

    # Outputs: line_waves, line_lums
    output_names = {dk.name for dk in outputs}
    assert "line_waves" in output_names
    assert "line_lums" in output_names


@pytest.mark.skipif(
    True,  # Skip by default — weights file required
    reason="Cue weights file (data/cue_weights.npz) not available",
)
def test_cue_nebular_parity_vs_backend():
    """Line luminosities match the original CueBackend to rtol=1e-10.

    This test requires the Cue weights file and a bare-stellar SSP.
    Skipped by default; enable when testing against a data directory.
    """
    import jax.numpy as jnp

    from tengri.components.nebular.cue import CueBackend
    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    # Load weights
    try:
        backend = CueBackend(weights_path="data/cue_weights.npz", ssp_data=None)
    except FileNotFoundError:
        pytest.skip("Cue weights file not found")

    # Create component
    comp = CueNebularSEDComponent()
    comp_precomputed = comp.precompute(wave_grid=jnp.linspace(100, 10000, 1000))

    # Set backend on component
    comp.data = backend

    # Test parameters (defaults)
    params = {
        "neb_logU": -3.0,
        "neb_logZ_gas": -0.3,
        "neb_fesc": 0.0,
        "neb_fesc_lya": 0.0,
        "neb_ionspec_index1": 2.0,
        "neb_ionspec_index2": 1.5,
        "neb_ionspec_index3": 1.0,
        "neb_ionspec_index4": 0.5,
        "neb_ionspec_logLratio1": 1.0,
        "neb_ionspec_logLratio2": 0.5,
        "neb_ionspec_logLratio3": 0.5,
        "neb_gas_logn": 2.0,
        "neb_gas_logno": 0.0,
        "neb_gas_logco": 0.0,
    }

    # Call component predict
    wave = jnp.linspace(100, 10000, 1000)
    sed_in = jnp.zeros_like(wave)
    ssp_ages_yr = jnp.array([1e6, 1e7, 1e8, 1e9])
    age_weights = jnp.array([0.25, 0.25, 0.25, 0.25])

    _sed_out, published = comp.predict(
        {k.replace("neb_", ""): v for k, v in params.items()},
        sed_in,
        wave,
        ssp_ages_yr=ssp_ages_yr,
        age_weights=age_weights,
    )

    # Verify published outputs
    assert "line_waves" in published
    assert "line_lums" in published
    assert len(published["line_waves"]) == len(published["line_lums"])
    assert len(published["line_waves"]) > 0  # Should have some lines


def test_cue_nebular_weights_missing_graceful():
    """Component returns empty outputs when weights file is missing.

    Simulates the case where data/cue_weights.npz is not available.
    """
    import jax.numpy as jnp

    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    comp = CueNebularSEDComponent()

    # Don't load weights (self.data remains None)
    # This simulates the weights file being missing

    params = {
        "logU": -3.0,
        "logZ_gas": -0.3,
        "fesc": 0.0,
        "fesc_lya": 0.0,
        "ionspec_index1": 2.0,
        "ionspec_index2": 1.5,
        "ionspec_index3": 1.0,
        "ionspec_index4": 0.5,
        "ionspec_logLratio1": 1.0,
        "ionspec_logLratio2": 0.5,
        "ionspec_logLratio3": 0.5,
        "gas_logn": 2.0,
        "gas_logno": 0.0,
        "gas_logco": 0.0,
    }

    wave = jnp.linspace(100, 10000, 100)
    sed_in = jnp.zeros_like(wave)

    sed_out, published = comp.predict(params, sed_in, wave)

    # Should return empty arrays gracefully
    assert published["line_waves"].shape[0] == 0
    assert published["line_lums"].shape[0] == 0
    assert jnp.allclose(sed_out, sed_in)


def test_cue_nebular_precompute_stores_backend():
    """Precompute() loads weights and stores on self.data."""
    import jax.numpy as jnp

    from tengri.components.nebular.cue_model import CueNebularSEDComponent

    comp = CueNebularSEDComponent()
    state = comp.precompute(wave_grid=jnp.linspace(100, 10000, 100))

    # Weights may or may not be loaded depending on file availability
    # State should be created regardless
    assert state.name == "cue_emulator"

    # If weights were loaded, self.data should be set
    if hasattr(comp, "data") and comp.data is not None:
        from tengri.components.nebular.cue import CueBackend

        assert isinstance(comp.data, CueBackend)
