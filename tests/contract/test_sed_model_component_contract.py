# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: RUF012
"""Contract tests for SEDModelComponent base class.

Tests auto-discovery of priors, registry collision detection, Protocol
conformance, input/output contract building, and end-to-end apply flow.

`inputs` / `outputs` are part of the astronomer-facing class-level contract
on `SEDModelComponent` subclasses: the base parses them once in
`__init_subclass__` and deletes them. The mutable-default rule (RUF012)
doesn't apply at that level.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent
from tengri.parameters.priors import Fixed, Gaussian, Uniform
from tengri.protocols.component import (
    ForwardState,
    SEDComponentConfig,
)

pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def _cleanup_test_registry_entries():
    """Cleanup test components registered during test execution.

    Fixture saves the registry state before each test and restores it after,
    ensuring test-created components don't pollute subsequent tests or the
    domain-scoping census. This is critical for xdist isolation — test
    components created in one worker pollute the registry for all tests in
    that worker if not cleaned up.
    """
    saved_registry = dict(_REGISTRY)
    yield
    # Restore the registry to its pre-test state
    _REGISTRY.clear()
    _REGISTRY.update(saved_registry)


class TestPriorDiscovery:
    """Test auto-discovery of Distribution-typed class attributes."""

    def test_uniform_prior_discovery(self):
        """Subclass declaring Uniform produces ParamDeclaration with prefix."""

        class DustTemp(SEDModelComponent):
            name = "test_dust_temp"
            parameter_prefix = "test_dust_"
            T = Uniform(20.0, 80.0, description="Temperature", units="K")

        instance = DustTemp()
        decls = instance.declared_parameters()

        assert len(decls) == 1
        assert decls[0].name == "test_dust_T"
        assert decls[0].description == "Temperature"
        assert decls[0].units == "K"
        assert isinstance(decls[0].prior, Uniform)

    def test_multiple_priors(self):
        """Subclass with multiple Distribution attributes discovers all."""

        class ModifiedBlackbody(SEDModelComponent):
            name = "test_mbb"
            parameter_prefix = "test_mbb_"
            T = Uniform(20.0, 80.0, description="Temperature", units="K")
            beta = Uniform(1.0, 3.0, description="Emissivity index", units="")

        instance = ModifiedBlackbody()
        decls = instance.declared_parameters()

        assert len(decls) == 2
        names = {d.name for d in decls}
        assert names == {"test_mbb_T", "test_mbb_beta"}

    def test_fixed_prior_discovery(self):
        """Fixed priors are discovered like any Distribution."""

        class FixedComponent(SEDModelComponent):
            name = "test_fixed"
            parameter_prefix = "test_fixed_"
            fixed_param = Fixed(0.5, description="Fixed value", units="dex")

        instance = FixedComponent()
        decls = instance.declared_parameters()

        assert len(decls) == 1
        assert decls[0].name == "test_fixed_fixed_param"
        assert decls[0].description == "Fixed value"
        assert isinstance(decls[0].prior, Fixed)

    def test_gaussian_prior_with_units(self):
        """Gaussian priors carry description and units."""

        class GaussianComponent(SEDModelComponent):
            name = "test_gaussian"
            parameter_prefix = "test_gauss_"
            param = Gaussian(0.0, 1.0, description="A gaussian", units="mag")

        instance = GaussianComponent()
        decls = instance.declared_parameters()

        assert len(decls) == 1
        assert decls[0].description == "A gaussian"
        assert decls[0].units == "mag"

    def test_prior_without_description_units(self):
        """Priors without explicit description/units use defaults."""

        class MinimalComponent(SEDModelComponent):
            name = "test_minimal"
            parameter_prefix = "test_"
            param = Uniform(0.0, 1.0)  # No description or units

        instance = MinimalComponent()
        decls = instance.declared_parameters()

        assert len(decls) == 1
        assert decls[0].description == ""
        assert decls[0].units == ""


class TestRegistry:
    """Test component registration and collision detection."""

    def test_registry_entry_created(self):
        """Defining a subclass registers it by name."""

        class RegistryTest(SEDModelComponent):
            name = "registry_test_unique_name_001"
            parameter_prefix = "reg_test_"

        # Should be in _REGISTRY
        assert "registry_test_unique_name_001" in _REGISTRY
        assert _REGISTRY["registry_test_unique_name_001"] is RegistryTest

    def test_collision_raises_error(self):
        """Defining two classes with the same name raises ValueError."""

        class First(SEDModelComponent):
            name = "collision_test_unique_002"
            parameter_prefix = "first_"

        # Second definition should fail
        with pytest.raises(
            ValueError,
            match="Component name 'collision_test_unique_002' already registered",
        ):

            class Second(SEDModelComponent):
                name = "collision_test_unique_002"
                parameter_prefix = "second_"

    def test_collision_error_message_includes_modules(self):
        """Collision error message names both the old and new module."""

        class FirstModule(SEDModelComponent):
            name = "collision_module_test_003"
            parameter_prefix = "first_"

        with pytest.raises(ValueError) as exc_info:

            class SecondModule(SEDModelComponent):
                name = "collision_module_test_003"
                parameter_prefix = "second_"

        error_msg = str(exc_info.value)
        assert "collision_module_test_003" in error_msg
        assert "FirstModule" in error_msg or "first_" in error_msg


class TestProtocolConformance:
    """Test that SEDModelComponent satisfies SEDComponent Protocol."""

    def test_isinstance_check(self):
        """An instance satisfies the SEDComponent Protocol."""
        from tengri.protocols.component import SEDComponent

        class SimpleComponent(SEDModelComponent):
            name = "test_protocol"
            parameter_prefix = "proto_"

        instance = SimpleComponent()
        # Runtime check — the Protocol is checkable
        assert isinstance(instance, SEDComponent)

    def test_has_required_attributes(self):
        """Required attributes present and accessible."""

        class FullComponent(SEDModelComponent):
            name = "test_attrs"
            parameter_prefix = "attrs_"

        instance = FullComponent()
        assert instance.name == "test_attrs"
        assert instance.parameter_prefix == "attrs_"
        assert isinstance(instance.config, SEDComponentConfig)


class TestInputOutputContract:
    """Test inputs/outputs dict parsing and tuple building."""

    def test_empty_inputs_outputs(self):
        """Subclass without inputs/outputs has empty tuples."""

        class Minimal(SEDModelComponent):
            name = "test_empty_contract"
            parameter_prefix = "min_"

        instance = Minimal()
        assert instance.inputs() == ()
        assert instance.outputs() == ()

    def test_inputs_dict_parsing(self):
        """inputs dict converts to DerivedKey tuples."""

        class WithInputs(SEDModelComponent):
            name = "test_with_inputs"
            parameter_prefix = "inp_"
            inputs = {"L_absorbed": "erg/s", "M_star": "Msun"}

        instance = WithInputs()
        inputs = instance.inputs()

        assert len(inputs) == 2
        names = {k.name for k in inputs}
        units = {k.units for k in inputs}
        assert names == {"L_absorbed", "M_star"}
        assert units == {"erg/s", "Msun"}

    def test_outputs_dict_parsing(self):
        """outputs dict converts to DerivedKey tuples."""

        class WithOutputs(SEDModelComponent):
            name = "test_with_outputs"
            parameter_prefix = "out_"
            outputs = {"L_ir": "erg/s"}

        instance = WithOutputs()
        outputs = instance.outputs()

        assert len(outputs) == 1
        assert outputs[0].name == "L_ir"
        assert outputs[0].units == "erg/s"

    def test_dicts_deleted_after_parsing(self):
        """inputs/outputs dicts are deleted so methods are found."""

        class DictDelete(SEDModelComponent):
            name = "test_dict_delete"
            parameter_prefix = "dd_"
            inputs = {"x": ""}
            outputs = {"y": ""}

        # Class dict should not have 'inputs' / 'outputs' attributes
        # (they were deleted by __init_subclass__)
        assert "inputs" not in vars(DictDelete)
        assert "outputs" not in vars(DictDelete)

        # But instance methods should return tuples
        instance = DictDelete()
        assert isinstance(instance.inputs(), tuple)
        assert isinstance(instance.outputs(), tuple)


class TestPrecompute:
    """Test precompute flow and data storage."""

    def test_precompute_with_no_load(self):
        """precompute returns SEDComponentState when load returns None."""

        class NoLoad(SEDModelComponent):
            name = "test_no_load"
            parameter_prefix = "nl_"

            def load(self, wave):
                return None

        instance = NoLoad()
        state = instance.precompute(wave_grid=jnp.linspace(100, 10000, 100))

        # Should get back a state object
        from tengri.protocols.component import SEDComponentState

        assert isinstance(state, SEDComponentState)
        assert state.name == "test_no_load"

    def test_precompute_with_data_storage(self):
        """precompute calls load() and stores non-None result on self.data."""

        test_data = {"template_grid": jnp.array([[1.0, 2.0]])}

        class WithData(SEDModelComponent):
            name = "test_with_data"
            parameter_prefix = "wd_"

            def load(self, wave):
                return test_data

        instance = WithData()
        state = instance.precompute(wave_grid=jnp.linspace(100, 10000, 100))

        # Check that data was stashed
        assert hasattr(instance, "data")
        assert instance.data is test_data


class TestApplyEndToEnd:
    """End-to-end test of apply() orchestration."""

    def test_apply_basic_flow(self):
        """apply() slices params, looks up inputs, calls predict, updates state."""

        class TestDustIR(SEDModelComponent):
            name = "test_dust_ir"
            parameter_prefix = "dust_"
            T = Uniform(20.0, 80.0, description="Temperature", units="K")

            inputs = {"L_absorbed": "erg/s"}
            outputs = {"L_ir": "erg/s"}

            def predict(self, p, sed_in, wave, *, L_absorbed):
                # Simple: emit a scaled version of the input
                addition = jnp.ones_like(wave) * p["T"] * L_absorbed * 1e-30
                L_ir = jnp.sum(addition)
                return sed_in + addition, {"L_ir": L_ir}

        instance = TestDustIR()

        # Create input state
        wave = jnp.linspace(100, 10000, 50)
        sed_in = jnp.ones(50)
        state = ForwardState(
            wave=wave,
            sed_intrinsic=sed_in,
            derived={"L_absorbed": 1e45},
        )

        # Create parameters with prefix
        params = {"dust_T": 50.0}

        # Apply
        new_state = instance.apply(state, params)

        # Verify SED updated
        assert new_state.sed_intrinsic is not None
        assert not jnp.allclose(new_state.sed_intrinsic, sed_in)

        # Verify published key added to derived
        assert "L_ir" in new_state.derived
        assert new_state.derived["L_ir"] is not None

    def test_apply_initializes_zero_sed(self):
        """apply() initializes sed_intrinsic to zeros if None."""

        class ZeroInit(SEDModelComponent):
            name = "test_zero_init"
            parameter_prefix = "zi_"

            def predict(self, p, sed_in, wave):
                # Emit ones
                return jnp.ones_like(wave), {}

        instance = ZeroInit()

        wave = jnp.linspace(100, 10000, 50)
        # Note: sed_intrinsic is None
        state = ForwardState(wave=wave, sed_intrinsic=None)

        new_state = instance.apply(state, {})

        # Should be initialized to zeros + ones
        assert new_state.sed_intrinsic is not None
        assert jnp.allclose(new_state.sed_intrinsic, jnp.ones(50))

    def test_apply_missing_required_input_fails(self):
        """apply() raises KeyError if required input is missing."""

        class RequiresInput(SEDModelComponent):
            name = "test_requires_input"
            parameter_prefix = "ri_"
            inputs = {"required_key": "erg/s"}

            def predict(self, p, sed_in, wave, *, required_key):
                return sed_in, {}

        instance = RequiresInput()

        wave = jnp.linspace(100, 10000, 50)
        # derived is empty — required_key missing
        state = ForwardState(wave=wave, sed_intrinsic=None, derived={})

        with pytest.raises(
            KeyError,
            match="required_key",
        ):
            instance.apply(state, {})

    def test_apply_no_mutation(self):
        """apply() returns new state; input state is unchanged."""

        class NoMutate(SEDModelComponent):
            name = "test_no_mutate"
            parameter_prefix = "nm_"

            def predict(self, p, sed_in, wave):
                return sed_in * 2.0, {}

        instance = NoMutate()

        wave = jnp.linspace(100, 10000, 50)
        sed_in = jnp.ones(50)
        state = ForwardState(wave=wave, sed_intrinsic=sed_in)

        new_state = instance.apply(state, {})

        # Original should be unchanged
        assert jnp.allclose(state.sed_intrinsic, sed_in)

        # New state should be different
        assert not jnp.allclose(new_state.sed_intrinsic, sed_in)
        assert jnp.allclose(new_state.sed_intrinsic, sed_in * 2.0)

    def test_apply_parameter_slicing(self):
        """apply() strips prefix from param dict before passing to predict."""

        class ParamSlice(SEDModelComponent):
            name = "test_param_slice"
            parameter_prefix = "ps_"
            X = Uniform(0.0, 10.0, description="X", units="")

            def predict(self, p, sed_in, wave):
                # p should have key "X" (prefix stripped)
                # Just verify the parameter is sliced by returning it in state
                return sed_in + p["X"], {}

        instance = ParamSlice()

        wave = jnp.linspace(100, 10000, 50)
        state = ForwardState(wave=wave, sed_intrinsic=jnp.ones(50))

        # Full param dict with prefix
        params = {"ps_X": 5.0}

        new_state = instance.apply(state, params)

        # Check SED was updated with the parameter value
        assert jnp.allclose(new_state.sed_intrinsic, jnp.ones(50) + 5.0)


class TestLoadDefault:
    """Test the default load() method."""

    def test_default_load_returns_none(self):
        """Default load() returns None."""

        class DefaultLoad(SEDModelComponent):
            name = "test_default_load"
            parameter_prefix = "dl_"

        instance = DefaultLoad()
        result = instance.load(None)
        assert result is None

        result = instance.load(jnp.linspace(100, 10000, 50))
        assert result is None


class TestPredictNotImplemented:
    """Test that default predict() raises NotImplementedError."""

    def test_predict_not_implemented(self):
        """Subclass must implement predict() or get NotImplementedError."""

        class NoPredict(SEDModelComponent):
            name = "test_no_predict"
            parameter_prefix = "np_"

        instance = NoPredict()

        wave = jnp.linspace(100, 10000, 50)
        state = ForwardState(wave=wave, sed_intrinsic=None)

        with pytest.raises(NotImplementedError, match="predict\\(\\) must be implemented"):
            instance.apply(state, {})
