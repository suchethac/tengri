# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the property registry and predict_properties interface.

These tests establish the correctness contract for Phase 1A:
- Bit-equality against state_to_sfh_quantities (same code path)
- Documented tolerance vs legacy predict_sfh_quantities (different path)
- JIT/vmap/grad compatibility
- Collision detection
- PropertyCatalog dict-like interface
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, FREE, Fixed, SEDModel
from tengri.forward.component_factory import state_to_sfh_quantities
from tengri.forward.properties import assemble_available_properties

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def base_model(synthetic_ssp_wide):
    """Base SFH model for property tests."""
    ssp = synthetic_ssp_wide
    spec = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "dpl", "*": FREE},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED, "tau_bc": 0.5},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )
    return spec


class TestBitEquality:
    """Test 1: Bit-equality against state_to_sfh_quantities (same path)."""

    def test_stellar_mass(self, base_model):
        """stellar_mass == 10**log_mstar_formed from state."""
        params = base_model.spec.sample(jax.random.PRNGKey(42))

        # Get from orchestrator path
        state = base_model.predict_state(params)
        sfh_qty = state_to_sfh_quantities(state)
        stellar_mass_ref = sfh_qty.stellar_mass

        # Get from predict_properties
        props = base_model.predict_properties(params, names=("stellar_mass",))
        stellar_mass = props["stellar_mass"]

        # Must match exactly (rtol=0, atol=0)
        chex.assert_trees_all_close(stellar_mass, stellar_mass_ref, rtol=0, atol=0)

    def test_stellar_mass_surviving(self, base_model):
        """stellar_mass_surviving == 10**log_mstar from state."""
        params = base_model.spec.sample(jax.random.PRNGKey(43))
        state = base_model.predict_state(params)
        sfh_qty = state_to_sfh_quantities(state)

        props = base_model.predict_properties(params, names=("stellar_mass_surviving",))
        chex.assert_trees_all_close(
            props["stellar_mass_surviving"],
            sfh_qty.stellar_mass_surviving,
            rtol=0,
            atol=0,
        )

    def test_sfr_100myr(self, base_model):
        """sfr_100myr matches state directly."""
        params = base_model.spec.sample(jax.random.PRNGKey(44))
        state = base_model.predict_state(params)
        sfh_qty = state_to_sfh_quantities(state)

        props = base_model.predict_properties(params, names=("sfr_100myr",))
        chex.assert_trees_all_close(
            props["sfr_100myr"],
            sfh_qty.sfr_100myr,
            rtol=0,
            atol=0,
        )

    def test_sfr_10myr(self, base_model):
        """sfr_10myr matches state directly."""
        params = base_model.spec.sample(jax.random.PRNGKey(45))
        state = base_model.predict_state(params)
        sfh_qty = state_to_sfh_quantities(state)

        props = base_model.predict_properties(params, names=("sfr_10myr",))
        chex.assert_trees_all_close(
            props["sfr_10myr"],
            sfh_qty.sfr_10myr,
            rtol=0,
            atol=0,
        )

    def test_ssfr(self, base_model):
        """ssfr == sfr_100myr / max(stellar_mass_surviving, 1e-30)."""
        params = base_model.spec.sample(jax.random.PRNGKey(46))
        state = base_model.predict_state(params)
        sfh_qty = state_to_sfh_quantities(state)

        props = base_model.predict_properties(params, names=("ssfr",))
        chex.assert_trees_all_close(
            props["ssfr"],
            sfh_qty.ssfr,
            rtol=0,
            atol=0,
        )

    def test_mass_weighted_age_gyr(self, base_model):
        """mass_weighted_age_gyr matches state computation exactly."""
        params = base_model.spec.sample(jax.random.PRNGKey(47))
        state = base_model.predict_state(params)
        sfh_qty = state_to_sfh_quantities(state)

        props = base_model.predict_properties(params, names=("mass_weighted_age_gyr",))
        chex.assert_trees_all_close(
            props["mass_weighted_age_gyr"],
            sfh_qty.mass_weighted_age_gyr,
            rtol=0,
            atol=0,
        )

    def test_mass_weighted_metallicity(self, base_model):
        """mass_weighted_metallicity matches state computation exactly."""
        params = base_model.spec.sample(jax.random.PRNGKey(48))
        state = base_model.predict_state(params)
        sfh_qty = state_to_sfh_quantities(state)

        props = base_model.predict_properties(params, names=("mass_weighted_metallicity",))
        chex.assert_trees_all_close(
            props["mass_weighted_metallicity"],
            sfh_qty.mass_weighted_metallicity,
            rtol=0,
            atol=0,
        )


class TestLegacyPathAgreement:
    """Test 2: Documented tolerance vs predict_sfh_quantities (different path)."""

    def test_stellar_mass_legacy(self, base_model):
        """stellar_mass vs predict_sfh_quantities.stellar_mass (legacy recompute)."""
        params = base_model.spec.sample(jax.random.PRNGKey(50))

        # New path via predict_properties
        props = base_model.predict_properties(params, names=("stellar_mass",))
        stellar_mass = props["stellar_mass"]

        # Legacy path via predict_sfh_quantities (independent recompute)
        sfh_legacy = base_model.predict_sfh_quantities(params)

        # Should agree to high tolerance (legacy path is a different recompute)
        rel_diff = jnp.abs(stellar_mass - sfh_legacy.stellar_mass) / (
            jnp.abs(sfh_legacy.stellar_mass) + 1e-30
        )
        max_rel_diff = float(jnp.max(rel_diff))

        print(f"Legacy path max relative diff (stellar_mass): {max_rel_diff:.2e}")
        assert max_rel_diff < 1e-6, f"Legacy mismatch: {max_rel_diff}"


class TestJITCompatibility:
    """Test 3: JIT/vmap/grad compatibility."""

    def test_jit_single_property(self, base_model):
        """predict_properties is JIT-compatible."""
        params = base_model.spec.sample(jax.random.PRNGKey(51))

        @jax.jit
        def compute_stellar_mass(p):
            return base_model.predict_properties(p, names=("stellar_mass",))["stellar_mass"]

        result = compute_stellar_mass(params)
        chex.assert_shape(result, ())
        chex.assert_tree_all_finite(result)

    def test_vmap_batch(self, base_model):
        """predict_properties vmap over batch of parameters."""
        params_batch = base_model.spec.sample_batch(jax.random.PRNGKey(52), n=5)

        def single_call(p):
            return base_model.predict_properties(p, names=("stellar_mass",))["stellar_mass"]

        vmapped = jax.vmap(single_call)
        result = vmapped(params_batch)

        chex.assert_shape(result, (5,))
        chex.assert_tree_all_finite(result)

    def test_grad_wrt_free_param(self, base_model):
        """Gradient through a free SFH parameter."""
        params = base_model.spec.sample(jax.random.PRNGKey(53))

        def mass_fn(p):
            return base_model.predict_properties(p, names=("stellar_mass",))["stellar_mass"]

        # Find a free SFH parameter
        free_sfh_params = [n for n in base_model.spec.free_params if n.startswith("sfh_")]
        if free_sfh_params:
            param_name = free_sfh_params[0]
            grad_fn = jax.grad(mass_fn)
            grads = grad_fn(params)

            # Gradient should be finite (possibly zero, but finite)
            assert jnp.isfinite(grads[param_name])


class TestKeyError:
    """Test 4: KeyError on unknown property."""

    def test_unknown_property_error(self, base_model):
        """predict_properties raises KeyError for unknown property."""
        params = base_model.spec.sample(jax.random.PRNGKey(54))

        with pytest.raises(KeyError) as exc_info:
            base_model.predict_properties(params, names=("nonexistent_prop",))

        assert "nonexistent_prop" in str(exc_info.value)
        assert "Unknown properties" in str(exc_info.value)

    def test_property_catalog_getitem_unknown(self, base_model):
        """PropertyCatalog raises KeyError on unknown property."""
        params = base_model.spec.sample(jax.random.PRNGKey(55))
        pred = base_model.predict(params)

        with pytest.raises(KeyError) as exc_info:
            _ = pred.properties["nonexistent"]

        assert "nonexistent" in str(exc_info.value)


class TestAttributeSugar:
    """Test 5: Attribute syntax on Prediction objects."""

    def test_attribute_access(self, base_model):
        """pred.stellar_mass == pred.properties["stellar_mass"]."""
        params = base_model.spec.sample(jax.random.PRNGKey(56))
        pred = base_model.predict(params)

        via_dict = pred.properties["stellar_mass"]
        via_attr = pred.stellar_mass

        chex.assert_trees_all_close(via_attr, via_dict, rtol=0, atol=0)

    def test_attribute_in_dir(self, base_model):
        """Property names appear in dir(pred)."""
        params = base_model.spec.sample(jax.random.PRNGKey(57))
        pred = base_model.predict(params)

        dir_list = dir(pred)
        assert "stellar_mass" in dir_list
        assert "ssfr" in dir_list

    def test_attribute_error_unknown(self, base_model):
        """Accessing unknown attribute raises AttributeError."""
        params = base_model.spec.sample(jax.random.PRNGKey(58))
        pred = base_model.predict(params)

        with pytest.raises(AttributeError) as exc_info:
            _ = pred.nonexistent_attribute

        assert "nonexistent_attribute" in str(exc_info.value)


class TestJAXArrayGuard:
    """Test 6: __jax_array__ guard."""

    def test_jax_array_guard(self, base_model):
        """Prediction.__jax_array__ raises on vmap/jit tracing."""
        params = base_model.spec.sample(jax.random.PRNGKey(59))
        pred = base_model.predict(params)

        with pytest.raises(TypeError) as exc_info:
            jax.numpy.asarray(pred)

        assert "not JIT/vmap-compatible" in str(exc_info.value)


class TestAvailableProperties:
    """Test 7: available_properties and collision detection."""

    def test_available_properties_contains_sfh_group(self, base_model):
        """available_properties includes all SFH group properties."""
        props = base_model.available_properties

        # 7 SFH properties from orchestrator ForwardState (no luminosity-weighted; Phase 1B)
        sfh_names = {
            "stellar_mass",
            "stellar_mass_surviving",
            "sfr_10myr",
            "sfr_100myr",
            "ssfr",
            "mass_weighted_age_gyr",
            "mass_weighted_metallicity",
        }

        for name in sfh_names:
            assert name in props

    def test_available_properties_sorted(self, base_model):
        """available_properties returns sorted names."""
        props = base_model.available_properties
        assert props == tuple(sorted(props))

    def test_stellar_manual_registration(self):
        """StellarSEDComponent is a bare-Protocol component (no SEDModelComponent
        base), so it registers properties manually at module scope; the module-level
        names are del'd, so no `properties` attr lingers on the class."""
        from tengri.components.stellar.component import StellarSEDComponent

        # The properties dict is NOT declared in the class body because Stellar
        # is a bare Protocol (no SEDModelComponent base), so no __init_subclass__.
        # Instead, properties are registered manually at module level and cleaned up.
        assert "properties" not in vars(StellarSEDComponent)

        # But the properties are registered (verified by list_properties)
        from tengri import list_properties

        props = list_properties(group="sfh")
        names = {p["name"] for p in props}
        expected = {
            "stellar_mass",
            "stellar_mass_surviving",
            "sfr_10myr",
            "sfr_100myr",
            "ssfr",
            "mass_weighted_age_gyr",
            "mass_weighted_metallicity",
        }
        assert expected.issubset(names)

    def test_init_subclass_auto_collection(self):
        """A SEDModelComponent subclass declaring in-body `properties` is
        auto-collected and the dict is deleted from the class (like priors)."""
        from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent
        from tengri.forward.properties import PROPERTY_REGISTRY, Property

        # Save both registries to restore after test
        saved_properties = {k: list(v) for k, v in PROPERTY_REGISTRY.items()}
        saved_components = dict(_REGISTRY)

        try:

            class _AutoCollectProbe(SEDModelComponent):
                name = "_autocollect_probe"
                parameter_prefix = "probe_"

                properties = {  # noqa: RUF012
                    "probe_value": Property(
                        units="",
                        group="probe",
                        doc="probe",
                        fn=lambda state, params: 1.0,
                    ),
                }

            # __init_subclass__ must have (a) deleted the class attr, (b) registered it
            assert "properties" not in vars(_AutoCollectProbe)
            assert "probe_value" in PROPERTY_REGISTRY
            assert any(
                e.component_name == "_autocollect_probe" for e in PROPERTY_REGISTRY["probe_value"]
            )
        finally:
            # Restore both registries so the probe doesn't pollute the session
            PROPERTY_REGISTRY.clear()
            PROPERTY_REGISTRY.update(saved_properties)
            _REGISTRY.clear()
            _REGISTRY.update(saved_components)

    def test_collision_detection_logic(self):
        """Collision detection logic filters properly."""
        from tengri.forward.properties import PropertyEntry

        # Create two entries for the same name
        entry1 = PropertyEntry(
            name="test_prop",
            units="Msun",
            group="test",
            doc="Test",
            component_name="comp1",
            fn=lambda state, params: 1.0,
        )
        entry2 = PropertyEntry(
            name="test_prop",
            units="Msun",
            group="test",
            doc="Test",
            component_name="comp2",
            fn=lambda state, params: 2.0,
        )

        # Create a manual registry with collision
        from tengri.forward import properties as props_mod

        original_registry = props_mod.PROPERTY_REGISTRY.copy()
        try:
            # Inject a collision scenario
            props_mod.PROPERTY_REGISTRY["collide_test"] = [entry1, entry2]

            # When both components are active, collision should be caught
            with pytest.raises(ValueError) as exc_info:
                assemble_available_properties({"comp1", "comp2"})

            assert "collide_test" in str(exc_info.value)

            # When only one component is active, no collision
            result = assemble_available_properties({"comp1"})
            assert "collide_test" in result
            assert result["collide_test"].component_name == "comp1"
        finally:
            # Restore the original registry
            props_mod.PROPERTY_REGISTRY.clear()
            props_mod.PROPERTY_REGISTRY.update(original_registry)


class TestPropertyCatalogInterface:
    """Test 8: PropertyCatalog dict-like interface."""

    def test_catalog_getitem(self, base_model):
        """PropertyCatalog.__getitem__ works."""
        params = base_model.spec.sample(jax.random.PRNGKey(60))
        pred = base_model.predict(params)

        value = pred.properties["stellar_mass"]
        chex.assert_shape(value, ())

    def test_catalog_contains(self, base_model):
        """PropertyCatalog.__contains__ works."""
        params = base_model.spec.sample(jax.random.PRNGKey(61))
        pred = base_model.predict(params)

        assert "stellar_mass" in pred.properties
        assert "nonexistent" not in pred.properties

    def test_catalog_iter(self, base_model):
        """PropertyCatalog iteration works."""
        params = base_model.spec.sample(jax.random.PRNGKey(62))
        pred = base_model.predict(params)

        names = list(pred.properties)
        assert "stellar_mass" in names
        assert len(names) > 0

    def test_catalog_to_dict(self, base_model):
        """PropertyCatalog.to_dict works."""
        params = base_model.spec.sample(jax.random.PRNGKey(63))
        pred = base_model.predict(params)

        d = pred.properties.to_dict(names=["stellar_mass", "ssfr"])
        assert "stellar_mass" in d
        assert "ssfr" in d
        assert len(d) == 2


class TestRegistryIntrospection:
    """Test registry listing and description."""

    def test_list_properties(self):
        """list_properties returns all registered properties."""
        from tengri import list_properties

        props = list_properties()
        assert len(props) > 0

        # Check structure
        assert all("name" in p for p in props)
        assert all("units" in p for p in props)
        assert all("group" in p for p in props)

    def test_list_properties_filter_group(self):
        """list_properties group filter works."""
        from tengri import list_properties

        sfh_props = list_properties(group="sfh")
        assert len(sfh_props) > 0
        assert all(p["group"] == "sfh" for p in sfh_props)

    def test_describe_property(self):
        """describe_property returns metadata."""
        from tengri import describe_property

        desc = describe_property("stellar_mass")
        assert "name" in desc
        assert desc["name"] == "stellar_mass"
        assert "units" in desc
        assert "group" in desc
