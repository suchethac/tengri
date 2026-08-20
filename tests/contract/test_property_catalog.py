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
        dust={
            "type": "two_component",
            "law": "calzetti",
            "*": FIXED,
            "tau_bc": 0.5,
        },
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
        # The rule, not the phrasing: a name in no registry is reported as
        # unknown. The plural "Unknown properties" was pinned here and is gone —
        # each bad name now gets its own diagnosis, because a *registered*
        # property missing from this model is not unknown at all (#1706).
        assert "Unknown property" in str(exc_info.value)

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


# ─────────────────────────────────────────────────────────────────────
# Phase 1B: SED, Lines, Radio, Xray, Ionizing, Luminosity-weighted
# ─────────────────────────────────────────────────────────────────────


class TestSEDGroupBitEquality:
    """Test SED group properties bit-equality against state_to_sed_quantities."""

    @pytest.fixture
    def sed_model(self, synthetic_ssp_wide):
        """Model with dust (required for full SED properties)."""
        ssp = synthetic_ssp_wide
        spec = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )
        return spec

    def test_l_bol(self, sed_model):
        """l_bol matches state_to_sed_quantities."""
        from tengri.forward.component_factory import state_to_sed_quantities

        params = sed_model.spec.sample(jax.random.PRNGKey(100))
        state = sed_model.predict_state(params)
        sed_qty = state_to_sed_quantities(state)

        props = sed_model.predict_properties(params, names=("l_bol",))
        chex.assert_trees_all_close(props["l_bol"], sed_qty.l_bol, rtol=0, atol=0)

    def test_l_tir(self, sed_model):
        """l_tir matches state_to_sed_quantities."""
        from tengri.forward.component_factory import state_to_sed_quantities

        params = sed_model.spec.sample(jax.random.PRNGKey(101))
        state = sed_model.predict_state(params)
        sed_qty = state_to_sed_quantities(state)

        props = sed_model.predict_properties(params, names=("l_tir",))
        chex.assert_trees_all_close(props["l_tir"], sed_qty.l_tir, rtol=0, atol=0)

    def test_irx(self, sed_model):
        """irx matches state_to_sed_quantities."""
        from tengri.forward.component_factory import state_to_sed_quantities

        params = sed_model.spec.sample(jax.random.PRNGKey(102))
        state = sed_model.predict_state(params)
        sed_qty = state_to_sed_quantities(state)

        props = sed_model.predict_properties(params, names=("irx",))
        chex.assert_trees_all_close(props["irx"], sed_qty.irx, rtol=0, atol=0)

    def test_uv_slope_beta(self, sed_model):
        """uv_slope_beta matches state_to_sed_quantities."""
        from tengri.forward.component_factory import state_to_sed_quantities

        params = sed_model.spec.sample(jax.random.PRNGKey(103))
        state = sed_model.predict_state(params)
        sed_qty = state_to_sed_quantities(state)

        props = sed_model.predict_properties(params, names=("uv_slope_beta",))
        chex.assert_trees_all_close(props["uv_slope_beta"], sed_qty.uv_slope_beta, rtol=0, atol=0)


class TestLuminosityWeightedSFHBitEquality:
    """Test luminosity-weighted properties bit-equality against prediction.sfh."""

    @pytest.fixture
    def lw_model(self, synthetic_ssp_wide):
        """Model for luminosity-weighted tests."""
        ssp = synthetic_ssp_wide
        spec = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )
        return spec

    def test_luminosity_weighted_age_gyr(self, lw_model):
        """luminosity_weighted_age_gyr matches Prediction.sfh."""
        params = lw_model.spec.sample(jax.random.PRNGKey(110))
        pred = lw_model.predict(params)

        props = lw_model.predict_properties(params, names=("luminosity_weighted_age_gyr",))
        # Documented float-precision tolerance (not rtol=0): the catalog fn
        # computes from the orchestrator's published per-age luminosity
        # (state.derived["L_age"]) while Prediction.sfh recomputes from the full
        # SED cube via compute_luminosity_weighted_age — two different routes to
        # the same quantity, agreeing to ~1 ULP (CI synthetic data exposed a
        # 3.6e-16 relative gap that real-data rounding happened to hide).
        chex.assert_trees_all_close(
            props["luminosity_weighted_age_gyr"],
            pred.sfh.luminosity_weighted_age_gyr,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_luminosity_weighted_metallicity(self, lw_model):
        """luminosity_weighted_metallicity matches Prediction.sfh."""
        params = lw_model.spec.sample(jax.random.PRNGKey(111))
        pred = lw_model.predict(params)

        props = lw_model.predict_properties(params, names=("luminosity_weighted_metallicity",))
        # Documented float-precision tolerance (see luminosity_weighted_age_gyr):
        # orchestrator-L_age route vs the lazy SED-cube helper agree to ~ULP.
        chex.assert_trees_all_close(
            props["luminosity_weighted_metallicity"],
            pred.sfh.luminosity_weighted_metallicity,
            rtol=1e-12,
            atol=1e-12,
        )


class TestIonizingGroupBitEquality:
    """Test ionizing group properties bit-equality against state_to_ionizing_quantities."""

    @pytest.fixture
    def ion_model(self, synthetic_ssp_wide):
        """Model for ionizing tests."""
        ssp = synthetic_ssp_wide
        spec = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )
        return spec

    def test_q_h(self, ion_model):
        """q_h matches state_to_ionizing_quantities."""
        from tengri.forward.component_factory import state_to_ionizing_quantities

        params = ion_model.spec.sample(jax.random.PRNGKey(120))
        state = ion_model.predict_state(params)
        ion_qty = state_to_ionizing_quantities(state)

        props = ion_model.predict_properties(params, names=("q_h",))
        chex.assert_trees_all_close(props["q_h"], ion_qty.q_h, rtol=0, atol=0)

    def test_xi_ion(self, ion_model):
        """xi_ion matches state_to_ionizing_quantities."""
        from tengri.forward.component_factory import state_to_ionizing_quantities

        params = ion_model.spec.sample(jax.random.PRNGKey(121))
        state = ion_model.predict_state(params)
        ion_qty = state_to_ionizing_quantities(state)

        props = ion_model.predict_properties(params, names=("xi_ion",))
        chex.assert_trees_all_close(props["xi_ion"], ion_qty.xi_ion, rtol=0, atol=0)


class TestRadioGroupBitEquality:
    """Test radio group properties bit-equality against state_to_radio_quantities."""

    @pytest.fixture
    def radio_model(self, synthetic_ssp_wide):
        """Model with radio component."""
        ssp = synthetic_ssp_wide
        spec = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            neb={"type": "none"},
            radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}, "*": FIXED},
            redshift=Fixed(0.1),
        )
        return spec

    def test_l_1p4ghz(self, radio_model):
        """l_1p4ghz matches state_to_radio_quantities."""
        from tengri.forward.component_factory import state_to_radio_quantities

        params = radio_model.spec.sample(jax.random.PRNGKey(130))
        state = radio_model.predict_state(params)
        radio_qty = state_to_radio_quantities(state)

        props = radio_model.predict_properties(params, names=("l_1p4ghz",))
        # Bit-equal: the property fn and state_to_radio_quantities interpolate
        # sed_radio at the identical 21 cm wavelength literal (_WAVE_21CM_AA).
        chex.assert_trees_all_close(props["l_1p4ghz"], radio_qty.l_1p4ghz, rtol=0, atol=0)

    def test_l_thermal(self, radio_model):
        """l_thermal matches state_to_radio_quantities."""
        from tengri.forward.component_factory import state_to_radio_quantities

        params = radio_model.spec.sample(jax.random.PRNGKey(131))
        state = radio_model.predict_state(params)
        radio_qty = state_to_radio_quantities(state)

        props = radio_model.predict_properties(params, names=("l_thermal",))
        chex.assert_trees_all_close(props["l_thermal"], radio_qty.l_thermal, rtol=0, atol=0)


class TestXRayGroupBitEquality:
    """Test xray group properties bit-equality against state_to_xray_quantities."""

    @pytest.fixture
    def xray_model(self, synthetic_ssp_wide):
        """Model with X-ray component."""
        ssp = synthetic_ssp_wide
        spec = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            neb={"type": "none"},
            xray={"type": "simple", "*": FIXED},
            redshift=Fixed(0.1),
        )
        return spec

    def test_l_x_xrb(self, xray_model):
        """l_x_xrb matches state_to_xray_quantities."""
        from tengri.forward.component_factory import state_to_xray_quantities

        params = xray_model.spec.sample(jax.random.PRNGKey(140))
        state = xray_model.predict_state(params)
        xray_qty = state_to_xray_quantities(state)

        props = xray_model.predict_properties(params, names=("l_x_xrb",))
        chex.assert_trees_all_close(props["l_x_xrb"], xray_qty.l_x_xrb, rtol=0, atol=0)

    def test_l_x_agn(self, xray_model):
        """l_x_agn matches state_to_xray_quantities (0 when inactive)."""
        from tengri.forward.component_factory import state_to_xray_quantities

        params = xray_model.spec.sample(jax.random.PRNGKey(141))
        state = xray_model.predict_state(params)
        xray_qty = state_to_xray_quantities(state)

        props = xray_model.predict_properties(params, names=("l_x_agn",))
        chex.assert_trees_all_close(props["l_x_agn"], xray_qty.l_x_agn, rtol=0, atol=0)


class TestLinesGroupNaNBehavior:
    """Test lines group returns NaN when catalog unavailable."""

    @pytest.fixture
    def no_lines_model(self, synthetic_ssp_wide):
        """Model with no discrete line catalog."""
        ssp = synthetic_ssp_wide
        spec = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "*": FREE},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            neb={"type": "none"},  # No nebular → no line catalog
            redshift=Fixed(0.1),
        )
        return spec

    def test_halpha_is_nan_when_no_catalog(self, no_lines_model):
        """halpha returns NaN when nebular component has no catalog."""
        params = no_lines_model.spec.sample(jax.random.PRNGKey(150))

        props = no_lines_model.predict_properties(params, names=("halpha",))
        assert jnp.isnan(props["halpha"])

    def test_bpt_nii_is_nan_when_no_catalog(self, no_lines_model):
        """bpt_nii returns NaN when nebular component has no catalog."""
        params = no_lines_model.spec.sample(jax.random.PRNGKey(151))

        props = no_lines_model.predict_properties(params, names=("bpt_nii",))
        assert jnp.isnan(props["bpt_nii"])


class TestPhase1BPropertyCounts:
    """Test that all Phase 1B properties are registered."""

    def test_total_property_count(self):
        """Verify total count includes Phase 1A + 1B."""
        from tengri import list_properties

        props = list_properties()
        # Phase 1A: 7 SFH
        # Phase 1B: 13 SED + 2 lum-weighted + 2 ionizing + 11 lines + 6 ratios + 4 radio + 3 xray
        # Total: 7 + 41 = 48
        assert len(props) >= 48, f"Expected at least 48 properties, got {len(props)}"

    def test_sed_group_count(self):
        """Verify SED group has all 15 properties."""
        from tengri import list_properties

        sed_props = list_properties(group="sed")
        sed_names = {p["name"] for p in sed_props}
        expected = {
            "l_bol",
            "l_tir",
            "l_dust_absorbed",
            "irx",
            "uv_slope_beta",
            "dn4000",
            "balmer_break",
            "m_uv",
            "fuv_flux",
            "nuv_flux",
            "fuv_flux_intrinsic",
            "nuv_flux_intrinsic",
            "rest_uv_color",
        }
        assert expected.issubset(sed_names), f"Missing SED properties: {expected - sed_names}"

    def test_lines_group_count(self):
        """Verify lines group has all 17 properties (11 + 6)."""
        from tengri import list_properties

        lines_props = list_properties(group="lines")
        lines_names = {p["name"] for p in lines_props}
        expected = {
            "lya",
            "civ_1549",
            "oii",
            "hbeta",
            "oiii_4959",
            "oiii_5007",
            "nii_6548",
            "halpha",
            "nii_6584",
            "sii_6717",
            "sii_6731",
            "bpt_nii",
            "bpt_sii",
            "o3hb",
            "r23",
            "o32",
            "balmer_decrement",
        }
        missing = expected - lines_names
        assert expected.issubset(lines_names), f"Missing lines properties: {missing}"

    def test_radio_group_count(self):
        """Verify radio group has all 4 properties."""
        from tengri import list_properties

        radio_props = list_properties(group="radio")
        radio_names = {p["name"] for p in radio_props}
        expected = {"l_1p4ghz", "l_thermal", "l_nonthermal", "q_ir"}
        missing = expected - radio_names
        assert expected.issubset(radio_names), f"Missing radio properties: {missing}"

    def test_xray_group_count(self):
        """Verify xray group has all 3 properties."""
        from tengri import list_properties

        xray_props = list_properties(group="xray")
        xray_names = {p["name"] for p in xray_props}
        expected = {"l_x_xrb", "l_x_agn", "l_x_total"}
        assert expected.issubset(xray_names), f"Missing xray properties: {expected - xray_names}"

    def test_ionizing_group_count(self):
        """Verify ionizing group has ionizing properties."""
        from tengri import list_properties

        ion_props = list_properties(group="ionizing")
        ion_names = {p["name"] for p in ion_props}
        expected = {"q_h", "xi_ion"}
        assert expected.issubset(ion_names), f"Missing ionizing properties: {expected - ion_names}"

    def test_sfh_group_count_extended(self):
        """Verify sfh group has Phase 1A + lum-weighted properties."""
        from tengri import list_properties

        sfh_props = list_properties(group="sfh")
        sfh_names = {p["name"] for p in sfh_props}
        # sfh group includes Phase 1A (7) + Phase 1B lum-weighted (2)
        expected_in_sfh = {
            "stellar_mass",
            "stellar_mass_surviving",
            "sfr_10myr",
            "sfr_100myr",
            "ssfr",
            "mass_weighted_age_gyr",
            "mass_weighted_metallicity",
            "luminosity_weighted_age_gyr",
            "luminosity_weighted_metallicity",
        }
        missing = expected_in_sfh - sfh_names
        assert expected_in_sfh.issubset(sfh_names), f"Missing sfh: {missing}"


class TestDictLikeInterface:
    """PropertyCatalog is documented as dict-like — complete the mapping protocol.

    Regression for the fresh-user audit (2026-07): the quickstart notebook did
    ``pred.properties.get(name)`` and hit ``AttributeError: 'PropertyCatalog'
    object has no attribute 'get'`` — the class implemented ``__getitem__`` /
    ``__contains__`` / ``keys`` but not ``get`` / ``values`` / ``items``.
    """

    def test_get_returns_value_or_default(self, base_model):
        params = base_model.spec.sample(jax.random.PRNGKey(0))
        props = base_model.predict(params).properties
        assert float(props.get("stellar_mass")) == float(props["stellar_mass"])
        assert props.get("definitely_not_a_property") is None
        assert props.get("definitely_not_a_property", 42) == 42

    def test_values_and_items_align_with_keys(self, base_model):
        params = base_model.spec.sample(jax.random.PRNGKey(0))
        props = base_model.predict(params).properties
        keys, vals, items = props.keys(), props.values(), props.items()
        assert len(keys) == len(vals) == len(items) > 0
        assert [name for name, _ in items] == keys
        assert float(items[0][1]) == float(props[keys[0]])
