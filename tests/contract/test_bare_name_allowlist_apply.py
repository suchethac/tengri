# SPDX-License-Identifier: BSD-3-Clause
"""Unit test for SEDModelComponent.apply() to honor BARE_NAME_ALLOWLIST.

Tests that bare-name allowlist parameters (e.g., redshift) are passed
through unstripped to predict() and precomp paths, ensuring that
redshift-evolution components and CMB-aware dust emission can access
these global parameters.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def _cleanup_bare_name_registry():
    """Cleanup registry pollution from auto-registered test components (#853)."""
    saved_registry = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved_registry)


@pytest.mark.unit
class TestBareNameAllowlistApply:
    """SEDModelComponent.apply() honors BARE_NAME_ALLOWLIST."""

    def test_redshift_passes_through_to_predict(self):
        """redshift (bare-name) is available in predict() when provided."""

        class _BareNameTestComponent(SEDModelComponent):
            """Minimal component that records whether 'redshift' is in params."""

            name = "bare_name_test"
            parameter_prefix = "test_"

            # One free parameter to satisfy component contract. Needs default= because
            # this component registers globally in _REGISTRY, and the test_param_defaults
            # contract iterates every registered component (no default → pollutes it).
            x = Uniform(0.0, 10.0, description="Test parameter", units="", default=5.0)

            def __init__(self):
                """Initialize tracking state."""
                # Track what was passed to predict
                self.last_p_had_redshift = False
                self.last_redshift_value = None

            def predict(self, p, sed_in, wave, **inputs):
                """Record redshift presence in p, then return passthrough SED."""
                self.last_p_had_redshift = "redshift" in p
                if "redshift" in p:
                    self.last_redshift_value = float(p["redshift"])
                return sed_in, {}

        comp = _BareNameTestComponent()

        # Build minimal ForwardState
        wave = jnp.linspace(1e3, 1e5, 50)
        state = ForwardState(
            wave=wave,
            sed_intrinsic=None,
            derived={},
        )

        # Params with both prefix-matched and bare-name params
        params = {
            "test_x": 5.0,
            "redshift": 2.5,
        }

        # Apply component
        new_state = comp.apply(state, params)

        # Verify redshift was passed through
        assert comp.last_p_had_redshift, "redshift was not in predict's params dict"
        assert comp.last_redshift_value == 2.5, "redshift value did not match"

    def test_redshift_absent_when_not_provided(self):
        """redshift is absent in predict() when not provided."""

        class _BareNameTestComponent(SEDModelComponent):
            """Minimal component that records whether 'redshift' is in params."""

            name = "bare_name_test"
            parameter_prefix = "test_"

            # One free parameter to satisfy component contract. Needs default= because
            # this component registers globally in _REGISTRY, and the test_param_defaults
            # contract iterates every registered component (no default → pollutes it).
            x = Uniform(0.0, 10.0, description="Test parameter", units="", default=5.0)

            def __init__(self):
                """Initialize tracking state."""
                # Track what was passed to predict
                self.last_p_had_redshift = False
                self.last_redshift_value = None

            def predict(self, p, sed_in, wave, **inputs):
                """Record redshift presence in p, then return passthrough SED."""
                self.last_p_had_redshift = "redshift" in p
                if "redshift" in p:
                    self.last_redshift_value = float(p["redshift"])
                return sed_in, {}

        comp = _BareNameTestComponent()

        wave = jnp.linspace(1e3, 1e5, 50)
        state = ForwardState(
            wave=wave,
            sed_intrinsic=None,
            derived={},
        )

        # Params with only prefix-matched param
        params = {
            "test_x": 5.0,
        }

        # Apply component
        new_state = comp.apply(state, params)

        # Verify redshift was not passed through
        assert not comp.last_p_had_redshift, "redshift should not be in predict's params"

    def test_redshift_zero_passes_through(self):
        """redshift=0.0 (not redshift=None) is passed correctly."""

        class _BareNameTestComponent(SEDModelComponent):
            """Minimal component that records whether 'redshift' is in params."""

            name = "bare_name_test"
            parameter_prefix = "test_"

            # One free parameter to satisfy component contract. Needs default= because
            # this component registers globally in _REGISTRY, and the test_param_defaults
            # contract iterates every registered component (no default → pollutes it).
            x = Uniform(0.0, 10.0, description="Test parameter", units="", default=5.0)

            def __init__(self):
                """Initialize tracking state."""
                # Track what was passed to predict
                self.last_p_had_redshift = False
                self.last_redshift_value = None

            def predict(self, p, sed_in, wave, **inputs):
                """Record redshift presence in p, then return passthrough SED."""
                self.last_p_had_redshift = "redshift" in p
                if "redshift" in p:
                    self.last_redshift_value = float(p["redshift"])
                return sed_in, {}

        comp = _BareNameTestComponent()

        wave = jnp.linspace(1e3, 1e5, 50)
        state = ForwardState(
            wave=wave,
            sed_intrinsic=None,
            derived={},
        )

        params = {
            "test_x": 5.0,
            "redshift": 0.0,
        }

        new_state = comp.apply(state, params)

        # Verify redshift=0.0 is available (not dropped)
        assert comp.last_p_had_redshift, "redshift=0.0 should be in predict's params"
        assert comp.last_redshift_value == 0.0, "redshift value should be 0.0"


@pytest.mark.contract
class TestEmissionComponentBareNameAllowlist:
    """EmissionComponent.apply() — the SECOND prefix-slicing path — honors BARE_NAME_ALLOWLIST.

    The base :class:`SEDModelComponent.apply` (covered above) is not the only
    method that slices params by prefix: dust IR emission components override
    ``apply`` with a bespoke three-branch implementation
    (:class:`~tengri.components.dust.emission._component_base.EmissionComponent`).
    That override strips the ``dust_`` prefix too, so it must re-inject the
    bare-name allowlist independently. This is the path that (before the base
    ``__init_subclass__`` shadowing fix) made dust emission a TOTAL no-op, and
    it is where CMB-at-high-z heating reads ``redshift``.

    Audit note (#853): these two are the ONLY ``apply()`` implementations that
    slice by prefix. Every bare-Protocol component (igm/agn/radio/xray/nebular)
    reads ``redshift`` from the full params dict by full key, so none can drop
    it.

    Uses a real registered component (``modified_blackbody``) rather than a
    bespoke subclass to avoid polluting ``_REGISTRY`` / the emission menu, and
    to exercise the exact production code path.
    """

    @pytest.fixture(autouse=True)
    def _registry_cleanup(self):
        """Ensure registry stays clean across tests."""
        saved_registry = dict(_REGISTRY)
        yield
        _REGISTRY.clear()
        _REGISTRY.update(saved_registry)

    @staticmethod
    def _mbb_component_with_spy():
        """Return an mbb emission component whose ``predict`` records the seen
        redshift.

        The spy records what ``apply()`` threads into ``predict`` and returns a
        passthrough SED — it does NOT run the real mbb physics (that would
        require the full ``dust_*`` param set, which is irrelevant to whether
        the bare-name ``redshift`` survived the prefix-slice).
        """
        from tengri.components.sed_model_component import _REGISTRY

        component = _REGISTRY["modified_blackbody"]()
        seen: dict[str, object] = {"redshift": "unset"}

        def _spy(p, sed_in, wave, **inputs):
            seen["redshift"] = float(p["redshift"]) if "redshift" in p else None
            return sed_in, {}

        component.predict = _spy  # instance-level shadow; apply() calls self.predict
        return component, seen

    def test_redshift_threads_into_predict_exact_path(self):
        """Exact full-wave branch: redshift reaches the component's predict()."""
        component, seen = self._mbb_component_with_spy()
        wave = jnp.linspace(1e4, 1e7, 200)
        state = ForwardState(wave=wave, sed_intrinsic=None, derived={"L_ir": jnp.asarray(1e44)})

        component.apply(state, {"dust_T": 40.0, "dust_beta_ir": 1.6, "redshift": 3.0})

        assert seen["redshift"] == 3.0, (
            "EmissionComponent.apply() (exact path) dropped the bare-name redshift"
        )

    def test_redshift_threads_into_predict_photometry_precomp(self):
        """Photometry-LUT branch: redshift reaches predict() there too (CMB-at-z)."""
        component, seen = self._mbb_component_with_spy()
        wave = jnp.linspace(1e4, 1e7, 200)
        # filter_eff_waves present -> apply() takes the photometry-precomp branch,
        # which itself calls predict() (line 180 of _component_base.py).
        state = ForwardState(
            wave=wave,
            sed_intrinsic=None,
            derived={"L_ir": jnp.asarray(1e44), "filter_eff_waves": jnp.array([1e5, 1e6])},
        )

        component.apply(state, {"dust_T": 40.0, "dust_beta_ir": 1.6, "redshift": 4.0})

        assert seen["redshift"] == 4.0, (
            "EmissionComponent.apply() (photometry-precomp path) dropped the bare-name redshift"
        )
