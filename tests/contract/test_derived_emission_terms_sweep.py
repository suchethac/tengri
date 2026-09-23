# SPDX-License-Identifier: BSD-3-Clause
"""Test that the emission_terms sweep is derived from the chain, not hardcoded.

The brief issue: a hardcoded tuple ("xray", "radio") at line 2974 of sed_model.py
means any new additive emitter added in future silently forfeits the band-response
optimization. This test verifies the sweep is now derived by probing which components
implement the emission_terms contract, so future emitters inherit the optimization
automatically.

This test must fail against the original hardcoded tuple and pass after the fix.
It constructs a mock component that implements emission_terms, adds it to a chain,
and verifies that the helper function finds it.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from tengri.forward.sed_model import _chain_implements_emission_terms


@dataclass
class MockAdditiveComponent:
    """Mock component for testing emission_terms detection.

    This is NOT a real SEDModelComponent, just a mock with a name and
    emission_terms method. The test uses it to verify that
    _chain_implements_emission_terms finds components by their method,
    not by a hardcoded list of names.
    """

    name: str = "mock_synchrotron"  # Not "xray" or "radio"

    def emission_terms(
        self, params: Mapping[str, jnp.ndarray], wave: jnp.ndarray, **kwargs
    ) -> dict[str, jnp.ndarray]:
        """Mock emission_terms method.

        Returns a single mock term for testing. In a real component, this would
        return a dict of rank-1 terms keyed by the term name (e.g., "HMXB",
        "LMXB", "hot_gas" for X-ray).
        """
        return {"mock_term": jnp.ones_like(wave) * 1e30}


class TestEmissionTermsSweepIsNotHardcoded:
    """Verify that the sweep is derived, not listed."""

    pytestmark = pytest.mark.contract

    def test_chain_implements_emission_terms_finds_radio_and_xray(self):
        """Known emitters (radio, xray) are found by the derived sweep.

        This is a sanity check that the helper function works at all. It must
        pass both before and after the fix (it verifies the function works with
        the known cases).
        """
        # Use mock components with names matching the real radio/xray names
        radio_mock = MockAdditiveComponent(name="radio")
        xray_mock = MockAdditiveComponent(name="xray")
        chain = [radio_mock, xray_mock]
        found = _chain_implements_emission_terms(chain)

        # Order is deterministic (sorted alphabetically)
        assert found == ["radio", "xray"], f"Expected ['radio', 'xray'], got {found}"

    def test_chain_implements_emission_terms_finds_mock_emitter(self):
        """A mock component with emission_terms is found even though it's not in hardcoded list.

        This is the key test: it constructs a component not in the hardcoded
        ("xray", "radio") tuple and verifies the derived sweep finds it. Against
        the original hardcoded tuple, this test FAILS. After the fix, it PASSES.
        """
        mock_component = MockAdditiveComponent(name="mock_synchrotron")
        chain = [mock_component]

        found = _chain_implements_emission_terms(chain)

        # If the sweep were still hardcoded as ("xray", "radio"), found would be [].
        # After the fix, the derived sweep finds the mock component.
        assert found == ["mock_synchrotron"], (
            f"Expected ['mock_synchrotron'], got {found}. "
            "This suggests the sweep is still hardcoded, not derived."
        )

    def test_chain_implements_emission_terms_mixed_chain(self):
        """Both hardcoded and mock emitters are found in a mixed chain."""
        radio_mock = MockAdditiveComponent(name="radio")
        xray_mock = MockAdditiveComponent(name="xray")
        mock_component = MockAdditiveComponent(name="mock_synchrotron")
        chain = [radio_mock, mock_component, xray_mock]

        found = _chain_implements_emission_terms(chain)

        # Deterministic order (sorted alphabetically)
        expected = ["mock_synchrotron", "radio", "xray"]
        assert found == expected, f"Expected {expected}, got {found}"

    def test_chain_implements_emission_terms_empty_chain(self):
        """An empty chain returns an empty list."""
        chain = []
        found = _chain_implements_emission_terms(chain)
        assert found == []

    def test_chain_implements_emission_terms_deterministic_order(self):
        """Order is deterministic (sorted by component name)."""
        # Create components with names in reverse alphabetical order
        components = [
            MockAdditiveComponent(name="zebra"),
            MockAdditiveComponent(name="apple"),
            MockAdditiveComponent(name="mango"),
        ]
        chain = components

        found = _chain_implements_emission_terms(chain)

        # Must be sorted alphabetically
        expected = ["apple", "mango", "zebra"]
        assert found == expected, (
            f"Expected {expected} (sorted), got {found}. "
            "Order must be deterministic for reproducible builds."
        )

    def test_chain_implements_emission_terms_ignores_non_emitters(self):
        """Components without emission_terms are ignored."""

        @dataclass
        class NonEmitterComponent:
            name: str = "some_other_component"

        non_emitter = NonEmitterComponent()
        mock_emitter = MockAdditiveComponent(name="mock_synchrotron")
        chain = [non_emitter, mock_emitter]

        found = _chain_implements_emission_terms(chain)

        # Only the mock_synchrotron (which has emission_terms) is found
        assert found == ["mock_synchrotron"]

    def test_chain_implements_emission_terms_handles_missing_name(self):
        """Components with missing name attribute are safely ignored."""

        @dataclass
        class BrokenComponent:
            # No name attribute
            def emission_terms(self, params, wave, **kwargs):
                return {"term": jnp.ones(5)}

        broken = BrokenComponent()
        good = MockAdditiveComponent(name="good_emitter")
        chain = [broken, good]

        found = _chain_implements_emission_terms(chain)

        # Only the component with a proper name is included
        assert found == ["good_emitter"]
