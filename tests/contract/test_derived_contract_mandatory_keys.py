# SPDX-License-Identifier: BSD-3-Clause
"""Mandatory derived keys are indexed, and a raising ``inputs()`` propagates.

``log_nion`` and ``lnu_age`` are unconditional outputs of the stellar
component, so a consumer that reads them with a default turns a missing key
into a silent zero (#1738). These tests pin the direct-index contract at the
three consumers that used to default, and the chain census that used to
swallow an exception raised by a component's ``inputs()``.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract


def test_state_to_sed_components_requires_lnu_age():
    """A derived dict without ``lnu_age`` raises KeyError instead of zeros."""
    from tengri.forward.component_factory import state_to_sed_components

    state = SimpleNamespace(
        derived={},
        wave=jnp.linspace(1000.0, 2000.0, 8),
        sed_intrinsic=None,
    )

    with pytest.raises(KeyError, match="lnu_age"):
        state_to_sed_components(state)


def test_radio_thermal_requires_log_nion():
    """A derived dict without ``log_nion`` raises KeyError instead of -inf."""
    from tengri.components.radio.component import _l_thermal_fn

    state = SimpleNamespace(derived={})

    with pytest.raises(KeyError, match="log_nion"):
        _l_thermal_fn(state, params=None)


def test_chain_consumes_propagates_raising_inputs():
    """An exception from ``component.inputs()`` reaches the caller."""
    from tengri.forward.sed_model import _chain_consumes

    class _Broken:
        def inputs(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _chain_consumes([_Broken()], "L_ir")
