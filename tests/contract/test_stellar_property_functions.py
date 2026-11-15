# SPDX-License-Identifier: BSD-3-Clause
"""Stellar property functions evaluated on a hand-built ForwardState.

The functions live in :mod:`tengri.components.stellar.properties`. Each is a pure
function of ``state.derived``, so its closed form can be checked without an SSP
grid, a model or a sampled parameter point.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.properties import (
    _ssfr_fn,
    _stellar_mass_fn,
    _timing_gyr_fn,
)
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.contract

_WAVE = jnp.linspace(1000.0, 10000.0, 16)


def _state(**derived) -> ForwardState:
    return ForwardState(wave=_WAVE, derived=derived)


def test_stellar_mass_is_the_formed_mass_key():
    """stellar_mass is 10**log_mstar_formed: the formed mass, not the surviving one."""
    state = _state(log_mstar_formed=9.0, log_mstar=8.7)
    value = _stellar_mass_fn(state, {})
    np.testing.assert_allclose(float(value), 1e9, rtol=1e-12)


def test_ssfr_is_sfr_over_surviving_mass():
    """ssfr = sfr_100myr / 10**log_mstar, with log_mstar the surviving mass."""
    state = _state(sfr_100myr=2.0, log_mstar=2.0)
    value = _ssfr_fn(state, {})
    np.testing.assert_allclose(float(value), 2.0 / 100.0, rtol=1e-12)


def test_timing_gyr_reads_its_key_and_is_nan_when_absent():
    """A timing property returns its resolved Gyr value, or NaN when the SFH lacks it."""
    fn = _timing_gyr_fn("sfh_delayed_age_gyr")
    present = _state(sfh_timing_gyr={"sfh_delayed_age_gyr": 0.75})
    absent = _state(sfh_timing_gyr={"sfh_other_gyr": 0.2})
    np.testing.assert_allclose(float(fn(present, {})), 0.75, rtol=1e-12)
    assert np.isnan(float(fn(absent, {})))
