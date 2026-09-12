# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for DIG short-circuit when frac=0 bug.

Bug: dig.py:110-113 — DIG forward pass always called even when neb_dig_frac=0.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestDIGShortCircuit:
    """Bug: dig.py:110-113 — DIG forward pass not short-circuited when neb_dig_frac=0."""

    def test_dig_zero_frac_short_circuits(self):
        """With neb_dig_frac=0.0 (Python float), predict_nebular_sed should only be called once."""
        call_count = {"n": 0}

        class _FakeBackend:
            def predict_nebular_sed(self, **kwargs):
                call_count["n"] += 1
                return jnp.zeros(100)

        from tengri.components.nebular.dig import mix_dig_emission

        wave = jnp.linspace(1000.0, 10000.0, 100)
        weights = jnp.ones(10) / 10.0
        log_ages = jnp.linspace(6.0, 10.0, 10)
        mix_dig_emission(
            _FakeBackend(),
            ssp_wave=wave,
            ssp_weights=weights,
            ssp_log_ages_yr=log_ages,
            log_z=-1.848,
            neb_logU=-3.0,
            neb_dig_frac=0.0,  # Python float — should short-circuit
            neb_dig_delta_logU=-1.0,
            line_sigma_aa=50.0,
        )
        assert call_count["n"] == 1, (
            f"DIG forward pass called {call_count['n']} times with neb_dig_frac=0.0; "
            "expected 1 (short-circuit)"
        )

    def test_dig_active_false_short_circuits(self):
        """With dig_active=False, DIG forward pass is skipped even with Python 0.0."""
        call_count = {"n": 0}

        class _FakeBackend:
            def predict_nebular_sed(self, **kwargs):
                call_count["n"] += 1
                return jnp.zeros(100)

        from tengri.components.nebular.dig import mix_dig_emission

        wave = jnp.linspace(1000.0, 10000.0, 100)
        weights = jnp.ones(10) / 10.0
        log_ages = jnp.linspace(6.0, 10.0, 10)
        mix_dig_emission(
            _FakeBackend(),
            ssp_wave=wave,
            ssp_weights=weights,
            ssp_log_ages_yr=log_ages,
            log_z=-1.848,
            neb_logU=-3.0,
            neb_dig_frac=0.0,
            neb_dig_delta_logU=-1.0,
            line_sigma_aa=50.0,
            dig_active=False,  # Explicit build-time decision
        )
        assert call_count["n"] == 1, (
            f"DIG forward pass called {call_count['n']} times with dig_active=False; "
            "expected 1 (skip DIG)"
        )

    def test_dig_active_true_forces_both(self):
        """With dig_active=True, both HII and DIG evaluations run even with Python 0.0."""
        call_count = {"n": 0}

        class _FakeBackend:
            def predict_nebular_sed(self, **kwargs):
                call_count["n"] += 1
                return jnp.zeros(100)

        from tengri.components.nebular.dig import mix_dig_emission

        wave = jnp.linspace(1000.0, 10000.0, 100)
        weights = jnp.ones(10) / 10.0
        log_ages = jnp.linspace(6.0, 10.0, 10)
        mix_dig_emission(
            _FakeBackend(),
            ssp_wave=wave,
            ssp_weights=weights,
            ssp_log_ages_yr=log_ages,
            log_z=-1.848,
            neb_logU=-3.0,
            neb_dig_frac=0.0,  # Python float is 0.0
            neb_dig_delta_logU=-1.0,
            line_sigma_aa=50.0,
            dig_active=True,  # Force both evaluations
        )
        assert call_count["n"] == 2, (
            f"DIG forward pass called {call_count['n']} times with dig_active=True; "
            "expected 2 (both HII and DIG)"
        )
