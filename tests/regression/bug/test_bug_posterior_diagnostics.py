# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for posterior summary_table diagnostic key names bug.

Bug: posterior.py:349-352 — checked 'acceptance_rate'/'n_divergences' but
raytrace stores 'accept_rate' and NUTS stores 'n_divergent'.
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


class TestPosteriorDiagnosticKeys:
    """Bug: posterior.py:349-352 — wrong diagnostic key names."""

    def test_accept_rate_key_shown(self):
        """summary_table() should display acceptance rate when 'accept_rate' is in diagnostics."""
        from tengri.inference.posterior import Posterior

        p = Posterior(
            samples=None,
            params={"x": jnp.array(1.0)},
            method="raytrace",
            wall_time_s=1.0,
            diagnostics={"accept_rate": 0.55},
        )
        table = p.summary_table()
        assert "accept=55.0%" in table, f"accept_rate not shown in summary_table:\n{table}"

    def test_n_divergent_key_shown(self):
        """summary_table() should display divergences when 'n_divergent' is in diagnostics."""
        from tengri.inference.posterior import Posterior

        key = jax.random.PRNGKey(0)
        samples = {"x": jax.random.normal(key, (100,))}
        p = Posterior(
            samples=samples,
            params={"x": jnp.array(1.0)},
            method="nuts",
            wall_time_s=1.0,
            diagnostics={"n_divergent": 3},
        )
        table = p.summary_table()
        assert "divergences=3" in table, f"n_divergent not shown in summary_table:\n{table}"

    def test_wrong_keys_not_used(self):
        """Old wrong key names should not trigger the diagnostic display."""
        from tengri.inference.posterior import Posterior

        p = Posterior(
            samples=None,
            params={"x": jnp.array(1.0)},
            method="raytrace",
            wall_time_s=1.0,
            diagnostics={"acceptance_rate": 0.55, "n_divergences": 3},
        )
        table = p.summary_table()
        # With the wrong keys, nothing should be shown
        assert "accept=" not in table, "Old wrong key 'acceptance_rate' is being read"
        assert "divergences=" not in table, "Old wrong key 'n_divergences' is being read"
