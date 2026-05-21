# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the canonical SFR-window-averaging helper.

The forward model exposes a single ``_sfr_current`` quantity to downstream
SFR-driven components (radio, X-ray, nebular Q_H scaling). This used to
be set to ``sfr[-1]`` — the SFR at the boundary of the lookback grid,
which actually returns the **oldest** bin for the canonical
``[1 Myr, 13.8 Gyr]`` ascending lookback grid. The replacement uses a
10 Myr time-weighted average (Murphy+2011 timescale).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.sfr_window import time_weighted_sfr


@pytest.mark.unit
def test_constant_sfr_recovers_input():
    """For a constant SFR, the window average must equal the input value."""
    lbt = jnp.logspace(6, 10, 64)
    sfr = jnp.full_like(lbt, 7.5)
    out = float(time_weighted_sfr(sfr, lbt, 1e7))
    assert out == pytest.approx(7.5, rel=1e-6)


@pytest.mark.unit
def test_declining_sfh_returns_recent_value():
    """For a strongly declining SFH, the window average must be close to the
    most recent SFR, not the oldest. Guards against ``sfr[-1]`` regression."""
    lbt = jnp.logspace(6, 10, 64)
    # Declining exponential: high SFR at low lookback, near-zero at old ages
    sfr = 100.0 * jnp.exp(-lbt / 5e8)
    avg_10myr = float(time_weighted_sfr(sfr, lbt, 1e7))
    # 10 Myr is << 5e8 yr e-folding, so SFR at the recent end is ~unchanged
    assert avg_10myr > 50.0, (
        f"10 Myr-averaged SFR for declining SFH should be near peak (>50), "
        f"got {avg_10myr:.2f} — likely picking up the OLD end of the grid"
    )
    # Compare against sfr[0] (most recent bin) — should be very close
    sfr_recent = float(sfr[0])
    assert avg_10myr == pytest.approx(sfr_recent, rel=0.05)


@pytest.mark.unit
def test_old_grid_value_does_not_dominate():
    """Regression guard: ``sfr[-1]`` (oldest bin) must NOT be returned even
    when it is the largest value in the array."""
    lbt = jnp.logspace(6, 10, 64)
    # Inverted: tiny recent SFR, huge old SFR — ``sfr[-1]`` would be 1e3.
    sfr = jnp.where(lbt > 1e9, 1e3, 1e-3)
    avg_10myr = float(time_weighted_sfr(sfr, lbt, 1e7))
    # The 10 Myr window only contains lbt < 1e7, where SFR = 1e-3
    assert avg_10myr == pytest.approx(1e-3, rel=0.1)
    assert avg_10myr < 1.0, f"Old-bin SFR (1e3) leaked into the 10 Myr average ({avg_10myr:.2e})"


@pytest.mark.unit
def test_window_size_changes_average():
    """100 Myr window must include more grid points than 10 Myr."""
    lbt = jnp.logspace(6, 10, 64)
    sfr = 100.0 * jnp.exp(-lbt / 5e7)  # 50 Myr e-folding
    avg_10myr = float(time_weighted_sfr(sfr, lbt, 1e7))
    avg_100myr = float(time_weighted_sfr(sfr, lbt, 1e8))
    # 10 Myr window: SFR ~ peak (lbt << tau)
    # 100 Myr window: SFR averages over the e-fold, so smaller
    assert avg_10myr > avg_100myr, (
        f"10 Myr avg ({avg_10myr:.2f}) must exceed 100 Myr avg ({avg_100myr:.2f}) "
        f"for a 50 Myr-decay SFH"
    )


@pytest.mark.unit
def test_jit_compatible():
    """Helper must compile under jax.jit (used inside fused kernels)."""
    import jax

    fn = jax.jit(time_weighted_sfr)
    lbt = jnp.logspace(6, 10, 64)
    sfr = jnp.full_like(lbt, 3.0)
    out = float(fn(sfr, lbt, 1e7))
    assert out == pytest.approx(3.0, rel=1e-6)


@pytest.mark.unit
def test_legacy_alias_delegates_to_canonical():
    """The private alias on StellarSEDComponent must call the canonical helper."""
    from tengri.components.stellar.component import _time_weighted_sfr

    lbt = jnp.logspace(6, 10, 64)
    sfr = jnp.linspace(0.0, 10.0, 64)
    a = float(_time_weighted_sfr(sfr, lbt, 1e7))
    b = float(time_weighted_sfr(sfr, lbt, 1e7))
    assert a == pytest.approx(b, abs=1e-12)


@pytest.mark.unit
def test_fallback_when_window_empty():
    """If no grid point falls inside the window, return the most recent bin."""
    # Grid where the smallest lbt is still bigger than the window
    lbt = jnp.array([1e8, 5e8, 1e9, 5e9])
    sfr = jnp.array([2.0, 1.0, 0.5, 0.1])
    out = float(time_weighted_sfr(sfr, lbt, 1e7))
    # Window is 10 Myr, smallest grid point is 100 Myr — window empty;
    # fallback returns sfr[0]
    assert out == pytest.approx(2.0, rel=1e-6)


def test_no_residual_sfr_minus_one_in_orchestrator():
    """Regression: ``sfr[-1]`` / ``sfr_on_ssp[-1]`` must not be set as
    ``_sfr_current`` anywhere in the live orchestrator path. The canonical
    helper is now the only correct source of present-day SFR.

    Phase 6 (PR #135): kernel adapter family deleted; only sed_model and
    the orchestrator carry the production SFR path now.
    """
    import inspect

    from tengri.forward import orchestrator, sed_model

    src = "\n".join(
        [
            inspect.getsource(sed_model),
            inspect.getsource(orchestrator),
        ]
    )

    # The forbidden patterns in production code:
    forbidden = [
        '"_sfr_current": sfr[-1]',
        '"_sfr_current": sfr_on_ssp[-1]',
        "_sfr_current = sfr[-1]",
        "_sfr_current = sfr_on_ssp[-1]",
    ]
    for pat in forbidden:
        assert pat not in src, (
            f"Found stale ``{pat}`` in orchestrator source. "
            f"Use ``time_weighted_sfr(sfr, lbt_grid, 1e7)`` instead."
        )


def test_numpy_array_input():
    """Sanity: helper accepts numpy arrays (auto-promotion to jnp)."""
    lbt = np.logspace(6, 10, 64)
    sfr = np.full_like(lbt, 5.0)
    out = float(time_weighted_sfr(jnp.asarray(sfr), jnp.asarray(lbt), 1e7))
    assert out == pytest.approx(5.0, rel=1e-6)
