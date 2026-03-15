"""Shared test fixtures for diffsed test suite."""

import jax
import pytest

from diffsed.models.sfh.gp_sfh import compute_sqrt_power_drw
from diffsed.utils.grid import grid_spacing, make_log_age_grid

# Enable 64-bit for numerical precision in tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def rng_key():
    """Default PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def log_age_grid():
    """Standard 256-point log-age grid."""
    return make_log_age_grid(256)


@pytest.fixture
def d_log_age(log_age_grid):
    """Grid spacing."""
    return grid_spacing(log_age_grid)


@pytest.fixture
def drw_params_moderate():
    """Moderate burstiness DRW parameters."""
    return {"sigma_ps": 1.0, "tau_ps": 50e6}  # 50 Myr


@pytest.fixture
def drw_params_smooth():
    """Smooth (low burstiness) DRW parameters."""
    return {"sigma_ps": 0.5, "tau_ps": 200e6}  # 200 Myr


@pytest.fixture
def drw_params_bursty():
    """Highly bursty DRW parameters."""
    return {"sigma_ps": 3.0, "tau_ps": 5e6}  # 5 Myr


@pytest.fixture
def sqrt_power_moderate(d_log_age, drw_params_moderate):
    """Pre-computed amplitude operator for moderate regime."""
    return compute_sqrt_power_drw(
        256,
        float(d_log_age),
        drw_params_moderate["sigma_ps"],
        drw_params_moderate["tau_ps"],
    )
