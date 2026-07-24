# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1177: MCLMC backends with blackjax >= 1.5.

Blackjax >= 1.5 introduced breaking API changes:
1. build_kernel no longer takes logdensity_fn or inverse_mass_matrix parameters
2. The kernel contract changed: no longer a factory, but a step function
3. mclmc_find_L_and_step_size takes logdensity_fn as a parameter

This test verifies that the MCLMC module imports and the blackjax adaptation
functions are callable in the expected way.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.regression


def test_mclmc_module_imports():
    """Smoke test: MCLMC module imports without errors."""
    try:
        import blackjax
        bx_version = tuple(int(x) for x in blackjax.__version__.split(".")[:2])
        if bx_version < (1, 5):
            pytest.skip(f"blackjax {blackjax.__version__} < 1.5; MCLMC not compatible")
    except ImportError:
        pytest.skip("blackjax not installed")

    # Import the MCLMC backends module
    from tengri.inference.backends.mcmc import mclmc

    # Verify the functions exist
    assert hasattr(mclmc, "run_mclmc")
    assert hasattr(mclmc, "run_adjusted_mclmc")
    assert callable(mclmc.run_mclmc)
    assert callable(mclmc.run_adjusted_mclmc)


def test_blackjax_adaptation_api_signature():
    """Verify that blackjax >= 1.5 adaptation functions have expected signatures."""
    try:
        import blackjax
        bx_version = tuple(int(x) for x in blackjax.__version__.split(".")[:2])
        if bx_version < (1, 5):
            pytest.skip(f"blackjax {blackjax.__version__} < 1.5")
    except ImportError:
        pytest.skip("blackjax not installed")

    import inspect

    # Verify that mclmc_find_L_and_step_size accepts logdensity_fn as a parameter
    sig = inspect.signature(blackjax.mclmc_find_L_and_step_size)
    assert "logdensity_fn" in sig.parameters, (
        f"mclmc_find_L_and_step_size missing logdensity_fn parameter; "
        f"signature: {sig}"
    )

    # Verify that adjusted_mclmc_find_L_and_step_size accepts logdensity_fn
    sig = inspect.signature(blackjax.adjusted_mclmc_find_L_and_step_size)
    assert "logdensity_fn" in sig.parameters, (
        f"adjusted_mclmc_find_L_and_step_size missing logdensity_fn parameter; "
        f"signature: {sig}"
    )


def test_blackjax_build_kernel_signature():
    """Verify that build_kernel doesn't take logdensity_fn or inverse_mass_matrix."""
    try:
        import blackjax
        bx_version = tuple(int(x) for x in blackjax.__version__.split(".")[:2])
        if bx_version < (1, 5):
            pytest.skip(f"blackjax {blackjax.__version__} < 1.5")
    except ImportError:
        pytest.skip("blackjax not installed")

    import inspect

    # Verify that mclmc.build_kernel does NOT take logdensity_fn
    sig = inspect.signature(blackjax.mcmc.mclmc.build_kernel)
    assert "logdensity_fn" not in sig.parameters, (
        f"mclmc.build_kernel should NOT have logdensity_fn parameter in >= 1.5; "
        f"signature: {sig}"
    )

    # Verify that adjusted_mclmc.build_kernel does NOT take logdensity_fn
    sig = inspect.signature(blackjax.mcmc.adjusted_mclmc.build_kernel)
    assert "logdensity_fn" not in sig.parameters, (
        f"adjusted_mclmc.build_kernel should NOT have logdensity_fn parameter in >= 1.5; "
        f"signature: {sig}"
    )
