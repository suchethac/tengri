# SPDX-License-Identifier: BSD-3-Clause
"""Contract: BlackJAX version floor validation before first kernel import.

Issue #1999: shared venv silently rebuilt with unsupported blackjax version,
causing frozen chains with no error. Guard must validate environment at the
seam where the drift lives — tengri's own blackjax import.

The guard checks the source tree's pyproject.toml first (editable installs),
then falls back to dist metadata (wheel installs). This prevents staleness:
metadata can be stale on editable installs, but pyproject.toml in the working
tree is always current.
"""

from unittest.mock import patch

_SHARED = "tengri.inference.backends.mcmc._shared"

import pytest

pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def reset_blackjax_floor_memoization():
    """Reset the blackjax floor check memoization flag and kernel caches before each test."""
    import tengri.inference.backends.mcmc._shared as _shared

    _shared._blackjax_floor_checked = False
    # Clear the functools.cache on the kernel getters
    _shared._get_nuts_kernel.cache_clear()
    _shared._get_hmc_kernel.cache_clear()
    _shared._get_dynamic_hmc_kernel.cache_clear()
    _shared._get_ghmc_kernel.cache_clear()
    yield
    # Reset after test too, for cleanliness
    _shared._blackjax_floor_checked = False
    _shared._get_nuts_kernel.cache_clear()
    _shared._get_hmc_kernel.cache_clear()
    _shared._get_dynamic_hmc_kernel.cache_clear()
    _shared._get_ghmc_kernel.cache_clear()


def test_blackjax_floor_violation_raises():
    """Simulated floor violation raises with versioning details and pip remedy."""
    from tengri.config.exceptions import BackendError
    from tengri.inference.backends import mcmc

    source_patch = "tengri.inference.backends.mcmc._shared._get_blackjax_floor_from_source"
    meta_patch = "tengri.inference.backends.mcmc._shared._get_blackjax_floor_from_metadata"

    with patch(source_patch) as mock_source:
        mock_source.return_value = None  # No source available

        with patch(meta_patch) as mock_meta:
            mock_meta.return_value = "1.6"  # Floor from metadata

            with patch("importlib.metadata.version") as mock_version:
                mock_version.return_value = "1.3"  # Simulated old version

                with pytest.raises(BackendError) as exc_info:
                    mcmc._shared._get_nuts_kernel()

                err_msg = str(exc_info.value)
                # Verify message contains all required information
                assert "1.3" in err_msg, f"Installed version not in error: {err_msg}"
                assert "1.6" in err_msg, f"Floor not in error: {err_msg}"
                assert "pip install -U" in err_msg, f"Remedy not in error: {err_msg}"
                assert "frozen" in err_msg.lower(), f"Consequence not mentioned: {err_msg}"


def test_blackjax_floor_satisfied_no_raise():
    """Satisfied floor (real env) imports without raising."""
    from tengri.inference.backends import mcmc

    # Real environment has blackjax 1.6.2; should not raise
    kernel = mcmc._shared._get_nuts_kernel()
    assert kernel is not None


def test_blackjax_floor_unresolvable_skips():
    """Unresolvable floor (no source, no metadata) silently skips the check."""
    from tengri.inference.backends import mcmc

    source_patch = "tengri.inference.backends.mcmc._shared._get_blackjax_floor_from_source"
    meta_patch = "tengri.inference.backends.mcmc._shared._get_blackjax_floor_from_metadata"

    with patch(source_patch) as mock_source:
        mock_source.return_value = None

        with patch(meta_patch) as mock_meta:
            mock_meta.return_value = None  # No floor found

            # Should not raise, just skip silently
            kernel = mcmc._shared._get_nuts_kernel()
            assert kernel is not None


def test_blackjax_floor_metadata_staleness():
    """Stale metadata does not override current source declaration.

    Editable installs have stale metadata. Verify that when source declares
    blackjax>=1.6 but metadata (mistakenly) declares >=1.0, the source wins
    and the floor is 1.6 (not 1.0). Installed version 1.3 should RAISE.
    """
    from tengri.config.exceptions import BackendError
    from tengri.inference.backends import mcmc

    with patch(f"{_SHARED}._get_blackjax_floor_from_source") as mock_source:
        mock_source.return_value = "1.6"  # Source declares 1.6

        with patch(f"{_SHARED}._get_blackjax_floor_from_metadata") as mock_meta:
            mock_meta.return_value = "1.0"  # Stale metadata says 1.0

            with patch("importlib.metadata.version") as mock_version:
                mock_version.return_value = "1.3"  # Installed is 1.3

                # Should raise because 1.3 < 1.6 (source floor wins)
                with pytest.raises(BackendError) as exc_info:
                    mcmc._shared._get_nuts_kernel()

                err_msg = str(exc_info.value)
                assert "1.3" in err_msg
                assert "1.6" in err_msg
                # Verify it did NOT use the stale 1.0 floor
                assert not ("1.0" in err_msg and "1.6" not in err_msg)


def test_blackjax_floor_memoization():
    """Second call after check does not re-parse."""
    from tengri.inference.backends import mcmc

    with patch(f"{_SHARED}._get_blackjax_floor_from_source") as mock_source:
        mock_source.return_value = "1.6"

        with patch(f"{_SHARED}._get_blackjax_floor_from_metadata") as mock_meta:
            mock_meta.return_value = None  # Metadata will not be consulted

            with patch("importlib.metadata.version") as mock_version:
                mock_version.return_value = "1.6.2"

                # First call
                mcmc._shared._get_nuts_kernel()
                source_call_count_1 = mock_source.call_count

                # Second call — should use memoization
                mcmc._shared._get_nuts_kernel()
                source_call_count_2 = mock_source.call_count

                # Both functions should have been called only once
                assert source_call_count_1 == 1
                assert source_call_count_2 == 1, "Memoization failed; source was re-parsed"
