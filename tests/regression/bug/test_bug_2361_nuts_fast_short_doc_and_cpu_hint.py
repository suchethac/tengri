# SPDX-License-Identifier: BSD-3-Clause
"""
Regression tests for issue #2361: mcmc_nuts_fast's short_doc and CPU hint.

Issue: https://github.com/suchethacooray/tengri/issues/2361

mcmc_nuts_fast claims "pmapped chains by default" but vmaps chains on single-device
CPU hosts. The TENGRI_HOST_DEVICES hint emitted unconditionally on CPU is
counterproductive: forces pmapping on a platform where vmap is 8% faster
(451.5 s pmap vs 418.3 s vmap for a 600-iteration warmup, same seed).

The registry short_doc must state the device-conditional truth, and the hint
must be gated to only emit on multi-device platforms.

Measured on 14-core CPU (10 performance cores) with a 5-parameter broadband
photometry fit, 11 detected bands:
  - baseline (unset TENGRI_HOST_DEVICES, vmap): 418.3 s
  - hint applied (TENGRI_HOST_DEVICES=4, pmap):  451.5 s

Mechanism: CPU vmap on single device owns all 14 cores; pmap partitions one CPU
into 4 logical devices of ~3 cores each, starving the kernel. NUTS tree depth
and vmap/pmap performance are platform-dependent; the hint should not emit on CPU.
"""

import jax
import pytest

pytestmark = pytest.mark.regression_bug


def test_nuts_fast_short_doc_no_pmapped_claims():
    """Assert short_doc for mcmc_nuts_fast does not falsely claim pmap."""
    from tengri.inference._backend_registry import get_backend

    entry = get_backend("mcmc_nuts_fast")
    short_doc = entry.short_doc

    # Should NOT contain the false claim
    assert "pmapped chains by default" not in short_doc, (
        f"mcmc_nuts_fast short_doc contains false 'pmapped chains by default'; got: {short_doc}"
    )

    # Should state the device-conditional truth
    assert "vmapped chains on one device; pmapped when the platform exposes" in short_doc, (
        f"mcmc_nuts_fast short_doc does not state device-conditional rule; got: {short_doc}"
    )


def test_nuts_fast_cpu_hint_not_emitted():
    """Assert TENGRI_HOST_DEVICES hint predicate returns False on CPU platform."""
    # Ensure we're on CPU
    if jax.devices()[0].platform != "cpu":
        pytest.skip(f"Test requires CPU platform, got {jax.devices()[0].platform}")

    # Test the logic directly: _pmap_hint_applies() should return False on CPU.
    from tengri.inference.backends.mcmc import _shared

    assert not _shared._pmap_hint_applies(), (
        "On CPU platform, _pmap_hint_applies() should return False"
    )


def test_nuts_fast_non_cpu_hint_applies_with_predicate(monkeypatch):
    """Assert _pmap_hint_applies() logic gates the hint on platform."""
    # Monkeypatch jax.devices() to return a non-CPU platform
    from tengri.inference.backends.mcmc import _shared

    class FakeDevice:
        def __init__(self, platform_name):
            self.platform = platform_name

    def mock_devices():
        return [FakeDevice("gpu")]

    monkeypatch.setattr(jax, "devices", mock_devices)

    # With fake GPU platform, _pmap_hint_applies() should return True
    assert _shared._pmap_hint_applies(), (
        "On non-CPU platform, _pmap_hint_applies() should return True"
    )


def test_pmap_error_message_cpu_platform():
    """Assert pmap error message does not mention TENGRI_HOST_DEVICES on CPU."""
    if jax.devices()[0].platform != "cpu":
        pytest.skip(f"Test requires CPU platform, got {jax.devices()[0].platform}")

    from tengri.inference.backends.mcmc._shared import _resolve_chain_parallel

    try:
        _resolve_chain_parallel("pmap", n_chains=4)
        pytest.fail("Should raise ValueError for pmap with insufficient devices")
    except ValueError as e:
        error_msg = str(e)
        assert "TENGRI_HOST_DEVICES" not in error_msg, (
            f"Error message should not mention TENGRI_HOST_DEVICES on CPU; got: {error_msg}"
        )
        assert "vmap is faster than pmap" in error_msg, (
            f"Error message should mention vmap is faster on CPU; got: {error_msg}"
        )


def test_pmap_error_message_gpu_platform(monkeypatch):
    """Assert pmap error message mentions TENGRI_HOST_DEVICES on non-CPU."""

    class FakeDevice:
        platform = "gpu"

    # Monkeypatch to simulate GPU platform
    monkeypatch.setattr(jax, "devices", lambda: [FakeDevice()])
    monkeypatch.setattr(jax, "device_count", lambda: 1)

    from tengri.inference.backends.mcmc._shared import _resolve_chain_parallel

    try:
        _resolve_chain_parallel("pmap", n_chains=4)
        pytest.fail("Should raise ValueError for pmap with insufficient devices")
    except ValueError as e:
        error_msg = str(e)
        assert "TENGRI_HOST_DEVICES" in error_msg, (
            f"Error message should mention TENGRI_HOST_DEVICES on GPU; got: {error_msg}"
        )
        assert "vmap is faster than pmap" not in error_msg, (
            f"Error message should not mention vmap speed comparison on GPU; got: {error_msg}"
        )
