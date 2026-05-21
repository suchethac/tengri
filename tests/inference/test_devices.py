"""Tests for tengri.utils.devices — JAX device/resource management."""

import os
import warnings

import jax
import pytest

from tengri.utils.devices import (
    check_resources,
    device_info,
    get_n_parallel_chains,
    print_device_info,
    setup_jax,
)

pytestmark = pytest.mark.contract


class TestSetupJax:
    def test_enable_x64_sets_env_var(self):
        """setup_jax(enable_x64=True) sets JAX_ENABLE_X64 in environment."""
        setup_jax(enable_x64=True)
        assert os.environ.get("JAX_ENABLE_X64") == "True"

    def test_platform_sets_env_var(self):
        """setup_jax(platform='cpu') writes JAX_PLATFORMS."""
        setup_jax(platform="cpu")
        assert os.environ.get("JAX_PLATFORMS") == "cpu"

    def test_no_preallocate_sets_env_var(self):
        """preallocate_gpu=False (default) sets XLA_PYTHON_CLIENT_PREALLOCATE=false."""
        setup_jax(preallocate_gpu=False)
        assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"

    def test_gpu_memory_fraction_sets_env_var(self):
        """gpu_memory_fraction sets XLA_PYTHON_CLIENT_MEM_FRACTION."""
        setup_jax(gpu_memory_fraction=0.75)
        assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "0.75"

    def test_returns_none(self):
        """setup_jax() returns None (side-effects only)."""
        result = setup_jax()
        assert result is None

    def test_x64_config_applied(self):
        """After setup_jax(enable_x64=True), jax.config.x64_enabled is True."""
        setup_jax(enable_x64=True)
        assert jax.config.x64_enabled is True


class TestDeviceInfo:
    def test_returns_dict(self):
        """device_info() returns a dict."""
        info = device_info()
        assert isinstance(info, dict)

    def test_required_keys_present(self):
        """All expected keys are present in the returned dict."""
        info = device_info()
        required = {
            "platform",
            "devices",
            "n_devices",
            "default_device",
            "x64_enabled",
            "gpu_memory_mb",
        }
        assert required <= set(info.keys())

    def test_platform_is_string(self):
        info = device_info()
        assert isinstance(info["platform"], str)

    def test_n_devices_positive_int(self):
        info = device_info()
        assert isinstance(info["n_devices"], int)
        assert info["n_devices"] >= 1

    def test_devices_is_list_of_strings(self):
        info = device_info()
        assert isinstance(info["devices"], list)
        assert all(isinstance(d, str) for d in info["devices"])

    def test_x64_enabled_is_bool(self):
        info = device_info()
        assert isinstance(info["x64_enabled"], bool)

    def test_gpu_memory_none_on_cpu(self):
        """On CPU, gpu_memory_mb is None."""
        info = device_info()
        if info["platform"] == "cpu":
            assert info["gpu_memory_mb"] is None

    def test_default_device_is_string(self):
        info = device_info()
        assert isinstance(info["default_device"], str)


class TestPrintDeviceInfo:
    def test_prints_platform(self, capsys):
        """print_device_info() outputs the platform."""
        print_device_info()
        out = capsys.readouterr().out
        assert "platform" in out.lower() or "JAX" in out

    def test_prints_jax_version(self, capsys):
        """print_device_info() outputs the JAX version."""
        print_device_info()
        out = capsys.readouterr().out
        assert jax.__version__ in out

    def test_prints_x64_status(self, capsys):
        """print_device_info() mentions 64-bit status."""
        print_device_info()
        out = capsys.readouterr().out
        assert "64" in out


class TestGetNParallelChains:
    def test_returns_positive_int(self):
        """get_n_parallel_chains() returns a positive integer."""
        n = get_n_parallel_chains()
        assert isinstance(n, int)
        assert n >= 1

    def test_memory_per_chain_scales_result_on_gpu(self):
        """On GPU, smaller memory_per_chain_mb gives more chains."""
        info = device_info()
        if info["platform"] != "gpu":
            pytest.skip("GPU-only scaling test")
        n_small = get_n_parallel_chains(memory_per_chain_mb=10.0)
        n_large = get_n_parallel_chains(memory_per_chain_mb=100.0)
        assert n_small >= n_large

    def test_cpu_returns_core_count(self):
        """On CPU, result matches os.cpu_count()."""
        info = device_info()
        if info["platform"] != "cpu":
            pytest.skip("CPU-only test")
        n = get_n_parallel_chains()
        expected = os.cpu_count() or 4
        assert n == expected


class TestCheckResources:
    def test_runs_without_error(self, capsys):
        """check_resources() completes without raising."""
        check_resources()

    def test_prints_compute_ok(self, capsys):
        """check_resources() confirms compute test passes."""
        check_resources()
        out = capsys.readouterr().out
        assert "OK" in out

    def test_prints_parallel_chains(self, capsys):
        """check_resources() reports recommended parallel chains."""
        check_resources()
        out = capsys.readouterr().out
        assert "chain" in out.lower()

    def test_warns_x64_disabled(self):
        """check_resources() warns when x64 is disabled."""
        # Temporarily disable x64
        jax.config.update("jax_enable_x64", False)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_resources()
            categories = [warning.category for warning in w]
            assert UserWarning in categories
        finally:
            jax.config.update("jax_enable_x64", True)
