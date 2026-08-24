# SPDX-License-Identifier: BSD-3-Clause
"""JAX device and resource management.

Handles GPU/CPU/TPU detection, memory reporting, and platform-aware
configuration. Ensures tengri uses all available resources correctly
whether running on a MacBook, a GPU workstation, or an HPC cluster.

Usage:
    from tengri.utils.devices import setup_jax, device_info
    setup_jax()          # auto-configure for current hardware
    print(device_info()) # see what's available
"""

import os
import warnings


def setup_jax(
    platform: str | None = None,
    enable_x64: bool = True,
    gpu_memory_fraction: float | None = None,
    preallocate_gpu: bool = False,
):
    """Configure JAX for the current platform. Call once at startup.

    Parameters
    ----------
    platform: str, optional
        Force platform: "cpu", "gpu", "tpu". If None, auto-detect.
    enable_x64: bool
        Enable 64-bit precision (recommended for SED fitting).
    gpu_memory_fraction: float, optional
        Fraction of GPU memory to use (0-1). If None, use JAX default.
        Set to e.g. 0.8 to leave room for other processes.
    preallocate_gpu: bool
        If True, pre-allocate all GPU memory at startup (faster but greedy).
        If False (default), allocate on demand (better for shared GPUs).

    Returns
    -------
    None

    Examples
    --------
    # MacBook (CPU only)
    >>> setup_jax()

    # GPU workstation, use 80% of GPU memory
    >>> setup_jax(gpu_memory_fraction=0.8)

    # Force CPU even when GPU is available (for debugging)
    >>> setup_jax(platform="cpu")

    # HPC cluster with pre-allocation
    >>> setup_jax(preallocate_gpu=True)
    """
    # 64-bit precision. Only ``jax.config.update`` below has any effect once
    # JAX is imported; writing JAX_ENABLE_X64 here used to leave a variable in
    # the environment that reads as authoritative and is not (#1840). Keep the
    # environment truthful instead, so a later reader -- or a subprocess that
    # inherits it -- sees the precision actually in force.
    os.environ["JAX_ENABLE_X64"] = "True" if enable_x64 else "0"

    # Platform selection
    if platform is not None:
        os.environ["JAX_PLATFORMS"] = platform

    # GPU memory management
    if not preallocate_gpu:
        # Default: allocate on demand, better for shared machines
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if gpu_memory_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_memory_fraction)

    # Now import JAX (env vars must be set before import)
    import jax

    jax.config.update("jax_enable_x64", enable_x64)


def device_info() -> dict:
    """Report available JAX devices and their properties.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Dictionary with keys: "platform" (str), "devices" (list), "n_devices" (int),
        "default_device" (str), "x64_enabled" (bool), "gpu_memory_mb" (float or None).
    """
    import jax

    devices = jax.devices()
    default = jax.devices()[0]
    platform = default.platform

    info = {
        "platform": platform,
        "devices": [str(d) for d in devices],
        "n_devices": len(devices),
        "default_device": str(default),
        "x64_enabled": jax.config.x64_enabled,
        "gpu_memory_mb": None,
    }

    # Try to get GPU memory info
    if platform == "gpu":
        try:
            for d in devices:
                mem = d.memory_stats()
                if mem:
                    info["gpu_memory_mb"] = mem.get("bytes_limit", 0) / 1e6
                    break
        except (AttributeError, KeyError, TypeError):
            # AttributeError: memory_stats() not available on this backend
            # KeyError: bytes_limit missing from memory stats dict
            # TypeError: memory_stats() returned non-dict
            pass

    return info


def print_device_info():
    """Print a human-readable summary of JAX configuration.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    info = device_info()
    print(f"JAX platform:    {info['platform'].upper()}")
    print(f"Devices:         {info['n_devices']}x {info['devices'][0]}")
    print(f"64-bit:          {'Yes' if info['x64_enabled'] else 'No'}")
    if info["gpu_memory_mb"]:
        print(f"GPU memory:      {info['gpu_memory_mb']:.0f} MB")

    import jax

    print(f"JAX version:     {jax.__version__}")


def get_n_parallel_chains(memory_per_chain_mb: float = 50.0) -> int:
    """Estimate how many HMC/NUTS chains can run in parallel.

    For GPU: limited by GPU memory.
    For CPU: limited by number of cores.

    Following Zacharegkas+2025: fit as many chains as GPU memory allows,
    since GPU HMC scales at near-zero cost up to memory saturation.

    Parameters
    ----------
    memory_per_chain_mb: float
        Estimated memory per chain (MB). Depends on model complexity.
        Default 50 MB is conservative for a 12-param photometry model.

    Returns
    -------
    int
        Recommended number of parallel chains.
    """
    import jax

    platform = jax.devices()[0].platform

    if platform == "gpu":
        try:
            mem = jax.devices()[0].memory_stats()
            total_mb = mem.get("bytes_limit", 0) / 1e6
            # Use 80% of available memory
            n = int(0.8 * total_mb / memory_per_chain_mb)
            return max(n, 1)
        except (AttributeError, KeyError, TypeError, ZeroDivisionError):
            # AttributeError: memory_stats() not available
            # KeyError: bytes_limit missing
            # TypeError: memory_stats() returned non-dict
            # ZeroDivisionError: memory_per_chain_mb is 0
            return 100  # conservative default for unknown GPU

    elif platform == "cpu":
        n_cores = os.cpu_count() or 4
        return n_cores

    else:  # TPU
        return len(jax.devices()) * 8  # TPU cores


def check_resources():
    """Run a quick diagnostic to verify JAX can use available hardware.

    Prints warnings if resources are underutilized.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    import jax
    import jax.numpy as jnp

    info = device_info()
    print_device_info()

    # Quick computation test
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    _ = jnp.dot(x, x.T).block_until_ready()
    print("Compute test:    OK")

    # Warnings
    if info["platform"] == "cpu":
        try:
            import subprocess

            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                warnings.warn(
                    "GPU detected but JAX is using CPU. "
                    "Install jaxlib with CUDA support: "
                    "pip install jax[cuda12]",
                    UserWarning,
                    stacklevel=2,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # No GPU, CPU is expected

    if not info["x64_enabled"]:
        warnings.warn(
            "64-bit precision disabled. SED fitting benefits from x64. "
            "Call setup_jax(enable_x64=True), or set JAX_ENABLE_X64=True "
            "before importing tengri, or jax.config.update("
            "'jax_enable_x64', True) after it. (Setting the environment "
            "variable after import has no effect -- #1840.)",
            UserWarning,
            stacklevel=2,
        )

    print("---")
    n_chains = get_n_parallel_chains()
    print(f"Recommended parallel chains: {n_chains}")
