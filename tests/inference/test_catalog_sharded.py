# SPDX-License-Identifier: BSD-3-Clause
"""Multi-device parity for sharded catalog sampling (A2 of the GPU plan).

``CatalogFitter.run("mcmc_nuts", devices="all")`` shards the galaxy axis across
devices (GSPMD, no cross-device reduction) and must return the *same* per-galaxy
posteriors as the single-device path up to float round-off.

Since JAX's device count is fixed at process start, this spawns a fresh
subprocess with ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` so the
multi-device code path genuinely executes in CI — not silently skipped for want
of real GPUs. The heavy lifting is in ``_sharded_parity_check.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tengri.inference.catalog_fitter import CatalogFitter

pytestmark = pytest.mark.contract

_CHECK = Path(__file__).parent / "_sharded_parity_check.py"


def test_resolve_devices():
    assert CatalogFitter._resolve_devices(None) is None
    assert CatalogFitter._resolve_devices("all") == list(jax.devices())
    with pytest.raises(ValueError, match="None, 'all', or a device list"):
        CatalogFitter._resolve_devices("gpu0")
    some = list(jax.devices())[:1]
    assert CatalogFitter._resolve_devices(some) == some


def test_devices_ignored_warns_for_non_mcmc(monkeypatch):
    """devices= on a non-sampling method is a no-op and must warn."""
    galaxies = [{"flux_obs": jnp.ones(4), "noise": jnp.ones(4)} for _ in range(2)]
    cf = CatalogFitter(None, galaxies)

    monkeypatch.setattr(
        cf,
        "_run_sequential",
        lambda method, *, key, **kw: None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cf.run("map", key=jax.random.PRNGKey(0), devices="all")
    assert any("devices=" in str(w.message) for w in caught)


def test_sharded_catalog_sampling_matches_single_device():
    env = dict(os.environ)
    # Emulate 4 devices on CPU so the shard path runs without real GPUs.
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"

    result = subprocess.run(
        [sys.executable, str(_CHECK)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if "SKIP_NO_DEVICES" in result.stdout:
        pytest.skip("XLA fake-device flag did not take on this platform")

    assert result.returncode == 0, (
        f"parity check failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr[-3000:]}"
    )
    assert "PARITY_OK" in result.stdout, f"missing PARITY_OK; stdout:\n{result.stdout}"
