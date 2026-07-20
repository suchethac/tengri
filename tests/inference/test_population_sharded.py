# SPDX-License-Identifier: BSD-3-Clause
"""Multi-device sharding for the hierarchical population VI fit (B1 of the GPU plan).

``Fitter(ForwardModel.build(population=...)).run("native_vi_linear", devices="all")``
shards the galaxy axis across devices so each device evaluates its own slice of
the population, and must return the same posterior as the single-device path.

The canonical population fit reaches the galaxy axis through ``jax.vmap``
(``forward/population_sed_model.py``), which GSPMD partitions from a sharding on
the data. That is *not* true of a ``lax.map`` galaxy loop, whose loop axis GSPMD
all-gathers — measured at exactly n_devices slower, correct results and zero
speedup. The subprocess check therefore asserts the work is genuinely divided,
not merely that the answer is right.

Device count is fixed at process start, so the real work runs in a fresh
subprocess with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``; the
heavy lifting is in ``_population_sharded_parity_check.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tengri.inference.sharding import resolve_devices, shard_leading_axis

pytestmark = pytest.mark.contract

_CHECK = Path(__file__).parent / "_population_sharded_parity_check.py"


def test_resolve_devices():
    assert resolve_devices(None) is None
    assert resolve_devices("all") == list(jax.devices())
    with pytest.raises(ValueError, match="None, 'all', or a device list"):
        resolve_devices("gpu0")
    some = list(jax.devices())[:1]
    assert resolve_devices(some) == some


def test_shard_leading_axis_is_a_noop_for_one_device():
    """A single device has nothing to shard; the tree must survive untouched."""
    tree = {"data": jnp.ones((4, 3)), "noise": jnp.full((4, 3), 2.0)}
    out = shard_leading_axis(tree, list(jax.devices())[:1])
    assert set(out) == set(tree)
    for k in tree:
        assert jnp.array_equal(out[k], tree[k])


def test_sharded_population_vi_matches_single_device():
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
        timeout=900,
    )

    if "SKIP_NO_DEVICES" in result.stdout:
        pytest.skip("XLA fake-device flag did not take on this platform")

    assert result.returncode == 0, (
        f"population shard check failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr[-3000:]}"
    )
    assert "PARITY_OK" in result.stdout, f"missing PARITY_OK; stdout:\n{result.stdout}"
