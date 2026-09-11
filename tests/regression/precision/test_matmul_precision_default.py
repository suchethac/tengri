# SPDX-License-Identifier: BSD-3-Clause
"""``import tengri`` raises matmul precision when x64 is off (#2022).

On Ampere and later, XLA lowers float32 matmuls to TF32 (10-bit mantissa
against float32's 24) unless ``jax_default_matmul_precision`` is set to
``"highest"``. tengri's own float32 numerics are calibrated on the full
float32 path: measured 4.5% error on Fisher-matrix parameter error bars.
Since ``import tengri`` already knows when x64 ends up off (#1840), it raises
matmul precision itself at that moment rather than leaving every float32
session to rediscover the hazard on a GPU.

These run in subprocesses for the same reason as
``test_x64_env_override.py``: the behavior under test is import-time and
process-global.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import tengri

pytestmark = pytest.mark.regression_bug

#: This process's own tengri, already checked by ``tests/conftest.py``'s
#: ``_check_tengri_source_tree_match`` at collection time (#2170). Every probe
#: below asserts its subprocess imported the identical file, not whatever a
#: stale editable-install ``.pth`` in the shared venv happens to point at --
#: a mismatch there would make these tests pass for the wrong tree entirely.
_TENGRI_FILE = str(Path(tengri.__file__).resolve())


def _probe(env, snippet):
    """Run `snippet` in a fresh interpreter with the given environment."""
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stdout}\n{r.stderr}"
    lines = r.stdout.splitlines()
    raw_file = next((ln.split(" ", 1)[1] for ln in lines if ln.startswith("TENGRI_FILE ")), None)
    got_file = str(Path(raw_file).resolve()) if raw_file is not None else None
    assert got_file == _TENGRI_FILE, (
        f"subprocess imported tengri from {got_file!r}, not this tree's "
        f"{_TENGRI_FILE!r} -- PYTHONPATH did not pin it to the worktree under "
        f"test (#2170), and every other assertion in this probe is about the "
        f"wrong source"
    )
    return r


def _env(**overrides):
    import os

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    # The subprocess does not inherit pytest's ``pythonpath = ["src"]`` ini
    # setting (that only patches ``sys.path`` in this process), so without
    # this it would pick up whatever tengri a stale editable install points
    # at instead of the tree under test (#2170).
    env["PYTHONPATH"] = str(Path(_TENGRI_FILE).parents[1])
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


_REPORT = """
    import warnings
    warnings.simplefilter("ignore")
    import jax
    import tengri
    print("TENGRI_FILE", tengri.__file__)
    print("X64", jax.config.jax_enable_x64)
    print("MATMUL", jax.config.jax_default_matmul_precision)
"""


def test_matmul_precision_defaults_to_highest_when_x64_is_off():
    """x64 off, env unset -- tengri raises matmul precision itself."""
    env = _env(JAX_ENABLE_X64="0", JAX_DEFAULT_MATMUL_PRECISION=None)
    out = _probe(env, _REPORT).stdout
    assert "X64 False" in out, f"x64 was not honored as off:\n{out}"
    assert "MATMUL highest" in out, f"matmul precision was not raised:\n{out}"


def test_explicit_matmul_precision_env_wins():
    """x64 off, env set to something else -- the user's choice survives."""
    env = _env(JAX_ENABLE_X64="0", JAX_DEFAULT_MATMUL_PRECISION="default")
    out = _probe(env, _REPORT).stdout
    assert "X64 False" in out, f"x64 was not honored as off:\n{out}"
    assert "MATMUL default" in out, f"tengri overrode the user's explicit setting:\n{out}"


def test_matmul_precision_untouched_when_x64_is_on():
    """x64 on (the default) -- tengri leaves matmul precision alone."""
    env = _env(JAX_ENABLE_X64=None, JAX_DEFAULT_MATMUL_PRECISION=None)
    out = _probe(env, _REPORT).stdout
    assert "X64 True" in out, f"x64 default was lost:\n{out}"
    assert "MATMUL None" in out, f"tengri set matmul precision even though x64 stayed on:\n{out}"


def test_setup_jax_raises_matmul_precision_when_x64_is_disabled():
    """``setup_jax(enable_x64=False)`` mirrors the import-time default."""
    env = _env(JAX_ENABLE_X64=None, JAX_DEFAULT_MATMUL_PRECISION=None)
    out = _probe(
        env,
        """
        import warnings
        warnings.simplefilter("ignore")
        import jax
        import tengri
        print("TENGRI_FILE", tengri.__file__)
        from tengri.utils.devices import setup_jax
        setup_jax(enable_x64=False, platform="cpu")
        print("X64", jax.config.jax_enable_x64)
        print("MATMUL", jax.config.jax_default_matmul_precision)
        """,
    ).stdout
    assert "X64 False" in out, f"setup_jax did not disable x64:\n{out}"
    assert "MATMUL highest" in out, f"setup_jax did not raise matmul precision:\n{out}"


def test_setup_jax_respects_explicit_matmul_precision_env():
    """``setup_jax(enable_x64=False)`` still lets an explicit env value win."""
    env = _env(JAX_ENABLE_X64=None, JAX_DEFAULT_MATMUL_PRECISION="default")
    out = _probe(
        env,
        """
        import warnings
        warnings.simplefilter("ignore")
        import jax
        import tengri
        print("TENGRI_FILE", tengri.__file__)
        from tengri.utils.devices import setup_jax
        setup_jax(enable_x64=False, platform="cpu")
        print("MATMUL", jax.config.jax_default_matmul_precision)
        """,
    ).stdout
    assert "MATMUL default" in out, f"setup_jax overrode the user's explicit setting:\n{out}"
