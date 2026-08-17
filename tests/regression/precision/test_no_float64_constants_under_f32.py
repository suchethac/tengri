# SPDX-License-Identifier: BSD-3-Clause
"""No module-scope ``jnp`` constant may be float64 in a float32 process (#1880).

Five DSPS modules flip ``jax_enable_x64`` on at import, transitively and early.
Every module-scope ``jnp`` array evaluated after that point was therefore
allocated as float64 — measured at **70 of 70** tree-wide. #1849's
``_reassert_x64_preference`` restores the *flag* at the end of the import but
cannot un-allocate an array.

On CPU that costs only a doubled footprint: with x64 off, JAX's promotion caps
results at float32 regardless of a float64 operand, so the numbers are
unaffected. On a backend with **no** float64 the allocation itself raises
(``MLX does not support float64``) and ``import tengri`` fails outright —
measured on Apple MPS via ``jax-mps``, where it made the package unimportable.

``tengri/__init__`` now holds x64 off for the duration of its own import when
the user has asked for float32.

These run in subprocesses because the behavior under test is import-time and
process-global: nothing in-process can observe it after the fact.
"""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.regression_bug

_CENSUS = """
    import warnings
    warnings.simplefilter("ignore")
    import importlib, pkgutil
    import jax, numpy as np
    import tengri

    bad = 0
    total = 0
    for mod in pkgutil.walk_packages(tengri.__path__, prefix="tengri."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue
        for attr in dir(m):
            if attr.startswith("__"):
                continue
            try:
                v = getattr(m, attr)
            except Exception:
                continue
            if isinstance(v, jax.Array) and np.issubdtype(v.dtype, np.floating):
                total += 1
                if v.dtype == np.float64:
                    bad += 1
    print("TOTAL", total)
    print("BAD", bad)
    print("X64", jax.config.jax_enable_x64)
"""


def _run(env_value, snippet):
    import os

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    if env_value is None:
        env.pop("JAX_ENABLE_X64", None)
    else:
        env["JAX_ENABLE_X64"] = env_value
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def _parse(out, key):
    for line in out.splitlines():
        if line.startswith(key + " "):
            return line.split(" ", 1)[1].strip()
    raise AssertionError(f"{key} not in output:\n{out}")


def test_no_module_scope_constant_is_float64_when_float32_requested():
    """The property, not a list of sites: zero float64 constants under f32."""
    out = _run("0", _CENSUS)
    assert _parse(out, "X64") == "False", f"arm is not float32:\n{out}"
    total = int(_parse(out, "TOTAL"))
    bad = int(_parse(out, "BAD"))
    # Non-vacuity: if the census stops finding constants at all, it stops
    # detecting the regression it exists for.
    assert total > 20, f"census found only {total} constants — it has gone blind"
    assert bad == 0, f"{bad} of {total} module-scope constants are float64 under float32"


def test_float64_default_still_allocates_float64_constants():
    """The default must be untouched: env unset means x64 on, constants float64."""
    out = _run(None, _CENSUS)
    assert _parse(out, "X64") == "True", f"float64 default was lost:\n{out}"
    total = int(_parse(out, "TOTAL"))
    bad = int(_parse(out, "BAD"))
    assert total > 20, f"census found only {total} constants"
    assert bad == total, (
        "under the float64 default every constant should be float64; "
        f"got {bad} of {total} — the guard is firing when it must not"
    )


def test_jax_config_update_is_restored_after_import():
    """The guard must not outlive the import — user calls must reach JAX."""
    out = _run(
        "0",
        """
        import warnings
        warnings.simplefilter("ignore")
        import jax
        import tengri  # noqa: F401
        print("PATCHED", hasattr(jax.config.update, "_tengri_original"))
        jax.config.update("jax_enable_x64", True)
        print("USERCALL", jax.config.jax_enable_x64)
        """,
    )
    assert _parse(out, "PATCHED") == "False", "jax.config.update left monkeypatched"
    assert _parse(out, "USERCALL") == "True", (
        "a user's own jax.config.update was swallowed after import — the guard "
        "must only cover tengri's import, not the rest of the process"
    )
