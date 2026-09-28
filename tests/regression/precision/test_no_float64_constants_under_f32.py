# SPDX-License-Identifier: BSD-3-Clause
"""No module-scope constant may be a device array (#1880, #2271).

A module-scope ``jnp`` array is a device buffer whose dtype is fixed by the x64
state at *import* time, and both states were defective:

* **x64 off** (#1880): five DSPS modules flip ``jax_enable_x64`` on at import,
  transitively and early, so every constant evaluated after that point was
  float64 — measured at **70 of 70** tree-wide. ``tengri/__init__`` now holds
  x64 off for the duration of its own import when the user has asked for
  float32. But the constants it then allocates are float32 for the life of the
  process, and a float64 reference computed later (the whole pytest suite,
  which forces x64 back on) is silently contaminated by them: measured as 16
  ``rtol=1e-12`` failures in two files from ``JAX_ENABLE_X64=0`` alone.
* **x64 on** (#2271, the default and the suite's state): every constant is a
  float64 device buffer, and a backend with no float64 (jax-mps / MLX) refuses
  the first one — measured on Apple MPS, where ``import tengri`` died at
  ``observation/eline_catalog.py`` while loading ``tests/conftest.py``.

The property is dtype-independent: a module-scope constant must not be a
device array of any dtype. A host-side numpy constant is canonicalized to the
working dtype at the op, so its values follow the x64 state at use time.

These run in subprocesses because the behavior under test is import-time and
process-global: nothing in-process can observe it after the fact.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression_bug

# Walks every tengri module AND the precision test tree (three test files had
# the same defect at collection time), reports import failures so a module that
# cannot be measured is visible rather than silently narrowing the census, and
# self-checks the detector on an array it knows is a device array.
_DEVICE_CENSUS = """
    import warnings
    warnings.simplefilter("ignore")
    import importlib, pkgutil, pathlib, sys
    import jax, jax.numpy as jnp, numpy as np
    import tengri

    def walk(pkg, prefix):
        for mod in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
            yield mod.name

    names = list(walk(tengri, "tengri."))
    tests_root = pathlib.Path("tests") / "regression" / "precision"
    if tests_root.is_dir():
        sys.path.insert(0, ".")
        names += [
            "tests.regression.precision." + p.stem for p in sorted(tests_root.glob("test_*.py"))
        ]

    hits, plain, failed, modules = [], [], [], 0
    for name in names:
        try:
            m = importlib.import_module(name)
        except Exception as e:
            failed.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
            continue
        modules += 1
        for attr, v in vars(m).items():
            if isinstance(v, jax.Array):
                hits.append(f"{name}.{attr} {v.dtype} {v.shape}")
            elif (
                name.startswith("tengri.")
                and isinstance(v, np.ndarray)
                and np.issubdtype(v.dtype, np.floating)
                and v.ndim > 0
                and (
                    type(v) is not np.ndarray
                    or not v.flags["C_CONTIGUOUS"]
                    or min(v.strides) < 0
                )
            ):
                kind = type(v).__name__
                plain.append(f"{name}.{attr} {kind} {v.dtype} {v.shape} strides={v.strides}")
    probe = jnp.zeros(3)
    print("SELFCHECK", isinstance(probe, jax.Array))
    print("MODULES", modules)
    print("FAILED", len(failed))
    for f in failed:
        print("FAIL", f)
    print("ARRAYS", len(hits))
    for h in hits:
        print("HIT", h)
    print("PLAIN", len(plain))
    for p in plain:
        print("PLAINHIT", p)
    print("X64", jax.config.jax_enable_x64)
"""


_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"

# Every snippet ends by naming the tengri it imported. The subprocess does not
# inherit pytest's ``pythonpath = ["src"]``, so without PYTHONPATH it imports
# whatever tengri the interpreter's site-packages resolve to, which in an
# editable install can be a different checkout than the one under test: the
# guard then measures a tree it was not asked about and reports it as this one.
_WHICH = """
    import tengri as _t
    print("TENGRI_FILE", _t.__file__)
"""


def _run(env_value, snippet):
    import os

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_SRC), env.get("PYTHONPATH")) if p)
    if env_value is None:
        env.pop("JAX_ENABLE_X64", None)
    else:
        env["JAX_ENABLE_X64"] = env_value
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet) + textwrap.dedent(_WHICH)],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
        timeout=900,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stdout}\n{r.stderr}"
    imported = Path(_parse(r.stdout, "TENGRI_FILE")).resolve()
    assert imported.is_relative_to(_SRC), (
        f"the probe imported tengri from {imported}, not from the tree under test ({_SRC})"
    )
    return r.stdout


def _parse(out, key):
    for line in out.splitlines():
        if line.startswith(key + " "):
            return line.split(" ", 1)[1].strip()
    raise AssertionError(f"{key} not in output:\n{out}")


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


def _device_census(env_value):
    out = _run(env_value, _DEVICE_CENSUS)
    assert _parse(out, "SELFCHECK") == "True", f"the detector does not see a jax.Array:\n{out}"
    modules = int(_parse(out, "MODULES"))
    failed = [ln for ln in out.splitlines() if ln.startswith("FAIL ")]
    hits = [ln[4:] for ln in out.splitlines() if ln.startswith("HIT ")]
    # Non-vacuity: a census that imported almost nothing measured almost nothing.
    assert modules > 100, f"census imported only {modules} modules — it has gone blind:\n{out}"
    assert not failed, (
        "modules the census could not import (unmeasured, not clean):\n" + "\n".join(failed)
    )
    return out, hits


@pytest.mark.parametrize(
    ("env_value", "x64"),
    [(None, "True"), ("0", "False")],
    ids=["float64_default", "float32_requested"],
)
def test_import_allocates_no_device_array_at_module_scope(env_value, x64):
    """#2271 (x64 on) and #1880 (x64 off): zero module-scope jax.Array attributes.

    Measured before the fix with x64 on: 74 device buffers across 25 modules
    (71 float64, one int64 index, re-exports included) plus four in this test
    tree, and ``import tengri`` dying at the first of them on jax-mps. With x64
    off the same constants were float32 and then contaminated every float64
    reference computed later in the process. A host-side numpy constant is
    canonicalized to the working dtype at the op, so the values follow the x64
    state at use time instead of at import time, under either flag.

    The flag assertion keeps #1880's two intents: a float32 request is honored,
    and the float64 default is untouched.
    """
    out, hits = _device_census(env_value)
    assert _parse(out, "X64") == x64, f"x64 state is not the one requested:\n{out}"
    assert not hits, (
        f"{len(hits)} module-scope device arrays materialized by import:\n" + "\n".join(hits)
    )


def test_every_module_scope_float_table_is_a_plain_contiguous_host_array():
    """The host half of the contract: a plain ``np.ndarray`` (no subclass; JAX
    refuses ``__jax_array__`` for jit arguments, so a wrapper buys nothing) that
    is C-contiguous with positive strides. jax-mps refuses a negative host
    stride at ``device_put``: the flipped SKIRTOR cubes failed every torus seam
    of the #1206 parity sweep with a misleading "GPU memory may be exhausted".
    The device half (``device_table`` at every use) is enforced statically by
    ``tools/check_host_tables.py``.
    """
    out, _ = _device_census(None)
    plain = [ln[9:] for ln in out.splitlines() if ln.startswith("PLAINHIT ")]
    assert not plain, (
        f"{len(plain)} module-scope float tables are not plain contiguous numpy:\n"
        + "\n".join(plain)
    )


def test_first_lazy_dsps_use_keeps_x64_off():
    """Lazy dsps imports at first use must respect the caller's x64_preference (#2504).

    Commit #2276 made DSPS imports lazy (function-local, inside __getattr__ or
    function bodies). Before the fix, the first lazy import still flipped x64 on
    unchecked, so ``JAX_ENABLE_X64=0`` would be honored at import, then silently
    clobbered at first use of any lazy DSPS path (cosmology distance functions,
    PLANCK15/WMAP5 attribute access). The result was silent dtype inflation for
    the rest of the process.

    This test runs in a subprocess with x64=off, touches the first lazy DSPS use,
    and asserts that x64 is still off and dtypes are still float32 afterwards.
    """
    out = _run(
        "0",
        """
        import warnings
        warnings.simplefilter("ignore")
        import jax
        import jax.numpy as jnp
        import tengri

        # First lazy dsps import: access PLANCK15
        _ = tengri.cosmology.PLANCK15
        print("AFTER_PLANCK15", jax.config.jax_enable_x64)
        print("DTYPE_AFTER_PLANCK15", (jnp.zeros(1) + 1.0).dtype)

        # Another lazy import: call luminosity_distance which imports from dsps
        _ = tengri.cosmology.luminosity_distance(0.1)
        print("AFTER_LUMINOSITY_DISTANCE", jax.config.jax_enable_x64)
        print("DTYPE_AFTER_LUMINOSITY_DISTANCE", (jnp.zeros(1) + 1.0).dtype)
        """,
    )
    assert _parse(out, "AFTER_PLANCK15") == "False", (
        "x64 flipped on at first lazy PLANCK15 access despite JAX_ENABLE_X64=0"
    )
    assert _parse(out, "DTYPE_AFTER_PLANCK15") == "float32", (
        "dtype inflated to float64 after lazy PLANCK15 import"
    )
    assert _parse(out, "AFTER_LUMINOSITY_DISTANCE") == "False", (
        "x64 flipped on at first lazy cosmology function call despite JAX_ENABLE_X64=0"
    )
    assert _parse(out, "DTYPE_AFTER_LUMINOSITY_DISTANCE") == "float32", (
        "dtype inflated to float64 after lazy cosmology function import"
    )


def test_first_ssp_load_keeps_x64_off():
    """Lazy dsps imports outside cosmology.py must respect x64_preference (#2504).

    Before the fix, the first lazy import in a non-cosmology path still flipped
    x64 on. This test exercises load_ssp_data (in dsps_wrapper.py), which is
    a non-cosmology first import, and asserts that x64 is kept off and dtypes
    remain float32 afterwards.
    """
    out = _run(
        "0",
        """
        import warnings
        warnings.simplefilter("ignore")
        import jax
        import jax.numpy as jnp
        import tengri
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

        # First lazy dsps import: load SSP data via dsps_wrapper
        # Use a relative path from repo root (where the test runs)
        try:
            ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
            print("SSP_LOAD_SUCCESS", True)
        except FileNotFoundError:
            # SSP data may not be available in all test environments
            print("SSP_LOAD_SUCCESS", False)

        print("AFTER_SSP_LOAD", jax.config.jax_enable_x64)
        print("DTYPE_AFTER_SSP_LOAD", (jnp.zeros(1) + 1.0).dtype)
        """,
    )
    # If SSP data is available, check that x64 is still off
    ssp_success = _parse(out, "SSP_LOAD_SUCCESS")
    if ssp_success == "True":
        assert _parse(out, "AFTER_SSP_LOAD") == "False", (
            "x64 flipped on at first SSP load despite JAX_ENABLE_X64=0"
        )
        assert _parse(out, "DTYPE_AFTER_SSP_LOAD") == "float32", (
            "dtype inflated to float64 after SSP load"
        )


def test_x64_hold_snapshots_config_update_not_env():
    """hold_x64_preference must snapshot jax.config, not parse JAX_ENABLE_X64 (#2504).

    The old implementation parsed the environment variable, so it only restored
    x64 if JAX_ENABLE_X64 was explicitly set to 0. If the caller set x64 to
    False via jax.config.update() with no environment variable, the context
    manager would not restore it. The new implementation snapshots the jax.config
    flag regardless of how it was set, ensuring restoration in all cases.

    This test sets x64 to False *after* importing tengri (which sets it to True
    by default when no env var is present), then calls a lazy DSPS import inside
    hold_x64_preference. If the context manager correctly snapshots jax.config
    instead of parsing the environment variable, x64 should be restored to False.
    """
    out = _run(
        None,  # No JAX_ENABLE_X64 env var
        """
        import warnings
        warnings.simplefilter("ignore")
        import jax
        import jax.numpy as jnp

        # Import tengri first (sets x64 to True by default)
        import tengri
        print("AFTER_IMPORT", jax.config.jax_enable_x64)

        # Now set x64 to False via jax.config.update (not env var)
        jax.config.update("jax_enable_x64", False)
        print("AFTER_MANUAL_UPDATE", jax.config.jax_enable_x64)

        # Trigger a lazy dsps import that uses hold_x64_preference
        # This should snapshot the current state (False) and restore it
        _ = tengri.cosmology.PLANCK15

        print("AFTER_LAZY_DSPS", jax.config.jax_enable_x64)
        print("DTYPE_AFTER", (jnp.zeros(1) + 1.0).dtype)
        """,
    )
    assert _parse(out, "AFTER_IMPORT") == "True", "tengri should set x64=True by default"
    assert _parse(out, "AFTER_MANUAL_UPDATE") == "False", "manual jax.config.update should work"
    assert _parse(out, "AFTER_LAZY_DSPS") == "False", (
        "x64 was not restored after lazy DSPS import: hold_x64_preference must "
        "snapshot jax.config, not parse the environment variable"
    )
    assert _parse(out, "DTYPE_AFTER") == "float32", (
        "dtype inflated despite x64 being restored via hold_x64_preference"
    )


@pytest.mark.parametrize(
    ("env_value", "x64"),
    [(None, "True"), ("0", "False")],
    ids=["float64_default", "float32_requested"],
)
def test_bare_import_pulls_no_dsps_or_blackjax_module(env_value, x64):
    """#2276: a bare import tengri imports no dsps or blackjax module.

    dsps.cosmology allocates a float64 device buffer at import
    (dsps/cosmology/defaults.py:8 TODAY = age_at_z0(...)), so an x64-on import
    fails on float64-less backends (jax-mps). Lazy-importing dsps inside the
    functions that use it allows the import to succeed; the first use of a dsps
    path under x64 on on MPS still fails, loudly, which is the acceptable
    failure mode.

    blackjax fails at import as well. Every other dsps import in src/ is
    already function-local; this test verifies the cosmology imports are too.
    """
    out = _run(
        env_value,
        """
        import sys, tengri
        thirdparty = sorted(m for m in sys.modules if m.split(".")[0] in ("dsps", "blackjax"))
        print("THIRDPARTY", thirdparty)
        """,
    )
    thirdparty = _parse(out, "THIRDPARTY")
    # Parse the list representation: "[]" or "['dsps.foo', ...]"
    import ast

    thirdparty_list = ast.literal_eval(thirdparty)
    assert not thirdparty_list, (
        f"a bare import tengri imported {len(thirdparty_list)} third-party modules "
        f"that allocate device buffers at import: {thirdparty_list} (#2276)"
    )
