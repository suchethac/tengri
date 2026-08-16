"""``import tengri`` must not silently discard ``JAX_ENABLE_X64`` (#1840).

tengri defaults to float64 because ``d_L^2`` at z > 0.01 overflows float32.
That default used to be enforced with an unconditional
``jax.config.update("jax_enable_x64", True)`` at import, which overrode
``JAX_ENABLE_X64=0`` — the documented JAX way to select float32 — with no
warning. Every float32 probe, benchmark and bug report that selected float32
that way ran in float64 and reported float32 as healthy.

Six DSPS modules also force the flag on at import time, so declining to force
it in ``tengri/__init__`` is necessary but not sufficient; the preference is
re-asserted after all transitive imports have run.

These run in subprocesses because the behavior under test is import-time and
process-global: nothing in-process can observe it after the fact.
"""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.regression_bug


def _probe(env_value, snippet):
    """Run `snippet` in a fresh interpreter with JAX_ENABLE_X64 set (or unset)."""
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
        timeout=600,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stdout}\n{r.stderr}"
    return r


_REPORT = """
    import warnings
    warnings.simplefilter("ignore")
    import jax
    import tengri  # noqa: F401
    import jax.numpy as jnp
    print("X64", jax.config.jax_enable_x64)
    print("DTYPE", jnp.zeros(1).dtype)
"""


@pytest.mark.parametrize("falsey", ["0", "false", "False", "no", "off"])
def test_env_request_for_float32_is_honored(falsey):
    """A falsey JAX_ENABLE_X64 survives `import tengri` and its dependencies."""
    out = _probe(falsey, _REPORT).stdout
    assert "X64 False" in out, f"JAX_ENABLE_X64={falsey} was overridden:\n{out}"
    assert "DTYPE float32" in out, f"default dtype is not float32:\n{out}"


def test_float64_remains_the_default_when_unset():
    """The default is unchanged: no variable, no surprise -- still float64."""
    out = _probe(None, _REPORT).stdout
    assert "X64 True" in out, f"float64 default was lost:\n{out}"
    assert "DTYPE float64" in out, out


@pytest.mark.parametrize("truthy", ["1", "True", "yes"])
def test_explicit_request_for_float64_is_honored(truthy):
    """An explicit truthy value keeps x64 on, same as the default."""
    out = _probe(truthy, _REPORT).stdout
    assert "X64 True" in out, out
    assert "DTYPE float64" in out, out


def test_overriding_to_float32_warns_about_the_distance_hazard():
    """Choosing float32 is allowed, but never silent.

    The overflow it risks (``d_L^2``) is catastrophic and invisible, so the
    import says so once rather than letting a user discover it as ``inf``.
    """
    r = _probe(
        "0",
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import tengri  # noqa: F401
        hits = [str(w.message) for w in caught if "JAX_ENABLE_X64" in str(w.message)]
        print("NWARN", len(hits))
        print("TEXT", hits[0] if hits else "")
        """,
    )
    assert "NWARN 1" in r.stdout, f"expected exactly one x64 warning:\n{r.stdout}"
    assert "float32" in r.stdout and "d_L^2" in r.stdout, (
        f"the warning must name the hazard it is warning about:\n{r.stdout}"
    )


def test_setup_jax_leaves_the_environment_truthful():
    """``setup_jax`` must not leave JAX_ENABLE_X64 disagreeing with the config.

    It used to ``setdefault`` the variable only when enabling x64, so a process
    that disabled it kept a stale ``True`` in the environment -- which a later
    reader, or an inheriting subprocess, would take as authoritative.
    """
    r = _probe(
        None,
        """
        import os
        import jax
        import tengri  # noqa: F401
        from tengri.utils.devices import setup_jax
        setup_jax(enable_x64=False, platform="cpu")
        print("ENV", os.environ.get("JAX_ENABLE_X64"))
        print("CFG", jax.config.jax_enable_x64)
        """,
    )
    assert "ENV 0" in r.stdout, f"environment still claims x64 is on:\n{r.stdout}"
    assert "CFG False" in r.stdout, r.stdout
