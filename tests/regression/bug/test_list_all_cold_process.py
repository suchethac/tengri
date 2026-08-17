# SPDX-License-Identifier: BSD-3-Clause
"""``tengri.list_all()`` must work in a process that imported nothing else.

``_check_deps_importable`` answered "is nifty8 installed?" by *importing*
nifty8, which runs the whole package body. Asked for twenty backends during a
registry listing, that pulled ``asyncio`` back in while it was already
mid-import, and the call died with ``NameError: name 'base_events' is not
defined`` -- a message naming neither tengri nor the package that caused it.

The bug only bites cold. Once anything in the process has imported those
packages the probe is a no-op and the listing succeeds, which is why it
survived interactive use, the notebooks, and the whole test suite: every one
of them imports something first. ``tengri.list_all()`` is documented in
``tengri/__init__.py`` as the way to enumerate every registry live and
"prefer it to any list", so the one call most likely to be someone's first
line was the one that failed.

Hence the subprocess. Running this in-process would pass against the unfixed
code, because pytest has long since imported everything involved -- the
isolation *is* the test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# A cold interpreter: import tengri, call the menu, print what came back.
_PROBE = textwrap.dedent(
    """
    import tengri
    registries = tengri.list_all()
    methods = tengri.list_inference_methods()
    print("OK", len(registries), len(methods))
    """
).strip()


@pytest.mark.regression_bug
def test_list_all_survives_a_cold_interpreter() -> None:
    """A fresh process must be able to enumerate the registries."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"tengri.list_all() failed in a cold process:\n{proc.stderr[-2000:]}"
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("OK")), "")
    assert line, f"probe produced no OK line; stdout was:\n{proc.stdout[-2000:]}"

    _, n_registries, n_methods = line.split()
    # Exact counts are a moving target; that the menus are non-empty is not.
    assert int(n_registries) > 0
    assert int(n_methods) > 0


@pytest.mark.regression_bug
def test_dep_probe_does_not_execute_optional_packages() -> None:
    """Listing backends must not import NIFTy, blackjax or optax as a side effect.

    The presence check is the fix's mechanism, so pin it: a menu call that
    imports its optional dependencies has regressed to the old behavior even
    if it happens not to crash on the machine running it.
    """
    probe = textwrap.dedent(
        """
        import sys
        import tengri
        tengri.list_inference_methods()
        leaked = sorted(p for p in ("nifty8", "blackjax", "optax") if p in sys.modules)
        print("LEAKED", ",".join(leaked))
        """
    ).strip()
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("LEAKED")), None)
    assert line is not None, proc.stdout[-2000:]
    leaked = line.removeprefix("LEAKED").strip()
    assert not leaked, f"listing backends imported optional packages: {leaked}"
