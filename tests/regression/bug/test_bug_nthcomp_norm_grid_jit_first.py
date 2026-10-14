# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the kubota_done disc works when its FIRST call is under ``jax.jit``.

``nthcomp_norm_grid`` read the cached template frequency grid through ``_get_nu_jax``,
which returns a device array built by ``jnp.asarray``; under a jit trace that is a tracer,
and ``np.asarray`` on it raises ``TracerArrayConversionError``. #2885 moved the cached
loaders to host arrays but left this host-side consumer on the device accessor (#2818 added
it). The check runs in a fresh interpreter so every cache is cold and the jit call is the
first call, which an in-process test cannot guarantee.

Marker: regression_bug
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.regression_bug

_SCRIPT = textwrap.dedent(
    """
    import jax, jax.numpy as jnp, numpy as np
    from tengri.components.agn.unified import kubota_done_full_agn

    wave = jnp.geomspace(1.0e3, 1.0e5, 64)
    run = jax.jit(lambda w, lb: kubota_done_full_agn(w, agn_log_lbol=lb))
    first = np.asarray(run(wave, 44.0))
    second = np.asarray(run(wave, 43.0))
    assert np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    print("OK")
    """
)


def test_kubota_done_first_call_under_jit_is_finite():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-1500:]
