# SPDX-License-Identifier: BSD-3-Clause
"""``jax.vmap(jax.jit(predict_photometry))`` agrees with eager for a constant SSP (#3350).

With an SSP flux constant along wavelength, the batched path returned exact
zeros while eager and single-draw ``jit`` were correct. The cause is an XLA CPU
miscompile (a YNNPACK fusion of a dot against a closure-captured broadcast
constant) in the jax releases listed in
``tengri._xla_cpu_workaround.AFFECTED_JAX_VERSIONS``; tengri disables that fusion
type at import. Each comparison runs in a fresh subprocess so the workaround is
exercised exactly as a user meets it: ``import tengri`` before any jax work,
and no ``XLA_FLAGS`` inherited from the test process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from importlib import import_module
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_SRC = str(Path(__file__).resolve().parents[2] / "src")

_MODEL_SCRIPT = r"""
import json, sys
import jax
import numpy as np
import jax.numpy as jnp
import tengri
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

mode = sys.argv[1]
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
n_met, n_age, n_wave = 3, 20, 100
wave = jnp.linspace(3000.0, 10000.0, n_wave)
ages = jnp.linspace(-1.0, 1.14, n_age)
lgmet = jnp.array([-1.5, -0.5, 0.0]) + (-1.848)
if mode == "const":
    flux = scale * jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64)
else:
    key = jax.random.PRNGKey(123)
    flux = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
ssp = SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet)

def tophat(center, frac=0.16, n=40):
    w = jnp.linspace(center * (1 - frac), center * (1 + frac), n)
    t = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=w, trans=t, name=f"b{int(center)}")

centers = (3500.0, 4800.0, 6200.0)
obs = Observation(photometry=Photometry(filters=tuple(tophat(c) for c in centers)))
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "all_params": tengri.FREE},
    neb={"type": "none"},
    redshift=tengri.Fixed(0.05),
)
p = dict(model.spec.sample(jax.random.PRNGKey(0)))
batch = jax.tree_util.tree_map(lambda x: jnp.stack([jnp.asarray(x)] * 4), p)
eager = np.asarray(model.predict_photometry(p))
vmapped = np.asarray(jax.vmap(jax.jit(model.predict_photometry))(batch))[0]
print(json.dumps({"eager": eager.tolist(), "vmap": vmapped.tolist()}))
"""

_PROBE_SCRIPT = r"""
import json
import jax
import numpy as np
import jax.numpy as jnp
import tengri  # noqa: F401  (import order is the point: workaround applied first)

n_met, n_age, n_wave = 3, 20, 100
ones_cube = jnp.ones((n_met, n_age, n_wave), dtype=jnp.float64)
jw = jnp.linspace(0.1, 1.0, n_met * n_age).reshape(n_met, n_age)
xc = jnp.float64(10.0)

def f(x, w):
    per_age = jnp.einsum("ma,maw->aw", w, ones_cube * 3.828e33)
    return (10.0**x) * jnp.sum(per_age, axis=0)

bx = jnp.stack([xc] * 4)
bw = jnp.stack([jw] * 4)
eager = np.asarray(f(xc, jw))
vmapped = np.asarray(jax.vmap(jax.jit(f))(bx, bw))[0]
print(json.dumps({"eager": eager.tolist(), "vmap": vmapped.tolist()}))
"""


def _run_fresh(script: str, *args: str, x64: bool = True) -> dict:
    """Run ``script`` in a new interpreter with a clean XLA_FLAGS and return its JSON."""
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_ENABLE_X64"] = "1" if x64 else "0"
    env["TENGRI_DISABLE_PRECOMP_CACHE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode", ["const", "nonconst"])
def test_vmap_jit_predict_photometry_matches_eager_float64(mode):
    """Constant and non-constant SSP: vmap(jit) equals eager to rtol 1e-10."""
    import numpy as np

    out = _run_fresh(_MODEL_SCRIPT, mode)
    eager, vmapped = np.asarray(out["eager"]), np.asarray(out["vmap"])
    assert np.all(np.isfinite(eager)) and np.all(eager > 0)
    np.testing.assert_allclose(vmapped, eager, rtol=1e-10, atol=0.0)


def test_pure_jax_closure_constant_matches_eager_after_import_tengri():
    """The minimal pure-JAX repro, in a process that imported tengri first."""
    import numpy as np

    out = _run_fresh(_PROBE_SCRIPT)
    np.testing.assert_allclose(
        np.asarray(out["vmap"]), np.asarray(out["eager"]), rtol=1e-10, atol=0.0
    )


def test_float32_constant_ssp_is_finite():
    """Pure float32 mode stays finite for the constant SSP under vmap(jit)."""
    import numpy as np

    # Physically scaled (peak ~1e-12 Lsun/Hz/Msun, as synthetic_ssp_wide_physical):
    # the unit-normalised test grid overflows float32 for reasons unrelated to #3350.
    out = _run_fresh(_MODEL_SCRIPT, "const", "1e-12", x64=False)
    vmapped = np.asarray(out["vmap"])
    assert np.all(np.isfinite(vmapped)) and np.all(vmapped > 0)


# -- gate logic -------------------------------------------------------------


def _workaround():
    """The gate module, imported lazily so the subprocess tests still run on a tree without it."""
    return import_module("tengri._xla_cpu_workaround")


def test_gate_leaves_unaffected_jax_alone():
    wa = _workaround()
    env: dict[str, str] = {}
    status = wa.apply_ynn_fusion_workaround(env, jax_version="0.10.2", backend_initialised=False)
    assert status == "not_affected"
    assert "XLA_FLAGS" not in env


def test_gate_affected_versions_are_the_measured_pair():
    assert frozenset({"0.11.1", "0.12.0.1"}) == _workaround().AFFECTED_JAX_VERSIONS


@pytest.mark.parametrize("version", ["0.11.1", "0.12.0.1"])
def test_gate_appends_flag_and_keeps_user_flags(version):
    wa = _workaround()
    env = {"XLA_FLAGS": "--xla_force_host_platform_device_count=4"}
    status = wa.apply_ynn_fusion_workaround(env, jax_version=version, backend_initialised=False)
    assert status == "applied"
    assert env["XLA_FLAGS"] == f"--xla_force_host_platform_device_count=4 {wa.YNN_FUSION_FLAG}"


def test_gate_respects_user_set_flag():
    wa = _workaround()
    env = {"XLA_FLAGS": "--xla_cpu_experimental_ynn_fusion_type=dot"}
    before = env["XLA_FLAGS"]
    status = wa.apply_ynn_fusion_workaround(env, jax_version="0.11.1", backend_initialised=False)
    assert status == "user_set"
    assert env["XLA_FLAGS"] == before


def test_gate_warns_when_backend_already_initialised():
    wa = _workaround()
    env: dict[str, str] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        status = wa.apply_ynn_fusion_workaround(
            env, jax_version="0.11.1", backend_initialised=True
        )
    assert status == "too_late"
    assert "XLA_FLAGS" not in env
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_repeated_application_is_idempotent():
    wa = _workaround()
    env: dict[str, str] = {}
    wa.apply_ynn_fusion_workaround(env, jax_version="0.12.0.1", backend_initialised=False)
    first = env["XLA_FLAGS"]
    status = wa.apply_ynn_fusion_workaround(env, jax_version="0.12.0.1", backend_initialised=False)
    assert status == "user_set"
    assert env["XLA_FLAGS"] == first
