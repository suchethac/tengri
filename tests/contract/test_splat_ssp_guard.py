# SPDX-License-Identifier: BSD-3-Clause
"""A splat SSP flux is refused at build; a non-splat SSP is served exactly (#3350).

On jaxlib 0.10.2 through 0.12.0.1 the XLA CPU YNNPACK fusion returns wrong values
(exact zeros) for ``jit(vmap(...))`` over a splat model constant, so a splat SSP
(every element equal) is refused at construction. The behavioural tests use a
non-splat SSP and must hold on every jaxlib. The pure-JAX upstream repro is an
expected failure on the affected releases, non-strict, so it flags when upstream
fixes it without blocking CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import ConfigError
from tengri.forward.ssp_guard import check_ssp_flux_not_splat

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_SRC = str(Path(__file__).resolve().parents[2] / "src")
_N_MET, _N_AGE, _N_WAVE = 3, 20, 100


def _jaxlib_release() -> tuple[int, ...]:
    try:
        text = version("jaxlib")
    except PackageNotFoundError:  # pragma: no cover - jaxlib is a jax dependency
        return (0,)
    parts: list[int] = []
    for piece in text.split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
        if len(digits) != len(piece):
            break
    return tuple(parts)


_AFFECTED_FROM = (0, 10, 2)


def _ssp(flux) -> SSPData:
    wave = jnp.linspace(3000.0, 10000.0, _N_WAVE)
    ages = jnp.linspace(-1.0, 1.14, _N_AGE)
    lgmet = jnp.array([-1.5, -0.5, 0.0]) + (-1.848)
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet)


def _smooth_flux() -> jnp.ndarray:
    """Deterministic, strictly positive, non-splat: a ramp over wavelength and age."""
    wave_ramp = jnp.linspace(1.0, 2.0, _N_WAVE)
    age_ramp = jnp.linspace(0.5, 1.5, _N_AGE)
    met_ramp = jnp.linspace(0.8, 1.2, _N_MET)
    return 1e-3 * met_ramp[:, None, None] * age_ramp[None, :, None] * wave_ramp[None, None, :]


# -- the guard ---------------------------------------------------------------


@pytest.mark.parametrize(
    "flux",
    [
        jnp.ones((_N_MET, _N_AGE, _N_WAVE), dtype=jnp.float64),
        jnp.full((_N_MET, _N_AGE, _N_WAVE), 1e-4),
        jnp.zeros((_N_MET, _N_AGE, _N_WAVE)),
    ],
    ids=["ones", "full", "zeros"],
)
def test_splat_ssp_is_refused_with_the_upstream_reason(flux):
    with pytest.raises(ConfigError) as info:
        check_ssp_flux_not_splat(flux)
    message = str(info.value)
    assert "splat" in message
    assert "YNNPACK" in message
    assert "jit(vmap" in message
    assert "non-constant" in message or "varies" in message


def test_near_constant_ssp_passes_the_guard():
    flux = jnp.ones((_N_MET, _N_AGE, _N_WAVE), dtype=jnp.float64) * (
        1.0 + 1e-12 * jnp.arange(_N_WAVE)[None, None, :]
    )
    check_ssp_flux_not_splat(flux)  # does not raise


def test_non_splat_ssp_passes_the_guard():
    check_ssp_flux_not_splat(_smooth_flux())


def test_traced_flux_is_not_checked():
    """A cube traced inside jit cannot be inspected and passes through unchanged."""

    def build_and_return(x):
        check_ssp_flux_not_splat(x)
        return x * 2.0

    out = jax.jit(build_and_return)(jnp.ones((2, 2, 2)))
    np.testing.assert_allclose(np.asarray(out), 2.0)


def test_sed_model_build_refuses_a_splat_ssp():
    import tengri
    from tengri.observation import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))
    with pytest.raises(ConfigError, match="splat"):
        tengri.SEDModel.build(
            _ssp(jnp.ones((_N_MET, _N_AGE, _N_WAVE), dtype=jnp.float64)),
            observation=obs,
            sfh={"type": "tsnorm", "all_params": tengri.FREE},
            neb={"type": "none"},
            redshift=tengri.Fixed(0.05),
        )


# -- behavioural: unconditional, subprocess with a clean environment ----------

_MODEL_SCRIPT = r"""
import json
import jax
import numpy as np
import jax.numpy as jnp
import tengri
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

n_met, n_age, n_wave = 3, 20, 100
wave = jnp.linspace(3000.0, 10000.0, n_wave)
ages = jnp.linspace(-1.0, 1.14, n_age)
lgmet = jnp.array([-1.5, -0.5, 0.0]) + (-1.848)
wave_ramp = jnp.linspace(1.0, 2.0, n_wave)
age_ramp = jnp.linspace(0.5, 1.5, n_age)
met_ramp = jnp.linspace(0.8, 1.2, n_met)
flux = 1e-3 * met_ramp[:, None, None] * age_ramp[None, :, None] * wave_ramp[None, None, :]
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
import tengri  # noqa: F401

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


def _run_fresh(script: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["TENGRI_DISABLE_PRECOMP_CACHE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_non_splat_model_vmap_jit_matches_eager_float64():
    """Unconditional on every jaxlib: a non-splat SSP is served exactly under vmap."""
    out = _run_fresh(_MODEL_SCRIPT)
    eager, vmapped = np.asarray(out["eager"]), np.asarray(out["vmap"])
    assert np.all(np.isfinite(eager)) and np.all(eager > 0)
    np.testing.assert_allclose(vmapped, eager, rtol=1e-10, atol=0.0)


@pytest.mark.xfail(
    _jaxlib_release() >= _AFFECTED_FROM,
    reason="upstream XLA YNNPACK bug",
    strict=False,
)
def test_upstream_repro_pure_jax_closure_constant():
    """Expected failure on jaxlib >= 0.10.2 (upstream bug); flags when upstream fixes it."""
    out = _run_fresh(_PROBE_SCRIPT)
    np.testing.assert_allclose(
        np.asarray(out["vmap"]), np.asarray(out["eager"]), rtol=1e-10, atol=0.0
    )
