# SPDX-License-Identifier: BSD-3-Clause
"""float32 stays float32 end to end on a spectroscopic fit (#2850).

#2504 (lazy DSPS imports flipping ``jax_enable_x64`` back on) was fixed by
#2512's ``hold_x64_preference``, and the photometry-only configuration was
covered by it. A GPU user still reports the flip on a build that includes
#2512, on a configuration no test had exercised: a DESI-style spectrum with a
banded resolution matrix, plus photometry, a continuity SFH, two-component dust,
a nebular backend and NUTS.

The behavior under test is process-global and import-time, so the whole
pipeline runs in a fresh interpreter with ``JAX_ENABLE_X64=0``. The subprocess
prints one ``RESULT`` JSON line recording the x64 flag and the dtypes after each
stage, so a failure names the stage that flipped.

Every on-disk cache tengri uses is redirected into a per-test directory, so the
result cannot depend on what an earlier process left on this machine. The
ionspec table cache is the case that matters: before #2849 its key ignored the
computation dtype, so a float32 build read a table computed in float64 and Cue's
float32 overflow stayed hidden.
"""

import json
import subprocess
import sys

import pytest

from tengri._data_setup import find_data

pytestmark = pytest.mark.regression_bug

_STAGES = ("import", "build", "predict", "loglik_grad", "jaxpr", "nuts")

_SSP_NAME = "fsps_prsc_miles_chabrier.h5"

# The subprocess body. ``__CONFIG__`` is replaced with a JSON literal (only
# strings are passed, so it is also a valid Python literal).
_SCRIPT = r"""
import json
import sys
import warnings

warnings.simplefilter("ignore")
CONFIG = __CONFIG__

import jax
import jax.numpy as jnp
import numpy as np

import tengri
from tengri import DEFAULT, FREE, Fitter, Fixed, Observation, Photometry, SEDModel
from tengri import Spectroscopy, Uniform
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.observation.banded import gaussian_resolution_bands

stages = {}


def record(name, **dtypes):
    entry = {"x64": bool(jax.config.jax_enable_x64)}
    entry.update(dtypes)
    stages[name] = entry


def emit(payload):
    print("RESULT " + json.dumps(payload), flush=True)
    raise SystemExit(0)


record("import")

try:
    ssp = load_ssp_data(CONFIG["ssp"])
except FileNotFoundError as exc:
    emit({"skip": f"SSP grid not on this machine: {exc}"})

try:
    wave = np.linspace(3600.0, 9800.0, 2000)
    resolution_matrix = gaussian_resolution_bands(jnp.asarray(wave), 2500.0, n_diag=11)
    observation = Observation(
        photometry=Photometry.from_names(["des_g", "des_r", "des_z", "wise_w1"]),
        spectroscopy=Spectroscopy(
            wave_obs=jnp.asarray(wave), resolution_matrix=resolution_matrix
        ),
    )
    if CONFIG["neb"] == "cue":
        neb = {"type": "cue", "all_params": Fixed(DEFAULT)}
    else:
        neb = {"type": "none"}
    model = SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "continuity",
            "log_total_mass": Fixed(10.0),
            "all_params": FREE,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Uniform(0.0, 2.0),
        },
        neb=neb,
        redshift=Fixed(0.1),
    )
    record("build")

    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    spec_flux = model.predict_spectrum(params)
    phot_flux = model.predict_photometry(params)
    record(
        "predict",
        spectrum=str(spec_flux.dtype),
        photometry=str(phot_flux.dtype),
    )

    rng = np.random.default_rng(0)
    phot_obs = np.asarray(phot_flux) * (1.0 + 0.02 * rng.standard_normal(phot_flux.shape))
    spec_obs = np.asarray(spec_flux) * (1.0 + 0.02 * rng.standard_normal(spec_flux.shape))
    data = np.concatenate([phot_obs, spec_obs])
    noise = np.concatenate(
        [0.05 * np.abs(phot_obs) + 1e-31, 0.05 * np.abs(spec_obs) + 1e-31]
    )
    fitter = Fitter(
        model,
        data=jnp.asarray(data),
        noise=jnp.asarray(noise),
        data_type="joint",
    )
    loglik = build_loglikelihood_fn(fitter)
    free_names = set(model.spec.free_params)
    free = {k: v for k, v in params.items() if k in free_names}

    def objective(q):
        return loglik(q, fitter._data_args)

    value, grad = jax.jit(jax.value_and_grad(objective))(free)
    grad_dtypes = sorted({str(leaf.dtype) for leaf in jax.tree_util.tree_leaves(grad)})
    record("loglik_grad", value=str(jnp.asarray(value).dtype), grad=grad_dtypes)

    jaxpr = str(jax.make_jaxpr(objective)(free))
    record("jaxpr", f64_in_jaxpr=("f64" in jaxpr) or ("float64" in jaxpr))

    posterior = fitter.run(
        "mcmc_nuts", n_warmup=5, n_samples=5, key=jax.random.PRNGKey(1)
    )
    record("nuts", posterior=type(posterior).__name__)
except Exception as exc:
    emit(
        {
            "error": {"type": type(exc).__name__, "message": str(exc)[:2000]},
            "stages": stages,
        }
    )

emit({"neb": CONFIG["neb"], "stages": stages})
"""


def _run_pipeline(neb, ssp_path, cache_root):
    config = {"neb": neb, "ssp": str(ssp_path)}
    script = _SCRIPT.replace("__CONFIG__", json.dumps(config))
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_subprocess_env(cache_root),
        timeout=900,
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"pipeline printed no RESULT line (rc={result.returncode}):\n" + (
        result.stdout[-2000:] + "\n" + result.stderr[-4000:]
    )
    return json.loads(lines[-1][len("RESULT ") :])


def _subprocess_env(cache_root):
    """Inherit the pytest environment, minus what hides Cue's guard or reuses caches.

    ``tests/conftest.py`` sets ``TENGRI_ALLOW_WNE_CUE=1`` suite-wide, which
    downgrades Cue's Q_H wNE heuristic (``CueWNESSPError``) to a warning. A
    user's session does not set it, so the subprocess must not either.

    Caches are isolated under ``cache_root``:

    * ``TENGRI_JAX_CACHE_DIR`` moves the JAX compile cache and, because
      ``_ionspec_disk_cache_dir`` resolves through it, the ionspec table cache.
    * ``TENGRI_PRECOMP_CACHE_DIR`` moves the precompute (WavePrecomp / SSP
      precompute) cache. The disable switches are removed so the isolated
      directory is the one exercised.
    * ``XDG_CACHE_HOME`` is also redirected as a fallback for any default path.
    """
    import os

    env = dict(os.environ)
    env.pop("TENGRI_ALLOW_WNE_CUE", None)
    env.pop("TENGRI_DISABLE_JAX_CACHE", None)
    env.pop("TENGRI_DISABLE_PRECOMP_CACHE", None)
    env["TENGRI_JAX_CACHE_DIR"] = str(cache_root / "jax")
    env["TENGRI_PRECOMP_CACHE_DIR"] = str(cache_root / "precomp")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    env["JAX_ENABLE_X64"] = "0"
    env["JAX_PLATFORMS"] = "cpu"
    return env


def _assert_stage_float32(stages, name, **dtypes):
    assert name in stages, f"stage {name!r} never completed: {sorted(stages)}"
    entry = stages[name]
    assert entry["x64"] is False, f"stage {name!r}: jax_enable_x64 flipped to True"
    for key, dtype in dtypes.items():
        got = entry[key]
        if isinstance(dtype, list):
            assert got == dtype, f"stage {name!r}: {key} dtypes {got} != {dtype}"
        else:
            assert got == dtype, f"stage {name!r}: {key} is {got}, expected {dtype}"


@pytest.mark.parametrize("neb", ["none", "cue"])
def test_float32_stays_float32_end_to_end_spectroscopic_fit(neb, tmp_path):
    """Spectrum+photometry, banded R, continuity SFH, 2-component dust, NUTS: x64 stays off.

    Asserts, per stage, that ``jax_enable_x64`` is False and the outputs are
    float32, and that the jitted log-likelihood jaxpr has no float64 type.
    All on-disk caches are redirected into ``tmp_path``.
    """
    ssp_path = find_data(_SSP_NAME)
    if ssp_path is None:
        ssp_path = _SSP_NAME  # missing; the subprocess skips on FileNotFoundError

    result = _run_pipeline(neb, ssp_path, tmp_path / "cache")

    if "skip" in result:
        pytest.skip(result["skip"])
    if "error" in result:
        err = result["error"]
        pytest.fail(
            f"pipeline raised {err['type']} after stages {sorted(result['stages'])}: "
            f"{err['message']}"
        )

    stages = result["stages"]
    _assert_stage_float32(stages, "import")
    _assert_stage_float32(stages, "build")
    _assert_stage_float32(stages, "predict", spectrum="float32", photometry="float32")
    _assert_stage_float32(stages, "loglik_grad", value="float32", grad=["float32"])
    _assert_stage_float32(stages, "jaxpr")
    assert stages["jaxpr"]["f64_in_jaxpr"] is False, "float64 type in the jitted loglik jaxpr"
    _assert_stage_float32(stages, "nuts")
