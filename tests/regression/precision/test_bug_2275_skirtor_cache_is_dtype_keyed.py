# SPDX-License-Identifier: BSD-3-Clause
r"""SKIRTOR grid caches are keyed on the process float dtype (#2275).

Two functools.cache loaders in skirtor.py hold the state: _load_raw_disk_dust_grid
and _load_skirtor_default_grid. They call _load_skirtor_grid_data which creates
jnp.array() device arrays at the FIRST call's float dtype and reuses them by
every later call. A float32 forward shifts a later float64 forward by 2.5e-9
relative; clearing only these two between arms restores bit-identical float64.

The cache key must include the process float dtype so a float32 arm never hands
its arrays to a float64 arm. Same disease as #1392 and #2024.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression_bug

_REPO = Path(__file__).resolve().parents[3]


def _float32_then_float64(ssp_path: str) -> tuple[str, str]:
    """Run float32 then float64, return the float64 hex() both times.

    Subprocess-isolated: the behavior under test is process-global dtype state.
    """
    code = """
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Suppress warnings to keep output clean
import warnings
warnings.simplefilter("ignore")

sys.path.insert(0, "{repo}")
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

ssp_path = {ssp_path}
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
ssp = load_ssp_data(ssp_path)

obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4"]))

# Arm A: float32 first, then float64
with jax.enable_x64(False):
    m32 = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={{"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0), "tau_gyr": 1.0, "age_gyr": 5.0}},
        dust_attenuation={{"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT), "tau_diff": 0.3, "tau_bc": 0.0}},
        agn={{"type": "composable", "all_params": Fixed(DEFAULT), "disc": {{"type": "kubota_done", "all_params": Fixed(DEFAULT)}}, "torus": {{"type": "skirtor", "all_params": Fixed(DEFAULT)}}, "norm": "cigale_joint", "log_lbol": Fixed(11.0), "fracAGN": 0.1}},
        redshift=Fixed(0.1)
    )
    p32 = {{"sfh_delayed_log_total_mass": jnp.asarray(10.0, dtype=jnp.float32), "agn_log_mbh": jnp.asarray(6.0, dtype=jnp.float32)}}
    pred32 = m32.predict(p32)

with jax.enable_x64(True):
    m64_after_f32 = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={{"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0), "tau_gyr": 1.0, "age_gyr": 5.0}},
        dust_attenuation={{"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT), "tau_diff": 0.3, "tau_bc": 0.0}},
        agn={{"type": "composable", "all_params": Fixed(DEFAULT), "disc": {{"type": "kubota_done", "all_params": Fixed(DEFAULT)}}, "torus": {{"type": "skirtor", "all_params": Fixed(DEFAULT)}}, "norm": "cigale_joint", "log_lbol": Fixed(11.0), "fracAGN": 0.1}},
        redshift=Fixed(0.1)
    )
    p64_after_f32 = {{"sfh_delayed_log_total_mass": jnp.asarray(10.0, dtype=jnp.float64), "agn_log_mbh": jnp.asarray(6.0, dtype=jnp.float64)}}
    pred64_after_f32 = m64_after_f32.predict(p64_after_f32)
    hex64_after_f32 = float(pred64_after_f32.rest_sed().sum()).hex()

# Arm B: float64 alone (reference)
with jax.enable_x64(True):
    m64_alone = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={{"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0), "tau_gyr": 1.0, "age_gyr": 5.0}},
        dust_attenuation={{"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT), "tau_diff": 0.3, "tau_bc": 0.0}},
        agn={{"type": "composable", "all_params": Fixed(DEFAULT), "disc": {{"type": "kubota_done", "all_params": Fixed(DEFAULT)}}, "torus": {{"type": "skirtor", "all_params": Fixed(DEFAULT)}}, "norm": "cigale_joint", "log_lbol": Fixed(11.0), "fracAGN": 0.1}},
        redshift=Fixed(0.1)
    )
    p64_alone = {{"sfh_delayed_log_total_mass": jnp.asarray(10.0, dtype=jnp.float64), "agn_log_mbh": jnp.asarray(6.0, dtype=jnp.float64)}}
    pred64_alone = m64_alone.predict(p64_alone)
    hex64_alone = float(pred64_alone.rest_sed().sum()).hex()

print("HEX64_AFTER_F32", hex64_after_f32)
print("HEX64_ALONE", hex64_alone)
""".format(repo=_REPO, ssp_path=repr(ssp_path))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO,
        env={**dict(os.environ), "PYTHONPATH": str(_REPO / "src")},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    output_lines = result.stdout.strip().split("\n")
    hex64_after_f32 = None
    hex64_alone = None
    for line in output_lines:
        if line.startswith("HEX64_AFTER_F32 "):
            hex64_after_f32 = line.split(" ", 1)[1]
        elif line.startswith("HEX64_ALONE "):
            hex64_alone = line.split(" ", 1)[1]

    if hex64_after_f32 is None or hex64_alone is None:
        raise RuntimeError(f"Failed to extract hex values from output:\n{result.stdout}")

    return hex64_after_f32, hex64_alone


def test_skirtor_cache_dtype_keyed():
    """SKIRTOR caches must separate float32 and float64 arms.

    A float32 forward creates device arrays at float32. When cached, a float64
    forward reuses those arrays and gets the wrong precision, shifting the result
    by ~2.5e-9 relative. Clearing the cache or keying on dtype restores
    bit-identity.
    """
    ssp_path = str(_REPO / "data" / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    if not Path(ssp_path).exists():
        pytest.skip(f"SSP data not found at {ssp_path}")

    hex64_after_f32, hex64_alone = _float32_then_float64(ssp_path)

    assert hex64_after_f32 == hex64_alone, (
        f"Float64 result differs depending on whether float32 ran first: "
        f"float64 after float32: {hex64_after_f32}, float64 alone: {hex64_alone}. "
        f"SKIRTOR grid caches (_load_raw_disk_dust_grid, _load_skirtor_default_grid) "
        f"are not keyed on the process float dtype, so float32 arrays are served to float64 "
        f"(#2275; same disease as #1392, #2024)"
    )
