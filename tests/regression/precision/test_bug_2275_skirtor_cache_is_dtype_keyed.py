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

_SCRIPT = r"""
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Suppress warnings to keep output clean
import warnings
warnings.simplefilter("ignore")

sys.path.insert(0, "{repo}")
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

MODE = sys.argv[1]
ssp_path = {ssp_path}
ssp = load_ssp_data(ssp_path)

obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4"]))
model_config = {{
    "ssp_data": ssp,
    "observation": obs,
    "sfh": {{
        "type": "delayed",
        "all_params": Fixed(DEFAULT),
        "log_total_mass": Uniform(9.0, 11.0),
        "tau_gyr": 1.0,
        "age_gyr": 5.0,
    }},
    "dust_attenuation": {{
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_diff": 0.3,
        "tau_bc": 0.0,
    }},
    "agn": {{
        "type": "composable",
        "all_params": Fixed(DEFAULT),
        "disc": {{"type": "kubota_done", "all_params": Fixed(DEFAULT)}},
        "torus": {{"type": "skirtor", "all_params": Fixed(DEFAULT)}},
        "norm": "cigale_joint",
        "log_lbol": Fixed(11.0),
        "fracAGN": 0.1,
    }},
    "redshift": Fixed(0.1),
}}

if MODE == "f32then64":
    # Both in the same process to share the cache (test the actual bug scenario)
    # First: float32 forward
    with jax.enable_x64(False):
        m32 = SEDModel.build(**model_config)
        p32 = {{
            "sfh_delayed_log_total_mass": jnp.asarray(10.0, dtype=jnp.float32),
            "agn_log_mbh": jnp.asarray(6.0, dtype=jnp.float32),
        }}
        _ = m32.predict(p32)  # run and discard

    # Then: float64 forward in SAME process (shares cache with above float32)
    with jax.enable_x64(True):
        m64 = SEDModel.build(**model_config)
        p64 = {{
            "sfh_delayed_log_total_mass": jnp.asarray(10.0, dtype=jnp.float64),
            "agn_log_mbh": jnp.asarray(6.0, dtype=jnp.float64),
        }}
        pred = m64.predict(p64)
        hex_val = float(np.asarray(pred.rest_sed()).sum()).hex()
        print(f"HEX64 {{hex_val}}")

elif MODE == "f64only":
    # Float64 alone in this process (clean cache)
    with jax.enable_x64(True):
        m64 = SEDModel.build(**model_config)
        p64 = {{
            "sfh_delayed_log_total_mass": jnp.asarray(10.0, dtype=jnp.float64),
            "agn_log_mbh": jnp.asarray(6.0, dtype=jnp.float64),
        }}
        pred = m64.predict(p64)
        hex_val = float(np.asarray(pred.rest_sed()).sum()).hex()
        print(f"HEX64 {{hex_val}}")
""".format(
    repo=_REPO,
    ssp_path=repr(str(_REPO / "data" / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")),
)


def test_skirtor_cache_dtype_keyed():
    """SKIRTOR caches must separate float32 and float64 arms.

    A float32 forward creates device arrays at float32. When cached, a float64
    forward reuses those arrays and gets the wrong precision, shifting the result
    by ~2.5e-9 relative. Clearing the cache or keying on dtype restores
    bit-identity.

    This test runs two separate subprocesses to ensure process-level cache
    isolation: one runs float32 then float64 in the SAME PROCESS (contaminated
    shared cache), the other runs float64 alone (clean cache). Both should
    produce bit-identical float64 results if the cache is properly keyed on
    dtype.
    """
    ssp_path = str(_REPO / "data" / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    if not Path(ssp_path).exists():
        pytest.skip(f"SSP data not found at {ssp_path}")

    env = {
        **os.environ,
        "PYTHONPATH": str(_REPO / "src"),
        "TENGRI_DISABLE_JAX_CACHE": "1",
        "TENGRI_DISABLE_PRECOMP_CACHE": "1",
    }

    # Run subprocess 1: float32 then float64 in the SAME process
    result_f32_then_64 = subprocess.run(
        [sys.executable, "-c", _SCRIPT, "f32then64"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result_f32_then_64.returncode != 0:
        raise RuntimeError(
            f"Subprocess (f32then64) failed:\n"
            f"stdout:\n{result_f32_then_64.stdout}\n"
            f"stderr:\n{result_f32_then_64.stderr}"
        )

    # Run subprocess 2: float64 alone
    result_f64_only = subprocess.run(
        [sys.executable, "-c", _SCRIPT, "f64only"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result_f64_only.returncode != 0:
        raise RuntimeError(
            f"Subprocess (f64only) failed:\n"
            f"stdout:\n{result_f64_only.stdout}\n"
            f"stderr:\n{result_f64_only.stderr}"
        )

    # Extract HEX64 from each output
    hex_f32_then_64 = None
    hex_f64_only = None

    for line in result_f32_then_64.stdout.strip().split("\n"):
        if line.startswith("HEX64 "):
            hex_f32_then_64 = line.split(" ", 1)[1]

    for line in result_f64_only.stdout.strip().split("\n"):
        if line.startswith("HEX64 "):
            hex_f64_only = line.split(" ", 1)[1]

    if hex_f32_then_64 is None:
        raise RuntimeError(
            f"Failed to extract HEX64 from f32then64 subprocess output:\n"
            f"{result_f32_then_64.stdout}"
        )
    if hex_f64_only is None:
        raise RuntimeError(
            f"Failed to extract HEX64 from f64only subprocess output:\n{result_f64_only.stdout}"
        )

    assert hex_f32_then_64 == hex_f64_only, (
        f"Float64 result differs when float32 ran first in the same process: "
        f"f32then64: {hex_f32_then_64}, f64only: {hex_f64_only}. "
        f"SKIRTOR grid caches (_load_raw_disk_dust_grid, _load_skirtor_default_grid) "
        f"are not keyed on the process float dtype, so float32 arrays are served "
        f"to float64 (#2275; same disease as #1392, #2024)"
    )
