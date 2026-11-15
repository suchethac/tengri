# SPDX-License-Identifier: BSD-3-Clause
"""Temporary probe: Linux x64-off Cue spectrum digest against the runner CPU (not for merge)."""

import hashlib
import importlib.util
import platform

import jax
import jax.numpy as jnp
import numpy as np

from tengri import load_ssp_data

_spec = importlib.util.spec_from_file_location(
    "t", "tests/components/nebular/test_cue_process_dtype.py"
)
t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t)

flags: set[str] = set()
model = "?"
with open("/proc/cpuinfo") as fh:
    for line in fh:
        if line.startswith("model name") and model == "?":
            model = line.split(":", 1)[1].strip()
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            break
isa = sorted(f for f in flags if f.startswith(("avx", "amx", "fma", "f16c")))
with jax.enable_x64(False):
    m = t._build(load_ssp_data(t._SSP))
    tr = {
        k: jnp.asarray(v, dtype=jnp.float32)
        for k, v in t._TRUE.items()
        if k in m.spec.free_params
    }
    s = np.asarray(m.predict_observables(tr).spec_fnu)
digest = hashlib.sha256(np.ascontiguousarray(s).tobytes()).hexdigest()
print("PROBE", digest[:8], "|", model, "|", " ".join(isa), "|", jax.__version__, platform.machine())
