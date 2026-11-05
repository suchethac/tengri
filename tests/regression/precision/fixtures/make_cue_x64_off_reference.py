"""Record the x64-off Cue reference spectrum, gradient and NLL.

Run under CI's pinned JAX (``.github/ci-constraints.txt``, jax 0.12.0) from the repository
root::

    python tests/regression/precision/fixtures/make_cue_x64_off_reference.py

It reuses the fixture and helpers of ``tests/components/nebular/test_cue_process_dtype.py``
so the recipe is the one the test replays. The output is ``cue_x64_off_reference.npz``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from tengri.inference.context import InferenceContext

HERE = Path(__file__).resolve().parent
TEST = HERE.parents[2] / "components" / "nebular" / "test_cue_process_dtype.py"
OUT = HERE / "cue_x64_off_reference.npz"


def _load_test_module():
    spec = importlib.util.spec_from_file_location("cue_dtype_test", TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    t = _load_test_module()
    with jax.enable_x64(False):
        model = t._build(t.load_ssp_data(t._SSP))
        fitter = t._mock_fitter(model)
        truth32 = {
            k: jnp.asarray(v, dtype=jnp.float32)
            for k, v in t._TRUE.items()
            if k in model.spec.free_params
        }
        spec = np.asarray(model.predict_observables(truth32).spec_fnu)
        flat, unravel = ravel_pytree(
            {
                n: fitter.spec.get_distribution(n).standardize(truth32[n])
                for n in fitter._free_names
            }
        )
        ctx = InferenceContext.from_target(fitter)
        nlp = ctx.neg_log_posterior_fn
        nll = nlp(unravel(flat), ctx.data_args)
        grad = np.asarray(jax.grad(lambda u: nlp(unravel(u), ctx.data_args))(flat))
    assert spec.dtype == np.float32 and grad.dtype == np.float32
    np.savez(OUT, spec=spec, grad=grad, nll=np.asarray(float(nll), dtype=np.float64))
    print(f"wrote {OUT} (jax {jax.__version__})")


if __name__ == "__main__":
    main()
