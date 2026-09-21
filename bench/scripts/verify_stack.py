#!/usr/bin/env python
"""Record what the benchmark stack actually is, as data rather than as a claim.

Answers the two checks requested on issue #2444:

1. **Device-matmul provenance as a JSON field, not only a precondition.** The
   plugin's CUDA version gate passes partly by an upstream scaling bug
   (``build_version // 100 <= runtime_version``, i.e. ``908 <= 90500``). A gate
   that passes by accident can begin failing by accident, and the fallback is
   CPU -- which would yield a plausible but wrong "GPU" curve. So an f32 and an
   f64 matmul are compiled and executed on the device and their results,
   dtypes and placements are written into the output.

2. **Whether cuDNN is ever actually called.** "tengri invokes no convolution
   kernels" was offered as reasoning; this measures it. A representative
   forward pass and its gradient are compiled with ``--xla_dump_to`` and the
   dumped HLO is scanned for cuDNN custom-call targets. Zero hits is the
   evidence; any hit falsifies the premise and must be reported, not filtered.

Writes one JSON object to ``--out``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

# cuDNN surfaces in HLO as custom-call targets. The convolution ones are
# `__cudnn$conv*`; fused attention/norm kernels also use the `__cudnn$` prefix.
# Match the prefix rather than a fixed list so a kernel we did not anticipate
# still trips it.
CUDNN_PATTERN = re.compile(r"__cudnn\$[A-Za-z0-9_]*")


def device_matmul_check() -> dict:
    """Compile and run a real matmul in both precisions on the device."""
    import jax
    import jax.numpy as jnp

    out: dict = {}
    devs = jax.devices()
    out["devices"] = [str(d) for d in devs]
    out["platform"] = devs[0].platform
    out["device_kind"] = getattr(devs[0], "device_kind", None)

    x = jnp.ones((512, 512), dtype=jnp.float32)
    r32 = jax.jit(lambda a: (a @ a).sum())(x)
    out["f32_matmul_value"] = float(r32)
    out["f32_matmul_dtype"] = str(r32.dtype)
    out["f32_matmul_device"] = str(list(r32.devices())[0])
    out["f32_matmul_expected"] = 512.0 * 512 * 512
    out["f32_matmul_ok"] = abs(float(r32) - 512.0**3) / 512.0**3 < 1e-3

    with jax.enable_x64():
        y = jnp.ones((512, 512), dtype=jnp.float64)
        r64 = jax.jit(lambda a: (a @ a).sum())(y)
        out["f64_matmul_value"] = float(r64)
        out["f64_matmul_dtype"] = str(r64.dtype)
        out["f64_matmul_device"] = str(list(r64.devices())[0])
        out["f64_matmul_ok"] = (
            str(r64.dtype) == "float64" and abs(float(r64) - 512.0**3) / 512.0**3 < 1e-12
        )

    out["all_ok"] = bool(
        out["platform"] == "gpu" and out["f32_matmul_ok"] and out["f64_matmul_ok"]
    )
    return out


def cuda_component_versions() -> dict:
    """Runtime versions of the CUDA components, as the plugin sees them."""
    try:
        from jax_plugins.xla_cuda12 import cuda_versions  # type: ignore
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    got = {}
    for name, fn in [
        ("cuda_runtime", "cuda_runtime_get_version"),
        ("cuda_runtime_build", "cuda_runtime_build_version"),
        ("cudnn", "cudnn_get_version"),
        ("cudnn_build", "cudnn_build_version"),
        ("cublas", "cublas_get_version"),
        ("cublas_build", "cublas_build_version"),
        ("cufft", "cufft_get_version"),
        ("cupti", "cupti_get_version"),
    ]:
        try:
            got[name] = int(getattr(cuda_versions, fn)())
        except Exception as exc:
            got[name] = f"unavailable: {type(exc).__name__}"
    return got


def hlo_cudnn_scan(dump_dir: str) -> dict:
    """Compile a representative forward + gradient and scan the dumped HLO."""
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    # Build the same fixture the device matrix uses, via the repo's own script.
    repo = os.environ.get("TENGRI_REPO", "/oak/stanford/projects/c4u/u/cooray/tengri")
    sys.path.insert(0, os.path.join(repo, "bench", "scripts"))
    import benchmark_device_matrix as bdm  # noqa: E402

    model, _, _ = bdm.build("wave_precomp", "photometry", 2000)
    p = bdm.reference_params(model)
    single, batch_fn, _ = bdm.observable(model, "photometry")

    pb = bdm.batch_params(model, 512)
    fwd = jax.jit(batch_fn)
    jax.block_until_ready(fwd(pb))

    grad_batch = jax.jit(jax.vmap(lambda q: jax.grad(lambda r: jnp.sum(single(r)))(q)))
    jax.block_until_ready(grad_batch(pb))

    files = sorted(glob.glob(os.path.join(dump_dir, "*")))
    hits: dict[str, list[str]] = {}
    scanned = 0
    for f in files:
        if not f.endswith((".txt", ".ll", ".hlo")) and "hlo" not in os.path.basename(f):
            continue
        try:
            text = pathlib.Path(f).read_text(errors="replace")
        except Exception:
            continue
        scanned += 1
        found = sorted(set(CUDNN_PATTERN.findall(text)))
        if found:
            hits[os.path.basename(f)] = found
    return {
        "dump_files_total": len(files),
        "dump_files_scanned": scanned,
        "cudnn_custom_call_hits": hits,
        "cudnn_custom_call_count": sum(len(v) for v in hits.values()),
        "clean": len(hits) == 0,
    }


def _fail(msg: str) -> int:
    print(f"FATAL: {msg}", file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-hlo", action="store_true")
    ap.add_argument(
        "--dump-dir",
        help="Directory XLA_FLAGS=--xla_dump_to already points at. XLA reads that "
        "flag at backend init, so the CALLER must set it before this process "
        "starts; setting it here would be too late and would silently dump "
        "nothing, which reads identically to 'no cuDNN found'.",
    )
    args = ap.parse_args()

    if not args.skip_hlo:
        if not args.dump_dir:
            return _fail("--dump-dir is required unless --skip-hlo")
        if f"--xla_dump_to={args.dump_dir}" not in os.environ.get("XLA_FLAGS", ""):
            return _fail(
                "XLA_FLAGS does not contain --xla_dump_to=%s (got %r). Refusing to "
                "report an empty scan as a clean one." % (args.dump_dir, os.environ.get("XLA_FLAGS"))
            )

    result: dict = {"argv": sys.argv}

    try:
        result["nvidia_smi_L"] = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception as exc:
        result["nvidia_smi_L"] = f"unavailable: {exc}"

    result["env"] = {
        k: os.environ.get(k)
        for k in ("JAX_PLATFORMS", "JAX_ENABLE_X64", "CUDA_HOME", "SLURM_JOB_ID")
    }

    result["device_matmul"] = device_matmul_check()
    result["cuda_component_versions"] = cuda_component_versions()

    if not args.skip_hlo:
        result["hlo_dump_dir"] = args.dump_dir
        result["xla_flags"] = os.environ.get("XLA_FLAGS")
        try:
            result["cudnn_scan"] = hlo_cudnn_scan(args.dump_dir)
        except Exception as exc:
            import traceback

            result["cudnn_scan"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["device_matmul"].get("all_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
