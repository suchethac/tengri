"""Compile-time benchmark for population-scale JIT kernels.

Measures Python-side tracing/lowering and XLA compile time as a function of
``N`` (number of galaxies) and ``K`` (forward_chunk_size) for several batching
strategies.  Pure-AOT path: ``jax.jit(f).lower(*abs_args).compile()`` —
*no execution*, so we are measuring compilation only.

The synthetic "forward" mimics tengri's signal_response shape:

    forward_one(ub_scalars, xi) -> data_per_gal

with closed-over "SSP-like" array (n_age, n_wave) and "filter-like" matrix
(n_band, n_wave).  Per-galaxy work is

    sfh = softmax(ub) -> sed = sfh @ ssp -> dust(sed) -> bands = filters @ sed

Variants compared:

    A. pure_lax_map     : lax.map(forward_one, xs)            # K=1
    B. manual_chunk     : lax.map(vmap(forward_one), chunked) # current code
    C. lax_map_batched  : lax.map(forward_one, xs, batch_size=K)  # new API
    D. pure_vmap        : vmap(forward_one)(xs)               # baseline
    E. checkpoint_lax   : lax.map(checkpoint(forward_one), xs)

Output: JSON with rows {variant, N, K, lower_s, compile_s, hlo_bytes}.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Force CPU + 64-bit
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TENGRI_DISABLE_JAX_CACHE", "1")  # measure cold compile

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------- #
#  Synthetic forward model (closed-over big arrays mimic SSP cube + filters)  #
# --------------------------------------------------------------------------- #

N_AGE = 64
N_WAVE = 512
N_BAND = 16
N_FREE = 8
N_GRID = 64  # field length per galaxy

# Closed-over constants (these are what may get constant-folded into HLO).
_KEY = jax.random.PRNGKey(0)
_SSP = jax.random.normal(jax.random.fold_in(_KEY, 1), (N_AGE, N_WAVE))
_FILT = jax.random.normal(jax.random.fold_in(_KEY, 2), (N_BAND, N_WAVE))
_AGE_BASIS = jax.random.normal(jax.random.fold_in(_KEY, 3), (N_GRID, N_AGE))


def forward_one(ub: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
    """Toy per-galaxy forward.  Mimics SED-model HLO weight."""
    # SFH: softmax(linear(ub) + (basis @ xi))  -> shape (N_AGE,)
    sfh_logits = jnp.tile(ub.sum(), (N_AGE,)) + _AGE_BASIS.T @ xi
    sfh = jax.nn.softmax(sfh_logits)
    sed = sfh @ _SSP  # (N_WAVE,)
    dust = jnp.exp(-0.4 * jnp.abs(ub[0]) * jnp.linspace(2.0, 0.1, N_WAVE))
    sed = sed * dust
    return _FILT @ sed  # (N_BAND,)


# --------------------------------------------------------------------------- #
#  Variants                                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Variant:
    name: str
    needs_K: bool

    def build_fn(self, N: int, K: int):
        ub = jax.ShapeDtypeStruct((N, N_FREE), jnp.float64)
        xi = jax.ShapeDtypeStruct((N, N_GRID), jnp.float64)

        if self.name == "pure_lax_map":

            def f(ub_arr, xi_arr):
                return lax.map(lambda args: forward_one(args[0], args[1]), (ub_arr, xi_arr))

            return jax.jit(f), (ub, xi)

        if self.name == "manual_chunk":
            assert N % K == 0, f"manual_chunk requires N={N} % K={K} == 0"
            n_chunks = N // K

            def f(ub_arr, xi_arr):
                ub_c = ub_arr.reshape(n_chunks, K, N_FREE)
                xi_c = xi_arr.reshape(n_chunks, K, N_GRID)
                return lax.map(
                    lambda args: jax.vmap(forward_one)(args[0], args[1]),
                    (ub_c, xi_c),
                )

            return jax.jit(f), (ub, xi)

        if self.name == "lax_map_batched":

            def f(ub_arr, xi_arr):
                return lax.map(
                    lambda args: forward_one(args[0], args[1]), (ub_arr, xi_arr), batch_size=K
                )

            return jax.jit(f), (ub, xi)

        if self.name == "pure_vmap":

            def f(ub_arr, xi_arr):
                return jax.vmap(forward_one)(ub_arr, xi_arr)

            return jax.jit(f), (ub, xi)

        if self.name == "checkpoint_lax":
            ckpt_fwd = jax.checkpoint(forward_one)

            def f(ub_arr, xi_arr):
                return lax.map(
                    lambda args: ckpt_fwd(args[0], args[1]), (ub_arr, xi_arr), batch_size=K
                )

            return jax.jit(f), (ub, xi)

        raise ValueError(self.name)


VARIANTS = [
    Variant("pure_lax_map", needs_K=False),
    Variant("manual_chunk", needs_K=True),
    Variant("lax_map_batched", needs_K=True),
    Variant("pure_vmap", needs_K=False),
    Variant("checkpoint_lax", needs_K=True),
]


def measure(variant: Variant, N: int, K: int) -> dict:
    """Return {lower_s, compile_s, hlo_bytes, error}."""
    try:
        jit_fn, abs_args = variant.build_fn(N, K)
    except AssertionError as e:
        return {"variant": variant.name, "N": N, "K": K, "error": str(e)}

    t0 = time.perf_counter()
    lowered = jit_fn.lower(*abs_args)
    t1 = time.perf_counter()

    # Get HLO size for diagnostic
    try:
        hlo_text = lowered.as_text(dialect="hlo")
        hlo_bytes = len(hlo_text)
    except Exception:
        hlo_bytes = -1

    t2 = time.perf_counter()
    try:
        _compiled = lowered.compile()
    except Exception as exc:
        return {
            "variant": variant.name,
            "N": N,
            "K": K,
            "lower_s": t1 - t0,
            "compile_s": -1.0,
            "hlo_bytes": hlo_bytes,
            "error": f"compile: {exc!r}",
        }
    t3 = time.perf_counter()

    return {
        "variant": variant.name,
        "N": N,
        "K": K,
        "lower_s": t1 - t0,
        "compile_s": t3 - t2,
        "hlo_bytes": hlo_bytes,
        "error": None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="bench/results/jit_compile_benchmark.json")
    p.add_argument("--Ns", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    p.add_argument("--Ks", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--variants", nargs="+", default=[v.name for v in VARIANTS])
    p.add_argument(
        "--max-compile-s",
        type=float,
        default=600.0,
        help="Skip later (variant, N, K) cells if compile exceeds this.",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    skip = set()  # (variant_name) — once a variant blows up at smaller N skip rest
    for variant in VARIANTS:
        if variant.name not in args.variants:
            continue
        for N in sorted(args.Ns):
            ks = args.Ks if variant.needs_K else [1]
            for K in ks:
                if variant.needs_K and K > N:
                    continue
                key = (variant.name, K)
                if key in skip:
                    continue
                print(f"  {variant.name:20s} N={N:>6d} K={K:>4d} ... ", end="", flush=True)
                row = measure(variant, N, K)
                rows.append(row)
                if row.get("error"):
                    print(f"ERROR: {row['error'][:60]}")
                    skip.add(key)
                    continue
                print(
                    f"lower={row['lower_s']:6.2f}s "
                    f"compile={row['compile_s']:6.2f}s "
                    f"hlo={row['hlo_bytes'] / 1e6:6.2f}MB"
                )
                if row["compile_s"] > args.max_compile_s:
                    print(f"  -> skipping rest of {variant.name} K={K}")
                    skip.add(key)
                # Persist incrementally
                out_path.write_text(json.dumps(rows, indent=2))

    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
