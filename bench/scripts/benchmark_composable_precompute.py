"""Benchmark: composable AGN runner vs precompute lookup.

Measures the three evaluation modes of the composable AGN block subsystem:

1. **Exact runtime** — ``composable_agn_l_nu`` without JIT (Python loop;
   the slowest, useful only for one-off plotting / debugging).
2. **JIT-composable** — ``jax.jit(composable_agn_l_nu)`` followed by a
   trapezoidal filter integration. Full physics traced into XLA on first
   call.
3. **Precompute lookup** — ``composable_precompute.precompute()`` builds a
   small triweight grid once; ``build_lookup()`` returns a JIT-compiled
   callable that takes the recipe's axis values and returns filter
   photometry.

Output: a small table comparing compile + per-call cost for a
template-heavy recipe (GRAHSP BBB + SKIRTOR torus + SMC attenuation).

Run::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_composable_precompute.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn.blocks import Recipe, composable_agn_l_nu
from tengri.components.agn.blocks.composable_precompute import (
    build_lookup,
    precompute,
)

_C_AA_PER_S: float = 2.99792458e18


def _toy_filter():
    wave = np.linspace(4500.0, 6500.0, 200)
    trans = np.exp(-0.5 * ((wave - 5500.0) / 300.0) ** 2)
    return wave, trans


def _bandpass_l_nu(l_nu: np.ndarray, wave_aa: np.ndarray,
                   filter_wave: np.ndarray, filter_trans: np.ndarray) -> float:
    """Filter-averaged :math:`L_\\nu` (matches preintegrate_grid convention)."""
    nu = _C_AA_PER_S / wave_aa
    trans_interp = np.interp(wave_aa, filter_wave, filter_trans, left=0.0, right=0.0)
    order = np.argsort(nu)
    num = np.trapezoid((l_nu * trans_interp / nu)[order], nu[order])
    den = np.trapezoid((trans_interp / nu)[order], nu[order])
    return float(num / den)


def _time(fn, n_warmup=1, n_repeat=20) -> tuple[float, float]:
    """Return (compile/first-call time [s], median cached time [s])."""
    t0 = time.time()
    out = fn()
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    t_first = time.time() - t0
    samples: list[float] = []
    for _ in range(n_warmup + n_repeat):
        t = time.time()
        out = fn()
        jax.tree.map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            out,
        )
        samples.append(time.time() - t)
    return t_first, float(np.median(samples[n_warmup:]))


def main() -> None:
    fw, ft = _toy_filter()
    wave_aa = np.logspace(2.0, 6.0, 1500)
    wave_jax = jnp.asarray(wave_aa)

    # Template-heavy recipe — GRAHSP BBB / lines / FeII / SKIRTOR torus / SMC.
    recipe_kw = dict(
        agn_disc_block="grahsp_sbpl",
        agn_lines_block="grahsp",
        agn_feii_block="grahsp",
        agn_torus_block="skirtor",
        agn_attenuation_block="smc_prevot",
    )

    fixed_kw = dict(
        agn_log_lbol=45.0,
        agn_grahsp_uvslope=0.0,
        agn_grahsp_plslope=-1.7,
        agn_grahsp_plbendloc_nm=100.0,
        agn_grahsp_plbendwidth=1.0,
        agn_grahsp_cutoff_nm=10000.0,
        agn_grahsp_a_lines=1.0,
        agn_grahsp_a_feii=5.0,
        agn_grahsp_linewidth_kms=5000.0,
        agn_tau_skirtor=7.0,
        agn_attenuation_ebv=0.1,
    )

    # ──────────────────────────────────────────────────────────────────
    # Mode 1: exact (no JIT) + filter integration
    # ──────────────────────────────────────────────────────────────────
    def _eager():
        l_nu = composable_agn_l_nu(
            wave_jax,
            **recipe_kw,
            **fixed_kw,
            agn_grahsp_l5100=1.0e44,
        )
        return _bandpass_l_nu(np.asarray(l_nu), wave_aa, fw, ft)

    t_eager_first, t_eager_med = _time(_eager)

    # ──────────────────────────────────────────────────────────────────
    # Mode 2: JIT-composable + filter integration (JAX trapz inside JIT)
    # ──────────────────────────────────────────────────────────────────
    @jax.jit
    def _jit_full(l5100):
        l_nu = composable_agn_l_nu(
            wave_jax,
            **recipe_kw,
            **fixed_kw,
            agn_grahsp_l5100=l5100,
        )
        nu = _C_AA_PER_S / wave_jax
        trans = jnp.interp(wave_jax, jnp.asarray(fw), jnp.asarray(ft), left=0.0, right=0.0)
        order = jnp.argsort(nu)
        num = jnp.trapezoid((l_nu * trans / nu)[order], nu[order])
        den = jnp.trapezoid((trans / nu)[order], nu[order])
        return num / den

    t_jit_first, t_jit_med = _time(lambda: _jit_full(jnp.array(1.0e44)))

    # ──────────────────────────────────────────────────────────────────
    # Mode 3: precompute + triweight lookup
    # ──────────────────────────────────────────────────────────────────
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        lines="grahsp",
        feii="grahsp",
        torus="skirtor",
        attenuation="smc_prevot",
        axis_params=("agn_grahsp_l5100",),
    )
    t_build = time.time()
    pre = precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={"agn_grahsp_l5100": np.logspace(43, 46, 5)},
    )
    fn = build_lookup(pre)
    t_build = time.time() - t_build

    t_pre_first, t_pre_med = _time(lambda: fn(jnp.array(1.0), jnp.array(1.0e44)))

    print()
    print("composable AGN forward-model evaluation modes")
    print("─" * 67)
    print(f"  Recipe: {dict(recipe_kw)}")
    print(f"  Wave grid: {wave_aa.size} points; 1 filter")
    print("  Repeat = 20, median reported.")
    print()
    print(f"{'mode':<25}{'first call':>15}{'cached':>15}{'speedup':>12}")
    print("─" * 67)
    print(
        f"{'exact (no JIT)':<25}"
        f"{t_eager_first*1e3:>12.1f} ms"
        f"{t_eager_med*1e3:>12.1f} ms"
        f"{1.0:>12.1f}x"
    )
    print(
        f"{'JIT-composable':<25}"
        f"{t_jit_first*1e3:>12.1f} ms"
        f"{t_jit_med*1e3:>12.3f} ms"
        f"{t_eager_med/t_jit_med:>12.1f}x"
    )
    print(
        f"{'precompute (build)':<25}"
        f"{t_build*1e3:>12.1f} ms"
        f"{'-':>15}"
        f"{'-':>12}"
    )
    print(
        f"{'precompute (lookup)':<25}"
        f"{t_pre_first*1e3:>12.1f} ms"
        f"{t_pre_med*1e3:>12.3f} ms"
        f"{t_eager_med/t_pre_med:>12.1f}x"
    )
    print()

    # ──────────────────────────────────────────────────────────────────
    # Mode 3b: multi-axis precompute — vary (l5100, plslope) jointly.
    # ──────────────────────────────────────────────────────────────────
    recipe2d = Recipe.from_selectors(
        disc="grahsp_sbpl",
        lines="grahsp",
        feii="grahsp",
        torus="skirtor",
        attenuation="smc_prevot",
        axis_params=("agn_grahsp_l5100", "agn_grahsp_plslope"),
    )
    t_build2 = time.time()
    pre2 = precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe2d,
        axis_grids={
            "agn_grahsp_l5100": np.logspace(43, 46, 5),
            "agn_grahsp_plslope": np.linspace(-2.5, -1.0, 5),
        },
    )
    fn2 = build_lookup(pre2)
    t_build2 = time.time() - t_build2
    t_pre2_first, t_pre2_med = _time(
        lambda: fn2(jnp.array(1.0), jnp.array(1.0e44), jnp.array(-1.7))
    )

    print("multi-axis (2D: l5100 × plslope, 5×5 grid)")
    print("─" * 67)
    print(
        f"{'precompute (build)':<25}"
        f"{t_build2*1e3:>12.1f} ms"
        f"{'-':>15}"
        f"{'-':>12}"
    )
    print(
        f"{'precompute (lookup)':<25}"
        f"{t_pre2_first*1e3:>12.1f} ms"
        f"{t_pre2_med*1e3:>12.3f} ms"
        f"{t_eager_med/t_pre2_med:>12.1f}x"
    )
    print()


if __name__ == "__main__":
    main()
