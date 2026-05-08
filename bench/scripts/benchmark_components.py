#!/usr/bin/env python
"""Benchmark all tengri forward model components.

Measures per-call timing for each physics module after JIT warmup.
Useful for identifying bottlenecks and tracking performance regressions.

Usage:
    python scripts/benchmark_components.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

N_WARMUP = 3
N_RUNS = 100


def bench(fn, label, n_warmup=N_WARMUP, n_runs=N_RUNS):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"  {label:<45} {ms:>8.3f} ms")
    return ms


def main():
    print("=" * 60)
    print("tengri Component Benchmarks")
    print(f"  Platform: {jax.devices()[0].platform.upper()}")
    print("  Precision: float64")
    print(f"  Runs: {N_RUNS} (after {N_WARMUP} warmup)")
    print("=" * 60)

    # --- Dust attenuation ---
    print("\nDust Attenuation:")
    from tengri.dust.attenuation import two_component_dust

    wave = jnp.linspace(1000, 30000, 5994)
    ages = jnp.array([1e6, 1e7, 1e8, 1e9, 1e10])
    bench(
        lambda: two_component_dust(
            wave, ages, 1.0, 0.3, law_bc="power_law", law_diff="power_law", n_slope=-0.7
        ),
        "two_component_dust power_law (5994 wave, 5 ages)",
    )

    # --- Dust emission ---
    print("\nDust IR Emission:")
    from tengri.dust.emission import dale2014, draine_li2007, modified_blackbody

    wave_ir = jnp.logspace(np.log10(5000), np.log10(5e6), 1000)
    bench(lambda: modified_blackbody(wave_ir, 1.0, dust_T=30.0), "Modified blackbody (1000 pts)")
    bench(lambda: dale2014(wave_ir, 1.0, dust_alpha_dale=2.0), "Dale+2014 analytic (1000 pts)")
    bench(
        lambda: draine_li2007(wave_ir, 1.0, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5),
        "DL07 analytic (1000 pts)",
    )

    try:
        from tengri.dust.emission import create_dl07_from_grid

        dl07 = create_dl07_from_grid("data/dl07_templates.npz")
        bench(
            lambda: dl07(wave_ir, 1.0, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5),
            "DL07 tabulated (1000 pts)",
        )
    except FileNotFoundError:
        print("  DL07 tabulated: SKIPPED (data/dl07_templates.npz not found)")

    # --- IGM ---
    print("\nIGM Absorption:")
    from tengri.igm import igm_transmission

    wave_obs = jnp.linspace(3000, 15000, 5994)
    bench(lambda: igm_transmission(wave_obs, 3.0), "Inoue+2014 (5994 pts, z=3)")

    # --- Mass remaining ---
    print("\nMass-Remaining Fraction:")
    from tengri.sps.mass_remaining import compute_mass_remaining_fraction

    ages_gyr = jnp.array([0.01, 0.1, 1.0, 5.0, 10.0])
    bench(
        lambda: compute_mass_remaining_fraction(ages_gyr, imf="chabrier"),
        "Internal Chabrier (5 ages, 500 mass pts)",
    )

    # --- CUE nebular ---
    print("\nCUE Nebular Emulator:")
    try:
        from tengri.nebular.cue import CueBackend

        cb = CueBackend("data/cue_weights.npz")
        cue_p = dict(
            ionspec_index1=-1.5,
            ionspec_index2=-3.0,
            ionspec_index3=-1.0,
            ionspec_index4=-2.0,
            ionspec_logLratio1=0.0,
            ionspec_logLratio2=0.0,
            ionspec_logLratio3=-0.5,
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=-0.5,
            gas_logno=-0.5,
            gas_logco=0.0,
        )
        bench(
            lambda: cb.predict_nebular_line_luminosities(cloudyfsps_only=False, **cue_p),
            "CUE lines (128 lines)",
        )
        bench(lambda: cb.predict_nebular_continuum(**cue_p), "CUE continuum")

        def loss_fn(logu):
            _, lum = cb.predict_nebular_line_luminosities(
                cloudyfsps_only=False,
                gas_logu=logu,
                **{k: v for k, v in cue_p.items() if k != "gas_logu"},
            )
            return jnp.sum(lum)

        grad_fn = jax.jit(jax.grad(loss_fn))
        bench(lambda: grad_fn(-2.5), "CUE lines + jax.grad")
    except FileNotFoundError:
        print("  CUE: SKIPPED (data/cue_weights.npz not found)")

    # --- AGN ---
    print("\nAGN Models:")
    from tengri.agn.unified import simple_agn, standard_agn

    wave_agn = jnp.linspace(500, 200000, 5000)
    bench(
        lambda: simple_agn(wave_agn, agn_log_lbol=11.0, agn_frac=1.0),
        "Simple AGN (powerlaw + torus, 5000 pts)",
    )
    bench(
        lambda: standard_agn(wave_agn, agn_log_lbol=11.0, agn_frac=1.0),
        "Standard AGN (multicolor + 2T torus, 5000 pts)",
    )

    # --- Radio ---
    print("\nRadio:")
    from tengri.radio import radio_total

    wave_radio = jnp.logspace(7, 10, 500)
    bench(
        lambda: radio_total(wave_radio, L_ir=1e10, L_agn_bol=1e11),
        "Radio total (SF + AGN, 500 pts)",
    )

    # --- X-ray ---
    print("\nX-ray:")
    from tengri.xray import xray_xrb

    wave_xray = jnp.linspace(0.1, 12.0, 500)
    bench(
        lambda: xray_xrb(wave_xray, sfr=1.0, stellar_mass=1e10),
        "XRB (HMXB + LMXB, 500 pts)",
    )

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
