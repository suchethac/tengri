# SPDX-License-Identifier: BSD-3-Clause
"""Dimension-scaling sweep for stochastic field-SFH recovery.

Companion to ``notebooks/stochastic_sfh_recovery.py``. For a ladder of SFH
dimensions ``n_grid`` and a set of inference backends, this fits the first
``n_fit`` star-forming mock galaxies and reports, per (n_grid, method):

* SFH 68 % / 95 % credible-band coverage of the injected history,
* the ``sfh_field_psd_sigma`` (burst amplitude) bias, and
* the median wall time.

Because the field truth ``sfh_field_xi`` has length ``n_grid``, the catalog is
regenerated **self-consistently at each dimension** — galaxy ``i`` is the same
star-forming backbone with an ``n_grid``-appropriate field realization — so the
sweep measures recovery *at* each dimension, not a fixed-truth mismatch.

One fit per process is the memory-safe unit; this loops sequentially and calls
``tengri.clear_cache()`` between fits. Run it under the OOM watchdog::

    LIMIT_GB=20 scripts/run_with_oom_monitor.sh -- \
        .venv/bin/python scripts/stochastic_sfh_dimension_scaling.py \
            --n-grid 16 32 64 128 --n-fit 4 --methods mcmc_hmc vi_nonlinear_fast mcmc_raytrace

The heavy trees are not exercised by the PR-gating test tier; this is an
operator script, not a test.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
# age_gyr=12 at z=0.1 forms ~1% of mass just before the Big Bang (truncated); benign here.
warnings.filterwarnings("ignore", message=r".*before the Big Bang.*")

import tengri
from tengri import (
    FREE,
    Fitter,
    Fixed,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
    WavePrecomp,
    builders,
    load_ssp_data,
)
from tengri.observation import LineFluxData
from tengri.observation.line_measurement import default_line_defs

# ── Fixed experiment definition (mirrors the notebook) ─────────────────────
Z_SPEC = 0.1
PHOT_BANDS = [
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
]
LINE_NAMES = [
    "Halpha",
    "Hbeta",
    "OIII_5007",
    "OIII_4959",
    "SII_6717",
    "SII_6731",
    "OII_3726",
    "OII_3729",
]
DPL_TEMPLATE = {
    "sfh_dpl_alpha": 2.0,
    "sfh_dpl_beta": 1.5,
    "sfh_dpl_age_gyr": 12.0,
    "sfh_dpl_tau_gyr": 13.0,
}
MET_FIXED = -0.3
SIGMA_TRUTH_KEY = "sfh_field_psd_sigma"
# Star-forming population prior (mirrors the notebook): physically-typical modest
# dust and moderate burstiness so per-galaxy recovery is not dominated by the
# heavy-dust age/SFR degeneracy the wide fitting prior would otherwise inject.
DUST_BC_RANGE = (0.1, 0.5)
DUST_DIFF_RANGE = (0.05, 0.35)
SIGMA_RANGE = (0.1, 0.5)
TAU_MYR_RANGE = (50.0, 400.0)
SFR0_RANGE = (3.0, 40.0)


def _present(t_gyr: np.ndarray) -> int:
    return int(np.argmin(np.asarray(t_gyr)))


def make_builder(ssp_data, phot, noise_model, line_template, n_grid):
    """Return a ``build(line_fluxes)`` closure at a fixed ``n_grid``."""

    def build(line_fluxes):
        return SEDModel.build(
            ssp_data=ssp_data,
            observation=Observation(photometry=phot, line_fluxes=line_fluxes, noise=noise_model),
            sfh={"type": ["dpl", "field"], "*": FREE},
            met={"logzsol": Fixed(MET_FIXED)},
            dust=builders.dust.two_component(defaults=FREE, law_bc="calzetti"),
            neb=builders.neb.ssp(),
            redshift=Fixed(Z_SPEC),
            apply_igm=False,
            n_grid=n_grid,
            approx=WavePrecomp(),
        )

    return build


def draw_truth(model, spec, fixed_values):
    """Return a ``draw_truth(seed)`` bound to this ``n_grid`` model."""

    def _draw(seed):
        k_xi, k_bc, k_diff, k_sig, k_tau, k_sfr = jax.random.split(jax.random.PRNGKey(seed), 6)
        drawn = spec.sample(k_xi)  # fresh field xi ~ N(0, I)

        def _u(k, lo, hi):
            return float(jax.random.uniform(k, minval=lo, maxval=hi))

        truth = {
            **drawn,
            **{k: jnp.array(v) for k, v in DPL_TEMPLATE.items()},
            "sfh_dpl_log_total_mass": jnp.array(11.0),
            "met_logzsol": jnp.array(MET_FIXED),
            "dust_tau_bc": jnp.array(_u(k_bc, *DUST_BC_RANGE)),
            "dust_tau_diff": jnp.array(_u(k_diff, *DUST_DIFF_RANGE)),
            "sfh_field_psd_sigma": jnp.array(_u(k_sig, *SIGMA_RANGE)),
            "sfh_field_psd_tau_myr": jnp.array(_u(k_tau, *TAU_MYR_RANGE)),
        }
        target = float(10.0 ** _u(k_sfr, np.log10(SFR0_RANGE[0]), np.log10(SFR0_RANGE[1])))
        sfh = model.predict_sfh({**fixed_values, **truth})
        sfr0 = float(np.asarray(sfh["sfr_mean"])[_present(sfh["t_gyr"])])
        truth["sfh_dpl_log_total_mass"] = jnp.array(11.0 + float(np.log10(target / sfr0)))
        return truth

    return _draw


def synthesize(model, line_defs, fixed_values, truth, seed, phot_snr, line_snr):
    truth_full = {**fixed_values, **truth}
    mp = model.mock(truth_full, snr=phot_snr, key=jax.random.PRNGKey(seed + 10_000))
    flux_phot, noise_phot = np.asarray(mp.flux_obs), np.asarray(mp.noise)
    lf_true = np.asarray(model.measure_line_fluxes(truth_full, line_defs, fast=False))
    lf_err = np.abs(lf_true) / line_snr
    lf_obs = lf_true + lf_err * np.random.default_rng(seed + 20_000).standard_normal(lf_true.shape)
    return flux_phot, noise_phot, lf_obs, lf_err, lf_true


def run_backend(fitter, method, res_map, key, dim):
    """Dispatch one approximate/exact backend from a shared MAP init."""
    if method == "mcmc_nuts":
        return fitter.run(
            method="mcmc_nuts",
            init_from=res_map,
            n_warmup=800,
            n_samples=500,
            dense_mass_matrix=(dim <= 40),
            key=key,
            verbose=False,
        )
    if method == "mcmc_hmc":
        # L=100: a long trajectory is required on this ill-conditioned field
        # geometry — a short one (blackjax default L=10) gets stuck near the mode
        # and returns overconfident, non-covering bands.
        return fitter.run(
            method="mcmc_hmc",
            init_from=res_map,
            n_warmup=500,
            n_samples=400,
            n_leapfrog_steps=100,
            dense_mass_matrix=(dim <= 40),
            key=key,
            verbose=False,
        )
    if method == "vi_nonlinear_fast":
        return fitter.run(
            method="vi_nonlinear_fast",
            init_from=res_map,
            n_iterations=25,
            n_samples=3,
            n_posterior_samples=800,
            key=key,
            verbose=False,
        )
    if method == "mcmc_raytrace":
        return fitter.run(
            method="mcmc_raytrace",
            init_from=res_map,
            step_size=0.05,
            n_samples=500,
            key=key,
            verbose=False,
        )
    raise ValueError(f"unknown method {method!r}")


def coverage(model_fit, samples, fixed_values, sfr_true):
    """Fraction of lookback bins where the true SFH lies in the 68 %/95 % bands."""
    n = int(next(iter(samples.values())).shape[0])

    def _p(i):
        d = {k: (float(v[i]) if v.ndim == 1 else np.asarray(v[i])) for k, v in samples.items()}
        return {**fixed_values, **d}

    draws = np.array([np.asarray(model_fit.predict_sfh(_p(i))["sfr_full"]) for i in range(n)])
    lo68, hi68 = np.percentile(draws, [16, 84], axis=0)
    lo95, hi95 = np.percentile(draws, [2.5, 97.5], axis=0)
    return (
        float(np.mean((sfr_true >= lo68) & (sfr_true <= hi68))),
        float(np.mean((sfr_true >= lo95) & (sfr_true <= hi95))),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-grid", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--n-fit", type=int, default=4)
    # HMC (L=100) is the practical honest reference (~9 min/fit at D=25); NUTS gives
    # the same recovery but ~8x slower, so pass --methods mcmc_nuts ... only for a
    # gold-standard cross-check on a few galaxies.
    ap.add_argument(
        "--methods", nargs="+", default=["mcmc_hmc", "vi_nonlinear_fast", "mcmc_raytrace"]
    )
    ap.add_argument("--ssp", default="data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    ap.add_argument("--phot-snr", type=float, default=20.0)
    ap.add_argument("--line-snr", type=float, default=10.0)
    ap.add_argument("--map-steps", type=int, default=3000)
    ap.add_argument("--map-restarts", type=int, default=8)
    ap.add_argument("--out", default="figures/stochastic_dimension_scaling.json")
    args = ap.parse_args()

    ssp_data = load_ssp_data(args.ssp)
    phot = Photometry.from_names(PHOT_BANDS)
    noise_model = NoiseModel(calibration_floor=0.01, student_t_dof=None)
    line_template = LineFluxData.from_dict({nm: (1e-16, 1e-17) for nm in LINE_NAMES})
    line_defs = default_line_defs(
        np.asarray(line_template.wavelengths), tuple(line_template.names)
    )

    rows = []
    for n_grid in args.n_grid:
        build = make_builder(ssp_data, phot, noise_model, line_template, n_grid)
        base = build(line_template)
        spec = base.spec
        fixed_values = spec.get_fixed_values()
        dim = spec.n_free + spec.n_grid
        draw = draw_truth(base, spec, fixed_values)
        print(f"\n=== n_grid={n_grid}  D={dim} ===", flush=True)

        accepted, seed = 0, 0
        while accepted < args.n_fit and seed < 500:
            truth = draw(seed)
            truth_full = {**fixed_values, **truth}
            fp, npn, lo, le, lt = synthesize(
                base, line_defs, fixed_values, truth, seed, args.phot_snr, args.line_snr
            )
            seed += 1
            if lt[LINE_NAMES.index("Halpha")] <= 0:
                continue  # star-forming, line-emitting catalog: skip net-absorption draws
            i, accepted = accepted, accepted + 1
            sfr_true = np.asarray(base.predict_sfh(truth_full)["sfr_full"])
            sig_true = float(truth[SIGMA_TRUTH_KEY])
            lfd = LineFluxData(
                names=tuple(LINE_NAMES),
                fluxes=jnp.asarray(lo),
                errors=jnp.asarray(le),
                wavelengths=line_template.wavelengths,
            )
            model_fit = build(lfd)
            fitter = Fitter(model_fit, fp, npn)
            res_map = fitter.run(
                method="map",
                n_steps=args.map_steps,
                n_restarts=args.map_restarts,
                key=jax.random.PRNGKey(i),
                verbose=False,
            )

            for m in args.methods:
                t0 = time.perf_counter()
                try:
                    res = run_backend(fitter, m, res_map, jax.random.PRNGKey(1000 + i), dim)
                    wall = time.perf_counter() - t0
                    c68, c95 = coverage(model_fit, res.samples, fixed_values, sfr_true)
                    sig = np.asarray(res.samples[SIGMA_TRUTH_KEY])
                    row = dict(
                        n_grid=n_grid,
                        dim=dim,
                        gal=i,
                        method=m,
                        wall=wall,
                        cov68=c68,
                        cov95=c95,
                        sigma_true=sig_true,
                        sigma_p50=float(np.median(sig)),
                        sigma_bias=float(np.median(sig) - sig_true),
                        ok=True,
                    )
                except Exception as e:
                    row = dict(
                        n_grid=n_grid, dim=dim, gal=i, method=m, ok=False, error=repr(e)[:200]
                    )
                    print(f"  [SKIP] n_grid={n_grid} gal={i} {m}: {row['error']}", flush=True)
                rows.append(row)
                if row.get("ok"):
                    print(
                        f"  gal={i} {m:18s} D={dim:3d} {row['wall']:6.1f}s "
                        f"cov68={row['cov68']:.2f} cov95={row['cov95']:.2f} "
                        f"sig_bias={row['sigma_bias']:+.3f}",
                        flush=True,
                    )
                tengri.clear_cache()  # keep the JAX compile cache from ballooning across fits

    # ── Aggregate table ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(
        f"{'n_grid':>6} {'D':>4} {'method':18s} {'cov68':>6} {'cov95':>6} "
        f"{'sig_bias':>9} {'wall_s':>7} {'n':>3}"
    )
    print("-" * 78)
    for n_grid in args.n_grid:
        for m in args.methods:
            sel = [r for r in rows if r.get("ok") and r["n_grid"] == n_grid and r["method"] == m]
            if not sel:
                print(f"{n_grid:>6} {'':>4} {m:18s} {'—':>6} {'—':>6} {'—':>9} {'—':>7}   0")
                continue
            dim = sel[0]["dim"]
            print(
                f"{n_grid:>6} {dim:>4} {m:18s} "
                f"{np.mean([r['cov68'] for r in sel]):6.2f} "
                f"{np.mean([r['cov95'] for r in sel]):6.2f} "
                f"{np.median([r['sigma_bias'] for r in sel]):+9.3f} "
                f"{np.median([r['wall'] for r in sel]):7.1f} {len(sel):>3}"
            )
    print("=" * 78)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
