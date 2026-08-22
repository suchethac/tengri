# SPDX-License-Identifier: BSD-3-Clause
"""Field-SFH recovery study: what observables constrain a bursty star-formation history.

Companion to ``notebooks/stochastic_sfh_recovery.py``. The notebook demonstrates
recovery on ONE galaxy; this script runs the population-level experiments that
support its conclusions, and writes results the notebook then loads and plots.
Separate because the full study is 2-3 CPU-hours -- far past what belongs inside
a notebook execution.

Three stages, selectable with ``--stage``:

``cells``
    18 HMC recovery cells: 2 observables (photometry, +8 line fluxes) x 3
    burstiness amplitudes x 3 INDEPENDENT realizations. Coverage is a frequentist
    property over data realizations, so each cell draws its own truth, its own
    photometric noise and its own line noise -- pinning the truth and varying only
    the sampler RNG measures sampler stability, not recovery.

``paired``
    The same photometry-vs-lines contrast at n=15 via MAP rather than HMC. The
    design is PAIRED (both arms see an identical truth, identical photometry and
    identical noise; only the observable differs), so the statistic is the
    per-realization DIFFERENCE. At n=3 the marginal scatter -- dominated by how
    bursty each drawn truth happens to be -- swamps a real effect: the 18-cell
    study appeared to show lines helping only at high sigma, which n=15 shows is
    noise. MAP costs ~12 s against HMC's ~4 min, which is what buys the n.

``spectrum``
    Adds a third arm, a full optical spectrum, and scores two age windows
    separately: young (<15 Myr), where emission lines act, and intermediate
    (15 Myr - 1 Gyr), where the 4000 A break and Balmer absorption should. If the
    spectrum only wins in the young window it is re-measuring the lines; if it
    wins in the intermediate one, the continuum is carrying information the line
    fluxes discard.

All scoring is on the model's NATIVE log-age grid
(``predict_sfh(..., grid="native")``). ``predict_sfh``'s default resamples onto a
uniform LINEAR time grid whose step is ``age_max / n_linear`` = 13.8 Myr at the
defaults, so of 37 samples inside the recent 0.5 Gyr only 2 land below 15 Myr,
while 5 of the 16 log-age nodes do. Scoring there weights every megayear equally
and 15-500 Myr swamps the young bins -- which turned a real ~60% improvement from
line fluxes into an apparent 0%.

Usage
-----
    python scripts/field_sfh_recovery_study.py --stage paired
    python scripts/field_sfh_recovery_study.py --stage spectrum --n-seeds 15
    python scripts/field_sfh_recovery_study.py --stage cells        # slowest

Results land in ``figures/field_sfh_study/`` as ``.npz``; the notebook's Stage-2
section loads whatever is present and skips what is not.
"""

from __future__ import annotations

import argparse
import os
import warnings
from math import comb
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore", message=r".*before the Big Bang.*")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from tengri import (
    FREE,
    FeaturePrecomp,
    Fixed,
    ForwardModel,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    SpectrumPrecomp,
    WavePrecomp,
    builders,
    load_ssp_data,
)
from tengri.observation import LineFluxData
from tengri.observation.line_measurement import default_line_defs

SSP_NAME = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0"
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
# Rising, still-star-forming backbone -- not quiescent.
DPL = {"sfh_dpl_alpha": 2.0, "sfh_dpl_beta": 1.5, "sfh_dpl_age_gyr": 12.0, "sfh_dpl_tau_gyr": 13.0}
TAU_MYR, Z_GAL, N_GRID = 120.0, 0.1, 16
SIGMAS = (0.2, 0.4, 0.6)
YOUNG_GYR, MID_GYR = 0.015, 1.0
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 260)  # rest 3455-8364 A at z=0.1

OUTDIR = Path(__file__).resolve().parents[1] / "figures" / "field_sfh_study"


def _ssp():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / f"{SSP_NAME}.h5"
    if not path.exists():
        import tengri

        path = Path(tengri.download_ssp(SSP_NAME))
    return load_ssp_data(str(path))


def build(ssp, observation):
    """Model for one observable configuration, with the right precompute."""
    if getattr(observation, "spectroscopy", None) is not None:
        approx = SpectrumPrecomp()
    elif getattr(observation, "has_line_fluxes", False):
        approx = (WavePrecomp(), FeaturePrecomp())
    else:
        approx = WavePrecomp()
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": ["dpl", "field"], "all_params": FREE},
        met={"logzsol": Fixed(-0.3)},
        dust_attenuation=builders.dust.two_component(defaults=FREE, law="calzetti"),
        neb=builders.neb.ssp(),
        redshift=Fixed(Z_GAL),
        igm={"type": "none"},
        n_grid=N_GRID,
        approx=approx,
    )


def sfr_on_nodes(model, params):
    """SFR at the native log-age nodes -- NOT the linear resampling."""
    return np.asarray(model.predict_sfh(params, grid="native")["sfr_full"])


def make_truth(ssp, seed, sigma, noise_model, line_template):
    """One realization: truth, photometry, line fluxes. Shared by every arm."""
    model = build(
        ssp,
        Observation(
            photometry=Photometry.from_names(PHOT_BANDS),
            line_fluxes=line_template,
            noise=noise_model,
        ),
    )
    fixed = model.spec.get_fixed_values()
    truth = {
        **model.spec.sample(jax.random.PRNGKey(seed)),
        **{k: jnp.array(v) for k, v in DPL.items()},
        "met_logzsol": jnp.array(-0.3),
        "dust_tau_bc": jnp.array(0.3),
        "dust_tau_diff": jnp.array(0.15),
        "sfh_field_psd_sigma": jnp.array(sigma),
        "sfh_field_psd_tau_myr": jnp.array(TAU_MYR),
        "sfh_dpl_log_total_mass": jnp.array(11.0),
    }
    sfh = model.predict_sfh({**fixed, **truth})
    idx = int(np.argmin(np.asarray(sfh["t_gyr"])))
    truth["sfh_dpl_log_total_mass"] = jnp.array(
        11.0 + np.log10(20.0 / float(np.asarray(sfh["sfr_mean"])[idx]))
    )
    tf = {**fixed, **truth}

    mock = model.mock(tf, snr=20.0, key=jax.random.PRNGKey(seed + 10_000))
    flux, noise = np.asarray(mock.flux_obs), np.asarray(mock.noise)

    defs = default_line_defs(np.asarray(line_template.wavelengths), tuple(line_template.names))
    lf_true = np.asarray(model.measure_line_fluxes(tf, defs, fast=True))
    if lf_true[0] <= 0:
        # Halpha in ABSORPTION -- physically real at high sigma (a bursty history
        # in a current lull), but such a galaxy would never enter a line-flux
        # sample, so it cannot inform a phot-vs-lines contrast. The caller counts
        # these; dropping them silently would bias survivors toward line-bright.
        return None
    lf_err = np.abs(lf_true) / 10.0
    rng = np.random.default_rng(seed + 20_000)
    lines = LineFluxData(
        names=tuple(LINE_NAMES),
        fluxes=jnp.asarray(lf_true + lf_err * rng.standard_normal(lf_true.shape)),
        errors=jnp.asarray(lf_err),
        wavelengths=line_template.wavelengths,
    )
    t_nod = np.asarray(model.predict_sfh(tf, grid="native")["t_gyr"])
    return dict(
        model=model,
        fixed=fixed,
        tf=tf,
        flux=flux,
        noise=noise,
        lines=lines,
        t_nod=t_nod,
        sfr_true=sfr_on_nodes(model, tf),
    )


def _fit(model, observation, data, noise, seed, data_type=None, **kw):
    """MAP through the canonical ForwardModel surface (Fitter(SEDModel) is deprecated)."""
    forward = ForwardModel.build(sed=model, observation=observation)
    extra = {"data_type": data_type} if data_type else {}
    return forward.fit(
        data,
        noise,
        method="map",
        n_steps=10_000,
        n_restarts=6,
        key=jax.random.PRNGKey(seed),
        verbose=False,
        **extra,
        **kw,
    )


def _dex(pred, truth, mask):
    return float(
        np.sqrt(
            np.mean(
                (
                    np.log10(np.clip(pred[mask], 1e-8, None))
                    - np.log10(np.clip(truth[mask], 1e-8, None))
                )
                ** 2
            )
        )
    )


def sign_test(k, n):
    """Two-sided exact binomial p. Honest at small n, unlike a t-statistic."""
    return min(1.0, 2 * sum(comb(n, j) for j in range(0, min(k, n - k) + 1)) / 2**n)


def run_paired(ssp, seeds, noise_model, line_template, with_spectrum):
    """Photometry vs +lines (vs +spectrum), paired, scored on the native grid."""
    arms = ("phot", "lines", "spec") if with_spectrum else ("phot", "lines")
    res = {a: {s: [] for s in SIGMAS} for a in arms}
    skipped = {s: [] for s in SIGMAS}
    phot = Photometry.from_names(PHOT_BANDS)

    for sigma in SIGMAS:
        for seed in seeds:
            r = make_truth(ssp, seed, sigma, noise_model, line_template)
            if r is None:
                skipped[sigma].append(seed)
                continue
            young = r["t_nod"] < YOUNG_GYR
            mid = (r["t_nod"] >= YOUNG_GYR) & (r["t_nod"] < MID_GYR)

            cfg = {
                "phot": (
                    Observation(photometry=phot, noise=noise_model),
                    r["flux"],
                    r["noise"],
                    None,
                ),
                "lines": (
                    Observation(photometry=phot, line_fluxes=r["lines"], noise=noise_model),
                    r["flux"],
                    r["noise"],
                    None,
                ),
            }
            if with_spectrum:
                spec_obs = Spectroscopy(wave_obs=WAVE_OBS, resolution=2000)
                m_spec = build(
                    ssp, Observation(photometry=phot, spectroscopy=spec_obs, noise=noise_model)
                )
                p_spec = np.asarray(m_spec.predict_spectrum(r["tf"], wave_obs=WAVE_OBS))
                n_spec = np.abs(p_spec) / 30.0
                f_spec = (
                    p_spec
                    + np.random.default_rng(seed + 30_000).normal(size=p_spec.shape) * n_spec
                )
                cfg["spec"] = (
                    Observation(photometry=phot, spectroscopy=spec_obs, noise=noise_model),
                    np.concatenate([r["flux"], f_spec]),
                    np.concatenate([r["noise"], n_spec]),
                    "joint",
                )

            for arm in arms:
                observation, data, noise, dtype = cfg[arm]
                model = build(ssp, observation)
                post = _fit(model, observation, data, noise, seed, data_type=dtype)
                pred = sfr_on_nodes(model, {**r["fixed"], **post.params})
                res[arm][sigma].append(
                    (_dex(pred, r["sfr_true"], young), _dex(pred, r["sfr_true"], mid))
                )
        print(
            f"  sigma={sigma}: {len(res['phot'][sigma])} realizations, "
            f"{len(skipped[sigma])} skipped (Halpha in absorption)",
            flush=True,
        )
    return res, skipped, arms


def report(res, arms, label, idx):
    print(f"\n{label}", flush=True)
    head = f"{'sigma':>6}" + "".join(f"{a:>10}" for a in arms)
    print(head + f"{'lines-phot':>13}{'improved':>11}{'sign p':>9}", flush=True)
    for sigma in SIGMAS:
        vals = {a: np.array([v[idx] for v in res[a][sigma]]) for a in arms}
        d = vals["phot"] - vals["lines"]
        k, n = int((d > 0).sum()), d.size
        row = f"{sigma:>6}" + "".join(f"{vals[a].mean():>10.3f}" for a in arms)
        print(row + f"{d.mean():>13.3f}{k:>7}/{n:<3}{sign_test(k, n):>9.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["paired", "spectrum", "cells"], default="paired")
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=15,
        help="realizations per sigma (n=3 is NOT enough; see module docstring)",
    )
    ap.add_argument("--seed0", type=int, default=101)
    args = ap.parse_args()

    if args.stage == "cells":
        raise SystemExit(
            "The 18-cell HMC stage is not yet wired into this script. Use --stage paired "
            "or --stage spectrum; both answer the observable-comparison question at higher "
            "n for a fraction of the cost."
        )
    if args.n_seeds < 8:
        print(
            f"WARNING: n={args.n_seeds} realizations. The paired difference this measures "
            f"is ~0.1 dex against ~0.15 dex realization scatter; n=3 produced a spurious "
            f"sigma-dependence that n=15 refuted.",
            flush=True,
        )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ssp = _ssp()
    noise_model = NoiseModel(calibration_floor=0.01, student_t_dof=None)
    line_template = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in LINE_NAMES})
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    with_spectrum = args.stage == "spectrum"

    print(
        f"stage={args.stage}  n={args.n_seeds} realizations per sigma  "
        f"scoring on the native log-age grid",
        flush=True,
    )
    res, skipped, arms = run_paired(ssp, seeds, noise_model, line_template, with_spectrum)

    print("\n" + "=" * 78, flush=True)
    print("PAIRED RECOVERY (MAP, native log-age grid). |dlog10 SFR| [dex]", flush=True)
    print("=" * 78, flush=True)
    report(res, arms, f"YOUNG (< {YOUNG_GYR * 1e3:.0f} Myr) -- where the LINES act", 0)
    if with_spectrum:
        report(
            res,
            arms,
            f"INTERMEDIATE ({YOUNG_GYR * 1e3:.0f} Myr - {MID_GYR:.0f} Gyr) -- "
            "where the CONTINUUM should act",
            1,
        )

    out = OUTDIR / f"paired_{args.stage}_n{args.n_seeds}.npz"
    np.savez(
        out,
        seeds=np.array(seeds),
        sigmas=np.array(SIGMAS),
        arms=np.array(arms),
        **{f"{a}_{s}": np.array(res[a][s]) for a in arms for s in SIGMAS},
        **{f"skipped_{s}": np.array(skipped[s]) for s in SIGMAS},
    )
    print(f"\nwrote {out}", flush=True)
    total_skipped = sum(len(v) for v in skipped.values())
    if total_skipped:
        print(
            f"NOTE: {total_skipped} realization(s) dropped for Halpha in absorption "
            f"(counted, not silently discarded).",
            flush=True,
        )


if __name__ == "__main__":
    main()
