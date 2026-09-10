"""Measure whether widening dust_bump_strength's free_prior costs sampler quality.

#2226 widened the ``dust_bump_strength`` ceiling ``Uniform(0.0, 2.0)`` ->
``Uniform(0.0, 4.0)`` so a ``kriek_conroy`` fit with ``dust_bump_strength: FREE``
can reach the Narayanan+2018 MUFASA-fitted bump multipliers (up to 3.634 at
z=4). A wider support can, in principle, hurt HMC geometry or the ``hmc_is``
importance-sampling proposal fit. This fits the SAME mock galaxy under both
ceilings, several seeds each, and reports min-ESS / max R-hat / wall time /
posterior coverage so the two are directly comparable.

Two things a first version of this script got wrong, both fixed here:

1. It built the model through the flat ``Parameters(...)`` escape hatch
   passed straight to ``SEDModel(...)``. That path carries no
   ``_group_provenance``, so ``SEDModel._requested_law_shape_params`` (#1808)
   reads every ``dust_*`` shape parameter as ``registry_default`` and never
   adds it to ``live_shape_params`` -- ``dust_bump_strength`` was silently
   never read by ``kriek_conroy`` at all, and the posterior was exactly the
   prior. Building through ``parse_groups`` (or ``SEDModel.build``) records
   ``user_prior`` provenance and the parameter genuinely reaches the law.
2. At z=1 with only broad top-hat bands the observed-frame bump (rest 2175 A,
   FWHM 350 A) was diluted below a measurable level anyway. Real GALEX NUV
   at low z sits squarely on the bump instead.

A :func:`_sensitivity_control` runs BEFORE any fit and hard-fails
(``SystemExit``) if no band moves by more than 5% between
``dust_bump_strength=0`` and the true value, so this script cannot silently
repeat that mistake.
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from tengri import Fitter, Fixed, Observation, Photometry, SEDModel, SSPData, Uniform, parse_groups
from tengri.components.dust.attenuation import _NARAYANAN_BUMP_STRENGTH

#: Low z puts the observed-frame 2175 A bump (rest * (1+z)) inside GALEX NUV.
_Z = 0.075
_FILTER_NAMES = (
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
)
_TRUE_PARAMS = {
    "sfh_dpl_alpha": 1.0,
    "sfh_dpl_beta": 1.0,
    "sfh_dpl_tau_gyr": 2.0,
    "sfh_dpl_log_total_mass": 10.5,
    "met_logzsol": -0.2,
    "dust_tau_v": 0.6,
    "dust_delta": -0.3,
    "dust_bump_strength": 3.3,
    "redshift": _Z,
}
_CEILINGS = (2.0, 4.0)
_SNR = 50.0
_SENSITIVITY_FLOOR = 0.05


def _build_ssp() -> SSPData:
    """Synthetic UV-far-IR SSP grid -- same construction as synthetic_ssp_wide."""
    ages_gyr = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 7.0, 1600)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet
    )


def _build_spec(ceiling: float):
    """Group-form spec: ``dust_bump_strength``/``dust_delta`` are ``user_prior``
    provenance, so ``kriek_conroy`` genuinely reads them (unlike the flat
    ``Parameters(...)`` escape hatch -- see module docstring point 1)."""
    return parse_groups(
        sfh={
            "type": "dpl",
            "alpha": Uniform(0.1, 3.0),
            "beta": Uniform(0.1, 3.0),
            "tau_gyr": Uniform(0.1, 10.0),
            "age_gyr": Fixed(5.0),  # << age_of_universe(z=0.075) ~ 13.7 Gyr
            "log_total_mass": Uniform(9.0, 11.5),
        },
        met={"logzsol": Uniform(-1.5, 0.3)},
        dust_attenuation={
            "type": "single_component",
            "law": "kriek_conroy",
            "tau_v": Uniform(0.0, 2.0),
            "delta": Uniform(-1.0, 0.4),
            "bump_strength": Uniform(0.0, ceiling),
        },
        redshift=Fixed(_Z),
    )


def _sensitivity_control(model) -> None:
    """Evaluate photometry at bump=0 and bump=truth, all else at truth.

    Raises ``SystemExit`` if no band moves by more than ``_SENSITIVITY_FLOOR``:
    a ceiling comparison is meaningless if the likelihood cannot see the
    parameter in the first place.
    """
    off = {**_TRUE_PARAMS, "dust_bump_strength": 0.0}
    on = dict(_TRUE_PARAMS)
    f_off = np.asarray(model.predict_photometry(off))
    f_on = np.asarray(model.predict_photometry(on))
    rel = np.abs(f_on - f_off) / np.abs(f_off)

    print(f"Sensitivity control (dust_bump_strength 0.0 -> {_TRUE_PARAMS['dust_bump_strength']}):")
    for name, r in zip(_FILTER_NAMES, rel):
        print(f"    {name:>10}: {r * 100:6.2f}% relative change")
    max_rel = float(np.max(rel))
    print(f"  max relative change across bands: {max_rel:.1%}\n")
    if max_rel < _SENSITIVITY_FLOOR:
        raise SystemExit(
            f"Sensitivity control FAILED: max relative change {max_rel:.2%} is below "
            f"the {_SENSITIVITY_FLOOR:.0%} floor -- the likelihood is flat in the bump "
            "direction for this fixture, so a ceiling comparison would measure the "
            "prior, not the data. Widen the filter set, lower z further, or raise S/N."
        )


def _run_one_seed(ceiling, seed, flux_obs, noise, ssp, obs, hmc_kwargs):
    spec = _build_spec(ceiling)
    model = SEDModel(spec, ssp, observation=obs)
    fitter = Fitter(model, flux_obs, noise)
    key = jr.PRNGKey(seed)
    t0 = time.perf_counter()
    posterior = fitter.run(method="hmc_is", key=key, **hmc_kwargs)
    wall = time.perf_counter() - t0
    ess = posterior.effective_sample_size()
    rhat = posterior.rhat()
    bump = np.asarray(posterior.samples["dust_bump_strength"])
    p16, p84 = (float(x) for x in np.percentile(bump, [16.0, 84.0]))
    truth = _TRUE_PARAMS["dust_bump_strength"]
    return {
        "wall_s": wall,
        "min_ess": min(ess.values()),
        "max_rhat": max(rhat.values()),
        "bump_median": float(np.median(bump)),
        "p16": p16,
        "p84": p84,
        "covers_truth": p16 <= truth <= p84,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--quick", action="store_true", help="fewer HMC warmup/samples")
    args = parser.parse_args()

    hmc_kwargs = (
        {"n_warmup": 150, "n_burnin": 50, "n_samples": 400}
        if args.quick
        else {"n_warmup": 300, "n_burnin": 100, "n_samples": 1000}
    )

    ssp = _build_ssp()
    obs = Observation(photometry=Photometry.from_names(list(_FILTER_NAMES)))
    true_model = SEDModel(_build_spec(_CEILINGS[-1]), ssp, observation=obs)

    print(f"z={_Z}, S/N={_SNR}, filters={_FILTER_NAMES}")
    print(f"free parameters: {sorted(true_model.spec.free_params)}")
    print(f"true params: {_TRUE_PARAMS}")
    print(f"max fitted MUFASA node: {float(max(_NARAYANAN_BUMP_STRENGTH)):.3f}\n")

    _sensitivity_control(true_model)

    mock = true_model.mock(_TRUE_PARAMS, snr=_SNR, key=jr.PRNGKey(0))
    print(f"hmc_is kwargs: {hmc_kwargs}, seeds: {args.seeds}\n")

    for ceiling in _CEILINGS:
        print(f"=== ceiling Uniform(0.0, {ceiling}) ===")
        header = f"{'seed':>5} {'min_ess':>10} {'max_rhat':>10} {'wall_s':>8} "
        header += f"{'bump_med':>10} {'[p16,p84]':>18} {'truth_in68':>11}"
        print(header)
        rows = []
        for seed in range(args.seeds):
            row = _run_one_seed(ceiling, seed, mock.flux_obs, mock.noise, ssp, obs, hmc_kwargs)
            rows.append(row)
            interval = f"[{row['p16']:.2f},{row['p84']:.2f}]"
            print(
                f"{seed:>5} {row['min_ess']:>10.1f} {row['max_rhat']:>10.4f} "
                f"{row['wall_s']:>8.2f} {row['bump_median']:>10.3f} {interval:>18} "
                f"{row['covers_truth']!s:>11}"
            )
        worst_ess = min(r["min_ess"] for r in rows)
        worst_rhat = max(r["max_rhat"] for r in rows)
        total_wall = sum(r["wall_s"] for r in rows)
        n_covering = sum(r["covers_truth"] for r in rows)
        print(
            f"  SUMMARY ceiling={ceiling}: worst-seed min_ess={worst_ess:.1f}, "
            f"worst-seed max_rhat={worst_rhat:.4f}, truth covered in "
            f"{n_covering}/{len(rows)} seeds' 68% interval, total wall={total_wall:.1f}s\n"
        )


if __name__ == "__main__":
    main()
