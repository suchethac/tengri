# SPDX-License-Identifier: BSD-3-Clause
"""MGVI vs HMC on identical galaxies, both initialized from the same MAP.

Why this probe exists. The two-step estimator needs per-galaxy posterior SAMPLES,
and HMC costs ~120 s per galaxy, which is what makes an N-sweep to N=500
expensive. MGVI is far cheaper. If its per-galaxy marginals agree with HMC's, the
sweep becomes affordable; if they do not, we learn that before trusting a cheap
result.

Two cautions this probe is designed around:

* ``native_vi_linear`` (pure-JAX MGVI) is registered ``tier="broken"`` and its own
  entry says it segfaults on DPL photometry mocks — exactly this model. This probe
  therefore uses ``vi_linear``, the NIFTy MGVI, which the broken entry points to.
* MGVI is a Gaussian variational family and is known in this codebase to
  UNDERESTIMATE variance on the correlated-field geometry. So the interesting
  comparison is not "does the median agree" but "does the WIDTH agree". A
  narrower MGVI interval at the same median is the documented failure, not a win.

Vary one thing at a time: both arms start from the same MAP, on the same galaxy,
with the same data.

Run with the worktree on the path::

  PYTHONPATH=<worktree>/src JAX_PLATFORMS=cpu python scripts/mgvi_vs_hmc_probe.py
"""

from __future__ import annotations

import time

import jax
import numpy as np

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, load_ssp_data
from tengri.analysis.population_mocks import assert_truth_against_model, make_population
from tengri.inference.fitter import Fitter

TRUTH_SIGMA = 0.75
TRUTH_TAU_MYR = 150.0
N_GAL = 2
N_GRID = 16
SNR_PHOT = 20.0
SNR_LINE = 10.0
BANDS = [
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
SHARED = ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr")


def _quantiles(post, name):
    """Median and 68% interval for one parameter, or None if absent."""
    s = getattr(post, "samples", None)
    if not s or name not in s:
        return None
    v = np.asarray(s[name]).ravel()
    if v.size == 0 or not np.all(np.isfinite(v)):
        return None
    lo, med, hi = np.percentile(v, [16, 50, 84])
    return float(med), float(lo), float(hi)


def main():
    import tengri

    print("tengri:", tengri.__file__)
    if "worktrees/hierarchical-psd-spec" not in tengri.__file__:
        raise SystemExit("WRONG CHECKOUT — set PYTHONPATH=<worktree>/src")

    ssp = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    obs = Observation(photometry=Photometry.from_names(BANDS))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": ["dpl", "field"], "*": FREE, "age_gyr": 11.0},
        dust_attenuation={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.1),
        n_grid=N_GRID,
    )
    for nm, val in (
        ("sfh_field_psd_sigma", TRUTH_SIGMA),
        ("sfh_field_psd_tau_myr", TRUTH_TAU_MYR),
    ):
        b = assert_truth_against_model(model, nm, val)
        print(f"  {nm}: truth {val} inside model prior {b}")

    pop = make_population(
        model,
        n_galaxies=N_GAL,
        sigma_true=TRUTH_SIGMA,
        tau_true_myr=TRUTH_TAU_MYR,
        key=jax.random.PRNGKey(0),
        snr_phot=SNR_PHOT,
        snr_line=SNR_LINE,
    )
    flux = np.asarray(pop.table["phot_flux_obs"])
    err = np.asarray(pop.table["phot_flux_err"])

    for i in range(N_GAL):
        print(f"\n{'=' * 70}\nGALAXY {i}\n{'=' * 70}")
        fitter = Fitter(model, flux[i], err[i])

        t0 = time.time()
        rmap = fitter.run("map", key=jax.random.PRNGKey(10 + i), n_steps=12000, n_restarts=3)
        t_map = time.time() - t0
        print(
            f"MAP: {t_map:.1f}s  sigma={float(rmap.params['sfh_field_psd_sigma']):.3f} "
            f"tau={float(rmap.params['sfh_field_psd_tau_myr']):.1f}"
        )

        for method, kwargs in (
            ("vi_linear", {"n_iterations": 20}),
            (
                "mcmc_hmc",
                {
                    "n_warmup": 1000,
                    "n_samples": 1000,
                    "n_leapfrog_steps": 100,
                    "dense_mass_matrix": True,
                },
            ),
        ):
            t0 = time.time()
            try:
                post = fitter.run(method, key=jax.random.PRNGKey(20 + i), init_from=rmap, **kwargs)
                dt = time.time() - t0
                bits = []
                for nm, truth in zip(SHARED, (TRUTH_SIGMA, TRUTH_TAU_MYR)):
                    q = _quantiles(post, nm)
                    if q is None:
                        bits.append(f"{nm}: no samples")
                    else:
                        med, lo, hi = q
                        bits.append(
                            f"{nm.split('_')[-1]}={med:.3f} [{lo:.3f},{hi:.3f}] "
                            f"w={hi - lo:.3f} truth={truth}"
                        )
                print(f"{method:10s}: {dt:6.1f}s  " + "  |  ".join(bits))
            except Exception as exc:
                print(
                    f"{method:10s}: FAILED after {time.time() - t0:.1f}s — "
                    f"{type(exc).__name__}: {str(exc)[:160]}"
                )

    print(
        "\nRead the WIDTHS, not just the medians. MGVI narrower than HMC at the "
        "same median is the documented variance underestimate, not agreement."
    )


if __name__ == "__main__":
    main()
