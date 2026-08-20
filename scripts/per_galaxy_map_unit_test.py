# Per-galaxy MAP unit test — the layer below the population step.
#
# Before trusting any shared posterior, each galaxy must recover its OWN
# parameters. The SFH recovery study's benchmark is chi2/N at the MAP in
# 0.32-1.06 across all 18 cells; anything far outside that means the
# per-galaxy fit is wrong and the population step cannot rescue it.
#
# MAP is also ~20x cheaper than HMC here (6 s vs 121 s per galaxy), which is
# what makes an N-sweep affordable at all.
import time

import jax
import numpy as np

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, load_ssp_data
from tengri.analysis.population_mocks import assert_truth_is_discriminating, make_population
from tengri.inference.fitter import Fitter

TRUTH_SIGMA = 0.75  # interior to the model's real Uniform(0.01, 1.0)
TRUTH_TAU_MYR = 150.0
N_GAL = 4
N_GRID = 16
SNR_PHOT = 20.0
SNR_LINE = 10.0
# NOTE: bounds are READ FROM THE MODEL below, never hardcoded. A previous
# version asserted the truth against (0.1, 4.0) while the model's prior was
# (0.01, 1.0), so sigma_true=1.3 was outside the support and unreachable.
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


def main():
    import tengri

    print("tengri:", tengri.__file__)
    assert "worktrees/hierarchical-psd-spec" in tengri.__file__, "WRONG CHECKOUT"

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
    print(f"model: D={len(model.spec.free_params)} free scalars, n_grid={model.spec.n_grid}")

    # Read the ACTUAL priors off the model and validate the truths against those.
    sigma_bounds = model.spec.get_distribution("sfh_field_psd_sigma").bounds
    tau_bounds = model.spec.get_distribution("sfh_field_psd_tau_myr").bounds
    print(f"model priors: sigma ~ U{sigma_bounds}, tau ~ U{tau_bounds} Myr")
    assert sigma_bounds[0] < TRUTH_SIGMA < sigma_bounds[1], (
        f"sigma truth {TRUTH_SIGMA} is OUTSIDE the model prior {sigma_bounds} — unreachable"
    )
    assert tau_bounds[0] < TRUTH_TAU_MYR < tau_bounds[1], (
        f"tau truth {TRUTH_TAU_MYR} is OUTSIDE the model prior {tau_bounds} — unreachable"
    )
    assert_truth_is_discriminating(TRUTH_SIGMA, sigma_bounds, name="sfh_field_psd_sigma")
    assert_truth_is_discriminating(TRUTH_TAU_MYR, tau_bounds, name="sfh_field_psd_tau_myr")
    print("truths are inside the prior AND discriminating")

    pop = make_population(
        model,
        n_galaxies=N_GAL,
        sigma_true=TRUTH_SIGMA,
        tau_true_myr=TRUTH_TAU_MYR,
        key=jax.random.PRNGKey(0),
        snr_phot=SNR_PHOT,
        snr_line=SNR_LINE,
    )
    print(f"generated {N_GAL} galaxies, Halpha absorption events: {pop.n_halpha_absorption}")

    flux = np.asarray(pop.table["phot_flux_obs"])
    err = np.asarray(pop.table["phot_flux_err"])

    print("\n galaxy |  chi2/N @ MAP |  sigma_hat |  tau_hat [Myr] |  wall")
    print(" -------|---------------|------------|----------------|------")
    rows = []
    for i in range(N_GAL):
        t0 = time.time()
        f = Fitter(model, flux[i], err[i])
        r = f.run(
            "map", key=jax.random.fold_in(jax.random.PRNGKey(1), i), n_steps=12000, n_restarts=3
        )
        dt = time.time() - t0
        pred = np.asarray(model.predict_photometry(r.params))
        chi2_n = float(np.sum(((flux[i] - pred) / err[i]) ** 2) / len(flux[i]))
        s_hat = float(r.params.get("sfh_field_psd_sigma", np.nan))
        t_hat = float(r.params.get("sfh_field_psd_tau_myr", np.nan))
        rows.append((chi2_n, s_hat, t_hat))
        print(
            f"   {i}    |     {chi2_n:8.3f}  |   {s_hat:6.3f}   |    {t_hat:8.1f}    | {dt:5.1f}s"
        )

    c = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    t = np.array([r[2] for r in rows])
    print("\n--- verdict against the SFH recovery study's benchmark ---")
    print(f"chi2/N at MAP : range [{c.min():.3f}, {c.max():.3f}]  (study: 0.32-1.06)")
    ok = (c.min() >= 0.1) and (c.max() <= 3.0)
    print(
        f"               {'PASS' if ok else 'FAIL'} — per-galaxy fits are "
        f"{'sound' if ok else 'NOT sound; the population step cannot rescue this'}"
    )
    print(
        f"sigma_hat     : median {np.median(s):.3f} vs truth {TRUTH_SIGMA} "
        f"(study predicts COMPRESSION toward the prior for photometry-only)"
    )
    print(
        f"tau_hat       : median {np.median(t):.1f} vs truth {TRUTH_TAU_MYR} "
        f"(bounds {tau_bounds}; at a bound => unconstrained)"
    )
    n_railed = int(np.sum((t <= tau_bounds[0] * 1.02) | (t >= tau_bounds[1] * 0.98)))
    print(f"tau at a bound: {n_railed}/{N_GAL} galaxies")


if __name__ == "__main__":
    main()
