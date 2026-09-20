"""Mock panchromatic joint spectro-photometric inference: data generation.

Builds the truth SED for the paper's ``sec:mock`` showcase and the synthetic
joint observation recovered from it. The model is the listing printed in
``3-mock-galaxy.tex``, executed rather than retyped -- it is imported from
:mod:`paper1.verify_mock_listing`, so the figure and the printed code cannot
drift apart.

Two choices here are deliberate and would be wrong if made the easy way.

**The truth is designed, not drawn.** A blind draw from the prior produced a
low-mass, heavily obscured host with a torus-dominated reddened AGN --
``agn_torus_frac`` 0.87, ``agn_ebv_disc`` 0.50, ``tau_bc`` 2.4, ``log M`` 8.6 --
whose WISE W4 flux exceeded GALEX NUV by a factor of 1.4e6. That is a Type 2
SED, and the section specifies a Type 1: an unobscured sightline where the
accretion disc dominates the ultraviolet. It is also useless as a *joint*
showcase, because an optical spectrum of that object is noise. A demonstration
figure is designed; only the recovery is random.

**Faint bands are upper limits, not detections.** A flat fractional error gives
every band the same signal-to-noise no matter how faint, which invents a
detection six decades below the brightest band. Each filter therefore carries a
depth, and anything under it is recorded as a non-detection with its limit. The
writing plan is explicit: do not invent detections without a noise model.

Note ``agn_log_lbol`` is not among the free parameters of this configuration --
a composable AGN with ``all_params: FREE`` pins the bolometric luminosity -- so
the AGN's appearance is set through ``agn_torus_frac`` and the reddening of the
disc sightline rather than through its luminosity.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

import tengri

from .verify_mock_listing import (
    MOCK_FILTERS,
    SPEC_WAVE_OBS,
    build_joint_observation,
    build_mock_model,
)

jax.config.update("jax_enable_x64", True)

RESULTS = Path(__file__).with_name("results")
TRUTH_NPZ = RESULTS / "mock_joint_truth.npz"

SEED = 20260920

#: Truth parameters set by hand to make a Type 1 AGN with a star-forming host.
#: Everything not named here keeps its prior draw, so the recovery still has to
#: work across the full 43-dimensional space rather than a hand-tuned corner.
DESIGNED_TRUTH = {
    # An unobscured sightline to the disc is what makes the object Type 1.
    "agn_ebv_disc": 0.02,
    "agn_grahsp_ebv": 0.05,
    "agn_grahsp_ebv_agn": 0.05,
    # Torus present and visible in the mid-infrared, but not dominant: the
    # prior draw put it at 0.87, which buries the disc.
    "agn_torus_frac": 0.25,
    # An ordinary star-forming host rather than a starburst behind a screen.
    "dust_tau_diff": 0.30,
    "dust_tau_bc": 1.00,
    "sfh_cont_log_total_mass": 10.5,
    "met_logzsol": 0.0,
    # Unobscured in X-rays too, which is what Type 1 means there.
    "xray_log_nh": 20.0,
}

#: Per-filter 1-sigma depth [erg/s/cm2/Hz]. Rough but honest: deep-field optical
#: and near-infrared, shallower in the ultraviolet and mid-infrared, which is
#: the ordering any real panchromatic dataset has.
#:
#: The X-ray depths are the 2 Ms Chandra Deep Field South limits converted out
#: of the integrated band: roughly 1e-17 erg/s/cm2 over 0.5-2 keV and 6e-17 over
#: 2-8 keV, divided by the bandwidth in frequency. The millimetre depths are a
#: deep pointed ALMA continuum observation, about 1 uJy rms. ALMA band 3 is
#: listed knowing the source falls under it -- a non-detection at the long-
#: wavelength end is the honest outcome there, and exercising the upper-limit
#: path is worth more than inventing a detection.
BAND_DEPTH = {
    "chandra_soft": 3.0e-35,
    "chandra_hard": 4.0e-35,
    "galex_nuv": 3.0e-31,
    "sdss_u": 1.0e-31,
    "sdss_g": 5.0e-32,
    "sdss_r": 5.0e-32,
    "sdss_i": 7.0e-32,
    "2mass_j": 2.0e-31,
    "2mass_ks": 3.0e-31,
    "wise_w1": 2.0e-31,
    "wise_w2": 3.0e-31,
    "wise_w3": 2.0e-30,
    "wise_w4": 1.0e-29,
    "alma_band7": 1.0e-29,
    "alma_band6": 5.0e-30,
    "alma_band3": 1.0e-29,
}

#: Fractional calibration term added in quadrature with the depth, so bright
#: bands are calibration-limited rather than infinitely precise.
#:
#: This is per-band because a single global value would have written "3 per cent
#: X-ray photometry" into the paper. Chandra's absolute effective-area
#: calibration is at the ten-per-cent level and degrades with the contamination
#: model at soft energies; ALMA quotes 10 per cent in band 3 through 7. The
#: optical and infrared 3 per cent is the ordinary broadband figure.
CALIBRATION_FRAC = 0.03
CALIBRATION_FRAC_BY_BAND = {
    "chandra_soft": 0.15,
    "chandra_hard": 0.15,
    "alma_band7": 0.10,
    "alma_band6": 0.10,
    "alma_band3": 0.10,
}
SPEC_FRAC = 0.10
DETECTION_SIGMA = 3.0


def designed_truth(model, key):
    """Prior draw with the showcase parameters overridden."""
    truth = dict(model.spec.sample(key=key))
    for name, value in DESIGNED_TRUTH.items():
        if name not in truth:
            raise KeyError(
                f"{name!r} is not a free parameter of this configuration; the "
                f"designed truth would silently do nothing. Free parameters: "
                f"{sorted(truth)}"
            )
        truth[name] = value
    return truth


def photometric_errors(flux, filters):
    """1-sigma errors from depth and calibration, and the detection mask.

    A band with no declared depth is refused rather than defaulted. A default
    would give a new filter the noise properties of whichever band the author
    last thought about, and the whole point of the depth table is that a flat
    error invents detections six decades below the brightest band.
    """
    missing = [f for f in filters if f not in BAND_DEPTH]
    if missing:
        raise KeyError(
            f"no depth declared for {missing}; add them to BAND_DEPTH. "
            f"Declared: {sorted(BAND_DEPTH)}"
        )
    depth = np.array([BAND_DEPTH[f] for f in filters])
    calib = np.array([CALIBRATION_FRAC_BY_BAND.get(f, CALIBRATION_FRAC) for f in filters])
    sigma = np.hypot(depth, calib * flux)
    detected = flux > DETECTION_SIGMA * depth
    return sigma, detected, depth


def main() -> int:
    ssp = tengri.load_ssp("fsps_mist_c3k_a_chabrier")
    obs = build_joint_observation()
    model = build_mock_model(ssp, obs)

    truth = designed_truth(model, jax.random.PRNGKey(SEED))
    phot_true = np.asarray(model.predict_photometry(truth))
    spec_true = np.asarray(model.predict(truth).spectrum())

    sigma, detected, depth = photometric_errors(phot_true, MOCK_FILTERS)
    rng = np.random.default_rng(SEED)
    phot_obs = phot_true + rng.normal(0.0, sigma)
    spec_sig = SPEC_FRAC * spec_true
    spec_obs = spec_true + rng.normal(0.0, spec_sig)

    free = sorted(model.spec.free_params)
    RESULTS.mkdir(exist_ok=True)
    np.savez(
        TRUTH_NPZ,
        free_params=np.array(free, dtype=object),
        truth_values=np.array([float(truth[k]) for k in free]),
        filters=np.array(MOCK_FILTERS, dtype=object),
        phot_true=phot_true,
        phot_obs=phot_obs,
        phot_sig=sigma,
        phot_depth=depth,
        detected=detected,
        wave_obs=SPEC_WAVE_OBS,
        spec_true=spec_true,
        spec_obs=spec_obs,
        spec_sig=spec_sig,
    )

    print(f"D = {len(free)}, {int(detected.sum())}/{len(MOCK_FILTERS)} bands detected\n")
    print(f"{'band':<12}{'flux':>11}{'S/N':>8}  status")
    for i, band in enumerate(MOCK_FILTERS):
        snr = phot_true[i] / sigma[i]
        status = "detected" if detected[i] else f"< {DETECTION_SIGMA:.0f}sigma limit"
        print(f"{band:<12}{phot_true[i]:11.3e}{snr:8.1f}  {status}")

    nuv = phot_true[MOCK_FILTERS.index("galex_nuv")]
    w4 = phot_true[MOCK_FILTERS.index("wise_w4")]
    print(f"\nNUV / W4 = {nuv / w4:.3e} (the prior draw gave 7.0e-07)")
    print(f"spectrum median S/N {np.median(spec_true / spec_sig):.1f}")
    print(f"saved {TRUTH_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
