"""Execute the mock listing printed in the paper, so it cannot rot unnoticed.

``3-mock-galaxy.tex`` prints a ``SEDModel.build`` call as the worked example of
the framework's central claim: that one call composes a panchromatic joint fit.
A reader copies that block. Nothing in CI ran it, and on 2026-09-20 it carried
eight defects: four names that do not exist, one retired API form, one prior
that admits negative luminosity, one declaration that builds cleanly and raises
only at prediction, and one that inverts the observation/data separation.

This script is the missing check. It builds the listing as corrected and
asserts the model predicts finite positive photometry AND a finite positive
spectrum: the section's claim is joint inference, so a check that exercised
only photometry would pass while the demonstrated thing was broken. Run it
whenever the listing or the grammar changes.

The defects found, recorded so a future edit does not reintroduce them:

===  ==========================================  ==================================
 #   As printed                                  Correct
===  ==========================================  ==================================
 1   ``Spectrum(...)``                           ``Spectroscopy(...)``
 2   ``dust_emission={"type": "draine2014"}``    ``"draine_li2014"``
 3   ``slope_bc=Uniform(...)`` per-screen prior  a scalar: per-screen law shapes
                                                 are build-time constants in the
                                                 compile signature, not sampled
 4   ``sfh={"n_bins": 7}``                       ``"bin_edges_gyr": <edges>``
 5   ``radio={"type": "sf_agn"}``                retired; composable
                                                 ``{"sf": ..., "agn": ...}``
 6   ``eta_balance=Gaussian(1.0, 0.3)``          needs ``lo=0.0``; untruncated it
                                                 admits negative L_IR
 7   ``met={"type": "table"}``                   needs an actual Z(t) table; omit
                                                 it or use ``{"logzsol": ...}``
 8   ``Observation(spectrum=Spectrum(wave=,     ``Observation(spectroscopy=
     fnu=, fnu_err=, resolution=))``             Spectroscopy(wave_obs=,
                                                 resolution=))`` and the fluxes
                                                 go to ``Data``, not here
===  ==========================================  ==================================

Defect 3 is not a transcription error. The writing plan locks "two-component
with free shapes", and the grammar cannot express that: ``dust_law_overrides``
is ``dict[str, dict[str, float]]`` and those values are folded into the compile
key. Either the paper drops per-screen free shapes or the framework grows them.
This script uses fixed per-screen shapes because that is what runs today.

Defect 8 is the one that misrepresents the design rather than misspelling it.
``Observation`` is the observation *model* -- which filters, what wavelength
grid, what resolution -- and ``Data`` carries the *measurements*. ``Photometry``
has no ``fnu`` parameter at all. The listing puts fluxes inside both observation
objects, which collapses a separation the framework maintains deliberately:
one ``Observation`` can be reused across many galaxies, each with its own
``Data``. That is what makes the catalog path possible, so the example teaching
joint fitting inverts the thing the section is demonstrating.
"""

from __future__ import annotations

import jax
import numpy as np

import tengri
from tengri import (
    DEFAULT,
    FREE,
    Data,
    Fixed,
    Gaussian,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    Uniform,
    WavePrecomp,
)

jax.config.update("jax_enable_x64", True)

REDSHIFT = 1.0

#: The SSP library this mock is built on. Named once so a figure rebuilding the
#: same model cannot drift from the script that wrote the truth file.
SSP_NAME = "fsps_mist_c3k_a_chabrier"

#: X-ray through millimeter: seven decades in wavelength, 4 Angstrom to 3 mm.
#:
#: The two Chandra bands are not decoration. ``xray={"type": "yang20",
#: "all_params": FREE}`` frees ``xray_log_nh``, and photoelectric absorption is
#: strongly energy-dependent -- it eats the soft band and spares the hard one --
#: so the soft/hard *ratio* is the only observable that carries column density.
#: With a UV-through-MIR filter set the model still declared an X-ray component
#: and MAP recovered ``xray_log_nh`` 2.04 dex from truth, because nothing
#: constrained it. One X-ray band would have pinned the luminosity and left the
#: column prior-driven; the pair is what closes it.
#:
#: The millimeter bands sit on the Rayleigh-Jeans tail of ``draine_li2014`` and
#: constrain the cold dust mass that ``dust_eta_balance`` controls. Band 3 is
#: included knowing it is a non-detection at a realistic depth: it bounds the
#: long-wavelength end from above rather than measuring it.
#:
#: **There is no radio band, and this is a limitation, not a choice.** The
#: registry's longest-wavelength entries are millimeter (ALMA, ACT, SPT,
#: TolTEC); it holds no VLA or LOFAR curve, so the centimetre regime is not
#: expressible today. The ``radio`` block still enters the forward model and
#: still emits, but every one of its parameters (``radio_q_ir``,
#: ``radio_alpha_sf``, ``radio_alpha_ff``, ``radio_T_e``) is ``Fixed(DEFAULT)``,
#: so none of them is a free parameter the data would have to constrain. The
#: model predicts a radio luminosity; the mock does not claim to measure one.
MOCK_FILTERS = [
    # X-ray: the soft/hard pair, which is what constrains N_H.
    "chandra_soft",
    "chandra_hard",
    # Ultraviolet through mid-infrared.
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "2mass_j",
    "2mass_ks",
    "wise_w1",
    "wise_w2",
    "wise_w3",
    "wise_w4",
    # Submillimeter and millimeter: the cold-dust Rayleigh-Jeans tail.
    "alma_band7",
    "alma_band6",
    "alma_band3",
]


def build_mock_model(ssp, observation, z=REDSHIFT):
    """The paper's listing, corrected to what the grammar accepts."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "continuity",
            "all_params": FREE,
            "age_kernel": "cic",
            "bin_edges_gyr": tengri.make_agebins_from_zred(z, n_bins=7),
        },
        # met={"type": "table"} as printed builds and then raises at predict:
        # it wants a Z(t) history the mock does not have. A free logzsol is the
        # nearest thing that means what the section describes.
        met={"logzsol": Uniform(-2.0, 0.4)},
        dust_attenuation={
            "type": "two_component",
            "law_bc": "power_law",
            "slope_bc": -0.7,
            "law_diff": "noll09",
            "delta_diff": 0.0,
            "tau_bc": Uniform(0.0, 3.0),
            "tau_diff": Uniform(0.0, 3.0),
        },
        dust_emission={
            "type": "draine_li2014",
            "eta_balance": Gaussian(1.0, 0.3, lo=0.0),
            "other_params": FREE,
        },
        # Nebular emission is ~38% of the optical photometry at this truth, so
        # most of this block is measurable: neb_logU scores 5e4 on the Delta
        # chi2 sweep, the largest of any parameter in the model.
        #
        # ``fesc`` and ``fdust`` are two fractions of the SAME ionizing-photon
        # budget. The CIGALE k-factor is
        #     k = (1 - fesc - fdust) / (1 + 0.597 (fesc + fdust)),
        # and ``lyc_dust_escape_factor`` clips the sum at 1.0, so ANY pair
        # summing to >= 1 makes k exactly 0 and switches nebular emission off
        # entirely -- lines and continuum. Nothing enforces that joint
        # constraint (#2436), and the declared Uniform(0, 1) priors put 50% of
        # their joint volume in that dead region, where the six parameters
        # below have exactly zero gradient. An earlier truth for this mock drew
        # 0.708 + 0.462 = 1.17 from those priors and produced a mock with no
        # nebular emission at all. Capping each at 0.35 holds the sum under 0.7,
        # so neither the truth nor the sampler can reach the plateau.
        neb={
            "type": "cue",
            "logU": FREE,
            "logZ_gas": FREE,
            "eline_sigma_kms": FREE,
            "dig_frac": FREE,
            "dig_delta_logU": FREE,
            "fesc": Uniform(0.0, 0.35),
            "fdust": Uniform(0.0, 0.35),
            # fesc_lya stays pinned: at z=1 Ly-alpha sits at 2432 A observed,
            # blueward of the spectrum and in the wing of GALEX NUV, so it
            # scores 5.6 -- weak, like the other parameters this mock pins.
            "all_params": Fixed(DEFAULT),
        },
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "all_params": FREE},
            # Four of the five SKIRTOR geometry knobs score below 1: p 0.466,
            # tau 0.112, radius_ratio 0.0528, q 0.0150. Four WISE bands and
            # three ALMA bands cannot determine a five-parameter torus. The
            # opening angle (3.29e3) and the covering factor (2.36e3) survive.
            "torus": {
                "type": "skirtor",
                "oa_skirtor": FREE,
                "torus_frac": FREE,
                "all_params": Fixed(DEFAULT),
            },
            "nlr": {"type": "analytic", "all_params": FREE},
            "blr": {"type": "analytic", "all_params": FREE},
            # One attenuation normalization on the AGN, by choice rather than
            # by evidence. An earlier sweep scored grahsp_ebv and
            # grahsp_ebv_agn identically to three figures and read that as an
            # exactly degenerate pair; that sweep was taken at a truth with
            # nebular emission switched off (#2436) and the equality did not
            # survive re-measurement -- they now score 4.88e3 and 2.89e3. They
            # are two distinct screens, both live, and pinning one is a
            # modeling choice for this mock, not a redundancy result.
            #
            # The key is "grahsp_ebv", not "ebv". Both are accepted; "ebv"
            # resolves to agn_ebv, a DIFFERENT parameter, so the shorter
            # spelling silently swaps one for the other and nothing raises.
            "atten": {
                "type": "grahsp_biatten",
                "grahsp_ebv": FREE,
                "all_params": Fixed(DEFAULT),
            },
            # The AGN's bolometric luminosity, which the Chandra pair constrains
            # through the disc-corona coupling that ``yang20`` supplies. Freeing
            # it is what makes this a recovery of AGN *energetics* rather than
            # only of the torus geometry: left implicit it sat at the registry
            # default of 10, and the section claimed an energy budget nothing
            # inferred.
            #
            # It is genuinely the luminosity scale here, measured rather than
            # assumed -- under the default ``cigale_joint`` normalization it was
            # not obvious that the composable AGN would ride on ``agn_log_lbol``
            # rather than an ``agn_power`` reference, and this configuration has
            # no ``agn_power`` at all. Sweeping its declared prior with
            # everything else pinned moves the photometry by a factor of 2.4e3.
            "log_lbol": FREE,
            # cos_inc, ir_frac and lum_ratio keep their defaults. Said out loud
            # rather than left to the wildcard's silence: all three are strongly
            # live (3.3e3, 7.3e8 and 1.0e5 on the sweep), so this is a choice,
            # not an absence. cos_inc in particular is degenerate with the torus
            # opening angle, which is already free, and ir_frac and lum_ratio
            # both re-scale AGN luminosity that agn_log_lbol already carries --
            # a one-at-a-time sweep scores each highly and cannot see that they
            # are the same degree of freedom three times over.
            "all_params": Fixed(DEFAULT),
        },
        # "all_params": FREE on this group never meant all of them: it frees
        # the six that carry a declared prior and silently leaves
        # xray_gamma_hmxb and xray_gamma_lmxb pinned. Of the six, E_cut scores
        # 0.00724 -- it is the coronal cutoff at 100-300 keV and Chandra stops
        # at 8 keV, so no band can see it -- and det_lmxb 2.05. The four named
        # here are what the soft/hard pair actually determines.
        xray={
            "type": "yang20",
            "log_nh": FREE,
            "gamma_agn": FREE,
            "det_hmxb": FREE,
            "delta_alpha_ox": FREE,
            "all_params": Fixed(DEFAULT),
        },
        # Stated, not left to the group's silence: omitting the disposition
        # pinned all four at defaults nobody chose. They stay pinned because
        # the mock observes no radio band -- the longest pivot in the set is
        # ALMA band 3 at 3.1 mm -- and the sweep confirms it: radio_q_ir,
        # radio_alpha_sf and radio_T_e all move the data by EXACTLY 0.000.
        # That exact zero is the honest kind: nothing observes the quantity.
        # Contrast the nebular six above, whose exact zeros came from an
        # amplitude clamped to zero upstream (#2436) while the bands that
        # would have seen them were right there in the filter set.
        radio={
            "sf": {"type": "bell2003"},
            "agn": {"type": "powerlaw"},
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


#: SDSS-like optical coverage for the spectroscopic channel.
SPEC_WAVE_OBS = np.linspace(3800.0, 9200.0, 1500)
SPEC_RESOLUTION = 2000.0


def build_joint_observation():
    """One ``Observation`` carrying both channels, as the section claims.

    The measurements do not live here. ``Data`` carries those, which is why one
    ``Observation`` serves a whole catalog.
    """
    return Observation(
        photometry=Photometry.from_names(MOCK_FILTERS),
        spectroscopy=Spectroscopy(wave_obs=SPEC_WAVE_OBS, resolution=SPEC_RESOLUTION),
    )


def main() -> int:
    ssp = tengri.load_ssp(SSP_NAME)
    obs = build_joint_observation()
    model = build_mock_model(ssp, obs)

    free = sorted(model.spec.free_params)
    print(f"listing builds: D = {len(free)}")
    for prefix, label in (("agn", "AGN"), ("sfh", "SFH"), ("dust", "dust"), ("neb", "nebular")):
        n = len([p for p in free if p.startswith(prefix)])
        print(f"  {label:<8} {n:>3}")

    params = model.spec.sample(key=jax.random.PRNGKey(0))
    phot = np.asarray(model.predict_photometry(params))

    assert phot.shape == (len(MOCK_FILTERS),), f"shape {phot.shape}"
    assert np.all(np.isfinite(phot)), "non-finite photometry"
    assert (phot > 0).all(), "non-positive photometry"
    print(
        f"\npredicts {len(phot)} bands, finite and positive ({phot.min():.2e} to {phot.max():.2e})"
    )

    # The section's actual claim is JOINT inference, so a check that exercises
    # only photometry would pass while the thing being demonstrated is broken.
    spectrum = np.asarray(model.predict(params).spectrum())
    assert spectrum.shape == (SPEC_WAVE_OBS.size,), f"shape {spectrum.shape}"
    assert np.all(np.isfinite(spectrum)), "non-finite spectrum"
    assert (spectrum > 0).all(), "non-positive spectrum"
    print(f"predicts {spectrum.size} spectral pixels, finite and positive")

    # Measurements are a separate object from the observation model. Building
    # one here keeps the distinction that defect 8 collapsed under test.
    data = Data(
        photometry=(phot, 0.05 * phot),
        spectrum=(spectrum, 0.05 * spectrum),
    )
    assert data is not None
    print("Data carries the measurements; Observation carries the model")
    print("[ok] the corrected listing executes, both channels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
