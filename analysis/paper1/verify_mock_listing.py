"""Execute the mock listing printed in the paper, so it cannot rot unnoticed.

``3-mock-galaxy.tex`` prints a ``SEDModel.build`` call as the worked example of
the framework's central claim: that one call composes a panchromatic joint fit.
A reader copies that block. Nothing in CI ran it, and on 2026-09-20 it carried
seven defects -- four names that do not exist, one retired API form, one prior
that admits negative luminosity, and one declaration that builds cleanly and
raises only at prediction.

This script is the missing check. It builds the listing as corrected and
asserts the model predicts finite positive photometry. Run it whenever the
listing or the grammar changes.

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
===  ==========================================  ==================================

Defect 3 is not a transcription error. The writing plan locks "two-component
with free shapes", and the grammar cannot express that: ``dust_law_overrides``
is ``dict[str, dict[str, float]]`` and those values are folded into the compile
key. Either the paper drops per-screen free shapes or the framework grows them.
This script uses fixed per-screen shapes because that is what runs today.
"""

from __future__ import annotations

import jax
import numpy as np

import tengri
from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Gaussian,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)

jax.config.update("jax_enable_x64", True)

REDSHIFT = 1.0

#: Ultraviolet through mid-infrared. The listing also names X-ray and radio,
#: which enter as model components rather than as filters.
MOCK_FILTERS = [
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
        neb={"type": "cue", "all_params": FREE},
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "all_params": FREE},
            "torus": {"type": "skirtor", "all_params": FREE},
            "nlr": {"type": "analytic", "all_params": FREE},
            "blr": {"type": "analytic", "all_params": FREE},
            "atten": {"type": "grahsp_biatten", "all_params": FREE},
        },
        xray={"type": "yang20", "all_params": FREE},
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}},
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


def main() -> int:
    ssp = tengri.load_ssp("fsps_mist_c3k_a_chabrier")
    obs = Observation(photometry=Photometry.from_names(MOCK_FILTERS))
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
    print("[ok] the corrected listing executes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
