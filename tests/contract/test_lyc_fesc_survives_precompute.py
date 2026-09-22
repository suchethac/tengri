#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""``neb_fesc`` must attenuate the stellar Lyman continuum on both paths.

``neb_fesc`` is the fraction of ionizing photons that escape the birth cloud.
``NebularSEDComponent`` applies it as ``where(lambda < 912, fesc, 1)`` on the
rest-frame SED array (``components/nebular/component.py``, the ``lyc_mask``
line), so on the exact wavelength-grid path a band probing rest-frame
lambda < 912 A scales with ``fesc``.

Under ``WavePrecomp`` the stellar photometry is not an array. It is a
pre-integrated SSP x filter scalar per band, computed at build time before any
component runs, so that multiplication has nothing to act on and is lost. The
band then reports the *unattenuated* stellar LyC no matter what ``fesc`` says.

Measured at z=1 through GALEX FUV, which probes rest-frame 670-903 A -- wholly
inside the Lyman continuum -- on a delayed-tau model with Cue nebular. On this
file's grid (``fsps_prsc_miles_chabrier``):

    neb_fesc   exact            LUT              LUT/exact
    0.05       1.115729e-31     2.231175e-30     19.9975
    0.50       1.115729e-30     2.231175e-30     1.9997

and on ``fsps_mist_c3k_a_chabrier``, which samples the Lyman continuum nine
times more densely (1106 points below 912 A against 122), 19.9992 and 1.9999.
Two grids of very different LyC resolution giving the same ratio is itself
evidence: a sampling problem would not.

The ratio is exactly ``1/fesc``, and the LUT column does not move: a tenfold
change in ``fesc`` moves it by 1.0000000000000104. That is the signature of a
factor that never reaches the number, not of an approximation that is coarse.
Refining the quadrature does not help either -- on the kitchen-sink mock the
disagreement *converges* (20.12% -> 20.22% from 5 to 51 sub-bands) rather than
shrinking, and the IGM fold is irrelevant to it (node and exact agree to
20.222% at 51 sub-bands).

This matters beyond one band. Every fit surface resolves ``approx="auto"`` to
``WavePrecomp`` for photometry, so this is the default path, and ``neb_fesc``
is routinely a free parameter -- the likelihood is then flat in the one
observable that constrains it.

The no-nebular control is the part that makes this a nebular finding rather
than a precompute one: without ``neb`` the two paths agree to 1.0000 in the
same band, so the sub-band quadrature and the IGM fold are both exonerated.

Marked ``xfail(strict=True)``: this is a live defect, and the marker is the
thing that fails once it is fixed, so whoever fixes it is told to delete it.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, WavePrecomp
from tengri.parameters import Fixed

pytestmark = pytest.mark.contract

#: Rest-frame 670-903 A at z=1: wholly below the 912 A Lyman limit.
PROBE_BAND = "galex_fuv"
CONTROL_BAND = "sdss_r"
BANDS = [PROBE_BAND, CONTROL_BAND]
PROBE, CONTROL = 0, 1
PROBE_Z = 1.0

SFH = {
    "type": "delayed",
    "all_params": Fixed(DEFAULT),
    "tau_gyr": Fixed(2.0),
    "age_gyr": Fixed(3.0),
    "log_total_mass": Fixed(10.0),
    "met_logzsol": Fixed(0.0),
}


@pytest.fixture(scope="module")
def ssp(ssp_data_fsps):
    """The bare-stellar FSPS grid: fsps_mist_c3k_a_chabrier.

    Bare rather than wNE deliberately. A with-nebular-emission grid already
    carries nebular emission baked in at a fixed escape fraction, so the
    quantity under test would be partly fixed inside the templates.
    """
    return ssp_data_fsps


@pytest.fixture(scope="module")
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _photometry(ssp, observation, approx, neb=None):
    extra = {"neb": neb} if neb is not None else {}
    model = SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh=dict(SFH),
        redshift=Fixed(PROBE_Z),
        igm={"type": "inoue"},
        approx=approx,
        **extra,
    )
    return np.asarray(model.predict_photometry({}), dtype=np.float64)


def _cue(fesc):
    return {"type": "cue", "fesc": Fixed(fesc), "fdust": Fixed(0.1), "all_params": Fixed(DEFAULT)}


def test_without_nebular_the_two_paths_agree_in_the_same_band(ssp, observation):
    """The control that makes this a nebular finding, not a precompute one.

    If this fails the comparison below is measuring the quadrature or the IGM
    fold, and the nebular conclusion does not follow.
    """
    exact = _photometry(ssp, observation, None)
    lut = _photometry(ssp, observation, WavePrecomp(n_subbands=21))

    assert lut[PROBE] / exact[PROBE] == pytest.approx(1.0, rel=1e-3, abs=0), (
        "the two paths already disagree with no nebular component, so the "
        "Lyman-continuum comparison below cannot be attributed to fesc"
    )


def test_the_exact_path_scales_the_lyman_continuum_by_fesc(ssp, observation):
    """Establish the reference behavior before asserting the LUT matches it."""
    bare = _photometry(ssp, observation, None)[PROBE]
    half = _photometry(ssp, observation, None, _cue(0.5))[PROBE]
    twentieth = _photometry(ssp, observation, None, _cue(0.05))[PROBE]

    assert half / bare == pytest.approx(0.5, rel=1e-3, abs=0)
    assert twentieth / bare == pytest.approx(0.05, rel=1e-3, abs=0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "neb_fesc does not reach the stellar Lyman continuum under WavePrecomp: "
        "the stellar photometry is pre-integrated per band before any component "
        "runs, so the where(lambda<912, fesc, 1) mask has no array to act on. "
        "Measured LUT/exact = 1/fesc exactly. Delete this marker when fixed."
    ),
)
@pytest.mark.parametrize("fesc", [0.05, 0.5])
def test_fesc_reaches_the_lyman_continuum_under_precompute(ssp, observation, fesc):
    """The defect. The LUT must honor fesc, as the exact path does."""
    exact = _photometry(ssp, observation, None, _cue(fesc))
    lut = _photometry(ssp, observation, WavePrecomp(n_subbands=21), _cue(fesc))

    # Ratios, not pytest.approx: these fluxes are ~1e-31 and approx keeps an
    # absolute tolerance of 1e-12 even when rel= is given, so every comparison
    # here passes for any pair of values. That is how this test first went
    # green against the very disagreement it exists to catch.
    assert lut[CONTROL] / exact[CONTROL] == pytest.approx(1.0, rel=1e-3, abs=0), (
        "the optical control disagrees too, so this is not specific to the LyC"
    )
    ratio = lut[PROBE] / exact[PROBE]
    assert ratio == pytest.approx(1.0, rel=5e-2, abs=0), (
        f"fesc={fesc}: exact {exact[PROBE]:.6e}, LUT {lut[PROBE]:.6e}, "
        f"ratio {ratio:.4f} (expected ~1, measured ~1/fesc = {1 / fesc:.1f})"
    )


@pytest.mark.xfail(
    strict=True,
    reason="same defect, stated as an invariant rather than a comparison",
)
def test_the_precomputed_band_moves_at_all_when_fesc_changes(ssp, observation):
    """The sharpest statement: under the LUT the band is inert in fesc.

    A tolerance comparison could in principle be met by a LUT that is merely
    inaccurate. This one cannot: it fails whenever the number does not move,
    which is what "the factor never arrives" actually looks like.
    """
    low = _photometry(ssp, observation, WavePrecomp(n_subbands=21), _cue(0.05))[PROBE]
    high = _photometry(ssp, observation, WavePrecomp(n_subbands=21), _cue(0.5))[PROBE]

    # Not ``low != high``: the two differ in their last bits, so inequality is
    # satisfied by an ulp and says nothing. A tenfold change in fesc has to
    # move the band by something like tenfold.
    assert high / low == pytest.approx(10.0, rel=0.2, abs=0), (
        f"a tenfold change in neb_fesc moved the band by {high / low:.6f}x "
        f"({low:.6e} -> {high:.6e}); the LyC term is not responding to fesc"
    )
