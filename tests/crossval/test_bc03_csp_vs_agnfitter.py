# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate tengri's declining-exponential CSP against AGNfitter-rX's BC03 library.

AGNfitter-rX's GALAXY component (``MODEL_AGNfitter.GALAXY()``, ``'BC03'``
branch) tabulates a pre-computed grid of Bruzual & Charlot (2003) composite
stellar populations over ``(tau, age)``. The pickle's own ``SFR`` array
declines monotonically with age at fixed tau -- the classic
declining-exponential tau-model (SFR(T) ~ exp(-T/tau)), NOT a delayed-tau
history (SFR(T) ~ T*exp(-T/tau), which rises before falling). Confusing the
two silently produced a wavelength-dependent residual before (#406, and its
near-repeat in the reproduction notebook, D1 in the AGNfitter-rX parity
audit). This file exercises tengri's ``sfh={'type': 'declining_exp', ...}``
(NOT ``'delayed'``) against the vendored reference for exactly this reason.

Library-edition difference (documented, not a defect)
------------------------------------------------------
Upstream's ``BC03_840seds.pickle`` tabulates 1221 wavelength points
(91 A - 1.6 mm); tengri's own ``bc03_pdva_stelib_chabrier.h5`` SSP grid has
6900 points. Both claim "BC03 + Chabrier" but are different library editions
(upstream's is smpy-generated, historically a lower-resolution BaSeL-based
grid; tengri's filename implies STELIB). This file compares SHAPE only (both
normalized to 1 at 5500 A) at a matched (tau, age) node -- an absolute-flux
comparison is not attempted (the two libraries' mass-formed normalization
conventions are not established to be equal).

Short-name grammar inconsistency (reported, not fixed here)
-------------------------------------------------------------
The public dict grammar's short-form parameter resolution accepts bare
``tau_gyr``/``age_gyr``/``log_total_mass`` for ``sfh={'type': 'delayed', ...}``
but RAISES ``ValueError`` for the literally identical spelling under
``sfh={'type': 'declining_exp', ...}`` -- only the fully-prefixed
``sfh_declining_exp_tau_gyr`` (etc.) resolves. This is a real inconsistency
in the short-name resolver (not the SFH-form defect this file exists to
catch), flagged here per the parity audit's UNVERIFIED note rather than
silently worked around. Pinned as an ``xfail(strict=True)`` below
(:func:`test_declining_exp_short_form_keys_resolve`) asserting the POSITIVE
claim ("bare keys resolve for both types") so a future resolver fix turns it
into a loud XPASS rather than leaving the inconsistency silently
un-noticed forever. Every other test in this file uses the fully-prefixed
spelling regardless of this xfail's outcome.

References
----------
.. [1] L. N. Martinez-Ramirez, et al., "AGNFITTER-RX: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). doi:10.1051/0004-6361/202449329. arXiv:2405.12111.
   bibcode: 2024A&A...688A..46M
.. [2] G. Bruzual & S. Charlot, "Stellar population synthesis at the
   resolution of 2003," MNRAS, 344, 1000 (2003).
   bibcode: 2003MNRAS.344.1000B
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.crossval, pytest.mark.regression_paper]

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_PATH = _DATA_DIR / "bc03_pdva_stelib_chabrier.h5"
_GALAXY_REF_PATH = _DATA_DIR / "agnfitter_galaxy_reference.h5"

#: Matched node: tau=1.0 Gyr is an exact upstream grid value; age=4.8939 Gyr
#: is the nearest upstream ``bc03_840`` age node to 5 Gyr (upstream is a
#: discrete, non-interpolated grid -- see D6 in the parity audit).
_TAU_GYR = 1.0
_AGE_GYR = 4.893900918477499

if not _GALAXY_REF_PATH.is_file():
    pytest.skip(
        f"AGNfitter GALAXY reference not found: {_GALAXY_REF_PATH} (build with "
        "scripts/build_agnfitter_galaxy_reference.py)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def ssp_data():
    """The shipped BC03 SSP grid; skip with an exact reason if absent.

    Fetches it via ``tengri.download_ssp`` when ``TENGRI_DOWNLOAD_SSP=1`` is
    set in the environment (mirrors ``reproduction/prospect_r/01_prospect_r.py``).
    """
    if not _SSP_PATH.is_file():
        if os.environ.get("TENGRI_DOWNLOAD_SSP") == "1":
            import tengri

            tengri.download_ssp("bc03_pdva_stelib_chabrier", dest=str(_DATA_DIR))
        else:
            pytest.skip(
                f"BC03 SSP not found: {_SSP_PATH}. Set TENGRI_DOWNLOAD_SSP=1 to "
                "fetch it via tengri.download_ssp, or place the file manually."
            )
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(_SSP_PATH))


def _build_declining_exp_model(ssp_data, *, tau_gyr, age_gyr, use_short_form=False):
    """Build a bare declining-exp CSP (no dust, no nebular, z=0)."""
    from tengri import DEFAULT, Fixed, SEDModel
    from tengri.observation import Photometry

    obs = Photometry.from_names(["sdss_g", "sdss_r"])
    if use_short_form:
        sfh_dict = {
            "type": "declining_exp",
            "tau_gyr": Fixed(tau_gyr),
            "age_gyr": Fixed(age_gyr),
            "log_total_mass": Fixed(10.0),
        }
    else:
        sfh_dict = {
            "type": "declining_exp",
            "sfh_declining_exp_tau_gyr": Fixed(tau_gyr),
            "sfh_declining_exp_age_gyr": Fixed(age_gyr),
            "sfh_declining_exp_log_total_mass": Fixed(10.0),
        }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="BakedInBackend")
        return SEDModel.build(
            ssp_data=ssp_data,
            observation=obs,
            sfh=sfh_dict,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
            neb={"type": "none"},
            redshift=Fixed(0.0),
        )


def _tengri_csp_sed(ssp_data, *, tau_gyr, age_gyr) -> tuple[np.ndarray, np.ndarray]:
    """Rest-frame (wave_aa, L_nu) for the fully-prefixed declining_exp build."""
    model = _build_declining_exp_model(ssp_data, tau_gyr=tau_gyr, age_gyr=age_gyr)
    params = model.spec.sample(jax.random.PRNGKey(0))
    pred = model.predict(params)
    return np.asarray(pred.wave_rest), np.asarray(pred.rest_sed())


def _norm_at(wave: np.ndarray, sed: np.ndarray, target: float = 5500.0) -> np.ndarray:
    idx = int(np.argmin(np.abs(wave - target)))
    return sed / sed[idx]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "short-name resolver: bare tau_gyr/age_gyr/log_total_mass resolve for "
        "sfh 'delayed' but raise for 'declining_exp' (parameters/groups.py, "
        "out of scope for this task); xfail(strict=True) so a future resolver "
        "fix turns this into a loud XPASS instead of staying silently green"
    ),
)
def test_declining_exp_short_form_keys_resolve(ssp_data):
    """Positive claim: bare short-form sfh keys resolve for BOTH SFH types.

    Currently FALSE for ``declining_exp`` -- only the fully-prefixed
    ``sfh_declining_exp_tau_gyr`` (etc.) works there, even though the
    identical bare spelling resolves for ``sfh={'type': 'delayed', ...}``.
    Every other test in this file uses the fully-prefixed spelling (that
    invariant is exercised, and must keep working, independently of this
    xfail's outcome).
    """
    _build_declining_exp_model(ssp_data, tau_gyr=_TAU_GYR, age_gyr=_AGE_GYR, use_short_form=True)


def test_csp_shape_matches_reference_at_matched_node(ssp_data):
    """tengri ``declining_exp`` CSP shape vs the vendored BC03 reference.

    tau=1.0 Gyr, age=4.8939 Gyr (nearest upstream grid node to 5 Gyr), both
    sides normalized to 1 at 5500 A. Asserts median|log10 ratio| < 0.05 over
    0.15-2.5 um, tightening to < 0.03 after smoothing tengri's finer
    (6900-point) grid onto the reference's coarser (1221-point) native
    sampling -- this only removes point-to-point resolution noise, it does
    not touch the one large localized Balmer-jump-region spike the audit's
    D6 finding attributes to a resolution/line-blending difference between
    library editions.
    """
    from reproduction.agnfitter._drivers import agnfitter_driver as d

    wave_t, sed_t = _tengri_csp_sed(ssp_data, tau_gyr=_TAU_GYR, age_gyr=_AGE_GYR)
    wave_r, sed_r = d.galaxy_template(tau=_TAU_GYR, age=_AGE_GYR * 1e9)

    sed_t_n = _norm_at(wave_t, sed_t)
    sed_r_n = _norm_at(wave_r, sed_r)

    lo_aa, hi_aa = 1500.0, 25000.0  # 0.15-2.5 um
    mask_r = (wave_r >= lo_aa) & (wave_r <= hi_aa)

    log_wave_t = np.log10(wave_t)
    log_sed_t_n = np.log10(sed_t_n)

    # No smoothing: log-log interpolate tengri onto the reference's grid.
    interp_t_on_r = np.interp(np.log10(wave_r[mask_r]), log_wave_t, log_sed_t_n)
    log_ratio = interp_t_on_r - np.log10(sed_r_n[mask_r])
    median_abs = float(np.median(np.abs(log_ratio)))
    assert median_abs < 0.05, (
        f"median|log10 ratio| = {median_abs:.4f} >= 0.05 over 0.15-2.5 um "
        f"(tau={_TAU_GYR} Gyr, age={_AGE_GYR:.4f} Gyr, normalized at 5500 A)"
    )

    # Smoothed: bin-average tengri within each reference wavelength cell.
    idx_r_full = np.where(mask_r)[0]
    smoothed = np.empty(idx_r_full.shape)
    for i, ridx in enumerate(idx_r_full):
        w_lo = wave_r[0] if ridx == 0 else 0.5 * (wave_r[ridx - 1] + wave_r[ridx])
        w_hi = wave_r[-1] if ridx == len(wave_r) - 1 else 0.5 * (wave_r[ridx] + wave_r[ridx + 1])
        sel = (wave_t >= w_lo) & (wave_t < w_hi)
        smoothed[i] = (
            np.mean(log_sed_t_n[sel])
            if np.any(sel)
            else np.interp(np.log10(wave_r[ridx]), log_wave_t, log_sed_t_n)
        )
    log_ratio_smoothed = smoothed - np.log10(sed_r_n[mask_r])
    median_abs_smoothed = float(np.median(np.abs(log_ratio_smoothed)))
    assert median_abs_smoothed < 0.03, (
        f"smoothed median|log10 ratio| = {median_abs_smoothed:.4f} >= 0.03"
    )


def test_reference_sfr_monotonically_declining_at_tau_1gyr():
    """Pins the SFH functional form the notebook must use (D1 in the audit).

    The vendored pickle's own SFR array declines monotonically with age at
    tau=1 Gyr -- classic declining-exponential, not delayed-tau (which would
    rise then fall). A future re-vendor that silently changes the tabulated
    SFH form would be caught here.
    """
    from reproduction.agnfitter._drivers import agnfitter_driver as d

    age_axis, sfr = d.galaxy_sfr(tau=_TAU_GYR)
    assert age_axis.shape == sfr.shape
    assert np.all(np.diff(sfr) <= 0), (
        "SFR(age) at tau=1 Gyr is not monotonically declining -- the vendored "
        "GALAXY reference no longer reflects a declining-exponential SFH"
    )
