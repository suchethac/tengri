# SPDX-License-Identifier: BSD-3-Clause
"""The ionizing-spectrum precompute must not degrade on a float32 SSP (#1206).

``fit_ionizing_spectrum`` fits a piecewise power law to the ionizing continuum
and publishes ``log_seglum`` -- the absolute integrated luminosity per segment.
Two of its guard floors are written below what float32 can hold::

    log_flux = np.log10(np.maximum(seg_flux, 1e-99))  # _fit_segment
    normalized = np.clip(flux * norm, 1e-70 * norm, np.inf)  # fit_ionizing_spectrum

float32's smallest subnormal is 1.4e-45, so ``float32(1e-99)`` and
``float32(1e-70 * norm)`` are both **exactly 0.0**. The floors evaporate, zero
flux survives the clamp, ``log10(0) = -inf`` enters the least-squares objective,
and the fit returns a degenerate slope. Measured on
``fsps_prsc_miles_chabrier.h5`` with float32 input:

===================  =========================  ===============================
table                float64 input              float32 input
===================  =========================  ===============================
``seglum_table``     finite, range [-99, 36.2]  **198 entries -inf**, max 102.7
``ionspec_table``    range [0, 42]              range [-1.0, 42], rel err 1.27
``logqion_table``    range [-99, 46.9]          unchanged (rel err 5.9e-09)
===================  =========================  ===============================

A ``log_seglum`` of 102.7 dex is 10**102 erg/s. Downstream, Cue's weighted
segment sum goes non-finite and the whole SED with it -- which is the
"pure-float32 Cue forward is NaN" symptom #1206 is blocked on.

``logqion_table`` is *unaffected* because #458 already cast that one integration
to float64 explicitly. The comment beside it asserts "the slope fits above are
float32-safe because ``normalized = flux * 1e-18 / ref_flux`` rescales each
segment before fitting". That claim is the leftover this file measures: the
rescale does keep the *slope* fit in range, but it does not make the two floors
representable, and ``log_seglum`` (added later, #1018) never got #458's cast.

Why the existing guard did not catch it
---------------------------------------
``test_representable_exponent.py::test_cue_forward_is_finite_in_pure_float32``
asserts exactly this end-to-end property and passes. It takes ``ssp_bare`` as a
**module-scoped fixture**, so the SSP is loaded while x64 is still enabled and
its arrays are float64; only the forward pass runs inside
``jax.enable_x64(False)``. The precompute therefore never sees a float32 SSP.
A process started with ``JAX_ENABLE_X64=0`` -- the actual pure-float32 mode --
loads the SSP in float32 and does. The fixture avoided the hard case, so the
test could not fail. Every test here loads or casts the SSP **inside** the
float32 context for that reason.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_SSP_PATH = "data/fsps_prsc_miles_chabrier.h5"

#: A segment is compared across dtypes only if it carries at least this share of
#: the largest segment in its own (Z, age) bin.
#:
#: Not a tolerance that was widened until the test passed — it is the difference
#: between the two things this file must not conflate. Measured after the fix:
#: zero entries lose finiteness, but 227 of 5580 still differ by >1e-3 dex, and
#: **every one of them sits at a luminosity share of 5.5e-15 or below** (the
#: worst, 8.2 dex, at 3.3e-18). Those are segments whose ionizing flux is
#: numerically zero — the HeII segment of a 10-100 Myr population — so the power
#: law is being fit to nothing and its normalization is unconstrained. Requiring
#: their logs to agree measures the conditioning of a fit to zero, not physics.
#:
#: End-to-end confirmation that the excluded entries are inert: photometry from a
#: float32-stored SSP versus the same SSP in float64 agrees to 7.8e-08 - 4.0e-07,
#: i.e. plain float32 input rounding, with no trace of an 8-dex segment.
#:
#: The strong assertion — no finite float64 entry may become non-finite — is
#: applied to EVERY entry regardless of share, because that is the actual defect.
_CONTRIBUTING_SHARE = 1e-6


def _ionizing_spectra_f64(max_bins=24):
    """Real (Z, age) ionizing spectra in float64, swept rather than hand-picked.

    Deliberately a sweep. The first draft of this file took the single youngest
    bin with ionizing flux and PASSED, because only 198 of 5580 segment entries
    degrade -- picking one spectrum has a good chance of missing every one of
    them. A test that selects its own subject can select away the defect, so
    this walks a spread of metallicities and young ages instead.
    """
    from tengri.components.nebular.ionizing_spectrum import HI_LIMIT, MAX_NEB_LOG_AGE
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    with jax.enable_x64(True):
        ssp = load_ssp_data(_SSP_PATH)
        wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
        flux = np.asarray(ssp.ssp_flux, dtype=np.float64)
        log_age = np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64) + 9.0

    ionizing = wave <= HI_LIMIT
    young = np.where(log_age <= MAX_NEB_LOG_AGE)[0]

    # A GRID striding the whole young range: a few metallicities x ages spread
    # from the youngest bin to the oldest.
    #
    # Two earlier drafts of this sweep passed while the full-grid comparison
    # beside it failed on 198 entries, and both failures were selection, not
    # tolerance. The degradation lives at log_age 7.0-8.0 (10-100 Myr, age
    # indices 30-50 on this grid) in segment 0, where the ionizing flux is
    # ~1e-17.7 of the normalization and underflows float32. Draft 1 took the
    # single youngest bin with ionizing flux; draft 2 filled greedily (all 15
    # metallicities at age 0, then age 6, ...) and hit the cap two ages in.
    # Both sampled only the young, bright populations -- precisely the ones that
    # do not break. Selection by convenience selects the defect away.
    n_met = flux.shape[0]
    n_met_samples = 3
    n_age_samples = max(1, max_bins // n_met_samples)
    met_samples = sorted({round(float(i)) for i in np.linspace(0, n_met - 1, n_met_samples)})
    age_samples = sorted({int(i) for i in np.linspace(young[0], young[-1], n_age_samples)})

    out = []
    for ia in age_samples:
        for im in met_samples:
            spec = flux[im, ia, :]
            if np.max(spec[ionizing]) > 0:
                out.append((im, int(ia), wave, spec))
    if not out:
        pytest.skip("no ionizing flux in this SSP")
    return out


def test_the_float32_floors_really_do_evaporate():
    """Non-vacuity control: the two floors ARE zero in float32.

    If numpy ever gained a representable-clamp semantic, every assertion below
    would pass for the wrong reason. Pin the premise.
    """
    assert np.float32(1e-99) == 0.0, "float32(1e-99) is no longer 0.0"
    assert np.float32(1e-70 * 1e-5) == 0.0, "float32(1e-70 * norm) is no longer 0.0"
    zeros32 = np.zeros(3, dtype=np.float32)
    assert np.all(np.maximum(zeros32, 1e-99) == 0.0), (
        "np.maximum(float32 zeros, 1e-99) now clamps above zero, so log10 of it "
        "is no longer -inf and this file's premise is gone"
    )


def test_fit_ionizing_spectrum_is_dtype_invariant():
    """The root cause: the same spectrum in float32 must fit the same as float64.

    This is the attributable test. It calls the fit directly -- no model, no
    table cache, no JAX precision context -- so a failure can only mean the fit
    itself depends on the input dtype.
    """
    from tengri.components.nebular.ionizing_spectrum import fit_ionizing_spectrum

    bad = []
    checked = 0
    for im, ia, wave64, flux64 in _ionizing_spectra_f64():
        wave32 = wave64.astype(np.float32)
        flux32 = flux64.astype(np.float32)

        # Control: the float32 cast must not itself lose the spectrum, or a
        # failure would be about representability rather than about the fit.
        assert np.all(np.isfinite(flux32)), f"({im},{ia}) float32 cast made input non-finite"
        assert np.max(flux32) > 0, f"({im},{ia}) float32 cast zeroed the spectrum"

        fit64 = fit_ionizing_spectrum(wave64, flux64)
        fit32 = fit_ionizing_spectrum(wave32, flux32)
        seg64 = np.asarray(fit64["log_seglum"], dtype=np.float64)
        seg32 = np.asarray(fit32["log_seglum"], dtype=np.float64)
        assert np.all(np.isfinite(seg64)), f"({im},{ia}) float64 reference broken: {seg64}"

        checked += 1
        # Finiteness is absolute: a finite float64 segment may never come back
        # non-finite. Agreement is asserted only where the segment CARRIES
        # something -- see ``_CONTRIBUTING_SHARE`` for why.
        contributing = (seg64 - seg64.max()) > np.log10(_CONTRIBUTING_SHARE)
        if not np.all(np.isfinite(seg32)) or not np.allclose(
            seg32[contributing], seg64[contributing], rtol=1e-3, atol=1e-3
        ):
            bad.append((im, ia, seg64, seg32))
        # gas_logqion was already float64-cast by #458 — the control showing it
        # is the rest of the fit that regressed, not the whole routine.
        assert np.isclose(
            float(fit32["gas_logqion"]), float(fit64["gas_logqion"]), rtol=1e-5, atol=1e-5
        ), f"({im},{ia}) gas_logqion drifted, which #458 should already prevent"

    assert checked > 0, "swept no spectra — the fixture selected nothing"
    assert not bad, (
        f"log_seglum depends on input dtype for {len(bad)}/{checked} spectra. "
        f"First: (met={bad[0][0]}, age={bad[0][1]}) float64={bad[0][2]} "
        f"float32={bad[0][3]}. The 1e-99 / 1e-70 guard floors are exactly 0.0 in "
        "float32, so zero flux survives the clamp and log10(0) = -inf enters the "
        "least-squares objective. A segment luminosity is a physical quantity; it "
        "cannot depend on how the SSP happened to be stored."
    )


def test_precompute_table_is_dtype_invariant():
    """The table the backend actually consumes must not depend on SSP dtype.

    Guards the seam between the per-spectrum fit and ``CueBackend``: the table is
    memoized on an SSP fingerprint, so a corrupt float32-built table would also
    persist to the on-disk cache.
    """
    from tengri.components.nebular.ionizing_spectrum import precompute_ionizing_params_table
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    with jax.enable_x64(True):
        ssp = load_ssp_data(_SSP_PATH)
        wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
        flux = np.asarray(ssp.ssp_flux, dtype=np.float64)
        lgmet = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
        logage = np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64) + 9.0

    r64 = precompute_ionizing_params_table(wave, flux, lgmet, ssp_log_age_yr=logage)
    r32 = precompute_ionizing_params_table(
        wave.astype(np.float32),
        flux.astype(np.float32),
        lgmet.astype(np.float32),
        ssp_log_age_yr=logage,
    )

    # The defect itself, asserted on EVERY entry of every table: finiteness.
    # This is what went wrong (198 -inf) and it admits no tolerance.
    for key in ("seglum_table", "ionspec_table", "logqion_table"):
        a = np.asarray(r64[key], dtype=np.float64)
        b = np.asarray(r32[key], dtype=np.float64)
        lost = int(np.sum(np.isfinite(a) & ~np.isfinite(b)))
        assert lost == 0, (
            f"{key}: {lost} entries are finite from a float64 SSP and non-finite "
            f"from the same SSP in float32. The ionizing-spectrum fit degraded on "
            "float32 input."
        )

    # log Q_H: already float64-cast by #458, so it is the control — it must agree
    # tightly, showing the comparison is not vacuous.
    q64 = np.asarray(r64["logqion_table"], dtype=np.float64)
    q32 = np.asarray(r32["logqion_table"], dtype=np.float64)
    assert np.allclose(q32, q64, rtol=1e-5, atol=1e-5), (
        f"logqion_table drifted (worst {np.max(np.abs(q64 - q32)):.3e} dex); #458's "
        "float64 cast should already make this dtype-invariant, so the control for "
        "the looser seglum comparison below is gone."
    )

    # Segment luminosities: compared where the segment carries something.
    s64 = np.asarray(r64["seglum_table"], dtype=np.float64)
    s32 = np.asarray(r32["seglum_table"], dtype=np.float64)
    binmax = np.max(s64, axis=-1, keepdims=True)
    contributing = (s64 - binmax) > np.log10(_CONTRIBUTING_SHARE)
    assert contributing.any(), "no contributing segments selected — comparison is vacuous"
    worst = float(np.max(np.abs(s64 - s32)[contributing]))
    assert worst < 1e-3, (
        f"seglum_table depends on SSP storage dtype for a segment that actually "
        f"carries luminosity: worst {worst:.4e} dex among segments above "
        f"{_CONTRIBUTING_SHARE:g} of their bin."
    )


def test_cue_forward_is_finite_when_the_ssp_itself_is_float32():
    """End-to-end, with the SSP loaded INSIDE the float32 context.

    The distinction from the existing guard in
    ``test_representable_exponent.py`` is the whole point: there the SSP fixture
    is built under x64 and only the forward runs in float32, so the precompute
    is fed float64 and the defect is invisible. Here the load happens inside
    ``enable_x64(False)``, which is what a ``JAX_ENABLE_X64=0`` process does.
    """
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    params = {"sfh_delayed_log_total_mass": 10.0, "dust_tau_diff": 0.5}

    with jax.enable_x64(False):
        ssp32 = load_ssp_data(_SSP_PATH)
        # Control: this really is the float32 path, else the test is vacuous.
        assert np.asarray(ssp32.ssp_flux).dtype == np.float32, (
            "the SSP loaded inside enable_x64(False) is not float32, so this test "
            "no longer exercises the path the existing guard misses"
        )
        model = SEDModel.build(
            ssp_data=ssp32,
            observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
            sfh={
                "type": "delayed",
                "all_params": FIXED,
                "log_total_mass": Uniform(9.0, 11.0),
                "tau_gyr": 1.0,
                "age_gyr": 5.0,
            },
            dust={"law_diff": 'calzetti', 
                "type": "two_component",
                "law_bc": "calzetti",
                "all_params": FIXED,
                "tau_diff": Uniform(0.0, 1.5),
                "tau_bc": 0.0,
            },
            neb={"type": "cue", "all_params": FIXED},
            redshift=Fixed(0.1),
            approx=None,
        )
        phot = np.asarray(model.predict_photometry(params), dtype=np.float64)

    assert np.isfinite(phot).all(), (
        f"pure-float32 Cue photometry is non-finite from a float32 SSP: {phot}. "
        "The ionizing-spectrum precompute degraded on float32 input."
    )
    assert np.all(phot > 0), f"pure-float32 Cue photometry collapsed to zero: {phot}"
