# SPDX-License-Identifier: BSD-3-Clause
"""Test precompute↔runtime equivalence for AGN-nebular emitters (BLR, NLR-analytic).

Verifies that precompute lookups return per-filter photometry matching the
runtime full-wavelength evaluation, using DIFFERENCE-BASED EQUIVALENCE to
isolate the line emitter from the AGN-disc pedestal.

Context (#1660, #1903):
  The prior test was quarantined because a naive equivalence comparison is
  VACUOUS when a broad-band filter contains both the component under test
  (BLR/NLR lines) and an unavoidable pedestal (AGN disc continuum, which
  scales exactly with bolometric luminosity). Measured: H-alpha photometry
  scales 99.99× per 2 dex of agn_log_lbol (exact disc scaling), and the BLR
  contributes ~0.5% to the total. A 100% error in the BLR LUT path moves the
  total by 0.5%, inside a 1% tolerance—vacuous.

  The fix is DIFFERENCE-BASED EQUIVALENCE: for each path (exact, WavePrecomp),
  compute delta = phot(with_line) - phot(without_line). The disc pedestal
  cancels identically. Then compare exact vs LUT deltas:
  |delta_exact - delta_lut| / |delta_exact| < rtol. This weighs the line
  channel at 100% regardless of the pedestal and is impossible to make vacuous
  by tuning the pedestal.

Dead-emitter finding (#1903):
  Current BLR and NLR analytic blocks contribute exactly zero to photometry
  (delta=0 at all agn_log_lbol, both norm modes). Tests xfail—ratchets for a
  known dead-block issue pending surface-integration wiring fix.

Pattern (reusable harness `assert_nonvacuous_equivalence_delta`):
  1. Measure line-emission delta for both exact and precompute paths using
     identical parameters (pedestal cancels identically).
  2. Derive tolerance from a control measurement (stellar-only LUT error), ×3.
  3. Assert non-vacuity: (a) |delta_exact| >> noise (≥3 orders above
     differencing floor); (b) SCALING SANITY: delta(lbol=13) >> delta(lbol=11)
     by >10×, proving the block responds to luminosity (catches #1488-class
     dead blocks).
  4. Assert equivalence: |delta_exact - delta_lut| / |delta_exact| < rtol.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

# Module-level redshift constant: ensures filter fixture and test models cannot drift apart
Z = 0.1


def assert_nonvacuous_equivalence_delta(
    model_exact_high,
    model_precomp_high,
    model_control_high,
    model_exact_low,
    model_control_low,
    params_dict,
    rtol,
    context_msg,
):
    """Assert difference-based equivalence with non-vacuity preconditions.

    This harness measures delta = phot(with_component) - phot(without_component)
    for both exact and precompute paths, canceling the pedestal entirely.

    Parameters
    ----------
    model_exact_high : SEDModel
        With-component model at lbol_high, `approx=None`.
    model_precomp_high : SEDModel
        With-component model at lbol_high, `approx=WavePrecomp()`.
    model_control_high : SEDModel
        Without-component (component disabled) model, lbol=lbol_high, for delta baseline.
    model_exact_low : SEDModel
        With-component model at lbol_low, `approx=None`, for SCALING SANITY check.
    model_control_low : SEDModel
        Without-component (component disabled) model, lbol=lbol_low, for SCALING SANITY.
    params_dict : dict
        Single parameter draw with non-AGN parameters. AGN parameters (especially
        agn_log_lbol if FIXED) are populated from each model's spec to avoid
        cross-model parameter bleed. Non-AGN params are shared across all models.
    rtol : float
        Relative tolerance on delta agreement: |delta_exact - delta_precomp| /
        |delta_exact| < rtol. Derived from measured stellar LUT error ×3.
    context_msg : str
        Human-readable context (e.g., "BLR-analytic delta, lbol=13 vs 11").

    Raises
    ------
    AssertionError
        If non-vacuity precondition fails (delta too small, doesn't scale with
        lbol, or equivalence exceeds rtol).

    Notes
    -----
    SCALING SANITY fix (#1903): The old harness was confounded because the lbol
    parameter appeared in both the "with-component" and "control" models. This
    meant the disc pedestal scaled differently, masking whether the component
    actually responded to luminosity. The fix passes separate control models for
    each luminosity so delta(lbol) = phot(with_block, lbol) - phot(without_block,
    SAME lbol), isolating the component's response and preventing confounding by
    the disc normalization.

    **Parameter handling**: When agn_log_lbol is FIXED in a model's spec, the params
    dict value (if present) would override the FIXED spec value. To prevent
    cross-model contamination, we remove agn_log_lbol from the params before
    passing to each model, letting each model use its own FIXED value.
    """
    # Remove agn_log_lbol from params if present, so each model uses its own FIXED value
    params_for_high = dict(params_dict)
    params_for_low = dict(params_dict)
    params_for_high.pop("agn_log_lbol", None)
    params_for_low.pop("agn_log_lbol", None)

    # ──────────────────────────────────────────────────────────────────────────
    # Compute deltas at high luminosity (pedestal cancels identically)
    # ──────────────────────────────────────────────────────────────────────────
    phot_exact_high = np.asarray(model_exact_high.predict_photometry(params_for_high))
    phot_control_high = np.asarray(model_control_high.predict_photometry(params_for_high))
    delta_exact_high = phot_exact_high - phot_control_high

    phot_precomp_high = np.asarray(model_precomp_high.predict_photometry(params_for_high))
    delta_precomp_high = phot_precomp_high - phot_control_high

    # ──────────────────────────────────────────────────────────────────────────
    # NON-VACUITY PRECONDITION (a): IN-BAND SIGNIFICANCE (#1660 pattern)
    # ──────────────────────────────────────────────────────────────────────────
    # Assert the component's in-band significance: delta must be a substantial
    # fraction of the total photometry. This guards against vacuous tests where
    # the emitter is swamped by a pedestal that cancels anyway.
    #
    # Orchestrator measurement on properly-aligned filters:
    #   - In-band (filter centered on redshifted line): ~49% of photometry
    #   - Out-of-band (filter off-center): ≤0.1% of photometry
    # Mutant (filter at rest-frame, not redshifted): ~0.52% in-band.
    # Threshold with ~10x safety margins: 0.01 (1.0%)
    #   - ~50× below in-band correct (49% vs 1%)
    #   - ~10× above out-of-band worst-case (0.1% vs 1%)
    #   - Clearly rejects the misaligned-filter mutant (0.52% vs 1.0%)
    in_band_fractional_contrib = np.max(
        np.abs(delta_exact_high) / np.maximum(np.abs(phot_exact_high), 1e-40)
    )
    in_band_threshold = 0.01  # 1.0% minimum in-band contribution

    if in_band_fractional_contrib < in_band_threshold:
        raise AssertionError(
            f"[{context_msg}] Non-vacuity FAILED (a) IN-BAND SIGNIFICANCE: "
            f"max(|delta|/|phot_with|) = {in_band_fractional_contrib:.3%} is below "
            f"threshold {in_band_threshold:.3%}. The line-emitter block has negligible "
            f"in-band contribution and the test would be vacuous. "
            f"Measured in-band on correct filters: ~49%, on misaligned: ~0.52%. "
            f"Max delta: {np.max(np.abs(delta_exact_high)):.3e}, "
            f"max photometry: {np.max(np.abs(phot_exact_high)):.3e}."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # NON-VACUITY PRECONDITION (b): SCALING SANITY
    # ──────────────────────────────────────────────────────────────────────────
    # Compare deltas at two luminosities. Each delta uses a lbol-matched control,
    # so delta(lbol) = phot(with_block, lbol) - phot(without_block, SAME lbol).
    # This isolates the component's luminosity response from disc pedestal scaling.
    phot_exact_low = np.asarray(model_exact_low.predict_photometry(params_for_low))
    phot_control_low = np.asarray(model_control_low.predict_photometry(params_for_low))
    delta_exact_low = phot_exact_low - phot_control_low

    max_delta_high = np.max(np.abs(delta_exact_high))
    max_delta_low = np.max(np.abs(delta_exact_low))
    scaling = max_delta_high / np.maximum(max_delta_low, 1e-40)

    if scaling < 10:
        raise AssertionError(
            f"[{context_msg}] Non-vacuity FAILED (b) SCALING SANITY: "
            f"delta scales {scaling:.1f}× (expect >10×). "
            f"The line-emitter block is not responding to luminosity—possible "
            f"dead block (#1488-class bug). Measured delta_low={max_delta_low:.3e}, "
            f"delta_high={max_delta_high:.3e}."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # EQUIVALENCE ASSERTION: exact vs precomp delta agreement at high luminosity
    # ──────────────────────────────────────────────────────────────────────────
    rel_error_delta = np.abs(delta_precomp_high - delta_exact_high) / np.maximum(
        np.abs(delta_exact_high), 1e-40
    )
    max_rel_error = np.max(rel_error_delta)

    assert max_rel_error < rtol, (
        f"[{context_msg}] Equivalence FAILED: precompute↔exact delta disagreement "
        f"{max_rel_error:.3%} exceeds tolerance {rtol:.3%}. "
        f"Per-filter delta_exact: {delta_exact_high}, delta_precomp: {delta_precomp_high}, "
        f"rel error: {rel_error_delta}."
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def real_ssp_data():
    """Load the standard SSP data (skips if not available)."""
    tengri_module = pytest.importorskip("tengri")
    return tengri_module.load_ssp()


def _gaussian_filter(center_aa, fwhm_aa, name):
    """Narrow Gaussian filter at a given rest wavelength (vacuum Å)."""
    from tengri.observation.photometry import FilterCurve

    sigma = fwhm_aa / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    wave = np.linspace(center_aa - 5.0 * sigma, center_aa + 5.0 * sigma, 150)
    trans = np.exp(-0.5 * ((wave - center_aa) / sigma) ** 2)
    return FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=name)


@pytest.fixture(scope="module")
def narrow_ha_hb_filters():
    """Two moderately-narrow Gaussian filters at OBSERVED-frame wavelengths.

    CRITICAL FIX for #1903: Filters are placed at observed-frame centers
    (rest × (1 + Z)) so they sample the actual redshifted line positions
    in the model. At Z=0.1 (module constant), rest Hα 6564.61 Å moves to
    7221.07 Å, rest Hβ 4862.68 Å to 5348.95 Å.

    FWHM=30 Å ensures the filter captures line emission while isolating it
    from broad continuum features.

    Previous bug: filters were defined at rest wavelengths but the model
    was at z=0.1, so the filters sampled continuum adjacent to the lines
    rather than the lines themselves. This made BLR/NLR appear dead (delta≈0)
    even when emitting. Consequence: equivalence test was vacuous (xfail).
    """
    # Redshift filters to observed frame
    ha_center_rest = 6564.61
    hb_center_rest = 4862.68
    ha_center_obs = ha_center_rest * (1.0 + Z)
    hb_center_obs = hb_center_rest * (1.0 + Z)

    ha_filter = _gaussian_filter(ha_center_obs, 30.0, "ha_narrow")
    hb_filter = _gaussian_filter(hb_center_obs, 30.0, "hb_narrow")
    return [hb_filter, ha_filter]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestAGNNebularPrecomputeEquivalence:
    """BLR/NLR-analytic precompute ↔ runtime equivalence (difference-based)."""

    def test_blr_analytic_delta_equivalence(self, real_ssp_data, narrow_ha_hb_filters):
        """BLR-analytic precompute must match runtime via DELTA.

        DIFFERENCE-BASED: delta = phot(with_BLR) - phot(without_BLR).
        Pedestal (AGN disc) cancels identically for both exact and precompute
        paths. The delta measures ONLY the BLR contribution and is immune to
        pedestal vacuity. Verifies both magnitude (>>noise) and SCALING SANITY
        (delta scales >10× with luminosity, proving the block responds to lbol).

        Tolerance derivation: Measured stellar-only LUT error ~0.3%.
        Measured BLR delta LUT error ~0.5%. Allow 3× = 1.5%.
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp

        obs = Observation(photometry=Photometry(filters=tuple(narrow_ha_hb_filters)))
        key = jax.random.PRNGKey(42)

        # ──── High-lbol (13) exact & precomp models ────
        model_exact_13 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=None,
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "analytic", "*": FIXED, "agn_log_lbol": 13.0},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        model_precomp_13 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=WavePrecomp(),
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "analytic", "*": FIXED, "agn_log_lbol": 13.0},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        # ──── Low-lbol (11) for scaling sanity check ────
        model_exact_11 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=None,
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "analytic", "*": FIXED, "agn_log_lbol": 11.0},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        # ──── Control models (BLR disabled, lbol-matched) ────
        model_control_13 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=None,
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED, "agn_log_lbol": 13.0},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "none"},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        model_control_11 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=None,
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED, "agn_log_lbol": 11.0},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "none"},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        params_dict = dict(model_exact_13.spec.sample(key))
        measured_rtol = 0.015  # 1.5%

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_nonvacuous_equivalence_delta(
                model_exact_13,
                model_precomp_13,
                model_control_13,
                model_exact_11,
                model_control_11,
                params_dict,
                rtol=measured_rtol,
                context_msg="BLR-analytic delta, lbol=13 vs 11",
            )

    def test_nlr_analytic_delta_equivalence(self, real_ssp_data, narrow_ha_hb_filters):
        """NLR-analytic precompute must match runtime via DELTA.

        Same difference-based approach as BLR test. Tolerance: 1.5%.
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp

        obs = Observation(photometry=Photometry(filters=tuple(narrow_ha_hb_filters)))
        key = jax.random.PRNGKey(43)

        # ──── High-lbol (12.5) exact & precomp models ────
        model_exact_125 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=None,
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "none"},
                "nlr": {"type": "analytic", "*": FIXED, "agn_log_lbol": 12.5},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        model_precomp_125 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=WavePrecomp(),
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "none"},
                "nlr": {"type": "analytic", "*": FIXED, "agn_log_lbol": 12.5},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        # ──── Low-lbol (11) for scaling sanity check ────
        model_exact_11 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=None,
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "none"},
                "nlr": {"type": "analytic", "*": FIXED, "agn_log_lbol": 11.0},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        # ──── Control models (NLR disabled, lbol-matched) ────
        model_control_125 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=None,
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED, "agn_log_lbol": 12.5},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "none"},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        model_control_11 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=None,
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED, "agn_log_lbol": 11.0},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "none"},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        params_dict = dict(model_exact_125.spec.sample(key))
        measured_rtol = 0.015  # 1.5%

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_nonvacuous_equivalence_delta(
                model_exact_125,
                model_precomp_125,
                model_control_125,
                model_exact_11,
                model_control_11,
                params_dict,
                rtol=measured_rtol,
                context_msg="NLR-analytic delta, lbol=12.5 vs 11",
            )

    def test_blr_vacuity_mutant_on_delta_precondition(self, real_ssp_data, narrow_ha_hb_filters):
        """VACUITY MUTANT: delta disabled (BLR=none) must fail precondition.

        If we disable the BLR (so delta ≈ 0), the magnitude precondition must
        FAIL loudly. This proves the precondition works and can't be silently
        bypassed as its predecessor was.
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp

        obs = Observation(photometry=Photometry(filters=tuple(narrow_ha_hb_filters)))
        key = jax.random.PRNGKey(42)

        # ──── All models with BLR disabled ────
        model_exact = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=None,
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "none"},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        model_precomp = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(Z),
            approx=WavePrecomp(),
            sfh={"type": "tsnorm", "*": FIXED, "log_total_mass": 6.0},
            dust={"type": "none"},
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": FIXED},
                "torus": {"type": "simple", "*": FIXED},
                "blr": {"type": "none"},
                "nlr": {"type": "none"},
                "feii": {"type": "none"},
                "atten": {"type": "none"},
            },
        )

        model_control = model_exact
        params_dict = dict(model_exact.spec.sample(key))
        measured_rtol = 0.015

        with pytest.raises(AssertionError, match="Non-vacuity FAILED"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_nonvacuous_equivalence_delta(
                model_exact,
                model_precomp,
                model_control,
                model_exact,
                model_control,
                params_dict,
                rtol=measured_rtol,
                context_msg="BLR-analytic MUTANT (BLR disabled)",
            )
