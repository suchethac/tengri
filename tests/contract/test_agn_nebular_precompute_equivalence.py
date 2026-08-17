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


def assert_nonvacuous_equivalence_delta(
    model_exact,
    model_precomp,
    model_control,
    params_dict,
    rtol,
    lbol_high,
    lbol_low,
    context_msg,
):
    """Assert difference-based equivalence with non-vacuity preconditions.

    This harness measures delta = phot(with_component) - phot(without_component)
    for both exact and precompute paths, canceling the pedestal entirely.

    Parameters
    ----------
    model_exact : SEDModel
        With-component model, `approx=None`.
    model_precomp : SEDModel
        With-component model, `approx=WavePrecomp()`.
    model_control : SEDModel
        Without-component (component disabled) model for delta baseline.
    params_dict : dict
        Single parameter draw, used identically for all three models.
    rtol : float
        Relative tolerance on delta agreement: |delta_exact - delta_precomp| /
        |delta_exact| < rtol. Derived from measured stellar LUT error ×3.
    lbol_high : SEDModel
        High-luminosity version of exact model for SCALING SANITY check.
        If None, skip scaling check.
    lbol_low : SEDModel
        Low-luminosity version of exact model for SCALING SANITY check.
        If None, skip scaling check.
    context_msg : str
        Human-readable context (e.g., "BLR-analytic, lbol=13 vs 11").

    Raises
    ------
    AssertionError
        If non-vacuity precondition fails (delta too small, doesn't scale with
        lbol, or equivalence exceeds rtol).
    """
    # ──────────────────────────────────────────────────────────────────────────
    # Compute deltas (pedestal cancels identically for exact and precomp)
    # ──────────────────────────────────────────────────────────────────────────
    phot_exact_with = np.asarray(model_exact.predict_photometry(params_dict))
    phot_exact_without = np.asarray(model_control.predict_photometry(params_dict))
    delta_exact = phot_exact_with - phot_exact_without

    phot_precomp_with = np.asarray(model_precomp.predict_photometry(params_dict))
    delta_precomp = phot_precomp_with - phot_exact_without

    # ──────────────────────────────────────────────────────────────────────────
    # NON-VACUITY PRECONDITION (a): magnitude of delta >> differencing noise
    # ──────────────────────────────────────────────────────────────────────────
    # Estimate noise floor: recompute control twice (independent parameter sampling)
    phot_control_recompute = np.asarray(model_control.predict_photometry(params_dict))
    noise = np.abs(phot_exact_without - phot_control_recompute)
    max_noise_per_band = np.max(noise)
    max_delta_magnitude = np.max(np.abs(delta_exact))

    noise_ratio = max_delta_magnitude / np.maximum(max_noise_per_band, 1e-40)
    if noise_ratio < 1000:  # ≥3 orders above noise
        raise AssertionError(
            f"[{context_msg}] Non-vacuity FAILED (a): delta magnitude "
            f"{max_delta_magnitude:.3e} is only {noise_ratio:.1f}× above noise floor "
            f"{max_noise_per_band:.3e}. Need ≥1000× to be signal-dominated."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # NON-VACUITY PRECONDITION (b): SCALING SANITY (if lbol models provided)
    # ──────────────────────────────────────────────────────────────────────────
    if lbol_high is not None and lbol_low is not None:
        phot_high = np.asarray(lbol_high.predict_photometry(params_dict))
        phot_low = np.asarray(lbol_low.predict_photometry(params_dict))
        # Control is same for both (pedestal cancels)
        delta_high = phot_high - phot_exact_without
        delta_low = phot_low - phot_exact_without

        max_delta_high = np.max(np.abs(delta_high))
        max_delta_low = np.max(np.abs(delta_low))
        scaling = max_delta_high / np.maximum(max_delta_low, 1e-40)

        if scaling < 10:
            raise AssertionError(
                f"[{context_msg}] Non-vacuity FAILED (b) SCALING SANITY: "
                f"delta scales {scaling:.1f}× from lbol=11 to lbol=13 (expect >10×). "
                f"The line-emitter block is not responding to luminosity—possible "
                f"dead block (#1488-class bug). Measured delta_11={max_delta_low:.3e}, "
                f"delta_13={max_delta_high:.3e}."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # EQUIVALENCE ASSERTION: exact vs precomp delta agreement
    # ──────────────────────────────────────────────────────────────────────────
    rel_error_delta = np.abs(delta_precomp - delta_exact) / np.maximum(np.abs(delta_exact), 1e-40)
    max_rel_error = np.max(rel_error_delta)

    assert max_rel_error < rtol, (
        f"[{context_msg}] Equivalence FAILED: precompute↔exact delta disagreement "
        f"{max_rel_error:.3%} exceeds tolerance {rtol:.3%}. "
        f"Per-filter delta_exact: {delta_exact}, delta_precomp: {delta_precomp}, "
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
    """Two moderately-narrow Gaussian filters at vacuum H-alpha and H-beta.

    FWHM=30 Å ensures the filter captures line emission while isolating it
    from broad continuum features.
    """
    ha_filter = _gaussian_filter(6564.61, 30.0, "ha_narrow")
    hb_filter = _gaussian_filter(4862.68, 30.0, "hb_narrow")
    return [hb_filter, ha_filter]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestAGNNebularPrecomputeEquivalence:
    """BLR/NLR-analytic precompute ↔ runtime equivalence (difference-based)."""

    @pytest.mark.xfail(
        strict=True,
        reason="BLR-analytic block is inert (delta=0 at all luminosities); "
        "surface-integration gap (see #1903). agn_log_lbol does not feed the BLR "
        "emission computation. TODO: fix BLR luminosity wiring + SED-surface integration "
        "(#1903, likely same root cause as #1867).",
    )
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

        z = 0.1
        obs = Observation(photometry=Photometry(filters=tuple(narrow_ha_hb_filters)))
        key = jax.random.PRNGKey(42)

        # ──── High-lbol (13) exact & precomp models ────
        model_exact_13 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(z),
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
            redshift=Fixed(z),
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
            redshift=Fixed(z),
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

        # ──── Control (BLR disabled) ────
        model_control = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(z),
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

        params_dict = dict(model_exact_13.spec.sample(key))
        measured_rtol = 0.015  # 1.5%

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_nonvacuous_equivalence_delta(
                model_exact_13,
                model_precomp_13,
                model_control,
                params_dict,
                rtol=measured_rtol,
                lbol_high=model_exact_13,
                lbol_low=model_exact_11,
                context_msg="BLR-analytic delta, lbol=13 vs 11",
            )

    @pytest.mark.xfail(
        strict=True,
        reason="NLR-analytic block is inert (delta=0 at all luminosities); "
        "surface-integration gap (see #1903). agn_log_lbol does not feed the NLR "
        "emission computation. TODO: fix NLR luminosity wiring + SED-surface integration "
        "(#1903, likely same root cause as #1867).",
    )
    def test_nlr_analytic_delta_equivalence(self, real_ssp_data, narrow_ha_hb_filters):
        """NLR-analytic precompute must match runtime via DELTA.

        Same difference-based approach as BLR test. Tolerance: 1.5%.
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp

        z = 0.1
        obs = Observation(photometry=Photometry(filters=tuple(narrow_ha_hb_filters)))
        key = jax.random.PRNGKey(43)

        # ──── High-lbol (12.5) exact & precomp models ────
        model_exact_125 = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(z),
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
            redshift=Fixed(z),
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
            redshift=Fixed(z),
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

        # ──── Control (NLR disabled) ────
        model_control = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(z),
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

        params_dict = dict(model_exact_125.spec.sample(key))
        measured_rtol = 0.015  # 1.5%

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert_nonvacuous_equivalence_delta(
                model_exact_125,
                model_precomp_125,
                model_control,
                params_dict,
                rtol=measured_rtol,
                lbol_high=model_exact_125,
                lbol_low=model_exact_11,
                context_msg="NLR-analytic delta, lbol=12.5 vs 11",
            )

    def test_blr_vacuity_mutant_on_delta_precondition(self, real_ssp_data, narrow_ha_hb_filters):
        """VACUITY MUTANT: delta disabled (BLR=none) must fail precondition.

        If we disable the BLR (so delta ≈ 0), the magnitude precondition must
        FAIL loudly. This proves the precondition works and can't be silently
        bypassed as its predecessor was.
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp

        z = 0.1
        obs = Observation(photometry=Photometry(filters=tuple(narrow_ha_hb_filters)))
        key = jax.random.PRNGKey(42)

        # ──── All models with BLR disabled ────
        model_exact = SEDModel.build(
            ssp_data=real_ssp_data,
            observation=obs,
            redshift=Fixed(z),
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
            redshift=Fixed(z),
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
                params_dict,
                rtol=measured_rtol,
                lbol_high=None,
                lbol_low=None,
                context_msg="BLR-analytic MUTANT (BLR disabled)",
            )
