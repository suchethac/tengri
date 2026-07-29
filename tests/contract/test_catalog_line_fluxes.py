# SPDX-License-Identifier: BSD-3-Clause
"""Catalog fitting with per-galaxy emission-line fluxes (#1480).

Direct plumbing tests for per-galaxy line flux threading. Tests verify the
MECHANISM (data reaches likelihood), not the physics (model predictions).
synthetic_ssp_wide has no nebular emission (Halpha_pred=0), but per-galaxy
chi-squared terms ((obs - 0) / err)**2 differ, proving threading works.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_engine_receives_per_galaxy_line_data(synthetic_ssp_wide,
                                              synthetic_tophat_obs):
    """Test 1: Verify catalog ingests correct per-galaxy line data.

    Check that the catalog's internal _catalog_arrays has the right
    line_flux_obs for each galaxy before any engine runs.
    """
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    cat, _ = build_two_galaxy_catalog(
        halpha=(1.0e-16, 4.0e-16),
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )

    ca = cat._catalog_arrays

    # Verify per-galaxy line fluxes were ingested correctly
    assert ca.line_flux_obs is not None, "line_flux_obs is None"
    assert ca.line_flux_obs.shape == (2, 1), (
        f"Expected shape (2, 1), got {ca.line_flux_obs.shape}"
    )

    # Galaxy 0 should have 1.0e-16
    assert np.allclose(ca.line_flux_obs[0], [1.0e-16]), (
        f"Galaxy 0: got {ca.line_flux_obs[0]}, expected [1.0e-16]"
    )

    # Galaxy 1 should have 4.0e-16
    assert np.allclose(ca.line_flux_obs[1], [4.0e-16]), (
        f"Galaxy 1: got {ca.line_flux_obs[1]}, expected [4.0e-16]"
    )


def test_likelihood_is_sensitive_to_line_data(synthetic_ssp_wide,
                                               synthetic_tophat_obs):
    """Test 2: Per-galaxy likelihood differs based on observed Halpha.

    Evaluate negative log-likelihood at same parameter vector for both
    galaxies. Chi-squared term ((obs - pred) / err)**2 differs between
    obs=1e-16 and obs=4e-16 even though pred=0.
    """
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog
    from tengri.inference.fitter import Fitter

    cat, truth = build_two_galaxy_catalog(
        halpha=(1.0e-16, 4.0e-16),
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )

    ca = cat._catalog_arrays

    # Build fitters and evaluate at truth
    fitter_g0 = Fitter(cat.fwd, ca.flux[0], ca.noise[0],
                       data_type="photometry")
    fitter_g1 = Fitter(cat.fwd, ca.flux[1], ca.noise[1],
                       data_type="photometry")

    nlp_fn_g0 = fitter_g0._get_or_build_logdensity_fn()
    nlp_fn_g1 = fitter_g1._get_or_build_logdensity_fn()

    data_args_g0 = dict(fitter_g0._data_args)
    data_args_g0["line_flux_obs"] = ca.line_flux_obs[0]
    data_args_g0["line_flux_err"] = ca.line_flux_err[0]

    data_args_g1 = dict(fitter_g1._data_args)
    data_args_g1["line_flux_obs"] = ca.line_flux_obs[1]
    data_args_g1["line_flux_err"] = ca.line_flux_err[1]

    nlp_g0 = nlp_fn_g0(truth, data_args_g0)
    nlp_g1 = nlp_fn_g1(truth, data_args_g1)

    # chi2_g0 = ((1e-16 - 0) / 0.1e-16)**2 = 1
    # chi2_g1 = ((4e-16 - 0) / 0.1e-16)**2 = 16
    # Expected nlp_diff ~= (16 - 1) / 2 = 7.5
    expected_chi2_diff = 16 - 1
    expected_nlp_diff = expected_chi2_diff * 0.5

    actual_diff = float(nlp_g1 - nlp_g0)
    tolerance = abs(actual_diff - expected_nlp_diff) / abs(expected_nlp_diff)

    assert tolerance < 0.2, (
        f"Likelihood diff {actual_diff:.4f} far from expected {expected_nlp_diff:.4f}"
    )
    assert nlp_g0 != nlp_g1, (
        f"Likelihoods identical — line data not reaching"
    )


def test_swapped_halpha_flips_outcomes(synthetic_ssp_wide,
                                        synthetic_tophat_obs):
    """Test 3: Swapping Halpha values swaps likelihood differences.

    Build catalog with halpha SWAPPED and verify Test 1 and Test 2 outcomes
    exchange. Permanent check for transposition bugs.
    """
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog
    from tengri.inference.fitter import Fitter

    # Original: (1x, 4x)
    cat_orig, truth_orig = build_two_galaxy_catalog(
        halpha=(1.0e-16, 4.0e-16),
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )
    ca_orig = cat_orig._catalog_arrays

    # Swapped: (4x, 1x)
    cat_swap, truth_swap = build_two_galaxy_catalog(
        halpha=(4.0e-16, 1.0e-16),
        ssp=synthetic_ssp_wide,
        obs_base=synthetic_tophat_obs,
    )
    ca_swap = cat_swap._catalog_arrays

    # Likelihood differences (g0 - g1)
    fitter_orig_g0 = Fitter(cat_orig.fwd, ca_orig.flux[0], ca_orig.noise[0],
                             data_type="photometry")
    fitter_orig_g1 = Fitter(cat_orig.fwd, ca_orig.flux[1], ca_orig.noise[1],
                             data_type="photometry")

    nlp_fn_orig_g0 = fitter_orig_g0._get_or_build_logdensity_fn()
    nlp_fn_orig_g1 = fitter_orig_g1._get_or_build_logdensity_fn()

    data_args_orig_g0 = dict(fitter_orig_g0._data_args)
    data_args_orig_g0["line_flux_obs"] = ca_orig.line_flux_obs[0]
    data_args_orig_g0["line_flux_err"] = ca_orig.line_flux_err[0]

    data_args_orig_g1 = dict(fitter_orig_g1._data_args)
    data_args_orig_g1["line_flux_obs"] = ca_orig.line_flux_obs[1]
    data_args_orig_g1["line_flux_err"] = ca_orig.line_flux_err[1]

    ratio_orig = float(nlp_fn_orig_g0(truth_orig, data_args_orig_g0) -
                      nlp_fn_orig_g1(truth_orig, data_args_orig_g1))

    # Same computation with swapped
    fitter_swap_g0 = Fitter(cat_swap.fwd, ca_swap.flux[0], ca_swap.noise[0],
                             data_type="photometry")
    fitter_swap_g1 = Fitter(cat_swap.fwd, ca_swap.flux[1], ca_swap.noise[1],
                             data_type="photometry")

    nlp_fn_swap_g0 = fitter_swap_g0._get_or_build_logdensity_fn()
    nlp_fn_swap_g1 = fitter_swap_g1._get_or_build_logdensity_fn()

    data_args_swap_g0 = dict(fitter_swap_g0._data_args)
    data_args_swap_g0["line_flux_obs"] = ca_swap.line_flux_obs[0]
    data_args_swap_g0["line_flux_err"] = ca_swap.line_flux_err[0]

    data_args_swap_g1 = dict(fitter_swap_g1._data_args)
    data_args_swap_g1["line_flux_obs"] = ca_swap.line_flux_obs[1]
    data_args_swap_g1["line_flux_err"] = ca_swap.line_flux_err[1]

    ratio_swap = float(nlp_fn_swap_g0(truth_swap, data_args_swap_g0) -
                      nlp_fn_swap_g1(truth_swap, data_args_swap_g1))

    # Ratios should flip sign (opposite sides of 0)
    assert ratio_orig * ratio_swap < 0, (
        f"Ratios did not flip: orig={ratio_orig:.4f}, "
        f"swap={ratio_swap:.4f}"
    )


def test_line_column_count_must_match_the_observation(synthetic_ssp_wide,
                                                       synthetic_tophat_obs):
    """Validate that line column count matches observation."""
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    with pytest.raises(ValueError, match="line"):
        build_two_galaxy_catalog(
            halpha=(1e-16, 4e-16),
            n_line_cols=3,
            ssp=synthetic_ssp_wide,
            obs_base=synthetic_tophat_obs,
        )
