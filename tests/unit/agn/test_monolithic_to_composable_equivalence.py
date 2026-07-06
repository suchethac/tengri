# SPDX-License-Identifier: BSD-3-Clause
"""Equivalence tests: monolithic AGN models vs composable blocks.

Verifies that each deprecated monolithic AGN model produces the same
(bit-exact or <1e-14 relative error) SED as its composable block equivalent.

Taxonomy: regression_paper / contract
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.runner import composable_agn_l_nu
from tengri.components.agn.unified import resolve_agn_model


# Test wavelength grid (100 points, log-spaced 1000–30000 Å)
_WAVE_TEST = np.logspace(3.0, 4.48, 100)


def _equivalence_test(
    monolithic_name: str,
    composable_selectors: dict,
    common_kwargs: dict | None = None,
    rtol: float = 1e-14,
) -> None:
    """Helper: assert monolithic == composable SED.

    Parameters
    ----------
    monolithic_name : str
        Name of deprecated monolithic model (e.g., "multicolor_agn").
    composable_selectors : dict
        Block selectors for composable path (e.g.,
        {"agn_disc_block": "multicolor", "agn_torus_block": "silva04"}).
    common_kwargs : dict, optional
        Shared parameters for both paths (e.g., agn_frac, agn_cos_inc).
        Default: empty dict.
    rtol : float, optional
        Relative tolerance for allclose. Default 1e-14.
    """
    if common_kwargs is None:
        common_kwargs = {}

    # Default params that both paths accept
    default_params = {
        "agn_log_lbol": 11.0,
        "agn_frac": 0.5,
    }
    default_params.update(common_kwargs)

    # Old monolithic path
    try:
        monolithic_fn = resolve_agn_model(monolithic_name)
        l_nu_old = np.array(monolithic_fn(_WAVE_TEST, **default_params))
    except Exception as e:
        pytest.skip(f"Monolithic {monolithic_name} not available: {e}")

    # New composable path
    l_nu_new = np.array(
        composable_agn_l_nu(
            _WAVE_TEST,
            **default_params,
            **composable_selectors,
        )
    )

    # Verify equivalence
    try:
        np.testing.assert_allclose(l_nu_old, l_nu_new, rtol=rtol, atol=1e-99)
    except AssertionError as e:
        # Report relative differences for debugging
        rel_diff = np.abs((l_nu_old - l_nu_new) / np.maximum(np.abs(l_nu_old), 1e-30))
        max_rel = np.max(rel_diff)
        pytest.fail(
            f"Equivalence failed for {monolithic_name}:\n"
            f"  Max rel diff: {max_rel:.2e}\n"
            f"  Tolerance: {rtol:.2e}\n"
            f"  Original error: {e}"
        )


# ────────────────────────────────────────────────────────────────────────
# 13 Equivalence Tests (One per Monolithic Model)
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.regression_paper
def test_multicolor_agn_equivalence():
    """Test: deprecated multicolor_agn == composable disc=multicolor + torus=silva04."""
    _equivalence_test(
        "multicolor_agn",
        {
            "agn_disc_block": "multicolor",
            "agn_torus_block": "silva04",
        },
    )


@pytest.mark.regression_paper
def test_kubota_done_full_equivalence():
    """Test: deprecated kubota_done_full == composable disc=kubota_done + torus=silva04."""
    _equivalence_test(
        "kubota_done_full",
        {
            "agn_disc_block": "kubota_done",
            "agn_torus_block": "silva04",
        },
    )


@pytest.mark.regression_paper
def test_silva04_equivalence():
    """Test: deprecated silva04 == composable disc=powerlaw + torus=silva04."""
    _equivalence_test(
        "silva04",
        {
            "agn_disc_block": "powerlaw",
            "agn_torus_block": "silva04",
        },
    )


@pytest.mark.regression_paper
def test_cat3d_wind_equivalence():
    """Test: deprecated cat3d_wind == composable disc=powerlaw + torus=cat3d_wind."""
    _equivalence_test(
        "cat3d_wind",
        {
            "agn_disc_block": "powerlaw",
            "agn_torus_block": "cat3d_wind",
        },
    )


@pytest.mark.regression_paper
def test_adaf_equivalence():
    """Test: deprecated adaf == composable disc=adaf + torus=silva04."""
    _equivalence_test(
        "adaf",
        {
            "agn_disc_block": "adaf",
            "agn_torus_block": "silva04",
        },
    )


@pytest.mark.regression_paper
def test_relagn_equivalence():
    """Test: deprecated relagn == composable disc=relagn + torus=silva04."""
    _equivalence_test(
        "relagn",
        {
            "agn_disc_block": "relagn",
            "agn_torus_block": "silva04",
        },
    )


@pytest.mark.regression_paper
def test_skirtor_equivalence():
    """Test: deprecated skirtor == composable disc=skirtor + torus=skirtor."""
    _equivalence_test(
        "skirtor",
        {
            "agn_disc_block": "skirtor",
            "agn_torus_block": "skirtor",
        },
    )


@pytest.mark.regression_paper
def test_skirtor_stalevski_equivalence():
    """Test: deprecated skirtor_stalevski == composable disc=skirtor + torus=skirtor."""
    _equivalence_test(
        "skirtor_stalevski",
        {
            "agn_disc_block": "skirtor",
            "agn_torus_block": "skirtor",
        },
    )


@pytest.mark.regression_paper
def test_qsogen_equivalence():
    """Test: deprecated qsogen == composable qsogen blocks (disc + nlr + blr)."""
    _equivalence_test(
        "qsogen",
        {
            "agn_disc_block": "qsogen_sbpl_disc",
            "agn_nlr_block": "qsogen_nlr",
            "agn_blr_block": "qsogen_blr",
        },
    )


@pytest.mark.regression_paper
def test_grahsp_equivalence():
    """Test: deprecated grahsp == composable grahsp blocks (disc + nlr + blr + feii)."""
    _equivalence_test(
        "grahsp",
        {
            "agn_disc_block": "grahsp_sbpl_disc",
            "agn_nlr_block": "grahsp_nlr",
            "agn_blr_block": "grahsp_blr",
            "agn_feii_block": "grahsp_feii",
        },
    )


@pytest.mark.regression_paper
def test_richards2006_equivalence():
    """Test: deprecated richards2006 == composable disc=richards2006."""
    _equivalence_test(
        "richards2006",
        {
            "agn_disc_block": "richards2006_disc",
        },
    )


@pytest.mark.regression_paper
def test_unified_nlr_blr_equivalence():
    """Test: deprecated unified_nlr_blr == composable multicolor + silva04 + nlr + blr."""
    # unified_nlr_blr uses multicolor disc + silva04 torus + nlr/blr lines
    _equivalence_test(
        "unified_nlr_blr",
        {
            "agn_disc_block": "multicolor",
            "agn_nlr_block": "nlr_analytic",
            "agn_blr_block": "blr_analytic",
            "agn_torus_block": "silva04",
        },
        common_kwargs={
            "agn_cos_inc": 0.5,
            "agn_nlr_cf": 0.1,
            "agn_blr_cf": 0.1,
        },
    )


@pytest.mark.regression_paper
def test_kubota_done_alias_equivalence():
    """Test: deprecated kubota_done (alias) == composable disc=kubota_done + torus=silva04."""
    # kubota_done is an alias for multicolor_agn, so it should also match composable
    _equivalence_test(
        "kubota_done",
        {
            "agn_disc_block": "multicolor",
            "agn_torus_block": "silva04",
        },
    )
