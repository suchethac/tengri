# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for xray_delta_alpha_ox as delta offset to L_2500-derived alpha_ox.

Verifies that the composable X-ray component now consumes xray_delta_alpha_ox as a
live offset parameter (not a no-op), consistent with CIGALE/Just+2007 convention.
"""

import jax.numpy as jnp
import pytest

from tengri.components.xray.xray import xray_agn_corona

pytestmark = [
    pytest.mark.contract,
    pytest.mark.regression_paper,
]


@pytest.fixture
def wave_xray_rest():
    """Rest-frame X-ray wavelength grid covering 0.1-124 Å (0.1-100 keV)."""
    return jnp.logspace(-1.0, 2.0, 256)  # 0.1 to 100 Å


@pytest.fixture
def l_2500_fixed():
    """Fixed 2500 Å luminosity for reproducible offset tests [erg/s/Hz]."""
    return 1.0e29


def test_xray_delta_alpha_ox_offset_zero_default(wave_xray_rest, l_2500_fixed):
    """Default delta_alpha_ox=0.0 gives pure empirical L_2500-derived corona.

    At default (0.0 offset), the X-ray corona follows the Just+2007 empirical
    relation with no modification.
    """
    # Compute corona at L_2500_fixed with zero offset
    sed_default = xray_agn_corona(
        wave_xray_rest,
        l_2500_30deg_erg_hz=l_2500_fixed,
        gamma=1.8,
        delta_alpha_ox=0.0,  # Default: pure empirical
    )
    # Should be non-zero and finite
    assert jnp.all(sed_default > 0.0), "Default corona should be positive"
    assert jnp.all(jnp.isfinite(sed_default)), "Default corona should be finite"


def test_xray_delta_alpha_ox_offset_hardening(wave_xray_rest, l_2500_fixed):
    """Negative delta_alpha_ox hardens the X-ray corona (steeper spectrum).

    A negative offset (e.g., -0.3) shifts alpha_ox to more negative values,
    making the spectrum steeper in alpha_ox (fewer hard X-rays).
    """
    # Compute corona at two different offsets
    sed_soft = xray_agn_corona(
        wave_xray_rest,
        l_2500_30deg_erg_hz=l_2500_fixed,
        gamma=1.8,
        delta_alpha_ox=0.0,  # Reference: empirical
    )

    sed_hard = xray_agn_corona(
        wave_xray_rest,
        l_2500_30deg_erg_hz=l_2500_fixed,
        gamma=1.8,
        delta_alpha_ox=-0.3,  # Harder corona (more negative alpha_ox)
    )

    # Both should be positive and finite
    assert jnp.all(sed_soft > 0.0)
    assert jnp.all(sed_hard > 0.0)

    # Hard and soft should differ (not be close)
    relative_diff = jnp.abs(sed_hard - sed_soft) / sed_soft
    mean_rel_diff = jnp.mean(relative_diff)
    assert mean_rel_diff > 0.01, (
        f"Hardening offset should change spectrum; mean rel diff = {mean_rel_diff}"
    )

    # alpha_ox sets the X-ray/UV NORMALIZATION (the spectral slope is gamma).
    # A more negative alpha_ox (delta < 0) steepens the UV->X-ray slope, i.e.
    # LESS X-ray per unit UV: L_2keV = L_2500 * 10**(alpha_ox / 0.3838) drops,
    # so the whole corona scales DOWN. The negative-offset corona is fainter.
    ratio = jnp.sum(sed_hard) / jnp.sum(sed_soft)
    assert ratio < 1.0, f"More negative alpha_ox should reduce X-ray flux; ratio = {ratio}"


def test_xray_delta_alpha_ox_offset_softening(wave_xray_rest, l_2500_fixed):
    """Positive delta_alpha_ox softens the X-ray corona (shallower spectrum).

    A positive offset (e.g., +0.3) shifts alpha_ox to less negative values,
    making the spectrum shallower in alpha_ox (more hard X-rays per unit soft).
    """
    # Compute corona at two different offsets
    sed_empirical = xray_agn_corona(
        wave_xray_rest,
        l_2500_30deg_erg_hz=l_2500_fixed,
        gamma=1.8,
        delta_alpha_ox=0.0,  # Reference: empirical
    )

    sed_soft = xray_agn_corona(
        wave_xray_rest,
        l_2500_30deg_erg_hz=l_2500_fixed,
        gamma=1.8,
        delta_alpha_ox=+0.3,  # Softer corona (less negative alpha_ox)
    )

    # Both should be positive and finite
    assert jnp.all(sed_empirical > 0.0)
    assert jnp.all(sed_soft > 0.0)

    # Soft and empirical should differ
    relative_diff = jnp.abs(sed_soft - sed_empirical) / sed_empirical
    mean_rel_diff = jnp.mean(relative_diff)
    assert mean_rel_diff > 0.01, (
        f"Softening offset should change spectrum; mean rel diff = {mean_rel_diff}"
    )


def test_xray_delta_alpha_ox_no_longer_noop():
    """Verify that xray_delta_alpha_ox parameter is no longer a no-op on composable path.

    This is the regression test proving the fix: directly calling
    xray_agn_corona (used by the SEDComponent) with different delta_alpha_ox
    produces different outputs, confirming the parameter is live.
    """
    wave = jnp.logspace(-1.0, 2.0, 128)
    l_2500 = 1.0e29

    # Compute with two different offsets
    sed_1 = xray_agn_corona(
        wave,
        l_2500_30deg_erg_hz=l_2500,
        gamma=1.8,
        delta_alpha_ox=-0.2,
    )

    sed_2 = xray_agn_corona(
        wave,
        l_2500_30deg_erg_hz=l_2500,
        gamma=1.8,
        delta_alpha_ox=+0.2,
    )

    # The SEDs must differ meaningfully (not close)
    assert not jnp.allclose(sed_1, sed_2, rtol=0.05), (
        "xray_delta_alpha_ox should produce different spectra when varied"
    )

    # Both should be physically reasonable
    assert jnp.all(sed_1 > 0.0) and jnp.all(sed_2 > 0.0)
    assert jnp.all(jnp.isfinite(sed_1)) and jnp.all(jnp.isfinite(sed_2))


def test_xray_delta_alpha_ox_has_free_prior_over_paper_range():
    """xray_delta_alpha_ox gets a declared free_prior (task-12 public-API
    audit): AGNfitter-rX fits Delta(alpha_ox) over [-0.4, 0.4] (the scatter
    around the Just+2007 empirical relation), but the declaration previously
    carried NO free_prior, so ``xray={'all_params': FREE}`` could never reach
    it even though it is a real, live, in-scope parameter for the
    agn_xray_corona/simple/... models (already listed in
    _XRAY_PARAMS_BY_MODEL) -- explicit naming was the only way to fit it.
    """
    from tengri.components.xray._params import PARAMS

    decl = next(pd for pd in PARAMS if pd.name == "xray_delta_alpha_ox")
    assert decl.free_prior is not None, "xray_delta_alpha_ox still has no free_prior"
    assert decl.free_prior.bounds == (-0.4, 0.4)


def test_xray_wildcard_now_frees_delta_alpha_ox():
    """``xray={'type': 'agn_xray_corona', 'all_params': FREE}`` frees
    xray_delta_alpha_ox with the paper-cited [-0.4, 0.4] bounds -- previously
    it stayed Fixed(0.0) under any wildcard because of the missing
    free_prior, even though the scoping table already listed it as a
    parameter that model reads."""
    import warnings

    from tengri import DEFAULT, FREE, Fixed
    from tengri.parameters.groups import parse_groups

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = parse_groups(
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            xray={"type": "agn_xray_corona", "all_params": FREE},
            agn={
                "type": "composable",
                "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
                "agn_log_lbol": Fixed(12.0),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )
    assert "xray_delta_alpha_ox" in params.free_params
    dist = params.get_distribution("xray_delta_alpha_ox")
    assert dist.bounds == (-0.4, 0.4)
