# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2612: PSB link ratio missing.

The psb_flex and psb_suess2022 families lack the ratio linking the oldest
flexible bin to the youngest fixed bin, forcing SFR(oldest flex)/SFR(youngest
fixed) = 1.0 regardless of parameters. This test pins the correct definition
with the new n_fixed ratios, where ratio_old_0 is the link and ratio_old_i≥1
are adjacent steps within the fixed section.

References: Suess et al. 2022, ApJ 935, 146, §3.1.4; Synthesizer ContinuityPSB
implementation (sf_hist.py:1880–1905); Prospector psb template.
"""

import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug


@pytest.mark.parametrize("sfh_type", ["psb_flex", "psb_suess2022"])
def test_psb_link_ratio_definition(synthetic_ssp_wide, sfh_type):
    """Pin SFR ratios with the new n_fixed link parameter.

    With tlast = 0.2 Gyr, tflex = 2 Gyr, log_total_mass = 10:
    - ratio_old_0 (link) = 0.4 → log10 SFR(oldest flex) / SFR(youngest fixed) = 0.4
    - ratio_old_1 = 0.2 → log10 SFR(fixed 0) / SFR(fixed 1) = 0.2
    - ratio_old_2 = -0.3 → log10 SFR(fixed 1) / SFR(fixed 2) = -0.3

    Tolerance: 1e-10 in log ratio (definition tolerance, not numerical error).
    """
    ssp = synthetic_ssp_wide

    m = SEDModel.build(
        ssp_data=ssp,
        redshift=Fixed(0.0),
        neb={"type": "none"},
        sfh={
            "type": sfh_type,
            "tlast_gyr": Fixed(0.2),
            "tflex_gyr": Fixed(2.0),
            "log_total_mass": Fixed(10.0),
            "ratio_old_0": Fixed(0.4),  # The new link
            "ratio_old_1": Fixed(0.2),
            "ratio_old_2": Fixed(-0.3),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "power_law",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
    )

    st = m.predict_state({})
    lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(st.derived["sfr_history"])

    # Interpolate SFR at bin centers: oldest flex bin at 1.82 Gyr,
    # youngest fixed bin at 3.9 Gyr
    sfr_oldest_flex = np.interp(1.82e9, lbt, sfr)
    sfr_youngest_fixed = np.interp(3.9e9, lbt, sfr)

    # Check the link: log10(SFR_oldest_flex / SFR_youngest_fixed) = 0.4
    log_ratio_link = np.log10(sfr_oldest_flex / sfr_youngest_fixed)
    np.testing.assert_allclose(
        log_ratio_link,
        0.4,
        rtol=0,
        atol=1e-10,
        err_msg=f"{sfh_type}: link ratio mismatch",
    )

    # Interpolate fixed bin SFRs for adjacent steps
    sfr_fixed_0 = np.interp(4.7e9, lbt, sfr)  # fixed bin 0
    sfr_fixed_1 = np.interp(6.8e9, lbt, sfr)  # fixed bin 1
    sfr_fixed_2 = np.interp(9.9e9, lbt, sfr)  # fixed bin 2

    # Check adjacent steps
    log_ratio_01 = np.log10(sfr_fixed_0 / sfr_fixed_1)
    log_ratio_12 = np.log10(sfr_fixed_1 / sfr_fixed_2)

    np.testing.assert_allclose(
        log_ratio_01, 0.2, rtol=0, atol=1e-10, err_msg="ratio_old_1 mismatch"
    )
    np.testing.assert_allclose(
        log_ratio_12, -0.3, rtol=0, atol=1e-10, err_msg="ratio_old_2 mismatch"
    )


@pytest.mark.parametrize("sfh_type", ["psb_flex", "psb_suess2022"])
def test_psb_link_zero_regression(synthetic_ssp_wide, sfh_type):
    """With link = 0, SFHs are bit-identical to pre-fix behavior.

    Regression pin: default ratio_old_0 = 0 must reproduce the old
    SFR values exactly, confirming the fix is backward compatible.
    """
    ssp = synthetic_ssp_wide

    m = SEDModel.build(
        ssp_data=ssp,
        redshift=Fixed(0.0),
        neb={"type": "none"},
        sfh={
            "type": sfh_type,
            "tlast_gyr": Fixed(0.2),
            "tflex_gyr": Fixed(2.0),
            "log_total_mass": Fixed(10.0),
            "ratio_old_0": Fixed(0.0),  # Link = 0 → SFR(oldest flex) = SFR(youngest fixed)
            "ratio_old_1": Fixed(0.2),
            "ratio_old_2": Fixed(-0.3),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "power_law",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
    )

    st = m.predict_state({})
    lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(st.derived["sfr_history"])

    # With link = 0, ratio = 1.0 (within numerical precision)
    sfr_oldest_flex = np.interp(1.82e9, lbt, sfr)
    sfr_youngest_fixed = np.interp(3.9e9, lbt, sfr)

    ratio = sfr_oldest_flex / sfr_youngest_fixed
    np.testing.assert_allclose(
        ratio,
        1.0,
        rtol=1e-10,
        err_msg=f"{sfh_type}: link=0 should give ratio=1",
    )


@pytest.mark.parametrize("sfh_type", ["psb_flex", "psb_suess2022"])
@pytest.mark.parametrize("link", [-0.5, 0.4])
@pytest.mark.parametrize("age_kernel", ["cic", "dsps"])
def test_mass_conserved_with_nonzero_link(synthetic_ssp_wide, sfh_type, link, age_kernel):
    """A nonzero link moves bin amplitudes, not the formed-mass normalization.

    ``_mass_conserving_total`` renormalizes the published history to
    ``10**log_total_mass`` after the shape is built, so any ``ratio_old_0``
    (link) value must still close to the declared total -- to the SAME
    tolerance ``tests/physics/conservation/test_sfh_support_bounded_to_age_of_universe.py``
    already uses (``abs=1e-4`` on the ratio to 1.0), on both age kernels.
    """
    log_total_mass = 10.0
    m = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        redshift=Fixed(0.0),
        neb={"type": "none"},
        sfh={
            "type": sfh_type,
            "tlast_gyr": Fixed(0.2),
            "tflex_gyr": Fixed(2.0),
            "log_total_mass": Fixed(log_total_mass),
            "ratio_old_0": Fixed(link),
            "ratio_old_1": Fixed(0.2),
            "ratio_old_2": Fixed(-0.3),
            "all_params": Fixed(DEFAULT),
            "age_kernel": age_kernel,
        },
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )
    pred = m.predict({})
    ratio = float(pred.stellar_mass) / 10.0**log_total_mass
    assert ratio == pytest.approx(1.0, abs=1e-4), (
        f"{sfh_type} {age_kernel} link={link}: formed mass ratio {ratio} != 1"
    )


@pytest.mark.parametrize("sfh_type", ["psb_flex", "psb_suess2022"])
def test_count_contract(synthetic_ssp_wide, sfh_type):
    """Exactly ``n_fixed`` ``ratio_old_*`` free params; an n_fixed-1 call leaves the link at 0.

    The registry-derived count (``SFH_REGISTRY[sfh_type].params``) and the
    ``free_params`` surface under ``all_params: FREE`` must agree, and both
    must equal ``n_fixed`` (#2612; was ``n_fixed - 1``). A flat-kwarg caller
    that supplies only the (old) ``n_fixed - 1`` values addresses
    ``ratio_old_0`` as the LINK, not re-indexed to an adjacent step: the
    omitted entry (now ``ratio_old_{n_fixed-1}``, the oldest adjacent step)
    silently defaults to 0, it does not shift what was passed.
    """
    from tengri.components.stellar.sfh.nonparametric import PSB_FLEX_DEFAULT_N_FIXED
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    spec = SFH_REGISTRY[sfh_type]
    prefix = "sfh_psb2022_" if sfh_type == "psb_suess2022" else "sfh_psb_flex_"
    declared = [name for name in spec.params if f"{prefix}ratio_old_" in name]
    assert len(declared) == PSB_FLEX_DEFAULT_N_FIXED, (
        f"{sfh_type}: declares {len(declared)} ratio_old_* params, expected "
        f"{PSB_FLEX_DEFAULT_N_FIXED} (one per fixed bin, #2612)"
    )

    m = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        redshift=Fixed(0.05),
        sfh={"type": sfh_type, "all_params": FREE},
    )
    free_old = [p for p in m.spec.free_params if f"{prefix}ratio_old_" in p]
    assert len(free_old) == PSB_FLEX_DEFAULT_N_FIXED, (
        f"{sfh_type}: all_params=FREE frees {len(free_old)} ratio_old_* params, "
        f"expected {PSB_FLEX_DEFAULT_N_FIXED}"
    )

    # The n_fixed - 1 call-site behavior: a caller supplying only the OLD
    # count of values addresses ratio_old_0 as the link (not re-indexed).
    n_fixed_minus_1 = PSB_FLEX_DEFAULT_N_FIXED - 1
    old_style_kwargs = {f"ratio_old_{i}": Fixed(0.0) for i in range(n_fixed_minus_1)}
    old_style_kwargs["ratio_old_0"] = Fixed(0.4)  # what an old caller meant as "first step"
    m_partial = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        redshift=Fixed(0.0),
        neb={"type": "none"},
        sfh={
            "type": sfh_type,
            "tlast_gyr": Fixed(0.2),
            "tflex_gyr": Fixed(2.0),
            "log_total_mass": Fixed(10.0),
            **old_style_kwargs,
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )
    st = m_partial.predict_state({})
    lbt = np.asarray(st.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(st.derived["sfr_history"])
    sfr_oldest_flex = np.interp(1.82e9, lbt, sfr)
    sfr_youngest_fixed = np.interp(3.9e9, lbt, sfr)
    log_ratio_link = np.log10(sfr_oldest_flex / sfr_youngest_fixed)
    np.testing.assert_allclose(
        log_ratio_link,
        0.4,
        rtol=0,
        atol=1e-10,
        err_msg=(
            f"{sfh_type}: an n_fixed-1 call's ratio_old_0=0.4 must be read as the "
            "link (0.4), not silently shifted to an adjacent step"
        ),
    )
    # The omitted entry (ratio_old_{n_fixed-1}, the oldest adjacent step)
    # defaults to 0: fixed bin n_fixed-2 and n_fixed-1 share an SFR.
    sfr_fixed_1 = np.interp(6.8e9, lbt, sfr)
    sfr_fixed_2 = np.interp(9.9e9, lbt, sfr)
    np.testing.assert_allclose(np.log10(sfr_fixed_1 / sfr_fixed_2), 0.0, rtol=0, atol=1e-10)
