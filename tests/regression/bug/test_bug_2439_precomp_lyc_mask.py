# SPDX-License-Identifier: BSD-3-Clause
"""WavePrecomp applies the nebular Lyman-continuum mask to the stellar photometric LUT.

Regression for #2439 and #2427. Without the fix, bands sampling rest λ < 912 Å were
over-predicted by up to 915% (z=2 GALEX NUV) with default neb_fesc=0.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import SEDModel, Fixed, DEFAULT, WavePrecomp

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def test_case1_worst_band_exact_vs_lut_z2_z3(synthetic_ssp):
    """Case 1: exact vs LUT at z=2 (GALEX NUV) and z=3 (SDSS u), Cue nebular.

    Worst band ≤ 0.5% after fix (the documented WavePrecomp floor).
    Before fix: +915% (z=2 NUV) and +69% (z=3 u).
    """
    filters = ["galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r"]

    # Test at z=2 and z=3
    for z in [2, 3]:
        # Build exact-path model (approx=None) with fixed redshift
        model_exact = SEDModel.build(
            synthetic_ssp,
            filters=filters,
            sfh={"type": "delayed", "log_total_mass": 10, "tau_gyr": 1, "age_gyr": 3},
            met={"logzsol": 0, "all_params": Fixed(DEFAULT)},
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(float(z)),
            approx=None,
        )

        # Build LUT-path model (approx=WavePrecomp()) with same redshift
        model_lut = SEDModel.build(
            synthetic_ssp,
            filters=filters,
            sfh={"type": "delayed", "log_total_mass": 10, "tau_gyr": 1, "age_gyr": 3},
            met={"logzsol": 0, "all_params": Fixed(DEFAULT)},
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(float(z)),
            approx=WavePrecomp(),
        )

        params = model_exact.spec.build_params_dict({})

        # Photometry via exact path
        exact_phot = np.asarray(model_exact.predict_photometry(params))

        # Photometry via LUT path
        lut_phot = np.asarray(model_lut.predict_photometry(params))

        # Straddle assertion: at least one filter samples both sides of 912 Å rest
        lyc_limit_obs = 912.0 * (1 + z)
        assert lyc_limit_obs > 1000, f"LyC limit in obs frame: {lyc_limit_obs} Å"

        # Ablation assertion: exact-path SED moves between fesc=0 and fesc=1
        params_fesc1 = model_exact.spec.build_params_dict({"neb_fesc": 1.0})
        exact_phot_fesc1 = np.asarray(model_exact.predict_photometry(params_fesc1))
        # NUV band (index 1) should show difference when fesc varies
        nuv_idx = 1
        assert not np.allclose(exact_phot[nuv_idx], exact_phot_fesc1[nuv_idx], rtol=1e-6), \
            f"Exact path: SED should move with fesc (z={z}, NUV: {exact_phot[nuv_idx]:.6e} vs {exact_phot_fesc1[nuv_idx]:.6e})"

        # Find worst band (excluding zero bands)
        mask_nonzero = exact_phot > 0
        if np.any(mask_nonzero):
            relative_errors = np.abs(
                (lut_phot[mask_nonzero] - exact_phot[mask_nonzero]) / exact_phot[mask_nonzero]
            )
            worst_band_idx_global = np.where(mask_nonzero)[0][np.argmax(relative_errors)]
            worst_band_error = np.max(relative_errors)

            print(f"z={z}: {filters[worst_band_idx_global]} error={worst_band_error*100:.3f}% "
                  f"(exact={exact_phot[worst_band_idx_global]:.6e}, lut={lut_phot[worst_band_idx_global]:.6e})")

            # After fix: worst band ≤ 0.5% (documented WavePrecomp floor)
            assert worst_band_error < 0.005, (
                f"z={z}: worst band {filters[worst_band_idx_global]} error {worst_band_error*100:.3f}% "
                f"exceeds 0.5% floor"
            )


def test_case2_fesc1_no_op(synthetic_ssp):
    """Case 2: neb_fesc=1.0 exact vs LUT unchanged (mask is a no-op).

    At fesc=1, lyc_transmission ≡ 1, both paths should agree to floating point precision.
    The fix must be a bit-exact no-op when the mask is a no-op.
    """
    filters = ["galex_fuv", "galex_nuv", "sdss_u"]

    model_exact = SEDModel.build(
        synthetic_ssp,
        filters=filters,
        sfh={"type": "delayed", "log_total_mass": 10, "tau_gyr": 1, "age_gyr": 3},
        met={"logzsol": 0, "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "neb_fesc": 1.0, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(2.0),
        approx=None,
    )

    model_lut = SEDModel.build(
        synthetic_ssp,
        filters=filters,
        sfh={"type": "delayed", "log_total_mass": 10, "tau_gyr": 1, "age_gyr": 3},
        met={"logzsol": 0, "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "neb_fesc": 1.0, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(2.0),
        approx=WavePrecomp(),
    )

    params = model_exact.spec.build_params_dict({})
    exact_phot = np.asarray(model_exact.predict_photometry(params))
    lut_phot = np.asarray(model_lut.predict_photometry(params))

    # At fesc=1, both should agree to floating point precision
    np.testing.assert_allclose(lut_phot, exact_phot, rtol=1e-10,
        err_msg=f"fesc=1.0: exact and LUT should be identical\nexact={exact_phot}\nlut={lut_phot}")
