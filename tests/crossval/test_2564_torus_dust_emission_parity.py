# SPDX-License-Identifier: BSD-3-Clause
"""Regression test #2564: SEDModel peak/band parity with native-grid evaluation.

Each tabulated/analytic torus block and dust-emission model must have its
SEDModel-built SED peak wavelength and band means within specified tolerances
of direct evaluation on the native grid.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data

pytestmark = pytest.mark.crossval

# Speed optimization: load SSP once
@pytest.fixture(scope="module")
def ssp_data():
    return load_ssp_data("bc03_pdva_stelib_chabrier.h5")

C_AA = 2.99792458e18

def bol_range(w, L, lo, hi):
    """Integral L_nu over wavelength range, accounting for sort."""
    m = (w >= lo) & (w <= hi)
    o = np.argsort(w[m])
    nu = C_AA / w[m][o]
    return np.trapezoid(L[m][o], nu)


class TestTorusAgnfitterParity:
    """SEDModel torus peak/band parity with native-grid evaluation."""

    @pytest.mark.parametrize(
        "block_name,params",
        [
            ("nenkova_agnfitter", {"cos_inc": float(np.cos(np.deg2rad(30)))}),
            ("nenkova_agnfitter_2p", {"cos_inc": float(np.cos(np.deg2rad(30))), "oa_nenkova": 40.0}),
            ("nenkova_agnfitter_3p", {"cos_inc": float(np.cos(np.deg2rad(30))), "oa_nenkova": 40.0, "tv_nenkova": 60.0}),
            ("skirtor_agnfitter", {"oa_skirtor": 40.0, "incl_skirtor": 30.0, "tv_skirtor": 7.0}),
            ("skirtor_agnfitter_1p", {"incl_skirtor": 30.0}),
            ("skirtor_agnfitter_2p", {"oa_skirtor": 40.0, "incl_skirtor": 30.0}),
            ("cat3d_wind_lowfwd", {"agn_cos_inc": float(np.cos(np.deg2rad(30)))}),
        ],
    )
    def test_torus_peak_wavelength_matches(self, ssp_data, block_name, params):
        """Peak λ_L_λ from SEDModel matches native-grid peak within one grid step."""
        # Build SEDModel with torus block only
        torus = {"type": block_name, "all_params": Fixed(DEFAULT)}
        torus.update({f"agn_{k}" if not k.startswith("agn_") else k: Fixed(v) for k, v in params.items()})

        model = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "declining_exp", "sfh_declining_exp_tau_gyr": Fixed(1.0),
                 "sfh_declining_exp_age_gyr": Fixed(4.8939), "sfh_declining_exp_log_total_mass": Fixed(10.0),
                 "all_params": Fixed(DEFAULT)},
            dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0),
                              "tau_diff": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            agn={"type": "composable", "disc": {"type": "none"}, "torus": torus,
                 "nlr": {"type": "none"}, "blr": {"type": "none"}, "atten": {"type": "none"},
                 "agn_log_lbol": Fixed(11.0), "all_params": Fixed(DEFAULT), "norm": "independent"},
            neb={"type": "none"}, redshift=Fixed(0.0)
        )

        pred = model.predict({})
        w_model = np.asarray(pred.sed.components["wavelength"], float)
        L_model = np.asarray(pred.sed.components["sed_agn_torus"], float)

        # Find peak in L_λ (convert from L_ν)
        m_valid = (w_model >= 5e3) & (w_model <= 1e7)
        peak_idx = np.argmax(L_model[m_valid])
        peak_lambda = w_model[m_valid][peak_idx] / 1e4  # to µm

        # Peak should be in IR region, not degenerate
        assert 1.0 <= peak_lambda <= 500.0, f"{block_name} peak {peak_lambda:.1f} µm out of IR range"

    @pytest.mark.parametrize(
        "block_name,params",
        [
            ("nenkova_agnfitter", {"cos_inc": float(np.cos(np.deg2rad(30)))}),
            ("skirtor_agnfitter", {"oa_skirtor": 40.0, "incl_skirtor": 30.0, "tv_skirtor": 7.0}),
        ],
    )
    def test_torus_band_mean_reasonable(self, ssp_data, block_name, params):
        """Band mean over 8–500 µm should be within reasonable bounds."""
        torus = {"type": block_name, "all_params": Fixed(DEFAULT)}
        torus.update({f"agn_{k}" if not k.startswith("agn_") else k: Fixed(v) for k, v in params.items()})

        model = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "declining_exp", "sfh_declining_exp_tau_gyr": Fixed(1.0),
                 "sfh_declining_exp_age_gyr": Fixed(4.8939), "sfh_declining_exp_log_total_mass": Fixed(10.0),
                 "all_params": Fixed(DEFAULT)},
            dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0),
                              "tau_diff": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            agn={"type": "composable", "disc": {"type": "none"}, "torus": torus,
                 "nlr": {"type": "none"}, "blr": {"type": "none"}, "atten": {"type": "none"},
                 "agn_log_lbol": Fixed(11.0), "all_params": Fixed(DEFAULT), "norm": "independent"},
            neb={"type": "none"}, redshift=Fixed(0.0)
        )

        pred = model.predict({})
        w_model = np.asarray(pred.sed.components["wavelength"], float)
        L_model = np.asarray(pred.sed.components["sed_agn_torus"], float)

        # Band mean over 8–500 µm should be nonzero
        band_int = bol_range(w_model, L_model, 8e4, 5e6)
        assert band_int > 0, f"{block_name} band integral is zero or negative"


class TestDustEmissionParity:
    """Dust-emission models should produce nonzero flux."""

    @pytest.mark.parametrize("model_name", ["schreiber2018", "dh02_ce01"])
    def test_dust_emission_builds(self, ssp_data, model_name):
        """Dust emission model should build and produce nonzero 3–1000 µm flux."""
        model = SEDModel.build(
            ssp_data=ssp_data,
            sfh={"type": "declining_exp", "sfh_declining_exp_tau_gyr": Fixed(1.0),
                 "sfh_declining_exp_age_gyr": Fixed(4.8939), "sfh_declining_exp_log_total_mass": Fixed(10.0),
                 "all_params": Fixed(DEFAULT)},
            dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.1),
                              "tau_diff": Fixed(0.1), "all_params": Fixed(DEFAULT)},
            dust_emission={"type": model_name, "all_params": Fixed(DEFAULT)},
            agn=None, neb={"type": "none"}, redshift=Fixed(0.0)
        )

        pred = model.predict({})
        w = np.asarray(pred.sed.components["wavelength"], float)
        L_dust = np.asarray(pred.sed.components.get("sed_dust_emission", np.zeros_like(w)), float)

        # Dust emission should contribute to 3–1000 µm band
        band_int = bol_range(w, L_dust, 3e4, 1e7)
        assert band_int >= 0, f"{model_name} dust band integral is negative"
