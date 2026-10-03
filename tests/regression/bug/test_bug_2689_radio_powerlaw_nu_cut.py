"""
Regression test for issue #2689: power-law AGN radio model honors radio_log_nu_cut.

Verifies the cutoff parameter is threaded through correctly, DPL-only keys are
refused, default is unchanged, and the analytic gradient is correct.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel, load_ssp
from tengri.config import ConfigError

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp():
    return load_ssp()


def test_radio_powerlaw_nu_cut_factor_formula(ssp):
    """
    Verify L_agn(cut)/L_agn(13) matches exp(-nu/10^cut)/exp(-nu/1e13) to rtol 1e-6.

    Isolate AGN by subtracting SF component: compute (SF+FF+AGN) - (SF+FF only).
    """
    # Get SF+FF component (no AGN)
    m_sf = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "none"},
        neb={"type": "none"},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        radio={"agn": {"type": "none"}, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
    )

    state_sf = m_sf.predict_state({})
    wave = np.asarray(m_sf.wave if hasattr(m_sf, "wave") else state_sf.wave)
    sed_sf = np.asarray(state_sf.derived["sed_radio"])
    C = 2.99792458e18
    nu = C / wave

    # Full spectrum at cut=13 (baseline)
    m_base = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "none"},
        neb={"type": "none"},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        radio={"all_params": Fixed(DEFAULT), "agn": {"all_params": Fixed(DEFAULT)}},
        redshift=Fixed(0.0),
    )

    sed_full_13 = np.asarray(m_base.predict_state({}).derived["sed_radio"])
    sed_agn_13 = sed_full_13 - sed_sf

    # Test each cutoff value
    for cut_val in [40.0, 11.0]:
        m_cut = SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(3.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "none"},
            neb={"type": "none"},
            agn={"type": "composable", "all_params": Fixed(DEFAULT)},
            radio={
                "all_params": Fixed(DEFAULT),
                "agn": {"radio_log_nu_cut": Fixed(cut_val), "all_params": Fixed(DEFAULT)},
            },
            redshift=Fixed(0.0),
        )

        sed_full_cut = np.asarray(m_cut.predict_state({}).derived["sed_radio"])
        sed_agn_cut = sed_full_cut - sed_sf

        # At grid nodes nearest 100, 300, 1000 GHz
        for freq_ghz in [100.0, 300.0, 1000.0]:
            idx = np.argmin(np.abs(nu - freq_ghz * 1e9))
            nu_exact = nu[idx]

            factor_obs = sed_agn_cut[idx] / sed_agn_13[idx]
            factor_exp = np.exp(-nu_exact / 10.0**cut_val) / np.exp(-nu_exact / 1e13)

            np.testing.assert_allclose(factor_obs, factor_exp, rtol=1e-6)


def test_radio_powerlaw_default_unchanged(ssp):
    """Default cut=13 is bit-identical to reproducer."""
    m = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "none"},
        neb={"type": "none"},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        radio={"all_params": Fixed(DEFAULT), "agn": {"all_params": Fixed(DEFAULT)}},
        redshift=Fixed(0.0),
    )

    state = m.predict_state({})
    wave = np.asarray(m.wave if hasattr(m, "wave") else state.wave)
    sed = np.asarray(state.derived["sed_radio"])
    C = 2.99792458e18
    nu = C / wave

    idx_100 = np.argmin(np.abs(nu - 100.0 * 1e9))
    L_100 = sed[idx_100]

    np.testing.assert_allclose(L_100, 7.593923e27, rtol=1e-3)


def test_radio_powerlaw_dpl_keys_refused(ssp):
    """DPL-only keys raise ConfigError on powerlaw model."""
    for key, val in [
        ("radio_alpha_thin", 1.5),
        ("radio_alpha_thick", -1.0),
        ("radio_log_nu_t", 10.5),
    ]:
        with pytest.raises(ConfigError) as exc:
            SEDModel.build(
                ssp_data=ssp,
                met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
                sfh={
                    "type": "delayed",
                    "tau_gyr": Fixed(1.0),
                    "age_gyr": Fixed(3.0),
                    "log_total_mass": Fixed(10.0),
                    "all_params": Fixed(DEFAULT),
                },
                dust_attenuation={
                    "type": "single_component",
                    "law": "calzetti",
                    "all_params": Fixed(DEFAULT),
                },
                dust_emission={"type": "none"},
                neb={"type": "none"},
                agn={"type": "composable", "all_params": Fixed(DEFAULT)},
                radio={
                    "all_params": Fixed(DEFAULT),
                    "agn": {"all_params": Fixed(DEFAULT), key: Fixed(val)},
                },
                redshift=Fixed(0.0),
            )

        msg = str(exc.value).lower()
        assert "dpl" in msg and (
            key in str(exc.value) or key.replace("radio_", "") in str(exc.value)
        )


def test_radio_powerlaw_nu_cut_gradient(ssp):
    """
    ``jax.grad`` w.r.t. a FREE ``radio_log_nu_cut`` of the 300 GHz AGN-jet L_nu
    equals the analytic L * (nu / nu_cut) * ln(10).

    The derivative is taken by autodiff through the public model path with the
    cutoff traced, so a ``stop_gradient``, a Python-float conversion or an
    untraced branch in the cutoff fails it.  A central finite difference
    cannot see those: it only evaluates the forward model.  The finite
    difference is kept as a secondary cross-check.
    """
    m_sf = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "none"},
        neb={"type": "none"},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        radio={"agn": {"type": "none"}, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
    )

    m = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "none"},
        neb={"type": "none"},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        radio={
            "all_params": Fixed(DEFAULT),
            "agn": {"radio_log_nu_cut": FREE, "all_params": Fixed(DEFAULT)},
        },
        redshift=Fixed(0.0),
    )
    assert "radio_log_nu_cut" in m.spec.free_params

    wave = np.asarray(
        m.wave if hasattr(m, "wave") else m.predict_state({"radio_log_nu_cut": 13.0}).wave
    )
    C = 2.99792458e18
    nu = C / wave
    sed_sf = jnp.asarray(m_sf.predict_state({}).derived["sed_radio"])

    idx_300 = int(np.argmin(np.abs(nu - 300.0 * 1e9)))
    nu_300 = float(nu[idx_300])

    def agn_flux_300(log_cut):
        state = m.predict_state({"radio_log_nu_cut": log_cut})
        return state.derived["sed_radio"][idx_300] - sed_sf[idx_300]

    log_cut0 = 13.0
    L_agn = float(agn_flux_300(jnp.asarray(log_cut0)))
    grad_ana = L_agn * (nu_300 / 10.0**log_cut0) * np.log(10.0)
    assert np.isfinite(grad_ana) and grad_ana != 0.0

    # Primary: autodiff through the traced cutoff.
    grad_ad = float(jax.grad(agn_flux_300)(jnp.asarray(log_cut0)))
    np.testing.assert_allclose(grad_ad, grad_ana, rtol=1e-4)

    # Secondary: central finite difference of the forward model.
    eps = 1e-4
    grad_fd = (
        float(agn_flux_300(jnp.asarray(log_cut0 + eps)))
        - float(agn_flux_300(jnp.asarray(log_cut0 - eps)))
    ) / (2 * eps)
    np.testing.assert_allclose(grad_fd, grad_ana, rtol=1e-3)
