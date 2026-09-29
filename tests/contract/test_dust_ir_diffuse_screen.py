# SPDX-License-Identifier: BSD-3-Clause
"""Contract: opt-in single-pass diffuse-screen attenuation of re-emitted IR (#2533).

The dust IR emission, after it is built and normalized to its absorbed budget
L_ir, is multiplied ONCE by the diffuse dust screen's transmission T(λ). The
IR energy absorbed on the way out is REMOVED: no re-emission, no iteration, no
renormalization back to L_ir. Default OFF: bit-identical to today.

These tests pin that the toggle (a) reaches the forward pass with correct
application to sed_dust_ir, (b) publishes log_L_ir_emergent correctly,
(c) round-trips through the build grammar, and (d) leaves attenuation and
input energy balance unchanged while changing the emergent IR spectrum.

CI-runnable on the synthetic wide SSP (no ``data/`` grids needed).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.dust._physics import integrate_lnu_over_nu
from tengri.observation.photometry import FilterCurve
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.contract


def _build(ssp, diffuse_screen: bool = False, *, dust_type: str = "modified_blackbody",
           law_bc: str = "calzetti", law_diff: str = "calzetti",
           tau_bc: float = 0.0, tau_diff: float = 3.0):
    dust_atten = {
        "type": "two_component",
        "law_bc": law_bc,
        "law_diff": law_diff,
        "tau_bc": Fixed(tau_bc),
        "tau_diff": Fixed(tau_diff),
        "all_params": Fixed(DEFAULT),
    }
    dust_emis = {
        "type": dust_type,
        "all_params": Fixed(DEFAULT),
    }
    if diffuse_screen:
        dust_emis["diffuse_screen"] = True
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=dust_atten,
        dust_emission=dust_emis,
        redshift=Fixed(0.0),
    )


class TestDustIrDiffuseScreen:
    def test_switch_off_bit_identical(self, synthetic_ssp_wide):
        """Off by default: switch-off build matches a build without the key."""
        m_off = _build(synthetic_ssp_wide, diffuse_screen=False)
        m_default = _build(synthetic_ssp_wide)  # diffuse_screen not specified

        s_off = m_off.predict_state({})
        s_default = m_default.predict_state({})

        # Every derived output must be bit-identical
        for key in ["L_ir", "L_absorbed", "log_L_ir", "log_L_absorbed",
                    "dust_attenuation_factor", "sed_dust_ir", "sed_dust_attenuated"]:
            off_val = jnp.asarray(s_off.derived[key])
            default_val = jnp.asarray(s_default.derived[key])
            np.testing.assert_array_equal(
                off_val, default_val,
                err_msg=f"{key} differs between off and default"
            )

    def test_calzetti_ir_unchanged(self, synthetic_ssp_wide):
        """Calzetti single_component has k=0 in IR (λ > 20 μm), so transmission=1."""
        m_off = _build(synthetic_ssp_wide, diffuse_screen=False,
                      dust_type="modified_blackbody",
                      law_bc="calzetti", law_diff="calzetti")
        m_on = _build(synthetic_ssp_wide, diffuse_screen=True,
                     dust_type="modified_blackbody",
                     law_bc="calzetti", law_diff="calzetti")

        s_off = m_off.predict_state({})
        s_on = m_on.predict_state({})

        # For Calzetti, k(λ) is defined in the optical/UV and extrapolates flat in the IR
        # so transmission ≈ 1 for λ > 20 μm (T = exp(-tau*k) with k≈0)
        sed_ir_off = jnp.asarray(s_off.derived["sed_dust_ir"])
        sed_ir_on = jnp.asarray(s_on.derived["sed_dust_ir"])

        # Check IR (λ > 20 μm)
        ir_mask = jnp.asarray(s_off.wave) > 20000.0
        np.testing.assert_allclose(
            sed_ir_on[ir_mask], sed_ir_off[ir_mask],
            rtol=1e-12, atol=1e-30,
            err_msg="Calzetti IR emission should be unchanged (k=0 in IR)"
        )

    def test_power_law_transmission_integral(self, synthetic_ssp_wide):
        """power_law at tau_diff=3: integral ratio and emergent fraction correct."""
        m_off = _build(synthetic_ssp_wide, diffuse_screen=False,
                      law_bc="power_law", law_diff="power_law", tau_diff=3.0)
        m_on = _build(synthetic_ssp_wide, diffuse_screen=True,
                     law_bc="power_law", law_diff="power_law", tau_diff=3.0)

        s_off = m_off.predict_state({})
        s_on = m_on.predict_state({})

        sed_ir_off = jnp.asarray(s_off.derived["sed_dust_ir"])
        sed_ir_on = jnp.asarray(s_on.derived["sed_dust_ir"])
        dust_diff_t = jnp.asarray(s_on.derived["dust_diff_transmission"])
        wave = jnp.asarray(s_off.wave)

        # Compute transmission-weighted integral ratio independently
        integral_full = integrate_lnu_over_nu(sed_ir_off, wave)
        integral_screened = integrate_lnu_over_nu(sed_ir_off * dust_diff_t, wave)
        transmission_fraction = integral_screened / integral_full

        # Check that published integral ratio matches computed one
        log_L_ir = float(jnp.asarray(s_off.derived["log_L_ir"]))
        log_L_ir_emergent_expected = log_L_ir + np.log10(float(transmission_fraction))
        log_L_ir_emergent_published = float(jnp.asarray(s_on.derived["log_L_ir_emergent"]))

        np.testing.assert_allclose(
            log_L_ir_emergent_published, log_L_ir_emergent_expected,
            rtol=1e-6,
            err_msg="log_L_ir_emergent should match the transmission-weighted integral"
        )

        # Transmission fraction should be < 1 (energy is removed)
        assert transmission_fraction < 1.0, (
            f"Transmission fraction should be < 1, got {transmission_fraction}"
        )

    def test_energy_balance_unchanged(self, synthetic_ssp_wide):
        """L_absorbed and L_ir identical with switch on and off."""
        m_off = _build(synthetic_ssp_wide, diffuse_screen=False)
        m_on = _build(synthetic_ssp_wide, diffuse_screen=True)

        s_off = m_off.predict_state({})
        s_on = m_on.predict_state({})

        # L_ir and L_absorbed are pre-screen budgets (radio reads L_ir)
        np.testing.assert_array_equal(
            jnp.asarray(s_off.derived["L_ir"]),
            jnp.asarray(s_on.derived["L_ir"]),
            err_msg="L_ir should not change"
        )
        np.testing.assert_array_equal(
            jnp.asarray(s_off.derived["L_absorbed"]),
            jnp.asarray(s_on.derived["L_absorbed"]),
            err_msg="L_absorbed should not change"
        )

    def test_emitted_ir_multiplication(self, synthetic_ssp_wide):
        """Emitted IR with switch on equals (switch-off emission) × dust_diff_transmission."""
        m_off = _build(synthetic_ssp_wide, diffuse_screen=False)
        m_on = _build(synthetic_ssp_wide, diffuse_screen=True)

        s_off = m_off.predict_state({})
        s_on = m_on.predict_state({})

        sed_ir_off = jnp.asarray(s_off.derived["sed_dust_ir"])
        sed_ir_on = jnp.asarray(s_on.derived["sed_dust_ir"])
        dust_diff_t = jnp.asarray(s_on.derived["dust_diff_transmission"])

        # Check pointwise multiplication
        expected = sed_ir_off * dust_diff_t
        np.testing.assert_allclose(
            sed_ir_on, expected,
            rtol=1e-10,
            err_msg="sed_dust_ir should equal (original) × dust_diff_transmission"
        )

    def test_grammar_roundtrip(self, synthetic_ssp_wide):
        """Grammar round-trip: parse_groups → Parameters → back keeps diffuse_screen."""
        from tengri.parameters import parse_groups, parameters_to_groups

        spec = parse_groups(
            met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(10.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "calzetti",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(1.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={
                "type": "modified_blackbody",
                "diffuse_screen": True,
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.0),
        )

        # Check that diffuse_screen was parsed
        assert spec.dust_ir_diffuse_screen is True, "diffuse_screen should be parsed as True"

        # Round-trip back to groups
        groups = parameters_to_groups(spec)
        assert groups["dust_emission"]["diffuse_screen"] is True, (
            "diffuse_screen should round-trip through grammar"
        )

    def test_diffuse_screen_requires_attenuation(self, synthetic_ssp_wide):
        """Validate: diffuse_screen=True requires dust_attenuation to be active."""
        from tengri.parameters import parse_groups

        with pytest.raises(ValueError, match="requires dust_attenuation to be active"):
            parse_groups(
                met={"logzsol": Fixed(0.0)},
                sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
                      "log_total_mass": Fixed(10.0)},
                dust_attenuation={"type": "none"},  # Attenuation OFF
                dust_emission={"type": "modified_blackbody", "diffuse_screen": True},
                redshift=Fixed(0.0),
            )

    def test_diffuse_screen_requires_emission_type(self, synthetic_ssp_wide):
        """Validate: diffuse_screen=True requires dust_emission type to be specified."""
        from tengri.parameters import parse_groups

        with pytest.raises(ValueError, match="requires a dust_emission type"):
            parse_groups(
                met={"logzsol": Fixed(0.0)},
                sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
                      "log_total_mass": Fixed(10.0)},
                dust_attenuation={"type": "two_component"},
                dust_emission={"diffuse_screen": True},  # No type specified
                redshift=Fixed(0.0),
            )

    def test_waveprecomp_vs_exact(self, synthetic_ssp_wide):
        """WavePrecomp photometry vs exact path agree with diffuse_screen=True."""
        from tengri.observation import Photometry, WavePrecomp

        # Build with WavePrecomp
        m = _build(synthetic_ssp_wide, diffuse_screen=True)

        # Create a simple filter set (use GALEX FUV + a few Herschel bands)
        filters = [
            FilterCurve(wave=np.array([1350.0, 1500.0]), trans=np.array([0.5, 1.0])),
            FilterCurve(wave=np.array([100000.0, 160000.0]), trans=np.array([0.8, 0.5])),
        ]
        phot_exact = Photometry(filters=filters, approx=None)
        phot_precomp = Photometry(filters=filters, approx=WavePrecomp())

        # Predict
        pred_exact = m.predict({"redshift": 0.0}, obs=phot_exact)
        pred_precomp = m.predict({"redshift": 0.0}, obs=phot_precomp)

        # Fluxes should agree (WavePrecomp with diffuse_screen uses exact, not LUT)
        np.testing.assert_allclose(
            pred_exact.fluxes, pred_precomp.fluxes,
            rtol=1e-4,  # Allow some numerical tolerance for interpolation
            err_msg="WavePrecomp and exact photometry should agree"
        )


@pytest.mark.slow
def test_must_keep_passing_dust_tests(synthetic_ssp_wide):
    """Sanity check: other dust tests still pass with diffuse_screen=False (default)."""
    # Build a model with default diffuse_screen=False
    m = _build(synthetic_ssp_wide, diffuse_screen=False)
    s = m.predict_state({})

    # Basic checks that dust emission works
    assert s.derived["sed_dust_ir"] is not None
    assert s.derived["L_ir"] is not None
    assert s.derived["L_absorbed"] is not None
    assert s.derived["dust_diff_transmission"] is not None

    # When switch is off, log_L_ir_emergent should not be published
    # (to keep off bit-identical)
    assert s.derived.get("log_L_ir_emergent") is None, (
        "log_L_ir_emergent should not be published when diffuse_screen=False"
    )
