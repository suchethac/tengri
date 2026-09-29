# SPDX-License-Identifier: BSD-3-Clause
"""Contract: opt-in single-pass diffuse-screen attenuation of re-emitted IR (#2533).

``dust_emission={'type': ..., 'diffuse_screen': True}`` multiplies the
re-emitted IR dust emission ONCE by the diffuse dust screen's transmission
T(lambda), published by every attenuator as the derived key
``dust_diff_transmission``. The absorbed IR energy is REMOVED -- no
iteration, no renormalization back to ``L_ir``. ``log_L_ir_emergent``
reports the escaping IR; ``L_ir``/``L_absorbed`` keep the pre-screen budget.
Default is off and must be bit-identical to a build without the key.

Modeled on ``tests/contract/test_energy_balance_lyc_toggle.py``: same
synthetic-SSP/SFH/met setup, same WavePrecomp-vs-exact tolerance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.dust.emission._physics import integrate_lnu_over_nu
from tengri.observation.photometry import FilterCurve
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract

# ── Shared build helpers (mirrors test_energy_balance_lyc_toggle.py) ───────


def _sfh_met_kwargs() -> dict:
    return dict(
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
    )


def _two_component(tau_diff: float, *, law: str = "power_law", tau_bc: float = 0.0) -> dict:
    return {
        "type": "two_component",
        "law_bc": law,
        "law_diff": law,
        "tau_bc": Fixed(tau_bc),
        "tau_diff": Fixed(tau_diff),
        "all_params": Fixed(DEFAULT),
    }


def _single_component(tau_v: float, *, law: str = "power_law") -> dict:
    return {
        "type": "single_component",
        "law": law,
        "tau_v": Fixed(tau_v),
        "all_params": Fixed(DEFAULT),
    }


def _wg00(tau_v: float) -> dict:
    return {
        "type": "wg00",
        "geometry": "shell",
        "tau_v": Fixed(tau_v),
        "all_params": Fixed(DEFAULT),
    }


def _emission(
    *,
    diffuse_screen: bool | None = None,
    emission_type: str = "modified_blackbody",
    dust_T: float | None = None,
) -> dict:
    d: dict = {"type": emission_type, "all_params": Fixed(DEFAULT)}
    if diffuse_screen is not None:
        d["diffuse_screen"] = diffuse_screen
    if dust_T is not None:
        d["T"] = Fixed(dust_T)
    return d


def _build(ssp, dust_attenuation: dict, dust_emission: dict, *, redshift: float = 0.0):
    return SEDModel.build(
        ssp_data=ssp,
        dust_attenuation=dust_attenuation,
        dust_emission=dust_emission,
        neb={"type": "none"},
        redshift=Fixed(redshift),
        **_sfh_met_kwargs(),
    )


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _build_emitting(ssp, diffuse_screen: bool, approx):
    dust = _two_component(tau_diff=0.3, law="power_law")
    dust["tau_bc"] = Uniform(0.0, 1.0)
    centers = (3500.0, 6200.0, 1.0e6)
    obs = Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=approx,
        dust_attenuation=dust,
        dust_emission=_emission(diffuse_screen=diffuse_screen),
        neb={"type": "none"},
        redshift=Fixed(0.05),
        **_sfh_met_kwargs(),
    )


# ── 1. Switch off: bit-identical to a build without the key ────────────────


class TestSwitchOffBitIdentical:
    def test_default_off_matches_explicit_false(self, synthetic_ssp_wide):
        """No ``diffuse_screen`` key and ``diffuse_screen=False`` must agree exactly."""
        atten = _two_component(tau_diff=1.0)
        s_default = _build(synthetic_ssp_wide, atten, _emission()).predict_state({})
        s_false = _build(synthetic_ssp_wide, atten, _emission(diffuse_screen=False)).predict_state(
            {}
        )

        np.testing.assert_array_equal(
            np.asarray(s_default.sed_intrinsic), np.asarray(s_false.sed_intrinsic)
        )
        d_default = s_default.derived.to_dict()
        d_false = s_false.derived.to_dict()
        assert set(d_default) == set(d_false), (
            f"derived key sets differ: only-default={set(d_default) - set(d_false)}, "
            f"only-false={set(d_false) - set(d_default)}"
        )
        for key, value in d_default.items():
            np.testing.assert_array_equal(
                np.asarray(value), np.asarray(d_false[key]), err_msg=f"derived[{key!r}] differs"
            )


# ── 2. Calzetti single_component: IR beyond 20um unchanged (k=0 there) ─────


class TestCalzettiLongWaveUnaffected:
    def test_ir_beyond_20_micron_unchanged(self, synthetic_ssp_wide):
        """Calzetti's curve clips to exactly k=0 above ~3.1um, so T=1 exactly
        past 20um: the screen must be a no-op there to 1e-12 relative."""
        atten = _single_component(tau_v=1.5, law="calzetti")
        s_off = _build(synthetic_ssp_wide, atten, _emission(diffuse_screen=False)).predict_state(
            {}
        )
        s_on = _build(synthetic_ssp_wide, atten, _emission(diffuse_screen=True)).predict_state({})

        wave_aa = np.asarray(s_off.wave)
        mask = wave_aa > 20.0 * 1e4  # 20 um in Angstrom
        assert mask.sum() > 5, "synthetic_ssp_wide grid does not reach past 20um"

        ir_off = np.asarray(s_off.derived["sed_dust_ir"])[mask]
        ir_on = np.asarray(s_on.derived["sed_dust_ir"])[mask]
        np.testing.assert_allclose(ir_on, ir_off, rtol=1e-12)

        # Sanity: the transmission really is 1.0 there (k=0), not coincidence.
        T = np.asarray(s_on.derived["dust_diff_transmission"])[mask]
        np.testing.assert_allclose(T, 1.0, rtol=1e-12)


# ── 3. Transmission-ratio identity across all three attenuators ────────────


def _atten_power_law(name: str) -> dict:
    if name == "two_component":
        return _two_component(tau_diff=3.0, law="power_law")
    if name == "single_component":
        return _single_component(tau_v=3.0, law="power_law")
    if name == "wg00":
        return _wg00(tau_v=3.0)
    raise ValueError(name)


#: WG00's vendored table (data/wg00_attenuation_grid.h5) is tabulated only
#: 1000-30001 A (0.1-3 um): a physically ordinary dust temperature's emission
#: peaks far redward of that, where the (pre-existing) WG00 curve is silent
#: and the screen is genuinely a no-op -- not a defect in the diffuse-screen
#: feature. Raise the dust temperature for this one sub-case only, so the
#: modified_blackbody emission overlaps WG00's tabulated domain and the
#: "some energy is removed" assertion is meaningful for this attenuator too.
_WG00_EMISSION_DUST_T = 1500.0


class TestTransmissionRatioIdentity:
    @pytest.mark.parametrize("atten_name", ["two_component", "single_component", "wg00"])
    def test_ratio_matches_log_l_ir_emergent_and_independent_integral(
        self, synthetic_ssp_wide, atten_name
    ):
        atten = _atten_power_law(atten_name)
        dust_T = _WG00_EMISSION_DUST_T if atten_name == "wg00" else None
        s_off = _build(
            synthetic_ssp_wide, atten, _emission(diffuse_screen=False, dust_T=dust_T)
        ).predict_state({})
        s_on = _build(
            synthetic_ssp_wide, atten, _emission(diffuse_screen=True, dust_T=dust_T)
        ).predict_state({})
        wave = s_on.wave

        L_ir = float(jnp.asarray(s_on.derived["L_ir"]))
        log_L_ir = float(jnp.asarray(s_on.derived["log_L_ir"]))
        log_L_ir_emergent = float(jnp.asarray(s_on.derived["log_L_ir_emergent"]))
        ratio_from_logs = 10.0 ** (log_L_ir_emergent - log_L_ir)

        integral_on = float(integrate_lnu_over_nu(jnp.asarray(s_on.derived["sed_dust_ir"]), wave))
        ratio_from_integral = integral_on / L_ir

        S_off = jnp.asarray(s_off.derived["sed_dust_ir"])
        T = jnp.asarray(s_off.derived["dust_diff_transmission"])
        ratio_independent = float(
            integrate_lnu_over_nu(S_off * T, wave) / integrate_lnu_over_nu(S_off, wave)
        )

        np.testing.assert_allclose(ratio_from_integral, ratio_from_logs, rtol=1e-6)
        np.testing.assert_allclose(ratio_independent, ratio_from_logs, rtol=1e-6)
        assert ratio_from_logs < 1.0, (
            f"{atten_name}: diffuse screen must remove some IR energy, got ratio={ratio_from_logs}"
        )
        print(
            f"[diffuse_screen] {atten_name} power_law tau=3: "
            f"integral S*T/S = {ratio_from_logs:.6f}"
        )


# ── 4. L_absorbed / L_ir unchanged by the switch ────────────────────────────


class TestEnergyBudgetUnchanged:
    def test_l_absorbed_and_l_ir_identical_on_vs_off(self, synthetic_ssp_wide):
        atten = _two_component(tau_diff=3.0, law="power_law")
        s_off = _build(synthetic_ssp_wide, atten, _emission(diffuse_screen=False)).predict_state(
            {}
        )
        s_on = _build(synthetic_ssp_wide, atten, _emission(diffuse_screen=True)).predict_state({})
        for key in ("L_absorbed", "L_ir", "log_L_absorbed", "log_L_ir"):
            np.testing.assert_array_equal(
                np.asarray(s_off.derived[key]), np.asarray(s_on.derived[key]), err_msg=key
            )


# ── 5. Pointwise: emitted IR (on) = emitted IR (off) * T ───────────────────


class TestPointwiseScreenMultiplication:
    def test_sed_dust_ir_equals_offswitch_times_transmission(self, synthetic_ssp_wide):
        atten = _two_component(tau_diff=3.0, law="power_law")
        s_off = _build(synthetic_ssp_wide, atten, _emission(diffuse_screen=False)).predict_state(
            {}
        )
        s_on = _build(synthetic_ssp_wide, atten, _emission(diffuse_screen=True)).predict_state({})

        T = np.asarray(s_on.derived["dust_diff_transmission"])
        S_off = np.asarray(s_off.derived["sed_dust_ir"])
        expected = S_off * T
        actual = np.asarray(s_on.derived["sed_dust_ir"])
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0.0)

        # The screen must actually do something (not a silent no-op): at
        # least part of the grid must be attenuated below the *unscreened*
        # (switch-off) emission -- comparing `actual` to `expected` here
        # would be circular, since `expected` is defined as `S_off * T`.
        assert np.any(actual < S_off * (1.0 - 1e-9))


# ── 6. WavePrecomp vs exact + band-response decline recorded ───────────────


class TestWavePrecompAgreesWithExact:
    def test_lut_matches_exact_and_declines_band_response(self, synthetic_ssp_wide):
        ssp = synthetic_ssp_wide
        m_lut = _build_emitting(ssp, True, WavePrecomp())
        m_exact = _build_emitting(ssp, True, None)

        decline = getattr(m_lut, "_dust_band_response_decline", None)
        assert decline is not None, "diffuse_screen=True must decline the band-response LUT"
        assert "diffuse" in decline.lower() or "screen" in decline.lower(), decline

        base = m_lut.spec.sample(jax.random.PRNGKey(0))
        for tau in (0.0, 0.5, 1.0):
            p = dict(base)
            p["dust_tau_bc"] = jnp.asarray(float(tau))
            a = np.asarray(m_lut.predict_photometry(p))
            b = np.asarray(m_exact.predict_photometry(p))
            np.testing.assert_allclose(a[-1], b[-1], rtol=5e-2)  # far-IR band carries L_ir


# ── 7. Grammar round-trip + parse-time refusals ─────────────────────────────


class TestGrammarRoundTripAndRefusals:
    def test_round_trip_keeps_diffuse_screen(self, synthetic_ssp_wide):
        atten = _two_component(tau_diff=3.0, law="power_law")
        m_on = _build(synthetic_ssp_wide, atten, _emission(diffuse_screen=True))
        groups_on = m_on.spec.to_groups()
        assert groups_on["dust_emission"].get("diffuse_screen") is True

        m_off = _build(synthetic_ssp_wide, atten, _emission())
        groups_off = m_off.spec.to_groups()
        assert "diffuse_screen" not in groups_off.get("dust_emission", {})

    def test_refuses_diffuse_screen_without_dust_attenuation(self):
        with pytest.raises(ValueError, match="diffuse_screen"):
            parse_groups(
                dust_attenuation={"type": "none"},
                dust_emission=_emission(diffuse_screen=True),
                redshift=Fixed(0.0),
                **_sfh_met_kwargs(),
            )

    def test_refuses_diffuse_screen_without_emission_type(self):
        atten = _two_component(tau_diff=1.0, law="power_law")
        with pytest.raises(ValueError, match="diffuse_screen"):
            parse_groups(
                dust_attenuation=atten,
                dust_emission={"diffuse_screen": True},
                redshift=Fixed(0.0),
                **_sfh_met_kwargs(),
            )
