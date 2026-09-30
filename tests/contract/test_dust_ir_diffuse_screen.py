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

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp
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


#: Real Spitzer/Herschel curves vendored in ``data/filters/`` (no network
#: access needed: Spitzer_MIPS_24mu.dat, Herschel_Pacs_blue/red.dat), used in
#: place of a UV/optical tophat because dust *emission* -- not attenuation --
#: is what the diffuse screen re-attenuates here, and it contributes nothing
#: at optical wavelengths. Ordered blue-to-red so a ratio array's indices
#: read MIPS24, PACS70, PACS160.
IR_BAND_NAMES = ("mips_24", "herschel_70", "herschel_160")


def _ir_filter_curves() -> tuple[FilterCurve, ...]:
    from tengri.observation.filters import load_filter_set

    _, _, curves = load_filter_set(list(IR_BAND_NAMES))
    return tuple(curves)


def _build_emitting(ssp, diffuse_screen: bool, approx, *, tau_diff: float = 3.0):
    """Two-component power_law diffuse screen, evaluated through real IR bands.

    ``tau_diff=3`` (matching :class:`TestTransmissionRatioIdentity`) and
    ``redshift=0`` so the Charlot & Fall (n=-0.7) power-law curve gives a
    transmission that is actually far from 1 at these rest-frame wavelengths
    (T ~= 0.81 at 24um, ~= 0.90 at 70um, ~= 0.95 at 160um) -- unlike the
    previous ``tau_diff=0.3`` tophat-at-100um fixture, where T ~= 0.992 was
    indistinguishable from a no-op at the tolerances a LUT-vs-exact comparison
    can use, which is exactly why the mutation-testing three (#2533) survived
    here: dropping the screen multiply moved the LUT photometry by less than
    the assertion's slack. ``tau_bc``/birth-cloud dust play no role in the
    *diffuse* screen (:func:`tengri.components.dust.two_component._screen_transmission`
    with ``choice="diffuse"`` reads only ``tau_diff``/``k_diff``), so it is
    held Fixed at the two_component helper's default (0) to keep every build
    here fully pinned (no free params to sample).
    """
    dust = _two_component(tau_diff=tau_diff, law="power_law")
    obs = Observation(photometry=Photometry(filters=_ir_filter_curves()))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=approx,
        dust_attenuation=dust,
        dust_emission=_emission(diffuse_screen=diffuse_screen),
        neb={"type": "none"},
        redshift=Fixed(0.0),
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
#
# Modeled on the diagnosis behind the #2533 mutation-testing three: the
# previous version of this class used UV/optical tophats (blind to dust
# emission) plus a single very-long-wavelength tophat at tau_diff=0.3, where
# the diffuse screen's transmission at that wavelength was ~0.992 -- a <1%
# effect invisible at the rtol=5e-2 the LUT-vs-exact comparison used. Real
# Spitzer/Herschel bands at tau_diff=3 make the screen's effect (5-19%
# depending on band) large enough to fail with any of the three lines
# mutated, and the full-state assertions below target the published dict
# mutations directly rather than only through a projected photometry number.


class TestWavePrecompAgreesWithExact:
    def test_declines_band_response(self, synthetic_ssp_wide):
        """diffuse_screen=True must decline the linear-model band-response LUT.

        ``_apply_photometry_precomp``'s ``band_response`` branch reads
        ``L_ir * band_response`` directly and never touches ``sed_ir``, so if
        the band-response cache were not declined, the diffuse screen would
        be silently invisible to any photometry that hits that fast path.
        """
        m_lut = _build_emitting(synthetic_ssp_wide, True, WavePrecomp())
        decline = getattr(m_lut, "_dust_band_response_decline", None)
        assert decline is not None, "diffuse_screen=True must decline the band-response LUT"
        assert "diffuse" in decline.lower() or "screen" in decline.lower(), decline

    def test_lut_photometry_matches_exact_and_is_observably_switched(self, synthetic_ssp_wide):
        """WavePrecomp photometry: on == exact's on, and on != off by a real amount."""
        ssp = synthetic_ssp_wide
        m_lut_on = _build_emitting(ssp, True, WavePrecomp())
        m_lut_off = _build_emitting(ssp, False, WavePrecomp())
        m_exact_on = _build_emitting(ssp, True, None)

        phot_on = np.asarray(m_lut_on.predict_photometry({}))
        phot_off = np.asarray(m_lut_off.predict_photometry({}))
        phot_exact_on = np.asarray(m_exact_on.predict_photometry({}))

        # The LUT path (switch on) must agree with the exact path (switch on):
        # both read the same published sed_dust_ir/T once band_response is
        # declined, so this is a tight numerical check, not a 5%-slack one.
        np.testing.assert_allclose(phot_on, phot_exact_on, rtol=1e-6)

        # The switch must be OBSERVABLE in every band (mutant (a): removing
        # ``sed_ir = sed_ir * dust_diff_t`` would make this ratio 1.0).
        ratio = phot_on / phot_off
        assert np.all(ratio < 0.999), (
            f"diffuse screen must measurably reduce every IR band "
            f"({dict(zip(IR_BAND_NAMES, ratio, strict=True))})"
        )
        # A power-law screen's k(lambda) decreases toward longer wavelengths,
        # so the fractional loss must shrink from MIPS24 -> PACS70 -> PACS160
        # (i.e. the ratio must increase): a flat, band-independent ratio would
        # mean the screen is not really being evaluated at each band's own
        # wavelength.
        assert ratio[0] < ratio[1] < ratio[2], (
            f"MIPS24/PACS70/PACS160 on/off ratios must increase with wavelength: "
            f"{dict(zip(IR_BAND_NAMES, ratio, strict=True))}"
        )

    def test_lut_full_grid_sed_and_log_l_ir_emergent_match_exact(self, synthetic_ssp_wide):
        """Full-state (not just projected) assertions targeting all three mutants.

        Built from ``predict_state`` (not ``predict_photometry``) on a
        WavePrecomp model so ``state.derived`` is read straight out of the LUT
        branch of ``EmissionComponent.apply``, the branch every #2533 mutant
        lived in.
        """
        ssp = synthetic_ssp_wide
        m_lut_on = _build_emitting(ssp, True, WavePrecomp())
        m_lut_off = _build_emitting(ssp, False, WavePrecomp())
        m_exact_on = _build_emitting(ssp, True, None)

        s_lut_on = m_lut_on.predict_state({})
        s_lut_off = m_lut_off.predict_state({})
        s_exact_on = m_exact_on.predict_state({})

        # Mutants (a) + (b): the LUT branch's own published sed_dust_ir must
        # equal the switch-off spectrum times the published transmission,
        # exactly like the exact-path pointwise test above.
        T = np.asarray(s_lut_on.derived["dust_diff_transmission"])
        S_off = np.asarray(s_lut_off.derived["sed_dust_ir"])
        S_on = np.asarray(s_lut_on.derived["sed_dust_ir"])
        np.testing.assert_allclose(S_on, S_off * T, rtol=1e-10, atol=0.0)
        assert np.any(S_on < S_off * (1.0 - 1e-9)), (
            "LUT-path published sed_dust_ir must not be a silent no-op"
        )

        # Mutant (c): the LUT branch's log_L_ir_emergent numerator must
        # include the ``* dust_diff_t`` factor, matching the exact path.
        log_emergent_lut = float(jnp.asarray(s_lut_on.derived["log_L_ir_emergent"]))
        log_emergent_exact = float(jnp.asarray(s_exact_on.derived["log_L_ir_emergent"]))
        np.testing.assert_allclose(log_emergent_lut, log_emergent_exact, rtol=1e-6)


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
