# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the FSPS-parity ``eb_include_lyc`` toggle changes L_absorbed (#961).

The canonical energy balance masks the Lyman continuum out of ``L_absorbed``
(#922: LyC photons ionize H, they don't heat dust). FSPS/Prospector re-emit
the *full* absorbed luminosity, which left tengri's far-IR ~10 % low at the
Prospector reproduction fiducial. ``dust={'eb_include_lyc': True}`` opts in
to the FSPS convention; these tests pin that the toggle (a) reaches the
forward pass (not a silent no-op), (b) reproduces the manual masked/unmasked
integrals against a dust-free twin's intrinsic SED, (c) round-trips through
the build grammar, and (d) is baked identically into the WavePrecomp
energy-balance LUT.

CI-runnable on the synthetic wide SSP (no ``data/`` grids needed).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.observation.photometry import FilterCurve
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.contract


def _build(ssp, include_lyc: bool, *, tau_diff: float = 1.0):
    dust = {
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(tau_diff),
        "*": FIXED,
    }
    if include_lyc:
        dust["eb_include_lyc"] = True
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "*": FIXED,
        },
        dust_attenuation=dust,
        redshift=Fixed(0.0),
    )


def _manual_absorbed(sed_intrinsic, sed_attenuated, wave, *, mask_lyc: bool) -> float:
    nu = np.asarray(C_AA) / np.asarray(wave)
    absorbed = np.asarray(sed_intrinsic) - np.asarray(sed_attenuated)
    if mask_lyc:
        absorbed = np.where(np.asarray(wave) >= 912.0, absorbed, 0.0)
    return float(abs(np.trapezoid(absorbed, nu)))


class TestEnergyBalanceLycToggle:
    def test_toggle_reaches_forward_pass(self, synthetic_ssp_wide):
        """FSPS-parity L_absorbed must exceed the LyC-masked default."""
        s_masked = _build(synthetic_ssp_wide, False).predict_state({})
        s_full = _build(synthetic_ssp_wide, True).predict_state({})
        L_masked = float(jnp.asarray(s_masked.derived["L_absorbed"]))
        L_full = float(jnp.asarray(s_full.derived["L_absorbed"]))
        # The synthetic SSP is LyC-bright, so including the LyC must add energy.
        assert L_full > L_masked * 1.0001, (
            f"eb_include_lyc is a no-op: L_absorbed {L_masked:.6e} -> {L_full:.6e}"
        )

    def test_matches_manual_integrals(self, synthetic_ssp_wide):
        """L_absorbed equals the masked/unmasked integral of intrinsic - attenuated.

        The final state's ``sed_intrinsic`` is the running (post-attenuation)
        SED, so the pre-dust intrinsic comes from a dust-free twin model with
        identical stellar parameters.
        """
        s_free = _build(synthetic_ssp_wide, False, tau_diff=0.0).predict_state({})
        s_masked = _build(synthetic_ssp_wide, False).predict_state({})
        s_full = _build(synthetic_ssp_wide, True).predict_state({})
        intrinsic = s_free.sed_intrinsic
        np.testing.assert_allclose(
            float(jnp.asarray(s_masked.derived["L_absorbed"])),
            _manual_absorbed(
                intrinsic, s_masked.derived["sed_dust_attenuated"], s_masked.wave, mask_lyc=True
            ),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            float(jnp.asarray(s_full.derived["L_absorbed"])),
            _manual_absorbed(
                intrinsic, s_full.derived["sed_dust_attenuated"], s_full.wave, mask_lyc=False
            ),
            rtol=1e-6,
        )

    def test_grammar_round_trip(self, synthetic_ssp_wide):
        m = _build(synthetic_ssp_wide, True)
        groups = m.spec.to_groups()
        assert groups["dust"].get("eb_include_lyc") is True
        m_default = _build(synthetic_ssp_wide, False)
        assert "eb_include_lyc" not in m_default.spec.to_groups()["dust"]


def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _build_emitting(ssp, include_lyc: bool, approx):
    dust = {
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "tau_bc": Uniform(0.0, 1.0),
        "tau_diff": Fixed(0.3),
        "emission": {"type": "modified_blackbody", "*": FIXED},
        "*": FIXED,
    }
    if include_lyc:
        dust["eb_include_lyc"] = True
    centers = (3500.0, 6200.0, 1.0e6)
    obs = Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=approx,
        met={"logzsol": Fixed(0.0), "*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "*": FIXED,
        },
        dust_attenuation=dust,
        neb={"type": "none"},
        redshift=Fixed(0.05),
    )


def test_lut_bakes_the_same_convention(synthetic_ssp_wide):
    """The WavePrecomp energy-balance LUT must carry the toggle too (#961).

    If the runtime path included the LyC but the LUT kept the mask, the fast
    and exact paths would silently disagree on L_ir — the far-IR band exposes
    it. Track the exact path across the free tau_bc range.
    """
    ssp = synthetic_ssp_wide
    m_lut = _build_emitting(ssp, True, WavePrecomp())
    m_exact = _build_emitting(ssp, True, None)
    assert getattr(m_lut, "_energy_balance_lut_cache", None) is not None

    base = {**m_lut.spec.get_fixed_values(), **m_lut.spec.sample(jax.random.PRNGKey(0))}
    for tau in (0.0, 0.5, 1.0):
        p = dict(base)
        p["dust_tau_bc"] = jnp.asarray(float(tau))
        a = np.asarray(m_lut.predict_photometry(p))
        b = np.asarray(m_exact.predict_photometry(p))
        np.testing.assert_allclose(a[-1], b[-1], rtol=5e-2)  # far-IR band carries L_ir
