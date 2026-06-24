# SPDX-License-Identifier: BSD-3-Clause
"""Tests for :class:`DustEmissionSEDComponent` and the dust energy-balance loop.

Two layers:

1. **Standalone IR emission** — given an L_ir scalar already on
   ``state.derived``, the adapter reproduces
   :func:`modified_blackbody` exactly.
2. **Closed loop** — DustAttenuation + DustEmission together approximately
   conserve energy across the attenuation/re-emission round-trip. The
   absolute energy balance won't be perfect (modified blackbody peaks
   in FIR, attenuation is in UV/optical/NIR, the integration grids
   differ), but sed_dust_ir integrated over frequency must equal
   ``state.derived["L_ir"]`` to ~1% in well-sampled bands.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.dust.component import DustAttenuationSEDComponent
from tengri.components.dust.emission import modified_blackbody
from tengri.components.dust.emission_component import DustEmissionSEDComponent
from tengri.forward.orchestrator import merge_declared_parameters, run_components
from tengri.protocols import ForwardState
from tengri.utils.physics_constants import C_AA

REL_TOL = 1e-10


@pytest.mark.parametrize(
    ("dust_T", "dust_beta_ir", "L_ir"),
    [
        (20.0, 1.5, 1e44),
        (30.0, 1.8, 5e44),
        (45.0, 2.0, 1e45),
        (60.0, 1.6, 1e43),
    ],
)
def test_emission_matches_modified_blackbody(dust_T, dust_beta_ir, L_ir):
    """Pipeline output equals direct modified_blackbody call."""
    wave = jnp.logspace(2, 8, 1024)  # 100 Å to 100 mm — covers UV through mm
    state = ForwardState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        derived={"L_ir": L_ir},
    )
    params = {
        "redshift": 0.0,
        "dust_T": dust_T,
        "dust_beta_ir": dust_beta_ir,
    }

    final = run_components([DustEmissionSEDComponent()], state, params)

    expected = modified_blackbody(
        wave,
        L_absorbed=L_ir,
        dust_T=dust_T,
        dust_beta_ir=dust_beta_ir,
        redshift=0.0,
    )

    assert jnp.allclose(final.sed_intrinsic, expected, rtol=REL_TOL, atol=0.0)
    assert "sed_dust_ir" in final.derived


@pytest.mark.unit
def test_emission_is_noop_when_l_ir_is_zero():
    """No upstream attenuator → L_ir = 0 → adapter contributes nothing."""
    wave = jnp.logspace(2, 8, 64)
    state = ForwardState(wave=wave, sed_intrinsic=jnp.zeros_like(wave))

    out = DustEmissionSEDComponent().apply(
        state, {"redshift": 0.0, "dust_T": 30.0, "dust_beta_ir": 1.8}
    )

    # modified_blackbody(L_absorbed=0) returns zeros.
    assert jnp.allclose(out.sed_intrinsic, 0.0, atol=1e-30)


@pytest.mark.unit
def test_attenuation_publishes_l_ir_for_emission_to_consume():
    """Energy-balance handshake: attenuation publishes L_ir; emission reads it."""
    wave = jnp.logspace(2, 8, 1024)
    intrinsic = jnp.ones_like(wave) * 1e30  # uniform L_nu in cgs

    state = ForwardState(wave=wave, sed_intrinsic=intrinsic)

    after_attenuation = DustAttenuationSEDComponent().apply(
        state, {"redshift": 0.0, "dust_tau_v": 0.5}
    )

    # Attenuation must publish a non-zero L_ir for any reasonable tau.
    assert "L_ir" in after_attenuation.derived
    L_ir = float(after_attenuation.derived["L_ir"])
    assert L_ir > 0.0
    assert jnp.isfinite(L_ir)


@pytest.mark.unit
def test_attenuation_l_ir_zero_when_no_attenuation():
    """tau_v = 0 → no absorbed luminosity → L_ir = 0."""
    wave = jnp.logspace(2, 8, 256)
    intrinsic = jnp.ones_like(wave) * 1e30

    state = ForwardState(wave=wave, sed_intrinsic=intrinsic)
    after = DustAttenuationSEDComponent().apply(state, {"redshift": 0.0, "dust_tau_v": 0.0})

    assert float(after.derived["L_ir"]) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.unit
def test_closed_loop_chain_runs():
    """Attenuation → Emission produces both attenuated and IR-augmented SEDs."""
    wave = jnp.logspace(2, 8, 1024)
    intrinsic = jnp.ones_like(wave) * 1e30

    state = ForwardState(wave=wave, sed_intrinsic=intrinsic)
    final = run_components(
        [DustAttenuationSEDComponent(), DustEmissionSEDComponent()],
        state,
        {"redshift": 0.0, "dust_tau_v": 0.5, "dust_T": 30.0, "dust_beta_ir": 1.8},
    )

    # Attenuated SED must exist (Phase II-1 attenuator wrote it).
    assert final.sed_attenuated is not None
    # Emission added FIR luminosity to sed_intrinsic.
    assert jnp.any(final.sed_intrinsic > intrinsic)
    # All energy-balance keys present.
    for key in ("L_ir", "dust_attenuation_factor", "sed_dust_ir"):
        assert key in final.derived


@pytest.mark.unit
def test_emitted_luminosity_integral_matches_l_ir():
    """∫ sed_dust_ir dν ≈ L_ir to within 1%.

    The modified-blackbody normalization is a JAX trapezoid in
    frequency; our attenuation L_ir uses the same trapezoid scheme.
    They should agree to numerical-integration precision.
    """
    wave = jnp.logspace(2, 8, 4096)  # finer grid for the integral
    intrinsic = jnp.ones_like(wave) * 1e30

    state = ForwardState(wave=wave, sed_intrinsic=intrinsic)
    final = run_components(
        [DustAttenuationSEDComponent(), DustEmissionSEDComponent()],
        state,
        {"redshift": 0.0, "dust_tau_v": 0.5, "dust_T": 30.0, "dust_beta_ir": 1.8},
    )

    L_ir = float(final.derived["L_ir"])
    L_emitted_lnu = final.derived["sed_dust_ir"]

    # Integrate emitted L_nu over frequency using the same convention
    # as the attenuator's L_ir computation (cgs trapezoid in ν).
    nu = C_AA / wave
    order = jnp.argsort(nu)
    L_emitted_total = float(jnp.trapezoid(L_emitted_lnu[order], nu[order]))

    assert L_emitted_total == pytest.approx(L_ir, rel=0.01), (
        f"L_emitted={L_emitted_total:.3e} should match L_ir={L_ir:.3e} to 1%"
    )


@pytest.mark.unit
def test_merge_dust_attenuation_and_emission_no_collision():
    """Both dust adapters share parameter_prefix='dust_' but declare disjoint names.

    ``merge_declared_parameters`` accepts the pair because attenuation
    declares ``dust_tau_v`` and emission declares ``dust_T`` /
    ``dust_beta_ir`` — no name collisions.
    """
    merged = merge_declared_parameters([DustAttenuationSEDComponent(), DustEmissionSEDComponent()])
    assert "dust_tau_v" in merged
    assert "dust_T" in merged
    assert "dust_beta_ir" in merged
