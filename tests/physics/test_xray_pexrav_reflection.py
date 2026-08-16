# SPDX-License-Identifier: BSD-3-Clause
"""Physics tests for pexrav cold-disc Compton reflection.

Verifies the Magdziarz & Zdziarski (1995) reflection approximation in
``pexrav_reflection``:

* ``regression_paper`` — albedo shape vs MZ95 Fig. 1: rises from zero
  in the soft band, peaks around 30 keV, falls off above ~ 100 keV.
* ``limit`` — R = 0 disables reflection entirely; large R linearly
  scales it.
* ``bounds`` — reflection is non-negative everywhere and zero outside
  the X-ray band.
* ``gradient`` — dL_refl/dR is finite and exactly positive (linear).
* ``regression_paper`` — full Eq. B6 model (primary + reflection + N_H
  + scattered) reproduces the Compton-thick "buried hump" signature.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.xray import (
    pexrav_reflection,
    xray_agn_corona,
)
from tests._grad_parity import assert_grad_matches_fd


@pytest.mark.regression_paper
def test_albedo_peaks_near_30_kev() -> None:
    """Reflection albedo (L_refl / L_primary) peaks in the 20–60 keV band.

    Matches Magdziarz & Zdziarski 1995 Fig. 1: cold-disc reflection
    has a Compton hump that peaks at ~30 keV due to (a) the soft band
    being absorbed by metals in the disc surface and (b) Klein-Nishina
    suppression above ~100 keV.
    """
    E_keV = np.logspace(-0.5, 2.5, 200)  # 0.3 – 300 keV
    wave = 12.398 / E_keV  # Å
    l_primary = jnp.ones_like(wave)  # unit primary spectrum
    l_refl = pexrav_reflection(jnp.asarray(wave), l_primary, R=1.0)

    # Albedo (= L_refl / L_primary) since L_primary = 1.
    A = np.asarray(l_refl)
    peak_E = E_keV[np.argmax(A)]
    assert 20.0 < peak_E < 60.0, f"peak at {peak_E:.1f} keV, expected 20–60"


@pytest.mark.regression_paper
def test_albedo_amplitude_matches_mz95() -> None:
    """Albedo at the Compton-hump peak ~ 0.3 for R=1, cos_inc=0.5.

    MZ95 Fig. 1 shows the peak albedo for a 60° viewing angle reaches
    A ≈ 0.3–0.5 depending on the primary spectral slope. Our simplified
    multiplicative model produces ~ 0.3–0.45 at R = 1 and 60°.
    """
    wave = jnp.array([12.398 / 30.0])  # 30 keV
    l_primary = jnp.array([1.0])
    A = float(pexrav_reflection(wave, l_primary, R=1.0, cos_inc=0.5)[0])
    assert 0.25 < A < 0.55, f"albedo at 30 keV = {A:.3f}, expected 0.25–0.55"


@pytest.mark.limit
def test_R_zero_disables_reflection() -> None:
    """R = 0 ⇒ zero reflection spectrum everywhere."""
    wave = jnp.logspace(0.0, 2.0, 100)
    l_primary = jnp.ones_like(wave)
    L = pexrav_reflection(wave, l_primary, R=0.0)
    chex.assert_trees_all_close(L, jnp.zeros_like(wave), atol=0.0)


@pytest.mark.limit
def test_R_linear_scaling() -> None:
    """L_refl ∝ R exactly (linear knob).

    The reflection-covering-fraction multiplier is a single scalar; the
    output must scale linearly. Locks the post-test refactor against
    accidental introduction of non-linear R dependence.
    """
    wave = jnp.logspace(0.0, 2.0, 50)
    l_primary = jnp.ones_like(wave) * 1.5
    L_a = pexrav_reflection(wave, l_primary, R=0.5)
    L_b = pexrav_reflection(wave, l_primary, R=2.0)
    chex.assert_trees_all_close(L_b, 4.0 * L_a, rtol=1e-12)


@pytest.mark.bounds
def test_reflection_non_negative_everywhere() -> None:
    """L_refl(E) ≥ 0 across the X-ray band, all R and inclinations.

    Physical bound: a reflection contribution can't be negative.
    """
    wave = jnp.logspace(0.0, 2.0, 200)
    l_primary = jnp.ones_like(wave) * 1e23
    for R in (0.1, 0.5, 1.0, 2.0):
        for cos_inc in (0.05, 0.3, 0.6, 1.0):
            L = pexrav_reflection(wave, l_primary, R=R, cos_inc=cos_inc)
            chex.assert_tree_all_finite(L)
            assert bool(jnp.all(L >= 0.0)), f"R={R} cos_inc={cos_inc}"


@pytest.mark.bounds
def test_zero_outside_xray_band() -> None:
    """Reflection vanishes outside λ < 124 Å (E < 0.1 keV).

    The optical / UV reflection from a cold disc is dominated by Wien-
    photosphere reprocessing (handled by the AGN disc model, not by
    pexrav). The X-ray reflection module returns zero outside the band.
    """
    wave = jnp.array([200.0, 1000.0, 5000.0])  # UV / optical
    l_primary = jnp.ones_like(wave) * 1e25
    L = pexrav_reflection(wave, l_primary, R=1.0)
    chex.assert_trees_all_close(L, jnp.zeros_like(wave), atol=0.0)


@pytest.mark.gradient
def test_dl_refl_dR_finite_and_positive() -> None:
    """dL_refl/dR is finite and strictly positive across the X-ray band.

    Required for gradient-based inference of the reflection covering
    fraction when fitting hard-X-ray data.
    """
    wave = jnp.array([12.398 / 30.0])  # 30 keV peak
    l_primary = jnp.array([1.0])

    def total_L(R):
        return pexrav_reflection(wave, l_primary, R=R)[0]

    grad = assert_grad_matches_fd(total_L, jnp.array(0.5))
    assert jnp.isfinite(grad)
    assert float(grad) > 0.0


@pytest.mark.regression_paper
def test_compton_thick_hump_visible_above_absorption() -> None:
    """At log N_H = 24, the pexrav hump dominates the 10–50 keV band.

    Ricci+2017 / Matsumoto+2026 Eq. B6 signature: when the line-of-sight
    absorber is Compton-thick, the *primary* corona is extinguished
    below ~ 10 keV, leaving the reflection hump as the dominant feature
    in the 10–50 keV band. This test reproduces the qualitative
    spectral shape behind Matsumoto+2026 Fig. 12.
    """
    wave = jnp.logspace(np.log10(0.5), np.log10(124.0), 800)
    L_2500 = 1e44 / (5.15 * 1.199e15)
    # Compton-thick, with reflection on
    L_with = xray_agn_corona(
        wave,
        l_2500_30deg_erg_hz=L_2500,
        log_nh=24.0,
        pexrav_R=0.5,
        apply_anisotropy=False,
    )
    # Compton-thick, no reflection (baseline)
    L_no = xray_agn_corona(
        wave,
        l_2500_30deg_erg_hz=L_2500,
        log_nh=24.0,
        pexrav_R=0.0,
        apply_anisotropy=False,
    )

    E_keV = 12.398 / np.asarray(wave)
    hump_band = (E_keV >= 10.0) & (E_keV <= 50.0)
    boost = float(np.mean(np.asarray(L_with - L_no)[hump_band] / np.asarray(L_no)[hump_band]))
    # Reflection must boost the hump band by ≥ 50 % at R = 0.5
    # over the unreflected Compton-thick baseline.
    assert boost > 0.5, f"hump boost = {boost:.2f}, expected > 0.5"
