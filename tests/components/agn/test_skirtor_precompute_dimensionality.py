# SPDX-License-Identifier: BSD-3-Clause
"""Regression: SKIRTOR WavePrecomp LUT carries L_ν dimensionality (issue #459).

The #459 fix corrected ``skirtor.py:_interpolate_and_normalize`` to normalise the
bolometric integral in the *wavelength* variable and convert L_λ → L_ν = L_λ·λ²/c
at the end (SKIRTOR v3 templates are stored L_λ-like). The WavePrecomp path in
``skirtor_precompute.py`` was missed and kept integrating against frequency and
treating the L_λ array as L_ν — leaving the photometry lookup off by a factor
~λ² relative to the runtime path.

This pins the precompute output to the exact runtime ``create_skirtor_from_grid``
SED via a convention-independent flux *ratio* between a mid-IR and a far-IR
filter. The λ² error would distort that ratio by ~(λ_mid/λ_far)²; the fixed
code keeps the two paths consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def _tophat(centre_aa: float, width_frac: float = 0.1, n: int = 64):
    """A narrow top-hat filter (wave [Å], transmission) around ``centre_aa``."""
    half = centre_aa * width_frac / 2.0
    wave = np.linspace(centre_aa - half, centre_aa + half, n)
    trans = np.ones_like(wave)
    return wave, trans


def test_precompute_matches_runtime_midir_to_farir_ratio():
    skirtor = pytest.importorskip("tengri.components.agn.skirtor")
    precompute_mod = pytest.importorskip("tengri.components.agn.skirtor_precompute")
    import jax.numpy as jnp

    try:
        grid_path = skirtor._find_skirtor_grid()
    except FileNotFoundError:
        pytest.skip("SKIRTOR template grid not available")

    # Two narrow filters straddling the torus bump and its FIR tail.
    mid_centre, far_centre = 1.2e5, 6.0e5  # 12 µm and 60 µm
    fw_mid, ft_mid = _tophat(mid_centre)
    fw_far, ft_far = _tophat(far_centre)

    precomp = precompute_mod.precompute_skirtor_photometry(
        grid_path,
        [jnp.asarray(fw_mid), jnp.asarray(fw_far)],
        [jnp.asarray(ft_mid), jnp.asarray(ft_far)],
        redshift=0.0,
    )
    grid_phot = np.asarray(precomp["grid_phot"])  # (..., n_filters), L_ν per L_sun
    # Flux ratio at one representative grid node (central index per axis).
    node = tuple(s // 2 for s in grid_phot.shape[:-1])
    precomp_ratio = float(grid_phot[node][0] / grid_phot[node][1])

    # Exact runtime SED at the same node parameters. The photometry precompute
    # is 5-D (tau, p, q, oa, cos_inc) with the radius ratio pinned to R=20
    # (#772), so map the node over the precompute's own axes and pass R=20 to
    # the exact (R-aware) runtime path.
    axes = [np.asarray(a) for a in precomp["axes"]]
    pt = {
        "agn_tau_skirtor": float(axes[0][node[0]]),
        "agn_p_skirtor": float(axes[1][node[1]]),
        "agn_q_skirtor": float(axes[2][node[2]]),
        "agn_oa_skirtor": float(axes[3][node[3]]),
        "agn_radius_ratio": 20.0,
        "agn_cos_inc": float(axes[4][node[4]]),
    }
    fn = skirtor.create_skirtor_from_grid(grid_path)
    wave = np.logspace(np.log10(91.0), np.log10(1.0e7), 4000)
    lnu = np.asarray(fn(jnp.asarray(wave), agn_log_lbol=0.0, agn_torus_frac=1.0, **pt))
    assert np.isfinite(lnu).all(), "runtime SKIRTOR L_ν is non-finite"
    lnu_mid = float(np.interp(mid_centre, wave, lnu))
    lnu_far = float(np.interp(far_centre, wave, lnu))
    exact_ratio = lnu_mid / lnu_far

    # The two paths should agree on the spectral shape (ratio) to ~15 %.
    # The old λ²-off precompute would miss by ~(60/12)² = 25×.
    assert exact_ratio > 0 and precomp_ratio > 0
    rel = abs(precomp_ratio - exact_ratio) / exact_ratio
    assert rel < 0.15, (
        f"precompute/runtime mid-IR↔far-IR flux ratio mismatch: "
        f"precompute {precomp_ratio:.3e} vs exact {exact_ratio:.3e} "
        f"(rel {rel:.2%}) — WavePrecomp path lost the #459 L_λ→L_ν conversion"
    )
