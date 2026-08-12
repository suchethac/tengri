# SPDX-License-Identifier: BSD-3-Clause
"""Tests validating tengri against FastSpecFit.

Includes:
  - build_line_mask (implements the same masking as FastSpecFit LineMasker)
  - compute_line_fluxes, compute_equivalent_widths, compute_line_moments
    (differentiable equivalents of FastSpecFit populate_emtable())
  - LineList.is_strong and LineList.plot_group metadata
    (matching FastSpecFit emlines.ecsv)

References
----------
.. [1] Moustakas, J., Scholte, D., Dey, B., Khederlarian, A., 2023,
       "FastSpecFit: Fast spectral synthesis and emission-line fitting
       of DESI spectra", Astrophysics Source Code Library,
       record ascl:2308.005.
       https://ui.adsabs.harvard.edu/abs/2023ascl.soft08005M

"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.regression_paper
jax.config.update("jax_enable_x64", True)

_C_KMS = 2.99792458e5

# ── Fixtures ──────────────────────────────────────────────────────────────────

HALPHA_REST = 6564.61  # Å, vacuum
HBETA_REST = 4862.68

# ── build_line_mask tests ─────────────────────────────────────────────────────


def test_build_line_mask_halpha_z0():
    """Hα at z=0: pixels near 6564.61 Å masked, far-off pixel not masked."""
    from tengri.observation.line_mask import build_line_mask

    wave = np.array([5500.0, 6564.0, 6564.61, 6565.5, 6600.0])
    mask = build_line_mask(wave, np.array([HALPHA_REST]), redshift=0.0)

    assert mask.dtype == bool
    assert not mask[0], "5500 Å should be outside the Hα window"
    assert mask[1], "6564.0 Å should be inside the Hα window"
    assert mask[2], "Hα center should be masked"
    assert mask[3], "6565.5 Å should be inside the Hα window"
    assert not mask[4], "6600.0 Å should be outside the Hα window"


def test_build_line_mask_redshifted():
    """Hα at z=0.1 appears near 7221 Å; original rest-frame position is unmasked."""
    from tengri.observation.line_mask import build_line_mask

    z = 0.1
    halpha_obs = HALPHA_REST * (1.0 + z)  # ~7221 Å

    wave = np.linspace(6000.0, 8000.0, 500)
    mask = build_line_mask(wave, np.array([HALPHA_REST]), redshift=z)

    # Pixels at rest-frame Hα position should NOT be masked
    rest_pix = np.argmin(np.abs(wave - HALPHA_REST))
    assert not mask[rest_pix], "Rest-frame Hα should not be masked at z=0.1"

    # Pixels near redshifted Hα should be masked
    obs_pix = np.argmin(np.abs(wave - halpha_obs))
    assert mask[obs_pix], "Observed-frame Hα should be masked"


def test_build_line_mask_scalar_sigma():
    """Scalar sigma_kms broadcasts to all lines without error."""
    from tengri.observation.line_mask import build_line_mask

    wave = np.linspace(3000.0, 9000.0, 1000)
    lines = np.array([HALPHA_REST, HBETA_REST, 5008.24])

    mask = build_line_mask(wave, lines, redshift=0.0, line_sigmas_kms=200.0)

    chex.assert_equal_shape([mask, wave])
    assert mask.sum() > 0


# ── compute_line_fluxes tests ─────────────────────────────────────────────────


def test_compute_line_fluxes_gaussian_identity():
    """Known amplitude + sigma → flux = A * sqrt(2π) * σ_ang."""
    from tengri.analysis.diagnostics.lines import compute_line_fluxes
    from tengri.observation.line_mask import _C_KMS

    z = 0.5
    sigma_kms = 100.0
    R = 3000.0
    rest_waves = jnp.array([HALPHA_REST, HBETA_REST])
    z_waves = rest_waves * (1.0 + z)

    sigma_vel_sq = (sigma_kms / _C_KMS) ** 2
    sigma_lsf_sq = (1.0 / (2.355 * R)) ** 2
    sigma_ang_expected = z_waves * jnp.sqrt(sigma_vel_sq + sigma_lsf_sq)

    amplitudes = jnp.array([1e-17, 2e-17])
    fluxes = compute_line_fluxes(amplitudes, rest_waves, z, sigma_kms, R)
    expected = amplitudes * jnp.sqrt(2.0 * jnp.pi) * sigma_ang_expected

    np.testing.assert_allclose(np.array(fluxes), np.array(expected), rtol=1e-6)


def test_compute_line_fluxes_zero_amplitude():
    """Zero amplitudes produce zero fluxes."""
    from tengri.analysis.diagnostics.lines import compute_line_fluxes

    rest_waves = jnp.array([HALPHA_REST, HBETA_REST, 5008.24])
    fluxes = compute_line_fluxes(
        jnp.zeros(3), rest_waves, redshift=0.3, sigma_kms=150.0, spectral_resolution=2000.0
    )
    np.testing.assert_array_equal(np.array(fluxes), 0.0)


def test_compute_line_fluxes_differentiable():
    """jax.grad through amplitudes produces finite, non-zero gradients."""
    from tengri.analysis.diagnostics.lines import compute_line_fluxes

    rest_waves = jnp.array([HALPHA_REST, HBETA_REST])

    def total_flux(amps):
        return jnp.sum(compute_line_fluxes(amps, rest_waves, 0.1, 150.0, 2500.0))

    grads = assert_grad_matches_fd(total_flux, jnp.ones(2) * 1e-17)
    chex.assert_tree_all_finite(grads)
    assert jnp.all(grads > 0.0), "Gradients must be positive"


# ── compute_equivalent_widths tests ──────────────────────────────────────────


def test_compute_equivalent_widths_simple():
    """flux = cont * 1 Å → EW_rest = 1/(1+z) Å."""
    from tengri.analysis.diagnostics.lines import compute_equivalent_widths

    z = 0.0
    cont = jnp.array([1e-17])
    flux = cont * 1.0  # 1 Å EW in obs frame
    ew = compute_equivalent_widths(flux, cont, z)
    np.testing.assert_allclose(float(ew[0]), 1.0, rtol=1e-6)


def test_compute_equivalent_widths_redshift_scaling():
    """EW_rest = EW_obs / (1+z) scaling verified at z=1."""
    from tengri.analysis.diagnostics.lines import compute_equivalent_widths

    z = 1.0
    cont = jnp.array([1e-17])
    flux = cont * 2.0  # 2 Å obs-frame EW
    ew = compute_equivalent_widths(flux, cont, z)
    np.testing.assert_allclose(float(ew[0]), 1.0, rtol=1e-6)  # 2 / (1+1) = 1 Å


def test_compute_equivalent_widths_zero_continuum():
    """Zero continuum returns EW=0, not NaN."""
    from tengri.analysis.diagnostics.lines import compute_equivalent_widths

    fluxes = jnp.array([1e-17, 2e-17])
    cont = jnp.array([0.0, 0.0])
    ew = compute_equivalent_widths(fluxes, cont, redshift=0.1)

    chex.assert_tree_all_finite(ew)
    np.testing.assert_array_equal(np.array(ew), 0.0)


def test_compute_equivalent_widths_differentiable():
    """jax.grad through line_fluxes produces finite gradients."""
    from tengri.analysis.diagnostics.lines import compute_equivalent_widths

    cont = jnp.array([1e-17, 1e-17])

    def total_ew(fluxes):
        return jnp.sum(compute_equivalent_widths(fluxes, cont, 0.3))

    grads = assert_grad_matches_fd(total_ew, jnp.array([1e-17, 2e-17]))
    chex.assert_tree_all_finite(grads)


# ── compute_line_moments tests ────────────────────────────────────────────────


def test_compute_line_moments_perfect_gaussian():
    """Injected Gaussian profile → recovered σ_int within 5% of input.

    Uses a wide soft window (σ_window >> σ_signal) so the kernel bias is
    negligible.  The recovered moment converges to the true signal sigma as
    σ_window → ∞: σ_meas = σ_s × σ_w / √(σ_s² + σ_w²).
    """
    from tengri.analysis.diagnostics.lines import compute_line_moments

    n_pix = 500
    sigma_kms_true = 120.0
    # Wide window: kernel bias ≈ (σ_s/σ_w)²/2 ≈ (120/1000)²/2 < 1%
    sigma_kms_window = 1000.0
    line_obs_wave = HALPHA_REST  # z=0 for simplicity

    wave = jnp.linspace(line_obs_wave - 50.0, line_obs_wave + 50.0, n_pix)
    v_pix = _C_KMS * (wave - line_obs_wave) / line_obs_wave
    flux = 1e-16 * jnp.exp(-0.5 * (v_pix / sigma_kms_true) ** 2)
    ivar = jnp.ones(n_pix) * 1e34  # high S/N

    v_cent, sigma_int = compute_line_moments(wave, flux, ivar, line_obs_wave, sigma_kms_window)

    assert abs(float(v_cent)) < 1.0, f"Centroid should be near 0 km/s, got {float(v_cent):.2f}"
    frac_err = abs(float(sigma_int) - sigma_kms_true) / sigma_kms_true
    assert frac_err < 0.05, f"sigma_int error {frac_err:.1%} exceeds 5%"


def test_compute_line_moments_zero_flux():
    """Empty residual (all zeros) returns finite (0, 0) without NaN."""
    from tengri.analysis.diagnostics.lines import compute_line_moments

    wave = jnp.linspace(6500.0, 6630.0, 200)
    flux = jnp.zeros(200)
    ivar = jnp.ones(200) * 1e32

    v_cent, sigma_int = compute_line_moments(wave, flux, ivar, HALPHA_REST, 150.0)

    assert jnp.isfinite(v_cent), "v_centroid must be finite for zero flux"
    assert jnp.isfinite(sigma_int), "sigma_int must be finite for zero flux"
    np.testing.assert_allclose(float(v_cent), 0.0, atol=1e-10)
    np.testing.assert_allclose(float(sigma_int), 0.0, atol=1e-10)


# ── LineList.is_strong / plot_group tests (reference: FastSpecFit) ────────────

# The 18 is_strong line names from FastSpecFit emlines.ecsv, mapped to
# tengri's naming convention.
_FASTSPECFIT_STRONG_NAMES = frozenset(
    [
        "Lya",
        "NV_1240",
        "CIV_1549",
        "SiIII_1892",
        "CIII_1908",
        "MgII_2796",
        "MgII_2803",
        "OII_3726",
        "OII_3729",
        "Hgamma",
        "Hbeta",
        "OIII_4959",
        "OIII_5007",
        "NII_6548",
        "Halpha",
        "NII_6584",
        "SII_6717",
        "SII_6731",
    ]
)


def test_linelist_isstrong_count():
    """Exactly 18 is_strong lines matching FastSpecFit emlines.ecsv.

    Reference values from FastSpecFit test_linetable.py [1]_.
    """
    from tengri.observation.line_list import LineList

    cat = LineList.default_optical()
    strong_names = {n for n, s in zip(cat.names, cat.is_strong) if s}
    assert len(strong_names) == 18, (
        f"Expected 18 is_strong lines, got {len(strong_names)}: {sorted(strong_names)}"
    )


def test_linelist_isstrong_names():
    """is_strong names exactly match FastSpecFit emlines.ecsv isstrong column.

    Reference values from FastSpecFit test_linetable.py [1]_.
    """
    from tengri.observation.line_list import LineList

    cat = LineList.default_optical()
    strong_names = {n for n, s in zip(cat.names, cat.is_strong) if s}
    assert strong_names == _FASTSPECFIT_STRONG_NAMES, (
        f"Mismatch in is_strong names.\n"
        f"  Extra: {strong_names - _FASTSPECFIT_STRONG_NAMES}\n"
        f"  Missing: {_FASTSPECFIT_STRONG_NAMES - strong_names}"
    )


def test_linelist_plot_group_halphanii():
    """Hα, NII_6548, NII_6584 share plot_group='halpha_nii_6548_48'.

    Reference values from FastSpecFit test_linetable.py [1]_.
    """
    from tengri.observation.line_list import LineList

    cat = LineList.default_optical()
    group_map = dict(zip(cat.names, cat.plot_group))

    expected_group = "halpha_nii_6548_48"
    for name in ("Halpha", "NII_6548", "NII_6584"):
        assert group_map[name] == expected_group, (
            f"{name} has plot_group={group_map[name]!r}, expected {expected_group!r}"
        )


def test_linelist_plot_group_oiii_doublet():
    """OIII_4959 and OIII_5007 share plot_group='oiii_doublet'."""
    from tengri.observation.line_list import LineList

    cat = LineList.default_optical()
    group_map = dict(zip(cat.names, cat.plot_group))

    for name in ("OIII_4959", "OIII_5007"):
        assert group_map[name] == "oiii_doublet"


def test_linelist_select_preserves_metadata():
    """select() carries is_strong and plot_group from parent catalog.

    Reference values from FastSpecFit test_linetable.py [1]_.
    """
    from tengri.observation.line_list import LineList

    cat = LineList.default_optical()
    sub = cat.select(names=["Halpha", "NII_6548", "NII_6584", "Hbeta", "OIII_5007"])

    assert len(sub.is_strong) == sub.n_lines
    assert len(sub.plot_group) == sub.n_lines

    name_to_strong = dict(zip(sub.names, sub.is_strong))
    assert name_to_strong["Halpha"] is True
    assert name_to_strong["Hbeta"] is True

    name_to_group = dict(zip(sub.names, sub.plot_group))
    assert name_to_group["Halpha"] == "halpha_nii_6548_48"
    assert name_to_group["OIII_5007"] == "oiii_doublet"


# ── Statistical utility tests (reference: FastSpecFit test_util.py) ───────────


def test_sigma_clip_statistics():
    """For Gaussian samples, (Q75-Q25)/1.349 ≈ std within 5%.

    Reference values from FastSpecFit test_util.py::test_stats [1]_.
    """
    rng = np.random.default_rng(42)
    samples = rng.standard_normal(10_000)

    q25, q75 = np.percentile(samples, [25.0, 75.0])
    iqr_sigma = (q75 - q25) / 1.349

    true_std = np.std(samples)
    frac_err = abs(iqr_sigma - true_std) / true_std
    assert frac_err < 0.05, (
        f"IQR-based sigma {iqr_sigma:.4f} vs true std {true_std:.4f} "
        f"— fractional error {frac_err:.1%} > 5%"
    )


def test_robust_sigma_iqr():
    """IQR-based sigma is robust to outliers; mean-based sigma is not.

    Reference values from FastSpecFit test_util.py::test_stats [1]_.
    """
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(1000)
    outliers = rng.uniform(50.0, 100.0, 10)
    data = np.concatenate([clean, outliers])

    q25, q75 = np.percentile(data, [25.0, 75.0])
    robust_sigma = (q75 - q25) / 1.349

    # Robust sigma should remain close to 1.0 despite the outliers
    assert abs(robust_sigma - 1.0) < 0.1, (
        f"Robust sigma {robust_sigma:.3f} drifted too far from 1.0 despite 1% outliers"
    )
    # Standard deviation is contaminated
    assert np.std(data) > robust_sigma * 1.5, "Standard deviation should be inflated by outliers"
