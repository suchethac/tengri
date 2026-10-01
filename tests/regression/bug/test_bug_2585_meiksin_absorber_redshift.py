# SPDX-License-Identifier: BSD-3-Clause
"""Meiksin (2006) Lyman-series terms evaluated at absorber redshift, not source redshift (#2585).

The n >= 3 terms in Meiksin 2006 must use tau_alpha evaluated at the absorber
redshift z_n, not at the source redshift z. Table 1 of the paper gives the
ratios tau_n/tau_alpha as functions of the absorber redshift (the (1+z) powers
"factor out the redshift dependence" of tau_alpha itself), so every term must
be evaluated at z_n — as the n = 2 term already is correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

from tengri.components.igm.meiksin06 import igm_transmission_meiksin06

pytestmark = pytest.mark.regression_bug


# ── Reference implementation from the issue reproducer ────────────────────
FACT = {3: 0.348, 4: 0.179, 5: 0.109, 6: 0.0722, 7: 0.0508, 8: 0.0373, 9: 0.0283}


def lam_n(n):
    """Lyman wavelength for transition n."""
    return 912.0 / (1 - 1 / n**2)


def tau_alpha(z):
    """Meiksin 2006 Eq. 2-3."""
    return 0.00211 * (1 + z) ** 3.7 if z < 4 else 0.00058 * (1 + z) ** 4.5


def tau_n_old(n, zn, z):
    """Old (wrong) tau_n implementation from the issue (evaluates n >= 3 at z, not z_n)."""
    if n == 2:
        return 0.00211 * (1 + zn) ** 3.7 if z <= 4 else 0.00058 * (1 + zn) ** 4.5
    base = tau_alpha(z)  # <-- WRONG: source redshift, not absorber redshift
    if n <= 9:
        return base * FACT[n] * (0.25 * (1 + zn)) ** (1 / 3 if (zn < 3 or n > 5) else 1 / 6)
    return tau_n_old(9, zn, z) * 720 / (n * (n * n - 1))


def tau_n_correct(n, zn, z, at_zn):
    """Correct tau_n (evaluates n >= 3 at z_n, not z)."""
    base = tau_alpha(zn) if at_zn else tau_alpha(z)
    if n == 2:
        if at_zn:
            return tau_alpha(zn)
        return 0.00211 * (1 + zn) ** 3.7 if z <= 4 else 0.00058 * (1 + zn) ** 4.5
    if n <= 9:
        exponent = 1 / 3 if (zn < 3 or n > 5) else 1 / 6
        return base * FACT[n] * (0.25 * (1 + zn)) ** exponent
    return tau_n_correct(9, zn, z, at_zn) * 720 / (n * (n * n - 1))


def tau_lines_old(lobs, z):
    """Old tau_lines (evaluates n >= 3 at source redshift)."""
    return sum(
        tau_n_old(n, lobs / lam_n(n) - 1, z) for n in range(2, 32) if 0 <= lobs / lam_n(n) - 1 < z
    )


def tau_lines_correct(lobs, z, at_zn):
    """Correct tau_lines with at_zn=True."""
    return sum(
        tau_n_correct(n, lobs / lam_n(n) - 1, z, at_zn)
        for n in range(2, 32)
        if 0 <= lobs / lam_n(n) - 1 < z
    )


def tau_ligm(lobs, z):
    """Lyman-alpha forest continuum opacity."""
    zL = lobs / 912.0 - 1
    return 0.805 * (1 + zL) ** 3 * (1 / (1 + zL) - 1 / (1 + z)) if zL < z else 0.0


def tau_lls(lobs, z, N0=0.25, beta=1.5, gam=1.5):
    """Lyman-Limit Systems opacity (Eq. 6, numerical)."""
    zL = lobs / 912.0 - 1
    if zL >= z:
        return 0.0

    def inner(zp):
        """Integrand for LLS opacity."""

        def integrand(t):
            return N0 * (beta - 1) * t ** (-beta) * (1 - np.exp(-t * ((1 + zL) / (1 + zp)) ** 3))

        return integrate.quad(integrand, 1, np.inf, limit=200)[0] * (1 + zp) ** gam

    return integrate.quad(inner, zL, z, limit=200)[0]


def transmission_correct(lobs, z):
    """Correct transmission (at_zn=True)."""
    return np.exp(-(tau_lines_correct(lobs, z, True) + tau_ligm(lobs, z) + tau_lls(lobs, z)))


# ── Cell 1a: Table 2 reference values at observed 1730 A ───────────────────
@pytest.mark.parametrize(
    "z,expected",
    [
        (1.5, 0.323241),
        (2.0, 0.150055),
        (2.5, 0.074558),
        (3.0, 0.038532),
        (3.5, 0.020407),
        (4.0, 0.010983),
        (4.5, 0.005976),
        (5.0, 0.003277),
        (5.5, 0.001807),
        (6.0, 0.001001),
        (6.5, 0.000556),
        (7.0, 0.000310),
    ],
)
def test_meiksin06_table2_at_1730_observed(z, expected):
    """Test transmission at observed 1730 Å equals Meiksin 2006 Table 2 to 3e-3."""
    T = float(igm_transmission_meiksin06(jnp.asarray([1730.0]), z)[0])
    assert abs(T - expected) < 3e-3 * abs(expected), (
        f"z={z}: tengri {T:.6f}, paper {expected:.6f}, rel_err={T / expected - 1:.2e}"
    )


# ── Cell 1d: Gradient ────────────────────────────────────────────────────
def test_meiksin06_grad_wrt_z():
    """Test that jax.grad of transmission w.r.t. z is finite at (1730 Å, z=3)."""

    def T_fn(z_val):
        return igm_transmission_meiksin06(jnp.asarray([1730.0]), z_val)[0]

    z_test = 3.0
    grad_T = jax.grad(T_fn)(z_test)
    assert jnp.isfinite(grad_T), f"Gradient is {grad_T}"


# ── Cell 1b: the series-only mechanism, independent of the paper table ────
_SERIES_RATIOS = {
    3.0: (1.171, 1.081, 1.036, 1.011),
    5.0: (2.617, 1.601, 1.220, 1.065),
    6.0: (8.182, 2.920, 1.616, 1.165),
}
_REST_LAMBDAS = (800.0, 900.0, 950.0, 1000.0)


def transmission_source_redshift(lobs, z):
    """Transmission with the n >= 3 series evaluated at the source redshift."""
    return np.exp(-(tau_lines_old(lobs, z) + tau_ligm(lobs, z) + tau_lls(lobs, z)))


@pytest.mark.parametrize(
    "z,rest_aa,expected",
    [
        (z, lam, ratio)
        for z, ratios in _SERIES_RATIOS.items()
        for lam, ratio in zip(_REST_LAMBDAS, ratios)
    ],
)
def test_meiksin06_series_ratio_to_source_redshift_reading(z, rest_aa, expected):
    """Absorber-redshift over source-redshift transmission matches the series-only ratio.

    Both sides carry the same LLS and LIGM terms, so the ratio isolates the n >= 3
    Lyman-series optical depths (Meiksin 2006, Table 1, evaluated at z_n).
    """
    lobs = rest_aa * (1.0 + z)
    t_fixed = float(igm_transmission_meiksin06(jnp.asarray([lobs]), z)[0])
    ratio = t_fixed / transmission_source_redshift(lobs, z)
    assert ratio == pytest.approx(expected, rel=1e-2), (
        f"z={z}, rest {rest_aa} A: T_fixed/T_source = {ratio:.4f}, expected {expected}"
    )


# ── Cell 2: the public path, SEDModel.build + photometry ───────────────────
_BAND_REST_AA = (850.0, 1000.0)
_PUBLIC_RTOL = 2e-2
_F32_VS_F64_RTOL = 1e-4


def _band_photometry_ratio(ssp_data, z, fold, *, x64):
    """Photometry(meiksin06) / photometry(none) in a rest 850-1000 A top-hat band.

    Returns the ratio, the filter edges and the observed-frame wavelength grid and
    intrinsic L_nu of the no-IGM model (the inputs the independent reference needs).
    """
    from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    lo, hi = (1.0 + z) * _BAND_REST_AA[0], (1.0 + z) * _BAND_REST_AA[1]
    with jax.enable_x64(x64):
        curve = FilterCurve(
            wave=jnp.linspace(lo, hi, 200), trans=jnp.ones(200), name=f"lyman_{z:g}"
        )
        observation = Observation(photometry=Photometry(filters=(curve,)))
        phot = {}
        intrinsic = None
        for igm in ("meiksin06", "none"):
            model = SEDModel.build(
                ssp_data=ssp_data,
                observation=observation,
                sfh={
                    "type": "delayed",
                    "tau_gyr": Fixed(1.0),
                    "age_gyr": Fixed(1.0),
                    "log_total_mass": Fixed(10.0),
                    "all_params": Fixed(DEFAULT),
                },
                dust_attenuation={
                    "law": "power_law",
                    "type": "two_component",
                    "tau_bc": Fixed(0.0),
                    "tau_diff": Fixed(0.0),
                    "all_params": Fixed(DEFAULT),
                },
                neb={"type": "none"},
                igm={"type": igm},
                redshift=Fixed(z),
                approx=WavePrecomp(band_integration="quadrature", n_subbands=16, igm_fold=fold),
            )
            phot[igm] = float(np.asarray(model.predict_photometry({}))[0])
            if igm == "none":
                pred = model.predict({})
                intrinsic = (np.asarray(pred.wave_obs), np.asarray(pred.obs_sed()))
    return phot["meiksin06"] / phot["none"], (lo, hi), intrinsic


def _reference_band_ratio(z, edges, intrinsic):
    """Photon-counting band average of T_meiksin06 weighted by the intrinsic L_nu."""
    wave_obs, l_nu = intrinsic
    lo, hi = edges
    fine = np.linspace(lo, hi, 20001)
    weight = np.interp(fine, wave_obs, l_nu) / fine
    trans = np.asarray(igm_transmission_meiksin06(jnp.asarray(fine), z))
    return np.trapezoid(trans * weight, fine) / np.trapezoid(weight, fine)


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
@pytest.mark.parametrize("fold", ["node", "exact"])
@pytest.mark.parametrize("z", [3.0, 5.0, 6.0])
def test_meiksin06_public_photometry_matches_band_averaged_transmission(
    ssp_data_fsps, z, fold, x64
):
    """SEDModel photometry in a Lyman-series band carries the absorber-redshift transmission.

    The fixed-redshift model with ``igm={"type": "meiksin06"}`` over the same model with
    ``igm={"type": "none"}`` must equal the transmission-weighted band average computed
    here from :func:`igm_transmission_meiksin06` on a fine grid; float32 must agree with
    float64 to 1e-4 in the ratio.
    """
    ratio64, edges, intrinsic = _band_photometry_ratio(ssp_data_fsps, z, fold, x64=True)
    expected = _reference_band_ratio(z, edges, intrinsic)
    assert ratio64 == pytest.approx(expected, rel=_PUBLIC_RTOL), (
        f"z={z}, fold={fold}: model ratio {ratio64:.5f}, band-averaged T {expected:.5f}"
    )
    if not x64:
        ratio32, _, _ = _band_photometry_ratio(ssp_data_fsps, z, fold, x64=False)
        assert ratio32 == pytest.approx(ratio64, rel=_F32_VS_F64_RTOL), (
            f"z={z}, fold={fold}: float32 {ratio32:.6f} vs float64 {ratio64:.6f}"
        )
