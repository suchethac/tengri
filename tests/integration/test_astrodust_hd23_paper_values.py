"""Numerical regression tests against H&D 2022/2023 published values.

These tests reproduce the computations performed in the canonical
``brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb``
notebook.  Anchoring against documented numbers (paper text + the
notebook's own arithmetic) guards against silent regressions in
either our FITS parser, unit conversions, or storage layout.

Reference
---------
* Notebook: brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb
* Paper:    Hensley & Draine 2023, ApJ 948, 55, arXiv:2208.12365.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


HDF5 = Path("data/astrodust_templates.h5")


@pytest.fixture(scope="module")
def hdf5():
    if not HDF5.is_file():
        pytest.skip(
            f"Astrodust HDF5 not built at {HDF5}; "
            "run scripts/build_astrodust_hdf5.py --download"
        )
    with h5py.File(HDF5, "r") as f:
        size_dist = np.asarray(f["size_distribution"])
        L_nu_total = np.asarray(f["L_nu_total"])
        L_nu_astrodust = np.asarray(f["L_nu_astrodust"])
        L_nu_pah = np.asarray(f["L_nu_pah"])
        L_nu_spdust_total = np.asarray(f["L_nu_spdust_total"])
        wavelength_um = np.asarray(f["wavelength_um"])
        lgU = np.asarray(f["lgU"])
        ext = np.asarray(f["extinction"])
        sca = np.asarray(f["scattering"])
    return dict(
        size_dist=size_dist,
        L_nu_total=L_nu_total,
        L_nu_astrodust=L_nu_astrodust,
        L_nu_pah=L_nu_pah,
        L_nu_spdust_total=L_nu_spdust_total,
        wave=wavelength_um,
        lgU=lgU,
        ext=ext,
        sca=sca,
    )


# ─────────────────────────────────────────────────────────────────────
# Size-distribution-derived volumes and masses
# (Notebook section "We can also compute the total grain volume and
# mass, both per H atom.")
# ─────────────────────────────────────────────────────────────────────


def test_grain_volumes_per_H(hdf5):
    """V_Ad, V_PAH per H atom integrated over the H&D 2022 fiducial
    size distribution.

    Reference values are computed by the upstream notebook itself
    (see model_file_tutorial.ipynb): the published HDU 1 ``dn/n_H``
    column already includes the ``da`` integration factor, so a plain
    sum reproduces the integral.
    """
    sd = hdf5["size_dist"]
    rad_um = sd[:, 0]
    dn_Ad = sd[:, 1]
    dn_PAH = sd[:, 2]
    V_Ad = (4.0 / 3.0) * np.pi * np.sum((rad_um * 1.0e-4) ** 3 * dn_Ad)
    V_PAH = (4.0 / 3.0) * np.pi * np.sum((rad_um * 1.0e-4) ** 3 * dn_PAH)

    # Notebook-computed reference values (cm^3 per H atom).
    np.testing.assert_allclose(V_Ad, 3.92e-27, rtol=1e-2)
    np.testing.assert_allclose(V_PAH, 5.51e-28, rtol=1e-2)


def test_dust_mass_per_H(hdf5):
    """M_Ad/M_H = 0.00642 and M_PAH/M_H = 0.000659 from H&D 2023
    (matches the values quoted in the dataverse README and stored in
    the HDF5 attrs)."""
    sd = hdf5["size_dist"]
    rad_um = sd[:, 0]
    dn_Ad = sd[:, 1]
    dn_PAH = sd[:, 2]

    rho_Ad = 2.74        # g/cm^3, porosity 0.2 (Hensley & Draine 2023, §2)
    rho_PAH = 2.0
    mp = 1.6726218e-24

    V_Ad = (4.0 / 3.0) * np.pi * np.sum((rad_um * 1.0e-4) ** 3 * dn_Ad)
    V_PAH = (4.0 / 3.0) * np.pi * np.sum((rad_um * 1.0e-4) ** 3 * dn_PAH)
    M_Ad_over_M_H = rho_Ad * V_Ad / mp
    M_PAH_over_M_H = rho_PAH * V_PAH / mp
    Md_over_MH = M_Ad_over_M_H + M_PAH_over_M_H

    np.testing.assert_allclose(M_Ad_over_M_H, 0.00642, rtol=2e-3)
    np.testing.assert_allclose(M_PAH_over_M_H, 0.000659, rtol=5e-3)
    # Notebook + paper: Sigma_d / Sigma_H = 0.0071.
    np.testing.assert_allclose(Md_over_MH, 0.0071, rtol=2e-2)


def test_hdf5_dust_mass_attrs_match_computation(hdf5):
    """The dust-mass-to-H constants stored as HDF5 attrs (used by
    downstream code for L/M conversions) must match the values you'd
    re-derive from the size distribution."""
    sd = hdf5["size_dist"]
    rad_um = sd[:, 0]
    rho_Ad, rho_PAH, mp = 2.74, 2.0, 1.6726218e-24
    V_Ad = (4.0 / 3.0) * np.pi * np.sum((rad_um * 1.0e-4) ** 3 * sd[:, 1])
    V_PAH = (4.0 / 3.0) * np.pi * np.sum((rad_um * 1.0e-4) ** 3 * sd[:, 2])
    M_Ad_calc = rho_Ad * V_Ad / mp
    M_PAH_calc = rho_PAH * V_PAH / mp

    with h5py.File(HDF5, "r") as f:
        attrs_Ad = float(f.attrs["M_Ad_over_M_H"])
        attrs_PAH = float(f.attrs["M_PAH_over_M_H"])
    np.testing.assert_allclose(attrs_Ad, M_Ad_calc, rtol=2e-3)
    np.testing.assert_allclose(attrs_PAH, M_PAH_calc, rtol=5e-3)


# ─────────────────────────────────────────────────────────────────────
# Emission grid linearity in U and feature peak locations
# ─────────────────────────────────────────────────────────────────────


def test_emission_peak_at_fiducial_U(hdf5):
    r"""At the H&D fiducial :math:`\log_{10}U = 0.2` (U=1.585), the
    :math:`\lambda I_\lambda` spectrum should peak in the 7.7 μm PAH
    complex — that's what defines this as a "PAH-feature-dominated
    sightline" in the paper."""
    L_nu = hdf5["L_nu_total"]
    wave = hdf5["wave"]
    lgU = hdf5["lgU"]

    i = int(np.argmin(np.abs(lgU - 0.2)))
    c_cgs = 2.99792458e10
    lam_cm = wave * 1.0e-4
    li = L_nu[i] * c_cgs / (4.0 * np.pi * lam_cm)
    peak_lambda = wave[np.argmax(li)]
    # The 7.7 μm PAH complex spans 6.9 to 9.7 μm in Draine's clip.
    assert 6.9 <= peak_lambda <= 9.7, (
        f"peak at {peak_lambda} μm; expected within the 7.7 μm complex"
    )


def test_bolometric_emission_scales_linearly_with_U(hdf5):
    r"""Energy conservation: the *bolometric* integral
    :math:`\int L_\nu \, d\nu` per H atom must scale exactly linearly
    with :math:`U` because dust re-emits all the absorbed power.

    At fixed wavelength in the FIR the spectrum is NOT linear in U
    because steady-state grain temperature scales like
    :math:`T \propto U^{1/(4+\beta)}`, shifting the SED peak; the
    notebook explicitly shows this departure by plotting
    :math:`\lambda I_\lambda / U` (notebook fig 8).  Only the
    bolometric integral is exactly linear.
    """
    L_nu = hdf5["L_nu_total"]
    wave = hdf5["wave"]
    lgU = hdf5["lgU"]
    c_cgs = 2.99792458e10
    nu_hz = c_cgs / (wave * 1.0e-4)
    order = np.argsort(nu_hz)

    def integrate(L_nu_row):
        return np.trapezoid(L_nu_row[order], nu_hz[order])

    # Compare bolometric values across two-decade U range.
    i_lo = int(np.argmin(np.abs(lgU - (-1.0))))
    i_mid = int(np.argmin(np.abs(lgU - 0.0)))
    i_hi = int(np.argmin(np.abs(lgU - 1.0)))

    bol_lo = integrate(L_nu[i_lo])
    bol_mid = integrate(L_nu[i_mid])
    bol_hi = integrate(L_nu[i_hi])

    np.testing.assert_allclose(bol_mid / bol_lo, 10.0, rtol=5e-3)
    np.testing.assert_allclose(bol_hi / bol_mid, 10.0, rtol=5e-3)


def test_FIR_peak_wavelength_decreases_with_U(hdf5):
    r"""Steady-state grain temperature scales as
    :math:`T \propto U^{1/(4+\beta)} \approx U^{0.17}`, so the FIR
    peak wavelength scales as :math:`\lambda_{\rm peak} \propto
    1/T \propto U^{-0.17}`.  The peak should move from ~120 μm at
    U=1 down to ~50 μm at U=10 in :math:`\nu I_\nu`.
    """
    L_nu = hdf5["L_nu_total"]
    wave = hdf5["wave"]
    lgU = hdf5["lgU"]

    # nu*L_nu (per-decade-energy form) makes the FIR peak unambiguous.
    c_cgs = 2.99792458e10
    nu = c_cgs / (wave * 1.0e-4)

    fir_mask = (wave > 30.0) & (wave < 500.0)
    peaks = []
    for tg in (0.0, 1.0):
        i = int(np.argmin(np.abs(lgU - tg)))
        nuLnu = nu * L_nu[i]
        peaks.append(wave[fir_mask][np.argmax(nuLnu[fir_mask])])
    # Higher U -> warmer dust -> shorter wavelength peak.
    assert peaks[1] < peaks[0], f"FIR peaks (U=1, U=10) = {peaks}"


def test_total_equals_components_to_published_precision(hdf5):
    """The "total" column equals astrodust + PAH within float32
    rounding tolerance (atol covers underflow at the UV end where
    dust emission is essentially zero)."""
    total = hdf5["L_nu_total"]
    summed = hdf5["L_nu_astrodust"] + hdf5["L_nu_pah"]
    np.testing.assert_allclose(total, summed, rtol=2e-3, atol=1e-40)


# ─────────────────────────────────────────────────────────────────────
# Extinction normalization — constraints from the paper
# ─────────────────────────────────────────────────────────────────────


def test_extinction_at_V_band_equals_R_V_3p1_value(hdf5):
    r"""For a MW R_V = 3.1 sightline, A_V/N_H is well-measured.  The
    paper's eq. for tau_V/N_H follows from the chosen normalisation
    Lenz, Hensley & Doré (2017): ``N_H/E(B-V) = 8.8e21 cm^-2 mag^-1``
    and ``A_V/E(B-V) = 3.1`` ⇒ ``A_V/N_H = 3.5e-22 cm^2/H``.
    Optical depth :math:`\tau_V = A_V \ln(10) / 2.5` ⇒
    ``tau_V/N_H ≈ 3.2e-22 cm^2/H``.

    The HDU 2 extinction is in tau units per H atom.  Check the V
    band (5500 Å = 0.55 μm) total extinction matches that.
    """
    ext = hdf5["ext"]  # (1000, 4): wave_um, tau_Ad, tau_PAH, tau_total
    wave = ext[:, 0]
    tau_total = ext[:, 3]

    iV = int(np.argmin(np.abs(wave - 0.55)))
    tau_V_per_H = tau_total[iV]
    # Expected from MW R_V=3.1 normalisation: ~3.2e-22 cm^2/H.
    np.testing.assert_allclose(tau_V_per_H, 3.2e-22, rtol=0.20)


def test_extinction_3p4_micron_aliphatic_feature_present(hdf5):
    """Aliphatic-hydrocarbon C-H stretch feature at 3.4 μm shows up
    as a small bump in tau_PAH(λ).  Just check tau_PAH at 3.4 μm
    exceeds the smooth interpolation of its neighbours, confirming
    the feature is in our HDF5."""
    ext = hdf5["ext"]
    wave = ext[:, 0]
    tau_PAH = ext[:, 2]

    i_feat = int(np.argmin(np.abs(wave - 3.4)))
    i_side1 = int(np.argmin(np.abs(wave - 3.0)))
    i_side2 = int(np.argmin(np.abs(wave - 3.7)))
    interp = 0.5 * (tau_PAH[i_side1] + tau_PAH[i_side2])
    assert tau_PAH[i_feat] > interp, (
        f"3.4 um feature missing? tau={tau_PAH[i_feat]:.2e} vs interp={interp:.2e}"
    )


def test_silicate_features_at_9p7_and_18_microns(hdf5):
    r"""The astrodust silicate stretch and bend features are at
    9.7 μm and 18 μm respectively — landmark features of the
    Astrodust grain composition (paper, §2)."""
    ext = hdf5["ext"]
    wave = ext[:, 0]
    tau_Ad = ext[:, 1]

    for lam_feat in (9.7, 18.0):
        i_feat = int(np.argmin(np.abs(wave - lam_feat)))
        # Use ±50% wavelength as off-feature continuum points.
        i_blue = int(np.argmin(np.abs(wave - lam_feat * 0.5)))
        i_red = int(np.argmin(np.abs(wave - lam_feat * 1.5)))
        # Geometric-mean continuum interpolation.
        log_blue = np.log10(tau_Ad[i_blue])
        log_red = np.log10(tau_Ad[i_red])
        cont = 10 ** (0.5 * (log_blue + log_red))
        assert tau_Ad[i_feat] > cont, (
            f"Astrodust {lam_feat} um feature absent: "
            f"tau={tau_Ad[i_feat]:.2e} vs continuum={cont:.2e}"
        )


# ─────────────────────────────────────────────────────────────────────
# Spinning dust microwave peak frequency
# ─────────────────────────────────────────────────────────────────────


def test_spinning_dust_peaks_in_AME_band(hdf5):
    """Anomalous Microwave Emission peaks in 20-30 GHz (Planck Int.
    XV/XXII).  Check the spinning-dust spectrum's peak in nu*I_nu
    (the natural unit for AME) lies in this range."""
    spd = hdf5["L_nu_spdust_total"]
    wave = hdf5["wave"]
    c_cgs = 2.99792458e10

    nu_hz = c_cgs / (wave * 1.0e-4)
    nu_Inu = nu_hz * spd  # ν L_ν per H per (4π sr)
    # Restrict to microwave for the peak search (1-200 GHz).
    mw = (nu_hz > 1e9) & (nu_hz < 2e11)
    if not np.any(mw & (spd > 0)):
        pytest.skip("spinning dust empty in microwave window")
    peak_nu = nu_hz[mw][np.argmax(nu_Inu[mw])]
    peak_ghz = peak_nu * 1.0e-9
    assert 15.0 <= peak_ghz <= 50.0, (
        f"AME peak at {peak_ghz:.1f} GHz; expected 20-30 GHz range"
    )
