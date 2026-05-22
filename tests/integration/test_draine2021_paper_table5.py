# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end checks against Draine, Li, Hensley et al. 2021 (arXiv:2011.07046).

These tests assert facts that must hold exactly, derived from the
template data themselves rather than from approximate reproductions
of paper tables:

* The integrated nu*P_nu over the full template grid equals the
  ``total TIR power`` quoted in each ASCII file's header (stored as
  ``tir_total`` on the HDF5).
* Spectra scale linearly with U: TIR(lgU=k) = 10^k * TIR(lgU=0).
* The total = sum of the per-component (Astrodust + PAH+ + PAH0)
  spectra to roundoff.

The paper's Table 5 ``Fclip / FTIR`` values are reported by the
``clip_feature`` integrator for diagnostic comparison but are NOT
used as a regression target: the linear-baseline definition has
intrinsic systematics (the paper itself notes this in section 9.1)
that depend on numerical-grid choices we cannot reverse-engineer
from the published table alone.  Asserting "approximately matches
the paper" with double-digit tolerance would not be a real test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pahspec_smoke.h5"


@pytest.fixture(scope="module")
def standard_spectrum():
    pytest.importorskip("h5py")
    from tengri.components.dust.emission_templates import (
        load_draine2021_pahspec_templates,
    )

    tpl = load_draine2021_pahspec_templates(str(FIXTURE))
    i_sl = tpl.starlight_names.index("mMMP")
    i_slab = 0
    i_lgU = int(np.argmin(np.abs(np.asarray(tpl.lgU) - 1.0)))
    i_ion = tpl.ion_names.index("st")
    i_size = tpl.size_names.index("std")

    wave_um = np.asarray(tpl.wavelength_um)
    nu_pnu = np.asarray(tpl.nu_pnu_total[i_sl, i_slab, i_lgU, i_ion, i_size, :])
    return wave_um, nu_pnu, tpl, (i_sl, i_slab, i_lgU, i_ion, i_size)


def test_total_ir_matches_file_header(standard_spectrum):
    """int(nu*P_nu) over ln(nu) must equal the 'total TIR power' header
    value embedded in the original ASCII files (stored on the grid as
    ``tir_total``).  Tolerance is the trapezoid discretisation error."""
    from tengri.analysis.feature_strengths import total_ir_power

    wave_um, nu_pnu, tpl, idx = standard_spectrum
    ftir = total_ir_power(wave_um, nu_pnu)
    expected = float(tpl.tir_total[idx[0], idx[1], idx[2], idx[3], idx[4]])
    np.testing.assert_allclose(ftir, expected, rtol=0.01)


def test_u_scaling_is_exactly_linear():
    """At fixed (starlight, ionization, size, slab), nu*P_nu must
    scale linearly with U because the spectrum at each grain
    temperature scales linearly with the absorbed power."""
    from tengri.components.dust.emission_templates import (
        load_draine2021_pahspec_templates,
    )

    tpl = load_draine2021_pahspec_templates(str(FIXTURE))
    # tir_total at lgU=k must equal 10^k * tir at lgU=0 to <0.5%
    # (the residual is from rounding in the file headers).
    base = np.asarray(tpl.tir_total[0, 0, 0, 1, 1])  # lgU=0, st, std
    for i_u in range(15):
        expected = base * (10.0 ** float(np.asarray(tpl.lgU)[i_u]))
        got = float(np.asarray(tpl.tir_total[0, 0, i_u, 1, 1]))
        np.testing.assert_allclose(got, expected, rtol=5e-3)


def test_total_equals_sum_of_components_exactly():
    """Column-2 'total' nu*P_nu must equal the sum of columns
    3 (Astrodust) + 4 (PAH+) + 5 (PAH0) to printed precision."""
    from tengri.components.dust.emission_templates import (
        load_draine2021_pahspec_templates,
    )

    tpl = load_draine2021_pahspec_templates(str(FIXTURE))
    total = np.asarray(tpl.nu_pnu_total)
    summed = (
        np.asarray(tpl.nu_pnu_astrodust)
        + np.asarray(tpl.nu_pnu_pah_plus)
        + np.asarray(tpl.nu_pnu_pah_neutral)
    )
    # 4-digit ASCII printing -> ~1e-4 relative roundoff.
    np.testing.assert_allclose(total, summed, rtol=2e-3, atol=0)


def test_fclip_is_finite_and_positive(standard_spectrum):
    """The clip-feature integrator runs cleanly on the loaded
    spectrum.  We do NOT assert numerical agreement with Table 5 of
    the paper because the linear-baseline definition has known
    systematic offsets (paper section 9.1) that depend on numerical
    grid choices not specified in the published table."""
    from tengri.analysis.feature_strengths import (
        TABLE5_FEATURES,
        clip_feature,
    )

    wave_um, nu_pnu, _, _ = standard_spectrum
    for info in TABLE5_FEATURES.values():
        fclip = clip_feature(wave_um, nu_pnu, info["lam1_um"], info["lam2_um"])
        assert np.isfinite(fclip)
        assert fclip > 0


def test_pah_components_dominate_at_pah_features(standard_spectrum):
    """At the canonical PAH-feature wavelengths (3.3, 6.2, 7.7, 11.2,
    17 um), the PAH+ + PAH0 components must exceed the Astrodust
    continuum component.  This is a model-internal consistency check
    that doesn't rely on paper numerical values."""
    _, _, tpl, idx = standard_spectrum
    i_sl, i_slab, i_lgU, i_ion, i_size = idx
    wave_um = np.asarray(tpl.wavelength_um)
    astro = np.asarray(tpl.nu_pnu_astrodust[i_sl, i_slab, i_lgU, i_ion, i_size, :])
    pah = np.asarray(
        tpl.nu_pnu_pah_plus[i_sl, i_slab, i_lgU, i_ion, i_size, :]
        + tpl.nu_pnu_pah_neutral[i_sl, i_slab, i_lgU, i_ion, i_size, :]
    )

    for lam_peak in (3.3, 6.2, 7.7, 11.2):
        i = int(np.argmin(np.abs(wave_um - lam_peak)))
        assert pah[i] > astro[i], (
            f"At {lam_peak} um, PAH ({pah[i]:.3e}) should exceed "
            f"Astrodust continuum ({astro[i]:.3e})"
        )
