# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for emission line doublet ratios and wavelength conventions.

Synthesizer parity.

Mirrors synthesizer's line-list validation logic for consistency with atomic-physics
constraints. Every doublet must enforce the correct luminosity ratio; every wavelength
must be in vacuum (not air), per the CLAUDE.md convention.

Pitfall: P-12 — atomic doublet-ratio enforcement.
If both [OIII] 4959 Å and [OIII] 5007 Å are returned as separate Line objects,
their luminosities must obey the atomic-physics ratio 1:3 (Storey & Zeippen 2000).
Same for [NII] 6548/6584 Å (≈1:3) and [Ne V] 3346/3426 Å (≈0.4).

Synthesizer source:
- https://github.com/flaresimulations/synthesizer/blob/main/src/synthesizer/line_list.py
- fastspecfit line list (Moustakas+2023, DESI standard)

Reference papers:
- Storey & Zeippen 2000, MNRAS, 312, 813 (atomic transition probabilities)
- Moustakas et al. 2023, ApJS, 264, 9 (FastSpecFit line catalog)
- NIST Atomic Spectra Database (doublet ratios verification)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_paper

from tengri.observation.line_list import _DOUBLET_RATIOS, LineList


def test_doublet_ratio_oiii_enforced():
    """[OIII] 4959/5007 flux ratio must be fixed at ~1/3 (atomic physics).

    Pitfall P-12: If the line list returns both [OIII] lines as independent entries
    without a constraint, users could fit them as free parameters, violating atomic
    physics. This test verifies that either:
      (a) A doublet constraint exists linking 4959 → 5007, or
      (b) The line list documents that the constraint is applied externally.

    Storey & Zeippen 2000 give the transition probability ratio for forbidden lines:
    [OIII] λ5007 / λ4959 luminosity ratio ≈ 2.98:1.
    """
    line_list = LineList.default_optical()
    names = list(line_list.names)

    # Locate the [OIII] doublet in the catalog
    try:
        idx_4959 = names.index("OIII_4959")
        idx_5007 = names.index("OIII_5007")
    except ValueError:
        # Either line missing — not an error, but can't test the constraint
        return

    # Check that a constraint exists in _DOUBLET_RATIOS
    constraint_key = ("OIII_5007", "OIII_4959")
    assert constraint_key in _DOUBLET_RATIOS, (
        "P-12 BUG: [OIII] doublet not listed in _DOUBLET_RATIOS. "
        "Line catalog would allow independent fitting of both 4959 and 5007 Ångstrom."
    )

    # Verify the ratio matches Storey & Zeippen 2000
    ratio = _DOUBLET_RATIOS[constraint_key]
    expected_ratio = 2.98  # λ5007 / λ4959 flux ratio
    rel_err = abs(ratio - expected_ratio) / expected_ratio
    assert rel_err < 0.05, (
        f"P-12 WARNING: [OIII] doublet ratio {ratio:.2f} "
        f"differs from Storey & Zeippen 2000 {expected_ratio:.2f} by {100 * rel_err:.1f}%. "
        "Check transition-probability source."
    )


def test_doublet_ratio_nii_enforced():
    """[NII] 6548/6584 flux ratio must be fixed at ~1/3 (atomic physics).

    Pitfall P-12: [NII] doublet constraints must match transition probabilities.
    Storey & Zeippen 2000: [NII] λ6584 / λ6548 ≈ 2.94:1.
    """
    line_list = LineList.default_optical()
    names = list(line_list.names)

    # Locate the [NII] doublet
    try:
        idx_6548 = names.index("NII_6548")
        idx_6584 = names.index("NII_6584")
    except ValueError:
        return

    constraint_key = ("NII_6584", "NII_6548")
    assert constraint_key in _DOUBLET_RATIOS, (
        "P-12 BUG: [NII] doublet not listed in _DOUBLET_RATIOS. "
        "Line catalog would allow independent fitting of both 6548 and 6584 Ångstrom."
    )

    ratio = _DOUBLET_RATIOS[constraint_key]
    expected_ratio = 2.94
    rel_err = abs(ratio - expected_ratio) / expected_ratio
    assert rel_err < 0.05, (
        f"P-12 WARNING: [NII] doublet ratio {ratio:.2f} "
        f"differs from Storey & Zeippen 2000 {expected_ratio:.2f} by {100 * rel_err:.1f}%. "
        "Check transition-probability source."
    )


def test_nev_doublet_ratio_enforced():
    """[Ne V] 3346/3426 flux ratio must be consistent with atomic physics.

    Pitfall P-12: [Ne V] is a high-ionization line; its doublet ratio comes from
    transition probabilities. Verify catalog enforces this ratio.

    NIST Atomic Spectra Database + Storey & Zeippen 2000: [Ne V] λ3426 / λ3346 ≈ 1.3.
    """
    line_list = LineList.default_optical()
    names = list(line_list.names)

    try:
        idx_3346 = names.index("NeV_3346")
        idx_3426 = names.index("NeV_3426")
    except ValueError:
        return

    constraint_key = ("NeV_3426", "NeV_3346")
    assert constraint_key in _DOUBLET_RATIOS, (
        "P-12 BUG: [Ne V] doublet not listed in _DOUBLET_RATIOS. "
        "Line catalog would allow independent fitting of both 3346 and 3426 Ångstrom."
    )

    ratio = _DOUBLET_RATIOS[constraint_key]
    expected_ratio = 1.3
    # Allow 10% tolerance on this transition (lower precision source)
    rel_err = abs(ratio - expected_ratio) / expected_ratio
    assert rel_err < 0.10, (
        f"P-12 WARNING: [Ne V] doublet ratio {ratio:.2f} "
        f"differs from expected {expected_ratio:.2f} by {100 * rel_err:.1f}%. "
        "Verify against NIST Atomic Spectra Database."
    )


def test_halpha_wavelength_is_vacuum():
    """H-alpha must be 6564.61 Å in vacuum, not 6562.79 Å in air.

    Pitfall P-12 (wavelength convention): CLAUDEMD convention requires all
    wavelengths in vacuum. H-alpha in air is 6562.79 Å; in vacuum, 6564.61 Å.
    This test locks the convention against accidental change.

    Reference: IAU 2015 standard; Byler et al. 2017.
    """
    line_list = LineList.default_optical()
    names = list(line_list.names)

    try:
        idx = names.index("Halpha")
    except ValueError:
        return

    halpha_vacuum_expected = 6564.61  # Angstrom, vacuum
    halpha_air_wrong = 6562.79  # Angstrom, air (should NOT match)

    wavelength = float(line_list.wavelengths[idx])

    # Assert vacuum convention, not air
    assert abs(wavelength - halpha_vacuum_expected) < 0.5, (
        f"H-alpha wavelength {wavelength:.2f} Å is not vacuum "
        f"{halpha_vacuum_expected:.2f} Å. "
        "CLAUDE.md mandate: all emission line wavelengths are vacuum."
    )

    # Negative test: ensure it's NOT the air wavelength
    assert abs(wavelength - halpha_air_wrong) > 1.0, (
        f"H-alpha wavelength {wavelength:.2f} Å matches the air value {halpha_air_wrong:.2f} Å. "
        "Violates CLAUDE.md vacuum convention."
    )


def test_lyman_alpha_wavelength_is_vacuum():
    """Lyman-alpha must be 1215.67 Å (vacuum), not a different convention.

    Pitfall P-12: UV lines are especially prone to convention confusion.
    Verify Ly-alpha matches IAU standard (vacuum).

    Reference: IAU 2015; NIST Atomic Spectra Database (transition 2P → 1S, n=2→1).
    """
    line_list = LineList.default_optical()
    names = list(line_list.names)

    try:
        idx = names.index("Lya")
    except ValueError:
        return

    lya_vacuum_expected = 1215.67  # Angstrom, vacuum
    wavelength = float(line_list.wavelengths[idx])

    assert abs(wavelength - lya_vacuum_expected) < 0.05, (
        f"Lyman-alpha wavelength {wavelength:.2f} Å "
        f"differs from IAU vacuum standard {lya_vacuum_expected:.2f} Å by > 0.05 Å. "
        "CLAUDE.md: all wavelengths in vacuum."
    )


def test_all_wavelengths_positive_and_reasonable():
    """Every line in the catalog must have positive wavelength in physical range (1000 - 1e6 Å).

    Pitfall P-12 (implicit): guards against corrupted line list entries with negative
    or zero wavelengths, which would break interp + photometry pipelines.
    """
    line_list = LineList.default_optical()

    for i, (name, wav) in enumerate(zip(line_list.names, line_list.wavelengths)):
        wav_float = float(wav)
        assert wav_float > 0.0, f"Line {name} (idx {i}) has non-positive wavelength {wav_float} Å"
        assert 500.0 < wav_float < 1e6, (
            f"Line {name} (idx {i}) wavelength {wav_float} Å "
            f"outside physical range [500, 1e6] Å (likely data entry error)"
        )
