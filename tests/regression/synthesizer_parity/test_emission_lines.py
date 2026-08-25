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

A missing line is a failure, not a pass
---------------------------------------

Each of the five tests this file used to hold opened with

    try:
        idx = names.index("Halpha")
    except ValueError:
        return

so a line disappearing from the catalog — the exact regression a file named for
locking conventions exists to catch — made the test return early and report as
passed. Every line is present today, so the handler was dead code; it was one
rename away from being a silent pass. Absence is asserted now.

Two of the three doublet tests also bound ``idx_4959``/``idx_5007`` and never
used them: the constraint is looked up in ``_DOUBLET_RATIOS`` by name, so the
index lookup existed only as the presence check that then swallowed itself.

Coverage the table added
------------------------

``_DOUBLET_RATIOS`` holds **five** constraints. Three were tested. MgII
2796/2803 and SIII 9069/9532 had none, and a constraint with no test is what
the file exists to prevent.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_paper

from tengri.observation.line_list import _DOUBLET_RATIOS, LineList

#: (strong, weak, expected ratio, tolerance, source).
#:
#: ``None`` for the expected ratio means "assert the constraint exists and is
#: physically usable, but do not pin a number": the value in the table has no
#: sourced reference here, and CLAUDE.md forbids writing a citation from
#: memory. Both such rows are flagged below for someone who can source them.
_DOUBLETS = [
    ("OIII_5007", "OIII_4959", 2.98, 0.05, "Storey & Zeippen 2000, MNRAS 312, 813"),
    ("NII_6584", "NII_6548", 2.94, 0.05, "Storey & Zeippen 2000, MNRAS 312, 813"),
    ("NeV_3426", "NeV_3346", 1.30, 0.10, "NIST ASD + Storey & Zeippen 2000 (lower precision)"),
    # Currently 1.0000 in the table. The optically-thin Mg II resonance doublet
    # is 2:1 in favor of 2796 (statistical weights of the 2P3/2 and 2P1/2 upper
    # levels), i.e. 2803/2796 ~ 0.5; 1:1 is the optically-thick limit. Which one
    # this table means is not recorded, so the value is not pinned here.
    ("MgII_2803", "MgII_2796", None, None, "reference not sourced — see the module docstring"),
    # Currently 2.4700. That is the familiar [S III] 9532/9069 value, but this
    # file has no citation for it, so it is not pinned either.
    ("SIII_9532", "SIII_9069", None, None, "reference not sourced — see the module docstring"),
]

#: (name, vacuum wavelength [Angstrom], tolerance, the air value it must NOT be).
#: CLAUDE.md: emission line wavelengths are vacuum throughout.
_VACUUM_WAVELENGTHS = [
    ("Halpha", 6564.61, 0.5, 6562.79),
    ("Lya", 1215.67, 0.05, None),
]


@pytest.fixture(scope="module")
def catalog():
    """The default optical line list, and a name -> wavelength map."""
    line_list = LineList.default_optical()
    names = list(line_list.names)
    return line_list, names, {n: float(w) for n, w in zip(names, line_list.wavelengths)}


@pytest.mark.parametrize(
    ("strong", "weak", "expected", "tol", "source"),
    _DOUBLETS,
    ids=[d[0].split("_")[0] for d in _DOUBLETS],
)
def test_doublet_ratio_enforced(catalog, strong, weak, expected, tol, source):
    """Both lines are in the catalog and a ratio constraint links them.

    P-12: if a doublet is returned as two independent entries with no
    constraint, a fit can vary them separately and violate atomic physics.
    """
    _line_list, names, _wav = catalog

    for name in (strong, weak):
        assert name in names, (
            f"P-12: {name} is not in default_optical(), so its doublet constraint "
            f"cannot be checked. This used to `return` and report as passed."
        )

    key = (strong, weak)
    assert key in _DOUBLET_RATIOS, (
        f"P-12 BUG: {strong}/{weak} not listed in _DOUBLET_RATIOS. "
        f"Line catalog would allow independent fitting of both lines."
    )

    ratio = _DOUBLET_RATIOS[key]
    assert ratio > 0.0, f"{strong}/{weak} ratio {ratio} is not positive"

    if expected is None:
        pytest.skip(f"{strong}/{weak}: {source}")

    rel_err = abs(ratio - expected) / expected
    assert rel_err < tol, (
        f"P-12: {strong}/{weak} ratio {ratio:.2f} differs from {expected:.2f} "
        f"({source}) by {100 * rel_err:.1f}%. Check the transition-probability source."
    )


@pytest.mark.parametrize(
    ("name", "vacuum", "tol", "air"),
    _VACUUM_WAVELENGTHS,
    ids=[w[0] for w in _VACUUM_WAVELENGTHS],
)
def test_wavelength_is_vacuum(catalog, name, vacuum, tol, air):
    """The catalog wavelength is the vacuum value, and not the air one.

    CLAUDE.md mandates vacuum throughout. The air check is the half that
    matters: a catalog rebuilt from an air-wavelength source lands 1.8 Å low on
    H-alpha, which is inside no tolerance anyone would notice by eye.
    """
    _line_list, names, wav = catalog

    assert name in names, (
        f"{name} is not in default_optical(), so the vacuum convention cannot be "
        f"checked. This used to `return` and report as passed."
    )

    wavelength = wav[name]
    assert abs(wavelength - vacuum) < tol, (
        f"{name} wavelength {wavelength:.2f} Å is not the vacuum value {vacuum:.2f} Å. "
        "CLAUDE.md mandate: all emission line wavelengths are vacuum."
    )

    if air is not None:
        assert abs(wavelength - air) > 1.0, (
            f"{name} wavelength {wavelength:.2f} Å matches the air value {air:.2f} Å. "
            "Violates the CLAUDE.md vacuum convention."
        )


def test_every_doublet_constraint_is_covered_here():
    """Non-vacuity: the table above tests every constraint the catalog declares.

    Two of the five entries in ``_DOUBLET_RATIOS`` had no test at all — MgII
    and SIII were added by this table. A constraint the catalog enforces and no
    test names is the gap this file exists to close, so the census is asserted
    rather than left to be noticed.
    """
    tested = {(strong, weak) for strong, weak, *_ in _DOUBLETS}
    declared = set(_DOUBLET_RATIOS)

    assert declared == tested, (
        f"untested constraints: {sorted(declared - tested)}; "
        f"table rows with no constraint: {sorted(tested - declared)}"
    )


def test_all_wavelengths_positive_and_reasonable(catalog):
    """Every line in the catalog must have positive wavelength in physical range.

    Pitfall P-12 (implicit): guards against corrupted line list entries with negative
    or zero wavelengths, which would break interp + photometry pipelines.
    """
    line_list, _names, _wav = catalog

    for i, (name, wav) in enumerate(zip(line_list.names, line_list.wavelengths)):
        wav_float = float(wav)
        assert wav_float > 0.0, f"Line {name} (idx {i}) has non-positive wavelength {wav_float} Å"
        assert 500.0 < wav_float < 1e6, (
            f"Line {name} (idx {i}) wavelength {wav_float} Å "
            f"outside physical range [500, 1e6] Å (likely data entry error)"
        )
