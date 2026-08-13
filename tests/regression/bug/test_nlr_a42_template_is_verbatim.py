# SPDX-License-Identifier: BSD-3-Clause
"""#1752: the NLR a42 template must stay verbatim, doublet "violations" included.

``_RICHARDSON_FLUXES`` is Richardson+2014 Table 3 column 'a42' — *dereddened
emission-line strengths for the AGN locus, relative to Hbeta*, measured off
fifteen high-S/N stacked SDSS composites forming a sequence in NLR ionization
level. It is a **measurement**, not a photoionization model.

That distinction is the whole point of this file. Three of the tabulated pairs
are forbidden doublets whose members decay from a single upper level, so in a
*model* their ratio is fixed by the transition probabilities alone. As
tabulated, a42 does not reproduce those atomic values:

===============  =========  ========  ==============================
doublet          tabulated  atomic    why the measurement deviates
===============  =========  ========  ==============================
[O III] 5007/4959     2.97      2.98   agrees; within the table's own
                                       two-decimal rounding
[N II]  6584/6548     2.70      2.94   6548 is a weak line sitting on
                                       the H-alpha wing of a stack
[O I]   6300/6363     3.67      3.00   6363 is quoted as 0.09 — one
                                       significant figure
===============  =========  ========  ==============================

#1752 read the [N II] deviation as a bug and rewrote 6548 to ``2.13 / 2.96``
(and 4959 to ``8.53 / 2.98``). That was wrong on both counts:

1. It imposed a photoionization-model constraint on an empirical table. The
   atomic ratio *is* enforced in tengri — in ``components/nebular/shock.py``
   and the Cloudy-grid NLR backends, where it belongs, because those generate
   line strengths from level populations.
2. It broke the parity this table exists for. The 23 published values agree
   value-for-value with the same table as carried by Prospector
   (``AGNSpecModel.init_aline_info``, on the same FSPS line indices), which is
   exactly what ``nlr.py`` claims in its Notes.

So the guard is a parity assertion, not a physics assertion. Anyone who
"corrects" a doublet here fails ``test_a42_fluxes_match_the_published_table``,
and ``test_forbidden_doublets_keep_their_measured_deviations`` states in the
failure message why the deviation is intentional.

References
----------
.. [1] C. T. Richardson, J. T. Allen, J. A. Baldwin, P. C. Hewett, and
   G. J. Ferland, "Interpreting the ionization sequence in AGN emission-line
   spectra," MNRAS, 2014, 437, 3, 2376-2403. Table 3, column 'a42'.
   https://doi.org/10.1093/mnras/stt2056
.. [2] B. D. Johnson, et al., "Prospector: Inferring the Star Formation
   Histories of Galaxies from Observed Spectral Energy Distributions,"
   ApJS, 254, 22, 2021. https://doi.org/10.3847/1538-4365/abef67
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

#: Richardson+2014 Table 3 'a42', in the wavelength order tengri and Prospector
#: both use (FSPS ``emline_wavelengths`` indices 38, 40, 41, 43, 45, 50, 51, 52,
#: 59, 61, 62, 64, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 80). Cross-checked
#: against Prospector ``prospect/models/sedmodel.py::AGNSpecModel.init_aline_info``,
#: which carries the same published column.
_PUBLISHED_A42_FLUXES = (
    2.96, 0.06, 0.10, 1.00, 0.20, 0.25, 0.48, 0.13, 1.00, 2.87, 8.53, 0.07,
    0.02, 0.10, 0.33, 0.09, 0.79, 2.86, 2.13, 0.03, 0.77, 0.65, 0.19,
)  # fmt: skip

#: Vacuum wavelengths [Angstrom] of the three forbidden doublets, as
#: ``(strong, weak, tabulated ratio, atomic ratio, why it deviates)``.
_DOUBLETS = (
    (5008.31, 4960.37, 8.53 / 2.87, 2.98, "agrees to within the table's rounding"),
    (6585.37, 6549.96, 2.13 / 0.79, 2.94, "[N II] 6548 is weak and on the H-alpha wing"),
    (6302.14, 6365.64, 0.33 / 0.09, 3.00, "[O I] 6363 is tabulated to one significant figure"),
)


def test_a42_fluxes_match_the_published_table():
    """The template is the published a42 column, unmodified.

    This is the guard #1752 should have had. It fails on any edit to the flux
    array — including a well-intentioned one that ties a doublet to its atomic
    ratio, which is what #1752 did and what this file exists to prevent.
    """
    from tengri.components.agn.nlr import _RICHARDSON_FLUXES

    np.testing.assert_allclose(
        np.asarray(_RICHARDSON_FLUXES),
        np.asarray(_PUBLISHED_A42_FLUXES),
        rtol=0.0,
        atol=0.0,
        err_msg=(
            "The a42 flux template has been modified. It is the published "
            "Richardson+2014 Table 3 'a42' column (the same values Prospector "
            "carries), and nlr.py claims that parity. If you changed a doublet "
            "to satisfy an atomic branching ratio: don't — Table 3 is measured off "
            "stacked observed spectra, not computed from level populations. See "
            "#1752."
        ),
    )


def test_template_has_one_entry_per_line():
    """Fluxes and wavelengths stay aligned — a shifted array is a silent rescale."""
    from tengri.components.agn.nlr import _RICHARDSON_FLUXES, _RICHARDSON_WAVES

    assert len(_RICHARDSON_WAVES) == len(_PUBLISHED_A42_FLUXES) == len(_RICHARDSON_FLUXES)


@pytest.mark.parametrize(
    "strong_aa,weak_aa,tabulated,atomic,reason",
    _DOUBLETS,
    ids=["oiii_5007_4959", "nii_6584_6548", "oi_6300_6363"],
)
def test_forbidden_doublets_keep_their_measured_deviations(
    strong_aa: float, weak_aa: float, tabulated: float, atomic: float, reason: str
):
    """Each doublet keeps the ratio a42 *measured*, not the one atoms require.

    Asserting the tabulated ratio (rather than the atomic one) is deliberate:
    it documents the deviation as a known, inherited property of the empirical
    template and makes any silent retie fail with an explanation attached.
    """
    from tengri.components.agn.nlr import _RICHARDSON_FLUXES, _RICHARDSON_WAVES

    waves = np.asarray(_RICHARDSON_WAVES)
    fluxes = np.asarray(_RICHARDSON_FLUXES)

    strong = fluxes[int(np.argmin(np.abs(waves - strong_aa)))]
    weak = fluxes[int(np.argmin(np.abs(waves - weak_aa)))]

    np.testing.assert_allclose(
        strong / weak,
        tabulated,
        rtol=1e-9,
        err_msg=(
            f"Doublet {strong_aa:.0f}/{weak_aa:.0f} should carry the a42 measured "
            f"ratio {tabulated:.3f}, not the atomic {atomic:.2f}. The deviation is "
            f"expected: {reason}. Richardson+2014 Table 3 is dereddened strengths "
            f"measured off stacked SDSS composites (#1752)."
        ),
    )
