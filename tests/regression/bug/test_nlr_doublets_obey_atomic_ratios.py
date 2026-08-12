# SPDX-License-Identifier: BSD-3-Clause
"""#1752: the NLR template carried [N II] 6583/6548 = 2.70 against an atomic 2.96.

Both members of a forbidden doublet decay from the *same* upper level, so their
intensity ratio is fixed by the two transition probabilities alone — independent
of density, temperature, ionization parameter and abundance. It is one of the
few numbers in a nebular spectrum with no physical freedom, and observers use it
as a sanity check on line fitting and flux calibration.

Richardson+2014 Table 3 'a42' tabulates the two [N II] lines as independent
numbers (6548 = 0.79, 6584 = 2.13), which is a ratio of 2.70 — 9% off. Carrying
them independently is what made the violation possible; the weak member of each
doublet is now derived from the strong one, so the constraint cannot drift.

The strong lines keep their tabulated values, so [O III] 5007 and [N II] 6583 —
the lines BPT diagnostics are built on — are unchanged and only the tied
partners move.

Ratios are measured from the emitted spectrum rather than the template arrays,
so this also pins that the constraint survives the line-profile machinery. The
measured ratio runs ~1% above the strength ratio because a Gaussian of fixed
velocity width has sigma_lambda proportional to lambda, so the redder line
integrates slightly more flux for the same peak strength.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

#: Vacuum wavelengths [Angstrom].
_OIII_5007, _OIII_4959 = 5008.24, 4960.30
_NII_6583, _NII_6548 = 6585.27, 6549.86

#: Storey & Zeippen 2000.
_OIII_ATOMIC = 2.98
_NII_ATOMIC = 2.96


def _doublet_ratio(bright_aa: float, faint_aa: float) -> float:
    """Flux ratio of two lines, measured from the emitted NLR spectrum."""
    from tengri.components.agn.nlr import compute_nlr_sed

    # Narrow lines so the [N II] doublet is not blended with Halpha; at the
    # 500 km/s default sigma_lambda is ~4.6 A against separations of 14 and 21 A.
    wave = jnp.linspace(3000.0, 7500.0, 200_000)
    sed = np.asarray(compute_nlr_sed(wave, l_disc_bol_erg=1e45, fwhm_kms=20.0))
    wave_np = np.asarray(wave)

    def flux(center: float) -> float:
        window = (wave_np > center - 6.0) & (wave_np < center + 6.0)
        return float(np.trapezoid(sed[window], wave_np[window]))

    return flux(bright_aa) / flux(faint_aa)


def test_nii_doublet_obeys_its_atomic_ratio():
    """[N II] 6583/6548. Was 2.70 before #1752; the atomic value is 2.96."""
    ratio = _doublet_ratio(_NII_6583, _NII_6548)
    np.testing.assert_allclose(
        ratio,
        _NII_ATOMIC,
        rtol=0.03,
        err_msg=f"[N II] 6583/6548 = {ratio:.4f}, atomic value {_NII_ATOMIC}",
    )


def test_oiii_doublet_obeys_its_atomic_ratio():
    """[O III] 5007/4959. Tabulated independently at 2.972; now tied to 2.98."""
    ratio = _doublet_ratio(_OIII_5007, _OIII_4959)
    np.testing.assert_allclose(
        ratio,
        _OIII_ATOMIC,
        rtol=0.03,
        err_msg=f"[O III] 5007/4959 = {ratio:.4f}, atomic value {_OIII_ATOMIC}",
    )


def test_strong_doublet_members_keep_their_tabulated_strengths():
    """The BPT diagnostic lines are untouched by the tie.

    [O III] 5007 and [N II] 6583 carry the Richardson+2014 values; only their
    weak partners are derived. If a future change ties the *strong* line
    instead, every [N II]/Halpha and [O III]/Hbeta in the literature comparison
    shifts, and this fails.
    """
    from tengri.components.agn.nlr import _RICHARDSON_FLUXES, _RICHARDSON_WAVES

    waves = np.asarray(_RICHARDSON_WAVES)
    fluxes = np.asarray(_RICHARDSON_FLUXES)

    for center, expected, name in (
        (_OIII_5007, 8.53, "[O III] 5007"),
        (_NII_6583, 2.13, "[N II] 6583"),
    ):
        idx = int(np.argmin(np.abs(waves - center)))
        np.testing.assert_allclose(
            fluxes[idx],
            expected,
            rtol=1e-9,
            err_msg=f"{name} must keep its tabulated Richardson+2014 strength",
        )
