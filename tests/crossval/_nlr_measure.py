# SPDX-License-Identifier: BSD-3-Clause
"""Measure NLR line strengths from the emitted SED, not from private tables.

Three crossval files asserted NLR doublet ratios by reading
``_NLR_LINE_STRENGTHS`` / ``_NLR_LINE_WAVELENGTHS`` / ``_NLR_LINES`` out of
``tengri.components.agn.nlr``. Those arrays were removed in a refactor, so all
of them broke at import — and because the crossval tree is in no CI gate
(#1728), that went unnoticed, along with the [NII] deviation it was hiding
(#1752).

Reading the spectrum instead is refactor-proof and tests what a user receives.
Shared here so the three call sites cannot drift apart, and so the two
conventions that matter are stated once:

* **Vacuum wavelengths.** [OIII] 5008.24 / 4960.30, [NII] 6585.27 / 6549.86,
  Halpha 6564.61 — per the naming contract, air wavelengths are never used.
* **Narrow lines.** ``fwhm_kms=20`` separates the [NII] doublet from Halpha.
  At the 500 km/s default, sigma_lambda is ~4.6 A against separations of 14 and
  21 A, so the three blend and no windowed flux is clean.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

#: Vacuum wavelengths [Angstrom].
OIII_5007 = 5008.24
OIII_4959 = 4960.30
NII_6583 = 6585.27
NII_6548 = 6549.86
HALPHA = 6564.61
HBETA = 4862.69

#: Narrow enough to resolve the [NII] doublet away from Halpha.
FWHM_KMS = 20.0

#: Bright enough to keep the lines well above numerical noise.
L_DISC_BOL_ERG = 1e45


def nlr_sed(
    lo_aa: float = 3000.0, hi_aa: float = 7500.0, n_points: int = 200_000
) -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelength, L_nu) for the NLR at narrow line width."""
    from tengri.components.agn.nlr import compute_nlr_sed

    wave = jnp.linspace(lo_aa, hi_aa, n_points)
    sed = compute_nlr_sed(wave, l_disc_bol_erg=L_DISC_BOL_ERG, fwhm_kms=FWHM_KMS)
    return np.asarray(wave), np.asarray(sed)


def line_flux(
    wave: np.ndarray, sed: np.ndarray, center_aa: float, half_width_aa: float = 6.0
) -> float:
    """Integrate one line over a window wide enough to contain it."""
    mask = (wave > center_aa - half_width_aa) & (wave < center_aa + half_width_aa)
    return float(np.trapezoid(sed[mask], wave[mask]))


def doublet_ratio(bright_aa: float, faint_aa: float) -> float:
    """Flux ratio of two lines, measured from the emitted spectrum."""
    wave, sed = nlr_sed()
    return line_flux(wave, sed, bright_aa) / line_flux(wave, sed, faint_aa)
