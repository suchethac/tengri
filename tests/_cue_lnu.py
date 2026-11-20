# SPDX-License-Identifier: BSD-3-Clause
"""Per-age ionizing spectrum of a population, as the stellar component publishes it.

Test helper only: the model receives this array from the stellar component as
``lnu_age_ion``. It is the SSP ionizing prefix (wavelength below 2 x the Lyman limit)
read at the population's log Z with the stellar code's own PCHIP metallicity read, then
weighted by the age weights.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.components.lyc import LYMAN_LIMIT_AA
from tengri.components.stellar.sps.met_pchip import read_met


def cue_lnu_age_ion(ssp_data, weights, log_z):
    """Per-age ionizing spectrum of a population, as the stellar component publishes it.

    Uses the stellar code's own metallicity read (tengri.components.stellar.sps.met_pchip.read_met)
    on the SSP ionizing prefix (wavelength < 2 * 911.76 A), times the age weights.
    Test helper only: the model gets this array from the stellar component as 'lnu_age_ion'.

    Parameters
    ----------
    ssp_data : SSPData
        SSP grid with ``ssp_wave (n_wave,)``, ``ssp_flux (n_met, n_age, n_wave)`` (or a
        4-D flux with an [alpha/Fe] axis at position 1) and ``ssp_lgmet``.
    weights : array_like, shape (n_age,)
        Age weights [Msun per age bin].
    log_z : scalar, possibly traced
        Absolute log10(Z).

    Returns
    -------
    jnp.ndarray, shape (n_age, n_ion)
        Age-weighted ionizing spectrum on the SSP ionizing prefix.
    """
    wave = np.asarray(ssp_data.ssp_wave)
    flux = jnp.asarray(ssp_data.ssp_flux)
    if flux.ndim == 4:
        i_alpha = int(np.argmin(np.abs(np.asarray(ssp_data.ssp_alpha_fe))))
        flux = flux[:, i_alpha]
    n_ion = int(np.count_nonzero(wave < 2.0 * LYMAN_LIMIT_AA))
    lnu = read_met(flux[..., :n_ion], ssp_data.ssp_lgmet, log_z)
    return jnp.asarray(weights)[:, None] * lnu
