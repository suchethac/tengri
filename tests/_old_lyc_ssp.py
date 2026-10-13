# SPDX-License-Identifier: BSD-3-Clause
"""A tracked FSPS grid with a post-AGB-like ionizing continuum added to its old bins.

The tracked FSPS/Padova grid (``data/fsps_prsc_miles_chabrier.h5``) has
``Q_H ~ 6e18`` photons/s/Msun at ages above 2.5 Gyr: no old-star Lyman continuum at all.
MIST grids carry ``Q_H ~ 3e40-1e41`` photons/s/Msun there, from post-AGB stars and
extreme horizontal-branch stars. This helper adds a 1e5 K blackbody below the Lyman
limit to every age bin from 1 Gyr on, normalized to a stated ``Q_H`` per Msun, so
tests can exercise the old-population path on a grid every runner has.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tengri.components.stellar.sps.dsps_wrapper import SSPData, load_ssp_data
from tengri.utils.blackbody import planck_bnu_wave
from tengri.utils.physics_constants import C_AA, H_PLANCK, L_SUN

FSPS_PRSC = Path(__file__).resolve().parents[1] / "data" / "fsps_prsc_miles_chabrier.h5"

#: Ionizing photon rate added to every age bin from OLD_FROM_LG_AGE_YR on [photons/s/Msun].
OLD_Q_H_PER_MSUN = 5.0e40
#: Blackbody temperature of the added old-star Lyman continuum [K].
OLD_T_EFF = 1.0e5
#: log10(age/yr) from which the added continuum is present.
OLD_FROM_LG_AGE_YR = 9.0
_LYMAN_LIMIT_AA = 911.76


def ssp_with_old_lyc(q_h_per_msun: float = OLD_Q_H_PER_MSUN) -> SSPData:
    """The tracked FSPS grid plus an old-star Lyman continuum.

    Parameters
    ----------
    q_h_per_msun : float
        Ionizing photon rate added at every age >= 1 Gyr [photons/s/Msun].

    Returns
    -------
    SSPData
        Same axes as ``fsps_prsc_miles_chabrier``; flux below the Lyman limit is replaced by
        a 1e5 K blackbody at the old ages.
    """
    ssp = load_ssp_data(str(FSPS_PRSC))
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    flux = np.array(ssp.ssp_flux, dtype=np.float64)
    log_age_yr = np.asarray(ssp.ssp_lg_age_gyr, dtype=np.float64) + 9.0
    old = log_age_yr >= OLD_FROM_LG_AGE_YR
    ion = wave < _LYMAN_LIMIT_AA
    shape = np.asarray(planck_bnu_wave(jnp.asarray(wave[ion]), OLD_T_EFF), dtype=np.float64)
    nu = C_AA / wave[ion]
    photons = shape * L_SUN / (H_PLANCK * nu)
    # Q = integral of L_nu / (h nu) d nu over the ionizing side; the grid points end below
    # the limit, so the trapezoid over them is the normalization (the tables read the same
    # integral with an exact edge step, which differs by the last half-bin).
    q_unit = float(np.abs(np.trapezoid(photons[::-1], nu[::-1])))
    scale = q_h_per_msun / q_unit
    flux[:, old, :][..., ion] = 0.0
    block = flux[:, old, :]
    block[..., ion] = scale * shape
    flux[:, old, :] = block
    return ssp._replace(ssp_flux=jnp.asarray(flux, dtype=ssp.ssp_flux.dtype))
