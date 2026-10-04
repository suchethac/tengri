# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic MIST-tagged SSP grid for the AGB dust-shell tests.

CI ships no ``fsps_mist_*`` grid, so the structural tests of the weighting
(seam, refusals, grammar, gradients) run on this grid and the tests that need
a real library skip when its file is absent. The grid sits on the template's
own wavelength, age and metallicity nodes, so resampling the ratio onto it is
exact; its flux is a smooth positive continuum with no physical content.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.components.stellar.agb_dust_shell import load_agb_dust_shell_template
from tengri.components.stellar.sps.dsps_wrapper import SSPData

#: Template metallicity nodes used (the middle one is the node nearest solar).
_Z_NODES = (7, 9, 11)


def synthetic_mist_ssp() -> SSPData:
    """A 3-D SSP grid whose ``source`` names the MIST isochrone.

    Returns
    -------
    SSPData
        Shape ``(3, 107, 5822)``: three template metallicity nodes, every
        template age node, every template wavelength node.
    """
    template = load_agb_dust_shell_template()
    lg_age_gyr = template.log_age_yr - 9.0
    wave = template.wave_angstrom
    rng = np.random.default_rng(11)
    continuum = (wave / 5000.0) ** -1.0
    age_scale = np.linspace(1.5, 0.5, lg_age_gyr.shape[0])
    z_scale = np.linspace(0.8, 1.2, len(_Z_NODES))
    flux = (
        1e-3
        * z_scale[:, None, None]
        * age_scale[None, :, None]
        * continuum[None, None, :]
        * rng.uniform(0.9, 1.1, (len(_Z_NODES), lg_age_gyr.shape[0], wave.shape[0]))
    )
    return SSPData(
        ssp_wave=jnp.asarray(wave),
        ssp_flux=jnp.asarray(flux),
        ssp_lg_age_gyr=jnp.asarray(lg_age_gyr),
        ssp_lgmet=jnp.asarray(template.log_z[list(_Z_NODES)]),
        ssp_mass_remaining=None,
        source="fsps_mist_synthetic",
    )
