# SPDX-License-Identifier: BSD-3-Clause
"""#3468: Catalog.from_histories silently drops SFR tabulated after the observation time.

The observation time is age_at_z(z). SFR tabulated after it cannot form stars
visible at z, and the catalog path used to discard that mass with no warning.
``from_histories`` now refuses a history that runs past age_at_z(z) for its
galaxy row, and accepts one that ends at it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.conservation

_Z = 0.1


@pytest.fixture(scope="module")
def grid_ssp():
    """A small SSP grid whose oldest template (12.589 Gyr) is younger than the universe."""
    import jax.numpy as jnp

    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    wave = jnp.logspace(2.0, 7.0, 1600)
    lg_age = jnp.linspace(-3.0, float(np.log10(12.589)), 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (lg_age - lg_age.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=lg_age,
        ssp_lgmet=lgmet,
    )


@pytest.fixture(scope="module")
def fwd(grid_ssp):
    from tengri import DEFAULT, Fixed, ForwardModel, Observation, Photometry, SEDModel

    obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
    sed = SEDModel.build(
        ssp_data=grid_ssp,
        observation=obs,
        redshift=Fixed(_Z),
        sfh={"type": "table"},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 0.0,
            "tau_diff": 0.0,
        },
        neb={"type": "none"},
    )
    return ForwardModel.build(sed=sed, observation=obs)


def test_table_running_past_the_observation_time_is_refused(fwd):
    """SFR tabulated to 13.7 Gyr at z = 0.1 (age 12.44 Gyr) is refused, not dropped."""
    from tengri.inference.catalog import Catalog

    t = np.linspace(0.05, 13.7, 2000)
    with pytest.raises(ValueError, match=r"galaxy row 0.*13\.7.*12\.4"):
        Catalog.from_histories(fwd, t_gyr=t, sfr=np.ones_like(t)[None, :])


def test_the_refusal_names_the_offending_galaxy_row(fwd):
    from tengri.cosmology import age_at_z
    from tengri.inference.catalog import Catalog

    t_ok = np.linspace(0.05, float(age_at_z(_Z)), 2000)
    t_bad = np.linspace(0.05, 13.7, 2000)
    sfr = np.ones((2, 2000))
    t = np.stack([t_ok, t_bad])
    with pytest.raises(ValueError, match="galaxy row 1"):
        Catalog.from_histories(fwd, t_gyr=t, sfr=sfr)


def test_table_ending_at_the_observation_time_is_accepted(fwd):
    """A history that ends at age_at_z(z) builds, and its formed mass is the input integral."""
    from tengri.cosmology import age_at_z
    from tengri.inference.catalog import Catalog

    t = np.linspace(0.05, float(age_at_z(_Z)), 2000)
    sfr = np.ones_like(t)
    cat = Catalog.from_histories(fwd, t_gyr=t, sfr=sfr[None, :])
    out = cat.simulate(properties=("stellar_mass",), chunk_size=1)
    mass = float(np.asarray(out.properties["stellar_mass"]).ravel()[0])
    assert mass == pytest.approx(float(np.trapezoid(sfr, t * 1e9)), rel=1e-3)
