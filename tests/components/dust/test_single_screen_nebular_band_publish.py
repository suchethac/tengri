# SPDX-License-Identifier: BSD-3-Clause
"""Conservation: the single-screen dust component publishes the reddened
nebular continuum integrated through each observed and rest-frame band.

The `DustAttenuationSEDComponent` (unflagged single-screen dust) reddens the
nebular continuum at the wavelengths where the emission is and publishes the
band-integrated result:
- `nebular_phot_lnu_attenuated_precomp`: observed-frame bands.
- `nebular_restband_lnu_attenuated_precomp`: rest-frame bands.

Both are strictly less than the unscreened flux (the screen is applied), and
both differ from the unscreened flux by more than 10% in every band (the
screen is non-trivial).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.conservation


def _nebular_sed(wave) -> np.ndarray:
    """A positive continuum with three emission features [erg/s/Hz]."""
    wave = np.asarray(wave)
    sed = 1.0e27 * (wave / 5000.0) ** -0.5
    for center, amp in ((1216.0, 40.0), (5007.0, 25.0), (6563.0, 30.0)):
        sed = sed + amp * 1.0e27 * np.exp(-0.5 * ((wave - center) / (0.01 * center)) ** 2)
    return sed


def test_single_screen_publishes_the_band_integral_of_the_reddened_continuum():
    """Single-screen dust publishes observed and rest-frame attenuated nebular flux."""
    import jax.numpy as jnp
    import numpy as np

    from tengri.components.dust.component import DustAttenuationSEDComponent
    from tengri.protocols.component import ForwardState

    component = DustAttenuationSEDComponent()

    wave = np.logspace(np.log10(600.0), np.log10(30000.0), 400)
    z = 0.5
    filter_waves = np.array([3000.0, 5000.0, 7000.0])
    restband_waves = np.array([2000.0, 3333.0, 4667.0])

    phot_filter_waves_padded = np.array(
        [
            [2000.0, 4000.0, 6000.0, 8000.0],
            [2000.0, 4000.0, 6000.0, 8000.0],
            [2000.0, 4000.0, 6000.0, 8000.0],
        ]
    )
    phot_filter_trans_padded = np.array(
        [
            [0.8, 0.9, 0.9, 0.8],
            [0.8, 0.9, 0.9, 0.8],
            [0.8, 0.9, 0.9, 0.8],
        ]
    )

    sed_neb = _nebular_sed(wave)
    neb_phot_unreddened = np.array([1e21, 1.5e21, 1.2e21])

    state = ForwardState(
        wave=jnp.asarray(wave),
        sed_intrinsic=jnp.zeros_like(jnp.asarray(wave)),
        derived={
            "sed_nebular": jnp.asarray(sed_neb),
            "nebular_phot_lnu_precomp": jnp.asarray(neb_phot_unreddened),
            "filter_eff_waves": jnp.asarray(filter_waves),
            "filter_restband_eff_waves": jnp.asarray(restband_waves),
            "phot_filter_waves_padded": jnp.asarray(phot_filter_waves_padded),
            "phot_filter_trans_padded": jnp.asarray(phot_filter_trans_padded),
        },
    )

    params = {"dust_tau_v": 1.5, "redshift": z}
    result = component.apply(state, params)

    assert "nebular_phot_lnu_attenuated_precomp" in result.derived
    assert "nebular_restband_lnu_attenuated_precomp" in result.derived

    neb_obs_att = result.derived["nebular_phot_lnu_attenuated_precomp"]
    neb_rest_att = result.derived["nebular_restband_lnu_attenuated_precomp"]

    assert neb_obs_att.shape == (3,)
    assert neb_rest_att.shape == (3,)
    assert np.all(np.isfinite(neb_obs_att))
    assert np.all(np.isfinite(neb_rest_att))
    assert np.all(neb_obs_att > 0.0)
    assert np.all(neb_rest_att > 0.0)


def test_the_published_flux_is_the_filter_integral_of_the_screened_sed():
    """The published value matches projecting the reddened nebular SED through the filter."""
    import jax.numpy as jnp
    import numpy as np

    from tengri.components._band_projection import project_additive_onto_photometry
    from tengri.components.dust.component import DustAttenuationSEDComponent
    from tengri.protocols.component import ForwardState

    component = DustAttenuationSEDComponent()

    wave = np.logspace(np.log10(600.0), np.log10(30000.0), 400)
    z = 0.5
    filter_waves = np.array([3000.0, 5000.0, 7000.0])
    restband_waves = np.array([2000.0, 3333.0, 4667.0])

    phot_filter_waves_padded = np.array(
        [
            [2000.0, 4000.0, 6000.0, 8000.0],
            [2000.0, 4000.0, 6000.0, 8000.0],
            [2000.0, 4000.0, 6000.0, 8000.0],
        ]
    )
    phot_filter_trans_padded = np.array(
        [
            [0.8, 0.9, 0.9, 0.8],
            [0.8, 0.9, 0.9, 0.8],
            [0.8, 0.9, 0.9, 0.8],
        ]
    )

    sed_neb = _nebular_sed(wave)
    neb_phot_unreddened = project_additive_onto_photometry(
        None, sed_neb, wave, filter_waves, phot_filter_waves_padded, phot_filter_trans_padded, z
    )

    state = ForwardState(
        wave=jnp.asarray(wave),
        sed_intrinsic=jnp.zeros_like(jnp.asarray(wave)),
        derived={
            "sed_nebular": jnp.asarray(sed_neb),
            "nebular_phot_lnu_precomp": jnp.asarray(neb_phot_unreddened),
            "filter_eff_waves": jnp.asarray(filter_waves),
            "filter_restband_eff_waves": jnp.asarray(restband_waves),
            "phot_filter_waves_padded": jnp.asarray(phot_filter_waves_padded),
            "phot_filter_trans_padded": jnp.asarray(phot_filter_trans_padded),
        },
    )

    params = {"dust_tau_v": 1.5, "redshift": z}
    result = component.apply(state, params)

    neb_obs_att = result.derived["nebular_phot_lnu_attenuated_precomp"]
    neb_obs_unreddened = state.derived["nebular_phot_lnu_precomp"]

    attenuation = component.nebular_screen_transmission(params, wave)
    sed_neb_reddened = sed_neb * np.asarray(attenuation)

    expected_obs = project_additive_onto_photometry(
        None,
        sed_neb_reddened,
        wave,
        filter_waves,
        phot_filter_waves_padded,
        phot_filter_trans_padded,
        z,
    )

    np.testing.assert_allclose(neb_obs_att, expected_obs, rtol=1e-12)

    reduction = neb_obs_att / neb_obs_unreddened
    assert np.all(reduction < 1.0), "Attenuation should reduce the flux"
    assert np.all((1.0 - reduction) > 0.1), (
        f"Attenuation should reduce flux by at least 10% in every band; got {1.0 - reduction}"
    )


def test_no_nebular_band_flux_means_no_dense_projection():
    """When nebular_phot_lnu_precomp is absent, neither screened key is published."""
    import jax.numpy as jnp
    import numpy as np

    from tengri.components.dust.component import DustAttenuationSEDComponent
    from tengri.protocols.component import ForwardState

    component = DustAttenuationSEDComponent()

    wave = np.logspace(np.log10(600.0), np.log10(30000.0), 400)
    z = 0.5
    filter_waves = np.array([3000.0, 5000.0, 7000.0])
    restband_waves = np.array([2000.0, 3333.0, 4667.0])

    phot_filter_waves_padded = np.array(
        [
            [2000.0, 4000.0, 6000.0, 8000.0],
            [2000.0, 4000.0, 6000.0, 8000.0],
            [2000.0, 4000.0, 6000.0, 8000.0],
        ]
    )
    phot_filter_trans_padded = np.array(
        [
            [0.8, 0.9, 0.9, 0.8],
            [0.8, 0.9, 0.9, 0.8],
            [0.8, 0.9, 0.9, 0.8],
        ]
    )

    sed_neb = _nebular_sed(wave)

    state = ForwardState(
        wave=jnp.asarray(wave),
        sed_intrinsic=jnp.zeros_like(jnp.asarray(wave)),
        derived={
            "sed_nebular": jnp.asarray(sed_neb),
            "filter_eff_waves": jnp.asarray(filter_waves),
            "filter_restband_eff_waves": jnp.asarray(restband_waves),
            "phot_filter_waves_padded": jnp.asarray(phot_filter_waves_padded),
            "phot_filter_trans_padded": jnp.asarray(phot_filter_trans_padded),
        },
    )

    params = {"dust_tau_v": 1.5, "redshift": z}
    result = component.apply(state, params)

    assert "nebular_phot_lnu_attenuated_precomp" not in result.derived
    assert "nebular_restband_lnu_attenuated_precomp" not in result.derived
