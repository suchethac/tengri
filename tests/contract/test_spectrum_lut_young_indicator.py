# SPDX-License-Identifier: BSD-3-Clause
"""SpectrumPrecomp LUT publishes the single _young_indicator definition.

Spectroscopy-only models with SpectrumPrecomp (the default for spectroscopy fits)
must apply the same birth-cloud selector as the exact screen and the photometry LUT.
The indicator was the base-10 logistic (2.3x sharper), causing young-star reddenings
to differ by 20%+ between paths. With the fix, all three paths use the natural
sigmoid.

References
----------
.. [1] Charlot, S. & Fall, S. M. 2000, ApJ 539, 718 — Birth cloud phase definition.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    SEDModel,
    SpectrumPrecomp,
    list_dust_laws,
)
from tengri.components.dust.two_component import _young_indicator
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "law",
    [row["name"] for row in list_dust_laws()],
    ids=lambda x: x,
)
def test_spectroscopy_only_spectrum_lut_publishes_the_single_indicator(synthetic_ssp_wide, law):
    """SpectrumPrecomp-only model publishes dust_young_indicator = _young_indicator.

    For every registered dust law, a spectroscopy-only model under SpectrumPrecomp
    must apply the same stellar birth-cloud selector (the natural sigmoid) that the
    exact path applies.

    Parameters
    ----------
    law : str
        Dust attenuation law name from list_dust_laws().
    """
    wave_obs = jnp.logspace(jnp.log10(3300.0), jnp.log10(8000.0), 80)
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "two_component", "law": law, "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.3),
        approx=SpectrumPrecomp(),
    )

    st = model.predict_state({})

    # Expected indicator from the definition. The defaults for two_component dust
    # are t_birth_yr=1e7 and transition_width_dex=0.3.
    ssp = synthetic_ssp_wide
    ssp_ages_yr = 10.0 ** np.asarray(ssp.ssp_lg_age_gyr) * 1e9
    t_birth_yr = 1e7
    transition_width_dex = 0.3
    expected = _young_indicator(
        jnp.asarray(ssp_ages_yr),
        t_birth_yr,
        transition_width_dex,
    )

    got = st.derived["dust_young_indicator"]
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(expected),
        atol=1e-12,
        rtol=0.0,
        err_msg=f"law={law}: published indicator does not match _young_indicator",
    )


def _young_sfh_and_laws():
    """SFH/law pairs for young-population LUT testing."""
    const_20_myr = {
        "type": "const",
        "start_gyr": Fixed(0.02),
        "end_gyr": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    }
    const_100_myr = {
        "type": "const",
        "start_gyr": Fixed(0.1),
        "end_gyr": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    }
    default_dpl = {"type": "delayed", "all_params": Fixed(DEFAULT)}
    return [
        (const_20_myr, "calzetti"),
        (const_100_myr, "calzetti"),
        (default_dpl, "calzetti"),
        (const_20_myr, "power_law"),
        (const_100_myr, "power_law"),
        (default_dpl, "power_law"),
        (const_20_myr, "noll09"),
        (const_100_myr, "noll09"),
        (default_dpl, "noll09"),
    ]


@pytest.mark.parametrize(
    "sfh,law",
    _young_sfh_and_laws(),
)
def test_spectrum_lut_matches_exact_for_young_populations(synthetic_ssp_wide, sfh, law):
    r"""Spectrum LUT agrees with exact for young 0-20/100 Myr and default SFH.

    The SpectrumPrecomp LUT and the exact path must redden the same stars. Before
    the fix, the LUT used a 2.3x sharper indicator, causing disagreement up to
    20% in young-dominated SFHs.

    Issue #617 documents the LUT residual class: 0.8–1.5 % for the two-component
    birth-cloud approximation. Deviation from that class signals the indicator bug.

    Parameters
    ----------
    sfh : dict
        Star formation history specification (constant 0–20 Myr, 0–100 Myr, or
        default delayed-tau).
    law : str
        Dust attenuation law ("calzetti", "power_law", or "noll09").
    """
    # Build observation with 80-point log grid 3300–8000 Å.
    wave_obs = jnp.logspace(jnp.log10(3300.0), jnp.log10(8000.0), 80)
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))

    # LUT and exact models with the same config.
    model_lut = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh=sfh,
        dust_attenuation={
            "type": "two_component",
            "law": law,
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(0.3),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.3),
        approx=SpectrumPrecomp(),
    )

    model_exact = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh=sfh,
        dust_attenuation={
            "type": "two_component",
            "law": law,
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(0.3),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.3),
        approx=None,
    )

    # Spectrum ratio.
    spec_lut = np.asarray(model_lut.predict_spectrum({}))
    spec_exact = np.asarray(model_exact.predict_spectrum({}))
    ratio = spec_lut / spec_exact

    # The bound is the documented LUT residual class (#617: 0.8–1.5 %).
    # The unfixed base-10 indicator is 20%+ too high for young populations.
    max_rel_error = np.max(np.abs(ratio - 1.0))
    err_msg = (
        f"sfh={sfh['type']}, law={law}: "
        f"LUT/exact max rel error {max_rel_error:.2%} exceeds 2.0% bound. "
        f"The young indicator may not be using the correct formula (#2591)."
    )
    assert max_rel_error < 0.02, err_msg
