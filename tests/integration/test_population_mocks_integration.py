# SPDX-License-Identifier: BSD-3-Clause
"""Integration test for mock galaxy population generator.

Tests that the population maker handles realistic forward model predictions
and correctly reports edge cases (Halpha in absorption) without silently
dropping galaxies.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, SEDModel
from tengri.analysis.population_mocks import make_population
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation import Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.slow

#: Vacuum rest wavelength of H-alpha [Angstrom] (tengri is vacuum-internal).
HALPHA_VACUUM = 6564.61
#: Gaussian sigma [Angstrom] of the injected stellar H-alpha absorption trough.
HALPHA_ABS_SIGMA = 2.0
#: Equivalent width [Angstrom] of the injected trough (realistic stellar Balmer value).
HALPHA_ABS_EW = 3.0


@pytest.fixture
def phot_obs():
    """10-band synthetic photometry observation for integration testing.

    Uses the same top-hat filter pattern as synthetic_tophat_obs but with
    more bands to exercise the forward model's photometry path thoroughly.
    """

    def _tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = [2500.0, 3500.0, 4500.0, 5500.0, 6500.0, 7500.0, 8500.0, 9500.0, 11000.0, 13000.0]
    curves = tuple(_tophat(c) for c in centers)
    return Observation(photometry=Photometry(filters=curves))


def _ssp_with_halpha_absorption(template: SSPData) -> SSPData:
    """``synthetic_ssp_wide`` with a genuine stellar H-alpha absorption trough.

    The template's continuum is smooth and has no H-alpha feature, so a
    linear side-band continuum measures a small negative residual from the
    continuum's own curvature (under exact windows, not edge leakage). Real
    stellar spectra carry a Balmer absorption line, so this injects one: a
    Gaussian trough of equivalent width ``HALPHA_ABS_EW`` at the vacuum
    H-alpha wavelength, applied to every age and metallicity.

    The wavelength grid is refined to 0.25 Angstrom across 6400-6750 Angstrom:
    the template's log grid has ~47 Angstrom spacing there, too coarse to
    sample a 2 Angstrom-wide trough at all. The continuum is re-evaluated
    analytically on the refined grid with the template's own formula, not
    interpolated, so its shape is unchanged.
    """
    wave = np.union1d(np.asarray(template.ssp_wave), np.arange(6400.0, 6750.0, 0.25))
    ages = np.asarray(template.ssp_lg_age_gyr)
    lgmet = np.asarray(template.ssp_lgmet)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = np.abs(flux) + 1e-12
    # EW = depth * sigma * sqrt(2 pi)  =>  depth = EW / (sigma * sqrt(2 pi)).
    depth = HALPHA_ABS_EW / (HALPHA_ABS_SIGMA * np.sqrt(2.0 * np.pi))
    trough = 1.0 - depth * np.exp(-0.5 * ((wave - HALPHA_VACUUM) / HALPHA_ABS_SIGMA) ** 2)
    return SSPData(
        ssp_wave=jnp.asarray(wave),
        ssp_flux=jnp.asarray(flux * trough[None, None, :]),
        ssp_lg_age_gyr=template.ssp_lg_age_gyr,
        ssp_lgmet=template.ssp_lgmet,
    )


@pytest.fixture
def field_model(synthetic_ssp_wide, phot_obs):
    """DPL + stochastic field at z = 0.1, 16 field latents.

    Uses synthetic_ssp_wide (no data/ files) so the test runs on every PR,
    with a stellar H-alpha absorption trough injected (see
    :func:`_ssp_with_halpha_absorption`). The stochastic field captures
    burstiness that can produce Halpha absorption during lulls in the SFH.
    """
    return SEDModel.build(
        ssp_data=_ssp_with_halpha_absorption(synthetic_ssp_wide),
        observation=phot_obs,
        sfh={"type": "dpl", "all_params": FREE, "age_gyr": 11.0, "field": {"all_params": FREE}},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        n_grid=16,
        redshift=Fixed(0.1),
    )


def test_halpha_absorption_galaxies_are_counted_not_dropped(field_model):
    """Galaxies with Halpha in absorption are counted and retained, never dropped.

    Dropping them would bias the survivors toward line-bright cases. The field
    SFH is a smooth continuum, so the SSP carries an injected stellar Halpha
    absorption trough (see :func:`_ssp_with_halpha_absorption`); every galaxy's
    predicted Halpha flux is then non-positive under the exact hard windows, and
    all of them must be counted in ``n_halpha_absorption`` and kept in the table.

    Before the trough was injected this test asserted zero absorption events.
    That expectation was an artifact: the smooth continuum's curvature gives a
    small negative residual under the linear side-band continuum, and the old
    soft windows masked it with a large positive edge-leakage term.
    """
    pop = make_population(
        field_model,
        n_galaxies=45,
        sigma_true=0.6,
        tau_true_myr=150.0,
        key=jax.random.PRNGKey(0),
        snr_phot=20.0,
        snr_line=10.0,
    )
    assert len(pop.truth_params) == 45, "galaxies were dropped"
    assert pop.n_halpha_absorption == 45, (
        "every galaxy's Halpha is in absorption (injected trough, no emission); "
        "all must be counted, not dropped or left uncounted"
    )
    assert pop.table is not None, "table should be populated"
    assert len(pop.table) == 45, "table row count should match n_galaxies"
