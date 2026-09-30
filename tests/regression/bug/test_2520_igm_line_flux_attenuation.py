# SPDX-License-Identifier: BSD-3-Clause
"""#2520: predict_line_fluxes / measure_line_fluxes must apply IGM transmission.

The model's own spectrum channel attenuates ``state.sed_observed`` by the
configured IGM component (``IGMSEDComponent.apply``), but the line-flux
surfaces read a discrete line catalog and divide by ``4 pi d_L^2`` directly,
never touching that transmission. A line at a redshift where the Lyman
forest bites (Ly-alpha at z >~ 2) therefore came out brighter than the same
galaxy's own spectrum -- an internal physical inconsistency, not merely an
enhancement.

Every check compares two models built identically except for ``igm=``,
sampled from the SAME PRNG key (the two builds declare the same free
parameters, so the samples are bit-identical), so dust/SFH/metallicity
cancel in the ratio and only the IGM factor survives.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

import tengri
from tengri import Fixed, SEDModel, recipes
from tengri.components.igm.igm import igm_absorption
from tengri.observation.line_measurement import default_line_defs
from tengri.utils.physics_constants import C_KM_S

pytestmark = pytest.mark.regression_bug

# Matches SEDModel._IGM_LINE_PROFILE_N_POINTS / _N_SIGMA and the declared
# Fixed default of neb_eline_sigma_kms (components/nebular/_params.py).
_N_POINTS = 201
_N_SIGMA = 3.0
_SIGMA_GAS_KMS = 100.0

# Every registered IGM model ('inoue' is a bare alias of 'inoue14': the same
# callable, so testing both is a trivial duplicate, not a boundary case) plus
# the off switch.
_IGM_TYPES = ("inoue14", "madau", "meiksin06", "asada25", "none")
_REDSHIFTS = (0.5, 3.0, 6.0)
_LINES = ((1215.67, "Lya"), (1549.0, "CIV"), (6564.61, "Halpha"))
_LINE_WAVES = np.array([w for w, _ in _LINES])

_BARE_SSP_CANDIDATES = ("fsps_prsc_miles_chabrier", "ssp_prsc_bc03_chabrier")


def _bare_ssp_name():
    for name in _BARE_SSP_CANDIDATES:
        if Path(f"data/{name}.h5").is_file():
            return name
    return None


@pytest.fixture(scope="module")
def ssp():
    name = _bare_ssp_name()
    if name is None:
        pytest.skip("No bare-stellar SSP file available")
    return tengri.load_ssp(name, download=False)


def _build(ssp_data, igm_type, z):
    recipe = dict(recipes.star_forming_photometry(), redshift=Fixed(z))
    recipe["igm"] = {"type": igm_type}
    return SEDModel.build(ssp_data=ssp_data, observation=None, **recipe)


def _profile_averaged_transmission(igm_type, z, line_wave_rest, sigma_kms=_SIGMA_GAS_KMS):
    """Independent numpy reimplementation of the Gaussian-stencil profile average.

    Mirrors ``SEDModel._line_igm_transmission``'s quadrature (same node
    count, same +/- 3 sigma window, same trapezoid), written fresh against
    :func:`igm_absorption` -- the model's own IGM dispatch -- rather than
    calling the method under test, so it checks the WIRING (does the model
    apply this factor to the flux ratio), not the quadrature itself.
    """
    z = float(z)
    line_wave_obs = float(line_wave_rest) * (1.0 + z)
    sigma_obs_aa = sigma_kms / C_KM_S * line_wave_obs
    offsets = np.linspace(-_N_SIGMA, _N_SIGMA, _N_POINTS)
    stencil = line_wave_obs + offsets * sigma_obs_aa
    T = np.asarray(igm_absorption(stencil, z, igm_model=igm_type))
    profile = np.exp(-0.5 * offsets**2)
    norm = np.trapezoid(profile, offsets)
    return float(np.trapezoid(T * profile, offsets) / norm)


class TestPredictLineFluxesIGM:
    """predict_line_fluxes must apply the model's own IGM transmission (#2520)."""

    @pytest.mark.parametrize("z", _REDSHIFTS)
    @pytest.mark.parametrize("igm_type", _IGM_TYPES)
    def test_flux_ratio_matches_profile_averaged_transmission(self, ssp, igm_type, z):
        key = jax.random.PRNGKey(0)
        model_igm = _build(ssp, igm_type, z)
        model_none = _build(ssp, "none", z)
        params_igm = model_igm.spec.sample(key)
        params_none = model_none.spec.sample(key)

        flux_igm = np.asarray(
            model_igm.predict_line_fluxes(params_igm, target_wavelengths=_LINE_WAVES)
        )
        flux_none = np.asarray(
            model_none.predict_line_fluxes(params_none, target_wavelengths=_LINE_WAVES)
        )
        ratio = flux_igm / flux_none

        for i, (wave, name) in enumerate(_LINES):
            expected = _profile_averaged_transmission(igm_type, z, wave)
            assert ratio[i] == pytest.approx(expected, abs=1e-6, rel=1e-6), (
                f"{name} at z={z}, igm={igm_type!r}: flux ratio {ratio[i]!r} != "
                f"profile-averaged transmission {expected!r}"
            )
            if igm_type == "none":
                assert ratio[i] == pytest.approx(1.0, abs=1e-9)

    def test_civ_beyond_forest_at_low_z(self, ssp):
        """CIV 1549 at z=0.5 (observed 2323.5 A) is redward of the Lyman forest: T~=1."""
        z = 0.5
        key = jax.random.PRNGKey(1)
        model_igm = _build(ssp, "inoue14", z)
        model_none = _build(ssp, "none", z)
        flux_igm = float(
            np.asarray(
                model_igm.predict_line_fluxes(
                    model_igm.spec.sample(key), target_wavelengths=np.array([1549.0])
                )
            )[0]
        )
        flux_none = float(
            np.asarray(
                model_none.predict_line_fluxes(
                    model_none.spec.sample(key), target_wavelengths=np.array([1549.0])
                )
            )[0]
        )
        assert flux_igm / flux_none == pytest.approx(1.0, abs=1e-3)

    def test_lya_z3_forest_attenuation(self, ssp):
        """Ly-alpha at z=3, sigma_gas=100 km/s: profile-averaged T ~= 0.835 (#2520)."""
        z = 3.0
        key = jax.random.PRNGKey(2)
        model_igm = _build(ssp, "inoue14", z)
        model_none = _build(ssp, "none", z)
        flux_igm = float(
            np.asarray(
                model_igm.predict_line_fluxes(
                    model_igm.spec.sample(key), target_wavelengths=np.array([1215.67])
                )
            )[0]
        )
        flux_none = float(
            np.asarray(
                model_none.predict_line_fluxes(
                    model_none.spec.sample(key), target_wavelengths=np.array([1215.67])
                )
            )[0]
        )
        ratio = flux_igm / flux_none
        assert ratio == pytest.approx(0.835, abs=0.02)

    def test_fast_grid_matches_exact_path(self, ssp):
        """The #950 per-Q_H grid path must apply IGM identically to the exact path.

        The grid caches a distance-free, IGM-free per-Q_H ratio (mirroring how
        it already cancels the luminosity-distance factor the build-time probe
        computes) and re-applies the transmission exactly once at read time;
        without that cancellation the transmission is squared (baked in once
        at build, again at read), which is what the mutation check below pins.
        """
        z = 3.0
        model = _build(ssp, "inoue14", z)
        key = jax.random.PRNGKey(3)
        params = model.spec.sample(key)
        exact = np.asarray(model.predict_line_fluxes(params, target_wavelengths=_LINE_WAVES))

        model.enable_fast_nebular(_LINE_WAVES)
        fast = np.asarray(model.predict_line_fluxes(params, target_wavelengths=_LINE_WAVES))

        # The #950 grid is itself an *interpolated* approximation (independent
        # of IGM: measured ~0.15-0.65% relative error on this fixture with
        # igm='none'), so parity is bounded by that pre-existing accuracy, not
        # by machine precision. 2% catches the IGM-squaring defect (which
        # moved the Ly-alpha line by ~-17%) with headroom above the grid's own
        # baseline error.
        np.testing.assert_allclose(fast, exact, rtol=2e-2)


class TestMeasureLineFluxesIGM:
    """measure_line_fluxes must also apply IGM transmission (#2520).

    It measures off the model's REST-frame SED (``_predict_rest_sed`` / the
    window LUT), which never carries IGM -- only ``state.sed_observed`` does.
    """

    def test_measured_flux_ratio_matches_predict_line_fluxes(self, ssp):
        z = 3.0
        key = jax.random.PRNGKey(4)
        model_igm = _build(ssp, "inoue14", z)
        model_none = _build(ssp, "none", z)
        params_igm = model_igm.spec.sample(key)
        params_none = model_none.spec.sample(key)

        line_defs = default_line_defs([1215.67], names=["Lya"])
        measured_igm = np.asarray(model_igm.measure_line_fluxes(params_igm, line_defs=line_defs))
        measured_none = np.asarray(
            model_none.measure_line_fluxes(params_none, line_defs=line_defs)
        )
        ratio = float(measured_igm[0] / measured_none[0])

        predicted_igm = float(
            np.asarray(
                model_igm.predict_line_fluxes(params_igm, target_wavelengths=np.array([1215.67]))
            )[0]
        )
        predicted_none = float(
            np.asarray(
                model_none.predict_line_fluxes(params_none, target_wavelengths=np.array([1215.67]))
            )[0]
        )
        predict_ratio = predicted_igm / predicted_none

        expected = _profile_averaged_transmission("inoue14", z, 1215.67)
        assert ratio == pytest.approx(expected, abs=1e-6, rel=1e-6)
        assert ratio == pytest.approx(predict_ratio, rel=1e-6)
