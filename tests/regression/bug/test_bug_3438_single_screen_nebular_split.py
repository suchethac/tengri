# SPDX-License-Identifier: BSD-3-Clause
"""#3438: single-screen dust republishes the nebular light it reddens.

The exact spectrum splits the rest SED by kernel. ``sed_nebular`` and ``sed_shock``
go through the instrument kernel alone; everything else in ``sed_intrinsic`` goes
through the stellar kernel (velocity dispersion plus instrument). A single screen
attenuates the summed ``sed_intrinsic`` but used to leave ``sed_nebular`` and
``sed_shock`` unscreened, so the split put the unattenuated lines in the instrument
group and ``(T - 1) * sed_nebular`` in the stellar group, which is broadened by
``sigma_v``. Two-component dust republishes its reddened nebular light; this test
pins the single-screen path to the same contract.
"""

from __future__ import annotations

import os
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform

pytestmark = pytest.mark.regression_bug

_BARE_SSP = "data/fsps_prsc_miles_chabrier.h5"
_CUE_WEIGHTS = "data/cue_weights.npz"
_SDSS = np.linspace(3800.0, 9200.0, 200)
_R = 2000.0
_SIGMA_V = 150.0
_Z = 0.1
_HALPHA = (6450.0, 6650.0)
_PARAMS = {"sfh_dpl_alpha": jnp.asarray(1.5), "dust_tau_v": jnp.asarray(0.5)}
_TWO_COMP_PARAMS = {
    "sfh_dpl_alpha": jnp.asarray(1.5),
    "dust_tau_bc": jnp.asarray(0.5),
    "dust_tau_diff": jnp.asarray(0.3),
}


@pytest.fixture(scope="module")
def ssp():
    from tengri import load_ssp_data

    if not os.path.isfile(_BARE_SSP):
        pytest.skip("bare-stellar SSP not available under data/")
    return load_ssp_data(_BARE_SSP)


def _require_cue():
    if not os.path.isfile(_CUE_WEIGHTS):
        pytest.skip("Cue weights not available under data/")


def _dust_block(kind):
    if kind == "single":
        return {
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Uniform(0.0, 1.5, "dust_tau_v"),
        }
    return {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_bc": Uniform(0.0, 2.0, "dust_tau_bc"),
        "tau_diff": Uniform(0.0, 1.5, "dust_tau_diff"),
    }


def _model(ssp, *, dust, cue=True, dust_emission=False):
    """Exact (``approx=None``) model on the conserving resample, R=2000, sigma_v=150.

    ``dust`` is ``"single"``, ``"two"`` or ``None`` (no dust).
    """
    obs = Observation(
        spectroscopy=Spectroscopy(
            resample="conserving", wave_obs=jnp.asarray(_SDSS), resolution=_R
        )
    )
    extra = {"neb": {"type": "cue", "all_params": Fixed(DEFAULT)} if cue else {"type": "none"}}
    if dust_emission:
        extra["dust_emission"] = {"type": "modified_blackbody", "all_params": Fixed(DEFAULT)}
    if dust is not None:
        extra["dust_attenuation"] = _dust_block(dust)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": "dpl",
                "alpha": Uniform(0.1, 5.0, "sfh_dpl_alpha"),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(_Z),
            **extra,
        )
        spec = base.spec.merge_observation_params(sigma_v_kms=Fixed(_SIGMA_V))
        return type(base)(spec, ssp, observation=obs, approx=None)


def _expected_spectrum(ssp, model, state, line_rest):
    """Exact projection of the pieces, with the projector the exact path uses per group.

    ``line_rest`` (the instrument-only light, already screened) goes through the
    instrument kernel alone; the rest of ``sed_intrinsic`` goes through the stellar
    kernel (sigma_v, instrument, and the library resolution curve).
    """
    from tengri.observation.spectrum import project_spectrum, resolve_sigma_lib_kms
    from tengri.utils.cosmology import luminosity_distance

    curve = type(model)._sigma_lib_curve_for(ssp)
    sigma_lib = resolve_sigma_lib_kms(
        jnp.asarray(_SDSS), _Z, model.observation.spectroscopy.sigma_lib_kms, curve
    )
    dl = jnp.asarray(luminosity_distance(jnp.asarray(_Z))).reshape(())
    continuum = jnp.asarray(state.sed_intrinsic) - line_rest
    line_part = project_spectrum(
        line_rest,
        state.wave,
        jnp.asarray(_SDSS),
        _Z,
        dl,
        resolution=_R,
        sigma_lib_kms=0.0,
        sigma_v_kms=0.0,
        conserving=True,
    )
    cont_part = project_spectrum(
        continuum,
        state.wave,
        jnp.asarray(_SDSS),
        _Z,
        dl,
        resolution=_R,
        sigma_lib_kms=sigma_lib,
        sigma_v_kms=_SIGMA_V,
        conserving=True,
    )
    return np.asarray(line_part + cont_part)


def test_single_screen_republishes_attenuated_nebular_light(ssp):
    """The published nebular light is T(lambda) times the unscreened Cue light."""
    _require_cue()
    state = _model(ssp, dust="single").predict_state(_PARAMS)
    bare = _model(ssp, dust=None).predict_state(_PARAMS)
    transmission = np.asarray(state.derived["dust_attenuation_factor"])
    np.testing.assert_allclose(
        np.asarray(state.derived["sed_nebular"]),
        transmission * np.asarray(bare.derived["sed_nebular"]),
        rtol=1e-6,
        atol=0.0,
    )


def test_sed_intrinsic_is_the_screened_total(ssp):
    """The screen still attenuates the whole summed SED; the split moves no light."""
    _require_cue()
    state = _model(ssp, dust="single").predict_state(_PARAMS)
    bare = _model(ssp, dust=None).predict_state(_PARAMS)
    transmission = np.asarray(state.derived["dust_attenuation_factor"])
    np.testing.assert_allclose(
        np.asarray(state.sed_intrinsic),
        transmission * np.asarray(bare.sed_intrinsic),
        rtol=1e-6,
        atol=0.0,
    )


def test_energy_balance_reads_the_summed_sed(ssp):
    """L_absorbed is the bolometric integral of (sed_intrinsic - sed_attenuated)."""
    from tengri.components.lyc import LYMAN_LIMIT_AA
    from tengri.forward.energy_balance import bolometric_absorbed_log10
    from tengri.utils.physics_constants import C_AA

    model = _model(ssp, dust="single", cue=False, dust_emission=True)
    state = model.predict_state(_PARAMS)
    wave = jnp.asarray(state.wave)
    log_abs, _ = bolometric_absorbed_log10(
        jnp.asarray(state.sed_intrinsic),
        jnp.asarray(state.sed_attenuated),
        C_AA / wave,
        wave=wave,
        lyman_cutoff_aa=LYMAN_LIMIT_AA,
    )
    got = float(np.asarray(state.derived["log_L_absorbed"]))
    # float32 model arithmetic: log10 L near 43 carries ~1e-6 dex of rounding.
    assert abs(got - float(np.asarray(log_abs))) <= 1e-6


def test_halpha_projection_has_no_broadened_negative_line(ssp):
    """Around Halpha the spectrum is T*line through the instrument kernel plus the rest.

    The stellar-kernel group carries the attenuated continuum only, so no sigma_v-broadened
    negative copy of the line remains.
    """
    _require_cue()
    model = _model(ssp, dust="single")
    state = model.predict_state(_PARAMS)
    bare = _model(ssp, dust=None).predict_state(_PARAMS)
    transmission = jnp.asarray(state.derived["dust_attenuation_factor"])
    line = transmission * jnp.asarray(bare.derived["sed_nebular"])
    spec = np.asarray(model.predict_observables(_PARAMS).spec_fnu)
    expected = _expected_spectrum(ssp, model, state, line)
    peak = float(np.max(np.abs(spec)))
    assert np.max(np.abs(spec - expected)) <= 1e-9 * peak
    lo, hi = _HALPHA
    window = np.logical_and(np.greater(_SDSS, lo), np.less(_SDSS, hi))
    assert np.max(np.abs(spec - expected)[window]) <= 1e-9 * peak


def test_two_component_split_uses_its_republished_nebular_light(ssp):
    """Two-component dust already republished reddened nebular light; the split agrees."""
    _require_cue()
    model = _model(ssp, dust="two")
    state = model.predict_state(_TWO_COMP_PARAMS)
    line = jnp.asarray(state.derived["sed_nebular"])
    spec = np.asarray(model.predict_observables(_TWO_COMP_PARAMS).spec_fnu)
    expected = _expected_spectrum(ssp, model, state, line)
    peak = float(np.max(np.abs(spec)))
    assert np.max(np.abs(spec - expected)) <= 1e-9 * peak


def _two_component_shock_model(ssp, *, dust):
    """Exact model with a shock component, two-component dust or none, R=2000, sigma_v=150."""
    obs = Observation(
        spectroscopy=Spectroscopy(
            resample="conserving", wave_obs=jnp.asarray(_SDSS), resolution=_R
        )
    )
    extra = {"neb": {"type": "none"}, "shock": {"frac": Fixed(0.5), "all_params": Fixed(DEFAULT)}}
    if dust is not None:
        extra["dust_attenuation"] = _dust_block(dust)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": "dpl",
                "alpha": Uniform(0.1, 5.0, "sfh_dpl_alpha"),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(_Z),
            **extra,
        )
        spec = base.spec.merge_observation_params(sigma_v_kms=Fixed(_SIGMA_V))
        return type(base)(spec, ssp, observation=obs, approx=None)


def test_two_component_republishes_attenuated_shock_light(ssp):
    """The published shock light is the screened shock light: below the intrinsic, positive."""
    state = _two_component_shock_model(ssp, dust="two").predict_state(_TWO_COMP_PARAMS)
    bare = _two_component_shock_model(ssp, dust=None).predict_state(_PARAMS)
    intrinsic = np.asarray(bare.derived["sed_shock"])
    published = np.asarray(state.derived["sed_shock"])
    assert np.any(intrinsic > 0.0)
    ratio = published[intrinsic > 0.0] / intrinsic[intrinsic > 0.0]
    assert np.all(ratio <= 1.0 + 1e-12)
    assert np.all(ratio > 0.0)
    assert ratio.min() < 1.0 - 1e-3
