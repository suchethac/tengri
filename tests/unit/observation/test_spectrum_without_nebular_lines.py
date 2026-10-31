# SPDX-License-Identifier: BSD-3-Clause
"""``Spectroscopy(nebular_lines=False)``: the Cue nebular lines leave the spectrum only.

A spectral fit that models emission lines itself (free-amplitude line columns in
the likelihood) needs the model spectrum without the model's own lines, while the
photometry keeps them. The Cue component publishes the line share of its SED,
``nebular_line_frac = lines / (continuum + lines)``, when built with
``neb={"type": "cue", "publish_line_fraction": True}``. The dust components that
republish ``sed_nebular`` reddened (``two_component``, ``age_binned``) multiply it
by one transmission T(lambda), so ``frac * sed_nebular`` is the attenuated line
SED for any law, and the spectrum drops it from the instrument-only group.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    load_ssp_data,
)
from tengri.observation.spectrum import project_spectrum

pytestmark = pytest.mark.contract

_SSP = "data/fsps_prsc_miles_chabrier.h5"
_CUE = "data/cue_weights.npz"
_Z = 0.05
_HALPHA_REST = 6564.61  # vacuum [Angstrom]
_WAVE_OBS = np.arange(3600.0, 9800.0, 0.8)  # [Angstrom]

# A dusty, star-forming draw so that T(lambda) < 1 matters and the lines are strong.
_PARAMS = {
    "sfh_dpl_log_total_mass": 9.0,
    "sfh_dpl_tau_gyr": 2.0,
    "sfh_dpl_alpha": 2.0,
    "sfh_dpl_beta": 2.0,
    "sfh_dpl_age_frac": 0.7,
    "dust_tau_bc": 0.6,
    "dust_tau_diff": 0.3,
}


@pytest.fixture(scope="module")
def ssp():
    if not Path(_SSP).is_file():
        pytest.skip(f"missing SSP grid {_SSP}")
    if not Path(_CUE).is_file():
        pytest.skip(f"Cue weights ({_CUE}) not present")
    return load_ssp_data(_SSP)


def _cue_model(
    ssp,
    *,
    nebular_lines,
    publish_line_fraction,
    dust="two_component",
    neb_extra=None,
    igm=None,
    approx=None,
):
    """Cue nebular model at z = 0.05 with an R = 3000 spectrum on 3600-9800 A and grz bands."""
    obs = Observation(
        photometry=Photometry.from_names(["des_g", "des_r", "des_z"]),
        spectroscopy=Spectroscopy(
            wave_obs=jnp.asarray(_WAVE_OBS), resolution=3000.0, nebular_lines=nebular_lines
        ),
    )
    neb = {"type": "cue", "all_params": Fixed(DEFAULT), **(neb_extra or {})}
    if publish_line_fraction:
        neb["publish_line_fraction"] = True
    dust_group = {"type": dust, "law": "calzetti", "all_params": FREE}
    extra = {} if igm is None else {"igm": {"type": igm}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation=dust_group,
            neb=neb,
            redshift=Fixed(_Z),
            approx=approx,
            **extra,
        )


def _params(model):
    return {k: jnp.asarray(_PARAMS[k]) for k in model.spec.free_params}


def _predict(ssp, nebular_lines, publish, igm=None):
    model = _cue_model(ssp, nebular_lines=nebular_lines, publish_line_fraction=publish, igm=igm)
    return model, model.predict_observables(_params(model))


def test_spectroscopy_default_keeps_lines():
    spec = Spectroscopy(wave_obs=jnp.asarray(_WAVE_OBS), resolution=3000.0)
    assert spec.nebular_lines is True


def test_default_bit_identical(ssp):
    """Publishing the fraction alone changes no observable, bit for bit."""
    _, a = _predict(ssp, True, False)
    _, b = _predict(ssp, True, True)
    np.testing.assert_array_equal(np.asarray(a.spec_fnu), np.asarray(b.spec_fnu))
    np.testing.assert_array_equal(np.asarray(a.phot_fnu), np.asarray(b.phot_fnu))


def test_split_reproduces_mix_dig_emission(ssp):
    """With the flag on, ``sed_nebular`` (continuum + rendered lines from the split)
    matches the ``mix_dig_emission`` SED of the default path; the fraction is in [0, 1]."""
    m_off = _cue_model(ssp, nebular_lines=True, publish_line_fraction=False)
    m_on = _cue_model(ssp, nebular_lines=True, publish_line_fraction=True)
    s_off = m_off.predict_state(_params(m_off))
    s_on = m_on.predict_state(_params(m_on))
    assert s_off.derived.get("nebular_line_frac") is None
    # predict_state's sed_nebular is after the dust republish; compare the
    # nebular SED itself through the attenuated copy, the same T on both sides.
    neb_off = np.asarray(s_off.derived["sed_nebular"])
    neb_on = np.asarray(s_on.derived["sed_nebular"])
    scale = np.max(np.abs(neb_off))
    assert scale > 0
    assert np.max(np.abs(neb_on - neb_off)) <= 1e-10 * scale
    frac = np.asarray(s_on.derived["nebular_line_frac"])
    assert frac.shape == neb_on.shape
    assert np.all(frac >= 0.0) and np.all(frac <= 1.0)
    wave = np.asarray(s_on.wave)
    at_ha = np.argmin(np.abs(wave - _HALPHA_REST))
    assert frac[at_ha] > 0.5  # H-alpha dominates the nebular SED at its center


@pytest.mark.parametrize("igm", [None, "inoue14"])
def test_lines_removed_from_spectrum_only(ssp, igm):
    model, full = _predict(ssp, True, True, igm=igm)
    _, cont = _predict(ssp, False, True, igm=igm)
    np.testing.assert_array_equal(np.asarray(full.phot_fnu), np.asarray(cont.phot_fnu))
    wave = np.asarray(model.observation.spectroscopy.wave_obs)
    ha = np.abs(wave - _HALPHA_REST * (1 + _Z)) < 3.0
    far = (wave > 6000 * (1 + _Z)) & (wave < 6100 * (1 + _Z))
    f_full = np.asarray(full.spec_fnu)
    d = f_full - np.asarray(cont.spec_fnu)
    assert d[ha].max() > 0.1 * f_full[ha].max()  # H-alpha removed
    # Continuum unchanged. The floor is the LSF convolution's own numerical
    # leakage: the line-only projection below is ~6e-10 of the spectrum peak in
    # this window, where the rest-frame line SED is exactly zero.
    assert np.max(np.abs(d[far])) <= 1e-8 * np.max(np.abs(f_full))

    # The removed light is exactly T x rendered lines (times the IGM, when there is
    # one) through the instrument-only kernel: project it independently.
    state = model.predict_state(_params(model))
    igm_trans = state.derived.get("igm_transmission")
    assert (igm_trans is None) == (igm is None)
    spectro = model.observation.spectroscopy
    lines_rest = jnp.asarray(state.derived["sed_nebular"]) * jnp.asarray(
        state.derived["nebular_line_frac"]
    )
    if igm_trans is not None:
        lines_rest = lines_rest * jnp.asarray(igm_trans)
    expected = np.asarray(
        project_spectrum(
            lines_rest,
            state.wave,
            spectro.wave_obs,
            _Z,
            model._get_dl_cm(_params(model)),
            resolution=spectro.resolution,
            sigma_lib_kms=0.0,
            n_bins=spectro.lsf_n_bins,
            sigma_v_kms=0.0,
            cal_coeffs=None,
            conserving=spectro.resolve_conserving(state.wave, _Z, wave_obs=spectro.wave_obs),
            has_lyc_edge=state.derived.get("lyc_transmission") is not None,
        )
    )
    assert np.max(np.abs(d - expected)) <= 1e-10 * np.max(np.abs(d))


def test_predict_spectrum_drops_lines_too(ssp):
    """The eager ``predict_spectrum`` surface honors the flag like ``predict_observables``."""
    model = _cue_model(ssp, nebular_lines=False, publish_line_fraction=True)
    p = _params(model)
    a = np.asarray(model.predict_observables(p).spec_fnu)
    b = np.asarray(model.predict_spectrum(p))
    np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-10 * np.max(np.abs(a)))


def test_flag_requires_cue_without_dig(ssp):
    with pytest.raises(ValueError, match="nebular_lines=False"):
        _cue_model(ssp, nebular_lines=False, publish_line_fraction=False)


def test_flag_refuses_dust_that_keeps_sed_nebular_intrinsic(ssp):
    """single_component leaves ``derived['sed_nebular']`` unattenuated, so the
    fraction times it would not be the line light in the spectrum."""
    with pytest.raises(ValueError, match="two_component"):
        _cue_model(ssp, nebular_lines=False, publish_line_fraction=True, dust="single_component")


def test_publish_line_fraction_refuses_active_dig(ssp):
    with pytest.raises(ValueError, match="DIG inactive"):
        _cue_model(
            ssp,
            nebular_lines=True,
            publish_line_fraction=True,
            neb_extra={"dig_frac": Fixed(0.3)},
        )


def test_spectrum_lut_refuses_line_free_spectrum(ssp):
    from tengri import SpectrumPrecomp

    with pytest.raises(ValueError, match="SpectrumPrecomp"):
        _cue_model(ssp, nebular_lines=False, publish_line_fraction=True, approx=SpectrumPrecomp())
