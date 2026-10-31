# SPDX-License-Identifier: BSD-3-Clause
"""``Spectroscopy(nebular_lines=False)`` on the batched spectroscopic path.

The batched forward builds its state through ``predict_state(observables_only=True)``,
which takes the cached chain's publication shortcuts, and then projects the spectrum
with the ``nebular_lines`` flag of the template. These tests pin three things on a
Cue template with ``publish_line_fraction``: the batched spectrum equals the
single-galaxy ``predict_spectrum`` for both flag values; the flag changes no
energy-balance or line-luminosity key of the state; and the flag removes exactly the
projected line light from the spectrum while the photometry stays bit-identical.
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
    Uniform,
    load_ssp_data,
)
from tengri.cosmology import luminosity_distance
from tengri.forward.batched_spectrum import make_batched_predict
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.observation.spectrum import project_spectrum

pytestmark = pytest.mark.contract

_SSP = "data/fsps_prsc_miles_chabrier.h5"
_CUE = "data/cue_weights.npz"
_Z = 0.1
_WAVE = np.linspace(3600.0, 9000.0, 600)  # [Angstrom]
_ENERGY_KEYS = (
    "line_lums",
    "log_line_lums",
    "log_line_lums_attenuated",
    "sed_nebular",
    "L_ir",
    "L_absorbed",
    "log_L_ir",
    "log_L_absorbed",
    "L_age",
    "log_L_age",
    "log_L_lyc",
    "log_L_lyc_age",
    "log_L_lyc_dust",
)
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


def _template(ssp, nebular_lines):
    """Cue template with free redshift, two-component dust and a batched-capable grid."""
    obs = Observation(
        photometry=Photometry.from_names(["des_g", "des_r", "des_z"]),
        spectroscopy=Spectroscopy(
            wave_obs=jnp.asarray(_WAVE), resolution=3000.0, nebular_lines=nebular_lines
        ),
    )
    neb = {"type": "cue", "all_params": Fixed(DEFAULT), "publish_line_fraction": True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
            neb=neb,
            redshift=Uniform(0.0, 0.5),
        )


def _params(model):
    values = {**_PARAMS, "redshift": _Z}
    return {k: jnp.asarray(values[k]) for k in model.spec.free_params}


def _batched(model, p):
    """Batched spectrum and photometry of one galaxy whose grid is the template grid."""
    galaxy = GalaxySpectrum(
        wave=_WAVE, flux=np.ones(_WAVE.size), ivar=np.ones(_WAVE.size), z=_Z, resolution=3000.0
    )
    spec, batch, _ = build_spectro_batches([galaxy], model=model)[0]
    stacked = {k: jnp.broadcast_to(v, (1,)) for k, v in p.items()}
    out = make_batched_predict(model, spec)(stacked, batch)
    return np.asarray(out["spec_fnu"])[0], np.asarray(out["phot_fnu"])[0]


def _state(model, p):
    fixed = {**dict(model.spec.get_fixed_values()), "redshift": _Z}
    free = {k: v for k, v in p.items() if k != "redshift"}
    return model.predict_state(free, fixed_values=fixed, observables_only=True)


@pytest.mark.parametrize("nebular_lines", [False, True])
def test_batched_spectrum_equals_single_galaxy(ssp, nebular_lines):
    """(a) The batched spectrum of the template's own grid matches ``predict_spectrum``."""
    model = _template(ssp, nebular_lines)
    p = _params(model)
    spec_batched, _ = _batched(model, p)
    spec_single = np.asarray(model.predict_spectrum(p))
    np.testing.assert_allclose(spec_batched, spec_single, rtol=1e-10, atol=0.0)


def test_flag_leaves_energy_balance_and_line_keys_bit_identical(ssp):
    """(b) The flag changes the spectrum only: every line and energy-balance key is unchanged."""
    m_on = _template(ssp, True)
    m_off = _template(ssp, False)
    p = _params(m_on)
    s_on = _state(m_on, p)
    s_off = _state(m_off, p)
    checked = 0
    for key in _ENERGY_KEYS:
        if key not in s_on.derived:
            continue
        assert key in s_off.derived, key
        a = jnp.asarray(s_on.derived[key])
        b = jnp.asarray(s_off.derived[key])
        assert jnp.array_equal(a, b), key
        checked += 1
    assert checked >= 8
    assert float(jnp.sum(jnp.asarray(s_on.derived["line_lums"]))) > 0.0
    assert float(jnp.sum(jnp.asarray(s_off.derived["sed_nebular"]))) > 0.0


def test_flag_removes_projected_line_light_only(ssp):
    """(c) Photometry is bit-identical; the spectrum drops exactly the projected line SED."""
    m_on = _template(ssp, True)
    m_off = _template(ssp, False)
    p = _params(m_on)
    spec_on, phot_on = _batched(m_on, p)
    spec_off, phot_off = _batched(m_off, p)
    np.testing.assert_array_equal(phot_on, phot_off)

    removed = spec_on - spec_off
    assert np.max(np.abs(removed)) > 0.0

    state = _state(m_on, p)
    lines_rest = jnp.asarray(state.derived["sed_nebular"]) * jnp.asarray(
        state.derived["nebular_line_frac"]
    )
    spectro = m_on.observation.spectroscopy
    expected = np.asarray(
        project_spectrum(
            lines_rest,
            state.wave,
            spectro.wave_obs,
            _Z,
            jnp.asarray(luminosity_distance(_Z)).reshape(()),
            resolution=spectro.resolution,
            sigma_lib_kms=0.0,
            n_bins=spectro.lsf_n_bins,
            sigma_v_kms=0.0,
            cal_coeffs=None,
            conserving=spectro.resolve_conserving(state.wave, _Z, wave_obs=spectro.wave_obs),
            has_lyc_edge=state.derived.get("lyc_transmission") is not None,
        )
    )
    np.testing.assert_allclose(removed, expected, rtol=1e-10, atol=1e-10 * np.max(np.abs(removed)))
