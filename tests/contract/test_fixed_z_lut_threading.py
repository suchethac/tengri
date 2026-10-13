# SPDX-License-Identifier: BSD-3-Clause
"""Fixed-z stellar LUTs ride ``template_data`` as runtime inputs (#2812).

With ``redshift=Fixed(z)`` and ``approx=WavePrecomp()`` / ``SpectrumPrecomp()``
the per-redshift SSP x filter / SSP x pixel tables used to be read off the
component closure, so they were baked into the compiled HLO as constants and
every distinct z compiled from scratch. They are now threaded through the
``"stellar"`` namespace of ``_template_data_for_jit()``. The contract pinned here:

* two fixed-z models lower to identical HLO (one compile serves every z);
* the threaded LUT gives bit-identical outputs to the closure path;
* all redshift dependence flows through the jit arguments (a z=0.3 kernel fed
  z=0.7 arguments reproduces the z=0.7 model).
"""

from __future__ import annotations

import pathlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Parameters, SEDModel, SpectrumPrecomp, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry, Spectroscopy
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.contract

_SSP_WNEREF = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp_wneref():
    if not _SSP_WNEREF.exists():
        pytest.skip(f"wNE SSP not available at {_SSP_WNEREF}")
    return load_ssp_data(str(_SSP_WNEREF))


@pytest.fixture(scope="module")
def phot_obs():
    return Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )


def _spec(z: float) -> Parameters:
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Uniform(8.0, 12.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        redshift=Fixed(z),
    )


def _silent_build(spec, ssp, obs, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, **kwargs)


def _lowered_text(model, field):
    jit_fn = model._get_or_build_predict_observables_jit()
    fv = model.spec.get_fixed_values()
    thr = model._resolve_threaded_data(None, None, None)
    p = model.spec.sample(jax.random.PRNGKey(0))
    loss = lambda q, fv, s, t, zt: jnp.sum(jnp.abs(getattr(jit_fn(q, fv, s, t, zt), field)))  # noqa: E731
    return jax.jit(jax.value_and_grad(loss)).lower(p, fv, *thr).as_text()


def test_fixed_z_photometry_kernels_lower_identically_across_redshift(ssp_wneref, phot_obs):
    """Two fixed redshifts with WavePrecomp lower to the same HLO text."""
    text_a = _lowered_text(
        _silent_build(_spec(0.3), ssp_wneref, phot_obs, approx=WavePrecomp()), "phot_fnu"
    )
    text_b = _lowered_text(
        _silent_build(_spec(0.7), ssp_wneref, phot_obs, approx=WavePrecomp()), "phot_fnu"
    )
    assert text_a == text_b, (
        f"fixed-z photometry kernels differ across z (len {len(text_a)} vs {len(text_b)}); "
        f"the stellar fixed-z LUT is still baked in as a constant"
    )


def test_fixed_z_spectrum_kernels_lower_identically_across_redshift(ssp_wneref):
    """Two fixed redshifts with SpectrumPrecomp lower to the same HLO text."""
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=np.linspace(3800.0, 9000.0, 400), resample="point")
    )
    text_a = _lowered_text(
        _silent_build(_spec(0.3), ssp_wneref, obs, approx=SpectrumPrecomp()), "spec_fnu"
    )
    text_b = _lowered_text(
        _silent_build(_spec(0.7), ssp_wneref, obs, approx=SpectrumPrecomp()), "spec_fnu"
    )
    assert text_a == text_b, (
        f"fixed-z spectrum kernels differ across z (len {len(text_a)} vs {len(text_b)}); "
        f"the stellar fixed-z LUT is still baked in as a constant"
    )


def test_threaded_lut_gives_the_closure_values_bit_for_bit(ssp_wneref, phot_obs):
    """Threaded LUT == closure LUT bit-for-bit; a foreign-z kernel reproduces its own z."""
    model = _silent_build(_spec(0.3), ssp_wneref, phot_obs, approx=WavePrecomp())
    ssp, tmpl, zt = model._resolve_threaded_data(None, None, None)
    assert "stellar" in tmpl, "fixed-z stellar LUT must be threaded under 'stellar'"

    jit_fn = model._get_or_build_predict_observables_jit()
    fv = model.spec.get_fixed_values()
    p = model.spec.sample(jax.random.PRNGKey(0))

    out_threaded = jit_fn(p, fv, ssp, tmpl, zt).phot_fnu
    tmpl_closure = {k: v for k, v in tmpl.items() if k != "stellar"}
    out_closure = jit_fn(p, fv, ssp, tmpl_closure, zt).phot_fnu
    np.testing.assert_array_equal(np.asarray(out_threaded), np.asarray(out_closure))

    # Guard: z dependence must flow through the arguments. A z=0.3 kernel fed the
    # z=0.7 model's template data and fixed values must reproduce the z=0.7 model.
    model_b = _silent_build(_spec(0.7), ssp_wneref, phot_obs, approx=WavePrecomp())
    ssp_b, tmpl_b, zt_b = model_b._resolve_threaded_data(None, None, None)
    fv_b = model_b.spec.get_fixed_values()
    own_b = model_b._get_or_build_predict_observables_jit()(p, fv_b, ssp_b, tmpl_b, zt_b).phot_fnu
    foreign = jit_fn(p, fv_b, ssp_b, tmpl_b, zt_b).phot_fnu
    np.testing.assert_allclose(np.asarray(foreign), np.asarray(own_b), rtol=1e-12, atol=0)
