# SPDX-License-Identifier: BSD-3-Clause
"""Pixel-integrated spectrum LUT serves the flux-conserving resample (#3413).

With ``resample="conserving"`` the exact path broadens the model on its own grid
(velocity dispersion, then the instrument line-spread function) and takes the mean
over each pixel's edges last. ``SpectrumPrecomp()`` must reproduce that per pixel,
for the flux and for its Jacobian with respect to the free parameters.

The metric is the largest per-pixel relative error. The denominator is floored at
1e-6 of the spectrum peak (flux) or of the column peak (Jacobian) so that pixels at
the edge of the model range, where both values vanish, do not divide by zero.
"""

import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import SpectrumPrecomp

pytestmark = pytest.mark.regression_paper

_TOL = 1e-4
_FLOOR = 1e-6

_BARE_SSP = "data/fsps_prsc_miles_chabrier.h5"
_CUE_WEIGHTS = "data/cue_weights.npz"

_SDSS = np.linspace(3800.0, 9200.0, 200)  # 27 Angstrom pixels
_DESI = np.linspace(6000.0, 6048.0, 61)  # 0.8 Angstrom pixels
_PRISM = np.geomspace(6000.0, 9000.0, 40)  # constant R_pix ~ 100

_TWO_COMP_FREE = ("sfh_dpl_alpha", "dust_tau_bc", "dust_tau_diff")
_SINGLE_FREE = ("sfh_dpl_alpha", "dust_tau_v")


def _dust(kind):
    from tengri import DEFAULT, Fixed, Uniform

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


def _model(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue, approx):
    """Exact (``approx=None``) or LUT model on the conserving resample."""
    from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform

    obs = Observation(
        spectroscopy=Spectroscopy(
            resample="conserving", wave_obs=jnp.asarray(wave_obs), resolution=resolution
        )
    )
    neb = {"type": "cue", "all_params": Fixed(DEFAULT)} if with_cue else {"type": "none"}
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
            dust_attenuation=_dust(dust_kind),
            neb=neb,
            redshift=Fixed(0.1),
        )
        spec = base.spec.merge_observation_params(sigma_v_kms=Fixed(float(sigma_v)))
        return type(base)(spec, ssp, observation=obs, approx=approx)


def _pixel_rel(a, b):
    """Largest per-pixel relative error, the denominator floored at a fraction of the peak."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    floor = _FLOOR * np.max(np.abs(b), axis=0, keepdims=True)
    denom = np.maximum(np.abs(b), floor)
    return float(np.max(np.abs(a - b) / np.where(denom > 0, denom, 1.0)))


def _flux_and_jacobian(model, names, base):
    def f(theta):
        p = dict(base)
        for i, name in enumerate(names):
            p[name] = theta[i]
        return model.predict_observables(p).spec_fnu

    theta0 = jnp.asarray([base[n] for n in names])
    return np.asarray(f(theta0)), np.asarray(jax.jacfwd(f)(theta0))


def pair_errors(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue):
    """Return ``(flux_err, jacobian_err)`` of the LUT against the exact conserving path."""
    names = _SINGLE_FREE if dust_kind == "single" else _TWO_COMP_FREE
    exact = _model(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue, None)
    lut = _model(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue, SpectrumPrecomp())
    base = {
        "sfh_dpl_alpha": jnp.asarray(1.5),
        "dust_tau_bc": jnp.asarray(0.5),
        "dust_tau_diff": jnp.asarray(0.3),
        "dust_tau_v": jnp.asarray(0.5),
    }
    f_e, j_e = _flux_and_jacobian(exact, names, base)
    f_l, j_l = _flux_and_jacobian(lut, names, base)
    return _pixel_rel(f_l, f_e), _pixel_rel(j_l, j_e)


_CASES = [
    pytest.param(
        _SDSS, None, 0.0, "two", False, id="sdss-2comp-nolines-sv0"
    ),
    pytest.param(
        _SDSS, 2000.0, 150.0, "two", False, id="sdss-2comp-R2000-sv150"
    ),
    pytest.param(
        _SDSS, 2000.0, 150.0, "single", True, id="sdss-1screen-cue-R2000-sv150"
    ),
    pytest.param(
        _DESI, 3000.0, 150.0, "two", False, id="desi-2comp-R3000-sv150"
    ),
    pytest.param(
        _PRISM, 100.0, 0.0, "two", False, id="prism-2comp-R100-sv0"
    ),
]


def _require_data(with_cue):
    if not os.path.isfile(_BARE_SSP):
        pytest.skip("bare-stellar SSP not available under data/")
    if with_cue and not os.path.isfile(_CUE_WEIGHTS):
        pytest.skip("Cue weights not available under data/")


@pytest.mark.parametrize(
    ("wave_obs", "resolution", "sigma_v", "dust_kind", "with_cue"), _CASES
)
def test_pixel_integral_lut_matches_conserving_exact(
    wave_obs, resolution, sigma_v, dust_kind, with_cue
):
    """SpectrumPrecomp under resample='conserving' matches the exact conserving path."""
    from tengri import load_ssp_data

    _require_data(with_cue)
    ssp = load_ssp_data(_BARE_SSP)
    if resolution is not None and with_cue:
        # A nebular family under a fixed instrument kernel has no per-draw kernel: refused.
        with pytest.raises(ValueError, match="instrument resolution"):
            _model(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue, SpectrumPrecomp())
        return
    flux_err, jac_err = pair_errors(ssp, wave_obs, resolution, sigma_v, dust_kind, with_cue)
    print(f"\nflux max rel err = {flux_err:.3e}   gradient max rel err = {jac_err:.3e}")
    assert flux_err <= _TOL, f"flux max per-pixel rel err {flux_err:.3e} > {_TOL:g}"
    assert jac_err <= _TOL, f"gradient max per-pixel rel err {jac_err:.3e} > {_TOL:g}"
