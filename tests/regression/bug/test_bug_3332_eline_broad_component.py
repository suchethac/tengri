# SPDX-License-Identifier: BSD-3-Clause
"""#3332: the emission-line broad component was sampled but never evaluated.

``eline_broad=True`` registers ``eline_broad_sigma_kms`` and the marginalized /
fitted line likelihood never builds broad design-matrix columns, so the
parameter changes the likelihood by exactly zero and its posterior is its
prior. Wiring the broad columns needs a design decision the issue leaves open
(which lines get a broad column, how their amplitudes and priors are indexed),
so enabling the component refuses with a clear error instead of running inert.
"""

from __future__ import annotations

import types

import jax.numpy as jnp
import pytest

from tengri.inference.fitter import Fitter
from tengri.observation.line_list import LineList
from tengri.observation.spectroscopy import Spectroscopy
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.regression_bug


def _wave(n: int = 300) -> jnp.ndarray:
    return jnp.linspace(4000.0, 7000.0, n)


def _spec(eline_broad: bool):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.1),
            dust_tau_diff=Fixed(0.1),
            redshift=Fixed(0.1),
            eline_broad=eline_broad,
        )


def _model(wave, spec, spec_cfg):
    continuum = jnp.ones(len(wave)) * 10.0
    return types.SimpleNamespace(
        spec=spec,
        wave_obs=wave,
        _spectral_resolution=2000.0,
        predict_spectrum=lambda params, w=None, **kwargs: continuum,
        observation=types.SimpleNamespace(spectroscopy=spec_cfg, photometry=None),
    )


def _fit(eline_broad_spec: bool, eline_broad_params: bool):
    wave = _wave()
    cfg = Spectroscopy(
        wave_obs=wave,
        resolution=2000.0,
        eline_mode="marginalized",
        eline_catalog=LineList.default_13(),
        eline_broad=eline_broad_spec,
    )
    model = _model(wave, _spec(eline_broad_params), cfg)
    return Fitter(
        model, jnp.ones(len(wave)) * 10.0, jnp.ones(len(wave)) * 0.1, data_type="spectroscopy"
    )


@pytest.mark.parametrize(
    ("spec_flag", "params_flag"), [(True, True), (True, False), (False, True)]
)
def test_enabling_broad_component_refuses(spec_flag, params_flag):
    """A broad component that no likelihood evaluates must not be accepted silently."""
    with pytest.raises(ValueError, match="eline_broad"):
        _fit(spec_flag, params_flag)


def test_broad_off_still_builds():
    """With the broad component off the fitter builds as before."""
    fitter = _fit(False, False)
    assert fitter._eline_marginalize is True
