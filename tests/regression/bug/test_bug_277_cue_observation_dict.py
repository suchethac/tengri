# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #277: Cue + Observation → 'dict has no batched_param_shifts'.

`SEDModel._template_data_for_jit` wraps the nebular weights bundle in a
namespaced dict (``{"nebular": <bundle>, ...}``) so the JIT closure can
thread heterogeneous template payloads. `NebularSEDComponent.apply` must
unwrap the nebular slot before passing it to the Cue backend; otherwise
the backend hits ``AttributeError: 'dict' object has no attribute
batched_param_shifts``.

See PR #306.
"""

import warnings

import jax
import pytest

pytestmark = pytest.mark.regression_bug


def test_cue_predict_paths_with_observation_attached():
    """Cue + Observation must not raise on any predict_* path."""
    pytest.importorskip("tengri")
    import tengri

    try:
        ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
    except FileNotFoundError:
        pytest.skip("bare-stellar SSP not available")

    obs = tengri.Observation(photometry=tengri.Photometry.from_names(["jwst_f277w", "jwst_f356w"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={
                "type": "dpl",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_gyr": 0.1,
                "log_total_mass": 1.5,
                "alpha": 4,
                "beta": 2,
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_diff": 0.05,
                "tau_bc": 0.1,
            },
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
            redshift=tengri.Fixed(3.0),
        )

    params = dict(model.spec.sample(jax.random.PRNGKey(0)))

    # Pre-fix: each of these raised AttributeError inside cue.predict_all_lines.
    sed = model.predict_rest_sed(params)
    assert sed.sed.shape[0] > 0
    fluxes = model.predict_photometry(params)
    assert fluxes.shape == (2,)
