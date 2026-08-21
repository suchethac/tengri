# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #444: tracer leak in preintegrate_grid.

Building two ``SEDModel`` instances on the same ``SSPData`` and fitting
each in turn used to leak a ``DynamicJaxprTracer`` from the first
model's compiled graph into the second's. The leak was traced to
``preintegrate_grid`` constructing the geometric flux scale via the
JIT-compiled ``lnu_to_fnu`` helper, which under a surrounding trace
caused ``float(traced)`` to fail. Fixed in 308742eb by inlining the
flux-scale math so it returns a Python float for concrete inputs.

This test pins the multi-fit pattern (variant SFH comparison) that
the issue reporter was attempting in their gallery example.
"""

import pytest

pytestmark = pytest.mark.regression_bug


def test_three_sfh_variants_fit_in_one_process():
    """Build mock, then MAP-fit three SFH variants on the same SSP."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import tengri

    try:
        ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
    except (FileNotFoundError, OSError):
        pytest.skip("fsps_prsc_miles_chabrier SSP not available")

    obs = tengri.Observation(
        photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    mock_model = tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "tsnorm", "all_params": tengri.FIXED, "log_total_mass": 10.0},
        redshift=tengri.Fixed(0.05),
        approx=tengri.WavePrecomp(n_z=20),
    )
    truth = dict(mock_model.spec.sample(jax.random.PRNGKey(0)))
    mock = mock_model.mock(truth, snr=20.0, key=jax.random.PRNGKey(1))

    # Touch predict_sfh on the truth model before the fits to exercise the
    # original leak path documented in the issue.
    mock_model.predict_sfh(truth)

    for sfh_kind in ("continuity", "dirichlet", "dense_basis"):
        model = tengri.SEDModel.build(
            ssp,
            observation=obs,
            sfh={"type": sfh_kind, "all_params": tengri.FREE},
            redshift=tengri.Fixed(0.05),
            approx=tengri.WavePrecomp(n_z=20),
        )
        forward = tengri.ForwardModel.build(sed=model, observation=obs)
        # 10 steps is enough to traverse the JIT trace + a few optimizer
        # iterations. We only assert the run finishes; no parameter
        # recovery claim is made.
        forward.fit(
            mock.flux_obs,
            mock.noise,
            method="map",
            optimizer="adam",
            n_steps=10,
            verbose=False,
        )

    mock_model.predict_sfh(truth)  # post-fit predict_sfh path
