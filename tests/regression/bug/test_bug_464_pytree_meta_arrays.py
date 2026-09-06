# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #464: PyTree metadata collision across SEDModels.

Two SEDModels built in the same process previously crashed on the second
``predict_photometry`` call when both used the Cue nebular backend. The
JIT pytree-equality check raised ``ValueError`` because
``CueWeights.line_wav_selections`` (a tuple of int arrays) was stored as
pytree aux_data — arrays in aux trigger ``__bool__`` on equality lookup.

Fix: move ``line_wav_selections`` from aux_data to children in
``_cue_weights_flatten`` / ``_cue_weights_unflatten``.
"""

import pytest

pytestmark = pytest.mark.regression_bug


def test_two_cue_models_predict_photometry_does_not_crash():
    """Two SEDModels sharing an Observation must both predict_photometry."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import numpy as np

    import tengri
    from tengri import Fixed, Observation, Photometry, SEDModel

    try:
        ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
    except (FileNotFoundError, OSError):
        pytest.skip("fsps_prsc_miles_chabrier SSP not available")

    try:
        obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))
        mA = SEDModel.build(
            ssp,
            observation=obs,
            sfh={
                "type": "dexp",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_gyr": 0.3,
                "log_total_mass": 10.0,
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_bc": 2.0,
                "tau_diff": 0.5,
                "slope": -0.7,
            },
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
            redshift=Fixed(0.5),
            igm={"type": "inoue"},
        )
        mB = SEDModel.build(
            ssp,
            observation=obs,
            sfh={
                "type": "dexp",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_gyr": 8.0,
                "log_total_mass": 10.0,
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "tau_bc": 0.1,
                "tau_diff": 0.05,
                "slope": -0.7,
            },
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
            redshift=Fixed(1.0),
            igm={"type": "inoue"},
        )
    except (FileNotFoundError, OSError):
        pytest.skip("required data files (e.g. cue_weights.npz) not available")

    pA = dict(mA.spec.sample(jax.random.PRNGKey(0)))
    pB = dict(mB.spec.sample(jax.random.PRNGKey(1)))

    # Both calls must succeed; pre-fix, the second one raised
    # "Exception raised while checking equality of metadata fields of pytree".
    fluxA = np.asarray(mA.predict_photometry(pA))
    fluxB = np.asarray(mB.predict_photometry(pB))

    assert np.all(np.isfinite(fluxA))
    assert np.all(np.isfinite(fluxB))
    assert fluxA.shape == (2,)
    assert fluxB.shape == (2,)


def test_cue_weights_aux_is_hashable():
    """CueWeights pytree aux_data must be hashable — no arrays."""
    pytest.importorskip("jax")
    try:
        from tengri.components.nebular.cue import load_cue_weights
    except ImportError:
        pytest.skip("cue module not importable")

    try:
        cw = load_cue_weights("data/cue_weights.npz")
    except (FileNotFoundError, OSError):
        pytest.skip("cue_weights.npz not available")

    import jax

    _, aux = jax.tree_util.tree_flatten(cw)
    # tree_flatten returns (leaves, treedef); treedef contains the aux.
    # Verify equality of two treedefs (the operation that previously failed).
    _, aux2 = jax.tree_util.tree_flatten(cw)
    assert aux == aux2  # must not raise
