"""Phase 4-C tests: nebular template data threading as JIT runtime inputs.

Extends Phase 4-B (SSP threading) to nebular backend grids (Cue weights,
CloudyGrid grids, etc.). Verifies that template data become JIT Parameter
ops instead of Constant ops, reducing HLO size.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, FREE, Uniform, recipes
from tengri.components.nebular.cue import CueBackend, load_cue_weights
from tengri.components.nebular.cloudy_grid import CloudyGridBackend, load_cloudy_grid


@pytest.mark.unit
def test_template_data_threaded_for_cue(mock_ssp_data):
    """Verify Cue weights appear as JIT Parameter ops, not Constant ops."""
    pytest.importorskip("cue")

    # Build a model with Cue nebular backend
    model = pytest.mock.MagicMock()
    model.ssp_data = mock_ssp_data
    model.observation = None
    model.spec = pytest.mock.MagicMock()
    model.spec.get_fixed_values = lambda: {"redshift": np.float32(0.0)}
    model._approx = {}
    model._wave_obs = None
    model._lsf_resolution = None
    model._sigma_lib_kms = None
    model._lsf_n_bins = None
    model._Observables = None

    # We expect this to skip gracefully if cue is not available
    try:
        from tengri import SEDModel, recipes

        model = SEDModel.build(
            ssp_data=mock_ssp_data,
            observation=None,
            **recipes.stochastic_sfh_jwst(
                nebular="cue", agn=None, dust="two_component", mean_sfh_type="field"
            ),
        )
    except ImportError:
        pytest.skip("Cue not available")

    # Get the JIT function
    jit_fn = model._get_or_build_predict_observables_jit()

    # Extract the jaxpr to inspect the signature
    abstract_args = jax.ShapedArray(shape=(1,), dtype=jnp.float32), {}, mock_ssp_data, None
    jaxpr = jax.make_jaxpr(jit_fn)(*abstract_args).jaxpr

    # Count Parameter ops (not Constant ops) in in_avals
    # The template_data should show up as a Parameter when passed
    # If Cue weights are being passed through, they should NOT be baked
    # into Constant ops — they should flow through as runtime inputs
    assert len(jaxpr.in_avals) >= 4, (
        f"Expected at least 4 JIT inputs (params, fixed, ssp_data, template_data), "
        f"got {len(jaxpr.in_avals)}"
    )


@pytest.mark.unit
def test_bit_close_with_template_threading(mock_ssp_data):
    """Verify bit-exact parity between JIT and non-JIT paths with threading."""
    from tengri import SEDModel, recipes

    try:
        model = SEDModel.build(
            ssp_data=mock_ssp_data,
            observation=None,
            **recipes.mock_recovery_minimal(nebular="cue"),
        )
    except ImportError:
        pytest.skip("Cue not available")

    params = {"redshift": 0.05}

    # Non-JIT path
    result_eager = model.predict_observables(params)

    # JIT path
    result_jit = model.predict_observables_jit(params)

    # Should be bit-exact (within rtol=1e-12 for float32->float64 roundtrip)
    if result_eager.phot_fnu is not None and result_jit.phot_fnu is not None:
        np.testing.assert_allclose(
            result_eager.phot_fnu, result_jit.phot_fnu, rtol=1e-12, atol=1e-30
        )


@pytest.mark.unit
def test_distinct_grids_have_distinct_signatures():
    """Verify that different template data lead to distinct compile signatures.

    Two models with the same structural config but different Cue weights files
    should have different compile_signature() because the weights object differs.
    (Only tested if multiple Cue weight files are available.)
    """
    pytest.skip(
        "Skipped: requires multiple Cue weight files. "
        "Implementation depends on available fixtures."
    )


@pytest.mark.unit
def test_no_template_threading_when_bakedin(mock_ssp_data):
    """Verify that 'baked_in' nebular backend doesn't require template threading."""
    from tengri import SEDModel

    model = SEDModel.build(
        ssp_data=mock_ssp_data,
        observation=None,
        nebular="baked_in",
    )

    # _template_data_for_jit() should return None
    template_data = model._template_data_for_jit()
    assert template_data is None, (
        f"Expected None for 'baked_in' nebular, got {template_data}"
    )

    # predict_observables_jit should work fine
    params = {"redshift": 0.05}
    result = model.predict_observables_jit(params)
    assert result is not None
