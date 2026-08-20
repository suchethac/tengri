# SPDX-License-Identifier: BSD-3-Clause
"""Template threading must survive a SECOND model build in the same process.

``_template_data_for_jit`` walks ``_cached_component_chain``. That chain is
pre-built in ``__init__`` only under ``spectrum_precomp`` / ``wave_precomp``;
otherwise the first ``predict_state`` warmup populates it.

Build a second model with the same compile signature and the structural-kernel
cache hits, so the warmup never runs. The publisher used to return ``None``
there, every template fell back to its in-block load, and the whole library
baked back into the graph. Measured on ``torus='skirtor'``: **0.05 MB on build
1, 29.94 MB on build 2**.

The threading contract test cannot catch this — it builds one fresh model per
test, so every measurement it takes is a "build 1". This asserts the repeat.
"""

from __future__ import annotations

import pathlib
import warnings

import jax
import pytest

from tengri import FIXED, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed
from tests.contract._jaxpr_consts import baked_mb

pytestmark = [pytest.mark.regression_bug]

#: Generous vs the ~0.05 MB floor, far under the 29.94 MB regression.
_BUDGET_MB = 1.0


def _build(ssp, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust_attenuation={"law": "power_law", "type": "two_component", "all_params": FIXED},
            redshift=Fixed(0.1),
            agn={
                "type": "composable",
                "all_params": FIXED,
                "disc": {"type": "multicolor"},
                "torus": {"type": "skirtor"},
            },
        )


def _baked(model):
    from tengri.inference._model_cache import _default_owner

    model._get_or_build_predict_observables_jit()
    impl = _default_owner.get_structural_kernel(model.compile_signature())[
        "predict_observables_impl"
    ]
    params = model.spec.sample(jax.random.PRNGKey(0))
    return baked_mb(
        jax.make_jaxpr(impl)(
            params,
            model.spec.get_fixed_values(),
            model.ssp_data,
            model._template_data_for_jit(),
        )
    )


def test_threading_holds_across_repeated_builds():
    """Three identical builds in one process must all thread."""
    candidates = sorted(pathlib.Path("data").glob("ssp_*.h5"))
    if not candidates:
        pytest.skip("no SSP grid available")
    try:
        ssp = load_ssp_data(str(candidates[0].resolve()))
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("SSP unavailable")
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))

    try:
        measured = [_baked(_build(ssp, obs)) for _ in range(3)]
    except FileNotFoundError:
        pytest.skip("SKIRTOR grid not available")

    # Assert every build, not just the last: the regression was specifically
    # "first one is fine, the rest are not".
    for i, mb in enumerate(measured, start=1):
        assert mb < _BUDGET_MB, f"build {i} baked {mb:.2f} MB (all builds: {measured})"
