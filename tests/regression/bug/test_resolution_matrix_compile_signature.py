# SPDX-License-Identifier: BSD-3-Clause
"""compile_signature must distinguish a resolution-matrix model (#1163).

The spectrum projector closes over whether it applies ``R @ model`` (banded
resolution matrix) or the Gaussian ``apply_lsf``. If ``resolution_matrix`` is
absent from ``compile_signature``, two models differing only in the matrix share
a compiled kernel from the structural cache and the matrix is silently dropped —
the same cache-collision class as #1135/#1149/#1166. This guard fails if the
signature element is removed (neuter-checkable).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import tengri  # noqa: F401
from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy
from tengri.observation.banded import gaussian_resolution_bands

pytestmark = pytest.mark.regression_bug


def _model(ssp, with_matrix):
    wave = np.geomspace(4000.0, 7000.0, 160)
    kw = {"wave_obs": jnp.asarray(wave), "resolution": 2500.0}
    if with_matrix:
        kw["resolution_matrix"] = gaussian_resolution_bands(jnp.asarray(wave), 2500.0, n_diag=21)
    obs = Observation(spectroscopy=Spectroscopy(**kw))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )


def test_resolution_matrix_changes_compile_signature(synthetic_ssp):
    sig_gauss = _model(synthetic_ssp, with_matrix=False).compile_signature()
    sig_matrix = _model(synthetic_ssp, with_matrix=True).compile_signature()
    assert sig_gauss != sig_matrix, (
        "resolution_matrix presence must change compile_signature — else the "
        "matrix model reuses the Gaussian kernel from the structural cache (#1163)"
    )
