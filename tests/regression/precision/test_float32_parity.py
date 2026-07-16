import numpy as np
import pytest

from .conftest import Z_MASS_GRID, build_model, forward_outputs

pytestmark = pytest.mark.regression_bug


@pytest.mark.parametrize("z,log10_mass", Z_MASS_GRID)
def test_f64_reference_is_finite(ssp_bare, z, log10_mass):
    """Baseline: the current f64 path produces finite outputs on the grid."""
    model = build_model(ssp_bare, "float64")
    out = forward_outputs(model, z, log10_mass)
    for k, v in out.items():
        assert np.all(np.isfinite(v)), f"{k} non-finite at z={z}, logM={log10_mass}"
