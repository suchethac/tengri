# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #302: BakedIn predict_emission_lines silent NaN.

`SEDModel.predict_emission_lines` on a BakedInBackend nebular returned
all-NaN silently — the backend bakes emission into the SSP grid and
doesn't publish a discrete line catalog. Fix raises NotImplementedError
with an actionable migration message.

See PR #306.
"""

import warnings

import jax
import pytest

pytestmark = pytest.mark.regression_bug


def test_bakedin_predict_emission_lines_raises():
    pytest.importorskip("tengri")
    import tengri

    try:
        ssp = tengri.load_ssp()  # default grid → BakedInBackend (the default neb)
    except FileNotFoundError:
        pytest.skip("default SSP not available")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tengri.SEDModel.build(
            ssp,
            sfh={"type": "const", "*": tengri.FIXED},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "*": tengri.FIXED,
                "tau_diff": 0.0,
                "tau_bc": 0.0,
            },
            redshift=tengri.Fixed(0.05),
        )
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    with pytest.raises(NotImplementedError, match="BakedIn"):
        model.predict_emission_lines(params)
