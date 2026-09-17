# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2255: lgmet_scatter kwarg is LIVE, not dead.

Before fix: Parameters(lgmet_scatter=0.3) vs default had bit-identical
predict_photometry outputs because the kwarg wasn't wired into the registry.

After fix: The flat lgmet_scatter kwarg SETS the registered
met_logzsol_scatter's Fixed value, making it LIVE and causing predict_photometry
to differ when scatter width changes.

Test markers:
- test_lgmet_scatter_both_spellings_refuse: Guards against silent shadowing
  when both flat (lgmet_scatter) and grammar (met_logzsol_scatter) spellings
  are passed in Parameters(). The refusal is loud and names both spellings.
- test_grammar_met_logzsol_scatter_free_prior_untouched: Grammar builds that
  free met_logzsol_scatter are outside the flat form's blast radius; the
  declared free_prior (Uniform(0.02, 0.4)) is preserved.
"""

from pathlib import Path

import pytest

from tengri import DEFAULT, Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tests.contract._signature_builds import BARE_STELLAR_NAME


def _bare_stellar_ssp():
    """Load the bare stellar SSP for testing."""
    return load_ssp_data(str(Path(__file__).resolve().parents[3] / "data" / BARE_STELLAR_NAME))


@pytest.mark.regression_bug
def test_lgmet_scatter_both_spellings_refuse():
    """Refuse loudly when both lgmet_scatter and met_logzsol_scatter are passed.

    The flat-form kwarg lgmet_scatter and the grammar parameter
    met_logzsol_scatter are the same underlying parameter. Passing both
    is silent shadowing, which is forbidden. The error message MUST name
    both spellings so the user knows what went wrong.
    """
    with pytest.raises(ValueError) as exc_info:
        Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            lgmet_scatter=0.3,
            met_logzsol_scatter=Fixed(0.2),  # Both spellings: refuse
        )

    error_msg = str(exc_info.value)
    assert "lgmet_scatter" in error_msg, "Error must name the flat-form spelling"
    assert "met_logzsol_scatter" in error_msg, "Error must name the grammar-form spelling"
    assert "both" in error_msg.lower() or "both" in error_msg, (
        "Error must indicate the problem is passing both"
    )


@pytest.mark.regression_bug
def test_grammar_met_logzsol_scatter_free_prior_untouched():
    """Free met_logzsol_scatter in grammar builds is unaffected by flat form.

    The flat-form lgmet_scatter kwarg is an expert escape hatch for backward
    compatibility. Grammar builds (via SEDModel.build(...)) should be able to
    free met_logzsol_scatter with a declared prior (Uniform(0.02, 0.4)) without
    interference from any flat-form kwarg. This test verifies the declared
    free_prior is respected when met_logzsol_scatter is freed.
    """
    ssp = _bare_stellar_ssp()
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))

    # Build via SEDModel.build with grammar that frees met_logzsol_scatter.
    # No flat-form Parameters involved; the grammar controls everything.
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},  # pinned SFH isolates the test
        met={
            "logzsol_scatter": Uniform(0.02, 0.4),  # free with a declared prior
        },
        redshift=Fixed(0.1),
    )

    # The free_params must include met_logzsol_scatter with the declared prior.
    assert "met_logzsol_scatter" in model.spec.free_params, (
        "met_logzsol_scatter should be free when declared Uniform(0.02, 0.4)"
    )

    # Check the distribution is correct (the declared prior, not the registry default).
    dist = model.spec.get_distribution("met_logzsol_scatter")
    assert isinstance(dist, Uniform), f"Expected Uniform prior, got {type(dist)}"
    # Bounds should match the declared prior (allowing small float tolerance).
    assert abs(float(dist.lo) - 0.02) < 1e-6, f"Lower bound should be 0.02, got {dist.lo}"
    assert abs(float(dist.hi) - 0.4) < 1e-6, f"Upper bound should be 0.4, got {dist.hi}"
