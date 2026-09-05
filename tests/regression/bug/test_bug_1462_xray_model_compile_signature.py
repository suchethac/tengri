# SPDX-License-Identifier: BSD-3-Clause
"""compile_signature must name WHICH X-ray model, not just that one exists (#1462).

``_xray_model`` was stored at construction but never entered the signature —
only ``uses_xray``, a boolean. So ``agn_xray_corona`` and ``xray_aird`` shared
a compiled kernel and the first one built won: the same class as the AGN block
selectors (#1450) and unlike the radio models beside it, which do carry their
selector.

Why the #1462 sweep could not settle it: X-ray emission lands at keV, so both
models contribute identically nothing to an optical/IR band set. A flux-based
priming test therefore reads the axis as *inert* — vacuous rather than clean —
and the audit correctly declined to clear it. The signature answers the
question directly and cannot be inert, which is what these tests assert.
"""

import pytest

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug


def _model(ssp, obs, xray_type):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        xray={"type": xray_type, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )


def test_two_xray_models_do_not_share_a_signature(synthetic_ssp_wide, synthetic_tophat_obs):
    sig_a = _model(synthetic_ssp_wide, synthetic_tophat_obs, "agn_xray_corona").compile_signature()
    sig_b = _model(synthetic_ssp_wide, synthetic_tophat_obs, "xray_aird").compile_signature()
    assert sig_a != sig_b, (
        "agn_xray_corona and xray_aird share a compile_signature, so they share "
        "a cached kernel and the first one built wins"
    )


def test_a_model_without_xray_still_differs_from_one_with(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """Guard the fix against collapsing the with/without distinction.

    Keying the model name while dropping ``uses_xray`` would leave 'no X-ray'
    and 'some X-ray' indistinguishable when the name string happens to match.
    """
    without = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.05),
    ).compile_signature()
    with_xray = _model(
        synthetic_ssp_wide, synthetic_tophat_obs, "agn_xray_corona"
    ).compile_signature()
    assert without != with_xray


def test_two_models_with_the_same_xray_type_still_match(synthetic_ssp_wide, synthetic_tophat_obs):
    """The point of a signature is reuse — over-keying breaks engine sharing."""
    a = _model(synthetic_ssp_wide, synthetic_tophat_obs, "xray_aird").compile_signature()
    b = _model(synthetic_ssp_wide, synthetic_tophat_obs, "xray_aird").compile_signature()
    assert a == b, "identical configurations must still share a kernel"
