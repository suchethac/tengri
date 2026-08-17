# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1767 - alpha_fe_evolving=True raises NotImplementedError.

#1767: `alpha_fe_evolving=True` is a documented setting that cannot build a
model. This test verifies that after the fix, it raises an honest loud error
with an actionable message instead of a silent conflicting-mappings failure.

The fix involves two parts:
1. Add _EVOLVING_ALPHA_PARAM_MAP to the manual precedence block in _build_param_map
   so it takes precedence over auto-derivation (avoids ParameterMapError).
2. Add a guard in _init_metallicity that raises NotImplementedError when
   alpha_fe_evolving=True, since the stellar component currently only accepts
   scalar met_alpha_fe, not per-age arrays.
"""

import pytest

from tengri import Parameters, SEDModel
from tengri.observation import Observation, Photometry


@pytest.mark.regression_bug
def test_alpha_fe_evolving_raises_not_implemented(ssp_data_wne):
    """Test that alpha_fe_evolving=True raises NotImplementedError with clear message (#1767).

    After the fix, alpha_fe_evolving=True should raise NotImplementedError
    with an actionable message explaining that per-age alpha enhancement
    is not yet supported and suggesting alternatives.

    The earlier ParameterMapError (conflicting mappings) is fixed by
    adding _EVOLVING_ALPHA_PARAM_MAP to the manual precedence block
    in _build_param_map.
    """

    with pytest.raises(NotImplementedError, match="alpha_fe_evolving=True is not yet supported"):
        # Use the flat-kwarg Parameters API (expert escape hatch) with alpha_fe_evolving=True
        spec = Parameters(
            mean_sfh_type="dpl",
            alpha_fe_evolving=True,  # This should raise NotImplementedError
            sfh_dpl_log_total_mass=10.0,
            sfh_dpl_alpha=2.0,
            sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=2.0,
            met_logzsol=0.0,
            redshift=0.1,
        )
        # Create an Observation with photometry filters
        filters = Photometry.from_names(["sdss_u", "sdss_g"])
        observation = Observation(photometry=filters)
        # SEDModel(...) constructor should raise NotImplementedError
        model = SEDModel(spec, ssp_data_wne, observation=observation)


if __name__ == "__main__":
    # Run with: PYTHONPATH=$PWD/src pytest test_repro_1767.py -v
    pytest.main([__file__, "-v"])
