# SPDX-License-Identifier: BSD-3-Clause
"""The one redshift-range rule of the runtime-redshift path (#3505).

``require_redshifts_in_range`` is shared by the catalog fit and the batched build. A
redshift outside a ``catalog_z_range`` is refused and named by row, and a non-finite
redshift is refused the way ``catalog_redshift_span`` refuses it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.inference.fitter import catalog_redshift_span, require_redshifts_in_range

pytestmark = pytest.mark.contract


def test_redshifts_inside_the_range_pass_including_its_ends():
    require_redshifts_in_range([0.02, 0.05, 0.4], (0.02, 0.4))


def test_redshifts_outside_the_range_name_their_rows():
    with pytest.raises(ValueError, match=r"row\(s\) \[1, 3\]") as err:
        require_redshifts_in_range([0.05, 0.7, 0.2, -0.1], (0.02, 0.4))
    message = str(err.value)
    assert "z=[0.7, -0.1]" in message
    assert "[0.020, 0.400]" in message
    assert "Widen catalog_z_range at model build time or filter the table" in message


def test_nan_is_refused_as_catalog_redshift_span_refuses_it():
    expected = r"not finite .* row\(s\) \[1\]"
    with pytest.raises(ValueError, match=expected):
        require_redshifts_in_range([0.1, np.nan], (0.0, 0.5))
    with pytest.raises(ValueError, match=expected):
        catalog_redshift_span([0.1, np.nan])
