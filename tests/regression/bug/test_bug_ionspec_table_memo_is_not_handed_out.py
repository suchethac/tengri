# SPDX-License-Identifier: BSD-3-Clause
"""The ionizing-parameter table memo is read-only to its callers.

``precompute_ionizing_params_table`` memoizes its result per SSP content for the
whole process. It used to return the stored dict writable, so a caller that assigned
into the result rewrote the table for every later Cue model on the same grid.
``test_new_user_errors.py::test_wne_guard_is_two_sided`` did exactly that (it set
``logqion_table`` to 30), and every later model on ``synthetic_ssp_wide`` in the
same worker then had no ionizing photons: ``enable_fast_nebular`` stopped moving
the photometry and two contract tests failed only when xdist put both files on one
worker.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.nebular.ionizing_spectrum import precompute_ionizing_params_table

pytestmark = pytest.mark.regression_bug


def _table(ssp):
    return precompute_ionizing_params_table(
        np.asarray(ssp.ssp_wave),
        np.asarray(ssp.ssp_flux),
        np.asarray(ssp.ssp_lgmet),
        ssp_log_age_yr=np.asarray(ssp.ssp_lg_age_gyr) + 9.0,
    )


def test_assigning_into_the_result_is_refused_and_the_table_is_unchanged(synthetic_ssp_wide):
    first = _table(synthetic_ssp_wide)
    reference = np.array(first["logqion_table"], copy=True)

    with pytest.raises(TypeError):
        first["logqion_table"] = np.full_like(reference, 30.0)

    second = _table(synthetic_ssp_wide)
    assert second is first  # callers key their bookkeeping on the entry's identity
    np.testing.assert_array_equal(second["logqion_table"], reference)


def test_a_copy_can_be_changed_without_reaching_the_table(synthetic_ssp_wide):
    table = _table(synthetic_ssp_wide)
    reference = np.array(table["logqion_table"], copy=True)

    rigged = dict(table)
    rigged["logqion_table"] = np.full_like(reference, 30.0)

    np.testing.assert_array_equal(_table(synthetic_ssp_wide)["logqion_table"], reference)


def test_writing_into_a_returned_array_is_refused(synthetic_ssp_wide):
    table = _table(synthetic_ssp_wide)
    with pytest.raises(ValueError, match="read-only"):
        table["logqion_table"][...] = 30.0
