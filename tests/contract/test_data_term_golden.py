# SPDX-License-Identifier: BSD-3-Clause
"""``DataTerm`` values are pinned bit for bit at the prior's reference draw.

``data/data_term_golden.npy`` holds ``(log_prob, log_normalization)`` of every Fitter structure
in :data:`tests.contract._channel_fixtures.KINDS`. ``log_prob + log_normalization`` is the
normalized log-density of the data: the diagonal, censored and correlated terms drop their
Gaussian constant from ``log_prob`` and return it in ``log_normalization``, and the
marginalized terms (``spec_cal``, ``spec_lines``, ``spec_cal_lines``) return it inside
``log_prob`` with ``log_normalization`` zero. ``spec_lines_fitted`` returns the diagonal
constant of its spectrum, 971.9137446159697 on these data. The independent dense-Gaussian
references of ``test_channel_engine_sweep.py`` check the same structures against scipy.
Regenerate with ``python -m tests.contract.test_data_term_golden OUT.npy`` only from a build
that is known to be right.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests.contract import _channel_fixtures as fixtures

pytestmark = pytest.mark.contract

GOLDEN = Path(__file__).parent / "data" / "data_term_golden.npy"


def term_values(kind: str) -> tuple:
    """``(log_prob, log_normalization)`` of one fixture structure."""
    fitter = fixtures.fitter_of(kind)
    term, physical = fixtures.build_term(fitter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            float(term.log_prob(physical, fitter._data_args)),
            float(term.log_normalization(fitter._data_args)),
        )


@pytest.mark.parametrize("kind", fixtures.KINDS)
def test_data_term_is_bit_identical_to_the_golden_values(kind):
    golden = np.load(GOLDEN)[fixtures.KINDS.index(kind)]
    actual = np.asarray(term_values(kind))
    np.testing.assert_array_equal(actual, golden)


if __name__ == "__main__":
    np.save(sys.argv[1], np.asarray([term_values(kind) for kind in fixtures.KINDS]))
