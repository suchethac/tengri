# SPDX-License-Identifier: BSD-3-Clause
"""#2297: the composable kwargs fallbacks read the declared default, not a literal copy.

``numeric_agn_kwargs`` carried six ``params.get(name, <literal>)`` fallbacks whose literal
disagreed with the value the parameter's declaration owns. Every build path supplies every
declared key, so the fallbacks never reach a built model; a direct call that omits the key
reaches them. Each test calls the function with the key ABSENT and compares it with the same
call given the declared default explicitly, and checks the old literal is not what it returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.agn._params import PARAMS
from tengri.components.agn.component import numeric_agn_kwargs
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.regression_bug

#: (kwargs key, declared parameter name, the literal the fallback used to hold)
_SITES = (
    ("agn_log_mbh", "agn_log_mbh", 8.0),
    ("agn_tau_torus", "agn_tau_torus", 3.0),
    ("agn_T_hot", "agn_T_hot", 1500.0),
    ("agn_frac_hot", "agn_frac_hot", 0.5),
    ("agn_cos_inc", "agn_cos_inc", 0.5),
    ("agn_polar_oa", "agn_polar_oa", 40.0),
)


@pytest.mark.parametrize(("key", "name", "old_literal"), _SITES)
def test_fallback_absent_equals_declared_default(key, name, old_literal):
    declared = declared_default(PARAMS, name)
    absent = numeric_agn_kwargs({}, 0.5)[key]
    explicit = numeric_agn_kwargs({name: declared}, 0.5)[key]
    assert np.array_equal(np.asarray(absent), np.asarray(explicit))
    assert not np.array_equal(np.asarray(absent), np.asarray(old_literal))
