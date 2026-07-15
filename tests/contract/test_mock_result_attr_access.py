# SPDX-License-Identifier: BSD-3-Clause
"""``generate_mock`` results support both dict and attribute access.

Fresh-user audit (2026-07): ``generate_mock`` returned a plain dict, but the
natural idiom — and what ``SEDModel.mock()``'s ``MockData`` uses — is
``mock.flux_obs``. Attribute access on a plain dict raised ``AttributeError``,
which broke several tutorial notebooks on first run. The result is now a dict
subclass that also allows attribute access, so both idioms work while remaining
a plain ``dict`` for existing consumers.
"""

from __future__ import annotations

import pytest

from tengri.analysis.mock import _MockResult

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


def test_dict_and_attribute_access_agree():
    r = _MockResult(flux_true=1.0, noise=0.1, flux_obs=1.05, params={"a": 1})
    assert isinstance(r, dict)  # still a plain dict for existing consumers
    assert r["flux_obs"] == r.flux_obs == 1.05
    assert r["noise"] == r.noise == 0.1
    assert sorted(r.keys()) == ["flux_obs", "flux_true", "noise", "params"]


def test_missing_attribute_raises_clear_error():
    r = _MockResult(flux_true=1.0, noise=0.1, params={})  # no flux_obs (key=None case)
    with pytest.raises(AttributeError, match="flux_obs"):
        _ = r.flux_obs
    with pytest.raises(AttributeError):
        _ = r.not_a_key
