# SPDX-License-Identifier: BSD-3-Clause
"""Silent fallbacks that must announce themselves.

Two places used to substitute a value without telling the caller:

1. ``parameters/defaults.py``: a malformed ``~/.tengri/defaults.toml`` fell
   back to a hand-written copy of the packaged defaults, which had drifted
   (``sfh = "tsnorm"`` against the packaged ``"dense_basis"``).
2. ``parameters/parameters.py``: a free ``sfh_field_*`` prior was dropped
   silently when the field was switched off (``stochastic=False``).
"""

from __future__ import annotations

import warnings

import pytest

from tengri import Fixed, Parameters, Uniform
from tengri.parameters import defaults

pytestmark = [pytest.mark.contract]


def _write_malformed_user_toml(tmp_path, monkeypatch):
    user_path = tmp_path / "defaults.toml"
    user_path.write_text('[from_config\nsfh = "tsnorm"\n', encoding="utf-8")
    monkeypatch.setattr(defaults, "_USER_DEFAULTS", user_path)
    return user_path


def test_malformed_user_defaults_warn_and_use_packaged_values(tmp_path, monkeypatch):
    user_path = _write_malformed_user_toml(tmp_path, monkeypatch)

    with pytest.warns(UserWarning, match=str(user_path)) as record:
        fc = defaults.get_from_config_defaults()

    assert fc["sfh"] == "dense_basis"
    assert any(str(user_path) in str(w.message) for w in record)


def test_malformed_user_defaults_load_packaged_tree(tmp_path, monkeypatch):
    _write_malformed_user_toml(tmp_path, monkeypatch)

    with pytest.warns(UserWarning):
        raw = defaults.load_defaults()

    assert raw["sfh"]["n_grid"] == 256


def test_hardcoded_from_config_fallback_agrees_with_packaged_toml():
    packaged = defaults.load_defaults()["from_config"]
    for key, value in defaults._FALLBACK_FROM_CONFIG.items():
        assert packaged[key] == value, key


def test_dropped_free_sfh_field_prior_warns():
    with pytest.warns(UserWarning, match="sfh_field_psd_sigma"):
        Parameters(
            mean_sfh_type="dpl",
            stochastic=False,
            sfh_field_psd_sigma=Uniform(0.1, 1.0),
        )


def test_dropped_fixed_sfh_field_value_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Parameters(
            mean_sfh_type="dpl",
            stochastic=False,
            sfh_field_psd_sigma=Fixed(0.5),
            sfh_field_psd_tau_myr=50.0,
        )
