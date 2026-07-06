# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the config-dataclass top-level demotion (#887).

The nested-dict grammar (``SEDModel.build``) is the one user-facing
construction surface; the config dataclasses are internal lowering
artifacts. Top-level access warns for one release; the ``tengri.config``
path stays warning-free for internal/expert use.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.contract

_DEMOTED = ("AGNConfig", "DustConfig", "NebularConfig", "SEDModelConfig", "SFHConfig")


@pytest.mark.parametrize("name", _DEMOTED)
def test_top_level_config_access_warns(name):
    """tengri.<Config> resolves but emits DeprecationWarning."""
    import tengri

    with pytest.warns(DeprecationWarning, match="nested-dict"):
        obj = getattr(tengri, name)
    assert obj is not None


@pytest.mark.parametrize("name", _DEMOTED)
def test_config_namespace_path_is_warning_free(name):
    """from tengri.config import <Config> stays silent (expert path)."""
    from tengri.config import settings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        obj = getattr(settings, name)
    assert obj is not None


def test_top_level_and_namespace_resolve_same_object():
    """The shim forwards to the identical class, not a copy."""
    import tengri
    from tengri.config.settings import SEDModelConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert tengri.SEDModelConfig is SEDModelConfig
