# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the Cue ``full_catalog`` opt-in (#303).

By default, ``CueBackend.predict_nebular_line_luminosities`` filters
output to the 128 CLOUDY/FSPS-matched lines (``cloudyfsps_only=True``).
Users requesting HeII 1640, HeI 10830, [OIII] 4363, etc. need access
to the full ~271-species catalog the Cue NN was trained on.

This PR plumbs a single boolean through the grammar →
:class:`Parameters` → :class:`NebularSEDComponentConfig` →
``predict_nebular_line_luminosities(cloudyfsps_only=...)``. The
headline named accessors (``halpha``, ``hbeta``, etc.) always work;
this changes only the size of ``pred.lines.all_waves`` /
``all_lums`` and what :meth:`EmissionLines.get` can resolve.
"""

from __future__ import annotations

import dataclasses

import pytest

import tengri
from tengri.components.nebular.component import NebularSEDComponentConfig
from tengri.parameters.groups import Fixed, parse_groups

pytestmark = pytest.mark.contract


class TestGrammarPlumbing:
    """``neb={'type': 'cue', 'full_catalog': True}`` reaches Parameters."""

    def test_default_is_false(self):
        params = parse_groups(
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)}, redshift=Fixed(0.05)
        )
        assert params.cue_full_catalog is False

    def test_opt_in_is_true(self):
        params = parse_groups(
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT), "full_catalog": True},
            redshift=Fixed(0.05),
        )
        assert params.cue_full_catalog is True

    def test_opt_in_only_affects_cue(self):
        """The flag is a no-op on non-cue backends — adding it on cb19
        shouldn't trip an unknown-key validator and shouldn't propagate."""
        params = parse_groups(
            neb={"type": "cb19", "all_params": tengri.Fixed(tengri.DEFAULT), "full_catalog": True},
            redshift=Fixed(0.05),
        )
        # Allowed by the validator, but not interpreted for cb19.
        assert params.cue_full_catalog is False

    def test_full_catalog_is_a_recognized_neb_key(self):
        """Don't trip the unknown-key validator when the user adds the
        flag on the cue group."""
        # No raise means the key is in _GROUP_STRUCTURAL_KEYS['neb'].
        parse_groups(
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT), "full_catalog": True},
            redshift=Fixed(0.05),
        )


class TestConfigDefault:
    """``NebularSEDComponentConfig`` has the new field and defaults False."""

    def test_default_false(self):
        cfg = NebularSEDComponentConfig()
        assert cfg.cue_full_catalog is False

    def test_constructor_opt_in(self):
        cfg = NebularSEDComponentConfig(backend="cue", cue_full_catalog=True)
        assert cfg.cue_full_catalog is True

    def test_field_is_frozen(self):
        """Config is a frozen dataclass; mutation raises FrozenInstanceError.

        Named rather than caught as bare ``Exception``: frozenness is the
        property, and any other error — a property with a broken setter, a
        renamed field — would also have satisfied ``raises(Exception)`` while
        the config was freely mutable. The comment already knew which
        exception it meant; asserting it costs nothing.
        """
        cfg = NebularSEDComponentConfig(backend="cue")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.cue_full_catalog = True  # type: ignore[misc]
