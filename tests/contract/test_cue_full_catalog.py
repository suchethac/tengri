# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the Cue ``full_catalog`` opt-in/opt-out (#303, #2239).

``CueBackend.predict_nebular_line_luminosities`` can filter its output to
the 128 legacy CLOUDY/FSPS-matched lines (``cloudyfsps_only=True``) or
return the full ~138-line catalog the Cue NN was trained on. Since #2239
the full catalog is the default, so HeII 1640, C IV 1549, etc. are reachable
without any grammar key at all; ``full_catalog: False`` is the explicit
opt-out kept for cross-code comparisons against the legacy subset.

This plumbs a single boolean through the grammar ->
:class:`Parameters` -> :class:`NebularSEDComponentConfig` ->
``predict_nebular_line_luminosities(cloudyfsps_only=...)``. Ten of the
~138 lines (headlined by C IV 1548/1550) are cue-only and absent from the
128-line subset; of the 11 ``KEY_LINES`` headline accessors only
``civ_1549`` is affected either way (#2239) -- the other ten (``halpha``,
``hbeta``, etc.) always work. This changes the size of
``pred.lines.all_waves`` / ``all_lums`` and what :meth:`EmissionLines.get`
can resolve.
"""

from __future__ import annotations

import dataclasses

import pytest

import tengri
from tengri.components.nebular.component import NebularSEDComponentConfig
from tengri.parameters.groups import Fixed, parse_groups

pytestmark = pytest.mark.contract


class TestGrammarPlumbing:
    """``neb={'type': 'cue', 'full_catalog': ...}`` reaches Parameters."""

    def test_default_is_true(self):
        """No ``full_catalog`` key at all: the #2239 default is the full catalog."""
        params = parse_groups(
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)}, redshift=Fixed(0.05)
        )
        assert params.cue_full_catalog is True

    def test_explicit_true_is_true(self):
        params = parse_groups(
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT), "full_catalog": True},
            redshift=Fixed(0.05),
        )
        assert params.cue_full_catalog is True

    def test_explicit_false_opts_out_to_the_legacy_subset(self):
        params = parse_groups(
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT), "full_catalog": False},
            redshift=Fixed(0.05),
        )
        assert params.cue_full_catalog is False

    def test_opt_out_only_affects_cue(self):
        """The flag is a no-op on non-cue backends — adding it on cb19
        shouldn't trip an unknown-key validator and shouldn't propagate."""
        params = parse_groups(
            neb={
                "type": "cb19",
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "full_catalog": False,
            },
            redshift=Fixed(0.05),
        )
        # Allowed by the validator, but not interpreted for cb19: the
        # attribute still holds the (now True) default, not the cb19 group's
        # unrelated 'full_catalog' key.
        assert params.cue_full_catalog is True

    def test_full_catalog_is_a_recognized_neb_key(self):
        """Don't trip the unknown-key validator when the user adds the
        flag on the cue group."""
        # No raise means the key is in _GROUP_STRUCTURAL_KEYS['neb'].
        parse_groups(
            neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT), "full_catalog": True},
            redshift=Fixed(0.05),
        )


class TestConfigDefault:
    """``NebularSEDComponentConfig`` has the field and defaults True (#2239)."""

    def test_default_true(self):
        cfg = NebularSEDComponentConfig()
        assert cfg.cue_full_catalog is True

    def test_constructor_opt_out(self):
        cfg = NebularSEDComponentConfig(backend="cue", cue_full_catalog=False)
        assert cfg.cue_full_catalog is False

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
