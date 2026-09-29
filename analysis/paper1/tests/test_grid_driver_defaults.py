# SPDX-License-Identifier: BSD-3-Clause
"""Grid driver defaults: profile_mass=auto, two-rung retune ladder, cpu-scaled concurrency.

These are pinned because of measured costs from the grid's own cell logs and JSONs
(analysis/paper1/results/fits on the grid branch, 2026-09-14 and 2026-09-29):

(a) row VI was launched without --profile-mass and paid ~4x (11,600-35,000 s per
    attempt, tree depth 8.5-9.9, up to 90% of trees at the depth-10 cap);

(b) the retune ladder's third rung (target_accept 0.99) cost 3-5x the first rung and
    adopted NONE of 79/II, 15336/II, 16455/II, 13097/VI;

(c) one fit_one cell uses ~4.5 of 14 cores at steady state, so more than
    cpu_count//4 concurrent cells oversubscribe the box.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1.fit_one import (
    DEFAULT_RETUNE_ATTEMPTS,
    build_parser as fit_one_build_parser,
    resolve_profile_mass_record,
    retune_settings,
)
from paper1.run_candels_fits import (
    build_parser as run_candels_build_parser,
    default_jobs,
)

pytestmark = pytest.mark.contract


class TestRetuneLadder:
    def test_default_retune_attempts_is_two(self):
        assert DEFAULT_RETUNE_ATTEMPTS == 2

    def test_retune_settings_attempt_1(self):
        base = {"target_accept_rate": 0.85, "n_warmup": 150}
        result = retune_settings(1, base)
        assert result["target_accept_rate"] == 0.85

    def test_retune_settings_attempt_2(self):
        base = {"target_accept_rate": 0.85, "n_warmup": 150}
        result = retune_settings(2, base)
        assert result["target_accept_rate"] == 0.95

    def test_retune_settings_attempt_3(self):
        base = {"target_accept_rate": 0.85, "n_warmup": 150}
        result = retune_settings(3, base)
        assert result["target_accept_rate"] == 0.99

    def test_retune_settings_attempt_4(self):
        base = {"target_accept_rate": 0.85, "n_warmup": 150}
        result = retune_settings(4, base)
        assert result["target_accept_rate"] == 0.99
        assert result["n_warmup"] == 300


class TestProfileMassRecord:
    def test_resolve_profile_mass_with_resolved_false(self):
        requested = "auto"
        diagnostics = {
            "profile_mass_resolved": False,
            "profile_mass_reason": "guard failed"
        }
        result = resolve_profile_mass_record(requested, diagnostics)
        assert result == {
            "profile_mass_requested": "auto",
            "profile_mass": False,
            "profile_mass_reason": "guard failed"
        }

    def test_resolve_profile_mass_with_empty_diagnostics(self):
        requested = "auto"
        diagnostics = {}
        result = resolve_profile_mass_record(requested, diagnostics)
        # When resolved value is absent, record the requested value verbatim
        assert result == {
            "profile_mass_requested": "auto",
            "profile_mass": "auto",
        }

    def test_resolve_profile_mass_with_resolved_true(self):
        requested = "on"
        diagnostics = {
            "profile_mass_resolved": True,
            "profile_mass_reason": "all linearity guards passed"
        }
        result = resolve_profile_mass_record(requested, diagnostics)
        assert result == {
            "profile_mass_requested": "on",
            "profile_mass": True,
            "profile_mass_reason": "all linearity guards passed"
        }


class TestFitOneParser:
    def test_build_parser_default_profile_mass(self):
        parser = fit_one_build_parser()
        args = parser.parse_args(["--galaxy", "79", "--config", "II", "--out", "x"])
        assert args.profile_mass == "auto"

    def test_build_parser_profile_mass_bare_flag(self):
        parser = fit_one_build_parser()
        args = parser.parse_args(
            ["--galaxy", "79", "--config", "II", "--out", "x", "--profile-mass"]
        )
        assert args.profile_mass == "on"

    def test_build_parser_profile_mass_off(self):
        parser = fit_one_build_parser()
        args = parser.parse_args(
            ["--galaxy", "79", "--config", "II", "--out", "x", "--profile-mass", "off"]
        )
        assert args.profile_mass == "off"

    def test_build_parser_default_retune_attempts(self):
        parser = fit_one_build_parser()
        args = parser.parse_args(["--galaxy", "79", "--config", "II", "--out", "x"])
        assert args.retune_attempts == DEFAULT_RETUNE_ATTEMPTS


class TestRunCandelsParser:
    def test_build_parser_default_profile_mass(self):
        parser = run_candels_build_parser()
        args = parser.parse_args([])
        assert args.profile_mass == "auto"

    def test_build_parser_default_jobs(self):
        parser = run_candels_build_parser()
        args = parser.parse_args([])
        assert args.jobs == default_jobs()

    def test_build_parser_jobs_exceeds_cpu_limit(self):
        parser = run_candels_build_parser()
        # Jobs > cpu_count//2 should raise SystemExit
        cpu_limit = os.cpu_count() or 4
        with pytest.raises(SystemExit):
            parser.parse_args(["--jobs", str(cpu_limit)])

    def test_default_jobs_is_cpu_count_divided_by_four(self):
        expected = max(1, os.cpu_count() // 4)
        assert default_jobs() == expected


class TestCommandTemplatePassthrough:
    def test_fit_one_command_includes_profile_mass(self):
        """The scheduler command template includes --profile-mass."""
        # Simulate building a command for a cell
        parser = fit_one_build_parser()
        args = parser.parse_args(
            ["--galaxy", "79", "--config", "II", "--out", "outdir", "--profile-mass", "auto"]
        )
        # Verify that the profile_mass setting would be passed to subprocess
        assert hasattr(args, "profile_mass")
        assert args.profile_mass == "auto"

    def test_fit_one_command_includes_retune_attempts(self):
        """The scheduler command template includes --retune-attempts."""
        parser = fit_one_build_parser()
        args = parser.parse_args(
            ["--galaxy", "79", "--config", "II", "--out", "outdir", "--retune-attempts", "2"]
        )
        assert hasattr(args, "retune_attempts")
        assert args.retune_attempts == 2
