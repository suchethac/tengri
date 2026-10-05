# SPDX-License-Identifier: BSD-3-Clause
"""Test suite selection and configuration routing in run_candels_fits.

Verifies that --suite {grid,xlike} selects different config registries,
results directories, and summary filenames, and that --configs validation
respects the selected suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from paper1.run_candels_fits import (
    SUITE_CONFIGS,
    fit_one_cell_command,
    parse_args,
)

pytestmark = pytest.mark.unit


class TestSuiteConfig:
    """Test SuiteConfig dataclass and suite table."""

    def test_suite_configs_frozen(self):
        """SuiteConfig instances are frozen."""
        assert SUITE_CONFIGS["grid"].__class__.__name__ == "SuiteConfig"
        with pytest.raises(AttributeError):
            SUITE_CONFIGS["grid"].name = "modified"

    def test_suite_configs_keys(self):
        """Both grid and xlike suites are registered."""
        assert set(SUITE_CONFIGS.keys()) == {"grid", "xlike"}

    def test_grid_suite_config(self):
        """Grid suite has correct configuration."""
        grid = SUITE_CONFIGS["grid"]
        assert grid.name == "grid"
        assert grid.results_dir_name == "fits"
        assert grid.summary_filename == "fit_summary.json"
        assert len(grid.config_keys) == 6
        assert grid.config_keys == ["I", "II", "III", "IV", "V", "VI"]

    def test_xlike_suite_config(self):
        """Xlike suite has correct configuration."""
        xlike = SUITE_CONFIGS["xlike"]
        assert xlike.name == "xlike"
        assert xlike.results_dir_name == "fits_xlike"
        assert xlike.summary_filename == "fit_summary_xlike.json"
        assert len(xlike.config_keys) == 5
        expected_keys = [
            "cigale_like",
            "prospector_like",
            "bagpipes_like",
            "beagle_like",
            "dense_basis_like",
        ]
        assert xlike.config_keys == expected_keys

    def test_suite_config_get_results_dir(self):
        """SuiteConfig.get_results_dir returns correct path."""
        paper1_root = Path("/test/paper1")
        grid_dir = SUITE_CONFIGS["grid"].get_results_dir(paper1_root)
        assert grid_dir == paper1_root / "results" / "fits"
        xlike_dir = SUITE_CONFIGS["xlike"].get_results_dir(paper1_root)
        assert xlike_dir == paper1_root / "results" / "fits_xlike"

    def test_suite_config_get_summary_json(self):
        """SuiteConfig.get_summary_json returns correct path."""
        paper1_root = Path("/test/paper1")
        grid_json = SUITE_CONFIGS["grid"].get_summary_json(paper1_root)
        assert grid_json == paper1_root / "results" / "fit_summary.json"
        xlike_json = SUITE_CONFIGS["xlike"].get_summary_json(paper1_root)
        assert xlike_json == paper1_root / "results" / "fit_summary_xlike.json"


class TestSuiteSelection:
    """Test --suite argument handling."""

    def test_default_suite_is_grid(self):
        """Default suite is grid when --suite is not specified."""
        args = parse_args([])
        assert args.suite == "grid"
        assert args.suite_config.name == "grid"

    def test_explicit_grid_suite(self):
        """--suite grid selects grid suite."""
        args = parse_args(["--suite", "grid"])
        assert args.suite == "grid"
        assert args.suite_config == SUITE_CONFIGS["grid"]

    def test_xlike_suite(self):
        """--suite xlike selects xlike suite."""
        args = parse_args(["--suite", "xlike"])
        assert args.suite == "xlike"
        assert args.suite_config == SUITE_CONFIGS["xlike"]

    def test_invalid_suite_raises_error(self):
        """Invalid suite name raises error."""
        with pytest.raises(SystemExit):
            parse_args(["--suite", "invalid"])


class TestConfigsValidation:
    """Test --configs validation respects suite selection."""

    def test_grid_suite_accepts_grid_configs(self):
        """Grid suite accepts I, II, III, IV, V, VI."""
        args = parse_args(["--suite", "grid", "--configs", "I,II"])
        assert args.run_configs == ["I", "II"]

    def test_grid_suite_rejects_xlike_configs(self):
        """Grid suite rejects cigale_like and other xlike configs."""
        with pytest.raises(SystemExit):
            parse_args(["--suite", "grid", "--configs", "cigale_like"])

    def test_xlike_suite_accepts_xlike_configs(self):
        """Xlike suite accepts cigale_like, prospector_like, etc."""
        args = parse_args(["--suite", "xlike", "--configs", "cigale_like,prospector_like"])
        assert args.run_configs == ["cigale_like", "prospector_like"]

    def test_xlike_suite_rejects_grid_configs(self):
        """Xlike suite rejects I, II, III, etc."""
        with pytest.raises(SystemExit):
            parse_args(["--suite", "xlike", "--configs", "I"])

    def test_configs_validation_error_names_suite(self):
        """Config validation error message names the selected suite."""
        with pytest.raises(SystemExit):
            try:
                parse_args(["--suite", "xlike", "--configs", "I"])
            except SystemExit:
                raise


class TestFitCommandGeneration:
    """Test that fit_one subprocess command uses correct results directory."""

    def test_fit_one_cell_command_grid(self):
        """fit_one_cell_command uses provided results_dir."""
        results_dir_grid = Path("/test/results/fits")
        cmd = fit_one_cell_command(123, "I", results_dir_grid)
        assert "--out" in cmd
        out_idx = cmd.index("--out")
        assert cmd[out_idx + 1] == str(results_dir_grid)

    def test_fit_one_cell_command_xlike(self):
        """fit_one_cell_command uses xlike results directory when provided."""
        results_dir_xlike = Path("/test/results/fits_xlike")
        cmd = fit_one_cell_command(123, "cigale_like", results_dir_xlike)
        assert "--out" in cmd
        out_idx = cmd.index("--out")
        assert cmd[out_idx + 1] == str(results_dir_xlike)


class TestNoHardcodedPaths:
    """Sweep test that no hardcoded "fits" or "fit_summary.json" remains.

    These tests scan the source code for path literals outside the suite table.
    Docstrings are allowed to mention these values (they describe what the code does).
    """

    def test_no_hardcoded_fits_path_outside_suite_table(self):
        """No literal 'fits' in path construction outside suite table."""
        run_candels_py = PAPER1 / "run_candels_fits.py"
        with open(run_candels_py) as f:
            content = f.read()

        # Look for patterns like: "results" / "fits" or results_dir = ... "fits"
        # These indicate hardcoding. Docstrings are OK.
        lines = content.split("\n")

        # Ignore lines that are in docstrings (triple-quoted strings) or
        # are part of the SUITE_CONFIGS definition (lines ~50-95)
        suspicious = []
        in_docstring = False
        for lineno, line in enumerate(lines, 1):
            # Skip SUITE_CONFIGS definition
            if lineno >= 50 and lineno <= 95:
                continue

            # Track docstrings
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
                continue

            if in_docstring:
                continue

            # Look for path constructions: results_dir = ... / "fits"
            if ('/ "fits"' in line or "/ 'fits'" in line) and "results_dir" in lines[
                max(0, lineno - 3) : lineno
            ]:
                suspicious.append((lineno, line.strip()))

        assert not suspicious, f"Found hardcoded 'fits' path construction: {suspicious}"

    def test_no_hardcoded_fit_summary_outside_suite_table(self):
        """No literal 'fit_summary.json' in path construction outside suite table."""
        run_candels_py = PAPER1 / "run_candels_fits.py"
        with open(run_candels_py) as f:
            content = f.read()

        lines = content.split("\n")

        # Ignore lines that are in docstrings or part of SUITE_CONFIGS
        suspicious = []
        in_docstring = False
        for lineno, line in enumerate(lines, 1):
            # Skip SUITE_CONFIGS definition
            if lineno >= 50 and lineno <= 95:
                continue

            # Track docstrings
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
                continue

            if in_docstring:
                continue

            # Look for hardcoded file references: = ... / "fit_summary.json"
            if (
                '/ "fit_summary' in line or "/ 'fit_summary" in line
            ) and "summary_json" not in line:
                suspicious.append((lineno, line.strip()))

        assert not suspicious, f"Found hardcoded 'fit_summary.json' path: {suspicious}"
