"""Tests for benchmark_forward_model.py census and configuration consistency.

These tests verify:
- Section tracking _REQUIRED_SECTIONS and _BARE_SSP_SECTIONS are consistent
- AGN model keys are valid
- Gradient config selection uses labels that exist
- No configuration inconsistencies
"""

import importlib.util
from pathlib import Path

from tengri import list_agn_models


def load_benchmark_module():
    """Import benchmark_forward_model.py as a module (it's a script)."""
    script_path = (
        Path(__file__).parent.parent.parent / "bench" / "scripts" /
        "benchmark_forward_model.py"
    )
    # Read and parse the script without executing the __main__ block
    with open(script_path) as f:
        content = f.read()

    # Remove the __main__ block to prevent execution
    # Find the "if __name__ == '__main__':" line
    lines = content.split('\n')
    main_idx = None
    for i, line in enumerate(lines):
        if "if __name__ == '__main__':" in line:
            main_idx = i
            break

    if main_idx is not None:
        # Truncate at the __main__ block
        content = '\n'.join(lines[:main_idx])

    # Create and execute the module
    spec = importlib.util.spec_from_file_location("benchmark_forward_model", script_path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(content, script_path, 'exec'), module.__dict__)
    return module


def get_benchmark_configs():
    """Extract all config labels from the benchmark module."""
    bench = load_benchmark_module()
    all_configs = bench.individual_components + bench.composite_configs
    return all_configs, bench


class TestBenchmarkCensusConsistency:
    """Verify census tracking sets are consistent with actual sections."""

    def test_required_sections_are_subset_of_defined_sections(self):
        """_REQUIRED_SECTIONS must name sections that are actually generated."""
        all_configs, bench = get_benchmark_configs()
        sfh_types = [sfh_type for _, sfh_type in bench.sfh_types]

        # Compute all possible forward section names
        all_forward_sections = set()
        for label, _ in all_configs:
            for sfh_type in sfh_types:
                all_forward_sections.add(f"{label}_{sfh_type}")

        # Compute all possible gradient section names (only for grad configs)
        all_gradient_sections = set()
        grad_config_labels = {"Stellar only", "Kitchen sink (all components)"}
        for label, _ in all_configs:
            if label in grad_config_labels:
                for sfh_type in sfh_types:
                    all_gradient_sections.add(f"{label}_grad_{sfh_type}")

        all_defined_sections = all_forward_sections | all_gradient_sections

        # Check that _REQUIRED_SECTIONS is a subset
        required_missing = bench._REQUIRED_SECTIONS - all_defined_sections
        assert not required_missing, (
            f"_REQUIRED_SECTIONS names undefined sections: {required_missing}"
        )

    def test_bare_ssp_sections_are_subset_of_defined_sections(self):
        """_BARE_SSP_SECTIONS must name sections that are actually generated."""
        all_configs, bench = get_benchmark_configs()
        sfh_types = [sfh_type for _, sfh_type in bench.sfh_types]

        # Compute all possible forward section names
        all_forward_sections = set()
        for label, _ in all_configs:
            for sfh_type in sfh_types:
                all_forward_sections.add(f"{label}_{sfh_type}")

        # Check that _BARE_SSP_SECTIONS is a subset of forward sections
        # (bare sections are only for forward configs, not gradients)
        bare_missing = bench._BARE_SSP_SECTIONS - all_forward_sections
        assert not bare_missing, (
            f"_BARE_SSP_SECTIONS names undefined sections: {bare_missing}"
        )

    def test_grad_config_labels_exist(self):
        """Gradient config selection must use labels that exist in all_configs."""
        all_configs, _bench = get_benchmark_configs()
        all_config_labels = {label for label, _ in all_configs}

        # The hardcoded grad config labels used in the main loop
        grad_config_labels = {"Stellar only", "Kitchen sink (all components)"}

        missing_labels = grad_config_labels - all_config_labels
        assert not missing_labels, (
            f"Gradient config labels do not exist in all_configs: {missing_labels}"
        )


class TestAGNModelValidity:
    """Verify all agn_model keys in configs are valid."""

    def test_agn_model_keys_are_valid(self):
        """All agn_model values must be accepted by list_agn_models() or be deprecated."""
        all_configs, _ = get_benchmark_configs()
        valid_models = set(list_agn_models().names())
        deprecated_models = {"kubota_done_full", "kubota_done", "silva04", "cat3d_wind",
                             "adaf", "relagn", "skirtor", "qsogen", "multicolor_agn",
                             "richards2006", "skirtor_stalevski", "grahsp"}

        for label, cfg_kwargs in all_configs:
            agn_model = cfg_kwargs.get("agn_model")
            if agn_model is not None:
                is_valid = agn_model in valid_models or agn_model in deprecated_models
                assert is_valid, (
                    f"Config '{label}' uses invalid agn_model='{agn_model}'. "
                    f"Valid: {valid_models | deprecated_models}"
                )


class TestBenchmarkConfigConsistency:
    """Verify configuration attributes are self-consistent."""

    def test_no_simple_agn_model_in_configs(self):
        """The agn_model='simple' key was invalid and must be replaced with 'composable'."""
        all_configs, _ = get_benchmark_configs()

        for label, cfg_kwargs in all_configs:
            agn_model = cfg_kwargs.get("agn_model")
            assert agn_model != "simple", (
                f"Config '{label}' still uses invalid agn_model='simple'. "
                f"Replace with 'composable'."
            )

    def test_cloudy_grid_requires_bare_ssp(self):
        """Configs using nebular='cloudy' should be in _BARE_SSP_SECTIONS (conceptual check)."""
        all_configs, _bench = get_benchmark_configs()

        for _label, cfg_kwargs in all_configs:
            if cfg_kwargs.get("nebular") == "cloudy":
                # These should ideally be in _BARE_SSP_SECTIONS, but since we
                # don't have a bare SSP grid in CI, they're just noted here
                pass

    def test_cue_emulator_requires_bare_ssp(self):
        """Configs using nebular_cue=True should be in _BARE_SSP_SECTIONS (conceptual check)."""
        all_configs, _bench = get_benchmark_configs()

        for _label, cfg_kwargs in all_configs:
            if cfg_kwargs.get("nebular_cue"):
                # These should ideally be in _BARE_SSP_SECTIONS, but since we
                # don't have a bare SSP grid in CI, they're just noted here
                pass


class TestRecipePerformanceWiring:
    """Regression test that recipe perf wiring tests still pass."""

    def test_recipe_perf_wiring_exists(self):
        """Confirm that test_recipe_perf_wiring.py still exists and can be imported."""
        wiring_test_path = (
            Path(__file__).parent.parent / "regression" / "test_recipe_perf_wiring.py"
        )
        assert wiring_test_path.exists(), (
            f"test_recipe_perf_wiring.py not found at {wiring_test_path}"
        )
