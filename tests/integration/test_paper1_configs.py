"""Integration tests for Paper I SED model configurations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

# Load configs module by path to avoid sys.path issues
CONFIG_MODULE_PATH = Path(__file__).parent.parent.parent / "analysis" / "paper1" / "configs.py"
spec = importlib.util.spec_from_file_location("configs", CONFIG_MODULE_PATH)
configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configs)

import tengri

CANDELS_FILTERS = [
    "hst_f435w",
    "hst_f606w",
    "hst_f775w",
    "hst_f814w",
    "hst_f850lp",
    "hst_f105w",
    "hst_f125w",
    "hst_f160w",
    "vista_ks",
    "irac_36",
    "irac_45",
    "irac_58",
    "irac_80",
]
TEST_Z = 1.0


@pytest.fixture
def observation():
    return tengri.Observation(photometry=tengri.Photometry.from_names(CANDELS_FILTERS))


@pytest.fixture
def test_key():
    return jax.random.PRNGKey(42)


@pytest.mark.parametrize("key", ["I", "II", "III"])
def test_config_builds(key: str, observation):
    try:
        ssp = configs.load_ssp_for(key)
        builder = getattr(configs, f"config_{key}")
        model = builder(ssp, observation, TEST_Z)
        assert model is not None
    except FileNotFoundError:
        pytest.skip(f"SSP grid for config {key} not found")


@pytest.mark.parametrize("key", ["I", "II", "III"])
def test_config_free_params_count(key: str, observation):
    try:
        ssp = configs.load_ssp_for(key)
        builder = getattr(configs, f"config_{key}")
        model = builder(ssp, observation, TEST_Z)
        expected_n_free = configs.CONFIGS[key]["n_free"]
        actual_n_free = len(model.spec.free_params)
        if expected_n_free is not None:
            assert actual_n_free == expected_n_free, (
                f"Expected {expected_n_free}, got {actual_n_free}"
            )
        else:
            assert actual_n_free > 0
    except FileNotFoundError:
        pytest.skip(f"SSP grid for config {key} not found")


@pytest.mark.parametrize("key", ["I", "II", "III"])
def test_config_prediction_finite_positive(key: str, observation, test_key):
    try:
        ssp = configs.load_ssp_for(key)
        builder = getattr(configs, f"config_{key}")
        model = builder(ssp, observation, TEST_Z)
        sample = model.spec.sample(key=test_key)
        pred = model.predict_photometry(sample)
        assert pred.shape == (len(CANDELS_FILTERS),)
        assert (pred > 0).all(), "Prediction contains non-positive values"
    except FileNotFoundError:
        pytest.skip(f"SSP grid for config {key} not found")


@pytest.mark.parametrize("key", ["I", "II", "III"])
def test_config_sfh_type_resolved(key: str, observation):
    """Assert the resolved SFH type matches the configuration."""
    try:
        ssp = configs.load_ssp_for(key)
        builder = getattr(configs, f"config_{key}")
        model = builder(ssp, observation, TEST_Z)
        groups = model.spec.to_groups()
        sfh_group = groups["sfh"]

        if key == "I":
            assert sfh_group["type"] == "delayed", (
                f"Config I SFH should be 'delayed', got {sfh_group['type']}"
            )
        elif key == "II":
            assert sfh_group["type"] == "dpl", (
                f"Config II SFH should be 'dpl', got {sfh_group['type']}"
            )
        elif key == "III":
            assert sfh_group["type"] == "continuity", (
                f"Config III SFH should be 'continuity', got {sfh_group['type']}"
            )
    except FileNotFoundError:
        pytest.skip(f"SSP grid for config {key} not found")


@pytest.mark.parametrize("key", ["I", "II", "III"])
def test_config_dust_attenuation_resolved(key: str, observation):
    """Assert the resolved dust attenuation law and type."""
    try:
        ssp = configs.load_ssp_for(key)
        builder = getattr(configs, f"config_{key}")
        model = builder(ssp, observation, TEST_Z)
        groups = model.spec.to_groups()
        dust_att = groups["dust_attenuation"]

        if key == "I":
            assert dust_att["type"] == "single_component", (
                f"Config I should be single_component, got {dust_att['type']}"
            )
            assert dust_att["law"] == "calzetti", (
                f"Config I dust law should be 'calzetti', got {dust_att['law']}"
            )
        elif key == "II":
            assert dust_att["type"] == "two_component", (
                f"Config II should be two_component, got {dust_att['type']}"
            )
            assert dust_att["law"] == "calzetti", (
                f"Config II dust law should be 'calzetti', got {dust_att['law']}"
            )
        elif key == "III":
            assert dust_att["type"] == "two_component", (
                f"Config III should be two_component, got {dust_att['type']}"
            )
            assert dust_att["law"] == "kriek_conroy", (
                f"Config III dust law should be 'kriek_conroy', got {dust_att['law']}"
            )
    except FileNotFoundError:
        pytest.skip(f"SSP grid for config {key} not found")


@pytest.mark.parametrize("key", ["I", "II", "III"])
def test_config_dust_emission_resolved(key: str, observation):
    """Assert the resolved dust emission type."""
    try:
        ssp = configs.load_ssp_for(key)
        builder = getattr(configs, f"config_{key}")
        model = builder(ssp, observation, TEST_Z)
        groups = model.spec.to_groups()
        dust_em = groups["dust_emission"]

        if key == "I":
            assert dust_em["type"] == "dale2014", (
                f"Config I dust emission should be 'dale2014', got {dust_em['type']}"
            )
        elif key == "II":
            assert dust_em["type"] == "dl07", (
                f"Config II dust emission should be 'dl07', got {dust_em['type']}"
            )
        elif key == "III":
            assert dust_em["type"] == "dl07", (
                f"Config III dust emission should be 'dl07', got {dust_em['type']}"
            )
    except FileNotFoundError:
        pytest.skip(f"SSP grid for config {key} not found")


@pytest.mark.parametrize("key", ["I", "II", "III"])
def test_config_nebular_backend_resolved(key: str, observation):
    """Assert the resolved nebular backend type."""
    try:
        ssp = configs.load_ssp_for(key)
        builder = getattr(configs, f"config_{key}")
        model = builder(ssp, observation, TEST_Z)
        groups = model.spec.to_groups()
        neb = groups["neb"]

        if key == "I":
            assert neb["type"] == "cue", f"Config I nebular should be 'cue', got {neb['type']}"
        elif key == "II":
            assert neb["type"] == "ssp", f"Config II nebular should be 'ssp', got {neb['type']}"
        elif key == "III":
            assert neb["type"] == "cue", f"Config III nebular should be 'cue', got {neb['type']}"
    except FileNotFoundError:
        pytest.skip(f"SSP grid for config {key} not found")


@pytest.mark.parametrize("key", ["I", "II", "III"])
def test_config_free_param_names_exact(key: str, observation):
    """Assert the exact list of free parameter names."""
    try:
        ssp = configs.load_ssp_for(key)
        builder = getattr(configs, f"config_{key}")
        model = builder(ssp, observation, TEST_Z)
        free_params = model.spec.free_params

        if key == "I":
            expected = [
                "dust_tau_v",
                "met_logzsol",
                "sfh_delayed_age_gyr",
                "sfh_delayed_log_total_mass",
                "sfh_delayed_tau_gyr",
            ]
            assert sorted(free_params) == sorted(expected), "Config I free params mismatch"
        elif key == "II":
            expected = [
                "dust_tau_bc",
                "dust_tau_diff",
                "met_logzsol",
                "sfh_dpl_age_gyr",
                "sfh_dpl_alpha",
                "sfh_dpl_beta",
                "sfh_dpl_log_total_mass",
                "sfh_dpl_tau_gyr",
            ]
            assert sorted(free_params) == sorted(expected), "Config II free params mismatch"
        elif key == "III":
            expected = [
                "dust_tau_bc",
                "dust_tau_diff",
                "met_logzsol",
                "neb_logU",
                "sfh_cont_log_total_mass",
                "sfh_cont_ratio_0",
                "sfh_cont_ratio_1",
                "sfh_cont_ratio_2",
                "sfh_cont_ratio_3",
                "sfh_cont_ratio_4",
                "sfh_cont_ratio_5",
            ]
            assert sorted(free_params) == sorted(expected), "Config III free params mismatch"
    except FileNotFoundError:
        pytest.skip(f"SSP grid for config {key} not found")
