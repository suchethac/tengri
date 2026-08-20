# SPDX-License-Identifier: BSD-3-Clause
"""Tests for fit_batch as a deprecated wrapper over Catalog.

Verifies that the legacy fit_batch function still works but emits a
one-shot DeprecationWarning, and that it returns the same shape as before.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract


@pytest.fixture
def minimal_sed_model(synthetic_ssp_wide, simple_observation):
    """Minimal SEDModel for testing."""
    from tengri import FIXED, FREE, SEDModel

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_bc": 0.5,
        },
        neb={"type": "none"},
        redshift=FIXED,
    )
    return model


@pytest.fixture
def catalog_single_row(minimal_sed_model):
    """Single-row catalog dict."""
    # Generate synthetic flux from the model using prior midpoints
    true_params = {
        "sfh_dpl_alpha": 1.5,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 4.0,
        "sfh_dpl_age_gyr": 8.0,
        "sfh_dpl_log_total_mass": 0.9,
        "met_logzsol": -0.3,
        "dust_tau_bc": 0.5,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }
    flux_true = minimal_sed_model.predict_photometry(true_params)
    noise = jnp.abs(flux_true) * 0.1 + 1e-30

    return [
        {
            "flux_u": float(flux_true[0]),
            "flux_g": float(flux_true[1]),
            "flux_r": float(flux_true[2]),
            "err_u": float(noise[0]),
            "err_g": float(noise[1]),
            "err_r": float(noise[2]),
        }
    ]


@pytest.fixture
def catalog_multi_row(minimal_sed_model):
    """Multi-row catalog dict (3 rows)."""
    true_params = {
        "sfh_dpl_alpha": 1.5,
        "sfh_dpl_beta": 1.0,
        "sfh_dpl_tau_gyr": 4.0,
        "sfh_dpl_age_gyr": 8.0,
        "sfh_dpl_log_total_mass": 0.9,
        "met_logzsol": -0.3,
        "dust_tau_bc": 0.5,
        "dust_tau_diff": 0.3,
        "dust_slope": -0.7,
        "redshift": 0.1,
    }
    flux_true = minimal_sed_model.predict_photometry(true_params)
    noise = jnp.abs(flux_true) * 0.1 + 1e-30

    return [
        {
            "flux_u": float(flux_true[0]),
            "flux_g": float(flux_true[1]),
            "flux_r": float(flux_true[2]),
            "err_u": float(noise[0]),
            "err_g": float(noise[1]),
            "err_r": float(noise[2]),
        }
        for _ in range(3)
    ]


def test_fit_batch_emits_one_shot_deprecation(minimal_sed_model, catalog_multi_row):
    """fit_batch must emit exactly one DeprecationWarning for a multi-row catalog.

    This ensures the deprecation is efficient (one warning, not per row).
    """
    # Capture all warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = minimal_sed_model.fit_batch(
            catalog_multi_row,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            verbose=False,
        )

    # Filter for DeprecationWarnings about Catalog
    deprecation_warnings = [
        warning
        for warning in w
        if issubclass(warning.category, DeprecationWarning) and "Catalog" in str(warning.message)
    ]

    # Should have exactly one DeprecationWarning mentioning Catalog
    assert len(deprecation_warnings) == 1, (
        f"Expected exactly 1 DeprecationWarning mentioning 'Catalog', "
        f"got {len(deprecation_warnings)}"
    )
    assert "Catalog" in str(deprecation_warnings[0].message)


def test_fit_batch_returns_list_of_posteriors(minimal_sed_model, catalog_single_row):
    """fit_batch must return a list of Posterior objects (legacy shape)."""
    from tengri.inference.posterior import Posterior

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        results = minimal_sed_model.fit_batch(
            catalog_single_row,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            verbose=False,
        )

    # Should return a list
    assert isinstance(results, list), f"Expected list, got {type(results)}"

    # Should have one result per row
    assert len(results) == len(catalog_single_row), (
        f"Expected {len(catalog_single_row)} results, got {len(results)}"
    )

    # Each result should be a Posterior
    for i, result in enumerate(results):
        assert isinstance(result, Posterior), f"Result {i} is not a Posterior, got {type(result)}"

    # Each Posterior should have .params (non-degenerate)
    for i, result in enumerate(results):
        assert result.params is not None, f"Result {i} has no params"
        assert len(result.params) > 0, f"Result {i} has empty params"


def test_fit_batch_multi_row_returns_correct_length(minimal_sed_model, catalog_multi_row):
    """fit_batch must return one result per catalog row."""
    from tengri.inference.posterior import Posterior

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        results = minimal_sed_model.fit_batch(
            catalog_multi_row,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            verbose=False,
        )

    # Should have 3 results for 3 rows
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # All should be Posterior objects with params
    for i, result in enumerate(results):
        assert isinstance(result, Posterior), f"Result {i} is not a Posterior"
        assert result.params is not None, f"Result {i} has no params"


def test_fit_batch_output_dir_checkpoint_preserved(tmp_path, minimal_sed_model, catalog_multi_row):
    """fit_batch's output_dir checkpoint/resume feature must survive the Catalog
    delegation: output_dir keeps the per-row loop (Catalog has no persistence)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = minimal_sed_model.fit_batch(
            catalog_multi_row,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            output_dir=str(tmp_path),
            verbose=False,
        )
        saved = sorted(p.name for p in tmp_path.glob("*.h5"))
        # Re-run: every galaxy loads from its checkpoint instead of refitting.
        r2 = minimal_sed_model.fit_batch(
            catalog_multi_row,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            output_dir=str(tmp_path),
            verbose=False,
        )
    assert len(r1) == 3 and len(saved) == 3 and len(r2) == 3


def test_fit_batch_forwards_explicit_approx(minimal_sed_model, catalog_multi_row, monkeypatch):
    """An explicit approx= must reach the model, not be silently dropped.

    Regression (#1336 follow-up): the delegation built the ForwardModel without
    applying approx=, so a caller passing e.g. WavePrecomp silently got "auto".
    """
    from tengri import WavePrecomp
    from tengri.forward import forward_model as fm

    seen = []
    orig = fm.ForwardModel.with_approx
    monkeypatch.setattr(
        fm.ForwardModel,
        "with_approx",
        lambda self, approx: (seen.append(approx), orig(self, approx))[1],
    )
    wp = WavePrecomp(catalog_z_range=(0.01, 2.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        minimal_sed_model.fit_batch(
            catalog_multi_row,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            approx=wp,
            verbose=False,
        )
    assert wp in seen, "explicit approx= was not forwarded to the model (silently dropped)"


def test_fit_batch_verbose_false_silences_progress(minimal_sed_model, catalog_multi_row, capsys):
    """verbose=False must suppress the engine's per-galaxy progress output."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        minimal_sed_model.fit_batch(
            catalog_multi_row,
            flux_cols=["flux_u", "flux_g", "flux_r"],
            err_cols=["err_u", "err_g", "err_r"],
            method="map",
            verbose=False,
        )
    out = capsys.readouterr().out
    assert "galaxies done" not in out and "Galaxy" not in out, (
        f"verbose=False did not silence: {out!r}"
    )
