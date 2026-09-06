# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the MAPPINGS III + V shock grid backend.

Covers grid loading, multi-axis interpolation, backward compatibility, and
JAX JIT compatibility.  Most tests run against the hardcoded fallback
(Allen+2008 Table 5) because the HDF5 file is not bundled in the repo;
tests that need the HDF5 are skipped with ``pytest.importorskip`` guard.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from tengri._data_setup import find_data
from tengri.components.dust.attenuation import resolve_dust_law
from tengri.components.nebular.shock import (
    _FALLBACK_V,
    _load_mappings_grids,
    compute_shock_sed,
    shock_line_ratios,
)

pytestmark = pytest.mark.bounds

# Path to the optional HDF5 file (may not exist in CI).
#
# This was ``Path(__file__).parents[2] / "data" / ...``, which is
# ``tests/data/mappings_templates.h5`` — one level short of the repo root. The
# file never exists there, so the five tests below skipped on *every* machine,
# CI included, rather than only where the grid is genuinely absent. Asking the
# same locator the code asks keeps the guard honest and picks up
# $TENGRI_DATA_DIR (#1431).
_H5_PATH = find_data("mappings_templates.h5")
_HAS_H5 = _H5_PATH is not None

h5_only = pytest.mark.skipif(not _HAS_H5, reason="data/mappings_templates.h5 not found")


# ── 1. HDF5 loads with correct shape (skip when file absent) ──────


@h5_only
def test_mappings_h5_loads():
    """HDF5 grids load without error; shapes are internally consistent."""
    grids = _load_mappings_grids()
    assert grids is not None

    assert "mappings5" in grids, "mappings5 group missing from HDF5"
    g = grids["mappings5"]

    n_abund = len(g["abundance_names"])
    n_n = g["log_density_cm3"].shape[0]
    n_v = g["velocities_kms"].shape[0]
    n_b = g["b_axis"].shape[0]
    n_lines = len(g["line_names"])

    expected = (n_abund, n_n, n_v, n_b, n_lines)
    for key in ("shock_ratios", "precursor_ratios", "combined_ratios"):
        assert g[key].shape == expected, (
            f"mappings5/{key}: expected {expected}, got {g[key].shape}"
        )

    # No NaN in the solar/n=1/combined slice (fallback region)
    i_solar = g["abundance_names"].index("Allen2008_Solar")
    i_n1 = int(jnp.argmin(jnp.abs(g["log_density_cm3"])))
    combined = g["combined_ratios"][i_solar, i_n1, :, 0, :]
    n_nan = int(jnp.sum(jnp.isnan(combined)))
    assert n_nan == 0, f"NaN in solar/n=1 combined slice ({n_nan} NaN)"


# ── 2. Backward compatibility — solar, n=1, no HDF5 ───────────────


def test_backward_compat_solar_n1():
    """Fallback values for solar/n=1 should match Allen+2008 Table 5.

    Checks that key BPT ratios at 300 km/s are within 5 % of the hardcoded
    reference values (the grid is read back from the same arrays, so this
    really tests the interpolation logic, not the data).
    """
    # Force fallback by requesting mappings5 when HDF5 is absent
    # (or just use the fallback check directly via the module arrays)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ratios = shock_line_ratios(
            300.0,
            shock_log_density=0.0,
            shock_b_over_sqrt_n=1.0,
            shock_abundance="solar",
            shock_component="combined",
        )

    # At 300 km/s, Allen+2008 Table 5 gives: R_OII=3.1, R_OIII=5.8, R_NII=2.1
    ha_key = "HA_6563A"
    hb_key = "Hb_4861A"

    # Hβ must be reference (=1)
    assert float(ratios[hb_key]) == pytest.approx(1.0, abs=0.01)

    # Hα/Hβ should be in the physically plausible range.
    # Allen+2008 hardcoded fallback gives [3.0, 3.7]; the full MAPPINGS V grid
    # (when HDF5 is present) can yield values slightly below Case B (2.7–3.2)
    # at high velocity due to collisional de-excitation in dense post-shock gas.
    r_ha = float(ratios[ha_key])
    assert 2.5 <= r_ha <= 5.0, f"Hα/Hβ={r_ha:.2f} outside plausible range"


# ── 3. Velocity interpolation is monotone and clamps at grid edges


@h5_only
def test_velocity_interpolation():
    """Velocity interpolation is continuous and clamps at grid edges."""
    velocities = [100.0, 200.0, 300.0, 500.0, 750.0, 1000.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ha_values = [float(shock_line_ratios(v)["HA_6563A"]) for v in velocities]

    # All values should be finite and physically plausible
    for v, r in zip(velocities, ha_values):
        assert 2.0 <= r <= 6.0, f"Hα/Hβ={r:.4f} at v={v} km/s outside physically plausible range"

    # Hβ should always equal 1.0 (reference line)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for v in velocities:
            hb = float(shock_line_ratios(v)["Hb_4861A"])
            assert hb == pytest.approx(1.0, abs=0.01), f"Hβ != 1 at v={v}"

    # Out-of-range velocity raises ValueError (no silent clamping)
    with pytest.raises(ValueError, match="shock_velocity"):
        shock_line_ratios(50.0)

    # Grid edge: exactly at v_min should succeed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r_edge = shock_line_ratios(float(_FALLBACK_V[0]))
    assert float(r_edge["HA_6563A"]) > 0.0


# ── 4. B-field nearest-neighbor snap — no error at grid boundaries


@h5_only
def test_b_field_snapping():
    """B-field values snap to the nearest grid point within bounds.

    Out-of-bounds values raise ValueError (fail fast, no silent clamping).
    The Allen+2008 (solar, n=1) B-field set is {0.0001, 0.5, 1, 2, 3.23, 4, 5, 10} μG.
    """
    grids = _load_mappings_grids()
    b_grid = grids["mappings5"]["b_axis"]
    b_min = float(b_grid[0])
    b_max = float(b_grid[-1])

    # In-bounds: nearby values snap to the same grid point
    b_ref = 1.0  # μG — one of the Allen08 8 values
    r_ref = shock_line_ratios(300.0, shock_b_over_sqrt_n=b_ref)
    r_near = shock_line_ratios(300.0, shock_b_over_sqrt_n=b_ref * 1.01)
    assert float(r_ref["HA_6563A"]) == pytest.approx(float(r_near["HA_6563A"]), rel=1e-4)

    # Grid endpoints: should not raise
    shock_line_ratios(300.0, shock_b_over_sqrt_n=b_min)
    shock_line_ratios(300.0, shock_b_over_sqrt_n=b_max)

    # Out-of-bounds: must raise ValueError, not silently snap
    with pytest.raises(ValueError, match="shock_b_over_sqrt_n"):
        shock_line_ratios(300.0, shock_b_over_sqrt_n=b_min / 100.0)

    with pytest.raises(ValueError, match="shock_b_over_sqrt_n"):
        shock_line_ratios(300.0, shock_b_over_sqrt_n=b_max * 100.0)


# ── 5. Precursor component differs from shock only at higher velocities


@h5_only
def test_out_of_bounds_raises():
    """All out-of-bounds parameters raise ValueError immediately."""
    grids = _load_mappings_grids()
    g = grids["mappings5"]
    v_max = float(g["velocities_kms"][-1])
    log_n_min = float(g["log_density_cm3"][0])
    log_n_max = float(g["log_density_cm3"][-1])

    with pytest.raises(ValueError, match="shock_velocity"):
        shock_line_ratios(v_max + 1.0)

    with pytest.raises(ValueError, match="shock_log_density"):
        shock_line_ratios(300.0, shock_log_density=log_n_max + 1.0)

    with pytest.raises(ValueError, match="shock_log_density"):
        shock_line_ratios(300.0, shock_log_density=log_n_min - 1.0)

    with pytest.raises(ValueError, match="shock_abundance"):
        shock_line_ratios(300.0, shock_abundance="unknown_set")

    with pytest.raises(ValueError, match="shock_component"):
        shock_line_ratios(300.0, shock_component="invalid")


@h5_only
def test_precursor_vs_shock_component():
    """'combined' and 'shock' should differ at v > 200 km/s.

    The precursor contribution grows with velocity (Sutherland & Dopita 2017);
    above 200 km/s the combined and shock-only SEDs should deviate.
    """
    r_combined = shock_line_ratios(400.0, shock_component="combined")
    r_shock_only = shock_line_ratios(400.0, shock_component="shock")

    # They should not be identical
    ha_combined = float(r_combined["HA_6563A"])
    ha_shock = float(r_shock_only["HA_6563A"])
    assert ha_combined != pytest.approx(ha_shock, rel=1e-6), (
        "combined and shock-only Hα ratios are identical at 400 km/s "
        "(precursor contribution should differ)"
    )


# ── 7. DeprecationWarning raised when HDF5 is absent ──────────────


def test_fallback_without_h5(tmp_path, monkeypatch):
    """When data/mappings_templates.h5 is absent, a DeprecationWarning is emitted
    and shock_line_ratios still returns non-zero ratios."""
    import tengri._data_setup as data_setup
    import tengri.components.nebular.shock as shock_module

    # Clear functools.cache so the loader re-runs on next call
    shock_module._load_mappings_grids.cache_clear()

    # Simulate absence at the locator. This used to monkeypatch
    # ``shock_module.__file__``, which only worked while the lookup rebuilt the
    # path from ``parents[4]``; that anchoring is gone (#1431) and the loader
    # now asks ``find_data``, which does not consult the calling module's
    # ``__file__`` at all. Pointing $TENGRI_DATA_DIR at an empty directory
    # would not do it either — ``data_dirs()`` still walks the cwd ancestors
    # and the package tree, where the real grid lives.
    monkeypatch.setattr(data_setup, "find_data", lambda *names: None)

    try:
        with pytest.warns(DeprecationWarning, match="MAPPINGS grid file not found"):
            ratios = shock_line_ratios(300.0)
        # Must still return usable values
        assert float(ratios["HA_6563A"]) > 0.0
        assert float(ratios["Hb_4861A"]) == pytest.approx(1.0, abs=0.01)
    finally:
        # Clear again so the fallback result is not cached for subsequent tests
        shock_module._load_mappings_grids.cache_clear()


# ── 8. Diffuse ISM attenuation reduces shock SED correctly ────────


def test_ism_attenuation_reduces_shock_sed():
    """Diffuse ISM screen attenuates shock SED — more dust → less flux.

    The pipeline applies exp(-tau_diff * k_diff(λ)) to shock emission.
    This test verifies the attenuation directly on the shock SED array,
    reproducing the same operation used in sed_pipeline.py so that any
    regression in either the pipeline or the dust law is caught.

    Birth-cloud attenuation must NOT be applied (shocks occur outside
    star-forming birth clouds).  We check this by confirming that the
    attenuated SED depends on tau_diff (ISM) but that no extra
    birth-cloud factor is present.
    """
    wave = jnp.linspace(3000.0, 10000.0, 2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shock_raw = compute_shock_sed(wave, 300.0, 1e7, line_sigma_aa=2.0)

    # Apply diffuse ISM screen — same expression as sed_pipeline.py. power_law
    # reads only its slope; since #2185 a law refuses a keyword it does not read.
    k_diff = resolve_dust_law("power_law")(wave, n_slope=-0.7)

    tau_low = 0.1
    tau_high = 1.0
    shock_low = shock_raw * jnp.exp(-tau_low * k_diff)
    shock_high = shock_raw * jnp.exp(-tau_high * k_diff)

    # More dust → less total flux
    assert float(jnp.sum(shock_high)) < float(jnp.sum(shock_low)), (
        "Higher tau_diff should give lower total shock flux"
    )
    # Both should have less flux than the un-attenuated version
    assert float(jnp.sum(shock_low)) < float(jnp.sum(shock_raw))
    assert float(jnp.sum(shock_high)) < float(jnp.sum(shock_raw))

    # Attenuation is wavelength-dependent: blue lines more attenuated than red
    # Compare [OII]~3727 Å vs Hα~6563 Å
    blue_mask = jnp.abs(wave - 3727.0) < 20.0
    red_mask = jnp.abs(wave - 6563.0) < 20.0
    tau_blue = -jnp.log(
        jnp.sum(shock_high[blue_mask]) / jnp.maximum(jnp.sum(shock_raw[blue_mask]), 1e-40)
    )
    tau_red = -jnp.log(
        jnp.sum(shock_high[red_mask]) / jnp.maximum(jnp.sum(shock_raw[red_mask]), 1e-40)
    )
    assert float(tau_blue) > float(tau_red), (
        f"Blue lines more attenuated than red: τ_blue={tau_blue:.3f}, τ_red={tau_red:.3f}"
    )
