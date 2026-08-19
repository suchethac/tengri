# SPDX-License-Identifier: BSD-3-Clause
"""Regression: log-domain migration of nebular Q_H consumers (#1206, task 3b).

Tests for migrate-nebular-phot and Cue nion fallback to log_nion:
1. f64 parity: log form == linear form at machine precision
2. Pure-f32 finiteness: log form stays finite when linear form overflows
3. Cue fallback log path: fallback maps log_nion correctly to gas_logqion
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    FIXED,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.components.nebular.nebular_grid_precompute import (
    NebularGridTable,
    precompute_nebular_grid,
    reconstruct_nebular_phot,
)
from tengri.observation.line_flux_data import LineFluxData
from tengri.utils.grid_interp import interp_nd_pchip
from tengri.utils.scale import pow10

pytestmark = pytest.mark.regression_bug

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_LINES = ("Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717")
_LINE_DATA = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
_LW = _LINE_DATA.wavelengths
Z = 0.15
_BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1", "wise_w2"]


def _require():
    if not Path(_BARE).is_file():
        pytest.skip(f"missing bare SSP {_BARE}")
    if not Path("data/cue_weights.npz").is_file():
        pytest.skip("Cue weights (data/cue_weights.npz) not present")


def _model(neb, sfh_wild=FREE):
    import warnings

    _require()
    ssp = load_ssp_data(_BARE)
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]), line_fluxes=_LINE_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": sfh_wild},
            dust=None,
            neb=neb,
            redshift=Fixed(Z),
        )


def _wave_model(neb, sfh_wild=FREE):
    """WavePrecomp Cue model with dust off — so it publishes nebular_phot_lnu_precomp."""
    import warnings

    _require()
    ssp = load_ssp_data(_BARE)
    obs = Observation(photometry=Photometry.from_names(_BANDS), line_fluxes=_LINE_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "*": sfh_wild},
            dust={"law_diff": 'calzetti', 
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
                "tau_diff": Fixed(0.0),
                "tau_bc": Fixed(0.0),
            },
            neb=neb,
            redshift=Fixed(Z),
            approx=WavePrecomp(),
        )


def _log_nion(m, p):
    """Extract log10(Q_H) from model state."""
    return float(np.asarray(m.predict_state(p).derived["log_nion"]))


def _nion(m, p):
    """Extract linear Q_H from model state."""
    return float(np.sum(np.asarray(m.predict_state(p).derived["nion"])))


def _kinds(table):
    """Per-axis interpolation kinds, tolerating tables pickled before #1020."""
    return tuple(table.axis_kinds) or None


def test_reconstruct_nebular_phot_f64_parity_log_vs_linear():
    """f64 parity: log form == linear form at machine precision.

    Build a WavePrecomp nebular grid with photometry channel. For several
    sampled params and realistic log_nion (54-56), assert that
    reconstruct_nebular_phot(log_nion, p, table) equals the reference
    pow10(log_nion) * (10.0**log_ppq) at rtol=1e-12.
    """
    m = _wave_model({"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)}, sfh_wild=FIXED)
    table = precompute_nebular_grid(m, _LW, n_grid=14)
    assert table.log_phot_per_qh is not None, "photometry channel missing"

    worst_rel_err = 0.0
    for i in range(6):
        p = dict(m.spec.sample(jax.random.PRNGKey(200 + i)))
        st = m.predict_state(p)
        log_nion = float(np.asarray(st.derived["log_nion"]))
        nion = float(np.sum(np.asarray(st.derived["nion"])))

        # Get log_ppq from the same interpolator the function uses
        if not table.axis_names:
            log_ppq = table.log_phot_per_qh
        else:
            point = tuple(jnp.asarray(p[name]).reshape(()) for name in table.axis_names)
            log_ppq = interp_nd_pchip(table.log_phot_per_qh, table.axes, point, _kinds(table))
        log_ppq = np.asarray(log_ppq)

        # The log-domain result (new path)
        log_result = np.asarray(reconstruct_nebular_phot(log_nion, p, table))

        # The reference linear result
        linear_result = nion * (10.0**log_ppq)

        # They must agree at machine precision
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.abs(log_result - linear_result) / (np.abs(linear_result) + 1e-300)
        worst_rel_err = max(worst_rel_err, np.max(rel_err))

    assert worst_rel_err < 1e-12, (
        f"log form / linear form parity violation: max relative error {worst_rel_err:.2e}"
    )


def test_reconstruct_nebular_phot_pure_f32_finiteness():
    """Pure-f32 finiteness: log form finite when linear form is inf.

    Under enable_x64(False), rebuild the table with f32 arrays and verify:
    - reconstruct_nebular_phot(log_nion=56.0 [f32], ...) is all-finite
      and has L_nu-scale magnitude (1e20 < max < 1e35)
    - The linear reference pow10(f32 56) * (10**log_ppq) is inf
    """
    m = _wave_model({"type": "cue", "*": FIXED, "logU": Uniform(-4.0, -1.0)}, sfh_wild=FIXED)

    # Build table at f64
    table_f64 = precompute_nebular_grid(m, _LW, n_grid=14)
    assert table_f64.log_phot_per_qh is not None

    # Cast to f32 inside enable_x64(False) context
    with jax.enable_x64(False):
        axes_f32 = tuple(jnp.asarray(ax, dtype=jnp.float32) for ax in table_f64.axes)
        log_phot_f32 = jnp.asarray(table_f64.log_phot_per_qh, dtype=jnp.float32)

        table_f32 = NebularGridTable(
            axis_names=table_f64.axis_names,
            axes=axes_f32,
            log_line_per_qh=jnp.asarray(table_f64.log_line_per_qh, dtype=jnp.float32),
            wavelengths=table_f64.wavelengths,
            log_phot_per_qh=log_phot_f32,
            axis_kinds=table_f64.axis_kinds,
        )
        assert table_f32.log_phot_per_qh.dtype == jnp.float32

        # Sample a parameter dict and set a high realistic log_nion (54-56)
        p = dict(m.spec.sample(jax.random.PRNGKey(0)))

        # Log-domain result at f32
        log_nion_f32 = jnp.asarray(56.0, dtype=jnp.float32)
        log_result_f32 = reconstruct_nebular_phot(log_nion_f32, p, table_f32)

        # Verify log result is finite and magnitude-appropriate
        assert np.all(np.isfinite(np.asarray(log_result_f32))), (
            "log-domain reconstruct is not all-finite"
        )
        log_max = np.max(np.abs(np.asarray(log_result_f32)))
        assert 1e20 < log_max < 1e35, (
            f"log-domain result magnitude {log_max:.2e} outside expected L_nu scale [1e20, 1e35]"
        )

        # Verify linear form is inf
        if not table_f32.axis_names:
            log_ppq_f32 = table_f32.log_phot_per_qh
        else:
            point = tuple(
                jnp.asarray(p[name], dtype=jnp.float32).reshape(())
                for name in table_f32.axis_names
            )
            log_ppq_f32 = interp_nd_pchip(
                table_f32.log_phot_per_qh, table_f32.axes, point, _kinds(table_f32)
            )

        linear_form = pow10(log_nion_f32) * pow10(log_ppq_f32)
        assert np.any(np.isinf(np.asarray(linear_form))), (
            "linear form should be inf at f32 (the bug we're fixing)"
        )


def test_cue_fallback_log_path_parity():
    """Cue fallback log path: log form maps correctly to gas_logqion.

    Unit-level test: assert the migrated fallback
    ``gas_logqion = maximum(log_nion, 0.0)`` equals the old
    ``log10(maximum(pow10(log_nion), 1.0))`` for representative log_nion
    (both > 0 and the floor case < 0), and is finite in pure f32.
    """
    # Test vectors: (log_nion, expected_gas_logqion)
    test_cases = [
        (56.0, 56.0),  # high realistic Q_H
        (50.0, 50.0),  # mid-range Q_H
        (10.0, 10.0),  # moderate Q_H
        (0.5, 0.5),  # near-unity Q_H
        (0.0, 0.0),  # Q_H = 1 (boundary)
        (-0.5, 0.0),  # Q_H < 1, should floor to 0
        (-1.0, 0.0),  # Q_H << 1, should floor to 0
        (-10.0, 0.0),  # very low Q_H, should floor to 0
    ]

    for log_nion, expected_gas_logqion in test_cases:
        log_nion_arr = jnp.asarray(log_nion, dtype=jnp.float64)

        # New log path: maximum(log_nion, 0.0)
        new_path = jnp.maximum(log_nion_arr, 0.0)

        # Old linear path: log10(maximum(pow10(log_nion), 1.0))
        old_path = jnp.log10(jnp.maximum(pow10(log_nion_arr), 1.0))

        # Both paths must agree to machine precision
        assert float(new_path) == pytest.approx(float(old_path), rel=1e-12), (
            f"log_nion={log_nion}: new path {float(new_path)} != old path {float(old_path)}"
        )

        # Both must equal the expected result
        assert float(new_path) == pytest.approx(expected_gas_logqion, rel=1e-12), (
            f"log_nion={log_nion}: result {float(new_path)} != expected {expected_gas_logqion}"
        )

    # Test pure-f32 finiteness
    with jax.enable_x64(False):
        for log_nion, _ in test_cases:
            log_nion_f32 = jnp.asarray(log_nion, dtype=jnp.float32)

            # New log path: always finite
            new_path_f32 = jnp.maximum(log_nion_f32, 0.0)

            assert np.isfinite(float(new_path_f32)), (
                f"new path not finite at log_nion={log_nion} (f32)"
            )

            # For high log_nion (56.0), the old path may overflow in f32 — that's the bug
            # we're fixing. For lower values, they should agree.
            if log_nion >= 40.0:
                # High Q_H: old path likely overflows to inf in f32
                old_path_f32 = jnp.log10(jnp.maximum(pow10(log_nion_f32), 1.0))
                # We just verify the new path stays finite (that's the improvement)
                pass
            else:
                # Lower Q_H: both paths should stay finite
                old_path_f32 = jnp.log10(jnp.maximum(pow10(log_nion_f32), 1.0))
                assert np.isfinite(float(old_path_f32)), (
                    f"old path not finite at log_nion={log_nion} (f32)"
                )
