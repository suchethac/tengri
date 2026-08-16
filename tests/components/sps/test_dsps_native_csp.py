# SPDX-License-Identifier: BSD-3-Clause
"""Tests for DSPS CSP integration modes: 'dsps_native' and 'dsps_met_table'.

Tests :func:`compute_dsps_native_weights`, :func:`compute_dsps_met_table_weights`,
and the SEDModel-level ``csp_integration='dsps_native'`` / ``'dsps_met_table'`` options.

All tests are CPU-only (no SSP file required).
"""

import chex
import pytest

pytestmark = pytest.mark.bounds

pytest.importorskip("dsps", reason="dsps not installed")

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.stellar.sps.dsps_wrapper import (
    compute_dsps_met_table_weights,
    compute_dsps_native_weights,
)
from tests._bounds import assert_non_negative
from tests._grad_parity import assert_grad_matches_fd


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Minimal synthetic SSP grid (no file I/O) ──────────────────────

N_MET = 5
N_AGE = 20
N_WAVE = 10
RNG = np.random.default_rng(42)

SSP_LGMET = np.linspace(-2.5, -0.5, N_MET)  # log10(Z), absolute
T_OBS_GYR = 10.0  # age of universe at observation (Gyr)
# All SSP ages must be strictly < T_OBS_GYR so that
# log10(t_obs - age) is finite inside DSPS's birth-time interpolation.
# Here: max age = 10^(0.9) Gyr ≈ 7.9 Gyr < 10.0 Gyr.
SSP_LG_AGE_GYR = np.linspace(-3.0, 0.9, N_AGE)  # log10(age/Gyr)
SSP_AGES_YR = 10.0 ** (SSP_LG_AGE_GYR + 9.0)  # years
SSP_FLUX = RNG.random((N_MET, N_AGE, N_WAVE)).astype(np.float64)

LGMET = -1.5  # log10(Z) of galaxy
LGMET_SCATTER = 0.2


def _flat_sfr(n_age: int = N_AGE) -> jnp.ndarray:
    """Flat SFR = 1 Msun/yr at all SSP ages."""
    return jnp.ones(n_age, dtype=jnp.float64)


# ── compute_dsps_native_weights: unit tests ───────────────────────


def test_output_shapes():
    """age_weights_msun and ssp_flux_at_z have correct shapes."""
    sfr = _flat_sfr()
    aw, flux_z = compute_dsps_native_weights(
        sfr,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET,
        LGMET_SCATTER,
    )
    assert aw.shape == (N_AGE,), f"age_weights shape {aw.shape} != ({N_AGE},)"
    assert flux_z.shape == (N_AGE, N_WAVE), f"ssp_flux_at_z shape {flux_z.shape}"


def test_age_weights_non_negative():
    """All age weights must be >= 0 (mass formed cannot be negative)."""
    sfr = _flat_sfr()
    aw, _ = compute_dsps_native_weights(
        sfr,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET,
        LGMET_SCATTER,
    )
    assert_non_negative(aw, name="aw", msg="Negative age weights found")


def test_ssp_flux_at_z_non_negative():
    """Metallicity-marginalized SSP flux must be >= 0."""
    sfr = _flat_sfr()
    _, flux_z = compute_dsps_native_weights(
        sfr,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        jnp.abs(jnp.array(SSP_FLUX)),  # ensure positive input
        T_OBS_GYR,
        LGMET,
        LGMET_SCATTER,
    )
    assert_non_negative(
        flux_z, name="flux_z", msg="Negative flux_at_z found (weights must sum to 1)"
    )


def test_total_mass_scales_with_sfr():
    """Doubling the SFR should double the total stellar mass formed."""
    sfr = _flat_sfr()
    aw1, _ = compute_dsps_native_weights(
        sfr,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET,
        LGMET_SCATTER,
    )
    aw2, _ = compute_dsps_native_weights(
        2.0 * sfr,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET,
        LGMET_SCATTER,
    )
    np.testing.assert_allclose(float(jnp.sum(aw2)), 2.0 * float(jnp.sum(aw1)), rtol=1e-6)


def test_zero_sfr_gives_zero_weights():
    """Zero SFR → zero total mass formed."""
    sfr = jnp.zeros(N_AGE, dtype=jnp.float64)
    aw, _ = compute_dsps_native_weights(
        sfr,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET,
        LGMET_SCATTER,
    )
    np.testing.assert_allclose(float(jnp.sum(aw)), 0.0, atol=1e-10)


def test_lgmet_scatter_zero_concentrates_on_nearest_bin():
    """Very small lgmet_scatter concentrates lgmet weights on nearest bin."""
    sfr = _flat_sfr()
    # Snap lgmet to nearest grid point to avoid interpolation ambiguity
    lgmet_snap = float(SSP_LGMET[N_MET // 2])
    _aw, flux_z = compute_dsps_native_weights(
        sfr,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        lgmet_snap,
        lgmet_scatter=1e-4,
    )
    # With scatter → 0, ssp_flux_at_z ≈ SSP_FLUX[N_MET//2]
    expected = SSP_FLUX[N_MET // 2]  # (n_age, n_wave)
    np.testing.assert_allclose(np.array(flux_z), expected, atol=1e-2)


def test_finite_output():
    """No NaN or Inf in output for well-formed inputs."""
    sfr = _flat_sfr()
    aw, flux_z = compute_dsps_native_weights(
        sfr,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET,
        LGMET_SCATTER,
    )
    chex.assert_tree_all_finite(aw)
    chex.assert_tree_all_finite(flux_z)


def test_jit_compatible():
    """compute_dsps_native_weights is JIT-traceable."""
    sfr = _flat_sfr()

    @jax.jit
    def _call(sfr):
        aw, fz = compute_dsps_native_weights(
            sfr,
            SSP_AGES_YR,
            SSP_LGMET,
            SSP_LG_AGE_GYR,
            SSP_FLUX,
            T_OBS_GYR,
            LGMET,
            LGMET_SCATTER,
        )
        return aw, fz

    aw, _fz = _call(sfr)
    chex.assert_shape(aw, (N_AGE,))


def test_grad_wrt_lgmet():
    """Gradient w.r.t. lgmet flows through to the metallicity-weighted SSP flux.

    Total CSP mass (``sum(age_weights_msun)``) is mathematically independent
    of lgmet by construction — it equals ``trapezoid(SFR, t) * joint.sum()``
    where ``joint.sum() = 1`` for any well-formed PDF. So that quantity has
    an exact-zero analytical gradient (autodiff returns 0; FD picks up float
    noise). To verify autodiff is wired through lgmet, check a quantity that
    genuinely depends on it: the metallicity-marginalized SSP flux.
    """
    sfr = _flat_sfr()

    def total_flux(lgmet):
        _, ssp_flux_at_z = compute_dsps_native_weights(
            sfr,
            SSP_AGES_YR,
            SSP_LGMET,
            SSP_LG_AGE_GYR,
            SSP_FLUX,
            T_OBS_GYR,
            lgmet,
            LGMET_SCATTER,
        )
        return jnp.sum(ssp_flux_at_z)

    grad_jax = float(jax.grad(total_flux)(jnp.array(LGMET)))
    grad_fd = fd_grad(lambda x: float(total_flux(x)), LGMET)
    np.testing.assert_allclose(
        grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
    )


# ── SEDModel-level: csp_integration='dsps_native' accepted ───────────


def _make_minimal_spec():
    """Minimal Parameters with fixed params (no SSP file needed)."""
    from tengri import Fixed, Parameters, Uniform

    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(1.5),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Uniform(-1.5, 0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )


def _make_minimal_ssp_data():
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    return SSPData(
        ssp_wave=jnp.array(np.linspace(3000.0, 10000.0, N_WAVE)),
        ssp_flux=jnp.array(SSP_FLUX),
        ssp_lg_age_gyr=jnp.array(SSP_LG_AGE_GYR),
        ssp_lgmet=jnp.array(SSP_LGMET),
    )


def test_model_accepts_dsps_native():
    """SEDModel.__init__ should not raise for csp_integration='dsps_native'."""
    from tengri.forward.sed_model import SEDModel

    ssp_data = _make_minimal_ssp_data()
    spec = _make_minimal_spec()

    model = SEDModel(spec, ssp_data, csp_integration="dsps_native")
    assert model._csp_integration == "dsps_native"
    assert model._csp_age_dt is None
    assert model._csp_matrix is None


def test_model_rejects_unknown_csp_mode():
    """SEDModel.__init__ raises ValueError for unknown csp_integration."""
    from tengri.forward.sed_model import SEDModel

    ssp_data = _make_minimal_ssp_data()
    spec = _make_minimal_spec()

    with pytest.raises(ValueError, match="csp_integration must be one of"):
        SEDModel(spec, ssp_data, csp_integration="bogus_mode")


# ── compute_dsps_met_table_weights: unit tests ────────────────────

LGMET_TABLE = np.linspace(-2.0, -1.0, N_AGE)  # per-age log10(Z), youngest first


def test_met_table_output_shapes():
    """age_weights_msun and ssp_flux_at_z have correct shapes."""
    sfr = _flat_sfr()
    aw, flux_z = compute_dsps_met_table_weights(
        sfr,
        jnp.array(LGMET_TABLE),
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET_SCATTER,
    )
    assert aw.shape == (N_AGE,), f"age_weights shape {aw.shape} != ({N_AGE},)"
    assert flux_z.shape == (N_AGE, N_WAVE), f"ssp_flux_at_z shape {flux_z.shape}"


def test_met_table_age_weights_non_negative():
    """All age weights must be >= 0."""
    sfr = _flat_sfr()
    aw, _ = compute_dsps_met_table_weights(
        sfr,
        jnp.array(LGMET_TABLE),
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET_SCATTER,
    )
    assert_non_negative(aw, name="aw", msg="Negative age weights found")


def test_met_table_finite_output():
    """No NaN or Inf in output for well-formed inputs."""
    sfr = _flat_sfr()
    aw, flux_z = compute_dsps_met_table_weights(
        sfr,
        jnp.array(LGMET_TABLE),
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET_SCATTER,
    )
    chex.assert_tree_all_finite(aw)
    chex.assert_tree_all_finite(flux_z)


def test_met_table_zero_sfr_gives_zero_weights():
    """Zero SFR → zero total mass formed."""
    sfr = jnp.zeros(N_AGE, dtype=jnp.float64)
    aw, _ = compute_dsps_met_table_weights(
        sfr,
        jnp.array(LGMET_TABLE),
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET_SCATTER,
    )
    np.testing.assert_allclose(float(jnp.sum(aw)), 0.0, atol=1e-10)


def test_met_table_mass_scales_with_sfr():
    """Doubling the SFR should double the total stellar mass formed."""
    sfr = _flat_sfr()
    lgmet = jnp.array(LGMET_TABLE)
    args = (lgmet, SSP_AGES_YR, SSP_LGMET, SSP_LG_AGE_GYR, SSP_FLUX, T_OBS_GYR, LGMET_SCATTER)
    aw1, _ = compute_dsps_met_table_weights(sfr, *args)
    aw2, _ = compute_dsps_met_table_weights(2.0 * sfr, *args)
    np.testing.assert_allclose(float(jnp.sum(aw2)), 2.0 * float(jnp.sum(aw1)), rtol=1e-6)


def test_met_table_constant_z_agrees_with_native():
    """Constant lgmet_table should give age_weights consistent with dsps_native.

    The two functions use different DSPS kernels so flux_at_z may differ,
    but the age weights (driven by SFH integration) should be close.
    """
    sfr = _flat_sfr()
    lgmet_scalar = float(SSP_LGMET[N_MET // 2])
    lgmet_const = jnp.full(N_AGE, lgmet_scalar, dtype=jnp.float64)

    aw_native, _ = compute_dsps_native_weights(
        sfr,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        lgmet_scalar,
        LGMET_SCATTER,
    )
    aw_table, _ = compute_dsps_met_table_weights(
        sfr,
        lgmet_const,
        SSP_AGES_YR,
        SSP_LGMET,
        SSP_LG_AGE_GYR,
        SSP_FLUX,
        T_OBS_GYR,
        LGMET_SCATTER,
    )
    # Both integrate the same SFH, so total mass should agree closely.
    np.testing.assert_allclose(float(jnp.sum(aw_table)), float(jnp.sum(aw_native)), rtol=1e-4)


def test_met_table_jit_compatible():
    """compute_dsps_met_table_weights is JIT-traceable."""
    sfr = _flat_sfr()
    lgmet = jnp.array(LGMET_TABLE)

    @jax.jit
    def _call(sfr, lgmet):
        return compute_dsps_met_table_weights(
            sfr, lgmet, SSP_AGES_YR, SSP_LGMET, SSP_LG_AGE_GYR, SSP_FLUX, T_OBS_GYR, LGMET_SCATTER
        )

    aw, fz = _call(sfr, lgmet)
    chex.assert_shape(aw, (N_AGE,))
    chex.assert_shape(fz, (N_AGE, N_WAVE))


def test_met_table_grad_wrt_lgmet():
    """Gradient of total CSP mass w.r.t. lgmet_table is finite."""
    sfr = _flat_sfr()

    def total_mass(lgmet):
        aw, _ = compute_dsps_met_table_weights(
            sfr, lgmet, SSP_AGES_YR, SSP_LGMET, SSP_LG_AGE_GYR, SSP_FLUX, T_OBS_GYR, LGMET_SCATTER
        )
        return jnp.sum(aw)

    lgmet = jnp.array(LGMET_TABLE)
    grad = assert_grad_matches_fd(total_mass, lgmet)
    assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient: {grad}"


# ── SEDModel-level: csp_integration='dsps_met_table' accepted ────────


def test_model_accepts_dsps_met_table():
    """SEDModel.__init__ should not raise for csp_integration='dsps_met_table'."""
    from tengri.forward.sed_model import SEDModel

    ssp_data = _make_minimal_ssp_data()
    spec = _make_minimal_spec()

    model = SEDModel(spec, ssp_data, csp_integration="dsps_met_table")
    assert model._csp_integration == "dsps_met_table"
    assert model._csp_age_dt is None
    assert model._csp_matrix is None
