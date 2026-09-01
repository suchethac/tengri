# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the composable AGN block subsystem.

Covers:
- registry sanity (categories + builtin none/grahsp/alternates)
- runner equivalence: all-grahsp recipe ≡ compute_grahsp_sed
- mix-and-match: GRAHSP BBB + simple two-T torus + SMC Prevot atten
- recipe validation: warnings on suspicious combos, ValueError on typos
- JIT compatibility
"""

from __future__ import annotations

import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds

from tengri.components.agn import AGN_MODELS, resolve_agn_model
from tengri.components.agn.blocks import (
    AGN_BLOCKS,
    BLOCK_CATEGORIES,
    RecipeWarning,
    composable_agn_l_nu,
    register_agn_block,
    resolve_agn_block,
    validate_block_recipe,
)
from tengri.components.agn.blocks.alternates import smc_prevot_block
from tengri.components.agn.reddening import redden_disc
from tests._data_skip import requires_grahsp
from tests._grad_parity import assert_grad_matches_fd

# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


def test_all_categories_have_none_block():
    for cat in BLOCK_CATEGORIES:
        assert "none" in AGN_BLOCKS[cat], f"{cat} missing 'none' block"


def test_grahsp_blocks_registered_for_all_categories():
    """GRAHSP must provide an impl for every pipeline stage."""
    expected = {
        "disc": "grahsp_sbpl",
        "nlr": "grahsp",
        "blr": "grahsp",
        "feii": "grahsp",
        "torus": "grahsp",
        "attenuation": "grahsp_biatten",
    }
    for cat, name in expected.items():
        assert name in AGN_BLOCKS[cat], f"{cat}/{name} not registered"


def test_at_least_one_alternate_per_category():
    """For mix-and-match to be meaningful, need >= 1 non-GRAHSP, non-none impl."""
    for cat in ("disc", "torus", "attenuation"):
        non_default = [n for n in AGN_BLOCKS[cat] if n != "none" and not n.startswith("grahsp")]
        assert non_default, f"{cat} has no non-GRAHSP alternate"


def test_composable_in_agn_models():
    assert "composable" in AGN_MODELS


def test_resolve_unknown_block_raises():
    with pytest.raises(ValueError, match="Unknown disc block"):
        resolve_agn_block("disc", "bogus_does_not_exist")


def test_register_duplicate_raises():
    """register_agn_block must reject duplicate (category, name) pairs."""
    with pytest.raises(ValueError, match="already registered"):

        @register_agn_block("disc", "grahsp_sbpl")
        def _redefine(*args, **kwargs):
            return None


# ──────────────────────────────────────────────────────────────────────
# Recipe validation: warnings for suspicious combos
# ──────────────────────────────────────────────────────────────────────


def test_validate_clean_recipe_emits_no_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RecipeWarning)
        issues = validate_block_recipe(
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
        )
    assert issues == []
    assert not any(issubclass(x.category, RecipeWarning) for x in w)


def test_validate_all_none_warns():
    with pytest.warns(RecipeWarning, match="every block selector is 'none'"):
        validate_block_recipe(
            agn_disc_block="none",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
        )


def test_validate_active_downstream_no_disc_warns():
    """Forgetting the disc but enabling nlr/blr/feii/torus must warn loudly."""
    with pytest.warns(RecipeWarning, match="agn_disc_block='none'"):
        validate_block_recipe(
            agn_disc_block="none",
            agn_nlr_block="grahsp",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="grahsp",
            agn_attenuation_block="none",
        )


def test_validate_unknown_block_raises_not_warns():
    """Typo in selector should be a hard error."""
    with pytest.raises(ValueError, match="Unknown disc block"):
        validate_block_recipe(
            agn_disc_block="grahps_sbpl",  # deliberate typo
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
        )


# ──────────────────────────────────────────────────────────────────────
# Regression: smc_prevot_block delegates to redden_disc (missing-R_V bug)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.regression_bug
def test_smc_prevot_block_matches_redden_disc():
    """smc_prevot_block must produce the same attenuation factor as redden_disc.

    Regression: smc_prevot_block previously computed
    A_lambda = k_norm(λ)·E(B−V) without the R_V=2.72 factor of the Prevot+1984
    prescription (A_lambda = k·R_V·E(B−V) with k = A_λ/A_V), causing 2.72×
    under-attenuation in magnitudes vs the canonical redden_disc function.
    Now it delegates to redden_disc, so the two disc-reddening paths are
    identical by construction.

    References
    ----------
    .. [1] M. L. Prevot et al., "The typical interstellar extinction in the
       Small Magellanic Cloud," A&A, 132, 389 (1984).
    .. [2] AGNfitter BBBred_Prevot in MODEL_AGNfitter.py.
    """
    wave_aa = jnp.logspace(3.0, 5.0, 200)  # 1000 Å – 100 µm
    agn_attenuation_ebv = 0.3

    # Block interface: returns L_lambda [erg/s/Å] multiplicative factor
    factor_block = smc_prevot_block(wave_aa, agn_attenuation_ebv=agn_attenuation_ebv)

    # Reference: redden_disc applied to a unit disc SED
    factor_redden = redden_disc(wave_aa, jnp.ones_like(wave_aa), agn_attenuation_ebv)

    # They must be bitwise identical (both delegate to the same prevot_smc + R_V)
    np.testing.assert_array_equal(np.asarray(factor_block), np.asarray(factor_redden))

    # Verify the implied absorption at 1500 Å is reasonable
    from tengri.components.dust.attenuation import prevot_smc

    wave_test = jnp.array([1500.0])
    factor_at_1500 = float(np.asarray(redden_disc(wave_test, jnp.ones(1), agn_attenuation_ebv))[0])
    # A(1500 Å) = -2.5 * log10(factor)
    a_1500 = -2.5 * np.log10(factor_at_1500)
    # Expected: k_norm(1500) * R_V * E(B-V)
    k_norm_1500 = float(np.asarray(prevot_smc(wave_test))[0])
    expected_a_1500 = k_norm_1500 * 2.72 * agn_attenuation_ebv
    # Within 2% due to numerical rounding
    np.testing.assert_allclose(a_1500, expected_a_1500, rtol=0.02)


# ──────────────────────────────────────────────────────────────────────
# Runner: all-grahsp recipe equals compute_grahsp_sed
# ──────────────────────────────────────────────────────────────────────


def _grahsp_params():
    """Single-source GRAHSP parameter set used by both pipelines.

    ``agn_grahsp_l5100`` is **explicit** here. Without it, the two paths
    use different conventions:

    - ``compute_grahsp_sed`` (monolithic): rescales l5100 so the total
      AGN-side bolometric integral matches ``10**agn_log_lbol * L_sun``.
    - ``composable_agn_l_nu`` (block runner): each block self-normalizes
      from its own params, with no post-hoc bolometric coupling.

    Setting ``agn_grahsp_l5100`` directly bypasses both the monolithic
    rescale and the composable disc auto-norm, so the two paths agree
    bit-for-bit.
    """
    return dict(
        agn_log_lbol=44.5,
        agn_grahsp_l5100=1.0e44,
        agn_grahsp_uvslope=0.0,
        agn_grahsp_plslope=-1.7,
        agn_grahsp_plbendloc_nm=100.0,
        agn_grahsp_plbendwidth=1.0,
        agn_grahsp_cutoff_nm=10000.0,
        agn_grahsp_a_lines=1.0,
        agn_grahsp_a_feii=5.0,
        agn_grahsp_linewidth_kms=5000.0,
        agn_grahsp_fcov=0.4,
        agn_grahsp_si=0.5,
        agn_grahsp_cool_lam_um=17.0,
        agn_grahsp_cool_width=0.45,
        agn_grahsp_hot_lam_um=2.0,
        agn_grahsp_hot_width=0.5,
        agn_grahsp_hot_fcov=1.0,
        agn_grahsp_ebv=0.05,
        agn_grahsp_ebv_agn=0.05,
        agn_type=1,
    )


@requires_grahsp
def test_all_grahsp_recipe_matches_compute_grahsp_sed():
    """Every selector = 'grahsp*' should reproduce compute_grahsp_sed exactly."""
    from tengri.components.agn.grahsp import compute_grahsp_sed

    wave_aa = jnp.logspace(2, 6, 400)
    p = _grahsp_params()

    out_composable = composable_agn_l_nu(
        wave_aa,
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="grahsp",
        agn_blr_block="grahsp",
        agn_feii_block="grahsp",
        agn_torus_block="grahsp",
        agn_attenuation_block="grahsp_biatten",
        **p,
    )
    out_monolithic = compute_grahsp_sed(wave_aa, **p)

    # Both paths exercise the same physics. The runner derives l5100_disc
    # via ``jnp.interp(5100Å, wave, L_λ_disc) × 5100`` from the disc grid;
    # the monolithic path uses the analytic l5100 directly. With explicit
    # ``agn_grahsp_l5100`` they agree to grid-interpolation precision.
    np.testing.assert_allclose(
        np.asarray(out_composable),
        np.asarray(out_monolithic),
        rtol=1e-3,
        atol=0.0,
    )


def test_mix_grahsp_disc_with_simple_torus():
    """GRAHSP BBB + simple graybody torus + SMC atten — runs and is finite."""
    wave_aa = jnp.logspace(2, 6, 400)
    out = composable_agn_l_nu(
        wave_aa,
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="two_temperature",
        agn_attenuation_block="smc_prevot",
        agn_log_lbol=45.0,
        agn_grahsp_l5100=1.0e44,
        agn_T_hot=1200.0,
        agn_T_warm=300.0,
        agn_frac_hot=0.3,
        agn_torus_frac=0.5,
        agn_attenuation_ebv=0.2,
    )
    chex.assert_equal_shape([out, wave_aa])
    chex.assert_tree_all_finite(out)
    # Some flux must come through after attenuation.
    assert float(out.sum()) > 0


def test_disc_only_recipe_is_pure_continuum():
    """No nlr/blr/feii/torus → output is just the disc continuum (in L_nu)."""
    wave_aa = jnp.logspace(2, 5, 200)
    out = composable_agn_l_nu(
        wave_aa,
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="none",
        agn_attenuation_block="none",
        agn_grahsp_l5100=1.0e44,
    )
    # Should be smooth (no line spikes) and positive across the disc's
    # physical range. The disc reads zero below the alpha_ox corona's blue
    # edge (124 A), a band GRAHSP does not model (#1168).
    chex.assert_tree_all_finite(out)
    physical = wave_aa >= 124.0
    assert jnp.all(out[physical] > 0)
    assert jnp.all(out[~physical] == 0)


# ──────────────────────────────────────────────────────────────────────
# JIT
# ──────────────────────────────────────────────────────────────────────


@requires_grahsp
def test_runner_jit_compatible():
    """Selectors are static; param values are dynamic."""
    wave_aa = jnp.logspace(2, 6, 200)

    @jax.jit
    def fwd(l5100, ebv):
        return composable_agn_l_nu(
            wave_aa,
            agn_log_lbol=44.5,
            agn_disc_block="grahsp_sbpl",
            agn_nlr_block="grahsp",
            agn_blr_block="grahsp",
            agn_feii_block="grahsp",
            agn_torus_block="grahsp",
            agn_attenuation_block="grahsp_biatten",
            agn_grahsp_l5100=l5100,
            agn_grahsp_ebv=ebv,
        )

    out = fwd(jnp.array(1.0e44), jnp.array(0.1))
    chex.assert_equal_shape([out, wave_aa])
    chex.assert_tree_all_finite(out)


def test_composable_polar_dust_jit_and_grad():
    """Regression (#846): composable AGN with polar_dust must be jittable and
    differentiable in ``agn_polar_ebv``.

    Before the fix, :func:`composable_agn_l_nu` called
    :func:`validate_block_recipe` from inside the jitted forward pass; that
    validator's Rule 7 did ``float(agn_polar_ebv)``, which raised
    ``ConcretizationTypeError`` on the tracer — so composing polar dust for an
    actual fit crashed. Recipe validation now runs only at construction time.
    Uses the analytic ``powerlaw`` disc + ``polar_dust`` atten (no HDF5) so the
    guard runs in CI without gridded template data.
    """
    wave_aa = jnp.logspace(2, 6, 200)

    @jax.jit
    def fwd(ebv):
        return composable_agn_l_nu(
            wave_aa,
            agn_log_lbol=44.5,
            agn_disc_block="multicolor",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="polar_dust",
            agn_cos_inc=0.95,  # face-on → Type-1 screen active
            agn_polar_ebv=ebv,
        )

    out = fwd(jnp.array(0.2))  # raised ConcretizationTypeError pre-fix
    chex.assert_equal_shape([out, wave_aa])
    chex.assert_tree_all_finite(out)

    grad = assert_grad_matches_fd(lambda e: jnp.sum(fwd(e)), jnp.array(0.2))
    chex.assert_tree_all_finite(grad)


def test_validate_polar_dust_guard_ignores_nonconcrete_ebv():
    """The Rule 7 concreteness guard must not crash (or warn) when
    ``agn_polar_ebv`` cannot be read as a concrete float — mimicking a JAX
    tracer, whose ``__float__`` raises a TypeError subclass."""

    class _Tracerish:
        def __float__(self):
            raise TypeError("Abstract tracer value encountered")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RecipeWarning)  # would raise if it warned
        issues = validate_block_recipe(
            agn_disc_block="multicolor",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="polar_dust",
            params={"agn_polar_ebv": _Tracerish()},
        )
    assert issues == []


def test_resolve_via_agn_models_registry():
    """Going through resolve_agn_model('composable') must match direct call."""
    fn_via_registry = resolve_agn_model("composable")
    wave_aa = jnp.logspace(2, 6, 100)
    p = dict(
        agn_disc_block="grahsp_sbpl",
        agn_nlr_block="none",
        agn_blr_block="none",
        agn_feii_block="none",
        agn_torus_block="grahsp",
        agn_attenuation_block="none",
        agn_grahsp_l5100=1.0e44,
    )
    out_registry = fn_via_registry(wave_aa, agn_log_lbol=44.5, **p)
    out_direct = composable_agn_l_nu(wave_aa, agn_log_lbol=44.5, **p)
    np.testing.assert_allclose(np.asarray(out_registry), np.asarray(out_direct), rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Contract: AGN E(B-V) parameters are settable via SEDModel.build
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.contract
def test_agn_ebv_disc_settable_via_sedbuild(synthetic_ssp_wide, synthetic_tophat_obs):
    """agn_ebv_disc must be settable via SEDModel.build and change predict_state.

    Regression: agn_ebv_disc (the AGNfitter ``EBVbbb`` analog, consumed by
    the composable runner since #916) had no ParamDeclaration and no partition
    entry, so the recommended SEDModel.build path could not redden the disc at
    all. It now lowers like any other shared agn-group parameter.
    """
    from tengri import DEFAULT, Fixed, SEDModel

    obs = synthetic_tophat_obs
    ssp = synthetic_ssp_wide

    # Build two models: one with agn_ebv_disc=0.0, one with 0.5
    # Fix all parameters to make predict_state work
    model_no_redd = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "none"},
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "agn_log_lbol": Fixed(11.0),
            "agn_ebv_disc": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
    )

    model_reddened = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "none"},
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "agn_log_lbol": Fixed(11.0),
            "agn_ebv_disc": Fixed(0.5),
            "all_params": Fixed(DEFAULT),
        },
    )

    # Predict SEDs and compare AGN components
    # (All parameters are fixed, so empty dict suffices for predict)
    state_no_redd = model_no_redd.predict_state({})
    state_reddened = model_reddened.predict_state({})

    sed_agn_no_redd = state_no_redd.derived["sed_agn"]
    sed_agn_reddened = state_reddened.derived["sed_agn"]

    # With reddening, the SED should be dimmer, especially at short wavelengths
    # Maximum relative difference must be > 1e-6 to confirm the parameter had an effect
    rel_diff = np.abs(sed_agn_reddened - sed_agn_no_redd) / np.maximum(
        np.abs(sed_agn_no_redd), 1e-100
    )
    max_rel_diff = float(np.nanmax(rel_diff))
    assert max_rel_diff > 1e-6, (
        f"agn_ebv_disc parameter had no effect on SED: max relative change = {max_rel_diff:.2e}"
    )


@pytest.mark.contract
def test_agn_attenuation_ebv_settable_via_sedbuild(synthetic_ssp_wide, synthetic_tophat_obs):
    """agn_attenuation_ebv (atten sub-block) must be settable and change predict_state.

    Regression: agn_attenuation_ebv had no ParamDeclaration and no partition
    entry, so ``atten={'type': 'smc_prevot'}`` built a block pinned at
    E(B−V)=0 — a silent no-op. It now lowers via the agn.atten sub-block.
    Updated to use law='prevot_smc' syntax (new form).
    """
    from tengri import DEFAULT, Fixed, SEDModel

    obs = synthetic_tophat_obs
    ssp = synthetic_ssp_wide

    # Build two models: one with agn_attenuation_ebv=0.0, one with 0.5
    # Use smc_prevot attenuation block via law='prevot_smc'; fix all parameters
    model_no_atten = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "none"},
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "atten": {
                "law": "prevot_smc",
                "attenuation_ebv": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            "agn_log_lbol": Fixed(11.0),
            "all_params": Fixed(DEFAULT),
        },
    )

    model_attenuated = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "none"},
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "atten": {
                "law": "prevot_smc",
                "attenuation_ebv": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            "agn_log_lbol": Fixed(11.0),
            "all_params": Fixed(DEFAULT),
        },
    )

    # Predict SEDs and compare AGN components
    # (All parameters are fixed, so empty dict suffices for predict)
    state_no_atten = model_no_atten.predict_state({})
    state_attenuated = model_attenuated.predict_state({})

    sed_agn_no_atten = state_no_atten.derived["sed_agn"]
    sed_agn_attenuated = state_attenuated.derived["sed_agn"]

    # With attenuation, the SED should be dimmer
    rel_diff = np.abs(sed_agn_attenuated - sed_agn_no_atten) / np.maximum(
        np.abs(sed_agn_no_atten), 1e-100
    )
    max_rel_diff = float(np.nanmax(rel_diff))
    assert max_rel_diff > 1e-6, (
        f"agn_attenuation_ebv parameter had no effect on SED: "
        f"max relative change = {max_rel_diff:.2e}"
    )
