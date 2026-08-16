# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the 14 non-GRAHSP block adapters wrapping existing tengri AGN
pieces (multicolor / kubota_done / adaf disc; nenkova / skirtor / silva04 /
cat3d_wind torus; BLR / NLR lines; polar dust attenuation; 5 qsogen blocks).

Coverage:
- Registry presence (every adapter shows up in :data:`AGN_BLOCKS`).
- Smoke (output finite + correct shape).
- JIT compatibility under :func:`jax.jit`.
- Cross-model recipes (e.g. multicolor disc + skirtor torus).
- Validation rule extensions (BLR/NLR + non-5100Å disc; polar_dust + EBV=0).
"""

from __future__ import annotations

import warnings

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.bounds

from tengri.components.agn.blocks import (
    AGN_BLOCKS,
    RecipeWarning,
    composable_agn_l_nu,
    validate_block_recipe,
)
from tests._data_skip import requires_nenkova

# ──────────────────────────────────────────────────────────────────────
# Registry presence
# ──────────────────────────────────────────────────────────────────────


def test_disc_alternates_registered():
    for name in ("multicolor", "kubota_done", "adaf", "qsogen"):
        assert name in AGN_BLOCKS["disc"], f"disc/{name!r} not registered"


def test_torus_alternates_registered():
    for name in ("nenkova", "skirtor", "silva04", "cat3d_wind", "qsogen"):
        assert name in AGN_BLOCKS["torus"], f"torus/{name!r} not registered"


def test_nlr_blr_alternates_registered():
    for name in ("analytic", "synthesizer", "synthesizer_spectra", "grahsp"):
        assert name in AGN_BLOCKS["nlr"], f"nlr/{name!r} not registered"
    for name in ("analytic", "synthesizer", "synthesizer_spectra", "grahsp", "qsogen"):
        assert name in AGN_BLOCKS["blr"], f"blr/{name!r} not registered"


def test_feii_alternates_registered():
    for name in ("qsogen_balmer",):
        assert name in AGN_BLOCKS["feii"], f"feii/{name!r} not registered"


def test_atten_alternates_registered():
    for name in ("polar_dust", "qsogen_smc"):
        assert name in AGN_BLOCKS["attenuation"], f"attenuation/{name!r} missing"


# ──────────────────────────────────────────────────────────────────────
# Smoke + JIT
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("disc", ["multicolor", "kubota_done"])
def test_disc_block_smoke(disc):
    """Each new disc block produces a finite L_nu when run via the runner.

    ADAF is skipped because its inner-flow physics produces zero/near-zero
    flux in the optical/NIR window we're sampling, which is exactly the
    case rule 4 of validate_block_recipe is designed to flag.
    """
    wave_aa = jnp.logspace(3, 5, 200)
    out = composable_agn_l_nu(
        wave_aa,
        agn_disc_block=disc,
        agn_log_lbol=45.0,
        agn_log_mbh=8.0,
        agn_log_ledd=-1.0,
    )
    chex.assert_equal_shape([out, wave_aa])
    chex.assert_tree_all_finite(out)


@pytest.mark.parametrize("torus", ["nenkova", "two_temperature", "simple"])
def test_torus_block_smoke(torus):
    if torus == "nenkova":
        from tests._data_skip import _nenkova_path as _np

        if _np is None or not _np.is_file():
            pytest.skip("Nenkova+2008 data not available (SPS_HOME unset)")
    """Each new torus block produces a finite L_nu (skirtor/silva04/cat3d need
    grids, smoke them via separate fixture-aware tests if grids are present)."""
    wave_aa = jnp.logspace(3, 6, 200)
    out = composable_agn_l_nu(
        wave_aa,
        agn_torus_block=torus,
        agn_log_lbol=44.0,
        agn_torus_frac=0.5,
    )
    chex.assert_tree_all_finite(out)


def test_polar_dust_factor_within_unit_interval():
    """Attenuation factor must lie in (0, 1]; this confirms the polar_dust
    block returns a true multiplicative factor, not a luminosity."""
    from tengri.components.agn.blocks import resolve_agn_block

    fn = resolve_agn_block("attenuation", "polar_dust")
    wave_aa = jnp.logspace(3, 5, 100)
    factor = fn(
        wave_aa,
        agn_polar_ebv=0.3,
        agn_cos_inc=1.0,  # face-on -> Type 1, full extinction
        agn_polar_oa=45.0,
    )
    assert jnp.all(factor > 0)
    assert jnp.all(factor <= 1.0 + 1e-10)


def test_blr_lines_smoke():
    """BLR block produces non-zero, finite output when paired with a UV/optical disc."""
    wave_aa = jnp.logspace(3, 5, 400)
    out = composable_agn_l_nu(
        wave_aa,
        agn_disc_block="powerlaw",
        agn_nlr_block="none",
        agn_blr_block="analytic",
        agn_log_lbol=44.0,
        agn_blr_cf=0.1,
        agn_blr_fwhm_kms=3000.0,
    )
    chex.assert_tree_all_finite(out)
    assert float(out.sum()) > 0


def test_nlr_lines_smoke():
    wave_aa = jnp.logspace(3, 5, 400)
    out = composable_agn_l_nu(
        wave_aa,
        agn_disc_block="powerlaw",
        agn_nlr_block="analytic",
        agn_blr_block="none",
        agn_log_lbol=44.0,
        agn_nlr_cf=0.05,
        agn_nlr_fwhm_kms=300.0,
    )
    chex.assert_tree_all_finite(out)
    assert float(out.sum()) > 0


@requires_nenkova
def test_jit_through_alternate_recipe():
    """JIT-compile a cross-model recipe (multicolor disc + nenkova torus)."""
    wave_aa = jnp.logspace(3, 6, 200)

    @jax.jit
    def fwd(log_lbol, log_mbh):
        return composable_agn_l_nu(
            wave_aa,
            agn_disc_block="multicolor",
            agn_torus_block="nenkova",
            agn_log_lbol=log_lbol,
            agn_log_mbh=log_mbh,
            agn_log_ledd=-0.5,
            agn_tau=30.0,
            agn_torus_frac=0.5,
        )

    out = fwd(jnp.array(44.5), jnp.array(8.0))
    chex.assert_equal_shape([out, wave_aa])
    chex.assert_tree_all_finite(out)


# ──────────────────────────────────────────────────────────────────────
# Validation rule extensions
# ──────────────────────────────────────────────────────────────────────


def test_validate_blr_with_adaf_disc_warns():
    """BLR is in _DOWNSTREAM_NEEDS_L5100; pairing with ADAF disc must warn."""
    with pytest.warns(RecipeWarning, match="lambda\\*L_lambda\\(5100A\\)"):
        validate_block_recipe(
            agn_disc_block="adaf",
            agn_nlr_block="none",
            agn_blr_block="analytic",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
        )


def test_validate_polar_dust_zero_ebv_warns():
    """Polar dust block selected without an E(B-V) is a no-op."""
    with pytest.warns(RecipeWarning, match="agn_polar_ebv=0"):
        validate_block_recipe(
            agn_disc_block="powerlaw",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="polar_dust",
            params={"agn_polar_ebv": 0.0},
        )


def test_validate_polar_dust_nonzero_ebv_clean():
    """Same combo with E(B-V) > 0 emits no rule-7 warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RecipeWarning)
        issues = validate_block_recipe(
            agn_disc_block="powerlaw",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="polar_dust",
            params={"agn_polar_ebv": 0.3},
        )
    assert all("agn_polar_ebv=0" not in m for m in issues)
    assert not any("agn_polar_ebv=0" in str(x.message) for x in w)


def test_validate_adaf_disc_no_longer_deprecation_warns():
    """After the faithful Mahadevan 1997 rewrite (#898) the adaf disc is a valid
    production choice — selecting it (with no downstream lines blocks) must NOT
    emit any deprecation RecipeWarning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RecipeWarning)
        validate_block_recipe(
            agn_disc_block="adaf",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
        )


def test_adaf_disc_block_status_is_production():
    """After the #898 rewrite the adaf disc block is registered as production."""
    from tengri.components.agn.blocks._protocol import AGN_BLOCK_META

    assert AGN_BLOCK_META[("disc", "adaf")]["status"] == "production"


def test_validate_boroson_green_feii_needs_5100A_disc():
    """boroson_green FeII normalizes to l5100_disc, so pairing it with a disc
    that lacks a 5100Å continuum (e.g. adaf) must warn (rule 4). It was missing
    from ``_DOWNSTREAM_NEEDS_L5100['feii']`` so the no-op went unflagged."""
    with pytest.warns(RecipeWarning, match=r"lambda\*L_lambda\(5100A\)"):
        validate_block_recipe(
            agn_disc_block="adaf",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="boroson_green",
            agn_torus_block="none",
            agn_attenuation_block="none",
        )


# ──────────────────────────────────────────────────────────────────────
# Selector typo via Parameters round-trip
# ──────────────────────────────────────────────────────────────────────


def test_parameters_typo_in_block_selector_raises():
    """Typo'd selector via tengri.Parameters must raise ValueError immediately,
    not silently fall through to a default."""
    from tengri import Parameters

    with pytest.raises(ValueError, match="Unknown torus block"):
        Parameters(
            agn_model="composable",
            agn_disc_block="powerlaw",
            agn_torus_block="bogus_torus",  # deliberate typo
        )


def test_parameters_clean_composable_recipe_constructs():
    """Sanity: a valid composable recipe constructs without errors."""
    from tengri import Parameters
    from tengri.parameters.priors import Fixed, Uniform

    p = Parameters(
        agn_model="composable",
        agn_disc_block="multicolor",
        agn_torus_block="skirtor",
        agn_attenuation_block="none",
        agn_log_lbol=Uniform(9.42, 13.42),
        agn_log_mbh=Fixed(8.0),
        agn_log_ledd=Fixed(-1.0),
        agn_tau_skirtor=Fixed(7.0),
    )
    assert p.agn_model == "composable"
    assert p.agn_disc_block == "multicolor"
    assert p.agn_torus_block == "skirtor"
