# SPDX-License-Identifier: BSD-3-Clause
"""Phase 3d — ``approx=`` kwarg contract on :class:`SEDModel`.

Tests the **user-observable** contract:

* ``approx=None`` (default) — exact wave-grid path; the LUT projection
  ``observation.predict_via_precomp`` is unavailable because the LUT was
  never published.
* ``approx=WavePrecomp()`` — opt-in LUT path with default ztable sampling.
  ``predict_photometry`` returns the same array as
  ``observation.predict_via_precomp`` (bit-exact, since the method now
  routes through it).
* ``approx=WavePrecomp(n_z=…, z_min=…, z_max=…)`` — ztable sampling
  knobs change the interpolation accuracy at off-grid redshifts.
* Anything else (dict, bool, any string) — ``TypeError`` naming the legal
  forms.

These tests intentionally avoid reaching into ``model._approx`` /
``ssp_phot_ztable.z_grid`` private fields. The contract being pinned is
behavioral, so renaming or restructuring internal state should not break
the suite.
"""

from __future__ import annotations

import warnings

import chex
import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel, WavePrecomp
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def ssp(synthetic_ssp_wide):
    # #613: synthetic SSP + synthetic filters so the approx-kwarg contract runs
    # on CI instead of skipping when data/ssp_*.h5 is absent.
    return synthetic_ssp_wide


@pytest.fixture(scope="module")
def obs(synthetic_tophat_obs):
    return synthetic_tophat_obs


@pytest.fixture(scope="module")
def fixed_z_spec():
    """Minimal Fixed-everywhere spec — validator tests don't need free params."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        igm={"type": "none"},
    )


@pytest.fixture(scope="module")
def free_z_spec():
    """Free-redshift spec — only test that needs ztable interpolation."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Uniform(0.5, 1.5),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        igm={"type": "none"},
    )


def _silent_build(spec, ssp, obs, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, **kwargs)


# ── Strict rejection of pre-3d forms ─────────────────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [
        {"wave_precomp": True},
        {"wave_precomp": True, "ztable": True},
        {},
        True,
        False,
        "wave_precomp",
        "precomp",
        "exact",
        0,
        1,
    ],
)
def test_approx_rejects_legacy_and_unknown_forms(fixed_z_spec, ssp, obs, bad):
    """Dict/bool/string forms must raise ``TypeError`` with a migration message."""
    with pytest.raises(TypeError, match="approx="):
        SEDModel(fixed_z_spec, ssp, observation=obs, approx=bad)


# ── Default semantics: no opt-in → exact path, LUT unavailable ──────────────


def test_default_disables_lut_projection(fixed_z_spec, ssp, obs):
    """``approx=None`` (default) builds a model whose ``predict_photometry``
    returns finite flux, but whose ``observation.predict_via_precomp`` fails
    because the LUT was never published.
    """
    model = _silent_build(fixed_z_spec, ssp, obs, approx=None)
    params = {}
    out = model.predict_photometry(params)
    assert out.shape[0] == 5
    chex.assert_tree_all_finite(out)
    state = model.predict_state(params)
    full = {**model.spec.get_fixed_values(), **params}
    with pytest.raises((AttributeError, KeyError, ValueError, TypeError)):
        model.observation.predict_via_precomp(state, full)


# ── Opt-in semantics: WavePrecomp() routes predict_photometry through LUT ────


def test_wave_precomp_routes_predict_photometry_through_lut(fixed_z_spec, ssp, obs):
    """The build-time opt-in is the speed knob: when built with
    ``WavePrecomp()``, ``predict_photometry`` returns exactly the array
    that ``observation.predict_via_precomp`` produces — proves the route-
    through actually fires.
    """
    model = _silent_build(fixed_z_spec, ssp, obs, approx=WavePrecomp())
    params = {}

    via_method = model.predict_photometry(params)

    state = model.predict_state(params)
    full = {**model.spec.get_fixed_values(), **params}
    via_obs = model.observation.predict_via_precomp(state, full)["phot_fnu"]

    # predict_photometry now goes through predict_observables_jit (which JITs
    # the projection); JIT op-reordering can introduce sub-machine-epsilon
    # differences from the non-JIT predict_via_precomp call. Floating-point
    # close is the correct guarantee.
    assert jnp.allclose(via_method, via_obs, rtol=1e-12, atol=0), (
        "predict_photometry must agree with observation.predict_via_precomp().phot_fnu "
        "when built with approx=WavePrecomp()."
    )


# ── ztable sampling knobs change physics for free-z models ──────────────────


def test_wave_precomp_n_z_changes_free_z_interpolation(free_z_spec, ssp, obs):
    """A coarse ztable (``n_z=10``) gives a different interpolated flux at
    an off-grid redshift than a fine one (``n_z=200``). If ``n_z`` is silently
    ignored both would be bit-identical — pinning that ``n_z`` actually
    flows through to the interpolation grid.
    """
    coarse = _silent_build(free_z_spec, ssp, obs, approx=WavePrecomp(n_z=10, z_min=0.5, z_max=1.5))
    fine = _silent_build(free_z_spec, ssp, obs, approx=WavePrecomp(n_z=200, z_min=0.5, z_max=1.5))

    # Off-grid redshift — coarse ztable resolves it through wider interpolation.
    params = {"redshift": jnp.asarray(0.73)}
    phot_coarse = coarse.predict_photometry(params)
    phot_fine = fine.predict_photometry(params)

    chex.assert_tree_all_finite(phot_coarse)
    chex.assert_tree_all_finite(phot_fine)
    # Close, but not bit-identical — proves the grids differ.
    assert jnp.allclose(phot_coarse, phot_fine, rtol=0.05)
    assert not jnp.array_equal(phot_coarse, phot_fine), (
        "n_z appears to be silently ignored — coarse and fine ztables give "
        "identical photometry, which means the user override didn't flow "
        "through to the stellar component's ztable construction."
    )


def test_wave_precomp_z_bounds_clip_the_grid(free_z_spec, ssp, obs):
    """Setting ``z_min=0.8, z_max=1.2`` should give a ztable that interpolates
    well inside that range and degrades sharply outside it. We don't reach
    into the z_grid array — we just confirm the model evaluates finitely
    inside and outside, and that bounds change the output.
    """
    narrow = _silent_build(free_z_spec, ssp, obs, approx=WavePrecomp(n_z=20, z_min=0.8, z_max=1.2))
    wide = _silent_build(free_z_spec, ssp, obs, approx=WavePrecomp(n_z=20, z_min=0.5, z_max=1.5))

    # Inside both grids — outputs are close.
    params_inside = {"redshift": jnp.asarray(1.0)}
    phot_narrow_inside = narrow.predict_photometry(params_inside)
    phot_wide_inside = wide.predict_photometry(params_inside)
    assert jnp.allclose(phot_narrow_inside, phot_wide_inside, rtol=0.02)

    # Outside the narrow grid but inside the wide one — outputs differ
    # measurably, proving z_min/z_max actually constrain the LUT.
    params_outside = {"redshift": jnp.asarray(0.6)}
    phot_narrow_outside = narrow.predict_photometry(params_outside)
    phot_wide_outside = wide.predict_photometry(params_outside)
    assert not jnp.array_equal(phot_narrow_outside, phot_wide_outside), (
        "z_min/z_max do not change the LUT — narrow and wide grids give "
        "identical photometry at z=0.6, which is outside the narrow grid."
    )


# ── Free-z transparency: ztable auto-extends, fixed-z stays as plain LUT ─────


def test_free_z_model_with_wave_precomp_handles_off_grid_redshifts(free_z_spec, ssp, obs):
    """When ``redshift`` is free, ``WavePrecomp()`` must publish a ztable
    so the model evaluates at arbitrary redshift in the prior — not just
    at the grid points. This is the behavioral check that ``ztable``
    auto-enabled internally.
    """
    model = _silent_build(free_z_spec, ssp, obs, approx=WavePrecomp(n_z=50))
    for z in (0.51, 0.73, 1.27, 1.49):
        params = {"redshift": jnp.asarray(z)}
        out = model.predict_photometry(params)
        assert jnp.all(jnp.isfinite(out)), f"non-finite photometry at z={z}"
        assert out.shape[0] == 5


# ── Catalog-inference safety (Phase 3d-4) ────────────────────────────────────


def test_wave_precomp_n_z_creates_distinct_compile_signatures(fixed_z_spec, ssp, obs):
    """``WavePrecomp(n_z=100)`` and ``WavePrecomp(n_z=200)`` must produce
    distinct ``compile_signature()`` tuples so the structural kernel cache
    doesn't reuse one model's compiled LUT for the other. If the n_z
    override falls out of the signature, catalog inference with mixed
    sampling silently miscompiles.
    """
    m100 = _silent_build(fixed_z_spec, ssp, obs, approx=WavePrecomp(n_z=100))
    m200 = _silent_build(fixed_z_spec, ssp, obs, approx=WavePrecomp(n_z=200))
    assert m100.compile_signature() != m200.compile_signature(), (
        "WavePrecomp(n_z=100) and WavePrecomp(n_z=200) share a compile_signature; "
        "they would collide in the structural kernel cache."
    )


def test_wave_precomp_z_bounds_create_distinct_compile_signatures(fixed_z_spec, ssp, obs):
    """Same guarantee for ``z_min`` / ``z_max`` overrides."""
    m1 = _silent_build(fixed_z_spec, ssp, obs, approx=WavePrecomp(z_min=0.5, z_max=1.5))
    m2 = _silent_build(fixed_z_spec, ssp, obs, approx=WavePrecomp(z_min=0.6, z_max=1.5))
    assert m1.compile_signature() != m2.compile_signature(), (
        "WavePrecomp(z_min=0.5) and WavePrecomp(z_min=0.6) share a compile_signature; "
        "they would collide in the structural kernel cache."
    )


def test_predict_observables_routes_through_lut_when_wave_precomp(fixed_z_spec, ssp, obs):
    """When built with ``approx=WavePrecomp()``, ``predict_observables``
    returns the LUT-projected flux.
    """
    model = _silent_build(fixed_z_spec, ssp, obs, approx=WavePrecomp())
    params = {}

    via_method = model.predict_observables(params).phot_fnu

    state = model.predict_state(params)
    full = {**model.spec.get_fixed_values(), **params}
    via_obs = model.observation.predict_via_precomp(state, full)["phot_fnu"]

    assert jnp.array_equal(via_method, via_obs), (
        "predict_observables must equal observation.predict_via_precomp().phot_fnu "
        "when built with approx=WavePrecomp()."
    )


def test_predict_observables_jit_routes_through_lut_when_wave_precomp(fixed_z_spec, ssp, obs):
    """Catalog inference uses ``predict_observables_jit``; the JIT'd path
    must also route through the LUT when built with ``approx=WavePrecomp()``,
    otherwise the build-time speed knob never reaches the fitter.
    Bit-exact agreement with the non-JIT path proves both go through the
    same projection.
    """
    model = _silent_build(fixed_z_spec, ssp, obs, approx=WavePrecomp())
    params = {}

    via_jit = model.predict_observables_jit(params).phot_fnu
    via_method = model.predict_observables(params).phot_fnu

    # JIT and non-JIT paths agree to floating-point precision (JIT may
    # reorder ops, so bit-equal isn't guaranteed).
    assert jnp.allclose(via_jit, via_method, rtol=1e-12, atol=0), (
        "predict_observables_jit must equal predict_observables when both are "
        "routed through the LUT (built with approx=WavePrecomp())."
    )


# ── Cross-galaxy compile reuse (Phase 4-A) ───────────────────────────────────


def _build_catalog_galaxy(ssp, obs, *, dust_tau_bc_fixed):
    """Build a catalog model where the per-galaxy parameter is a
    *runtime-read* fixed value (``dust_tau_bc``), not a chain-construction-
    time one (``redshift``).

    Redshift specifically gets baked into the stellar component's LUT at
    chain-build time when no ``WavePrecomp(ztable)`` is in use, so two
    models with different fixed ``redshift`` would have structurally
    different chains. Per-galaxy ``dust_tau_bc`` is the cleaner test
    because dust attenuation is evaluated at runtime from
    ``params['dust_tau_bc']`` regardless of fixed/free status.
    """
    spec = Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),  # SAME for both galaxies — chain matches
        dust_tau_bc=Fixed(dust_tau_bc_fixed),  # ← per-galaxy
        dust_tau_diff=Fixed(0.0),
        igm={"type": "none"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs)


def test_compile_signature_drops_fixed_value(ssp, obs):
    """Two SEDModels that differ only in fixed-parameter VALUES (not in
    the set of fixed parameter NAMES) must share a compile_signature.
    Pre-Phase 4-A the per-galaxy fixed dust_tau_bc was baked into the
    signature; Phase 4-A drops it so the structural cache hits.
    """
    m1 = _build_catalog_galaxy(ssp, obs, dust_tau_bc_fixed=0.1)
    m2 = _build_catalog_galaxy(ssp, obs, dust_tau_bc_fixed=0.8)
    assert m1.compile_signature() == m2.compile_signature(), (
        "Two models with identical structural config but different fixed "
        "dust_tau_bc values must share a compile_signature after Phase 4-A — "
        "fixed values are threaded as JIT runtime inputs, not baked into "
        "the cache key."
    )


def test_cross_galaxy_predict_observables_jit_uses_per_galaxy_values(ssp, obs):
    """The Phase 4-A payoff: two galaxies with identical structure but
    different runtime-read fixed values (``dust_tau_bc``) each get the
    correct per-galaxy result from ``predict_observables_jit`` — even
    though they share one compiled function under the hood.
    """
    m1 = _build_catalog_galaxy(ssp, obs, dust_tau_bc_fixed=0.1)
    m2 = _build_catalog_galaxy(ssp, obs, dust_tau_bc_fixed=0.8)

    # Shared compile slot.
    assert m1.compile_signature() == m2.compile_signature()

    # Both calls produce finite photometry.
    phot1 = m1.predict_observables_jit({}).phot_fnu
    phot2 = m2.predict_observables_jit({}).phot_fnu
    chex.assert_tree_all_finite(phot1)
    chex.assert_tree_all_finite(phot2)
    # The two galaxies produce DIFFERENT photometry — proves each call
    # threaded its own fixed_values rather than reusing the first model's.
    # Heavier dust attenuation (galaxy 2) gives fainter flux.
    assert jnp.all(phot2 < phot1), (
        "Galaxy 2 (dust_tau_bc=0.8) should be fainter than galaxy 1 "
        "(dust_tau_bc=0.1) — heavier attenuation. If they're equal or "
        "reversed, the closure leaked the first model's fixed values."
    )

    # Cross-check: each result agrees with the non-JIT path on the same
    # model, confirming the per-galaxy fixed value flowed through both
    # the orchestrator chain and the observation projection.
    assert jnp.allclose(phot1, m1.predict_observables({}).phot_fnu, rtol=1e-12)
    assert jnp.allclose(phot2, m2.predict_observables({}).phot_fnu, rtol=1e-12)
