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

import pathlib
import warnings

import jax.numpy as jnp
import pytest

from tengri import Parameters, SEDModel, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation import Observation, Photometry
from tengri.parameters.priors import Fixed, Uniform

_SSP = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp():
    if not _SSP.exists():
        pytest.skip(f"SSP not available at {_SSP}")
    return load_ssp_data(str(_SSP))


@pytest.fixture(scope="module")
def obs():
    return Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )


@pytest.fixture(scope="module")
def fixed_z_spec():
    """Minimal Fixed-everywhere spec — validator tests don't need free params."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_peak_sfr=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Fixed(0.1),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )


@pytest.fixture(scope="module")
def free_z_spec():
    """Free-redshift spec — only test that needs ztable interpolation."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(5.0),
        sfh_dpl_log_peak_sfr=Fixed(1.0),
        met_logzsol=Fixed(-0.5),
        redshift=Uniform(0.5, 1.5),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
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
    assert jnp.all(jnp.isfinite(out))

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

    assert jnp.array_equal(via_method, via_obs), (
        "predict_photometry must equal observation.predict_via_precomp().phot_fnu "
        "when built with approx=WavePrecomp() — bit-identical, not just close."
    )


# ── ztable sampling knobs change physics for free-z models ──────────────────


def test_wave_precomp_n_z_changes_free_z_interpolation(free_z_spec, ssp, obs):
    """A coarse ztable (``n_z=10``) gives a different interpolated flux at
    an off-grid redshift than a fine one (``n_z=200``). If ``n_z`` is silently
    ignored both would be bit-identical — pinning that ``n_z`` actually
    flows through to the interpolation grid.
    """
    coarse = _silent_build(
        free_z_spec, ssp, obs, approx=WavePrecomp(n_z=10, z_min=0.5, z_max=1.5)
    )
    fine = _silent_build(
        free_z_spec, ssp, obs, approx=WavePrecomp(n_z=200, z_min=0.5, z_max=1.5)
    )

    # Off-grid redshift — coarse ztable resolves it through wider interpolation.
    params = {"redshift": jnp.asarray(0.73)}
    phot_coarse = coarse.predict_photometry(params)
    phot_fine = fine.predict_photometry(params)

    assert jnp.all(jnp.isfinite(phot_coarse))
    assert jnp.all(jnp.isfinite(phot_fine))
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
    narrow = _silent_build(
        free_z_spec, ssp, obs, approx=WavePrecomp(n_z=20, z_min=0.8, z_max=1.2)
    )
    wide = _silent_build(
        free_z_spec, ssp, obs, approx=WavePrecomp(n_z=20, z_min=0.5, z_max=1.5)
    )

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
    at the grid points. This is the behavioural check that ``ztable``
    auto-enabled internally.
    """
    model = _silent_build(free_z_spec, ssp, obs, approx=WavePrecomp(n_z=50))
    for z in (0.51, 0.73, 1.27, 1.49):
        params = {"redshift": jnp.asarray(z)}
        out = model.predict_photometry(params)
        assert jnp.all(jnp.isfinite(out)), f"non-finite photometry at z={z}"
        assert out.shape[0] == 5
