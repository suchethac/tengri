"""Phase 3d — ``approx=`` kwarg contract on :class:`SEDModel`.

Pins the post-3d surface:

* ``approx=None`` (default) → exact wave-grid path; no LUTs.
* ``approx=WavePrecomp()`` → opt-in LUT path with default ztable grid.
* ``approx=WavePrecomp(n_z=..., z_min=..., z_max=...)`` → same path, custom
  ztable sampling.
* Anything else (dict, bool, any string) → ``TypeError`` with a migration
  message that names the new legal forms.
"""

from __future__ import annotations

import pathlib
import warnings

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


def _make_spec(*, free_z: bool):
    return Parameters(
        mean_sfh_type=["tsnorm"],
        sfh_tsnorm_log_peak_sfr=Uniform(-1, 3),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Fixed(-0.5),
        redshift=Uniform(0.01, 2.0) if free_z else Fixed(0.05),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Fixed(0.0),
        apply_igm=False,
    )


@pytest.fixture(scope="module")
def obs():
    phot = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    return Observation(photometry=phot)


def _build(spec, ssp, obs, approx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(spec, ssp, observation=obs, approx=approx)


# ── default / opt-in semantics ───────────────────────────────────────────────


def test_approx_none_is_default_and_disables_lut(ssp, obs):
    spec = _make_spec(free_z=False)
    model = _build(spec, ssp, obs, approx=None)
    assert model._approx["wave_precomp"] is False
    assert model._approx["ztable"] is False
    assert model._approx_config is None


def test_default_wave_precomp_has_default_config(ssp, obs):
    spec = _make_spec(free_z=False)
    model = _build(spec, ssp, obs, approx=WavePrecomp())
    assert model._approx["wave_precomp"] is True
    assert isinstance(model._approx_config, WavePrecomp)
    assert model._approx_config.n_z == 100
    assert model._approx_config.z_min is None
    assert model._approx_config.z_max is None


def test_approx_wave_precomp_object_carries_user_config(ssp, obs):
    spec = _make_spec(free_z=True)
    cfg = WavePrecomp(n_z=250, z_min=0.0, z_max=3.5)
    model = _build(spec, ssp, obs, approx=cfg)
    assert model._approx["wave_precomp"] is True
    assert model._approx["ztable"] is True
    assert model._approx_config is cfg


def test_free_z_auto_enables_ztable_under_wave_precomp(ssp, obs):
    spec = _make_spec(free_z=True)
    model = _build(spec, ssp, obs, approx=WavePrecomp())
    assert model._approx["wave_precomp"] is True
    assert model._approx["ztable"] is True


def test_fixed_z_does_not_enable_ztable(ssp, obs):
    spec = _make_spec(free_z=False)
    model = _build(spec, ssp, obs, approx=WavePrecomp())
    assert model._approx["wave_precomp"] is True
    assert model._approx["ztable"] is False


# ── strict-rejection of pre-3d forms ─────────────────────────────────────────


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
        "wave-precomp",
        "exact",
        0,
        1,
    ],
)
def test_approx_rejects_legacy_and_unknown_forms(ssp, obs, bad):
    spec = _make_spec(free_z=False)
    with pytest.raises(TypeError, match="approx="):
        SEDModel(spec, ssp, observation=obs, approx=bad)


# ── ztable sampling override flows through to redshift_spec ──────────────────


def _stellar_state(model):
    return model._build_component_chain()[0]._state


def test_wave_precomp_n_z_override_flows_to_state(ssp, obs):
    spec = _make_spec(free_z=True)
    model = _build(spec, ssp, obs, approx=WavePrecomp(n_z=137))
    ztable = _stellar_state(model).ssp_phot_ztable
    assert ztable is not None
    assert ztable.z_grid.shape[0] == 137


def test_wave_precomp_z_bounds_override_flows_to_state(ssp, obs):
    spec = _make_spec(free_z=True)
    cfg = WavePrecomp(n_z=50, z_min=0.5, z_max=1.5)
    model = _build(spec, ssp, obs, approx=cfg)
    ztable = _stellar_state(model).ssp_phot_ztable
    assert ztable is not None
    z_grid = ztable.z_grid
    assert float(z_grid.min()) == pytest.approx(0.5)
    assert float(z_grid.max()) == pytest.approx(1.5)
    assert z_grid.shape[0] == 50


# ── build-time approx drives runtime path (Phase 3d-2) ───────────────────────


def test_predict_photometry_routes_through_lut_when_wave_precomp(ssp, obs):
    """Building with approx=WavePrecomp() makes predict_photometry use the
    LUT path: predict_photometry == observation.predict_via_precomp().phot_fnu.
    """
    import jax

    spec = _make_spec(free_z=False)
    model = _build(spec, ssp, obs, approx=WavePrecomp())
    params = spec.sample(jax.random.PRNGKey(0))

    via_method = model.predict_photometry(params)

    state = model.predict_state(params)
    full = {**model.spec.get_fixed_values(), **params}
    via_obs = model.observation.predict_via_precomp(state, full)["phot_fnu"]

    import jax.numpy as jnp

    assert jnp.allclose(via_method, via_obs, rtol=0, atol=0), (
        f"predict_photometry must equal observation.predict_via_precomp().phot_fnu "
        f"when built with approx=WavePrecomp(). Got {via_method} vs {via_obs}."
    )


def test_predict_photometry_does_not_use_lut_when_approx_none(ssp, obs):
    """Building without approx leaves predict_photometry on the strategy/
    kernel path, NOT the LUT (which isn't built when wave_precomp=False).
    """
    import jax

    spec = _make_spec(free_z=False)
    model = _build(spec, ssp, obs, approx=None)
    params = spec.sample(jax.random.PRNGKey(1))
    # Just check it runs without going through predict_via_precomp.
    out = model.predict_photometry(params)
    assert out.shape[0] == 5  # five SDSS filters
    # The LUT wasn't published, so predict_via_precomp would fail.
    with pytest.raises((AttributeError, KeyError, AssertionError, ValueError, TypeError)):
        state = model.predict_state(params)
        full = {**model.spec.get_fixed_values(), **params}
        model.observation.predict_via_precomp(state, full)
