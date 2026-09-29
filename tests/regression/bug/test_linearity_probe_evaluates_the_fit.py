# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for linearity probe evaluation at the fit's own fixed values.

Tests that:
1. Model-evaluation errors propagate from the probe instead of being logged
2. The probe evaluates at the Fitter's resolved fixed values (including params_override)
3. The numeric-invalid path still works when other thetas are valid
4. Every other site that either (a) rebuilds fixed values from the spec
   alone, or (b) swallows a model-evaluation error into a changed decision,
   is fixed the same way -- see ``_classify_nonproportional`` (the affine
   retest's third-mass evaluation) and ``configure_profile_mass``'s
   ``"auto"`` branch (the guard-check try/except), plus the params_override
   threading into ``_check_guards`` that closes the construction-time gap
   (Fitter.__init__ calls configure_profile_mass before self._fixed_values /
   self._params_override exist, fitter.py ~1570 vs ~1574/~1630).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import Fixed, Observation, Photometry, SEDModel, recipes
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.inference import mass_profile
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


def synthetic_ssp_wide():
    """Minimal synthetic SSP for testing — no untracked data files required."""
    n_met, n_age = 3, 25
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    flux = jnp.abs(flux) + 1e-12
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


def _minimal_photometry_model(ssp_data):
    """Minimal model for testing (photometry only)."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())


def _minimal_photometry_model_at_redshift(ssp_data, z):
    """Minimal photometry model with redshift Fixed at ``z``.

    ``recipes.mock_recovery_minimal()`` already declares its own
    ``redshift=Fixed(0.05)``, so overriding it means replacing that key in
    the recipe's own dict rather than also passing ``redshift=`` as a kwarg
    (``SEDModel.build(..., redshift=Fixed(z), **recipe_kwargs)`` raises
    ``TypeError: got multiple values for keyword argument 'redshift'``).
    """
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    recipe_kwargs = recipes.mock_recovery_minimal()
    recipe_kwargs["redshift"] = Fixed(z)
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipe_kwargs)


def _dummy_obs_data():
    """Create a minimal dummy flux vector for ``Fitter``'s ``data=`` argument.

    ``Fitter.__init__`` does ``self.data = jnp.asarray(data)``: ``data`` must
    be the flat flux array directly, not a dict of named arrays (that raised
    ``ValueError: entry not a 2- or 3- tuple`` from ``jnp.asarray`` trying to
    interpret a dict as a structured-array dtype spec).
    """
    n_bands = len(_FILTERS)
    return jnp.array([1.0] * n_bands)


def test_linearity_probe_model_error_propagates(monkeypatch, caplog):
    """Model-evaluation errors propagate out instead of being logged as invalid thetas."""
    model = _minimal_photometry_model(synthetic_ssp_wide())
    obs_data = _dummy_obs_data()
    # Build with profile_mass=False so probe doesn't run during construction
    fitter = Fitter(model, data=obs_data, noise=0.1, profile_mass=False)

    mass_name = "sfh_tsnorm_log_total_mass"
    mass_bounds = fitter.spec.get_distribution(mass_name).bounds

    # Monkeypatch _predict_full_vector to raise KeyError on first call
    original_predict = mass_profile._predict_full_vector
    call_count = [0]

    def mock_predict_full_vector(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise KeyError("boom")
        return original_predict(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(mass_profile, "_predict_full_vector", mock_predict_full_vector)

        # The probe should propagate the KeyError, not log it as a warning
        with pytest.raises(KeyError, match="boom"):
            mass_profile._linearity_max_deviation(fitter, mass_name, mass_bounds)

    # No "linearity probe: ... valid bands" warning should be present
    assert "linearity probe:" not in caplog.text


def test_linearity_probe_evaluates_at_fitter_fixed_values(monkeypatch):
    """Probe evaluates at Fitter's resolved fixed values, including params_override redshift."""
    # Build model with redshift Fixed at z1; the fit itself runs at z2 via
    # params_override, so a probe reading the spec's OWN fixed value instead
    # of the override would evaluate at the wrong redshift entirely.
    z1 = 0.5
    z2 = 1.5
    model = _minimal_photometry_model_at_redshift(synthetic_ssp_wide(), z1)

    # Create fitter with params_override for runtime redshift
    obs_data = _dummy_obs_data()
    fitter = Fitter(
        model,
        data=obs_data,
        noise=0.1,
        profile_mass=False,
        params_override={"redshift": z2},
    )

    mass_name = "sfh_tsnorm_log_total_mass"
    mass_bounds = fitter.spec.get_distribution(mass_name).bounds

    # Capture the redshift values used in _predict_full_vector calls
    recorded_redshifts = []
    original_predict = mass_profile._predict_full_vector

    def track_redshift_predict(model, data_type, params, **kwargs):
        z_val = params.get("redshift", None)
        if z_val is not None:
            recorded_redshifts.append(float(z_val))
        return original_predict(model, data_type, params, **kwargs)

    # Monkeypatch to track what redshift is used
    with monkeypatch.context() as mp:
        mp.setattr(mass_profile, "_predict_full_vector", track_redshift_predict)
        # Call the probe directly
        mass_profile._linearity_max_deviation(fitter, mass_name, mass_bounds)

    # Verify that all recorded redshifts are z2 (the params_override value)
    # (with some tolerance for floating point)
    assert len(recorded_redshifts) >= 2, "Probe should call predict_full_vector multiple times"
    for z_recorded in recorded_redshifts:
        assert abs(z_recorded - z2) < 1e-12, (
            f"Expected redshift {z2} (from params_override) but probe used {z_recorded}"
        )


def test_linearity_probe_numeric_invalid_path_works(monkeypatch):
    """Numeric-invalid (NaN) predictions skip without raising when other thetas valid."""
    model = _minimal_photometry_model(synthetic_ssp_wide())
    obs_data = _dummy_obs_data()
    # Build with profile_mass=False so probe doesn't run during construction
    fitter = Fitter(model, data=obs_data, noise=0.1, profile_mass=False)

    mass_name = "sfh_tsnorm_log_total_mass"
    mass_bounds = fitter.spec.get_distribution(mass_name).bounds

    # Monkeypatch _predict_full_vector to return NaN for theta index 0
    original_predict_fn = mass_profile._predict_full_vector
    call_count = [0]

    def mock_predict_full_vector(*args, **kwargs):
        call_count[0] += 1
        result = original_predict_fn(*args, **kwargs)
        # Return NaN for the first two calls (theta 0's two masses)
        if call_count[0] <= 2:
            return jnp.full_like(result, jnp.nan)
        return result

    with monkeypatch.context() as mp:
        mp.setattr(mass_profile, "_predict_full_vector", mock_predict_full_vector)
        # This should not raise - the probe should skip the NaN theta
        # and continue with the other 8 thetas
        max_dev, tol, kind = mass_profile._linearity_max_deviation(fitter, mass_name, mass_bounds)

        # Verify the probe still returned valid results (from other 8 thetas)
        assert jnp.isfinite(max_dev), "Max deviation should be finite"
        assert jnp.isfinite(tol), "Tolerance should be finite"
        assert kind in ("proportional", "affine", "nonlinear"), "Kind should be one of the three"


def _call_linearity_max_deviation(fitter, mass_name, mass_bounds):
    """Invoke the probe's own evaluation loop directly."""
    mass_profile._linearity_max_deviation(fitter, mass_name, mass_bounds)


def _call_classify_nonproportional(fitter, mass_name, mass_bounds):
    """Invoke the affine-vs-nonlinear retest's third-mass evaluation directly.

    ``worst`` is a synthetic ``(phys, pred_a, pred_b, valid)`` tuple; its
    contents are never read once ``_predict_full_vector`` (the third-mass
    evaluation) is patched to raise before returning, so a minimal shape-only
    stand-in is enough to reach the site.
    """
    ell_lo, _ = mass_bounds
    pred_a = jnp.array([1.0, 2.0])
    pred_b = jnp.array([10.0, 20.0])
    valid = jnp.array([True, True])
    worst = ({}, pred_a, pred_b, valid)
    mass_profile._classify_nonproportional(
        fitter,
        mass_name,
        worst,
        ell_a=ell_lo,
        ell_b=ell_lo + 1.0,
        use_components=False,
        line_flux_block=None,
    )


def _call_configure_profile_mass_auto(fitter, mass_name, mass_bounds):
    """Invoke the ``profile_mass="auto"`` guard-check dispatch directly."""
    mass_profile.configure_profile_mass(fitter, "auto", None)


@pytest.mark.parametrize(
    "call_site",
    [
        _call_linearity_max_deviation,
        _call_classify_nonproportional,
        _call_configure_profile_mass_auto,
    ],
    ids=["linearity_max_deviation", "classify_nonproportional", "configure_profile_mass_auto"],
)
def test_linearity_probe_every_changed_swallow_site_propagates(monkeypatch, caplog, call_site):
    """Every site this fix touched propagates a model-evaluation error.

    Parametrized over the three (b)-class sites the whole-class fix changed:
    the probe's own evaluation loop (``_linearity_max_deviation``, the
    original fix), the affine-vs-nonlinear retest's third-mass evaluation
    (``_classify_nonproportional``), and the ``profile_mass="auto"``
    guard-check dispatch (``configure_profile_mass``). Each used to fold a
    KeyError/ValueError/etc. raised while evaluating the model into a
    warning, a "nonlinear" refusal, or an "auto-disabled" reason -- hiding a
    real configuration bug behind unrelated-looking text instead of letting
    it surface.
    """
    model = _minimal_photometry_model(synthetic_ssp_wide())
    obs_data = _dummy_obs_data()
    # Build with profile_mass=False so the probe doesn't run during
    # construction; each call_site re-invokes the relevant function directly.
    fitter = Fitter(model, data=obs_data, noise=0.1, profile_mass=False)

    mass_name = "sfh_tsnorm_log_total_mass"
    mass_bounds = fitter.spec.get_distribution(mass_name).bounds

    def raise_boom(*args, **kwargs):
        raise KeyError("boom")

    with monkeypatch.context() as mp:
        mp.setattr(mass_profile, "_predict_full_vector", raise_boom)
        with pytest.raises(KeyError, match="boom"):
            call_site(fitter, mass_name, mass_bounds)

    # No warning should paper over the propagated error: neither the probe's
    # own "N of 9 thetas" warning nor configure_profile_mass's retired
    # "guard check raised" fallback text.
    assert "linearity probe:" not in caplog.text
    assert "guard check raised" not in caplog.text


@pytest.mark.parametrize(
    "simulate_construction_time",
    [False, True],
    ids=["direct_call_post_construction", "construction_time_fixed_values_absent"],
)
def test_linearity_probe_honors_params_override_redshift(monkeypatch, simulate_construction_time):
    """The fit's own redshift override is used, not the spec's declared Fixed value.

    Parametrized over the two states ``_linearity_max_deviation``'s own
    fixed-values resolution must handle: called after ``Fitter.__init__``
    completes (``fitter._fixed_values`` already merged with the override),
    and called with that attribute absent -- reproducing the state this
    function actually sees during real construction, where
    ``configure_profile_mass`` calls it (fitter.py ~1570) BEFORE
    ``self._fixed_values``/``self._params_override`` are set (~1574/~1630).
    The second case is exercised directly, by deleting the already-set
    attribute and passing ``params_override`` as this function's own
    explicit 4th argument, rather than by re-running ``Fitter.__init__``
    with ``profile_mass="auto"``: threading the override through
    ``_check_guards`` during a real construction-time run is a parallel,
    separately owned change (#2509) and out of scope here -- this test
    targets ``_linearity_max_deviation``'s own resolution logic directly,
    independent of how any caller reaches it.
    """
    z1 = 0.5
    z2 = 1.5
    recorded_redshifts = []
    original_predict = mass_profile._predict_full_vector

    def track_redshift_predict(model, data_type, params, **kwargs):
        z_val = params.get("redshift", None)
        if z_val is not None:
            recorded_redshifts.append(float(z_val))
        return original_predict(model, data_type, params, **kwargs)

    model = _minimal_photometry_model_at_redshift(synthetic_ssp_wide(), z1)
    obs_data = _dummy_obs_data()
    fitter = Fitter(
        model,
        data=obs_data,
        noise=0.1,
        profile_mass=False,
        params_override={"redshift": z2},
    )
    mass_name = "sfh_tsnorm_log_total_mass"
    mass_bounds = fitter.spec.get_distribution(mass_name).bounds

    with monkeypatch.context() as mp:
        mp.setattr(mass_profile, "_predict_full_vector", track_redshift_predict)

        if simulate_construction_time:
            # Both fitter._fixed_values AND fitter._params_override exist
            # post-construction (fitter.py ~1574/~1594); delete both to
            # reproduce the state _linearity_max_deviation actually sees when
            # configure_profile_mass calls it during __init__ (~1570, before
            # either is set), and supply the raw params_override as this
            # function's own argument, exactly as a caller mid-construction
            # would. Deleting only _fixed_values would leave
            # fitter._params_override already populated with this same z2
            # (set at real construction time regardless of profile_mass),
            # which would let a reader of that fitter attribute pass by
            # coincidence without ever using the explicit argument this test
            # means to exercise.
            mp.delattr(fitter, "_fixed_values")
            mp.delattr(fitter, "_params_override")
            mass_profile._linearity_max_deviation(
                fitter, mass_name, mass_bounds, params_override={"redshift": z2}
            )
        else:
            mass_profile._linearity_max_deviation(fitter, mass_name, mass_bounds)

    assert len(recorded_redshifts) >= 1, "Probe should call _predict_full_vector at least once"
    for z_recorded in recorded_redshifts:
        assert abs(z_recorded - z2) < 1e-12, (
            f"Expected redshift {z2} (from params_override) but probe used {z_recorded}"
        )
