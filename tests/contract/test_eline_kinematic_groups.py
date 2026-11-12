# SPDX-License-Identifier: BSD-3-Clause
"""#3332: emission lines in named kinematic groups.

Each group of ``Spectroscopy(eline_kinematics=...)`` owns a velocity offset and an
intrinsic width; ``"add": True`` gives a line a second component with its own
amplitude. The mock spectra here are built from plain NumPy Gaussians, not from the
model's design-matrix builder, so the comparison does not check the builder against
itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.observation.eline_marginalization import (
    build_eline_design_matrix,
    kinematic_param_names,
    line_kinematics,
)
from tengri.observation.line_list import LineList

pytestmark = pytest.mark.contract

_C_KMS = 299792.458
_Z = 0.1
_R = 1500.0
_LINES = ["Hbeta", "OIII_4959", "OIII_5007", "Halpha"]
_WAVE = np.linspace(5200.0, 7400.0, 2000)
_GROUPS = {
    "balmer": {"lines": ["Hbeta", "Halpha"]},
    "broad": {"lines": ["Halpha"], "add": True},
}
# Truth: (sigma [km/s], delta_v [km/s]) per group; "" is the default group ([O III]).
_TRUTH = {"": (80.0, 0.0), "balmer": (150.0, 250.0), "broad": (1200.0, -100.0)}
# Line fluxes [F_nu x Angstrom], by catalog entry.
_AMPS = {
    "Hbeta": 3.0e-16,
    "OIII_4959": 5.0e-16 / 2.98,
    "OIII_5007": 5.0e-16,
    "Halpha": 9.0e-16,
    "Halpha__broad": 2.0e-15,
}
_SIGMA_NOISE = 2.0e-19


def _gaussian(wave, rest, sigma_kms, delta_v_kms, resolution=_R, z=_Z):
    """Unit-area Gaussian line profile on ``wave`` [1/Angstrom], NumPy only."""
    lam = rest * (1.0 + z)
    center = lam * (1.0 + delta_v_kms / _C_KMS)
    width = np.hypot(lam / (2.355 * resolution), lam * sigma_kms / _C_KMS)
    return np.exp(-0.5 * ((wave - center) / width) ** 2) / (np.sqrt(2.0 * np.pi) * width)


def _model(ssp, kinematics=_GROUPS, redshift=None, **priors):
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_WAVE),
        resolution=_R,
        eline_mode="marginalized",
        eline_catalog=LineList.from_names(_LINES),
        eline_prior_sigma=1.0e-14,
        eline_kinematics=kinematics,
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(_Z) if redshift is None else redshift,
        **priors,
    )


def _truth_params(groups=("", "balmer", "broad")):
    params = {}
    for group in groups:
        sigma_name, dv_name = kinematic_param_names(group)
        params[sigma_name], params[dv_name] = _TRUTH[group]
    return params


def _free_all():
    return {
        "eline_sigma_kms": Uniform(0.0, 500.0),
        "eline_delta_v_kms": Uniform(-500.0, 500.0),
        "eline_sigma_kms_balmer": Uniform(0.0, 500.0),
        "eline_delta_v_kms_balmer": Uniform(-500.0, 500.0),
        "eline_sigma_kms_broad": Uniform(300.0, 3000.0),
        "eline_delta_v_kms_broad": Uniform(-500.0, 500.0),
    }


@pytest.fixture(scope="module")
def mock(synthetic_ssp_module):
    """Continuum of the model plus three kinematic groups of lines, with noise."""
    model = _model(synthetic_ssp_module, **_free_all())
    # The continuum does not read the line kinematics; any values serve.
    continuum = np.asarray(model.predict_spectrum(_truth_params()))
    catalog = model.observation.spectroscopy.effective_catalog
    rest = np.asarray(catalog.wavelengths)
    lines = np.zeros_like(_WAVE)
    for name, lam, group in zip(catalog.names, rest, catalog.line_groups):
        lines += _AMPS[name] * _gaussian(_WAVE, lam, *_TRUTH[group])
    rng = np.random.default_rng(3332)
    flux = continuum + lines + rng.normal(0.0, _SIGMA_NOISE, _WAVE.size)
    return model, flux, np.full(_WAVE.size, _SIGMA_NOISE)


@pytest.fixture(scope="module")
def synthetic_ssp_module():
    """Module-scoped copy of the ``synthetic_ssp`` fixture's grid."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    key = jax.random.PRNGKey(0)
    wave = jnp.linspace(3000.0, 10000.0, 100)
    flux = jnp.abs(jax.random.normal(key, (3, 20, 100))) * 1e-3 + 1e-5
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-2.0, -1.0, 0.0]),
    )


def _loglik(model, flux, noise):
    fitter = Fitter(
        model, jnp.asarray(flux), jnp.asarray(noise), data_type="spectroscopy", approx=None
    )
    fn = build_loglikelihood_fn(fitter)
    return lambda params: fn(params, fitter._data_args)


# ── Catalog ───────────────────────────────────────────────────────


def test_groups_label_lines_and_added_component_is_a_new_entry():
    catalog = LineList.from_names(_LINES).with_kinematics(_GROUPS)
    assert catalog.names == (*_LINES, "Halpha__broad")
    assert catalog.line_groups == ("balmer", "", "", "balmer", "broad")
    assert catalog.kinematic_groups == ("balmer", "", "broad")
    np.testing.assert_array_equal(catalog.group_index, [0, 1, 1, 0, 2])
    assert float(catalog.wavelengths[-1]) == float(catalog.wavelengths[3])
    # One fixed-ratio doublet ([O III]) removes one amplitude; the added entry adds one.
    assert catalog.n_independent == 4


def test_catalog_without_groups_is_one_default_group():
    catalog = LineList.from_names(_LINES)
    assert catalog.kinematic_groups == ("",)
    assert catalog.kinematic_group == ()


def test_added_component_of_a_doublet_keeps_the_ratio():
    catalog = LineList.from_names(_LINES).with_kinematics(
        {"wind": {"lines": ["OIII_4959", "OIII_5007"], "add": True}}
    )
    constraint = np.asarray(catalog.build_constraint_matrix())
    names = list(catalog.names)
    row = constraint[names.index("OIII_4959__wind")]
    assert row[row != 0].tolist() == pytest.approx([1.0 / 2.98])
    assert catalog.n_independent == 4


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        ({"Bad-Name": {"lines": ["Halpha"]}}, "lowercase letters"),
        ({"a": {"lines": ["NoSuchLine"]}}, "not in the catalog"),
        ({"a": {"lines": ["Halpha"]}, "b": {"lines": ["Halpha"]}}, "belongs to one group"),
        ({"a": {"lines": ["OIII_5007"]}}, "must stay in one kinematic group"),
        ({"a": {"lines": []}}, "names no lines"),
        ({"a": {"lines": ["Halpha"], "width": 3}}, "unknown keys"),
    ],
)
def test_bad_group_specifications_are_refused(groups, match):
    with pytest.raises(ValueError, match=match):
        LineList.from_names(_LINES).with_kinematics(groups)


def test_resolved_uv_catalog_and_names():
    uv = LineList.default_uv()
    assert {"NIV_1483", "NIV_1486", "CIII_1907", "CIII_1909", "CIV_1548", "CIV_1551"} <= set(
        uv.names
    )
    assert uv.doublets == ()
    assert np.all(np.diff(np.asarray(uv.wavelengths)) > 0)
    mixed = LineList.from_names(["NIV_1486", "Halpha", "CIV_1549"])
    assert set(mixed.names) == {"NIV_1486", "Halpha", "CIV_1549"}


def test_from_names_optical_selection_is_unchanged():
    got = LineList.from_names(_LINES)
    ref = LineList.default_optical().select(names=_LINES)
    assert got.names == ref.names
    assert got.doublets == ref.doublets
    np.testing.assert_array_equal(np.asarray(got.wavelengths), np.asarray(ref.wavelengths))


# ── Design matrix ─────────────────────────────────────────────────


def test_per_line_arrays_match_numpy_gaussians():
    rest = np.array([4862.68, 5008.24, 6564.61])
    sigma = np.array([80.0, 150.0, 1200.0])
    delta_v = np.array([0.0, 250.0, -100.0])
    resolution = np.array([1200.0, 1500.0, 2100.0])
    design = np.asarray(
        build_eline_design_matrix(
            jnp.asarray(_WAVE),
            jnp.asarray(rest),
            jnp.asarray(resolution),
            _Z,
            eline_sigma_kms=jnp.asarray(sigma),
            eline_delta_v_kms=jnp.asarray(delta_v),
        )
    )
    for j in range(3):
        ref = _gaussian(_WAVE, rest[j], sigma[j], delta_v[j], resolution[j])
        np.testing.assert_allclose(design[:, j], ref, rtol=1e-9, atol=1e-300)


def test_scalar_arguments_agree_with_equal_arrays():
    """The per-line kernel reproduces the scalar kernel; scalars keep the scalar kernel."""
    rest = jnp.asarray([4862.68, 5008.24, 6564.61])
    scalar = build_eline_design_matrix(jnp.asarray(_WAVE), rest, _R, _Z, 120.0, 30.0)
    arrays = build_eline_design_matrix(
        jnp.asarray(_WAVE), rest, jnp.full(3, _R), _Z, jnp.full(3, 120.0), jnp.full(3, 30.0)
    )
    np.testing.assert_allclose(np.asarray(scalar), np.asarray(arrays), rtol=1e-10, atol=1e-300)
    mixed = build_eline_design_matrix(jnp.asarray(_WAVE), rest, _R, _Z, jnp.full(3, 120.0), 30.0)
    np.testing.assert_allclose(np.asarray(scalar), np.asarray(mixed), rtol=1e-10, atol=1e-300)


def test_line_kinematics_gathers_each_group():
    catalog = LineList.from_names(_LINES).with_kinematics(_GROUPS)
    params = _truth_params(("", "balmer"))
    fixed = _truth_params(("broad",))
    sigma, delta_v = line_kinematics(params, fixed, catalog)
    np.testing.assert_allclose(np.asarray(sigma), [150.0, 80.0, 80.0, 150.0, 1200.0])
    np.testing.assert_allclose(np.asarray(delta_v), [250.0, 0.0, 0.0, 250.0, -100.0])


# ── Model and likelihood ──────────────────────────────────────────


def test_group_parameters_register_and_free(synthetic_ssp_module):
    model = _model(
        synthetic_ssp_module,
        eline_sigma_kms_broad=Uniform(300.0, 3000.0),
        eline_delta_v_kms_balmer=FREE,
    )
    assert model.spec.eline_groups == ("balmer", "broad")
    assert {"eline_sigma_kms_balmer", "eline_delta_v_kms_broad"} <= set(model.spec.all_params)
    assert set(model.spec.free_params) == {"eline_sigma_kms_broad", "eline_delta_v_kms_balmer"}
    lo, hi = model.spec.get_distribution("eline_delta_v_kms_balmer").bounds
    assert (lo, hi) == (-1000.0, 1000.0)


def test_truth_beats_shifted_kinematics_in_every_group(mock):
    model, flux, noise = mock
    loglik = _loglik(model, flux, noise)
    truth = _truth_params()
    best = float(loglik(truth))
    for name, shift in [
        ("eline_delta_v_kms", 150.0),
        ("eline_sigma_kms", 150.0),
        ("eline_delta_v_kms_balmer", -200.0),
        ("eline_sigma_kms_balmer", 200.0),
        ("eline_delta_v_kms_broad", 400.0),
        ("eline_sigma_kms_broad", 1000.0),
    ]:
        moved = float(loglik({**truth, name: truth[name] + shift}))
        assert best - moved > 10.0, name


def test_gradient_matches_finite_differences_for_every_group_parameter(mock):
    model, flux, noise = mock
    loglik = _loglik(model, flux, noise)
    # Away from the optimum, where the gradient is not near zero.
    point = {k: v + (30.0 if "sigma" in k else 20.0) for k, v in _truth_params().items()}
    point = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in point.items()}
    grad = jax.grad(loglik)(point)
    for name in point:
        step = 1.0e-3
        up = float(loglik({**point, name: point[name] + step}))
        down = float(loglik({**point, name: point[name] - step}))
        fd = (up - down) / (2.0 * step)
        assert abs(fd) > 1.0e-3, name
        assert float(grad[name]) == pytest.approx(fd, rel=2.0e-4), name


def test_one_group_model_is_worse_on_three_group_data(mock, synthetic_ssp_module):
    model, flux, noise = mock
    three = float(_loglik(model, flux, noise)(_truth_params()))
    single = _model(
        synthetic_ssp_module,
        kinematics=None,
        eline_sigma_kms=Uniform(0.0, 500.0),
        eline_delta_v_kms=Uniform(-500.0, 500.0),
    )
    loglik = _loglik(single, flux, noise)
    # The best one-group fit over a grid that brackets every group's truth.
    best = max(
        float(loglik({"eline_sigma_kms": s, "eline_delta_v_kms": v}))
        for s in np.linspace(40.0, 500.0, 24)
        for v in np.linspace(-150.0, 300.0, 19)
    )
    assert three - best > 100.0


def test_broad_component_changes_the_likelihood(mock, synthetic_ssp_module):
    model, flux, noise = mock
    with_broad = float(_loglik(model, flux, noise)(_truth_params()))
    narrow_only = _model(
        synthetic_ssp_module,
        kinematics={"balmer": _GROUPS["balmer"]},
        eline_sigma_kms=Uniform(0.0, 500.0),
        eline_delta_v_kms=Uniform(-500.0, 500.0),
        eline_sigma_kms_balmer=Uniform(0.0, 500.0),
        eline_delta_v_kms_balmer=Uniform(-500.0, 500.0),
    )
    without = float(_loglik(narrow_only, flux, noise)(_truth_params(("", "balmer"))))
    assert with_broad - without > 100.0


def test_no_kinematics_matches_explicit_single_group_arithmetic(synthetic_ssp_module):
    """Without ``eline_kinematics`` the likelihood reads the two default parameters."""
    model = _model(
        synthetic_ssp_module,
        kinematics=None,
        eline_sigma_kms=Uniform(0.0, 500.0),
        eline_delta_v_kms=Uniform(-500.0, 500.0),
    )
    assert model.spec.eline_groups == ()
    assert not [n for n in model.spec.all_params if n.startswith("eline_sigma_kms_")]
    continuum = np.asarray(
        model.predict_spectrum({"eline_sigma_kms": 0.0, "eline_delta_v_kms": 0.0})
    )
    rest = np.asarray(LineList.from_names(_LINES).wavelengths)
    amps = [_AMPS[n] for n in _LINES]
    flux = continuum + sum(a * _gaussian(_WAVE, lam, 120.0, 60.0) for a, lam in zip(amps, rest))
    loglik = _loglik(model, flux, np.full(_WAVE.size, _SIGMA_NOISE))
    at_truth = float(loglik({"eline_sigma_kms": 120.0, "eline_delta_v_kms": 60.0}))
    off = float(loglik({"eline_sigma_kms": 120.0, "eline_delta_v_kms": 160.0}))
    assert at_truth - off > 100.0


def test_fitted_mode_reads_the_group_kinematics(mock, synthetic_ssp_module):
    """With sampled amplitudes the added component is its own amplitude parameter."""
    _, flux, noise = mock
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_WAVE),
        resolution=_R,
        eline_mode="fitted",
        eline_catalog=LineList.from_names(_LINES),
        eline_prior_sigma=1.0e-14,
        eline_kinematics=_GROUPS,
    )
    model = SEDModel.build(
        ssp_data=synthetic_ssp_module,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(_Z),
        **_free_all(),
    )
    fitter = Fitter(
        model, jnp.asarray(flux), jnp.asarray(noise), data_type="spectroscopy", approx=None
    )
    assert "eline_amp_Halpha__broad" in fitter._eline_amplitude_names
    fn = build_loglikelihood_fn(fitter)
    amps = {
        f"eline_amp_{name}": _AMPS[name]
        for name in ("Hbeta", "OIII_5007", "Halpha", "Halpha__broad")
    }
    truth = {**_truth_params(), **amps}
    best = float(fn(truth, fitter._data_args))
    for name, value in [
        ("eline_delta_v_kms_balmer", 0.0),
        ("eline_sigma_kms_broad", 300.0),
        ("eline_amp_Halpha__broad", 0.0),
    ]:
        assert best - float(fn({**truth, name: value}, fitter._data_args)) > 100.0, name


def test_automatic_precompute_builds_without_a_nebular_backend(mock):
    """#3374: nuisance line amplitudes on a model with no nebular backend fit under 'auto'."""
    model, flux, noise = mock
    exact = _loglik(model, flux, noise)
    auto = Fitter(model, jnp.asarray(flux), jnp.asarray(noise), data_type="spectroscopy")
    fn = build_loglikelihood_fn(auto)
    truth = _truth_params()
    assert float(fn(truth, auto._data_args)) == pytest.approx(float(exact(truth)), rel=1e-9)


def test_kinematic_priors_survive_a_round_trip(synthetic_ssp_module):
    """parameters_to_groups emits the line widths and offsets a rebuild needs."""
    from tengri.parameters.groups import parameters_to_groups

    model = _model(
        synthetic_ssp_module,
        eline_sigma_kms=Uniform(0.0, 300.0),
        eline_sigma_kms_broad=Uniform(300.0, 3000.0),
        eline_delta_v_kms_balmer=Fixed(120.0),
    )
    groups = parameters_to_groups(model.spec)
    assert {"eline_sigma_kms", "eline_sigma_kms_broad", "eline_delta_v_kms_balmer"} <= set(groups)
    assert "eline_delta_v_kms" not in groups  # default Fixed(0.0) stays implicit
    rebuilt = SEDModel.build(
        ssp_data=synthetic_ssp_module, observation=model.observation, **groups
    )
    for name in ("eline_sigma_kms", "eline_sigma_kms_broad", "eline_delta_v_kms_balmer"):
        assert rebuilt.spec.get_distribution(name) == model.spec.get_distribution(name), name


# ── Refusals ──────────────────────────────────────────────────────


def test_eline_broad_is_retired_with_a_pointer():
    with pytest.raises(ValueError, match="eline_kinematics"):
        Spectroscopy(wave_obs=jnp.asarray(_WAVE), eline_mode="marginalized", eline_broad=True)
    from tengri import Parameters

    with pytest.raises(ValueError, match="eline_kinematics"):
        Parameters(eline_mode="marginalized", eline_broad=True)


def test_bad_group_fails_at_spectroscopy_construction():
    with pytest.raises(ValueError, match="not in the catalog"):
        Spectroscopy(
            wave_obs=jnp.asarray(_WAVE),
            eline_mode="marginalized",
            eline_catalog=LineList.from_names(_LINES),
            eline_kinematics={"a": {"lines": ["CIV_1549"]}},
        )


def test_added_component_needs_the_flat_amplitude_prior():
    with pytest.raises(ValueError, match="eline_prior_type='flat'"):
        Spectroscopy(
            wave_obs=jnp.asarray(_WAVE),
            eline_mode="marginalized",
            eline_catalog=LineList.from_names(_LINES),
            eline_prior_type="model",
            eline_kinematics={"broad": {"lines": ["Halpha"], "add": True}},
        )


def test_batched_path_refuses_kinematic_groups(synthetic_ssp_module):
    from tengri.forward.batched_spectrum import check_template_model

    model = _model(synthetic_ssp_module, redshift=Uniform(0.0, 0.5))
    with pytest.raises(ValueError, match="kinematic groups"):
        check_template_model(model)
