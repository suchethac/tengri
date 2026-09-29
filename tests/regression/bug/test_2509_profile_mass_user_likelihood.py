# SPDX-License-Identifier: BSD-3-Clause
"""#2509: profile_mass guards refuse user-supplied likelihoods across all data types.

The profiled mass marginalization uses a diagonal Gaussian chi-square on the
Fitter's own data/noise arrays to analytically absorb a mass amplitude. A
user-supplied likelihood owns the data; if the profiled quadratic is engaged,
it silently replaces the user's likelihood in the mass direction, corrupting
the posterior. This bug affected only measured-line-flux channels (which
themselves have a mass-proportional prediction) before this fix; now all
data types are protected.

Similarly, a spectral covariance makes the likelihood a full multivariate
Gaussian, which the diagonal profiled quadratic cannot absorb.

``tests/inference/`` is auto-marked ``slow`` (see ``tests/conftest.py``); run
with ``-m slow``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    ForwardModel,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    builders,
    generate_mock,
    recipes,
)
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]


def _user_likelihood():
    """A minimal user-supplied likelihood satisfying the protocol."""

    class _UserLikelihood:
        name = "user_supplied"

        def log_prob(self, prediction):  # pragma: no cover
            return jnp.asarray(0.0)

    return _UserLikelihood()


def _model_phot_only(ssp_data):
    """Photometry only."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    recipe = recipes.mock_recovery_minimal()
    recipe["neb"] = builders.neb.ssp()
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipe), obs


def _model_spec_only(ssp_data):
    """Spectroscopy only."""
    obs = Observation(
        spectroscopy=Spectroscopy(
            wave_obs=jnp.logspace(2.0, 5.0, 300),
            calibration_order=0,
        ),
    )
    recipe = recipes.mock_recovery_minimal()
    recipe["neb"] = builders.neb.ssp()
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipe), obs


def _model_joint(ssp_data):
    """Joint photometry + spectroscopy."""
    obs = Observation(
        photometry=Photometry.from_names(_FILTERS),
        spectroscopy=Spectroscopy(
            wave_obs=jnp.logspace(2.0, 5.0, 300),
            calibration_order=0,
        ),
    )
    recipe = recipes.mock_recovery_minimal()
    recipe["neb"] = builders.neb.ssp()
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipe), obs


def _model_spec_with_covariance(ssp_data):
    """Spectroscopy with a spectral covariance (small grid, for speed)."""
    wave_obs = jnp.logspace(2.0, 5.0, 50)
    cov = np.eye(len(wave_obs)) * 0.01
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=wave_obs, calibration_order=0, covariance=cov),
    )
    recipe = recipes.mock_recovery_minimal()
    recipe["neb"] = builders.neb.ssp()
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipe), obs


def _model_joint_with_covariance(ssp_data):
    """Joint photometry + spectroscopy-with-covariance (small grid, for speed)."""
    wave_obs = jnp.logspace(2.0, 5.0, 50)
    cov = np.eye(len(wave_obs)) * 0.01
    obs = Observation(
        photometry=Photometry.from_names(_FILTERS),
        spectroscopy=Spectroscopy(wave_obs=wave_obs, calibration_order=0, covariance=cov),
    )
    recipe = recipes.mock_recovery_minimal()
    recipe["neb"] = builders.neb.ssp()
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipe), obs


def _model_phot_with_noise(ssp_data, *, calibration_floor=0.0, student_t_dof=None):
    """Photometry with a variable-noise model (calibration floor and/or Student-t dof).

    ``calibration_floor > 0`` alone (``student_t_dof=None``) exercises the
    ``has_noise_model`` branch of ``build_base_likelihood`` / the "a
    variable-noise model (noise_frac_cal) is configured" guard.
    ``student_t_dof`` set exercises the ``uses_student_t`` branch / the
    "likelihood is Student-t (noise_dof != 0)" guard.
    """
    obs = Observation(
        photometry=Photometry.from_names(_FILTERS),
        noise=NoiseModel(calibration_floor=calibration_floor, student_t_dof=student_t_dof),
    )
    recipe = recipes.mock_recovery_minimal()
    recipe["neb"] = builders.neb.ssp()
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipe), obs


def _fitter(model, obs, ssp_data, profile_mass="auto", likelihood=None, data_type=None, **kwargs):
    """Build a fitter with the given configuration.

    ``**kwargs`` forwards additional ``Fitter`` constructor arguments (e.g.
    ``data_mask=``, ``calibration_marginalize=``) for configurations the
    plain phot/spec/joint call sites above don't need.
    """
    key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
    truth = model.spec.sample(key_truth)

    # For spectroscopy-only, build mock spectrum manually since generate_mock is photometry-only
    if data_type == "spectroscopy" or (
        hasattr(obs, "spectroscopy")
        and obs.spectroscopy is not None
        and (not hasattr(obs, "photometry") or obs.photometry is None)
    ):
        flux_true = model.predict_spectrum(truth)
        # Use a fixed SNR of 30 with a floor to avoid zero noise
        noise = jnp.maximum(jnp.abs(flux_true) / 30.0, 1e-3 * jnp.max(jnp.abs(flux_true)))
        flux = flux_true + noise * jax.random.normal(key_mock, flux_true.shape)
        flux_obs = flux
    else:
        mock = generate_mock(model, truth, key=key_mock, snr=30.0)
        flux_obs = jnp.asarray(mock["flux_obs"])
        noise = jnp.asarray(mock["noise"])

    forward = ForwardModel.build(sed=model, observation=obs)
    fitter = Fitter(
        forward,
        flux_obs,
        noise,
        profile_mass=profile_mass,
        likelihood=likelihood,
        data_type=data_type,
        **kwargs,
    )
    return fitter


class TestUserSuppliedLikelihoodRefusal:
    """User-supplied likelihoods must not be silently replaced by profiled mass."""

    def test_photometry_user_likelihood_auto_disables(self, ssp_data_wne):
        """Photometry with user likelihood: auto disables profiling with reason."""
        model, obs = _model_phot_only(ssp_data_wne)
        fitter = _fitter(model, obs, ssp_data_wne, likelihood=_user_likelihood())
        assert not fitter._profile_mass
        reason = fitter._profile_mass_reason or ""
        assert "user-supplied" in reason, reason

    def test_photometry_user_likelihood_true_raises(self, ssp_data_wne):
        """Photometry with user likelihood: True raises ValueError."""
        model, obs = _model_phot_only(ssp_data_wne)
        with pytest.raises(ValueError, match="user-supplied"):
            _fitter(model, obs, ssp_data_wne, profile_mass=True, likelihood=_user_likelihood())

    def test_joint_user_likelihood_auto_disables(self, ssp_data_wne):
        """Joint data with user likelihood: auto disables profiling with reason."""
        model, obs = _model_joint(ssp_data_wne)
        fitter = _fitter(model, obs, ssp_data_wne, likelihood=_user_likelihood())
        assert not fitter._profile_mass
        reason = fitter._profile_mass_reason or ""
        assert "user-supplied" in reason, reason

    def test_joint_user_likelihood_true_raises(self, ssp_data_wne):
        """Joint data with user likelihood: True raises ValueError."""
        model, obs = _model_joint(ssp_data_wne)
        with pytest.raises(ValueError, match="user-supplied"):
            _fitter(model, obs, ssp_data_wne, profile_mass=True, likelihood=_user_likelihood())


class TestProfileMassWithStandardData:
    """Control: standard data without user likelihood should still engage."""

    def test_photometry_standard_engages(self, ssp_data_wne):
        """Photometry without complications should engage profiling."""
        model, obs = _model_phot_only(ssp_data_wne)
        fitter = _fitter(model, obs, ssp_data_wne)
        assert fitter._profile_mass

    def test_joint_standard_engages(self, ssp_data_wne):
        """Joint data without complications should engage profiling."""
        model, obs = _model_joint(ssp_data_wne)
        fitter = _fitter(model, obs, ssp_data_wne)
        assert fitter._profile_mass


class TestSpectroscopyUserLikelihood:
    """Spectroscopy data with user-supplied likelihood must refuse profile_mass."""

    def test_spectroscopy_user_likelihood_auto_disables(self, ssp_data_wne):
        """Spectroscopy with user likelihood: auto disables profiling with reason."""
        model, obs = _model_spec_only(ssp_data_wne)
        fitter = _fitter(
            model, obs, ssp_data_wne, likelihood=_user_likelihood(), data_type="spectroscopy"
        )
        assert not fitter._profile_mass
        reason = fitter._profile_mass_reason or ""
        assert "user-supplied" in reason, reason

    def test_spectroscopy_user_likelihood_true_raises(self, ssp_data_wne):
        """Spectroscopy with user likelihood: True raises ValueError."""
        model, obs = _model_spec_only(ssp_data_wne)
        with pytest.raises(ValueError, match="user-supplied"):
            _fitter(
                model,
                obs,
                ssp_data_wne,
                profile_mass=True,
                likelihood=_user_likelihood(),
                data_type="spectroscopy",
            )


class TestCanonicalForwardModelPath:
    """Test via explicit ForwardModel.build + Fitter(data_type=...) API path."""

    def test_photometry_via_forward_model(self, ssp_data_wne):
        """Photometry via canonical ForwardModel.build path."""
        model, obs = _model_phot_only(ssp_data_wne)
        key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
        truth = model.spec.sample(key_truth)
        mock = generate_mock(model, truth, key=key_mock, snr=30.0)

        forward = ForwardModel.build(sed=model, observation=obs)
        fitter = Fitter(
            forward,
            jnp.asarray(mock["flux_obs"]),
            jnp.asarray(mock["noise"]),
            data_type="photometry",
            profile_mass="auto",
            likelihood=_user_likelihood(),
        )
        assert not fitter._profile_mass
        reason = fitter._profile_mass_reason or ""
        assert "user-supplied" in reason, reason

    def test_spectroscopy_via_forward_model(self, ssp_data_wne):
        """Spectroscopy via canonical ForwardModel.build path."""
        model, obs = _model_spec_only(ssp_data_wne)
        key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
        truth = model.spec.sample(key_truth)
        flux_true = model.predict_spectrum(truth)
        noise = jnp.maximum(jnp.abs(flux_true) / 30.0, 1e-3 * jnp.max(jnp.abs(flux_true)))
        flux = flux_true + noise * jax.random.normal(key_mock, flux_true.shape)

        forward = ForwardModel.build(sed=model, observation=obs)
        fitter = Fitter(
            forward,
            flux,
            noise,
            data_type="spectroscopy",
            profile_mass="auto",
            likelihood=_user_likelihood(),
        )
        assert not fitter._profile_mass
        reason = fitter._profile_mass_reason or ""
        assert "user-supplied" in reason, reason


class TestSpectralCovarianceRefusal:
    """Spectral covariance requires full multivariate likelihood, incompatible with profiling."""

    def test_spectroscopy_with_covariance_auto_disables(self, ssp_data_wne):
        """Spectroscopy with spectral covariance: auto disables profiling with reason."""

        # Build spectroscopy with covariance
        wave_obs = jnp.logspace(2.0, 5.0, 50)  # Smaller for speed
        cov = np.eye(len(wave_obs)) * 0.01  # Diagonal covariance for testing
        obs = Observation(
            spectroscopy=Spectroscopy(
                wave_obs=wave_obs,
                calibration_order=0,
                covariance=cov,
            ),
        )
        recipe = recipes.mock_recovery_minimal()
        recipe["neb"] = builders.neb.ssp()
        model = SEDModel.build(ssp_data=ssp_data_wne, observation=obs, **recipe)

        # Build fitter
        key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
        truth = model.spec.sample(key_truth)
        flux_true = model.predict_spectrum(truth)
        noise = jnp.maximum(jnp.abs(flux_true) / 30.0, 1e-3 * jnp.max(jnp.abs(flux_true)))
        flux = flux_true + noise * jax.random.normal(key_mock, flux_true.shape)

        forward = ForwardModel.build(sed=model, observation=obs)
        fitter = Fitter(
            forward,
            flux,
            noise,
            data_type="spectroscopy",
            profile_mass="auto",
        )
        assert not fitter._profile_mass
        reason = fitter._profile_mass_reason or ""
        assert "covariance" in reason, reason

    def test_spectroscopy_with_covariance_true_raises(self, ssp_data_wne):
        """Spectroscopy with spectral covariance: True raises ValueError."""

        # Build spectroscopy with covariance
        wave_obs = jnp.logspace(2.0, 5.0, 50)
        cov = np.eye(len(wave_obs)) * 0.01
        obs = Observation(
            spectroscopy=Spectroscopy(
                wave_obs=wave_obs,
                calibration_order=0,
                covariance=cov,
            ),
        )
        recipe = recipes.mock_recovery_minimal()
        recipe["neb"] = builders.neb.ssp()
        model = SEDModel.build(ssp_data=ssp_data_wne, observation=obs, **recipe)

        key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
        truth = model.spec.sample(key_truth)
        flux_true = model.predict_spectrum(truth)
        noise = jnp.maximum(jnp.abs(flux_true) / 30.0, 1e-3 * jnp.max(jnp.abs(flux_true)))
        flux = flux_true + noise * jax.random.normal(key_mock, flux_true.shape)

        forward = ForwardModel.build(sed=model, observation=obs)
        with pytest.raises(ValueError, match="covariance"):
            Fitter(
                forward,
                flux,
                noise,
                data_type="spectroscopy",
                profile_mass=True,
            )

    def test_photometry_not_refused_by_spectral_covariance(self, ssp_data_wne):
        """Photometry fit is NOT refused merely because model has spectral covariance capability.

        The covariance guard should only apply to spectroscopy/joint data_type,
        not to photometry-only fits of a model that happens to have spectroscopy
        configured with covariance.
        """

        # Build joint model with covariance, but fit photometry only
        wave_obs = jnp.logspace(2.0, 5.0, 50)
        cov = np.eye(len(wave_obs)) * 0.01
        obs = Observation(
            photometry=Photometry.from_names(_FILTERS),
            spectroscopy=Spectroscopy(
                wave_obs=wave_obs,
                calibration_order=0,
                covariance=cov,
            ),
        )
        recipe = recipes.mock_recovery_minimal()
        recipe["neb"] = builders.neb.ssp()
        model = SEDModel.build(ssp_data=ssp_data_wne, observation=obs, **recipe)

        key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
        truth = model.spec.sample(key_truth)
        mock = generate_mock(model, truth, key=key_mock, snr=30.0)

        forward = ForwardModel.build(sed=model, observation=obs)
        fitter = Fitter(
            forward,
            jnp.asarray(mock["flux_obs"]),
            jnp.asarray(mock["noise"]),
            data_type="photometry",
            profile_mass="auto",
        )
        # Photometry should engage profiling despite spectral covariance being available
        assert fitter._profile_mass


def _is_plain_gaussian_likelihood(likelihood) -> bool:
    """Whether ``likelihood`` is a plain diagonal Gaussian the profiled quadratic may absorb.

    Per ``mass_profile``'s guards (module docstring, "Guards" section): a plain
    ``PhotometryLikelihood`` / ``SpectroscopyLikelihood`` with zero ``sigma_floor``,
    or a ``CompositeLikelihood`` made ENTIRELY of such members (joint phot+spec).
    Everything else -- ``None`` (no Protocol adapter built, e.g. the legacy
    censored-spec/joint bail-out), ``StudentTLikelihood``, ``CensoredLikelihood``,
    ``MultivariateGaussianLikelihood``, ``CalibrationMarginalizedLikelihood``,
    ``ELineMarginalizedLikelihood``/``ELineFittedLikelihood``, or a user-supplied
    object -- is not.
    """
    from tengri.inference.composite_likelihood import CompositeLikelihood
    from tengri.inference.photometry_likelihood import PhotometryLikelihood
    from tengri.inference.spectroscopy_likelihood import SpectroscopyLikelihood

    if likelihood is None:
        return False

    if isinstance(likelihood, (PhotometryLikelihood, SpectroscopyLikelihood)):
        return getattr(likelihood, "sigma_floor", 0.0) == 0.0

    if isinstance(likelihood, CompositeLikelihood):
        return all(_is_plain_gaussian_likelihood(member) for member in likelihood.likelihoods)

    return False


# ── One cheap Fitter-builder per ``build_base_likelihood`` branch ──────────
# (``tengri.inference.likelihood``), plus the user-supplied-likelihood path
# that sits outside that dispatch entirely (``Fitter(likelihood=...)``).
# Each entry is ``(builder, both_directions, case_id)``:
# - ``builder(ssp_data) -> Fitter``, built with ``profile_mass="auto"``.
# - ``both_directions``: assert the full iff (engaged <=> plain Gaussian) when
#   True; when False, assert only "never engaged on a non-plain likelihood"
#   because this is a should-refuse case whose disengagement is expected for
#   a *different* reason than the guard under test (documented per-case below)
#   and the reverse direction would just re-assert that unrelated guard.


def _case_plain(model_fn, data_type):
    def _build(ssp_data):
        model, obs = model_fn(ssp_data)
        return _fitter(model, obs, ssp_data, profile_mass="auto", data_type=data_type)

    return _build


def _case_user_likelihood(model_fn, data_type):
    def _build(ssp_data):
        model, obs = model_fn(ssp_data)
        return _fitter(
            model,
            obs,
            ssp_data,
            profile_mass="auto",
            likelihood=_user_likelihood(),
            data_type=data_type,
        )

    return _build


def _case_student_t(ssp_data):
    model, obs = _model_phot_with_noise(ssp_data, student_t_dof=5.0)
    return _fitter(model, obs, ssp_data, profile_mass="auto", data_type="photometry")


def _case_noise_model(ssp_data):
    model, obs = _model_phot_with_noise(ssp_data, calibration_floor=0.05)
    return _fitter(model, obs, ssp_data, profile_mass="auto", data_type="photometry")


def _case_censored_photometry(ssp_data):
    model, obs = _model_phot_only(ssp_data)
    mask = jnp.zeros(len(_FILTERS), dtype=jnp.int32).at[0].set(1)
    return _fitter(
        model, obs, ssp_data, profile_mass="auto", data_type="photometry", data_mask=mask
    )


def _case_censored_spectroscopy(ssp_data):
    # build_base_likelihood only builds an explicit CensoredLikelihood for
    # data_type="photometry"; for spectroscopy/joint it returns None (bails
    # to the legacy censored dispatch) -- see likelihood.py's comment "Censored
    # mask on spec / joint isn't covered by a single-channel adapter". This
    # case pins that distinction: the guard still refuses on `fitter.data_mask`
    # directly (independent of which likelihood object gets built), so
    # `_user_likelihood` stays None here rather than becoming a CensoredLikelihood.
    model, obs = _model_spec_only(ssp_data)
    n_pix = obs.spectroscopy.wave_obs.shape[0]
    mask = jnp.zeros(n_pix, dtype=jnp.int32).at[0].set(1)
    return _fitter(
        model, obs, ssp_data, profile_mass="auto", data_type="spectroscopy", data_mask=mask
    )


def _case_covariance(model_fn, data_type):
    def _build(ssp_data):
        model, obs = model_fn(ssp_data)
        return _fitter(model, obs, ssp_data, profile_mass="auto", data_type=data_type)

    return _build


def _case_calibration_marginalize(ssp_data):
    model, obs = _model_spec_only(ssp_data)
    return _fitter(
        model,
        obs,
        ssp_data,
        profile_mass="auto",
        data_type="spectroscopy",
        calibration_marginalize=True,
    )


_CONTRACT_CASES = [
    pytest.param(_case_plain(_model_phot_only, "photometry"), True, id="plain_photometry"),
    pytest.param(_case_plain(_model_spec_only, "spectroscopy"), True, id="plain_spectroscopy"),
    pytest.param(_case_plain(_model_joint, "joint"), True, id="plain_joint"),
    pytest.param(
        _case_user_likelihood(_model_phot_only, "photometry"),
        True,
        id="user_likelihood_photometry",
    ),
    pytest.param(
        _case_user_likelihood(_model_spec_only, "spectroscopy"),
        True,
        id="user_likelihood_spectroscopy",
    ),
    pytest.param(_case_user_likelihood(_model_joint, "joint"), True, id="user_likelihood_joint"),
    pytest.param(_case_student_t, True, id="student_t_photometry"),
    pytest.param(_case_noise_model, True, id="noise_model_photometry"),
    pytest.param(_case_censored_photometry, True, id="censored_photometry"),
    pytest.param(_case_censored_spectroscopy, True, id="censored_spectroscopy"),
    pytest.param(
        _case_covariance(_model_spec_with_covariance, "spectroscopy"),
        True,
        id="spectral_covariance_spectroscopy",
    ),
    pytest.param(
        _case_covariance(_model_joint_with_covariance, "joint"),
        True,
        id="spectral_covariance_joint",
    ),
    pytest.param(_case_calibration_marginalize, True, id="calibration_marginalize_spectroscopy"),
]


class TestContractProfileMassLikelihood:
    """Contract test: profile_mass engages if, and only if, the likelihood is plain Gaussian.

    This is the class sweep for #2509: one cheap configuration per
    ``build_base_likelihood`` branch (``tengri.inference.likelihood``), plus
    the user-supplied-likelihood path. For each, the contract asserted is

        fitter._profile_mass_resolved == _is_plain_gaussian_likelihood(actual)

    in BOTH directions: a non-plain likelihood must never get profiled (the
    #2509 defect class), and every plain config must actually engage (so the
    guards aren't silently over-broad). ``actual`` is read off
    ``fitter._user_likelihood`` *after* ``Fitter.__init__`` returns: that
    attribute is set to the user-supplied object (or ``None``) early in
    ``__init__``, read by ``mass_profile._check_guards`` in that state, and
    then -- for the auto-protocol path (default) -- overwritten in place with
    the auto-built adapter cohort (``PhotometryLikelihood`` /
    ``SpectroscopyLikelihood`` / ``CompositeLikelihood`` / ``StudentTLikelihood``
    / ``CensoredLikelihood`` / ``MultivariateGaussianLikelihood`` /
    ``CalibrationMarginalizedLikelihood`` / e-line adapters) once the guard
    check is behind it. By the time a caller outside ``__init__`` reads
    ``fitter._user_likelihood``, it names the likelihood the fit actually
    scores with.
    """

    @pytest.mark.parametrize("build_fitter,both_directions", _CONTRACT_CASES)
    def test_contract(self, ssp_data_wne, build_fitter, both_directions):
        fitter = build_fitter(ssp_data_wne)
        actual = getattr(fitter, "_user_likelihood", None)
        resolved = fitter._profile_mass_resolved
        plain = _is_plain_gaussian_likelihood(actual)
        lk_type = type(actual).__name__

        if both_directions:
            assert resolved == plain, (
                f"contract violated: _profile_mass_resolved={resolved} but "
                f"is_plain_gaussian(likelihood)={plain} (likelihood={lk_type}, "
                f"reason={fitter._profile_mass_reason!r})"
            )
        else:
            # Only "never engage on a non-plain likelihood": this case's
            # disengagement is expected for an unrelated guard (documented at
            # its builder), so asserting the reverse direction here would
            # merely re-assert that unrelated guard instead of this contract.
            assert not resolved or plain, (
                f"profile_mass engaged on a non-plain likelihood: {lk_type} "
                f"(reason={fitter._profile_mass_reason!r})"
            )
