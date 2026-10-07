# SPDX-License-Identifier: BSD-3-Clause
r"""IGM transmission must act after the galaxy's own kinematics, not before (#2589).

Physical order: stellar kinematics (rest frame, intrinsic to the source) ->
redshift -> IGM transmission (observed frame, a line-of-sight absorber) ->
instrument LSF (on the ground, last). Before this fix, ``Observation.predict``
and ``SEDModel._predict_spectrum_on_grid`` multiplied the IGM transmission
into the rest-frame SED *before* calling
:func:`~tengri.observation.observation.project_spectrum_kernel_split`, which
then convolved the whole thing with one kernel carrying both
:math:`\sigma_v` and :math:`\sigma_{\rm inst}`. A galaxy's own velocity
dispersion cannot physically broaden the IGM's sharp Lyman-limit /
Lyman-:math:`\alpha`-forest edge -- that edge is imprinted along the line of
sight, after the light has left the galaxy -- so putting :math:`\sigma_v` on
the same side of the multiply as ``T`` smeared a feature it should never
touch.

Measured on main (commit 06461cfea), ``igm='inoue14'``, z=6,
:math:`\sigma_v=300` km/s, R=3000: the old order's edge differed from the
physical order by 87% of peak flux with an edge 7.15x shallower (this
file's independent numpy reference reproduces both numbers).

``project_spectrum_kernel_split`` (``src/tengri/observation/observation.py``)
now convolves the stellar piece with :math:`\sigma_v` (via
:func:`~tengri.observation.spectrum.broaden_velocity_only`) before
multiplying by ``igm_trans``, and reassigns the library-deconvolved
instrument kernel (:math:`\sqrt{\sigma_{\rm inst}^2 \cdot
\text{lsf\_scale}^2-\sigma_{\rm lib}^2}`) to the final stage alone. The
fallback (no IGM component configured, ``igm_trans`` is ``None``,
:math:`T\equiv 1` structurally) takes the pre-#2589 single-kernel code path
unchanged.

References
----------
.. [1] Inoue, A. K., Shimizu, I., Iwata, I., & Tanaka, M. (2014).
       "An updated analytic model for attenuation by the intergalactic
       medium." MNRAS, 442, 1805. arXiv:1402.0677.
.. [2] Madau, P. (1995). "Radiative transfer in a clumpy universe: the
       colors of high-redshift galaxies." ApJ, 441, 18.
.. [3] Johnson, B. D., Leja, J., Conroy, C., & Speagle, J. S. (2021).
       "Stellar Population Inference with Prospector." ApJS, 254, 22.
       arXiv:2012.01426.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import curve_fit

from tengri import DEFAULT, FREE, Fixed, Observation, SEDModel, Uniform, load_ssp_data
from tengri.components.igm.igm import igm_absorption
from tengri.observation.observation import (
    _split_stellar_and_instrument_only_sed,
    project_spectrum_kernel_split,
)
from tengri.observation.spectroscopy import Spectroscopy
from tengri.observation.spectrum import project_spectrum
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.regression_bug

_C_KMS = 299792.458
_FWHM_TO_SIGMA = 2.354820045030949  # 2*sqrt(2*ln(2))
_MILES_BARE = "data/fsps_prsc_miles_chabrier.h5"
_HBETA_REST = 4862.68  # vacuum, = EMISSION_LINES["Hbeta"]


def _resolution_for_sigma_inst(sigma_inst_kms: float) -> float:
    """Invert sigma_inst_kms = c / (FWHM_TO_SIGMA * R)."""
    return _C_KMS / (_FWHM_TO_SIGMA * sigma_inst_kms)


def _numpy_log_gaussian_convolve(
    flux: np.ndarray, wave: np.ndarray, sigma_kms: float
) -> np.ndarray:
    """Direct FFT Gaussian convolution in ln(lambda), written fresh in plain
    numpy -- independent of any tengri convolution code (``apply_lsf``,
    ``velocity_broaden``, ``broaden_velocity_only``) -- for the physical-order
    reference of test (a). Exact on a grid uniform in ln(lambda).
    """
    if sigma_kms <= 0.0:
        return np.asarray(flux).copy()
    flux = np.asarray(flux, dtype=np.float64)
    n = flux.size
    dlnwave = np.log(wave[1] / wave[0])
    sigma_pix = (sigma_kms / _C_KMS) / dlnwave
    freq = np.fft.rfftfreq(n)
    kernel = np.exp(-2.0 * np.pi**2 * sigma_pix**2 * freq**2)
    return np.fft.irfft(np.fft.rfft(flux) * kernel, n=n)


def _edge_steepness(wave_obs: np.ndarray, flux: np.ndarray, z: float) -> float:
    """Steepest single-pixel step across the Ly-alpha-forest onset (#2589 probe)."""
    lo_obs, hi_obs = 900.0 * (1.0 + z), 1230.0 * (1.0 + z)
    edge = (wave_obs > lo_obs) & (wave_obs < hi_obs)
    return float(np.max(np.abs(np.diff(flux[edge]))))


# ── (a) tengri's actual order vs. an independent numpy physical-order reference ──


def _fine_log_grid(z: float, n: int = 1 << 16):
    """Fine, log-uniform rest grid spanning the Lyman edge (mirrors the probe)."""
    wave_rest = np.geomspace(700.0, 4000.0, n)
    wave_obs = wave_rest * (1.0 + z)
    sed_rest = (wave_rest / 1000.0) ** (-1.5)  # smooth power-law continuum
    return jnp.asarray(wave_rest), jnp.asarray(wave_obs), jnp.asarray(sed_rest)


@pytest.mark.parametrize("sigma_inst_kms", [50.0, 150.0])
@pytest.mark.parametrize("sigma_v", [0.0, 150.0, 300.0])
@pytest.mark.parametrize("z", [3.0, 6.0])
@pytest.mark.parametrize("igm_model", ["inoue14", "madau"])
def test_matches_physical_order_numpy_reference(igm_model, z, sigma_v, sigma_inst_kms):
    """tengri's actual kernel-split order agrees with an independent numpy
    implementation of the physical order (sigma_v -> IGM -> instrument) to
    max |delta| / peak < 1e-3, and the Ly-forest edge steepness agrees to 2%.

    The numpy reference convolves directly in log(lambda) with its own
    from-scratch FFT kernel (``_numpy_log_gaussian_convolve``), never calling
    tengri's convolution code. ``igm_absorption`` (the model's own Inoue14 /
    Madau dispatch) is reused for T(lambda) -- the quantity under test is the
    ORDER of operations, not a reimplementation of the mean-IGM physics.
    """
    wave_rest, wave_obs, sed_rest = _fine_log_grid(z)
    T = jnp.asarray(igm_absorption(wave_obs, z, igm_model=igm_model))

    # -- physical order, independent numpy implementation --
    sed_v = _numpy_log_gaussian_convolve(np.asarray(sed_rest), np.asarray(wave_rest), sigma_v)
    sed_post_igm = sed_v * np.asarray(T)
    flux_physical = _numpy_log_gaussian_convolve(
        sed_post_igm, np.asarray(wave_obs), sigma_inst_kms
    )

    # -- tengri's actual order, via project_spectrum_kernel_split --
    fake_state = ForwardState(
        wave=wave_rest,
        sed_intrinsic=sed_rest,
        derived={"sed_nebular": jnp.zeros_like(sed_rest), "sed_shock": jnp.zeros_like(sed_rest)},
    )
    resolution = _resolution_for_sigma_inst(sigma_inst_kms)
    # dl_cm chosen so the cosmological (1+z)/(4 pi dl^2) dimming is exactly 1
    # and wave_obs == wave_rest*(1+z) exactly, so project_spectrum's internal
    # rest<->obs resample is the identity: tengri's raw output is directly
    # comparable to the numpy reference's rest-frame amplitude.
    dl_cm = float(np.sqrt((1.0 + z) / (4.0 * np.pi)))
    flux_tengri = np.asarray(
        project_spectrum_kernel_split(
            fake_state,
            sed_rest,
            T,
            wave_rest,
            wave_obs,
            z,
            dl_cm,
            resolution=resolution,
            sigma_lib_kms=0.0,
            sigma_v_kms=sigma_v,
            lsf_scale=1.0,
            n_bins=64,
            conserving=False,
        )
    )

    peak = np.max(flux_physical)
    max_diff = np.max(np.abs(flux_tengri - flux_physical))
    assert max_diff / peak < 1e-3, (igm_model, z, sigma_v, sigma_inst_kms, max_diff / peak)

    jump_tengri = _edge_steepness(np.asarray(wave_obs), flux_tengri, z)
    jump_physical = _edge_steepness(np.asarray(wave_obs), flux_physical, z)
    assert jump_physical / jump_tengri == pytest.approx(1.0, rel=0.02), (
        igm_model,
        z,
        sigma_v,
        sigma_inst_kms,
        jump_physical / jump_tengri,
    )


# ── (b) sigma_v = 0 identical to the pre-change single-kernel result ──


@pytest.mark.parametrize("igm_model", ["inoue14", "madau"])
def test_zero_sigma_v_identical_to_pre_change_single_kernel(igm_model):
    """At sigma_v=0, the new IGM-present three-stage split reproduces the
    pre-#2589 single-kernel formula (reconstructed inline from the retained
    ``_split_stellar_and_instrument_only_sed`` + ``project_spectrum``
    building blocks, which are unchanged by this fix) to 1e-12 relative.
    """
    z = 6.0
    wave_rest, wave_obs, sed_rest = _fine_log_grid(z, n=1 << 14)
    T = jnp.asarray(igm_absorption(wave_obs, z, igm_model=igm_model))
    fake_state = ForwardState(
        wave=wave_rest,
        sed_intrinsic=sed_rest,
        derived={"sed_nebular": jnp.zeros_like(sed_rest), "sed_shock": jnp.zeros_like(sed_rest)},
    )
    resolution = _resolution_for_sigma_inst(100.0)
    dl_cm = jnp.asarray(1.0)

    flux_new = project_spectrum_kernel_split(
        fake_state,
        sed_rest,
        T,
        wave_rest,
        wave_obs,
        z,
        dl_cm,
        resolution=resolution,
        sigma_lib_kms=20.0,
        sigma_v_kms=0.0,
        lsf_scale=1.0,
        n_bins=32,
    )

    # Pre-#2589 formula: IGM folded into sed_atten BEFORE the split, single
    # combined kernel (sigma_v=0 here, so the kernel is sigma_inst/sigma_lib only).
    sed_atten_old = sed_rest * T
    sed_stellar_old, sed_instrument_only_old = _split_stellar_and_instrument_only_sed(
        fake_state, sed_atten_old, T
    )
    flux_stellar_old = project_spectrum(
        sed_stellar_old,
        wave_rest,
        wave_obs,
        z,
        dl_cm,
        resolution=resolution,
        sigma_lib_kms=20.0,
        n_bins=32,
        sigma_v_kms=0.0,
    )
    flux_instr_old = project_spectrum(
        sed_instrument_only_old,
        wave_rest,
        wave_obs,
        z,
        dl_cm,
        resolution=resolution,
        sigma_lib_kms=0.0,
        n_bins=32,
        sigma_v_kms=0.0,
    )
    flux_old = flux_stellar_old + flux_instr_old

    chex.assert_trees_all_close(flux_new, flux_old, rtol=1e-12, atol=0.0)


# ── (c) no-IGM model identical to the pre-change two-piece split ──


def test_no_igm_identical_to_two_piece_split():
    """With no IGM component configured (igm_trans=None), the result is
    bit-for-bit the pre-#2589 two-piece split: the structural fallback
    branch is literally the unchanged code path, not merely numerically
    close.
    """
    wave_rest, wave_obs, sed_rest = _fine_log_grid(3.0, n=1 << 12)
    fake_state = ForwardState(
        wave=wave_rest,
        sed_intrinsic=sed_rest,
        derived={"sed_nebular": jnp.zeros_like(sed_rest), "sed_shock": jnp.zeros_like(sed_rest)},
    )
    resolution = _resolution_for_sigma_inst(80.0)
    dl_cm = jnp.asarray(1.0)

    flux_new = project_spectrum_kernel_split(
        fake_state,
        sed_rest,
        None,
        wave_rest,
        wave_obs,
        3.0,
        dl_cm,
        resolution=resolution,
        sigma_lib_kms=15.0,
        sigma_v_kms=250.0,
        n_bins=16,
    )
    sed_stellar_old, sed_instrument_only_old = _split_stellar_and_instrument_only_sed(
        fake_state, sed_rest, None
    )
    flux_stellar_old = project_spectrum(
        sed_stellar_old,
        wave_rest,
        wave_obs,
        3.0,
        dl_cm,
        resolution=resolution,
        sigma_lib_kms=15.0,
        n_bins=16,
        sigma_v_kms=250.0,
    )
    flux_instr_old = project_spectrum(
        sed_instrument_only_old,
        wave_rest,
        wave_obs,
        3.0,
        dl_cm,
        resolution=resolution,
        sigma_lib_kms=0.0,
        n_bins=16,
        sigma_v_kms=0.0,
    )
    flux_old = flux_stellar_old + flux_instr_old

    chex.assert_trees_all_close(flux_new, flux_old, rtol=0.0, atol=0.0)


# ── (d) line widths unchanged with IGM on ──


def _gaussian_fit_width_kms(wave_obs: np.ndarray, flux: np.ndarray, center: float) -> float:
    def model(x, amp, sigma_aa, cont):
        return cont + amp * np.exp(-0.5 * ((x - center) / sigma_aa) ** 2)

    wave_obs = np.asarray(wave_obs)
    flux = np.asarray(flux)
    p0 = [flux.max() - flux.min(), 2.0, flux.min()]
    popt, _ = curve_fit(model, wave_obs, flux, p0=p0)
    return float(popt[1] / center * _C_KMS)


def _cue_or_skip():
    if not Path("data/cue_weights.npz").is_file():
        pytest.skip("Cue weights (data/cue_weights.npz) not present")


def _ssp_or_skip(path):
    if not Path(path).is_file():
        pytest.skip(f"missing SSP grid {path}")
    return load_ssp_data(path)


def test_hbeta_width_unchanged_with_igm_on():
    """H-beta's fitted width is invariant to sigma_v in {0, 300} km/s and
    matches sqrt(sigma_gas^2 + sigma_inst^2) to 2%, with an IGM component
    configured (#2519 is unaffected by #2589: the nebular/shock piece never
    receives sigma_v in either the old or new code, and its own ``*
    igm_trans`` multiply is the same formula before and after this fix).

    Compares the two sigma_v widths to each other first (immune to the
    windowed-fit line-blend contamination documented in
    ``test_line_kernel_split.py``'s module docstring, a common bias at every
    sigma_v), then both against the absolute formula.
    """
    _cue_or_skip()
    ssp = _ssp_or_skip(_MILES_BARE)
    z = 0.05
    hbeta_obs = _HBETA_REST * (1.0 + z)
    wave_obs = jnp.linspace(hbeta_obs - 15.0, hbeta_obs + 15.0, 2000)
    sigma_inst_kms = 150.0
    resolution = _resolution_for_sigma_inst(sigma_inst_kms)
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=resolution, sigma_lib_kms=70.0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            igm={"type": "inoue14"},
            redshift=Fixed(z),
        )
        merged = base.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
        model = SEDModel(merged, ssp, observation=obs)
    sigma_gas = float(model.spec.get_distribution("neb_eline_sigma_kms").value)
    p0 = dict(model.spec.sample(jax.random.PRNGKey(1)))
    p0["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    p0["dust_tau_bc"] = jnp.asarray(0.0)
    p0["dust_tau_diff"] = jnp.asarray(0.0)

    widths = {}
    for sigma_v in (0.0, 300.0):
        p = dict(p0)
        p["sigma_v_kms"] = jnp.asarray(sigma_v)
        flux = np.asarray(model.predict_spectrum(p))
        widths[sigma_v] = _gaussian_fit_width_kms(np.asarray(wave_obs), flux, hbeta_obs)

    assert widths[300.0] == pytest.approx(widths[0.0], rel=2e-2), widths
    expected = float(np.sqrt(sigma_gas**2 + sigma_inst_kms**2))
    for sigma_v, w in widths.items():
        assert w == pytest.approx(expected, rel=3e-2), (sigma_v, w, expected)


# ── (e) cross-path agreement with IGM on, incl. the LUT and banded paths ──


def _build_igm_model(ssp, wave_obs, z, *, resolution=None, resolution_matrix=None, approx=None):
    spec = Spectroscopy(
        wave_obs=wave_obs,
        resolution=resolution,
        sigma_lib_kms=70.0,
        resolution_matrix=resolution_matrix,
    )
    obs = Observation(spectroscopy=spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
            neb={"type": "none"},
            igm={"type": "inoue14"},
            redshift=Fixed(z),
            approx=approx,
        )
        merged = base.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
        return SEDModel(merged, ssp, observation=obs)


def _params_for(model, sigma_v):
    p = dict(model.spec.sample(jax.random.PRNGKey(11)))
    p["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    p["dust_tau_bc"] = jnp.asarray(0.0)
    p["dust_tau_diff"] = jnp.asarray(0.0)
    p["sigma_v_kms"] = jnp.asarray(sigma_v)
    return p


@pytest.mark.parametrize("sigma_v", [0.0, 300.0])
@pytest.mark.parametrize("z", [3.0, 6.0])
def test_cross_path_agreement_with_igm_gaussian(z, sigma_v):
    """Eager, grid, compiled-kernel, and Prediction-accessor surfaces agree
    to 1e-10 with an IGM component configured, for the default Gaussian LSF.
    """
    ssp = _ssp_or_skip(_MILES_BARE)
    wave_obs = jnp.linspace(900.0 * (1 + z), 1300.0 * (1 + z), 400)
    model = _build_igm_model(ssp, wave_obs, z, resolution=3000.0)
    p = _params_for(model, sigma_v)

    flux_eager = model._spectrum_via_state(p, wave_obs=wave_obs)
    flux_grid = model.predict_spectrum(p, wave_obs=wave_obs)
    flux_jit = model.predict_spectrum(p)
    pred = model.predict(p)
    flux_pred = pred.spectrum()
    flux_pred_grid = pred.spectrum(wave_obs=wave_obs)

    chex.assert_trees_all_close(flux_eager, flux_grid, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_jit, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_pred, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_pred_grid, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("sigma_v", [0.0, 300.0])
def test_banded_path_agrees_with_igm(sigma_v):
    """The banded resolution-matrix path applies sigma_v to the stellar
    piece before T (pre-IGM), then R to everything, and agrees with the
    eager/grid/compiled/Prediction surfaces to 1e-10 with IGM configured.
    """
    from tengri.observation.banded import gaussian_resolution_bands

    ssp = _ssp_or_skip(_MILES_BARE)
    z = 3.0
    wave_obs = jnp.linspace(900.0 * (1 + z), 1300.0 * (1 + z), 300)
    matrix = gaussian_resolution_bands(wave_obs, resolution=3000.0, n_diag=15)
    model = _build_igm_model(ssp, wave_obs, z, resolution_matrix=matrix)
    p = _params_for(model, sigma_v)

    flux_eager = model._spectrum_via_state(p, wave_obs=wave_obs)
    flux_grid = model.predict_spectrum(p, wave_obs=wave_obs)
    flux_jit = model.predict_spectrum(p)
    pred = model.predict(p)
    flux_pred = pred.spectrum()

    chex.assert_trees_all_close(flux_eager, flux_grid, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_jit, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_pred, rtol=1e-10, atol=0.0)


def test_spectrum_precomp_lut_agrees_at_zero_sigma_v():
    """SpectrumPrecomp (the spectrum LUT path) applies no kinematic
    broadening at all -- a documented, pre-existing limitation orthogonal to
    #2589 (``src/tengri/forward/sed_model.py``, the ``spec_lut`` branch of
    the compiled-kernel ``_impl``: "Velocity dispersion / LSF are not
    applied on the per-pixel continuum LUT"). At sigma_v=0 there is no
    kinematics to misorder against IGM, so the LUT must still agree with
    the exact path; #2589 does not touch ``predict_spectrum_via_precomp``
    and does not change this agreement.
    """
    from tengri import SpectrumPrecomp

    ssp = _ssp_or_skip(_MILES_BARE)
    z = 3.0
    wave_obs = jnp.linspace(900.0 * (1 + z), 1300.0 * (1 + z), 200)
    model_exact = _build_igm_model(ssp, wave_obs, z, resolution=None)
    model_lut = _build_igm_model(ssp, wave_obs, z, resolution=None, approx=SpectrumPrecomp())
    p = _params_for(model_exact, 0.0)

    flux_exact = model_exact.predict_spectrum(p)
    flux_lut = model_lut.predict_spectrum(p)
    chex.assert_trees_all_close(flux_exact, flux_lut, rtol=1e-6, atol=0.0)


# ── (f) mutation: restore IGM-before-kinematics ──


def _exec_isolated_module(mutated_path: Path):
    """Execute a mutated source file as a throwaway, uniquely-named module.

    Never touches ``sys.modules`` for the real ``tengri.*`` module path and
    never calls ``importlib.reload`` -- both would replace the live classes
    (e.g. ``ForwardState``, ``Observation``) that every OTHER already-loaded
    module holds a reference to with NEW, distinct class objects, so an
    ``isinstance`` check anywhere else in the same pytest process (worker)
    can start comparing against the wrong incarnation of a class with the
    same name. Measured: reloading ``tengri.observation.observation`` from
    this file's original (reload-based) mutation test, when sharing a
    pytest process with ``tests/unit/inference/test_catalog_histories.py``
    and ``tests/components/sps/test_stellar_precomp_bounds.py``, broke 56
    unrelated tests in those files with
    ``TypeError: observation= must be an Observation (...), got Observation``
    -- two distinct ``Observation`` classes alive at once. Loading the
    mutated file under its own synthetic module name sidesteps this
    entirely: the canonical ``tengri.observation.observation`` module object
    is never replaced, so every class it defines stays singular for the
    whole process.

    Parameters
    ----------
    mutated_path : Path
        Path to the mutated ``.py`` file (not the real source file).

    Returns
    -------
    module
        The executed, isolated module object.
    """
    import importlib.util
    import sys
    import uuid

    module_name = f"_tengri_2589_mutation_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, mutated_path)
    module = importlib.util.module_from_spec(spec)
    # ``dataclass()`` (used by this file's own ``ForwardState``-adjacent
    # dataclasses) looks itself up via ``sys.modules[cls.__module__]`` while
    # processing fields, so the module must be registered under its own
    # (unique) name before exec, exactly as a normal import would -- then
    # removed again immediately after, so nothing of this throwaway module
    # lingers in ``sys.modules`` once the test is done with it.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_mutation_igm_before_kinematics_would_fail(tmp_path):
    """Reintroducing the pre-#2589 order (IGM folded in before the kinematic
    split) must fail this module's own physical-order check.

    Loads the mutated file as an isolated module (see
    ``_exec_isolated_module``); the real source file is never written to,
    so there is nothing to restore.
    """
    src_path = Path("src/tengri/observation/observation.py")
    original_text = src_path.read_text()

    mutated_text = original_text.replace(
        "sed_stellar_v = broaden_velocity_only(sed_stellar_rest, wave_rest, sigma_v_kms, n_bins)\n"
        "        sed_stellar = sed_stellar_v * igm_trans\n"
        "        sed_instrument_only = sed_instrument_only_rest * igm_trans\n",
        "sed_stellar = (sed_stellar_rest * igm_trans)\n"
        "        sed_stellar = broaden_velocity_only(\n"
        "            sed_stellar, wave_rest, sigma_v_kms, n_bins\n"
        "        )\n"
        "        sed_instrument_only = sed_instrument_only_rest * igm_trans\n",
        1,
    )
    assert mutated_text != original_text, "mutation target string not found -- update this test"

    mutated_path = tmp_path / "observation_mutated.py"
    mutated_path.write_text(mutated_text)
    obs_mod = _exec_isolated_module(mutated_path)

    z, sigma_v, sigma_inst_kms = 6.0, 300.0, 50.0
    wave_rest, wave_obs, sed_rest = _fine_log_grid(z)
    T = jnp.asarray(igm_absorption(wave_obs, z, igm_model="inoue14"))
    sed_v = _numpy_log_gaussian_convolve(np.asarray(sed_rest), np.asarray(wave_rest), sigma_v)
    flux_physical = _numpy_log_gaussian_convolve(
        np.asarray(sed_v) * np.asarray(T), np.asarray(wave_obs), sigma_inst_kms
    )
    fake_state = ForwardState(
        wave=wave_rest,
        sed_intrinsic=sed_rest,
        derived={
            "sed_nebular": jnp.zeros_like(sed_rest),
            "sed_shock": jnp.zeros_like(sed_rest),
        },
    )
    resolution = _resolution_for_sigma_inst(sigma_inst_kms)
    dl_cm = float(np.sqrt((1.0 + z) / (4.0 * np.pi)))
    flux_mutated = np.asarray(
        obs_mod.project_spectrum_kernel_split(
            fake_state,
            sed_rest,
            T,
            wave_rest,
            wave_obs,
            z,
            dl_cm,
            resolution=resolution,
            sigma_lib_kms=0.0,
            sigma_v_kms=sigma_v,
            lsf_scale=1.0,
            n_bins=64,
            conserving=False,
        )
    )
    peak = np.max(flux_physical)
    max_diff = np.max(np.abs(flux_mutated - flux_physical))
    jump_tengri = _edge_steepness(np.asarray(wave_obs), flux_mutated, z)
    jump_physical = _edge_steepness(np.asarray(wave_obs), flux_physical, z)

    failures = []
    try:
        assert max_diff / peak < 1e-3
    except AssertionError:
        failures.append(f"FAILED max_diff/peak = {max_diff / peak:.4f} (expected < 1e-3)")
    try:
        assert jump_physical / jump_tengri == pytest.approx(1.0, rel=0.02)
    except AssertionError:
        failures.append(
            f"FAILED edge steepness ratio = {jump_physical / jump_tengri:.4f} (expected ~1.0)"
        )
    assert failures, "mutation did not reproduce #2589 -- the test is not sensitive to it"
    print("\n".join(failures))


# ── (g) blast radius: simulate.spectrum_from_sfh shares the #2589 order ──


def test_spectrum_from_sfh_matches_physical_order(synthetic_ssp_wide):
    """``simulate.spectrum_from_sfh`` applied the IGM transmission before its
    own ``velocity_broaden`` call (#2589, same defect class as
    ``project_spectrum_kernel_split``): a galaxy's own kinematics cannot
    broaden a line-of-sight absorption feature. Pins the fixed order against
    an independent numpy physical-order reference (its own
    ``_numpy_log_gaussian_convolve``, not ``spectrum_from_sfh``'s
    ``velocity_broaden``) at z=6, sigma_v=300, inoue14 (the only IGM model
    this function wires in).

    Uses the ``synthetic_ssp_wide`` fixture (log-uniform 100 A -- 1 mm,
    smooth continuum): ``velocity_broaden`` requires a grid uniform in
    ln(lambda), and the grid must reach below 1216 A (rest) to resolve the
    Lyman edge at z=6.
    """
    from tengri.analysis.simulate import spectrum_from_sfh
    from tengri.components.igm import igm_transmission
    from tengri.observation.spectrum import compute_spectrum
    from tengri.utils.cosmology import luminosity_distance

    z = 6.0
    sigma_v = 300.0
    t_gyr = jnp.linspace(0.05, 0.9, 50)
    sfr = jnp.ones_like(t_gyr)

    wave_obs = jnp.geomspace(900.0 * (1.0 + z), 1300.0 * (1.0 + z), 1 << 14)

    out = spectrum_from_sfh(
        t_gyr,
        sfr,
        synthetic_ssp_wide,
        wave_obs,
        log_z=-0.3,
        redshift=z,
        sigma_v=sigma_v,
        apply_igm=True,
    )
    flux_tengri = np.asarray(out["flux"])

    sed = out["sed"]
    wave_rest = synthetic_ssp_wide.ssp_wave
    dl_cm = luminosity_distance(z)
    flux_pre_igm = np.asarray(compute_spectrum(sed, wave_rest, wave_obs, z, dl_cm))
    flux_broadened = _numpy_log_gaussian_convolve(flux_pre_igm, np.asarray(wave_obs), sigma_v)
    T = np.asarray(igm_transmission(wave_obs, z))
    flux_physical = flux_broadened * T

    peak = np.max(flux_physical)
    max_diff = np.max(np.abs(flux_tengri - flux_physical))
    assert max_diff / peak < 1e-6, (max_diff / peak,)

    jump_tengri = _edge_steepness(np.asarray(wave_obs), flux_tengri, z)
    jump_physical = _edge_steepness(np.asarray(wave_obs), flux_physical, z)
    assert jump_physical / jump_tengri == pytest.approx(1.0, rel=1e-3), (
        jump_physical / jump_tengri,
    )


def test_mutation_spectrum_from_sfh_igm_before_kinematics_would_fail(synthetic_ssp_wide, tmp_path):
    """Reintroducing IGM-before-kinematics into a throwaway copy of
    ``simulate.py`` must fail the physical-order check above.

    Loads the mutated file as an isolated module (see
    ``_exec_isolated_module``); the real source file is never written to,
    so there is nothing to restore.
    """
    src_path = Path("src/tengri/analysis/simulate.py")
    original_text = src_path.read_text()

    mutated_text = original_text.replace(
        "    # Compute observed spectrum (pre-IGM: stellar kinematics act on the\n"
        "    # galaxy's own light, which the IGM has not yet absorbed).\n"
        "    z_static = static_redshift(redshift)\n"
        "    conserving = resolve_resample_mode(resample, wave_obs, wave, z_static)\n"
        "    resampler = compute_spectrum_conserving if conserving else compute_spectrum\n"
        "    if conserving and sigma_v > 0:\n"
        "        # The pixel integral is taken of the kinematically broadened model (#2530).\n"
        "        from tengri.observation.spectrum import broaden_velocity_only\n"
        "\n"
        "        sed = broaden_velocity_only(sed, wave, sigma_v)\n"
        "    flux = resampler(sed, wave, wave_obs, redshift, dl_cm)\n"
        "\n"
        "    # Velocity broadening -- intrinsic to the source, applied BEFORE the\n"
        "    # line-of-sight IGM transmission (#2589): a galaxy's own kinematics\n"
        "    # cannot physically broaden an absorption feature imprinted after the\n"
        "    # light has left the galaxy. Broadening is a constant-width Gaussian in\n"
        "    # ln(lambda), and redshifting only shifts ln(lambda) by the constant\n"
        "    # ln(1+z), so broadening this observed-frame, pre-IGM ``flux`` is\n"
        "    # equivalent to broadening the rest-frame SED before redshifting.\n"
        "    if sigma_v > 0 and not conserving:\n"
        "        from tengri.observation.spectrum import velocity_broaden\n"
        "\n"
        "        flux = velocity_broaden(flux, wave_obs, sigma_v)\n"
        "\n"
        "    # IGM: multiplies the already-kinematically-broadened observed flux.\n"
        "    if apply_igm and redshift > 0:\n"
        "        from tengri.components.igm import igm_transmission\n"
        "\n"
        "        igm_trans = igm_transmission(wave_obs, redshift)\n"
        "        flux = flux * igm_trans\n",
        # Pre-#2589 order, restored exactly: IGM folded into the rest-frame
        # SED (on the fine model grid) before resampling; kinematics applied
        # after, to the resampled observed-frame flux alone.
        "    # IGM\n"
        "    if apply_igm and redshift > 0:\n"
        "        from tengri.components.igm import igm_transmission\n"
        "\n"
        "        wave_obs_full = wave * (1.0 + redshift)\n"
        "        igm_trans = igm_transmission(wave_obs_full, redshift)\n"
        "        sed = sed * igm_trans\n"
        "\n"
        "    z_static = static_redshift(redshift)\n"
        "    conserving = resolve_resample_mode(resample, wave_obs, wave, z_static)\n"
        "    resampler = compute_spectrum_conserving if conserving else compute_spectrum\n"
        "    flux = resampler(sed, wave, wave_obs, redshift, dl_cm)\n"
        "\n"
        "    # Velocity broadening\n"
        "    if sigma_v > 0:\n"
        "        from tengri.observation.spectrum import velocity_broaden\n"
        "\n"
        "        flux = velocity_broaden(flux, wave_obs, sigma_v)\n",
        1,
    )
    assert mutated_text != original_text, "mutation target string not found -- update this test"

    mutated_path = tmp_path / "simulate_mutated.py"
    mutated_path.write_text(mutated_text)
    sim_mod = _exec_isolated_module(mutated_path)

    from tengri.components.igm import igm_transmission
    from tengri.observation.spectrum import compute_spectrum
    from tengri.utils.cosmology import luminosity_distance

    z, sigma_v = 6.0, 300.0
    t_gyr = jnp.linspace(0.05, 0.9, 50)
    sfr = jnp.ones_like(t_gyr)
    wave_obs = jnp.geomspace(900.0 * (1.0 + z), 1300.0 * (1.0 + z), 1 << 14)

    out = sim_mod.spectrum_from_sfh(
        t_gyr,
        sfr,
        synthetic_ssp_wide,
        wave_obs,
        log_z=-0.3,
        redshift=z,
        sigma_v=sigma_v,
        apply_igm=True,
    )
    flux_mutated = np.asarray(out["flux"])

    sed = out["sed"]
    wave_rest = synthetic_ssp_wide.ssp_wave
    dl_cm = luminosity_distance(z)
    flux_pre_igm = np.asarray(compute_spectrum(sed, wave_rest, wave_obs, z, dl_cm))
    flux_broadened = _numpy_log_gaussian_convolve(flux_pre_igm, np.asarray(wave_obs), sigma_v)
    T = np.asarray(igm_transmission(wave_obs, z))
    flux_physical = flux_broadened * T

    peak = np.max(flux_physical)
    max_diff = np.max(np.abs(flux_mutated - flux_physical))
    jump_mutated = _edge_steepness(np.asarray(wave_obs), flux_mutated, z)
    jump_physical = _edge_steepness(np.asarray(wave_obs), flux_physical, z)

    failures = []
    try:
        assert max_diff / peak < 1e-6
    except AssertionError:
        failures.append(f"FAILED max_diff/peak = {max_diff / peak:.4f} (expected < 1e-6)")
    try:
        assert jump_physical / jump_mutated == pytest.approx(1.0, rel=1e-3)
    except AssertionError:
        failures.append(
            f"FAILED edge steepness ratio = {jump_physical / jump_mutated:.4f} (expected ~1.0)"
        )
    assert failures, "mutation did not reproduce #2589 -- the test is not sensitive to it"
    print("\n".join(failures))
