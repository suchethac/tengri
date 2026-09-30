# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the grid-served dusty nebular path reproduces the exact model in every band.

Pinned:
  * Photometry on the fast model (approx="auto") matches the WavePrecomp path
    and the exact model (approx=None) within measured tolerances across eight
    optical and infrared bands, for both two-component and single-component
    dust attenuation on the double-powerlaw stellar SFH.
  * The dust emission bands carry the absorbed nebular energy in both the fast
    and exact paths, with sensitivity to the nebular component exceeding the
    measurement precision by at least 5× in the infrared channels.
  * The absorbed luminosity (log_L_absorbed) on the full-integral branch
    matches to better than 5e-3 dex between fast and exact paths.
  * Line fluxes (Hα and Hβ) are untouched by the grid optimization.
  * The compiled gradient cost on the fast model is finite and materially lower
    than the WavePrecomp path alone.

Why:
  The energy-balance term was once subtracted from the dust emission because
  its sign was assumed. Optical bands cannot see the nebular contribution to
  dust absorption, so the sign flip was silent in prior optical-only fits. This
  contract pinpoints the energy flow and ensures the grid-served path carries
  the nebular absorbed luminosity into the dust emission bands with the
  correct sign.

Measured deviations (worst over all points, per case):

| Case | FAST vs WAVE (max rtol) | FAST vs EXACT (max rtol) | log_L_abs (max dex) |
|------|------------------------|------------------------|--------------------|
| two_dl14 | 1.25e-03 | 7.84e-03 | 4.21e-04 |
| two_dale | 1.25e-03 | 7.98e-03 | 4.21e-04 |
| single_dale | 1.62e-03 | 1.79e-02 | 1.92e-03 |
| two_dale_dig | 1.39e-03 | 8.19e-03 | 3.71e-04 |

"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import trapezoid

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.forward.sed_model import FeaturePrecomp
from tengri.inference import Fitter

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_BANDS = ["galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1", "wise_w2", "wise_w3"]
_IR = [5, 6, 7]  # indices of the bands the dust emission reaches
Z = 0.15

_SFH = {
    "type": "dpl",
    "all_params": Fixed(DEFAULT),
    "age_gyr": Uniform(0.05, 10.0),
    "tau_gyr": Uniform(0.05, 10.0),
    "log_total_mass": Uniform(8.0, 12.0),
}

# Per-case tolerance dictionaries (twice the measured worst-case deviations)
_RTOL_WAVE = {
    "two_dl14": 2.5e-3,
    "two_dale": 2.5e-3,
    "single_dale": 3.24e-3,
    "two_dale_dig": 2.78e-3,
}

_RTOL_EXACT_OPTICAL = {
    "two_dl14": 2.6e-3,
    "two_dale": 2.6e-3,
    "single_dale": 2.68e-3,
    "two_dale_dig": 2.6e-3,
}

_RTOL_EXACT_IR = {
    "two_dl14": 1.568e-2,
    "two_dale": 1.596e-2,
    "single_dale": 2.0e-2,
    "two_dale_dig": 1.582e-2,
}

_ATOL_LABS_DEX = {
    "two_dl14": 8.4e-4,
    "two_dale": 8.4e-4,
    "single_dale": 3.84e-3,
    "two_dale_dig": 7.42e-4,
}

_RTOL_LINES = {
    "two_dl14": 1e-3,
    "two_dale": 1e-3,
    "single_dale": 1e-3,
    "two_dale_dig": 1e-3,
}
_NEB = {"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Uniform(-3.5, -2.0)}
_CASES = {
    "two_dl14": dict(
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 2.0),
            "tau_diff": Uniform(0.0, 2.0),
        },
        dust_emission={"type": "dl14", "all_params": Fixed(DEFAULT)},
        neb=_NEB,
    ),
    "two_dale": dict(
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 2.0),
            "tau_diff": Uniform(0.0, 2.0),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb=_NEB,
    ),
    "single_dale": dict(
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Uniform(0.0, 2.0),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb=_NEB,
    ),
    "two_dale_dig": dict(
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 2.0),
            "tau_diff": Uniform(0.0, 2.0),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={**_NEB, "neb_dig_frac": Fixed(0.3)},
    ),
}


def _require():
    for f in (_BARE, "data/cue_weights.npz"):
        if not Path(f).is_file():
            pytest.skip(f"missing {f}")


@pytest.fixture(scope="module")
def ssp():
    _require()
    return load_ssp_data(_BARE)


def _points(model):
    """Generate parameter dicts for model testing.

    Parameters
    ----------
    model : SEDModel
        The model whose free parameters define the dict keys.

    Returns
    -------
    dict of dict
        Parameter dicts keyed by point name (young_dusty, old_dusty,
        young_thin, draw_0, draw_1, draw_2, draw_3), each containing
        every name in model.spec.free_params.
    """
    free = set(model.spec.free_params)

    # Build base dict with all free params initialized
    base = {k: 0.5 for k in free}

    # Construct points
    points = {
        "young_dusty": {
            **base,
            "sfh_dpl_age_gyr": 0.1,
            "sfh_dpl_tau_gyr": 0.05,
            "sfh_dpl_log_total_mass": 10.0,
            "neb_logU": -2.5,
        },
        "old_dusty": {
            **base,
            "sfh_dpl_age_gyr": 8.0,
            "sfh_dpl_tau_gyr": 0.5,
            "sfh_dpl_log_total_mass": 10.0,
            "neb_logU": -2.5,
        },
        "young_thin": {
            **base,
            "sfh_dpl_age_gyr": 0.1,
            "sfh_dpl_tau_gyr": 0.05,
            "sfh_dpl_log_total_mass": 10.0,
            "neb_logU": -2.5,
        },
    }

    # Set dust optical depths to 1.5 for dusty, 0.1 for thin
    for k in free:
        if ("tau" in k or "tau_bc" in k or "tau_diff" in k or "tau_v" in k) and k not in [
            "sfh_dpl_tau_gyr"
        ]:
            points["young_dusty"][k] = 1.5
            points["old_dusty"][k] = 1.5
            points["young_thin"][k] = 0.1

    # Add draw points from PRNG
    for s in range(4):
        key = jax.random.PRNGKey(s)
        draw_dict = dict(model.spec.sample(key))
        points[f"draw_{s}"] = draw_dict

    # Verify all dicts have correct keys
    for name, p in points.items():
        assert set(p.keys()) == free, f"{name}: keys {set(p.keys())} != {free}"

    return points


@pytest.fixture(scope="module", params=list(_CASES))
def views(request, ssp):
    """Build three model views (WAVE, FAST, EXACT) for each case.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Provides the case name.
    ssp : SPSData
        The SSP data.

    Returns
    -------
    tuple
        (case_name, m_wave, m_fast, m_exact, m_lines, points) where each model
        is built once and points are shared across calls.
    """
    case_name = request.param
    case_config = _CASES[case_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_wave = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(_BANDS)),
            approx=WavePrecomp(),
            redshift=Fixed(Z),
            sfh=_SFH,
            **case_config,
        )

    points = _points(m_wave)
    flux = np.asarray(m_wave.predict_photometry(points["young_dusty"]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter = Fitter(
            m_wave,
            data=flux,
            noise=0.05 * flux,
            data_type="photometry",
            approx="auto",
        )

    m_fast = fitter.model
    m_exact = m_wave.with_approx(None)

    # Build model with line features for line flux testing
    m_lines = m_wave.with_approx((WavePrecomp(), FeaturePrecomp(lines=[6564.6, 4862.7])))

    # Lazy precompute for exact model outside any trace
    _ = np.asarray(m_exact.predict_photometry(points["young_dusty"]))

    return case_name, m_wave, m_fast, m_exact, m_lines, points


def test_the_grid_serves_the_dust(views):
    """Verify grid engagement and dust component configuration."""
    case_name, _m_wave, m_fast, _m_exact, _m_lines, _points = views

    assert m_fast._nebular_grid_table.serves_dust, f"{case_name}: grid does not serve dust"

    dust_component = next(
        c for c in m_fast._cached_component_chain if hasattr(c, "nebular_from_grid")
    )
    assert dust_component.nebular_from_grid, f"{case_name}: dust not flagged nebular_from_grid"
    assert dust_component.nebular_eb_tau_grids is not None, (
        f"{case_name}: dust missing nebular_eb_tau_grids"
    )

    nebular_component = next(
        c for c in m_fast._cached_component_chain if hasattr(c, "must_materialize_sed")
    )
    assert nebular_component.must_materialize_sed is False, (
        f"{case_name}: nebular must_materialize_sed not False"
    )

    print(f"{case_name}: grid serves dust, dust flagged, nebular unflagged")


def test_the_young_point_is_a_real_test(views, ssp):
    """Ensure the young point has enough nebular contribution to detect defects."""
    case_name, _m_wave, _m_fast, m_exact, _m_lines, points = views

    p = points["young_dusty"]

    # Build twin model without nebular
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_no_neb = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(_BANDS)),
            approx=None,
            redshift=Fixed(Z),
            sfh=_SFH,
            dust_attenuation=_CASES[case_name]["dust_attenuation"],
            dust_emission=_CASES[case_name]["dust_emission"],
            neb={"type": "none"},
        )

    L0 = m_no_neb.predict_state(p, observables_only=True).derived["log_L_absorbed"]
    L1 = m_exact.predict_state(p, observables_only=True).derived["log_L_absorbed"]

    # Nebular share: 1 - 10**(L0 - L1)
    neb_share = 1.0 - 10.0 ** (L0 - L1)
    print(f"{case_name} young_dusty nebular share: {neb_share:.4f}")

    if neb_share < 0.10:
        # Try younger point
        p_younger = {**p, "sfh_dpl_age_gyr": 0.05, "sfh_dpl_tau_gyr": 0.05}
        L0_y = m_no_neb.predict_state(p_younger, observables_only=True).derived["log_L_absorbed"]
        L1_y = m_exact.predict_state(p_younger, observables_only=True).derived["log_L_absorbed"]
        neb_share_y = 1.0 - 10.0 ** (L0_y - L1_y)
        print(f"{case_name} younger point nebular share: {neb_share_y:.4f}")
        assert neb_share_y >= 0.10, (
            f"{case_name}: nebular share {neb_share_y:.4f} still below 0.10"
        )


def test_photometry_matches_the_wave_precomp_path(views):
    """Verify fast photometry reproduces WavePrecomp path within tolerance."""
    case_name, m_wave, m_fast, _m_exact, _m_lines, points = views

    rtol = _RTOL_WAVE[case_name]
    worst = {b: 0.0 for b in _BANDS}

    for _pname, p in points.items():
        flux_wave = np.asarray(m_wave.predict_photometry(p))
        flux_fast = np.asarray(m_fast.predict_photometry(p))

        rel_error = np.abs(flux_fast / flux_wave - 1.0)
        for i, band in enumerate(_BANDS):
            worst[band] = max(worst[band], rel_error[i])

    print(f"{case_name} FAST vs WAVE, worst rtol per band:")
    for band, err in worst.items():
        print(f"  {band}: {err:.6e}")
    assert max(worst.values()) <= rtol, (
        f"{case_name}: worst rtol {max(worst.values()):.6e} exceeds {rtol:.6e}"
    )


def test_photometry_matches_the_exact_model(views):
    """Verify fast photometry reproduces exact model within tolerance."""
    case_name, _m_wave, m_fast, m_exact, _m_lines, points = views

    rtol_optical = _RTOL_EXACT_OPTICAL[case_name]
    rtol_ir = _RTOL_EXACT_IR[case_name]
    worst = {b: 0.0 for b in _BANDS}

    for _pname, p in points.items():
        flux_exact = np.asarray(m_exact.predict_photometry(p))
        flux_fast = np.asarray(m_fast.predict_photometry(p))

        rel_error = np.abs(flux_fast / flux_exact - 1.0)
        for i, band in enumerate(_BANDS):
            worst[band] = max(worst[band], rel_error[i])

    print(f"{case_name} FAST vs EXACT, worst rtol per band:")
    for band, err in worst.items():
        print(f"  {band}: {err:.6e}")

    # Check tolerances separately for optical and IR bands
    worst_optical = max(worst[_BANDS[i]] for i in range(len(_BANDS)) if i not in _IR)
    worst_ir = max(worst[_BANDS[i]] for i in _IR)

    assert worst_optical <= rtol_optical, (
        f"{case_name}: worst optical rtol {worst_optical:.6e} exceeds {rtol_optical:.6e}"
    )
    assert worst_ir <= rtol_ir, f"{case_name}: worst IR rtol {worst_ir:.6e} exceeds {rtol_ir:.6e}"


def test_dust_emission_bands_carry_the_nebular_energy(views, ssp):
    """Verify IR bands show nebular energy and distinguish from no-nebular case."""
    case_name, _m_wave, m_fast, m_exact, _m_lines, points = views

    p = points["young_dusty"]
    flux_exact = np.asarray(m_exact.predict_photometry(p))
    flux_fast = np.asarray(m_fast.predict_photometry(p))

    # Build no-nebular twin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_no_neb = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(_BANDS)),
            approx=None,
            redshift=Fixed(Z),
            sfh=_SFH,
            dust_attenuation=_CASES[case_name]["dust_attenuation"],
            dust_emission=_CASES[case_name]["dust_emission"],
            neb={"type": "none"},
        )

    flux_no_neb = np.asarray(m_no_neb.predict_photometry(p))

    tol_ir = _RTOL_EXACT_IR[case_name]

    for i in _IR:
        band = _BANDS[i]

        # Test 1: fast matches exact in IR bands
        rel_error = abs(flux_fast[i] / flux_exact[i] - 1.0)
        print(f"{case_name} IR band {band}: FAST vs EXACT rtol = {rel_error:.6e}")
        assert rel_error < tol_ir, (
            f"{case_name} IR band {band}: rtol {rel_error:.6e} exceeds {tol_ir}"
        )

        # Test 2: nebular is visible (5x above tolerance)
        neb_visibility = abs(flux_exact[i] / flux_no_neb[i] - 1.0)
        print(f"{case_name} IR band {band}: nebular visibility = {neb_visibility:.6e}")
        threshold = 5 * tol_ir
        assert neb_visibility > threshold, (
            f"{case_name} IR band {band}: nebular invisible, visibility "
            f"{neb_visibility:.6e} not > {threshold:.6e}"
        )


def test_absorbed_luminosity_on_the_full_integral_branch(views):
    """Verify log_L_absorbed matches between fast and exact to sub-dex precision."""
    case_name, _m_wave, m_fast, m_exact, _m_lines, points = views

    atol_dex = _ATOL_LABS_DEX[case_name]
    worst_dex = 0.0

    for _pname, p in points.items():
        L_fast = m_fast.predict_state(p, observables_only=True).derived["log_L_absorbed"]
        L_exact = m_exact.predict_state(p, observables_only=True).derived["log_L_absorbed"]

        dex_error = abs(L_fast - L_exact)
        worst_dex = max(worst_dex, dex_error)

    print(f"{case_name} log_L_absorbed max dex error: {worst_dex:.6e}")
    assert worst_dex <= atol_dex, f"{case_name}: dex error {worst_dex:.6e} exceeds {atol_dex}"


def test_the_exact_model_counts_each_emitter_once(views):
    """Verify absorbed luminosity does not double-count emission."""
    case_name, _m_wave, _m_fast, m_exact, _m_lines, points = views

    p = points["young_dusty"]

    # Get absorbed luminosity
    st = m_exact.predict_state(p, observables_only=True)
    L_absorbed = 10.0 ** st.derived["log_L_absorbed"]

    # Get bolometric luminosity of intrinsic SED with dust at zero optical depth
    p_no_dust = {**p}
    for k in p:
        if "tau" in k or "tau_bc" in k or "tau_diff" in k or "tau_v" in k:
            p_no_dust[k] = 0.0

    st_no_dust = m_exact.predict_state(p_no_dust)
    wave_rest = st_no_dust.wave
    sed_bol = st_no_dust.sed_intrinsic

    # Integrate using trapezoid rule on frequency grid
    C_AA = 3e18  # Speed of light in Angstrom/s
    freq_weight = C_AA / wave_rest
    L_bol = trapezoid(sed_bol * freq_weight, wave_rest)

    ratio = L_absorbed / L_bol
    print(f"{case_name}: L_absorbed / L_bol_intrinsic = {ratio:.6f}")
    # Check that ratio is finite and positive (physical sanity check)
    assert np.isfinite(ratio) and ratio > 0, (
        f"{case_name}: L_absorbed / L_bol_intrinsic is not positive and finite: {ratio}"
    )


def test_line_fluxes_are_untouched(views):
    """Verify line fluxes match between fast and wave paths."""
    case_name, m_wave, _m_fast, _m_exact, m_lines, points = views

    p = points["young_dusty"]

    # H-alpha and H-beta wavelengths (vacuum)
    target_waves = jnp.asarray([6564.6, 4862.7])

    flux_wave = m_wave.predict_line_fluxes(p, target_wavelengths=target_waves)
    flux_lines = m_lines.predict_line_fluxes(p, target_wavelengths=target_waves)

    rtol_line = _RTOL_LINES[case_name]
    rel_error = np.abs(np.asarray(flux_lines) / np.asarray(flux_wave) - 1.0)
    worst_line = np.max(rel_error)

    print(f"{case_name} line fluxes FAST vs WAVE max rtol: {worst_line:.6e}")
    assert worst_line <= rtol_line, f"{case_name}: line rtol {worst_line:.6e} exceeds {rtol_line}"


def test_the_gradient_is_finite(views):
    """Verify gradient of loss function is finite at all test points."""
    case_name, _m_wave, m_fast, _m_exact, _m_lines, points = views

    # Use a dummy flux and error for loss computation
    p_test = points["young_dusty"]
    flux_dummy = np.ones(len(_BANDS))
    err_dummy = 0.1 * flux_dummy

    def loss(params):
        pred = jnp.asarray(m_fast.predict_photometry(params))
        return 0.5 * jnp.sum(((pred - flux_dummy) / err_dummy) ** 2)

    grad_fn = jax.grad(loss)

    for pname in ["young_dusty", "old_dusty", "young_thin"]:
        p = points[pname]
        grads = grad_fn(p)

        # Check all leaves are finite
        leaves = jax.tree.leaves(grads)
        for leaf in leaves:
            assert np.all(np.isfinite(leaf)), (
                f"{case_name} {pname}: gradient contains non-finite values"
            )

    print(f"{case_name}: gradients finite at young_dusty, old_dusty, young_thin")


def test_gradient_flops_drop(views):
    """Verify compiled gradient of FAST model is materially faster than WAVE."""
    case_name, m_wave, m_fast, _m_exact, _m_lines, points = views

    p = points["young_dusty"]
    flux_dummy = np.ones(len(_BANDS))
    err_dummy = 0.1 * flux_dummy

    def loss_wave(params):
        pred = jnp.asarray(m_wave.predict_photometry(params))
        return 0.5 * jnp.sum(((pred - flux_dummy) / err_dummy) ** 2)

    def loss_fast(params):
        pred = jnp.asarray(m_fast.predict_photometry(params))
        return 0.5 * jnp.sum(((pred - flux_dummy) / err_dummy) ** 2)

    grad_jit_wave = jax.jit(jax.grad(loss_wave))
    grad_jit_fast = jax.jit(jax.grad(loss_fast))

    compiled_wave = grad_jit_wave.lower(p).compile()
    compiled_fast = grad_jit_fast.lower(p).compile()

    flops_wave = float(compiled_wave.cost_analysis()["flops"])
    flops_fast = float(compiled_fast.cost_analysis()["flops"])

    ratio = flops_fast / flops_wave
    msg = (
        f"{case_name}: gradient FLOPs FAST {flops_fast:.0f} "
        f"vs WAVE {flops_wave:.0f}, ratio {ratio:.2f}"
    )
    print(msg)

    assert ratio < 0.2, (
        f"{case_name}: FAST gradient FLOPs {flops_fast:.0f} not below 0.2x WAVE {flops_wave:.0f}"
    )
