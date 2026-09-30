# SPDX-License-Identifier: BSD-3-Clause
"""Nebular continuum continues past its tabulated end as optically thin free-free (#2346).

Cue's nebular continuum (src/tengri/components/nebular/cue.py, line 1972) and
CloudyGrid's (src/tengri/components/nebular/cloudy_grid.py, line 1074) interpolate
onto the model grid with ``jnp.interp(ssp_wave, cont_wav, cont_lum, left=0.0,
right=0.0)``, so continuum is exactly zero above the emulator's last node (1e8 Å
for Cue). Reference codes tabulate further: pcigale to 1 m (1e10 Å) with
spectral slope L_nu ∝ nu^-0.096, bagpipes to 3.6 cm. Cue's last decade
(1e7 to 1e8 Å) is a clean power law L_nu ∝ lambda^+0.147 (rms 0.0013 dex).

Fix: continue every tabulated nebular continuum past its last node as optically
thin thermal free-free, L_nu ∝ nu^-0.1 (anchored at the last node; an analytic
extension, NOT an emulator prediction). Extend Cue's declared support to 1 m
so a model without a radio block has nodes there. ``interp_continuum_with_freefree_tail``
in ``src/tengri/components/nebular/_shared.py`` implements this analytic extension.

Tests:
- ``test_cue_model_grid_reaches_one_meter_without_radio``: Cue model, NO radio.
  Model's rest grid max >= 1e10 Å, and every native_wave_nebular("cue") node
  above 1e8 Å is present in it.
- ``test_cue_continuum_continues_as_freefree_past_one_cm``: Cue continuum > 0
  above 1e8 Å; every consecutive log-log slope from 1e8 Å onward equals 0.1
  (free-free); 1e9/1e8 flux ratio == 10^0.1 to relative 1e-6.
- ``test_cloudy_grid_continuum_carries_the_same_tail``: CloudyGrid WITH radio
  block; continuum > 0 above 1e8 Å; every consecutive log-log slope from 1e8 Å
  onward equals 0.1 (free-free) to absolute tolerance 1e-6.
- ``test_tail_is_differentiable``: Cue with neb_logU FREE; jax.grad of
  sed_nebular at 1e9 Å w.r.t. neb_logU is finite.
- ``test_cue_red_edge_sits_in_the_freefree_regime`` (premise guard): uses five
  low-level cases (logU -4, -1; log Z_gas -1.5, +0.4; plus unmodified base);
  red-edge slope 1e7-1e8 Å must be 0.05 <= slope <= 0.25 for all variants.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri
from tengri._data_setup import find_data_str

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def cue_backend(ssp_data_fsps):
    """Build a CueBackend with the git-tracked bare-stellar FSPS PRSC SSP."""
    from tengri.components.nebular.cue import CueBackend

    weights_path = find_data_str("cue_weights.npz")
    if weights_path is None:
        pytest.skip("cue_weights.npz not found")
    return CueBackend(weights_path, ssp_data_fsps)


def _build_cue_model(ssp_data):
    """A minimal Cue-only model with no radio block."""
    return tengri.SEDModel.build(
        ssp_data=ssp_data,
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={"type": "two_component", "law": "calzetti"},
        dust_emission={"type": "draine_li2014"},
        neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
        redshift=tengri.Fixed(0.0),
    )


def _build_cue_model_free_logu(ssp_data):
    """A Cue model with neb_logU FREE."""
    return tengri.SEDModel.build(
        ssp_data=ssp_data,
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={"type": "two_component", "law": "calzetti"},
        dust_emission={"type": "draine_li2014"},
        neb={
            "type": "cue",
            "logU": tengri.FREE,
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        redshift=tengri.Fixed(0.0),
    )


def test_cue_model_grid_reaches_one_meter_without_radio(ssp_data_fsps):
    """Cue model without radio has max wavelength >= 1 m (1e10 Å).

    Before the fix: max = 1e8 Å (Cue's last node).
    After the fix: max >= 1e10 Å (extended by native_wave_nebular).
    """
    model = _build_cue_model(ssp_data_fsps)

    # Find the actual rest wavelength attribute
    wave_rest = np.asarray(model._rest_wavelength)
    wave_max = float(wave_rest.max())

    # Expected: at least 1e10 Å (1 m), allowing 1e-9 relative tolerance
    assert wave_max >= 1e10 * (1.0 - 1e-9), (
        f"Model rest wavelength max {wave_max:.3e} Å is below 1e10 Å. "
        "Cue's native grid extension to 1 m (#2346) is not working."
    )

    # Every native_wave_nebular("cue") node above 1e8 Å must be in the model grid
    from tengri.forward.wavelength_extension import native_wave_nebular

    native_cue = native_wave_nebular("cue")
    assert native_cue is not None, "native_wave_nebular('cue') must be discoverable"
    native_cue = np.asarray(native_cue)

    above_1cm = native_cue[native_cue > 1e8]
    assert above_1cm.size > 0, "native_wave_nebular('cue') should have nodes above 1e8 Å"

    # Check that all these native nodes appear in the model grid
    for node in above_1cm:
        found = np.any(np.isclose(wave_rest, node, rtol=1e-9))
        assert found, (
            f"Native Cue node at {node:.3e} Å (above 1e8 Å) is missing from "
            f"model grid. All native nodes must be present (#2346)."
        )


def test_cue_continuum_continues_as_freefree_past_one_cm(ssp_data_fsps):
    """Cue's nebular continuum is positive above 1e8 Å and follows free-free slope.

    Before the fix: continuum = 0 everywhere above 1e8 Å (clipped by right=0.0).
    After the fix: continuum > 0, slope d(ln L)/d(ln nu) ≈ -0.1 (free-free).
    """
    model = _build_cue_model(ssp_data_fsps)

    # Get default parameters and predict state
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(params)

    wave = np.asarray(state.wave)
    sed_nebular = np.asarray(state.derived["sed_nebular"])

    # Find indices for 1e8 Å and 1e9 Å
    above_1cm = wave > 1e8
    assert np.any(above_1cm), "Model should have wavelengths above 1e8 Å"

    # Check all values above 1e8 Å are positive
    assert np.all(sed_nebular[above_1cm] > 0.0), (
        "Nebular continuum must be positive above 1e8 Å (optically thin "
        "free-free tail); found zeros or negative values (#2346)."
    )

    # Find nodes at 1e8 and 1e9 Å (or nearest)
    idx_1e8 = np.argmin(np.abs(wave - 1e8))
    idx_1e9 = np.argmin(np.abs(wave - 1e9))
    assert idx_1e8 < idx_1e9, "1e8 Å should be at a lower index than 1e9 Å"

    # Flux ratio should be 10^0.1 (since L_nu ∝ nu^-0.1 and nu ∝ 1/lambda)
    # nu_1e8 / nu_1e9 = (1e9 / 1e8) = 10, so L_1e8 / L_1e9 = 10^0.1 ≈ 1.2589
    # => L_1e9 / L_1e8 = 10^(-0.1) ≈ 0.7943  NO WAIT:
    # nu ∝ 1/lambda, so lambda_1e8 < lambda_1e9 => nu_1e8 > nu_1e9
    # nu_1e8 / nu_1e9 = (1e9 / 1e8) = 10
    # L_nu ∝ nu^-0.1, so L_1e8 = L_1e9 × (10^-0.1) = L_1e9 × 10^-0.1
    # => L_1e9 / L_1e8 = 10^0.1
    expected_ratio = 10.0**0.1  # ≈ 1.2589254...
    flux_ratio = sed_nebular[idx_1e9] / sed_nebular[idx_1e8]
    rel_error = abs(flux_ratio - expected_ratio) / expected_ratio
    assert flux_ratio == pytest.approx(expected_ratio, rel=1e-6), (
        f"Flux ratio sed[1e9]/sed[1e8] = {flux_ratio:.6f}, expected {expected_ratio:.6f} "
        f"(free-free L_nu ∝ nu^-0.1). Relative error {rel_error:.3e} exceeds 1e-6 (#2346)."
    )

    # Check continuity: log-log slope between 1e8 and first node above
    # should be 0.1 (free-free with L_nu ∝ nu^-0.1 ⟹ L_λ ∝ λ^0.1)
    above_indices = np.where(wave > 1e8)[0]
    above_and_edge = np.concatenate([[idx_1e8], above_indices])
    for i in range(len(above_and_edge) - 1):
        idx_lo = above_and_edge[i]
        idx_hi = above_and_edge[i + 1]
        wave_lo = wave[idx_lo]
        wave_hi = wave[idx_hi]
        sed_lo = sed_nebular[idx_lo]
        sed_hi = sed_nebular[idx_hi]
        slope = np.log(sed_hi / sed_lo) / np.log(wave_hi / wave_lo)
        assert abs(slope - 0.1) < 1e-6, (
            f"Slope between {wave_lo:.3e} and {wave_hi:.3e} Å is {slope:.6f}, "
            f"expected 0.1 (free-free). Continuity check failed (#2346)."
        )


@pytest.mark.skipif(
    find_data_str("cloudy_grid_prsc.h5") is None,
    reason=(
        "cloudy_grid_prsc.h5 is not shipped in git; set TENGRI_DATA_DIR to a checkout that has it"
    ),
)
def test_cloudy_grid_continuum_carries_the_same_tail(ssp_data_fsps):
    """CloudyGrid (with radio block) continuum follows free-free above 1e8 Å.

    CloudyGrid declares no native wavelength grid, so nodes above 1e8 Å come
    from the radio wing. The tail should still follow L_nu ∝ nu^-0.1.

    The PRSC ``cloudy_grid_prsc.h5`` grid is paired with the PRSC-matched
    FSPS SSP (``ssp_data_fsps``); the BC03 SSP with this grid yields an
    all-NaN ``sed_nebular`` on main (pre-existing, tracked separately from
    #2346).
    """

    # CloudyGrid only makes sense with a radio block (it declares no native grid)
    # Now build a CloudyGrid model with radio
    model = tengri.SEDModel.build(
        ssp_data=ssp_data_fsps,
        sfh={"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={"type": "two_component", "law": "calzetti"},
        dust_emission={"type": "draine_li2014"},
        neb={"type": "cloudy", "all_params": tengri.Fixed(tengri.DEFAULT)},
        radio={
            "sf": {"type": "bell2003"},
            "agn": {"type": "powerlaw"},
            "all_params": tengri.Fixed(tengri.DEFAULT),
        },
        redshift=tengri.Fixed(0.0),
    )

    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(params)

    wave = np.asarray(state.wave)
    sed_nebular = np.asarray(state.derived["sed_nebular"])

    above_1e8 = wave > 1e8
    assert np.any(above_1e8), "Model should have wavelengths above 1e8 Å"

    assert np.isfinite(sed_nebular).all(), (
        "CloudyGrid nebular continuum must be finite everywhere; found NaN/inf (#2346)."
    )

    assert (sed_nebular[above_1e8] > 0.0).all(), (
        "CloudyGrid nebular continuum must be positive above 1e8 Å. "
        "The radio block should provide wavelength nodes, and the tail "
        "should continue the continuum (#2346)."
    )

    # Local log-log slope between every pair of nodes at/above 1e8 Å must be
    # 0.1 (free-free, L_nu ∝ nu^-0.1 ⟹ L_λ ∝ λ^0.1).
    at_and_above_1e8 = wave >= 1e8
    wave_tail = wave[at_and_above_1e8]
    sed_tail = sed_nebular[at_and_above_1e8]
    slopes = np.diff(np.log(sed_tail)) / np.diff(np.log(wave_tail))
    assert slopes == pytest.approx(0.1, abs=1e-6), (
        f"CloudyGrid tail slopes are {slopes}, expected 0.1 everywhere (free-free) (#2346)."
    )


def test_tail_is_differentiable(ssp_data_fsps):
    """Gradient of sed_nebular w.r.t. neb_logU is finite and non-zero at 1e9 Å.

    The helper function interp_continuum_with_freefree_tail must be
    gradient-safe: jax.grad should work through it.
    """
    model = _build_cue_model_free_logu(ssp_data_fsps)

    # Get a reference parameter sample and compute the index outside the loss
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state_ref = model.predict_state(params)
    wave = np.asarray(state_ref.wave)
    idx_1e9 = np.argmin(np.abs(wave - 1e9))

    # Define a loss function: sed_nebular at the 1e9 Å node
    def loss(p):
        state = model.predict_state(p)
        return state.derived["sed_nebular"][idx_1e9]

    # Take gradient w.r.t. all parameters
    grad_fn = jax.grad(loss)
    grads = grad_fn(params)

    # Find the free parameter name (neb_logU from the model's grammar)
    free_param_name = "neb_logU"
    assert free_param_name in params, f"expected neb_logU among {sorted(params)}"
    assert free_param_name in grads, (
        f"Parameter {free_param_name} not in gradients. "
        "Check that it is marked as FREE in the model."
    )

    grad_val = float(np.asarray(grads[free_param_name]))
    assert np.isfinite(grad_val), (
        f"Gradient of sed_nebular w.r.t. {free_param_name} is {grad_val} (not finite). "
        "interp_continuum_with_freefree_tail must be gradient-safe (#2346)."
    )
    assert grad_val != 0.0, (
        f"Gradient of sed_nebular w.r.t. {free_param_name} at 1e9 Å is exactly zero. "
        "neb_logU sets the continuum amplitude at the emulator's last node, which the "
        "tail is anchored to, so a zero gradient would mean the tail was detached from "
        "the emulator output (#2346)."
    )


def test_cue_red_edge_sits_in_the_freefree_regime(cue_backend):
    """Premise guard: Cue red edge slopes 0.05-0.25 via low-level gas_logu/gas_logz.

    Tests logU -4 and -1, log Z_gas -1.5 and +0.4 via the low-level
    gas_logu/gas_logz keys. This premise guard asserts that the tabulated
    Cue continuum sits in the free-free regime (slope ~0.147); it should PASS
    both before and after the fix and guards against future Cue changes.
    """
    # Low-level test parameters (from test_nebular_continuum.py)
    base_params = {
        "ionspec_index1": 0.5,
        "ionspec_index2": -1.5,
        "ionspec_index3": -2.5,
        "ionspec_index4": -3.0,
        "ionspec_logLratio1": 0.0,
        "ionspec_logLratio2": 0.0,
        "ionspec_logLratio3": 0.0,
        "gas_logu": 0.0,
        "gas_logn": 2.5,
        "gas_logz": 0.0,
        "gas_logco": -0.3,
        "gas_logno": -1.0,
    }

    # Test variants: gas_logu -4 and -1, gas_logz -1.5 and +0.4, plus unmodified base
    test_variants = [
        {"gas_logu": -4.0, "gas_logz": -1.5},
        {"gas_logu": -4.0, "gas_logz": 0.4},
        {"gas_logu": -1.0, "gas_logz": -1.5},
        {"gas_logu": -1.0, "gas_logz": 0.4},
        {},  # unmodified base
    ]

    slopes = []
    for variant in test_variants:
        params = {**base_params, **variant}
        cont_wav, cont_lum = cue_backend.predict_nebular_continuum(**params)

        cont_wav = np.asarray(cont_wav)
        cont_lum = np.asarray(cont_lum)

        # Select nodes in 1e7-1e8 Å range (the red edge)
        mask = (cont_wav >= 1e7) & (cont_wav <= 1e8)
        assert np.any(mask), "no Cue nodes between 1e7 and 1e8 Å"

        wav_sub = cont_wav[mask]
        lum_sub = cont_lum[mask]

        # Fit log-log slope: d(ln L) / d(ln w)
        log_lum = np.log10(lum_sub)
        log_wav = np.log10(wav_sub)
        slope = float(np.polyfit(log_wav, log_lum, 1)[0])
        slopes.append(slope)

        # Free-free is L ∝ lambda^0.147, so slope in log-log should be ~+0.147
        # Allow 0.05 to 0.25 margin for data quality and fitting
        assert 0.05 <= slope <= 0.25, (
            f"Cue red edge slope at gas_logu={variant.get('gas_logu', 0.0)}, "
            f"gas_logz={variant.get('gas_logz', 0.0)} is {slope:.4f}, outside [0.05, 0.25]. "
            "This premise guard expects Cue's continuum to sit in the free-free regime."
        )

    # Ensure the variants actually produce distinct continua (not all inert)
    assert len(set(slopes)) > 1, (
        f"All five variants yielded identical slopes {set(slopes)}, "
        "the test has become inert and does not verify distinct parameter effects."
    )
