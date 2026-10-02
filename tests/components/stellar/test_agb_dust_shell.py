# SPDX-License-Identifier: BSD-3-Clause
"""AGB circumstellar dust-shell weighting (#2534).

``agb_dust={'type': 'fsps_shell', 'weight': ...}`` rescales the Villaume,
Conroy & Johnson (2015) circumstellar AGB dust-shell reprocessing FSPS bakes
into every ``fsps_mist_*`` SSP grid at its own default weight (``agb_dust=
1.0``). Covers: identity at the default weight; the template's pinned numbers
and recorded provenance; exact identity outside the stored window; the seam
(baked Fixed weight == live free weight == SSP cube times R, and a hand
computation at a weight between stored nodes); every refusal for a free
weight (precompute tables, window LUTs, non-MIST grids); the batch-fit
resolver; the nebular ionizing-rate tables; and the nested-dict grammar.
"""

from __future__ import annotations

import dataclasses
import os
import warnings

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, Fixed, SEDModel, SpectrumPrecomp, Uniform, WavePrecomp, load_ssp
from tengri.components.stellar.agb_dust_shell import (
    SUPPORTED_GRIDS,
    ResampledAGBDustShell,
    agb_dust_ratio,
    bake_agb_dust_shell,
    load_agb_dust_shell_template,
    resample_agb_dust_shell,
)
from tengri.components.stellar.component import StellarSEDComponent
from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S
from tengri.inference.fitter import Fitter, _resolve_batch_fit_approx
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectral_indices import STANDARD_INDICES

pytestmark = pytest.mark.bounds

_MIST_C3K = "fsps_mist_c3k_a_chabrier"
_MIST_MILES = "fsps_mist_miles_chabrier"
_NON_MIST = "fsps_prsc_miles_chabrier"
_FREE_WEIGHT = {"type": "fsps_shell", "weight": Uniform(0.0, 3.0)}

#: FSPS 0.4.7 / libfsps v3.2-61-g82a8735 output (MIST + MILES + Chabrier) at the
#: node nearest solar metallicity (log10 Z = -1.8477) and 1 Gyr.
_PIN_MEDIAN_INV_R0 = 1.0470  # median of 1/R(w=0) over 2-10 um
_PIN_MAX_INV_R0 = 2.3621  # max of 1/R(w=0) over 2-10 um
_PIN_R3_AT_10UM = 0.8546  # R(w=3) at 10 um


def _template_path():
    from tengri import data_path

    return data_path("agb_dust_shell_ratios_mist.h5")


def _observation() -> Observation:
    """Synthetic top-hat filters from the optical to 10 um."""

    def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    curves = tuple(_tophat(c) for c in (4800.0, 8000.0, 22000.0, 50000.0, 100000.0))
    return Observation(photometry=Photometry(filters=curves))


def _build(ssp_name: str, *, agb_dust=None, age_kernel: str = "cic", approx=None, **extra):
    return SEDModel.build(
        ssp_data=load_ssp(ssp_name),
        observation=_observation(),
        sfh={"type": "const", "age_kernel": age_kernel},
        redshift=Fixed(0.0),
        agb_dust=agb_dust,
        approx=approx,
        **extra,
    )


def _resampled(ssp) -> ResampledAGBDustShell:
    return resample_agb_dust_shell(
        load_agb_dust_shell_template(),
        np.asarray(ssp.ssp_lgmet),
        np.asarray(ssp.ssp_lg_age_gyr) + 9.0,
        np.asarray(ssp.ssp_wave),
    )


def _stellar(model) -> StellarSEDComponent:
    return next(c for c in model._feature_chain() if isinstance(c, StellarSEDComponent))


# ── 1. Identity at the default weight ───────────────────────────────────


@pytest.mark.parametrize("ssp_name", [_MIST_C3K, _MIST_MILES])
@pytest.mark.parametrize("age_kernel", ["cic", "dsps"])
def test_fixed_default_weight_is_bit_identical_to_omitted(ssp_name, age_kernel):
    """``Fixed(1.0)`` is bit-identical to omitting the group: R(w=1) is exactly
    1 everywhere, so baking it multiplies the SSP cube by an all-ones array."""
    plain = _build(ssp_name, agb_dust=None, age_kernel=age_kernel)
    with_default = _build(
        ssp_name,
        agb_dust={"type": "fsps_shell", "weight": Fixed(1.0)},
        age_kernel=age_kernel,
    )
    key = jax.random.PRNGKey(0)
    params = plain.spec.sample(key)

    sed_plain = plain.predict(params).rest_sed()
    sed_default = with_default.predict(with_default.spec.sample(key)).rest_sed()
    assert float(jnp.max(jnp.abs(sed_plain - sed_default))) == 0.0

    phot_plain = jnp.asarray(plain.predict_photometry(params))
    phot_default = jnp.asarray(with_default.predict_photometry(with_default.spec.sample(key)))
    assert float(jnp.max(jnp.abs(phot_plain - phot_default))) == 0.0


@pytest.mark.parametrize("ssp_name", [_MIST_C3K, _MIST_MILES])
def test_fixed_default_weight_is_bit_identical_under_wave_precomp(ssp_name):
    """Same identity through the WavePrecomp photometry table (the weight is
    baked into the SSP grid before the table is built)."""
    plain = _build(ssp_name, agb_dust=None, approx=WavePrecomp())
    with_default = _build(
        ssp_name, agb_dust={"type": "fsps_shell", "weight": Fixed(1.0)}, approx=WavePrecomp()
    )
    key = jax.random.PRNGKey(1)
    params = plain.spec.sample(key)
    phot_plain = jnp.asarray(plain.predict_photometry(params))
    phot_default = jnp.asarray(with_default.predict_photometry(with_default.spec.sample(key)))
    assert float(jnp.max(jnp.abs(phot_plain - phot_default))) == 0.0


# ── 2. The shipped template ──────────────────────────────────────────────


class TestTemplate:
    """Numbers and provenance of ``data/agb_dust_shell_ratios_mist.h5``."""

    def test_w1_is_not_stored_and_is_the_exact_identity_plane(self):
        with h5py.File(_template_path(), "r") as f:
            assert 1.0 not in np.asarray(f["weights"][:])
        template = load_agb_dust_shell_template()
        plane = template.ratio_planes[int(np.argmin(np.abs(template.weights - 1.0)))]
        assert np.all(plane == 1.0)

    def test_pins_match_the_fsps_output(self):
        with h5py.File(_template_path(), "r") as f:
            median_pin = float(f.attrs["pin_median_inv_r0_2_10um"])
            max_pin = float(f.attrs["pin_max_inv_r0_2_10um"])
            r3_pin = float(f.attrs["pin_r3_at_10um"])
        assert median_pin == pytest.approx(_PIN_MEDIAN_INV_R0, abs=1e-4)
        assert max_pin == pytest.approx(_PIN_MAX_INV_R0, abs=1e-4)
        assert r3_pin == pytest.approx(_PIN_R3_AT_10UM, abs=1e-4)

    def test_stored_planes_reproduce_the_pins(self):
        """The pins re-derived from the stored planes, not only the attrs."""
        ssp = load_ssp(_MIST_C3K)
        resampled = _resampled(ssp)
        z = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - (-1.8477))))
        a = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr))))
        wave = np.asarray(ssp.ssp_wave)
        i10 = int(np.argmin(np.abs(wave - 1.0e5)))
        assert float(agb_dust_ratio(resampled, 3.0)[z, a, i10]) == pytest.approx(
            _PIN_R3_AT_10UM, abs=3e-3
        )
        band = (wave >= 2.0e4) & (wave <= 1.0e5)
        inv_r0 = 1.0 / np.asarray(agb_dust_ratio(resampled, 0.0)[z, a, band])
        assert np.max(inv_r0) == pytest.approx(_PIN_MAX_INV_R0, abs=0.1)

    def test_ratio_is_stored_unclamped(self):
        """FSPS's own extremes are kept: R reaches below 0.1 (far-infrared at
        w=0) and above 10 (old, metal-poor populations below 1500 A)."""
        template = load_agb_dust_shell_template()
        assert template.ratio_planes.min() < 0.1
        assert template.ratio_planes.max() > 10.0

    def test_provenance_names_the_libraries_the_ratio_was_computed_with(self):
        with h5py.File(_template_path(), "r") as f:
            libraries = str(f.attrs["libraries"])
            reuse = str(f.attrs["reuse_note"])
            window_note = str(f.attrs["window_edge_note"])
        assert "mist" in libraries and "miles" in libraries
        assert "c3k" not in libraries
        for token in ("c3k_a", "Kroupa", "Salpeter", "cannot be measured"):
            assert token in reuse
        assert "not by shell absorption" in window_note

    def test_imf_sensitivity_is_recorded_with_its_scope(self):
        with h5py.File(_template_path(), "r") as f:
            diff = float(f.attrs["imf_sensitivity_max_diff"])
            scope = str(f.attrs["imf_sensitivity_scope"])
        assert 0.0 < diff < 0.03
        assert "Z_sun only" in scope

    def test_interpolation_error_against_direct_fsps_is_recorded(self):
        """Midpoint error of every interval between stored weights, measured
        against direct FSPS (2-30 um, Z_sun, 0.3/1/3 Gyr)."""
        with h5py.File(_template_path(), "r") as f:
            weights = np.concatenate([np.asarray(f["weights"][:]), [1.0]])
            weights.sort()
            midpoints = np.asarray(f.attrs["interp_midpoints"])
            errors = np.asarray(f.attrs["interp_max_rel_err"])
        np.testing.assert_allclose(midpoints, 0.5 * (weights[:-1] + weights[1:]))
        # R is nearly discontinuous at w = 0 (FSPS switches the shell model off
        # there); no plane spacing resolves the first interval.
        assert errors[0] < 0.04
        assert np.all(errors[1:] <= 0.02)

    def test_file_size_under_cap(self):
        size_mb = os.path.getsize(_template_path()) / 1e6
        assert size_mb <= 10.0, f"{_template_path()} is {size_mb:.2f} MB, over the 10 MB cap"


# ── 3. Window behavior ───────────────────────────────────────────────────


def test_ratio_is_exactly_one_outside_the_stored_window():
    template = load_agb_dust_shell_template()
    ssp = load_ssp(_MIST_C3K)
    wave = np.asarray(ssp.ssp_wave)
    outside = (wave < template.wave_angstrom.min()) | (wave > template.wave_angstrom.max())
    assert np.count_nonzero(outside) > 0
    resampled = _resampled(ssp)
    for w in (0.0, 0.4, 3.0):
        assert np.all(np.asarray(agb_dust_ratio(resampled, w))[:, :, outside] == 1.0)


def test_ratio_is_piecewise_linear_between_stored_weights():
    """Between nodes R(w) is the straight line through the bracketing planes."""
    ssp = load_ssp(_MIST_C3K)
    resampled = _resampled(ssp)
    weights = np.asarray(resampled.weights)
    planes = np.asarray(resampled.ratio_planes)
    i = int(np.searchsorted(weights, 0.4)) - 1  # 0.3125 < 0.4 < 0.4375
    frac = (0.4 - weights[i]) / (weights[i + 1] - weights[i])
    hand = planes[i] + frac * (planes[i + 1] - planes[i])
    np.testing.assert_allclose(np.asarray(agb_dust_ratio(resampled, 0.4)), hand, rtol=1e-6)


def test_ratio_monotonic_in_weight_at_10um_for_intermediate_age():
    """At 10 um for 1 Gyr, solar, R rises into the reference weight w=1 and
    falls away from it: monotonic on each side of the stored node at w=1."""
    ssp = load_ssp(_MIST_C3K)
    resampled = _resampled(ssp)
    age_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr))))
    met_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - (-1.848))))
    wave_idx = int(np.argmin(np.abs(np.asarray(ssp.ssp_wave) - 1.0e5)))

    def _value(w):
        return float(agb_dust_ratio(resampled, w)[met_idx, age_idx, wave_idx])

    below = [_value(w) for w in (0.0, 0.0625, 0.25, 0.5, 0.75, 1.0)]
    above = [_value(w) for w in (1.0, 1.25, 1.5, 2.0, 3.0)]
    assert np.all(np.diff(below) >= -1e-9), f"not monotonic for w<=1: {below}"
    assert np.all(np.diff(above) <= 1e-9), f"not monotonic for w>=1: {above}"


# ── 4. The seam: baked == live == SSP cube times R ───────────────────────


def test_baked_cube_is_the_ssp_cube_times_the_ratio():
    """The cube a Fixed weight bakes is the SSP cube times R(w), cell by cell;
    for a single-metallicity single-age burst this is the whole prediction."""
    ssp = load_ssp(_MIST_C3K)
    ratio = np.asarray(agb_dust_ratio(_resampled(ssp), 2.0))
    baked = np.asarray(bake_agb_dust_shell(ssp, 2.0).ssp_flux)
    np.testing.assert_allclose(baked, np.asarray(ssp.ssp_flux) * ratio, rtol=1e-6)
    z = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - (-1.848))))
    a = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr))))
    np.testing.assert_allclose(
        baked[z, a], np.asarray(ssp.ssp_flux)[z, a] * ratio[z, a], rtol=1e-6
    )
    assert not np.allclose(ratio[z, a], 1.0)


@pytest.mark.parametrize("weight", [2.0, 0.4])
def test_fixed_weight_equals_free_weight_live(weight):
    """A Fixed weight baked at build time and the same value fed live to a free
    weight give the same spectrum and photometry."""
    baked = _build(_MIST_C3K, agb_dust={"type": "fsps_shell", "weight": Fixed(weight)})
    free = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    params = baked.spec.sample(jax.random.PRNGKey(3))
    free_params = {**params, "agb_dust_weight": weight}

    sed_baked = np.asarray(baked.predict(params).rest_sed())
    sed_free = np.asarray(free.predict(free_params).rest_sed())
    np.testing.assert_allclose(sed_free, sed_baked, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(free.predict_photometry(free_params)),
        np.asarray(baked.predict_photometry(params)),
        rtol=1e-5,
    )
    default = np.asarray(_build(_MIST_C3K).predict(params).rest_sed())
    assert np.max(np.abs(sed_free / default - 1.0)) > 1e-3


def test_prediction_at_a_non_node_weight_matches_a_hand_computation():
    """The model's rest-frame SED at w=0.4 (between the stored nodes 0.3125 and
    0.4375) is total_mass * sum(joint weights * SSP cube * R) * L_sun."""
    model = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    params = {**model.spec.sample(jax.random.PRNGKey(4)), "agb_dust_weight": 0.4}
    ssp = model.ssp_data
    ratio = np.asarray(agb_dust_ratio(_resampled(ssp), 0.4))

    joint_weights, total_mass, _ = _stellar(model).compute_joint_weights(
        model._evaluation_params(params, None)
    )
    hand = (
        float(total_mass)
        * np.einsum("ma,maw->w", np.asarray(joint_weights), np.asarray(ssp.ssp_flux) * ratio)
        * LSUN_ERG_PER_S
    )
    got = np.asarray(model.predict(params).rest_sed())
    np.testing.assert_allclose(got, hand, rtol=1e-5)


def test_compute_log_nion_applies_the_live_ratio():
    """The SED-free ionizing rate reads the corrected cube: with a ratio that
    doubles the flux shortward of the Lyman limit it rises by log10(2)."""
    model = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    params = model._evaluation_params(
        {**model.spec.sample(jax.random.PRNGKey(5)), "agb_dust_weight": 2.0}, None
    )
    stellar = _stellar(model)
    wave = np.asarray(model.ssp_data.ssp_wave)
    plane_shape = (1, *model.ssp_data.ssp_flux.shape)
    boost = np.where(wave < 912.0, 2.0, 1.0)[None, None, None, :]
    boosted = ResampledAGBDustShell(
        weights=np.array([0.0, 3.0]),
        ratio_planes=jnp.asarray(np.broadcast_to(boost, (2, *plane_shape[1:])).copy()),
    )
    base = float(dataclasses.replace(stellar, agb_dust_ratio=None).compute_log_nion(params))
    live = float(dataclasses.replace(stellar, agb_dust_ratio=boosted).compute_log_nion(params))
    assert live - base == pytest.approx(np.log10(2.0), abs=1e-6)


def test_compute_log_nion_matches_the_published_log_nion_with_the_real_template():
    model = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    params = {**model.spec.sample(jax.random.PRNGKey(6)), "agb_dust_weight": 0.0}
    full = model._evaluation_params(params, None)
    stellar = _stellar(model)
    published = float(model.predict(params).properties["log_q_h"])
    assert float(stellar.compute_log_nion(full)) == pytest.approx(published, abs=1e-6)


def test_nebular_ionizing_tables_are_unaffected_by_the_shell_weight():
    """Nebular backends tabulate Q_H(Z, age) from the SSP cube at build time,
    for ages up to 100 Myr. R differs from 1 below the Lyman limit only for
    some 45-100 Myr populations, whose ionizing output is negligible: the
    table error is below 1e-4 dex for ages up to 10 Myr and the constant-SFH
    integrated rate changes by less than 1e-5."""
    from tengri.components.nebular.cloudy_grid import _compute_log_qh_grid

    ssp = load_ssp(_MIST_C3K)
    resampled = _resampled(ssp)
    age_yr = 10.0 ** (np.asarray(ssp.ssp_lg_age_gyr) + 9.0)
    base = np.asarray(_compute_log_qh_grid(ssp.ssp_wave, ssp.ssp_flux))
    young = age_yr <= 1e8
    for w in (0.0, 3.0):
        ratio = np.asarray(agb_dust_ratio(resampled, w))
        shifted = np.asarray(_compute_log_qh_grid(ssp.ssp_wave, ssp.ssp_flux * ratio))
        assert np.abs((shifted - base)[:, age_yr <= 1e7]).max() < 1e-4
        dt = np.gradient(age_yr)
        before = (10.0**base * dt)[:, young].sum(axis=1)
        after = (10.0**shifted * dt)[:, young].sum(axis=1)
        assert np.abs(after / before - 1.0).max() < 1e-5


# ── 5. Refusals ──────────────────────────────────────────────────────────


def test_non_mist_grid_refuses_naming_supported_grids():
    with pytest.raises(ValueError, match="MIST") as exc:
        _build(_NON_MIST, agb_dust={"type": "fsps_shell", "weight": Fixed(1.0)})
    for name in SUPPORTED_GRIDS:
        assert name in str(exc.value)


@pytest.mark.parametrize("approx", [WavePrecomp(), SpectrumPrecomp()])
def test_free_weight_refuses_the_precompute_tables_naming_the_exact_path(approx):
    """Raised in ``SEDModel.__init__`` before any table is attempted: the
    refusal is deterministic, so leaving it to the build-time catch-warn-and-
    fall-back handling would give a model that builds and then always fails."""
    with pytest.raises(ValueError, match="exact path"):
        _build(_MIST_C3K, agb_dust=_FREE_WEIGHT, approx=approx)


def test_free_weight_refuses_the_window_lut_fast_paths():
    """``approx=True`` on indices and line fluxes reads a window table built
    once from the SSP cube; it refuses for a free weight and still serves a
    Fixed weight, whose cube is baked."""
    from tengri.observation.line_measurement import default_line_defs

    free = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    params = free.spec.sample(jax.random.PRNGKey(7))
    dn4000 = (STANDARD_INDICES["Dn4000"],)
    with pytest.raises(ValueError, match="free agb_dust_weight"):
        free.predict_spectral_indices(params, dn4000, approx=True)
    lines = default_line_defs(np.asarray([6564.61, 5008.24]))
    with pytest.raises(ValueError, match="free agb_dust_weight"):
        free.measure_line_fluxes(params, lines, approx=True)

    fixed = _build(_MIST_C3K, agb_dust={"type": "fsps_shell", "weight": Fixed(2.0)})
    fixed_params = fixed.spec.sample(jax.random.PRNGKey(7))
    fast = np.asarray(fixed.predict_spectral_indices(fixed_params, dn4000, approx=True))
    exact = np.asarray(fixed.predict_spectral_indices(fixed_params, dn4000, approx=False))
    assert np.all(np.isfinite(fast))
    np.testing.assert_allclose(fast, exact, rtol=0.02)


def test_auto_approx_resolves_to_exact_for_free_weight():
    model = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    n_bands = len(model.observation.photometry.filters)
    fitted = Fitter(model, jnp.ones(n_bands), jnp.ones(n_bands), approx="auto").model
    assert not fitted._has_modern_approx()


@pytest.mark.parametrize("data_type", ["photometry", "spectroscopy", "joint"])
def test_batch_fit_resolver_keeps_the_exact_model_for_free_weight(data_type):
    """Catalog and population fits resolve ``approx='auto'`` through
    ``_resolve_batch_fit_approx``: a free weight stays on the exact model,
    silently (no table is attempted, so there is nothing to warn about)."""
    model = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_batch_fit_approx(model, "auto", data_type) is model


def test_batch_fit_resolver_warns_and_stays_exact_for_an_explicit_table():
    model = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    with pytest.warns(UserWarning, match="exact path"):
        resolved = _resolve_batch_fit_approx(model, WavePrecomp(), "photometry")
    assert resolved is model


def test_exact_spectrum_applies_the_live_weight():
    """The exact spectroscopy path (the one a free weight uses) responds to the
    weight; precompute tables are refused above."""
    model = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    params = model.spec.sample(jax.random.PRNGKey(8))
    wave = np.asarray(model.ssp_data.ssp_wave)
    at_10um = int(np.argmin(np.abs(wave - 1.0e5)))
    w0 = np.asarray(model.predict({**params, "agb_dust_weight": 0.0}).rest_sed())[at_10um]
    w1 = np.asarray(model.predict({**params, "agb_dust_weight": 1.0}).rest_sed())[at_10um]
    assert w0 < w1


# ── 6. Grammar ───────────────────────────────────────────────────────────


def test_none_and_omitted_are_bit_identical():
    omitted = _build(_MIST_C3K, agb_dust=None)
    none_type = _build(_MIST_C3K, agb_dust={"type": "none"})
    key = jax.random.PRNGKey(2)
    params = omitted.spec.sample(key)
    sed_omitted = omitted.predict(params).rest_sed()
    sed_none = none_type.predict(none_type.spec.sample(key)).rest_sed()
    assert float(jnp.max(jnp.abs(sed_omitted - sed_none))) == 0.0
    assert "agb_dust_weight" not in omitted.spec.free_params
    assert "agb_dust_weight" not in omitted.spec.get_fixed_values()


def test_all_params_free_frees_exactly_agb_dust_weight():
    model = _build(_MIST_C3K, agb_dust={"type": "fsps_shell", "all_params": FREE})
    agb_free = {p for p in model.spec.free_params if p.startswith("agb_dust_")}
    assert agb_free == {"agb_dust_weight"}


def test_summary_tags_agb_dust_module():
    model = _build(_MIST_C3K, agb_dust={"type": "fsps_shell", "weight": Fixed(2.0)})
    assert "agb_dust" in model.spec.summary_str()


def test_to_groups_round_trip():
    from tengri.parameters.groups import parse_groups

    model = _build(_MIST_C3K, agb_dust=_FREE_WEIGHT)
    groups = model.spec.to_groups()
    assert groups.get("agb_dust", {}).get("type") == "fsps_shell"
    roundtrip_spec = parse_groups(ssp_data=model.ssp_data, **groups)
    assert set(roundtrip_spec.free_params) == set(model.spec.free_params)
    assert roundtrip_spec.get_fixed_values() == model.spec.get_fixed_values()
