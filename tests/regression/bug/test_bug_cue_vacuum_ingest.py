# SPDX-License-Identifier: BSD-3-Clause
r"""Cue and CB_19 ingest vacuum wavelengths; no post-hoc air/vacuum vote.

Follow-up to the owner's vacuum-wavelength ruling (2026-10-07): every
wavelength inside tengri is vacuum and an air input converts exactly once, at
ingestion. ``data/cue_weights.npz`` stores Cloudy AIR labels (6562.80, 5006.84,
...) and the hosted ``cb19_templates.h5`` mixes air and vacuum entries. Both
loaders now convert at load, and the Balmer-vote correction that used to
re-derive the frame at publication (``nebular_line_waves_to_vacuum``) is gone.

Before this change the Cue loader returned the air labels, so the published
catalog sat 1.2 to 1.8 Angstrom blueward of the vacuum catalog and the SED
placed the lines there; CB_19 returned [N II] 6583.45 against a vacuum 6585.27.
"""

from __future__ import annotations

import warnings

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.nebular._line_ingest import CLOUDY_LABEL_AIR_MAX_AA
from tengri.observation.eline_catalog import EMISSION_LINES
from tengri.observation.line_measurement import DESI_LINES, LineDef
from tests._data_skip import CUE_WEIGHTS, DATA_DIR, requires_cue_weights
from tests.regression.bug.test_bug_nebular_grid_eval_redshift import (  # noqa: F401
    _Models,
    _point,
    _shared_stellar_ztables,
)

pytestmark = pytest.mark.regression_bug

_TOL_AA = 0.05
_SSP = DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_WNE_SSP = DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_CB19 = DATA_DIR / "cb19_templates.h5"

#: Optical lines Cue carries, by their key in the vacuum catalog.
_CUE_CHECK = (
    "OII3727",
    "Hgamma",
    "Hbeta",
    "OIII4959",
    "OIII5007",
    "OI6300",
    "NII6548",
    "Halpha",
    "NII6583",
    "SII6716",
    "SII6731",
)
_ALIAS = {"OII3727": "OII3726"}


def _catalog_wave(key: str) -> float:
    return float(EMISSION_LINES[_ALIAS.get(key, key)][0])


@pytest.fixture(scope="module")
def cue_waves() -> np.ndarray:
    if not CUE_WEIGHTS.is_file():
        pytest.skip("cue_weights.npz not present")
    from tengri.components.nebular.cue import CueBackend

    return CueBackend(str(CUE_WEIGHTS)).published_line_wavelengths(cloudyfsps_only=False)


def _nearest(waves: np.ndarray, target: float) -> float:
    return float(waves[int(np.argmin(np.abs(waves - target)))])


# ── Cue loader ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", _CUE_CHECK)
def test_cue_published_centers_land_on_the_vacuum_catalog(cue_waves, key):
    target = _catalog_wave(key)
    assert abs(_nearest(cue_waves, target) - target) < _TOL_AA, key


def test_cue_published_centers_land_on_the_desi_vacuum_lines(cue_waves):
    for line in DESI_LINES:
        if 3000.0 < line.wavelength < 9000.0:
            got = _nearest(cue_waves, line.wavelength)
            assert abs(got - line.wavelength) < _TOL_AA, (line.name, got)


def test_cue_weights_arrays_are_the_vacuum_conversion_of_the_stored_air_labels():
    """The file stores air; the loaded arrays are air_to_vac of it, once."""
    if not CUE_WEIGHTS.is_file():
        pytest.skip("cue_weights.npz not present")
    from tengri.components.nebular.cue import load_cue_weights
    from tengri.utils.air_vacuum import air_to_vac

    raw = np.load(CUE_WEIGHTS)["nn_line_wavelength"]
    w = load_cue_weights(str(CUE_WEIGHTS))
    # Cloudy air-labels every line from 2000 A to 100 micron (Pa-alpha 18751.0, [Ne II]
    # 12.8101 micron), so the converted window is the Cloudy one, not the 1e4 A optical one.
    optical = (raw >= 2000.0) & (raw <= CLOUDY_LABEL_AIR_MAX_AA)
    assert optical.sum() > 50
    np.testing.assert_allclose(w.nn_line_wav[optical], air_to_vac(raw[optical]), rtol=0, atol=1e-9)
    np.testing.assert_array_equal(w.nn_line_wav[~optical], raw[~optical])
    assert np.all(w.nn_line_wav[optical] - raw[optical] > 0.5)  # vacuum is longer than air


@requires_cue_weights
def test_component_publishes_the_ingested_catalog_unchanged():
    """No second conversion: ``line_waves`` is the backend's array bit for bit."""
    if not _SSP.is_file():
        pytest.skip("FSPS SSP not present")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    model = SEDModel.build(
        ssp_data=load_ssp_data(str(_SSP)),
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )
    state = model.predict_state({}, fixed_values=model.spec.get_fixed_values())
    published = np.asarray(state.derived["line_waves"])
    backend = next(c for c in model._cached_component_chain if hasattr(c, "backend")).backend
    np.testing.assert_array_equal(
        published, backend.published_line_wavelengths(cloudyfsps_only=False)
    )
    assert abs(_nearest(published, 6564.61) - 6564.61) < _TOL_AA  # not 6566.4 (double)


def test_no_post_hoc_air_vacuum_vote_remains():
    from tengri.components.nebular import _shared, component
    from tengri.forward import properties

    for module in (_shared, component, properties):
        assert not hasattr(module, "nebular_line_waves_to_vacuum"), module.__name__
        assert not hasattr(module, "_BALMER_AIR_VAC"), module.__name__


# ── window measurement of the blended pair ─────────────────────────────────


def _half(name: str, center: float, lo: float, hi: float) -> LineDef:
    return LineDef(
        name,
        center,
        ((center - 14, center - 8), (center + 8, center + 14)),
        (center + lo, center + hi),
    )


def _halves_ratio(model, center: float) -> float:
    """Flux in [c-4, c] over flux in [c, c+4]: 1 for a line centered on c."""
    defs = [_half("blue", center, -4.0, 0.0), _half("red", center, 0.0, 4.0)]
    blue, red = np.asarray(model.measure_line_fluxes({}, defs, approx=False), dtype=float)
    assert np.isfinite(blue) and np.isfinite(red) and blue > 0.0 and red > 0.0
    return blue / red


@requires_cue_weights
@pytest.mark.parametrize("center", [6564.61, 6585.28])
def test_blended_pair_lines_are_centered_on_the_vacuum_centers(center):
    """[N II] 6584 and H-alpha sit on their vacuum centers in the rendered SED.

    A line rendered at its air label (1.8 Angstrom blueward) puts about twice
    the flux in the blue half-window as in the red one (measured ratio 2.1).
    """
    if not _SSP.is_file():
        pytest.skip("FSPS SSP not present")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=load_ssp_data(str(_SSP)),
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Fixed(10.0),
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(0.3),
            },
            dust_attenuation={"type": "none"},
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.05),
        )
    assert abs(_halves_ratio(model, center) - 1.0) < 0.05


@pytest.mark.parametrize("center", [6564.61, 6585.28])
def test_window_lut_measures_the_blended_pair_on_the_vacuum_centers(center):
    """The window LUT (baked-in grid) agrees with the exact path on the vacuum windows."""
    if not _WNE_SSP.is_file():
        pytest.skip("wNE SSP grid not available")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=load_ssp_data(str(_WNE_SSP)),
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Fixed(10.0),
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
            },
            dust_attenuation={"type": "none"},
            neb={"type": "none"},
            redshift=Fixed(0.0),
        )
    defs = [_half("blue", center, -4.0, 0.0), _half("red", center, 0.0, 4.0)]
    lut = np.asarray(model.measure_line_fluxes({}, defs, approx=True), dtype=float)
    exact = np.asarray(model.measure_line_fluxes({}, defs, approx=False), dtype=float)
    assert np.all(np.isfinite(lut)) and np.all(lut != 0.0)
    np.testing.assert_allclose(lut, exact, rtol=1e-9, atol=0.0)


# ── #2701 split-band line catalog ──────────────────────────────────────────


@requires_cue_weights
def test_grid_line_phot_waves_rest_is_the_vacuum_catalog(ssp_data_fsps):
    """``nebular_line_phot_waves_rest`` (#2701) carries vacuum rest wavelengths."""
    models = _Models(ssp_data_fsps)
    fast = models.fast(Uniform(0.05, 2.0))
    state = fast.predict_state(
        _point(fast, redshift=0.5),
        fixed_values=fast.spec.get_fixed_values(),
        observables_only=True,
    )
    waves = np.asarray(state.derived["nebular_line_phot_waves_rest"])
    published = np.asarray(state.derived["line_waves"]) if "line_waves" in state.derived else None
    assert waves.size > 100
    for key in ("Hbeta", "OIII5007", "Halpha", "NII6583"):
        target = _catalog_wave(key)
        assert abs(_nearest(waves, target) - target) < _TOL_AA, key
    if published is not None:
        np.testing.assert_array_equal(np.sort(waves), np.sort(published))
    assert jnp.all(jnp.isfinite(jnp.asarray(waves)))


# ── CB_19 ───────────────────────────────────────────────────────────────────


def _load_cb19_ignoring_the_placeholder_guard(monkeypatch):
    from tengri.components.nebular import cloudy_cb19

    monkeypatch.setattr(cloudy_cb19, "_reject_degenerate_line_ratios", lambda *a, **k: None)
    return cloudy_cb19.load_cb19_grid(str(_CB19))


def test_cb19_hosted_file_is_mixed_air_and_vacuum_and_loads_as_vacuum(monkeypatch):
    if not _CB19.is_file():
        pytest.skip("cb19_templates.h5 not present")
    with h5py.File(_CB19) as f:
        raw = np.asarray(f["line_wavelengths_aa"][:], dtype=float)
    # the convention, established from the values: Hbeta/Halpha/[OIII]5008 are
    # vacuum, Hgamma/[OI]6300/[NII]6548,6583 are Cloudy air labels
    assert abs(_nearest(raw, 6564.61) - 6564.61) < _TOL_AA
    assert abs(_nearest(raw, 6583.45) - 6583.45) < _TOL_AA
    got = np.asarray(_load_cb19_ignoring_the_placeholder_guard(monkeypatch).line_wavelengths)
    for key in ("Hgamma", "Hbeta", "OIII5007", "OI6300", "NII6548", "Halpha", "NII6583"):
        target = _catalog_wave(key)
        assert abs(_nearest(got, target) - target) < 0.06, (key, _nearest(got, target))

    changed = np.abs(got - raw) > 0.5
    assert changed.sum() == 4  # exactly the four air labels, nothing twice


def test_cb19_air_mask_leaves_a_vacuum_catalog_alone():
    from tengri.components.nebular._line_ingest import catalog_air_to_vacuum
    from tengri.components.nebular.cloudy_cb19 import _cb19_air_mask

    vac = np.array([1215.67, 4341.68, 4862.68, 6302.04, 6549.86, 6564.61, 6585.27])
    assert not _cb19_air_mask(vac).any()
    np.testing.assert_array_equal(catalog_air_to_vacuum(vac, air_mask=_cb19_air_mask(vac)), vac)
