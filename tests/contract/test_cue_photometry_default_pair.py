# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``approx="auto"`` photometry fits of a Cue model take the pair (#2377 follow-up).

Owner decision (2026-11-06): ``(WavePrecomp, FeaturePrecomp)`` is the default
approximation for Cue photometry fits. Without dust it already was (#1596). With
dust the resolvers checked ``fast_nebular_can_engage`` on the RAW model, where the
dust energy-balance LUT does not exist until ``WavePrecomp`` is attached, so a dusty
Cue fit silently kept ``WavePrecomp`` alone. The grid serves the dusty photometry
once the clone carries ``WavePrecomp``, measured at 60.8 MFLOP / 4.0 ms against
7.5 MFLOP / 1.4 ms for the pair on five bands.

The table pinned here, for the single-galaxy ``Fitter`` and the batch resolver:

* dusty Cue photometry, fixed or free redshift, under ``"auto"`` -> pair;
* no-dust Cue photometry -> pair (unchanged);
* a free nebular parameter the grid bakes (``gas_logn``) -> ``WavePrecomp`` only;
* a non-Cue backend (``gutkin16``) -> ``WavePrecomp`` only, although the same
  predicate on its WavePrecomp clone is True: the Cue gate is load-bearing;
* explicit ``approx=None`` -> exact, explicit ``approx=WavePrecomp()`` -> respected;
* spectroscopy and joint fits -> as before (no pair).
"""

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_BANDS = ["galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1"]


def _require():
    from pathlib import Path

    if not Path(_BARE).is_file():
        pytest.skip(f"missing {_BARE}")


@pytest.fixture(scope="module")
def ssp():
    _require()
    from tengri import load_ssp_data

    return load_ssp_data(_BARE)


@pytest.fixture(scope="module")
def data():
    fnu = np.array([2.0e-6, 4.0e-6, 6.0e-6, 8.0e-6, 1.0e-5, 1.5e-5])
    return fnu, 0.1 * fnu


def _dust_groups(dust: bool, ir: bool = True):
    from tengri import DEFAULT, Fixed, Uniform

    if not dust:
        return {"dust_attenuation": {"type": "none"}}
    groups = {
        "dust_attenuation": {
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": Uniform(0.0, 2.0),
            "tau_diff": Uniform(0.0, 2.0),
        },
    }
    if ir:
        groups["dust_emission"] = {"type": "dale2014", "all_params": Fixed(DEFAULT)}
    return groups


def _build(ssp, neb, *, dust=True, z=None, approx="unset", ir=True):
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    kwargs = dict(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        redshift=Fixed(0.15) if z is None else z,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(8, 12),
        },
        neb=neb,
        **_dust_groups(dust, ir),
    )
    if approx != "unset":
        kwargs["approx"] = approx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kwargs)


def _cue(**extra):
    from tengri import DEFAULT, Fixed, Uniform

    return {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "neb_logU": Uniform(-3.5, -2.0),
        **extra,
    }


def _gutkin16():
    from tengri import DEFAULT, Fixed

    return {"type": "gutkin16", "all_params": Fixed(DEFAULT)}


def _pair(state) -> bool:
    return bool(state.wave_precomp and state.feature_precomp)


def _fitter_state(model, data, approx="auto"):
    from tengri.inference import Fitter

    fnu, sigma = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter = Fitter(model, data=fnu, noise=sigma, data_type="photometry", approx=approx)
    return fitter.model.approx


def _batch_state(model, data_type="photometry"):
    from tengri.inference.fitter import _resolve_batch_fit_approx

    return _resolve_batch_fit_approx(model, "auto", data_type).approx


# --- single-galaxy Fitter -------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "dust", "free_z", "ir"),
    [
        ("dusty_fixed_z", True, False, True),
        ("dusty_free_z", True, True, True),
        ("dusty_no_ir_fixed_z", True, False, False),
        ("no_dust", False, False, True),
    ],
)
def test_fitter_auto_takes_pair_for_cue_photometry(ssp, data, label, dust, free_z, ir):
    from tengri import Uniform

    z = Uniform(0.1, 0.3) if free_z else None
    model = _build(ssp, _cue(), dust=dust, z=z, ir=ir)
    assert _pair(_fitter_state(model, data)), label


def test_fitter_auto_keeps_wave_only_when_grid_bakes_a_free_parameter(ssp, data):
    from tengri import Uniform

    model = _build(ssp, _cue(gas_logn=Uniform(1.0, 3.0)), dust=True)
    state = _fitter_state(model, data)
    assert state.wave_precomp and not state.feature_precomp


def test_fitter_auto_keeps_wave_only_for_non_cue_backend(ssp, data):
    model = _build(ssp, _gutkin16(), dust=True)
    state = _fitter_state(model, data)
    assert state.wave_precomp and not state.feature_precomp


def test_fitter_explicit_none_stays_exact(ssp, data):
    model = _build(ssp, _cue(), dust=True)
    state = _fitter_state(model, data, approx=None)
    assert not state.wave_precomp and not state.feature_precomp


def test_fitter_explicit_wave_precomp_is_respected(ssp, data):
    from tengri import WavePrecomp

    model = _build(ssp, _cue(), dust=True, approx=WavePrecomp())
    state = _fitter_state(model, data)
    assert state.wave_precomp


# --- batch resolver (PopulationFitter / CatalogFitter / hierarchical) -----------------


def test_batch_auto_takes_pair_for_dusty_cue_photometry(ssp):
    model = _build(ssp, _cue(), dust=True)
    assert _pair(_batch_state(model))


def test_batch_auto_takes_pair_for_no_dust_cue_photometry(ssp):
    model = _build(ssp, _cue(), dust=False)
    assert _pair(_batch_state(model))


def test_batch_auto_takes_pair_for_dusty_no_ir_cue_photometry(ssp):
    model = _build(ssp, _cue(), dust=True, ir=False)
    assert _pair(_batch_state(model))


def test_batch_auto_keeps_wave_only_when_grid_bakes_a_free_parameter(ssp):
    from tengri import Uniform

    model = _build(ssp, _cue(gas_logn=Uniform(1.0, 3.0)), dust=True)
    state = _batch_state(model)
    assert state.wave_precomp and not state.feature_precomp


def test_batch_auto_keeps_wave_only_for_non_cue_backend(ssp):
    model = _build(ssp, _gutkin16(), dust=True)
    state = _batch_state(model)
    assert state.wave_precomp and not state.feature_precomp


def test_batch_spectroscopy_is_unchanged(ssp):
    model = _build(ssp, _cue(), dust=True)
    state = _batch_state(model, data_type="spectroscopy")
    assert not state.feature_precomp


def test_batch_joint_is_unchanged(ssp):
    model = _build(ssp, _cue(), dust=True)
    state = _batch_state(model, data_type="joint")
    assert state.wave_precomp and not state.feature_precomp


def test_batch_explicit_none_stays_exact(ssp):
    from tengri.inference.fitter import _resolve_batch_fit_approx

    model = _build(ssp, _cue(), dust=True)
    resolved = _resolve_batch_fit_approx(model, None, "photometry")
    assert not resolved.approx.wave_precomp and not resolved.approx.feature_precomp
