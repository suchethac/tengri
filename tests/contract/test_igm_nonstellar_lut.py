# SPDX-License-Identifier: BSD-3-Clause
"""The IGM on non-stellar LUT photometry, against the wavelength-grid integrator.

Under ``WavePrecomp`` the stellar continuum carries the IGM inside its sub-band
quadrature; nebular, shock and AGN band fluxes used to take the filter-averaged
transmission ``<T>_b``, unweighted by their own spectrum. A Cue model's Ly-alpha
line sits on the break at high redshift, so that was the dominant LUT error there.
Measured, bare FSPS grid, dpl SFH, Cue at its defaults, inoue, fixed z, exact
stellar fold, vs ``approx=None``:

========  ======  ==========  ===========
z         band    ``<T>_b``   own ``T``
========  ======  ==========  ===========
7.0       sdss_z  +14.6 %     0.0000 %
7.3       sdss_z  +5.6 %      0.0000 %
7.3       F115W   +10.3 %     0.0000 %
========  ======  ==========  ===========

With a two-component screen (tau_bc 0.7, tau_diff 0.3) the stellar term also
needs its dust screen evaluated where the IGM-surviving light sits: at the bare
sub-band node sdss_z read 8.7 % off, at the IGM-weighted node 0.11 % (worst band
0.69 %, F090W). A composable AGN beside Cue reads <= 0.28 %, against 48 % under
the node fold.

(``sdss_i`` keeps the -0.1 to -0.8 % Lyman-limit mask-edge residual of #2447,
which is not an IGM term; F090W keeps the stellar -0.35 % that a bare-stellar
model shows too.) ``spectral_igm_correction`` is patched to zero for the
anti-vacuity arm: the ``<T>_b``-only answer these tests would otherwise accept.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, WavePrecomp
from tengri.observation import _igm_weighting
from tengri.parameters import Fixed

pytestmark = pytest.mark.contract

BANDS = ["sdss_z", "JWST_NIRCam_F090W", "JWST_NIRCam_F115W"]
PROBE_Z = 7.3
#: Measured worst 0.35 % (F090W, the stellar term a bare model shares).
_TOL = 0.01
#: Measured 10.3 % (F115W) with the correction zeroed.
_WITHOUT_FLOOR = 0.05


@pytest.fixture(scope="module")
def ssp(ssp_data_fsps):
    return ssp_data_fsps


def _model(ssp, bands, z, approx, **groups):
    return SEDModel.build(
        ssp_data=ssp,
        observation=tengri.Observation(photometry=tengri.Photometry.from_names(bands)),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        igm={"type": "inoue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(z),
        approx=approx,
        **groups,
    )


def _worst(lut, ref):
    return float(np.max(np.abs(np.asarray(lut, float) / np.asarray(ref, float) - 1.0)))


def _zeroed(*args, **kwargs):
    return 0.0 * args[-1]  # the band factor's shape and dtype


@pytest.fixture
def without_correction(monkeypatch):
    """Zero the correction, in a kernel of its own.

    Models with one compile signature share a compiled kernel process-wide, so
    the patched build would otherwise reuse the unpatched one, and a patched
    kernel left behind would serve every later test of that signature.
    """
    from tengri.inference._model_cache import clear_structural_kernel_cache

    clear_structural_kernel_cache()
    monkeypatch.setattr(_igm_weighting, "spectral_igm_correction", _zeroed)
    yield
    clear_structural_kernel_cache()


@pytest.mark.parametrize(
    "groups",
    [
        pytest.param({}, id="dust-free"),
        pytest.param(
            {
                "dust_attenuation": {
                    "type": "two_component",
                    "law": "calzetti",
                    "tau_bc": Fixed(0.7),
                    "tau_diff": Fixed(0.3),
                    "other_params": Fixed(DEFAULT),
                }
            },
            id="two-component",
        ),
        pytest.param(
            {
                "agn": {
                    "type": "composable",
                    "disc": {"type": "multicolor"},
                    "all_params": Fixed(DEFAULT),
                }
            },
            id="agn",
        ),
    ],
)
def test_nebular_flux_near_lyman_alpha_takes_its_own_transmission(ssp, groups):
    reference = _model(ssp, BANDS, PROBE_Z, None, **groups).predict_photometry({})
    lut = _model(ssp, BANDS, PROBE_Z, WavePrecomp(igm_fold="exact"), **groups)
    worst = _worst(lut.predict_photometry({}), reference)
    assert worst < _TOL, f"LUT off the integrator by {worst:.3%}"


@pytest.mark.parametrize("groups", [pytest.param({}, id="dust-free")])
def test_the_band_averaged_transmission_alone_is_wrong_here(ssp, without_correction, groups):
    """Anti-vacuity: the probe must be one the old ``<T>_b`` answer fails."""
    reference = _model(ssp, BANDS, PROBE_Z, None, **groups).predict_photometry({})
    without = _model(ssp, BANDS, PROBE_Z, WavePrecomp(igm_fold="exact"), **groups)
    worst = _worst(without.predict_photometry({}), reference)
    assert worst > _WITHOUT_FLOOR, (
        f"<T>_b alone is off by only {worst:.3%}: the probe no longer puts "
        "nebular emission on the Lyman-alpha break"
    )


def test_a_band_the_igm_cannot_reach_publishes_nothing(ssp):
    """Low redshift, optical bands: no table, no correction, no runtime cost."""
    model = _model(ssp, ["sdss_r", "sdss_i"], 0.1, WavePrecomp(igm_fold="exact"))
    state = model.predict_state({})
    assert state.derived.get("igm_rest_transmission_precomp") is None


def test_a_dusty_nebular_grid_takes_the_igm_at_its_sub_band_nodes(ssp):
    """#2679: the grid serves dusty nebular flux as screened chunks, no spectrum.

    With dust emission the energy-balance LUT exists and the per-Q_H grid serves
    the dusty nebular band flux (#2570), so there is no dense nebular spectrum to
    weight by. Each chunk takes ``T`` at its own node instead. Measured, z = 6.5
    to 7.3: worst band 0.94 % with the chunk correction, 10.1 % without (F090W,
    z = 7.0).
    """
    from tengri import FeaturePrecomp

    groups = {
        "dust_attenuation": {
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.7),
            "tau_diff": Fixed(0.3),
            "other_params": Fixed(DEFAULT),
        },
        "dust_emission": {"type": "dale2014", "all_params": Fixed(DEFAULT)},
    }
    z = 7.0
    lut = _model(ssp, BANDS, z, (WavePrecomp(igm_fold="exact"), FeaturePrecomp()), **groups)
    state = lut.predict_state({}, observables_only=True)
    assert state.derived.get("nebular_phot_lnu_subband_screened_precomp") is not None, (
        "the nebular grid no longer serves this dusty model; the probe tests nothing"
    )
    reference = _model(ssp, BANDS, z, None, **groups).predict_photometry({})
    worst = _worst(lut.predict_photometry({}), reference)
    assert worst < 0.02, f"grid-served dusty nebular off the integrator by {worst:.3%}"


#: Free-redshift probes where Ly-alpha sits inside F090W (6.5, 7.0) and F115W (8.0, 8.5).
_GRID_Z = (6.5, 7.0, 8.0, 8.5)


@pytest.fixture(scope="module")
def free_z_grid(ssp):
    """A dust-free, free-redshift nebular grid model and the integrator's photometry."""
    from tengri import FeaturePrecomp, Uniform

    common = dict(
        ssp_data=ssp,
        observation=tengri.Observation(photometry=tengri.Photometry.from_names(BANDS)),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        igm={"type": "inoue", "all_params": Fixed(DEFAULT)},
        redshift=Uniform(6.0, 9.0),
    )
    lut = SEDModel.build(approx=(WavePrecomp(z_min=6.0, z_max=9.0), FeaturePrecomp()), **common)
    exact = SEDModel.build(approx=None, **common)
    state = lut.predict_state({"redshift": 7.0}, observables_only=True)
    assert state.derived.get("nebular_phot_lnu_lines_precomp") is not None, (
        "the nebular grid no longer serves this dust-free model; the probe tests nothing"
    )
    return lut, {z: exact.predict_photometry({"redshift": z}) for z in _GRID_Z}


def _grid_worst(free_z_grid):
    lut, reference = free_z_grid
    return max(_worst(lut.predict_photometry({"redshift": z}), r) for z, r in reference.items())


def test_a_dust_free_nebular_grid_takes_the_igm_over_each_line_profile(free_z_grid):
    """The dust-free grid serves line rows and a continuum, no dense spectrum.

    It took ``<T>_b`` for both. Each line now takes the transmission averaged over
    its rendered profile and the continuum over its sub-band chunks. Measured,
    z = 6.5 to 8.5: worst band 0.07 %, against 20.6-23.8 % with ``<T>_b``.
    """
    worst = _grid_worst(free_z_grid)
    assert worst < _TOL, f"grid-served nebular off the integrator by {worst:.3%}"


def test_a_dust_free_nebular_grid_without_its_correction_is_wrong(free_z_grid, monkeypatch):
    """Anti-vacuity: with the chunk correction zeroed the probe reads ``<T>_b``.

    The correction is resolved at trace time, so clearing the kernel cache makes the
    same model recompile with the patched function.
    """
    from tengri.inference._model_cache import clear_structural_kernel_cache

    clear_structural_kernel_cache()
    monkeypatch.setattr(_igm_weighting, "subband_igm_correction", _zeroed)
    try:
        worst = _grid_worst(free_z_grid)
    finally:
        clear_structural_kernel_cache()
    assert worst > _WITHOUT_FLOOR, f"<T>_b alone is off by only {worst:.3%}"
