# SPDX-License-Identifier: BSD-3-Clause
"""Every consumer of the sub-band quadrature nodes reads the nodes the build stored (#3505).

A model with ``n_subbands > 1`` and a live Lyman-continuum gate stores, at build, one set of
observed-frame sub-band node wavelengths per grid point: at the fixed redshift
(``ssp_phot_lut.ssp_subband_waves_rest``) or on the redshift table
(``ssp_phot_ztable.subband_waves_rest_table``), each ``K + 1`` wide (the gate forces the physical
Lyman edge into the partition). Four consumers read them:

* the IGM node fold at build time reads the stored tensor itself: its argument must be
  ``array_equal`` to the stored nodes;
* the stellar apply contracts the stored nodes over the population weights and publishes
  ``stellar_subband_waves_rest_precomp``. The node is a flux-weighted ratio, so it equals the
  stored node to round-off (1e-12), not bit for bit; where a sub-band carries no weight the
  published node is the filter's effective wavelength by design, and those entries are skipped;
* the dust screens (:func:`resolve_dust_law`), the nebular sub-band decomposition and the
  Lyman-continuum gate (``ionizing_mask``), and the runtime IGM node fold of a patchy IGM read
  the published array: each call's argument must be ``array_equal`` to it.

The SSP is separable (a wavelength shape times an age and a metallicity factor), so every grid
point has the same nodes and the contracted node can be compared with one stored row. Runtime
reads are recorded under ``jax.disable_jit`` (a spy cannot see inside a compiled program).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.photometry import FilterConvention, FilterCurve
from tests._data_skip import requires_cue_weights

pytestmark = [pytest.mark.contract, requires_cue_weights]

CONVENTIONS = (FilterConvention.BESSELL, FilterConvention.ENERGY)
LYC_LIMIT = 912.0
N_SUBBANDS = 5
#: Contracted node against stored node: a ratio of two sums over the same population weights.
NODE_RTOL = 1e-12
#: Redshifts of the fixed-redshift builds; the free build is read at two table nodes.
FIXED_REDSHIFTS = (1.5, 2.5)
FREE_RANGE = (1.5, 2.5)


def _ssp() -> SSPData:
    """A separable wide SSP with a Lyman break (so the Cue ionizing rate stays physical)."""
    wave = jnp.logspace(2.0, 7.0, 1600)
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    shape = (5000.0 / wave) ** 2 * jnp.where(wave < LYC_LIMIT, 0.03, 1.0)
    flux = (
        shape[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave, ssp_flux=jnp.abs(flux) + 1e-12, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet
    )


def _band(center_rest: float, z: float, frac: float, name: str) -> FilterCurve:
    center = center_rest * (1.0 + z)
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), 60)
    return FilterCurve(wave=wave, trans=jnp.sin(jnp.linspace(0.0, jnp.pi, 60)) * 0.6, name=name)


#: IGM variants: (patchy, igm_fold). The node fold of a mean IGM and of a patchy IGM read the
#: stored nodes at build and at runtime respectively; the exact fold publishes its own nodes.
VARIANTS = {
    "mean_node": (False, "node"),
    "patchy_node": (True, "node"),
    "mean_exact": (False, "exact"),
}


def _build(convention, redshift, variant, z_band):
    patchy, fold = VARIANTS[variant]
    """Cue nebular (live Lyman gate), two-component dust and an IGM, under WavePrecomp."""
    filters = (_band(LYC_LIMIT, z_band, 0.35, "lyman"), _band(5000.0, z_band, 0.1, "clean"))
    igm = {"type": "inoue", "all_params": Fixed(DEFAULT)}
    if patchy:
        igm["patchy"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp(),
            observation=Observation(photometry=Photometry(filters=filters, convention=convention)),
            sfh={
                "type": "delayed",
                "log_total_mass": 10.0,
                "tau_gyr": 1.0,
                "age_gyr": Fixed(1.0),
                "all_params": Fixed(DEFAULT),
            },
            met={"logzsol": 0.0, "all_params": Fixed(DEFAULT)},
            neb={"type": "cue", "all_params": Fixed(DEFAULT), "lyc_max_age_yr": 1.0e7},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_bc": 0.5,
                "tau_diff": 0.3,
                "lyc_reprocessed_by": "young",
            },
            igm=igm,
            redshift=redshift,
            approx=WavePrecomp(igm_fold=fold, n_subbands=N_SUBBANDS),
        )


class _Recorder:
    """Spies on every consumer of the nodes; records the array each one is handed."""

    def __init__(self, monkeypatch):
        self.calls: dict[str, list[np.ndarray]] = {
            "ionizing_mask_nebular": [],
            "ionizing_mask_dust": [],
            "dust_law": [],
            "igm_node_fold": [],
            "igm_runtime_fold": [],
        }
        self._mp = monkeypatch

    def _wrap(self, key, fn, arg_index=0):
        def spy(*args, **kwargs):
            self.calls[key].append(np.asarray(args[arg_index]))
            return fn(*args, **kwargs)

        return spy

    def patch_runtime(self):
        from tengri.components import lyc
        from tengri.components.dust import attenuation
        from tengri.components.igm import component as igm_component
        from tengri.components.nebular import component as neb_component

        self._mp.setattr(
            neb_component,
            "ionizing_mask",
            self._wrap("ionizing_mask_nebular", neb_component.ionizing_mask),
        )
        self._mp.setattr(lyc, "ionizing_mask", self._wrap("ionizing_mask_dust", lyc.ionizing_mask))
        resolve = attenuation.resolve_dust_law

        def resolve_spy(name):
            return self._wrap("dust_law", resolve(name))

        self._mp.setattr(attenuation, "resolve_dust_law", resolve_spy)
        self._mp.setattr(
            igm_component,
            "igm_absorption_for_component",
            self._wrap("igm_runtime_fold", igm_component.igm_absorption_for_component, 2),
        )

    def patch_build(self):
        from tengri.components.igm.component import IGMSEDComponent

        original = IGMSEDComponent.subband_node_transmission

        def spy(component, waves, *args, **kwargs):
            self.calls["igm_node_fold"].append(np.asarray(waves))
            return original(component, waves, *args, **kwargs)

        self._mp.setattr(IGMSEDComponent, "subband_node_transmission", spy)


def _stored_nodes(model, z_index=None):
    """The build's stored node tensor and the redshift(s) it is read at."""
    state = model._cached_component_chain[0]._state
    lut = getattr(state, "ssp_phot_lut", None)
    if lut is not None and lut.ssp_subband_waves_rest is not None:
        return np.asarray(lut.ssp_subband_waves_rest), None
    ztable = state.ssp_phot_ztable
    return np.asarray(ztable.subband_waves_rest_table), np.asarray(ztable.z_grid)


def _check_build_fold(recorder, stored, fold):
    folds = recorder.calls["igm_node_fold"]
    if fold != "node":
        assert not folds
        return
    assert folds, "the IGM node fold was never called at build"
    for a in folds:
        assert np.array_equal(a, stored), "the IGM node fold read nodes other than the stored ones"


def _derived(model, params):
    with jax.disable_jit():
        return model.predict_state(params).derived


def _check_consumers(recorder, derived, *, patchy, z):
    nodes = np.asarray(derived["stellar_subband_waves_rest_precomp"])
    igm_published = derived.get("stellar_subband_waves_rest_igm_precomp")
    igm_nodes = None if igm_published is None else np.asarray(igm_published)
    shape = nodes.shape
    for key in ("ionizing_mask_nebular", "ionizing_mask_dust"):
        seen = [a for a in recorder.calls[key] if a.shape == shape]
        assert seen, f"{key}: no sub-band-shaped read recorded"
        for a in seen:
            assert np.array_equal(a, nodes), f"{key}: read nodes other than the published ones"
    law = [a for a in recorder.calls["dust_law"] if a.shape == shape]
    assert law, "dust screens: no sub-band-shaped law evaluation recorded"
    for a in law:
        assert np.array_equal(a, nodes) or (
            igm_nodes is not None and np.array_equal(a, igm_nodes)
        ), "dust screens: evaluated at nodes other than the published ones"
    assert any(np.array_equal(a, nodes) for a in law)
    if patchy:
        folds = recorder.calls["igm_runtime_fold"]
        assert folds, "patchy IGM: the runtime node fold was never called"
        expected = np.asarray(jnp.asarray(nodes).reshape(-1) * (1.0 + z))
        assert any(np.array_equal(a, expected) for a in folds)


def _check_stored_against_published(stored, derived, index):
    nodes = np.asarray(derived["stellar_subband_waves_rest_precomp"])
    fallback = np.asarray(derived["filter_eff_waves"])[:, None]
    row = stored[index]
    flat = row.reshape(-1, *row.shape[-2:])
    np.testing.assert_allclose(
        flat, np.broadcast_to(flat[0], flat.shape), rtol=NODE_RTOL, atol=0.0
    )
    live = nodes != fallback
    assert live.mean() > 0.3, "most sub-bands carry no weight: the comparison is vacuous"
    np.testing.assert_allclose(
        nodes[live], np.broadcast_to(flat[0], nodes.shape)[live], rtol=NODE_RTOL, atol=0.0
    )


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("convention", CONVENTIONS, ids=lambda c: c.value)
@pytest.mark.parametrize("z", FIXED_REDSHIFTS)
def test_fixed_redshift_nodes(monkeypatch, z, convention, variant):
    recorder = _Recorder(monkeypatch)
    recorder.patch_build()
    model = _build(convention, Fixed(z), variant, z)
    stored, z_grid = _stored_nodes(model)
    assert z_grid is None
    assert stored.shape[-1] == N_SUBBANDS + 1, "the Lyman gate must add one edge to the partition"
    _check_build_fold(recorder, stored, VARIANTS[variant][1])
    recorder.patch_runtime()
    derived = _derived(model, {})
    _check_stored_against_published(stored, derived, (slice(None), slice(None)))
    _check_consumers(recorder, derived, patchy=VARIANTS[variant][0], z=z)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("convention", CONVENTIONS, ids=lambda c: c.value)
@pytest.mark.parametrize("fraction", [0.25, 0.75])
def test_redshift_table_nodes(monkeypatch, fraction, convention, variant):
    recorder = _Recorder(monkeypatch)
    recorder.patch_build()
    model = _build(convention, Uniform(*FREE_RANGE), variant, 2.0)
    stored, z_grid = _stored_nodes(model)
    assert z_grid is not None
    assert stored.shape[-1] == N_SUBBANDS + 1, "the Lyman gate must add one edge to the partition"
    _check_build_fold(recorder, stored, VARIANTS[variant][1])
    index = int(fraction * (z_grid.shape[0] - 1))
    z = float(z_grid[index])
    recorder.patch_runtime()
    derived = _derived(model, {"redshift": z})
    _check_stored_against_published(stored, derived, (index, slice(None), slice(None)))
    _check_consumers(recorder, derived, patchy=VARIANTS[variant][0], z=z)
