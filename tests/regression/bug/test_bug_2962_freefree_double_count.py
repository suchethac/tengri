# SPDX-License-Identifier: BSD-3-Clause
"""
Thermal free-free counted once when a nebular continuum carries it (#2962).

Invariant: at every rest wavelength, the thermal free-free emission of a model is
owned by exactly one component. A nebular continuum that carries free-free (Cue,
CloudyGrid, a ``NebularContinuumFallback`` whose analytic tier supplies the
free-free and two-photon continuum, or a wNE SSP whose flux holds it up to the grid
edge) takes that share, and the radio block's thermal term steps aside.

Two configurations broke it:

1. ``NebularContinuumFallback`` wrapped around a line-only backend (CB19, MAPPINGS).
   The factory gated the radio Murphy+2011 free-free on the declared backend name
   only, so the wrapper's analytic free-free was emitted beside it.
2. ``radio_sfr_mode="bell2003_split"`` on a nebular-included (wNE) SSP. The split's
   own 10 % thermal law ran from the radio edge (100 um) through the SSP edge, where
   the SSP already holds the thermal continuum. The #2574 wave-min rule excluded
   this mode.

The fixes gate both on one predicate, :func:`nebular_carries_freefree`, and window the
split's thermal law at the SSP edge like the Murphy+2011 term.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.nebular._shared import NebularContinuumFallback
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.radio.radio import radio_sfr_bell2003_split
from tengri.forward.component_factory import build_components
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.regression_bug

_Z = 0.05
_LINES_ONLY_NAME = "lines_only"


class _LinesOnlyBackend:
    """A line-only nebular backend: no continuum of its own (CB19-like)."""

    has_continuum = False
    has_free_params = False
    name = _LINES_ONLY_NAME

    def predict_nebular_sed(self, *args, **kwargs):
        wave = kwargs.get("wavelength", args[0] if args else None)
        return jnp.zeros_like(jnp.asarray(wave, dtype=jnp.float64))


def _radio_component(components) -> RadioSEDComponent:
    for component in components:
        if isinstance(component, RadioSEDComponent):
            return component
    raise AssertionError("radio component not built")


def _tophat(lo_obs, hi_obs, name, n=60):
    wave = jnp.linspace(lo_obs, hi_obs, n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=name)


_OBS = Observation(photometry=Photometry(filters=(_tophat(5000.0, 6000.0, "opt"),)))


def _build_split(ssp, *, ir_window="total"):
    """A bell2003_split radio block with no AGN term, on the given SSP."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=_OBS,
        sfh={"type": "const", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "draine_li2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "ssp"},
        radio={
            "sf": {"type": "bell2003_split", "ir_window": ir_window},
            "agn": {"type": "none"},
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(_Z),
    )


def test_wrapped_analytic_continuum_suppresses_radio_freefree(ssp_data_fsps):
    """A NebularContinuumFallback analytic tier carries free-free: no radio free-free.

    The wrapper is the factory's ``nebular_backend_instance`` for the declared
    ``cb19`` backend. Its Tier 2 supplies the free-free over the whole grid, so the
    radio block's ``include_freefree`` must resolve to False rather than the default
    True.
    """
    wrapped = NebularContinuumFallback(_LinesOnlyBackend())
    components = build_components(
        ssp_data=ssp_data_fsps,
        use_radio=True,
        nebular_backend="cb19",
        nebular_backend_instance=wrapped,
    )
    radio = _radio_component(components)
    assert radio.config.include_freefree is False


def test_split_thermal_on_wne_ssp_counts_once_at_the_edge(ssp_data_wne):
    """bell2003_split on a wNE SSP: the split's thermal law stops at the SSP edge.

    Below the SSP edge (1 cm here) the SSP holds the thermal continuum, so the radio
    block carries synchrotron only. Above it the SSP has no flux and the radio block
    carries the full split (synchrotron plus the 10 % thermal law). Checked on the
    radio SED grid against the closed-form split.
    """
    edge = float(np.max(np.asarray(ssp_data_wne.ssp_wave)))
    model = _build_split(ssp_data_wne)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    state = model.predict_state(params)

    wave = np.asarray(state.wave)
    sed = np.asarray(state.derived["sed_radio"])
    L_ir = jnp.asarray(state.derived["L_ir"])
    assert float(L_ir) > 0.0
    # The radio block's Fixed(DEFAULT) q_IR: the registry default, not a sampled key.
    q_ir = jnp.asarray(declared_default(RadioSEDComponent().declared_parameters(), "radio_q_ir"))

    sync_only = np.asarray(
        radio_sfr_bell2003_split(jnp.asarray(wave), L_ir, q_ir, include_thermal=False)
    )
    full_split = np.asarray(
        radio_sfr_bell2003_split(jnp.asarray(wave), L_ir, q_ir, include_thermal=True)
    )

    below = (wave < edge) & (wave >= 1.0e6)  # 100 um to the SSP edge
    above = wave >= edge
    assert below.any() and above.any()

    np.testing.assert_allclose(sed[below], sync_only[below], rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(sed[above], full_split[above], rtol=1e-10, atol=0.0)
    # The thermal law is not zero below the edge: it is what the window removed.
    assert np.all(full_split[below] > sync_only[below])


def test_split_thermal_window_is_set_from_the_wne_ssp_edge(ssp_data_wne):
    """The factory windows the split's thermal term at the wNE SSP edge."""
    edge = float(np.max(np.asarray(ssp_data_wne.ssp_wave)))
    model = _build_split(ssp_data_wne)
    cfg = next(config for name, config in model._component_configs if name == "radio")
    assert cfg.sfr_mode == "bell2003_split"
    assert cfg.freefree_wave_min == edge
    assert cfg.split_thermal is True
