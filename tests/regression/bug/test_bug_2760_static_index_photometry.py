# SPDX-License-Identifier: BSD-3-Clause
"""The production photometry path takes the build-time X-ray index of the torus sightline.

The photometry kernel traces the rest-frame grid, so the library sightline's X-ray
absorption cannot pick its samples from the grid inside the kernel. The SEDModel build
therefore carries the below-library data as a constant (``AGNSEDComponentState.
torus_below_index``). These tests pin that the production path carries it, and that the
photometry it produces equals the photometry of the same model without it (the dense
traced form) to rounding.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, WavePrecomp
from tengri.components.agn import component as agn_component
from tengri.components.agn.blocks import runner
from tengri.components.agn.component import AGNSEDComponent
from tengri.observation.photometry_config import Photometry
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.regression_bug

#: Relative agreement of the static-index photometry with the dense traced photometry.
RTOL = 1.0e-14

AGN = {
    "type": "composable",
    "torus": {"type": "skirtor"},
    "all_params": Fixed(DEFAULT),
    "norm": "cigale_joint",
    "log_lbol": Fixed(45.5),
}

OPTICAL = ("sdss_g", "sdss_r", "sdss_i")


def _model():
    return SEDModel.build(
        ssp_data=pytest.importorskip("tengri").load_ssp(),
        observation=Observation(photometry=Photometry.from_names(list(OPTICAL))),
        redshift=Fixed(0.1),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
        agn=AGN,
        approx=WavePrecomp(),
        xray={"type": "lopez24", "all_params": Fixed(DEFAULT), "gamma_agn": Fixed(2.1)},
    )


def _agn_component(model):
    model._template_data_for_jit()  # builds the component chain if it is not cached yet
    return next(c for c in model._cached_component_chain if isinstance(c, AGNSEDComponent))


def _params(model):
    return {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()}


def test_production_photometry_carries_the_build_time_index():
    comp = _agn_component(_model())
    index = comp._state.torus_below_index
    assert index is not None, "the production build did not carry the X-ray index"
    assert index.below.any(), "the grid has no samples below the library edge"
    assert index.below.shape == (index.n_wave,)


def test_static_index_photometry_matches_dense_photometry(monkeypatch):
    static_model = _model()
    static = np.asarray(
        jax.jit(lambda p: static_model.predict_photometry(p))(_params(static_model))
    )

    # The same model built without the index: the dense traced form. The structural kernel
    # cache is keyed on compile_signature, so the reference gets its own signature.
    monkeypatch.setattr(agn_component, "below_library_index", lambda *a, **k: None)
    dense_model = _model()
    dense_model._cached_component_chain = None
    comp = _agn_component(dense_model)
    assert comp._state.torus_below_index is None
    monkeypatch.undo()
    reference_signature = ("dense-reference-2760", *static_model.compile_signature())
    dense_model.compile_signature = lambda: reference_signature
    dense = np.asarray(jax.jit(lambda p: dense_model.predict_photometry(p))(_params(dense_model)))

    nz = dense != 0.0
    assert nz.any()
    rel = np.max(np.abs(static[nz] / dense[nz] - 1.0))
    assert rel <= RTOL, f"static-index photometry differs from dense by {rel:.3e}"


def _agn_on_grid(model, wave, *, closed_over: bool = False):
    """The AGN component's apply on ``wave``, as sed_intrinsic, under jit.

    ``closed_over`` keeps ``wave`` a constant of the trace, as the production predict path
    does with the master grid; otherwise the grid is a traced argument.
    """
    comp = _agn_component(model)
    params = {**model.spec.get_fixed_values(), **_params(model)}
    if closed_over:
        return jax.jit(lambda: comp.apply(ForwardState(wave=wave), params).sed_intrinsic)()
    return jax.jit(lambda w: comp.apply(ForwardState(wave=w), params).sed_intrinsic)(
        jnp.asarray(wave)
    )


def test_index_reaches_the_sightline_only_on_the_master_grid(monkeypatch):
    """The master grid gets the build-time index; a same-length explicit grid gets None."""
    model = _model()
    master = model._rest_wavelength
    seen = []
    original = runner.library_torus_sightline

    def spy(*args, **kwargs):
        seen.append(kwargs.get("below_index"))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "library_torus_sightline", spy)
    _agn_on_grid(model, master, closed_over=True)
    assert seen and all(index is not None for index in seen)

    seen.clear()
    _agn_on_grid(model, np.asarray(master) * 1.5)
    assert seen and all(index is None for index in seen)


def test_same_length_explicit_grid_matches_the_dense_built_model(monkeypatch):
    """On a grid with the master's length but other wavelengths, the output is the dense one.

    The master grid is stretched by 1.5: same length, different samples below the library
    edge. The static-index model must give the dense-built model's AGN emission on it.
    """
    static_model = _model()
    shifted = np.asarray(static_model._rest_wavelength) * 1.5

    static = np.asarray(_agn_on_grid(static_model, shifted))

    monkeypatch.setattr(agn_component, "below_library_index", lambda *a, **k: None)
    dense_model = _model()
    dense_model._cached_component_chain = None
    dense = np.asarray(_agn_on_grid(dense_model, shifted))
    monkeypatch.undo()

    nz = dense != 0.0
    assert nz.any()
    rel = np.max(np.abs(static[nz] / dense[nz] - 1.0))
    assert rel <= 1.0e-12, f"same-length grid got the master's absorption ({rel:.3e})"
