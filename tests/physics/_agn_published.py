# SPDX-License-Identifier: BSD-3-Clause
"""Shared builders and box draws for the AGN published-luminosity tests (#2842, #2851)."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

#: Standalone AGN ``SEDModelComponent`` classes that publish luminosity diagnostics:
#: name -> (module, class, config class, template file or None).
SED_MODEL_COMPONENTS = {
    "silva04": ("silva04_model", "Silva04Torus", "Silva04TorusConfig", "silva04_torus_grid.h5"),
    "cat3d": ("cat3d_torus_model", "CAT3DTorus", "CAT3DTorusConfig", "cat3d_wind_torus_grid.h5"),
    "skirtor_agnfitter": (
        "skirtor_agnfitter_model",
        "SKIRTORAgnfitterTorus",
        "SKIRTORAgnfitterTorusConfig",
        "skirtor_mean3p_torus_grid.h5",
    ),
    "skirtor": ("skirtor_model", "SKIRTORTorus", "SKIRTORTorusConfig", "skirtor_templates_v3.h5"),
    "kd18": ("kd18_disc_model", "KD18Disc", "KD18DiscConfig", None),
    "powerlaw": ("powerlaw_disc_model", "PowerLawDisc", "PowerLawDiscConfig", None),
}

#: Rest-frame caller grid [Angstrom] for the SED evaluations.
WAVE = np.geomspace(100.0, 1.0e6, 300)


def build_component(name):
    """One standalone component with its templates loaded; skips when the data are absent."""
    from tengri._data_setup import find_data

    module, cls_name, cfg_name, grid_file = SED_MODEL_COMPONENTS[name]
    mod = importlib.import_module(f"tengri.components.agn.{module}")
    kwargs = {}
    if grid_file is not None:
        found = find_data(grid_file)
        if found is None:
            pytest.skip(f"template data unavailable ({grid_file})")
        kwargs["grid_path"] = str(found)
    comp = getattr(mod, cls_name)(config=getattr(mod, cfg_name)(**kwargs))
    # Frozen dataclasses refuse plain assignment; the pipeline stores ``data`` the same way.
    object.__setattr__(comp, "data", comp.load(None))
    return comp


def declared_box(comp):
    """``{name: (lo, hi)}`` of a component's free parameters, prefix stripped."""
    prefix = comp.parameter_prefix
    box = {}
    for decl in comp.declared_parameters():
        prior = decl.prior
        lo = getattr(prior, "lo", None)
        key = decl.name[len(prefix) :]
        box[key] = (float(prior.default),) * 2 if lo is None else (float(lo), float(prior.hi))
    return box


def box_points(box, n_random, seed=0):
    """The box center, both edges of every parameter, and ``n_random`` uniform draws."""
    rng = np.random.default_rng(seed)
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in box.items()}
    points = [dict(mid)]
    for key, (lo, hi) in box.items():
        points.extend({**mid, key: edge} for edge in (lo, hi))
    points.extend({k: rng.uniform(lo, hi) for k, (lo, hi) in box.items()} for _ in range(n_random))
    return points


def predict(comp, params, *, float32):
    """``comp.predict`` in pure float32 or float64: ``(published dict of floats, SED)``."""
    dtype = jnp.float32 if float32 else jnp.float64
    with jax.enable_x64(not float32):
        wave = jnp.asarray(WAVE, dtype=dtype)
        p = {k: jnp.asarray(v, dtype=dtype) for k, v in params.items()}
        sed, published = comp.predict(p, jnp.zeros_like(wave), wave)
        return {k: float(v) for k, v in published.items()}, np.asarray(sed, dtype=np.float64)


def relative_error(key, value32, value64):
    """Relative error of a float32 published value against float64.

    ``log_*`` keys are in dex, so the error is that of the luminosity, ``10**(d) - 1``. An
    absent component (``-inf`` in float64) must be ``-inf`` in float32 too.
    """
    if not np.isfinite(value64):
        return 0.0 if value32 == value64 else np.inf
    if not np.isfinite(value32):
        return np.inf
    if key.startswith("log_"):
        return abs(10.0 ** (value32 - value64) - 1.0)
    return abs(value32 / value64 - 1.0) if value64 != 0.0 else abs(value32)
