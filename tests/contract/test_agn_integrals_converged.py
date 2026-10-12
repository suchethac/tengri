# SPDX-License-Identifier: BSD-3-Clause
"""Published AGN luminosities are converged on their own native grids.

Each published ``erg/s`` key of an AGN template or analytic model is an integral that the model
evaluates on a fixed internal grid (a publication grid, a normalization band, a template's own
nodes, or a closed form on those nodes). None of them may depend on the caller's wavelength
array, and each must be converged on its own grid: the value must agree, to 1e-6 relative,
with the same integral taken on a grid refined by at least 20x. The refined evaluation is a
second predict call with the internal grid replaced by its refinement, so the integrand is
evaluated on the refined grid and not interpolated from the native one. Template-backed tori
integrate in closed form on their native nodes (log-log between nodes), so their refinement
inserts nodes inside each segment and must not move the value.

Monochromatic diagnostics (erg/s/Hz) are not integrals and are not checked here.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import (
    disc as _disc,
    kd18_disc_model,
    powerlaw_disc_model,
    skirtor_model,
)
from tengri.forward.orchestrator import default_params_dict

pytestmark = pytest.mark.contract

#: Agreement required between a published luminosity and its refined-grid value [relative].
_RTOL = 1.0e-6
#: Refinement factor of the reference grid (node count of the reference over the native one).
_REFINE = 20

#: Caller grid for predict. The published values do not depend on it (checked by the ledger test).
_CALLER_WAVE = np.geomspace(1.0e3, 1.0e7, 600)

_DATA = Path(__file__).resolve().parents[2] / "data"

#: Parameter points per model: "defaults" and one perturbed point (bare names, no prefix).
_POINTS = {
    "cat3d_wind": {
        "defaults": {},
        "perturbed": {"cos_inc": 0.4, "a_cat3d": -1.5, "torus_frac": 0.3},
    },
    "kd18_disc": {
        "defaults": {},
        "perturbed": {"log_lbol": 11.5, "cos_inc": 0.4, "log_mbh": 8.5, "kt_hot": 30.0},
    },
    "powerlaw_disc": {"defaults": {}, "perturbed": {"alpha": -1.7, "T_max": 3.0e4}},
    "silva04": {
        "defaults": {},
        "perturbed": {"log_lbol": 11.0, "log_nh_silva": 24.0, "torus_frac": 0.3},
    },
    "skirtor": {
        "defaults": {},
        "perturbed": {
            "cos_inc": 0.4,
            "oa_skirtor": 60.0,
            "tau_skirtor": 9.0,
            "polar_ebv": 0.1,
            "torus_frac": 0.3,
        },
    },
    "skirtor_agnfitter": {
        "defaults": {},
        "perturbed": {
            "oa_skirtor": 20.0,
            "incl_skirtor": 60.0,
            "tv_skirtor": 5.0,
            "torus_frac": 0.3,
        },
    },
}

#: Monochromatic diagnostics (erg/s/Hz, node values of the publication grid): not integrals.
_MONOCHROMATIC = frozenset({"L_2500_30deg", "L_6um", "L_12um"})

_CASES = [(model, point) for model in sorted(_POINTS) for point in ("defaults", "perturbed")]


def _make_component(name):
    """One SEDModelComponent that reads its template library from ``data/``."""
    from tengri.components.agn.cat3d_torus_model import CAT3DTorus, CAT3DTorusConfig
    from tengri.components.agn.kd18_disc_model import KD18Disc
    from tengri.components.agn.powerlaw_disc_model import PowerLawDisc
    from tengri.components.agn.silva04_model import Silva04Torus, Silva04TorusConfig
    from tengri.components.agn.skirtor_agnfitter_model import (
        SKIRTORAgnfitterTorus,
        SKIRTORAgnfitterTorusConfig,
    )
    from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig

    makers = {
        "cat3d_wind": lambda: CAT3DTorus(
            config=CAT3DTorusConfig(grid_path=str(_DATA / "cat3d_wind_torus_grid.h5"))
        ),
        "kd18_disc": KD18Disc,
        "powerlaw_disc": PowerLawDisc,
        "silva04": lambda: Silva04Torus(
            config=Silva04TorusConfig(grid_path=str(_DATA / "silva04_torus_grid.h5"))
        ),
        "skirtor_agnfitter": lambda: SKIRTORAgnfitterTorus(
            config=SKIRTORAgnfitterTorusConfig(
                grid_path=str(_DATA / "skirtor_mean3p_torus_grid.h5")
            )
        ),
        "skirtor": lambda: SKIRTORTorus(
            config=SKIRTORTorusConfig(grid_path=str(_DATA / "skirtor_templates_v3.h5"))
        ),
    }
    return makers[name]()


def _load(model):
    """Component with its template data loaded; skips only when the data file is absent."""
    comp = _make_component(model)
    try:
        data = comp.load(jnp.asarray(_CALLER_WAVE))
    except (FileNotFoundError, OSError) as err:
        pytest.skip(f"{model}: template data unavailable ({err})")
    object.__setattr__(comp, "data", data)
    return comp


def _published_erg(comp, overrides):
    """Published erg/s luminosities of ``comp`` at its defaults with ``overrides`` applied."""
    prefix = comp.parameter_prefix
    wave = jnp.asarray(_CALLER_WAVE)
    p = {k.removeprefix(prefix): v for k, v in default_params_dict([comp]).items()}
    for key, value in overrides.items():
        p[key] = jnp.asarray(value, dtype=wave.dtype)
    _, published = comp.predict(p, jnp.zeros_like(wave), wave)
    return {
        k: float(np.asarray(v))
        for k, v in published.items()
        if k.startswith("L_") and k not in _MONOCHROMATIC
    }


def _refine_nodes(nodes, factor=_REFINE):
    """Insert ``factor - 1`` geometric points inside every segment of ascending ``nodes``."""
    nodes = np.asarray(nodes, dtype=np.float64)
    pieces = [nodes]
    for lo, hi in itertools.pairwise(nodes):
        pieces.append(np.geomspace(lo, hi, factor + 1)[1:-1])
    return np.unique(np.concatenate(pieces))


def _refine_internal(model, comp, monkeypatch):
    """Replace each model's internal integration grid by a ``_REFINE`` times denser one."""
    if model == "kd18_disc":
        n_pub = _REFINE * kd18_disc_model._KD18_PUBLICATION_NODES
        monkeypatch.setattr(kd18_disc_model, "_KD18_PUBLICATION_NODES", n_pub)
        monkeypatch.setattr(_disc, "_CORONA_NU_NODES", _REFINE * _disc._CORONA_NU_NODES)
    elif model == "powerlaw_disc":
        monkeypatch.setattr(
            powerlaw_disc_model,
            "_PUBLISHED_BAND_NODES",
            _REFINE * powerlaw_disc_model._PUBLISHED_BAND_NODES,
        )
        monkeypatch.setattr(_disc, "_PL_BAND_NODES", _REFINE * _disc._PL_BAND_NODES)
    elif model == "skirtor":
        native_pub = skirtor_model._native_publication_wave

        def refined_pub(native, breaks, dtype):
            base = np.asarray(native_pub(native, breaks, dtype), dtype=np.float64)
            return jnp.asarray(_refine_nodes(base), dtype=dtype)

        monkeypatch.setattr(skirtor_model, "_native_publication_wave", refined_pub)
        # The polar absorbed power is Gauss-Legendre per segment; the reference uses 8 points.
        monkeypatch.setattr(skirtor_model, "_POLAR_GL_ORDER", 8)
    else:
        # Closed form on the template's own nodes: the reference integrates the same closed form
        # on a refined node set.
        monkeypatch.setattr(
            comp.data, "native_wave", _refine_nodes(comp.data.native_wave), raising=True
        )


@pytest.mark.parametrize(("model", "point"), _CASES)
def test_published_luminosity_matches_refined_grid(model, point, monkeypatch):
    """Each published erg/s key equals its value on a 20x refined internal grid to 1e-6."""
    comp = _load(model)
    native = _published_erg(comp, _POINTS[model][point])
    _refine_internal(model, comp, monkeypatch)
    refined = _published_erg(comp, _POINTS[model][point])

    assert native, f"{model}: no erg/s keys published"
    failures = []
    for key in sorted(native):
        ref = refined[key]
        assert ref > 0.0, f"{model} {key}: non-positive reference {ref:.3e}"
        rel = abs(native[key] / ref - 1.0)
        if rel >= _RTOL:
            failures.append(
                f"{key}: native {native[key]:.12e} vs refined {ref:.12e}, rel {rel:.3e}"
            )
    assert not failures, f"{model} ({point}) not converged to {_RTOL:.0e}: " + "; ".join(failures)


def test_polar_absorbed_power_is_caller_grid_independent():
    """The polar absorbed power (``blocks/atten.py``) does not depend on the caller's grid.

    The absorbed power is the trapezoid of the per-bin absorbed profile over the caller's
    reference wavelengths. On a 300-point caller grid it sits 1e-5 from the 6000-point value
    (the log-log rule is 1.1e-4 off there), so the published absorbed power depends on how the
    caller samples the spectrum. This is a known open site (#2881): the fix is a fixed internal
    integration grid, which is not in place, so this test fails until it is.
    """
    from tengri.components.agn.blocks.atten import polar_dust_reemission_lnu

    def absorbed(n):
        wave = np.geomspace(912.0, 1.0e5, n)
        l_in = (wave / 5100.0) ** -0.5 * 1.0e43
        _, total = polar_dust_reemission_lnu(
            jnp.asarray(wave),
            jnp.asarray(l_in),
            return_absorbed=True,
            agn_polar_ebv=0.3,
            agn_cos_inc=0.5,
            agn_polar_oa=60.0,
        )
        return float(np.asarray(total))

    reference = absorbed(6000)
    coarse = absorbed(300)
    rel = abs(coarse / reference - 1.0)
    assert rel < _RTOL, (
        f"polar absorbed power on a 300-point caller grid differs from the 6000-point value by "
        f"{rel:.3e} (tolerance {_RTOL:.0e}); caller-grid dependence, #2881"
    )
