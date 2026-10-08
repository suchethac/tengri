"""The AGN energy ledger closes: each published luminosity is the power of its spectrum.

Every AGN component that publishes an ``L_*`` integrates the same SED it emits, on a
fixed budget grid, so the value depends on the component's parameters and not on the
caller's wavelength array. Two contracts follow and both are checked here:

(a) the spectrum integrated on a converged grid (four times the budget density) equals
    the published luminosity it describes, to 1e-4;
(b) the published luminosity is the same on a coarse and a fine caller grid, to 1e-6.

The composable-runner discs (kubota_done, ADAF) are not in this module: their
luminosities come from the runner's ledger, which is checked there.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn._phys import bolometric_integral_nu, wavelength_to_nu
from tengri.components.agn._template_grid import BUDGET_WAVE
from tengri.components.agn.grahsp import GRAHSPSEDComponent, GRAHSPSEDComponentConfig
from tengri.forward.orchestrator import default_params_dict
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.contract

_LEDGER_RTOL = 1.0e-4
_GRID_RTOL = 1.0e-6

#: Converged own grid: four times the density of the budget grid [Angstrom].
_CONVERGED = np.geomspace(BUDGET_WAVE[0], BUDGET_WAVE[-1], 4 * BUDGET_WAVE.size)
_L_COARSE = np.geomspace(1.0e3, 1.0e7, 60)  # 0.1-1000 um [Angstrom]
_L_FINE = np.geomspace(1.0e3, 1.0e7, 6000)

_DATA = Path(__file__).resolve().parents[2] / "data"

#: Known failures, keyed by ``case: measured``. Each entry is xfail(strict=True), so the
#: case starts passing and the entry must be removed once the leak is closed.
KNOWN_LEAKS = {
    "grahsp_bbb": (
        "L_agn_bol is the BBB above the Lyman limit, a step on a 13001-node budget grid: "
        "the converged value is 4.1e-4 away (measured -4.14e-4 at 8x density)"
    ),
}

_TORUS_KEYS = ("L_agn_torus",)
_DISC_KEYS = ("L_agn_disc",)

#: Component -> (published keys whose sum the spectrum integrates to).
_SEDMODEL_LEDGER = {
    "cat3d_wind": _TORUS_KEYS,
    "kd18_disc": _DISC_KEYS,
    "powerlaw_disc": _DISC_KEYS,
    "silva04": _TORUS_KEYS,
    "skirtor_agnfitter": _TORUS_KEYS,
    # The spectrum is the attenuated disc plus torus plus polar re-emission. The disc
    # and torus sum is the closure target; the re-emission returns the absorbed power.
    "skirtor": ("L_agn_disc", "L_agn_torus"),
}

_GRAHSP_PARAMS = {
    "agn_grahsp_l5100": jnp.array(1.0e44),
    "agn_grahsp_uvslope": jnp.array(0.0),
    "agn_grahsp_plslope": jnp.array(-1.7),
    "agn_grahsp_plbendloc_nm": jnp.array(100.0),
    "agn_grahsp_plbendwidth": jnp.array(1.0),
    "agn_grahsp_cutoff_nm": jnp.array(10000.0),
    "agn_grahsp_a_lines": jnp.array(1.0),
    "agn_grahsp_a_feii": jnp.array(5.0),
    "agn_grahsp_linewidth_kms": jnp.array(5000.0),
    "agn_grahsp_fcov": jnp.array(0.4),
    "agn_grahsp_si": jnp.array(0.0),
    "agn_grahsp_cool_lam_um": jnp.array(17.0),
    "agn_grahsp_cool_width": jnp.array(0.45),
    "agn_grahsp_hot_lam_um": jnp.array(2.0),
    "agn_grahsp_hot_width": jnp.array(0.5),
    "agn_grahsp_hot_fcov": jnp.array(1.0),
    "agn_grahsp_ebv": jnp.array(0.05),
    "agn_grahsp_ebv_agn": jnp.array(0.05),
}


def _make_component(name):
    """Build one SEDModelComponent that reads its template library from ``data/``."""
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


def _predict(name, wave):
    """Spectrum and published L_* of one SEDModelComponent on ``wave``, at defaults."""
    comp = _make_component(name)
    wave = jnp.asarray(wave)
    try:
        data = comp.load(wave)
    except (FileNotFoundError, OSError) as err:
        pytest.skip(f"{name}: template data unavailable ({err})")
    object.__setattr__(comp, "data", data)
    prefix = comp.parameter_prefix
    p = {k.removeprefix(prefix): v for k, v in default_params_dict([comp]).items()}
    sed, published = comp.predict(p, jnp.zeros_like(wave), wave)
    return sed, {k: float(np.asarray(v)) for k, v in published.items() if k.startswith("L_")}


def _grahsp(wave, **config):
    """GRAHSP output on rest-frame ``wave`` [Angstrom] for one configuration."""
    comp = GRAHSPSEDComponent(config=GRAHSPSEDComponentConfig(**config))
    out = comp.apply(ForwardState(wave=jnp.asarray(wave)), _GRAHSP_PARAMS)
    return out


def _power(sed, wave, mask=None):
    """Bolometric power |int L_nu dnu| [erg/s] of ``sed`` on ``wave``, optionally masked."""
    w = jnp.asarray(wave)
    s = jnp.asarray(sed)
    if mask is not None:
        s = jnp.where(jnp.asarray(mask), s, 0.0)
    return float(np.abs(np.asarray(bolometric_integral_nu(s, wavelength_to_nu(w)))))


def _rel(a, b):
    return abs(a / b - 1.0)


@pytest.mark.parametrize("name", sorted(_SEDMODEL_LEDGER))
def test_spectrum_integrates_to_its_published_luminosity(name):
    """The spectrum on a converged grid carries the published power to 1e-4."""
    keys = _SEDMODEL_LEDGER[name]
    sed, published = _predict(name, _CONVERGED)
    target = sum(published[k] for k in keys)
    assert target > 0.0, f"{name}: non-positive published target {target:.3e}"
    spectrum = _power(sed, _CONVERGED)
    rel = _rel(spectrum, target)
    assert rel < _LEDGER_RTOL, (
        f"{name}: spectrum {spectrum:.9e} erg/s vs {' + '.join(keys)} = {target:.9e} erg/s, "
        f"rel {rel:.3e}"
    )


@pytest.mark.parametrize("name", sorted(_SEDMODEL_LEDGER))
def test_published_luminosity_is_grid_independent(name):
    """The published L_* on a coarse caller grid equals the fine-grid value to 1e-6."""
    _, coarse = _predict(name, _L_COARSE)
    _, fine = _predict(name, _L_FINE)
    assert fine, f"{name}: publishes no L_* luminosity"
    for key, ref in fine.items():
        assert ref > 0.0, f"{name}.{key}: non-positive reference {ref:.3e}"
        rel = _rel(coarse[key], ref)
        assert rel < _GRID_RTOL, f"{name}.{key}: coarse-vs-fine rel diff {rel:.3e}"


def test_grahsp_torus_spectrum_integrates_to_L_agn_torus():
    """GRAHSP torus alone (lines, FeII and Balmer off) carries L_agn_torus to 1e-4."""
    out = _grahsp(
        _CONVERGED,
        include_bbb=False,
        include_lines=False,
        include_feii=False,
        include_balmer=False,
        apply_attenuation=False,
    )
    target = float(out.derived["L_agn_torus"])
    spectrum = _power(out.sed_intrinsic, _CONVERGED)
    rel = _rel(spectrum, target)
    assert rel < _LEDGER_RTOL, f"grahsp torus: spectrum vs L_agn_torus rel {rel:.3e}"


@pytest.mark.xfail(strict=True, reason=KNOWN_LEAKS["grahsp_bbb"])
def test_grahsp_bbb_spectrum_integrates_to_L_agn_bol():
    """GRAHSP BBB, lines and FeII above 91.2 nm carry L_agn_bol to 1e-4."""
    out = _grahsp(
        _CONVERGED,
        include_torus=False,
        include_si=False,
        apply_attenuation=False,
    )
    target = float(out.derived["L_agn_bol"])
    spectrum = _power(out.sed_intrinsic, _CONVERGED, mask=_CONVERGED >= 912.0)
    rel = _rel(spectrum, target)
    assert rel < _LEDGER_RTOL, f"grahsp bbb: spectrum vs L_agn_bol rel {rel:.3e}"


def test_grahsp_published_luminosity_is_grid_independent():
    """GRAHSP L_* on a coarse caller grid equals the fine-grid value to 1e-6."""

    def _published(wave):
        out = _grahsp(wave)
        return {k: float(v) for k, v in out.derived.items() if k.startswith("L_")}

    coarse = _published(_L_COARSE)
    fine = _published(_L_FINE)
    assert fine, "grahsp publishes no L_* luminosity"
    for key, ref in fine.items():
        assert ref > 0.0, f"grahsp.{key}: non-positive reference {ref:.3e}"
        rel = _rel(coarse[key], ref)
        assert rel < _GRID_RTOL, f"grahsp.{key}: coarse-vs-fine rel diff {rel:.3e}"
