# SPDX-License-Identifier: BSD-3-Clause
"""#1718: FeaturePrecomp could not build against a tabulated SFH, in any met mode.

Both nebular precompute builders take their reference point from
``spec.sample(...)`` and then run a full forward pass per grid node. A tabulated
SFH declares **zero** parameters — the table *is* the SFH — so no prior can
produce ``sfh_t_gyr`` / ``sfh_sfr`` and the stellar component raised the #996
guard on the first row. ``WavePrecomp`` and ``SpectrumPrecomp`` were unaffected.

A stand-in SFH is legitimate because both tables store luminosity **per ionizing
photon** and divide Q_H back out, which ``line_precompute`` states in its own
docstring: "a property of the gas, independent of the reference SFH". The LUT is
already built at one ``spec.sample`` draw and reused for every SFH a fit visits,
so parametric models rely on the same invariance.

Making it merely *run* was the trap. The grid axes are the model's **free
parameters**, so ``met_mode='table'`` removes ``met_logzsol`` and the metallicity
axis disappears with nothing raised — the table is then built at one reference
metallicity. Measured against exact on a history spanning Z = -2.1 to +0.4:
OIII_5007 17.5% wrong, NII_6584 5.3%. That is refused, not warned about.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.regression_bug]


def _find_data_file(name):
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / name
        if candidate.is_file():
            return candidate
    return None


# Cue needs its trained weights and a bare-stellar SSP; both are gitignored, so
# the model-level arms skip on CI and run wherever the grids are present.
requires_cue = pytest.mark.skipif(
    _find_data_file("cue_weights.npz") is None,
    reason="Cue weights not found in any parent data/",
)

_N_T = 40
_T = np.linspace(0.0, 12.5, _N_T)


def _history(tau=2.0):
    sfr = (_T / tau) * np.exp(-_T / tau)
    sfr = sfr * (40.0 / sfr.max())
    cum = np.cumsum(sfr)
    cum /= cum[-1]
    return sfr, np.clip(-2.6 + 2.7 * cum, -2.1, 0.4)


# ── the stand-in itself, no model needed ─────────────────────────────


class _Cfg:
    """Only the two fields the stand-in reads."""

    def __init__(self, sfh_model, metallicity_model="delta"):
        self.sfh_model = sfh_model
        self.metallicity_model = metallicity_model


def test_a_parametric_model_gets_no_stand_in():
    """``{}`` for anything that is not tabulated, so callers merge unconditionally."""
    from tengri.components.stellar.reference_history import reference_history_for_config

    assert reference_history_for_config(_Cfg("dpl")) == {}
    assert reference_history_for_config(None) == {}


def test_the_stand_in_is_a_constant_sfh_inside_cosmic_time():
    """Anchored at t=0 and ending at the cosmic age, so no mass predates the Big Bang."""
    from tengri.components.stellar.reference_history import reference_history_for_config
    from tengri.utils.cosmology import age_at_z

    out = reference_history_for_config(_Cfg("table"), redshift=0.1, n_nodes=32)
    t = np.asarray(out["sfh_t_gyr"])
    sfr = np.asarray(out["sfh_sfr"])
    assert t.shape == sfr.shape == (32,)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(float(age_at_z(0.1)), rel=1e-9)
    assert np.all(np.diff(t) > 0), "the component requires strictly increasing time"
    assert sfr[0] == 0.0, "anchored so nothing extrapolates past the Big Bang"
    assert np.all(sfr[1:] == sfr[1]), "constant by construction"
    assert "met_history" not in out, "no Z table was asked for"


def test_a_tabulated_metallicity_also_gets_a_stand_in_on_the_same_axis():
    """Z(t) shares the SFH's node axis by contract (#996), so the lengths must match."""
    from tengri.components.stellar.reference_history import reference_history_for_config

    out = reference_history_for_config(_Cfg("table", "table"), n_nodes=16)
    assert np.asarray(out["met_history"]).shape == np.asarray(out["sfh_t_gyr"]).shape


# ── through the real builders ────────────────────────────────────────


def _build(ssp, obs, *, met_mode, approx):
    from tengri import FIXED, SEDModel
    from tengri.parameters.priors import Fixed, Uniform

    met = {"type": "table"} if met_mode == "table" else {"logzsol": Uniform(-2.0, 0.4)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp,
            observation=obs,
            sfh={"type": "table"},
            met=met,
            neb={
                "type": "cue",
                "all_params": FIXED,
                "logU": Uniform(-4.0, -1.0),
                "logZ_gas": Uniform(-1.5, 0.3),
            },
            dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
            redshift=Fixed(0.1),
            approx=approx,
        )


@pytest.fixture(scope="module")
def line_obs():
    import jax.numpy as jnp

    from tengri.observation import Observation
    from tengri.observation.line_list import LineList
    from tengri.observation.photometry_config import Photometry

    try:
        from tengri.observation.line_flux_data import LineFluxData
    except ImportError:  # pragma: no cover - layout drift
        from tengri.observation.data import LineFluxData

    wanted = ["Hbeta", "OIII_5007", "Halpha", "NII_6584"]
    cat = LineList.default_optical()
    waves = jnp.asarray([float(w) for n, w in zip(cat.names, cat.wavelengths) if n in wanted])
    names = tuple(n for n in cat.names if n in wanted)
    f = jnp.ones(len(names)) * 1e-16
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
        line_fluxes=LineFluxData(names=names, fluxes=f, errors=f * 0.1, wavelengths=waves),
    )
    return obs, waves


@requires_cue
def test_feature_precomp_builds_against_a_tabulated_sfh(line_obs):
    """The defect: this raised the #996 guard before building a single row."""
    import tengri
    from tengri import FeaturePrecomp, WavePrecomp

    obs, _ = line_obs
    _build(tengri.load_ssp(), obs, met_mode="delta", approx=(WavePrecomp(), FeaturePrecomp()))


@requires_cue
def test_the_lut_agrees_with_the_exact_line_path_on_a_tabulated_sfh(line_obs):
    """Building is not enough — the table has to be right, across SFH shapes.

    The stand-in is one constant-SFR history; these are delayed exponentials from
    burst-like to near-constant. Agreement across them is what shows the per-Q_H
    invariance the stand-in relies on actually holds.
    """
    import jax.numpy as jnp

    import tengri
    from tengri import FeaturePrecomp, WavePrecomp
    from tengri.parameters.resolve import resolve_fixed_params

    obs, waves = line_obs
    ssp = tengri.load_ssp()
    exact = _build(ssp, obs, met_mode="delta", approx=(WavePrecomp(),))
    lut = _build(ssp, obs, met_mode="delta", approx=(WavePrecomp(), FeaturePrecomp()))

    worst = 0.0
    for tau in (0.5, 2.0, 6.0):
        sfr, _ = _history(tau)
        p = {
            "neb_logU": jnp.asarray(-2.5),
            "neb_logZ_gas": jnp.asarray(-0.2),
            "met_logzsol": jnp.asarray(-0.3),
            "sfh_t_gyr": jnp.asarray(_T),
            "sfh_sfr": jnp.asarray(sfr),
        }
        a = np.asarray(
            exact.predict_line_fluxes(resolve_fixed_params(exact, p), target_wavelengths=waves)
        )
        b = np.asarray(
            lut.predict_line_fluxes(resolve_fixed_params(lut, p), target_wavelengths=waves)
        )
        worst = max(worst, float(np.abs(b / a - 1).max()))
    assert worst < 0.01, f"LUT departs from exact by {worst:.2%} (measured 0.35%)"


@requires_cue
def test_a_tabulated_metallicity_is_refused_rather_than_silently_flattened(line_obs):
    """The axis vanishes with no complaint, and metal lines go 17.5% wrong.

    Refused rather than warned: line ratios are the thing a nebular fit measures,
    so a quietly single-metallicity table is not a degraded answer, it is a wrong
    one. The message has to name the way out, because a tabulated SFH on its own
    is supported and the reader needs to know the metallicity is the blocker.
    """
    import tengri
    from tengri import FeaturePrecomp, WavePrecomp

    obs, _ = line_obs
    with pytest.raises(ValueError, match="cannot serve a tabulated metallicity") as exc:
        _build(tengri.load_ssp(), obs, met_mode="table", approx=(WavePrecomp(), FeaturePrecomp()))
    message = str(exc.value)
    assert "met_logzsol" in message, "name the parameter whose absence removes the axis"
    assert "17.5%" in message, "quote the measurement, so the guard is not read as caution"


@requires_cue
def test_wave_precomp_alone_still_serves_a_tabulated_metallicity(line_obs):
    """The refusal is scoped to the feature LUT and must not spill onto WavePrecomp."""
    import tengri
    from tengri import WavePrecomp

    obs, _ = line_obs
    _build(tengri.load_ssp(), obs, met_mode="table", approx=(WavePrecomp(),))
