# SPDX-License-Identifier: BSD-3-Clause
"""Each band-table provider's table carries the CORRECT filter convention (#3505, M1a).

The sweep (``test_band_projection_sweep.py``) bounds the WavePrecomp photometry against the
exact path at 6e-3, and its canary shows that a provider's tables DIFFER between the photon and
energy conventions. Bessell and energy photometry differ by only ~1e-3 on a mildly sloped SED, so
a provider that applied the wrong convention inside its table could pass both. Two checks close
that gap:

* the model-level check: a model in which the provider's emitter is steep across a band, so the
  exact Bessell and energy photometry of the emitter differ by at least 1e-2. The emitter's own
  contribution is the photometry of the model with the emitter minus that of the same model
  without it (the stellar part cancels, on the LUT path and on the exact path alike). Per
  convention ``c`` the LUT contribution must sit within ``tol`` of the exact contribution under
  ``c`` and at least ``3 tol`` away from the exact contribution under the other convention;
* the node-level check: the table entries at the grid nodes equal ``project_bands`` of the
  emitter's shape on the same basis at rtol 1e-6 (nodes need no interpolation).

The providers are enumerated from ``BAND_TABLE_PROVIDERS``; a registry entry without a case here
fails ``test_every_provider_has_a_case``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cache
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.forward.precompute.band_table import BandTable
from tengri.forward.precompute.registry import BAND_TABLE_PROVIDERS, resolve_band_table_provider
from tengri.observation.photometry import FilterConvention, FilterCurve
from tengri.utils.band_projection import project_bands
from tests._cloudy_grid import synthetic_ionizing_ssp
from tests.contract import _band_table_cases as cases

pytestmark = pytest.mark.contract

BESSELL, ENERGY = FilterConvention.BESSELL, FilterConvention.ENERGY
CONVENTIONS = (BESSELL, ENERGY)
_OTHER = {BESSELL: ENERGY, ENERGY: BESSELL}

#: The exact Bessell and energy photometry of the emitter must differ by at least this much in
#: every band of a case, or the case cannot tell the two conventions apart.
MIN_SEPARATION = 1e-2

#: Node-level agreement of a table entry with ``project_bands`` of the same basis.
NODE_RTOL = 1e-6

_FIXED_SFH = {"type": "dpl", "all_params": Fixed(DEFAULT)}
_DUST = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
_NO_RADIO = {"sf": {"type": "none"}, "agn": {"type": "none"}}


def _curve(lo: float, hi: float, name: str, n: int = 60) -> FilterCurve:
    """A sine-shaped band from ``lo`` to ``hi`` [Angstrom]."""
    wave = jnp.linspace(lo, hi, n)
    return FilterCurve(wave=wave, trans=jnp.sin(jnp.linspace(0.0, jnp.pi, n)), name=name)


def _wide(center: float, name: str) -> FilterCurve:
    """A band spanning 0.5 center either side, so a steep emitter shows its slope."""
    return _curve(center * 0.5, center * 1.5, name)


def _bands_from_names(*names):
    return lambda convention: Photometry.from_names(list(names), convention=convention)


def _bands_from_curves(*curves):
    return lambda convention: Photometry(filters=tuple(curves), convention=convention)


@dataclass(frozen=True)
class _Case:
    """One steep-emitter model per provider slot.

    Parameters
    ----------
    provider : str
        Registry key of the provider the case engages.
    slot : str
        The band-table slot that must hold a built table in the LUT model.
    ssp : callable
        Returns the SSP data.
    photometry : callable
        ``photometry(convention)`` returns the observation's ``Photometry``.
    groups : dict
        Group kwargs of the model with the emitter.
    off : dict
        Group kwargs that replace the emitter's in the model without it.
    base : dict
        Group kwargs shared by both models.
    tol : float
        Measured LUT error of this provider's table against the exact path, with margin.
    per_unit_power : bool
        The table's amplitude is a power computed elsewhere (the dust emission's absorbed
        stellar energy), so the table is compared per unit of that power: each band after
        the first, divided by the first.
    """

    provider: str
    slot: str
    ssp: Any
    photometry: Any
    groups: dict
    off: dict
    base: dict
    tol: float
    per_unit_power: bool = False

    def model(self, convention, approx, *, emitter=True, redshift=None):
        groups = {**self.base, **(self.groups if emitter else self.off)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SEDModel.build(
                ssp_data=self.ssp(),
                observation=Observation(photometry=self.photometry(convention)),
                approx=approx,
                redshift=Fixed(0.1) if redshift is None else redshift,
                **groups,
            )


def _agn(disc: str, slot: str, tol: float) -> _Case:
    bands = (_wide(1.2e5, "ir12"), _wide(2.5e5, "ir25"), _wide(8.0e5, "ir80"))
    return _Case(
        provider="agn",
        slot=slot,
        ssp=cases._synthetic_agn_ssp,
        photometry=_bands_from_curves(*bands),
        groups={
            "agn": {
                "type": "composable",
                "all_params": Fixed(DEFAULT),
                "disc": {"type": disc},
                "torus": {"type": "simple"},
                "agn_log_lbol": Fixed(13.0),
            }
        },
        off={"agn": {"type": "none"}},
        base={
            "sfh": {**_FIXED_SFH, "log_total_mass": Fixed(0.0)},
            "dust_attenuation": {"type": "none"},
            "neb": {"type": "none"},
        },
        tol=tol,
    )


#: Each ``tol`` is the measured |LUT - exact| / exact of that provider's emitter contribution
#: (worst band, both conventions, stated in the comment above it) rounded up to the next
#: decade; every one is far below the LUT's own 6e-3 bound, because the contribution is the
#: emitter's table alone.
CASES: dict[str, _Case] = {
    # dust emission: 1.4e-8 (wise_w4 over wise_w3, per unit L_ir). The absolute contribution
    # also carries the power, which is tracked by ``test_dust_emission_absolute_agreement``.
    "dust_emission": _Case(
        provider="dust_emission",
        slot="emission_band_response",
        ssp=cases._bare_ssp,
        photometry=_bands_from_names("wise_w3", "wise_w4"),
        groups={"dust_emission": {"type": "dale2014", "all_params": Fixed(DEFAULT)}},
        off={"dust_emission": {"type": "none"}},
        base={
            "sfh": _FIXED_SFH,
            "dust_attenuation": _DUST,
            "neb": {"type": "none"},
        },
        tol=1e-7,
        per_unit_power=True,
    ),
    # composable AGN, power-law disc: term_band_response, 8.3e-7 (the stellar remainder at the
    # SFH's unit mass).
    "agn_term": _agn("powerlaw", "term_band_response", 1e-5),
    # composable AGN, multicolor disc: band_table over the luminosity axis, 1.8e-6.
    "agn_table": _agn("multicolor", "band_table", 1e-5),
    # CLOUDY grid nebular (continuum and lines): 1.3e-14.
    "nebular": _Case(
        provider="nebular",
        slot="band_table",
        ssp=synthetic_ionizing_ssp,
        photometry=_bands_from_curves(
            _curve(6800.0, 14000.0, "red"),
            _curve(3500.0, 9000.0, "opt"),
            _curve(9000.0, 20000.0, "nir"),
        ),
        groups={
            "neb": {
                "type": "cloudy",
                "all_params": Fixed(DEFAULT),
                "grid": cases._synthetic_cloudy_grid(),
            }
        },
        off={"neb": {"type": "none"}},
        base={"sfh": _FIXED_SFH, "dust_attenuation": {"type": "none"}},
        tol=1e-9,
    ),
    # MAPPINGS shock lines: 6.9e-15.
    "shock": _Case(
        provider="shock",
        slot="band_table",
        ssp=cases._bare_ssp,
        photometry=_bands_from_curves(
            _curve(6800.0, 14000.0, "red"),
            _curve(3500.0, 9000.0, "opt"),
            _curve(9000.0, 20000.0, "nir"),
        ),
        groups={
            "shock": {
                "norm": "lhalpha",
                "log_lhalpha": Fixed(41.0),
                "velocity": Fixed(400.0),
                "log_density": Fixed(1.0),
            }
        },
        off={},
        base={
            "sfh": {**_FIXED_SFH, "log_total_mass": Fixed(0.0)},
            "dust_attenuation": {"type": "none"},
            "neb": {"type": "none"},
        },
        tol=1e-9,
    ),
    # radio (SF synchrotron and AGN jet), 1.4 GHz-scale band: 0 (bit-identical).
    "radio": _Case(
        provider="radio",
        slot="term_band_response",
        ssp=cases._synthetic_agn_ssp,
        photometry=_bands_from_curves(_curve(1.0e8, 4.0e8, "radio")),
        groups={
            "radio": {
                "sf": {"type": "bell2003"},
                "agn": {"type": "powerlaw"},
                "all_params": Fixed(DEFAULT),
            }
        },
        off={"radio": _NO_RADIO},
        base={"sfh": _FIXED_SFH, "dust_attenuation": _DUST, "neb": {"type": "none"}},
        tol=1e-9,
    ),
    # X-ray, 2-12 Angstrom band: 0 (bit-identical).
    "xray": _Case(
        provider="xray",
        slot="term_band_response",
        ssp=cases._synthetic_agn_ssp,
        photometry=_bands_from_curves(_curve(2.0, 12.0, "xray")),
        groups={"xray": {"type": "simple", "all_params": Fixed(DEFAULT)}},
        off={"xray": {"type": "none"}},
        base={
            "sfh": _FIXED_SFH,
            "dust_attenuation": {"type": "none"},
            "neb": {"type": "none"},
        },
        tol=1e-9,
    ),
}

#: Slots no steep-emitter case engages, each with the reason. A slot of a registered provider
#: that is in neither ``CASES`` nor here fails ``test_every_slot_is_covered_or_declared``.
_UNCOVERED_SLOTS = {
    ("dust_emission", "emission_shape_table"): "built only with a free emission shape",
    ("agn", "term_shape_table"): "built only with a free AGN shape parameter",
    ("radio", "term_shape_table"): "built only with a free radio shape parameter",
    ("xray", "term_shape_table"): "built only with a free X-ray shape parameter",
}


@cache
def _photometry(name: str) -> dict:
    """Emitter contribution per path and convention: ``{(exact?, convention): array}``."""
    case = CASES[name]
    out = {}
    for approx in (None, WavePrecomp()):
        for convention in CONVENTIONS:
            on = case.model(convention, approx)
            off = case.model(convention, approx, emitter=False)
            if approx is not None:
                entry = on._band_table(case.provider, case.slot)
                assert isinstance(entry, BandTable), (
                    f"{name}: slot {case.slot} was declined: {entry}"
                )
            phot_on = np.asarray(on.predict_photometry({}), dtype=np.float64)
            phot_off = np.asarray(off.predict_photometry({}), dtype=np.float64)
            out[(approx is None, convention)] = phot_on - phot_off
    return out


def _per_unit_power(contribution):
    """Bands after the first over the first: the table's shape with the power divided out."""
    return contribution[1:] / contribution[0]


def test_every_provider_has_a_case():
    assert {case.provider for case in CASES.values()} == set(BAND_TABLE_PROVIDERS)


def test_every_slot_is_covered_or_declared():
    declared = set(_UNCOVERED_SLOTS)
    covered = {(case.provider, case.slot) for case in CASES.values()}
    every = {
        (provider, slot)
        for provider in BAND_TABLE_PROVIDERS
        for slot in resolve_band_table_provider(provider).BAND_SLOTS
    }
    assert every == covered | declared
    assert not covered & declared


@pytest.mark.parametrize("name", list(CASES))
def test_the_emitter_separates_the_conventions(name):
    exact = _photometry(name)
    bessell, energy = exact[(True, BESSELL)], exact[(True, ENERGY)]
    assert np.all(bessell != 0.0), f"{name}: the emitter adds nothing in a band: {bessell}"
    separation = np.abs(bessell / energy - 1.0)
    assert np.all(separation >= MIN_SEPARATION), (
        f"{name}: Bessell-vs-energy exact separation per band {separation}"
    )


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("convention", CONVENTIONS, ids=lambda c: c.value)
def test_the_table_carries_the_right_convention(name, convention):
    case = CASES[name]
    tol = case.tol
    phot = _photometry(name)
    view = _per_unit_power if case.per_unit_power else (lambda a: a)
    lut = view(phot[(False, convention)])
    own = view(phot[(True, convention)])
    other = view(phot[(True, _OTHER[convention])])
    to_own = np.abs(lut / own - 1.0)
    to_other = np.abs(lut / other - 1.0)
    assert np.all(to_own <= tol), f"{name}/{convention.value}: LUT vs own convention {to_own}"
    assert np.all(to_other >= 3.0 * tol), (
        f"{name}/{convention.value}: LUT vs the other convention {to_other}"
    )


@pytest.mark.parametrize("convention", CONVENTIONS, ids=lambda c: c.value)
def test_dust_emission_absolute_agreement(convention):
    """The absolute dust-emission contribution of the table equals the exact path (1e-7)."""
    phot = _photometry("dust_emission")
    gap = np.abs(phot[(False, convention)] / phot[(True, convention)] - 1.0)
    assert np.all(gap <= CASES["dust_emission"].tol), f"{convention.value}: {gap}"


# ------------------------------------------------------------------------------- node level


def _table_context(model):
    return model._band_table_context(model._cached_component_chain)


def _wavepre_model(name, convention, redshift):
    return CASES[name].model(convention, WavePrecomp(), redshift=redshift)


@pytest.mark.parametrize("name", ["agn_term", "radio", "xray"])
@pytest.mark.parametrize("convention", CONVENTIONS, ids=lambda c: c.value)
@pytest.mark.parametrize("redshift", [Fixed(0.1), Uniform(0.05, 0.3)], ids=["fixed", "free"])
def test_term_response_nodes_equal_project_bands(name, convention, redshift):
    """``R`` at each redshift node is ``project_bands`` of each term's shape (rtol 1e-6)."""
    case = CASES[name]
    model = _wavepre_model(name, convention, redshift)
    entry = model.band_table(case.provider, case.slot)
    assert entry is not None, f"{name}: no {case.slot} table"
    table = entry.arrays
    ctx = _table_context(model)
    comp = next(c for c in ctx.chain if getattr(c, "name", "") == case.provider)
    fixed = {k: jnp.asarray(v) for k, v in ctx.spec.get_fixed_values().items()}
    probe = next(iter(comp.EMITTER_PROBE_INPUTS))
    nodes = np.asarray(table["ln1pz"])
    assert ctx.bands.convention is convention
    live = False
    for i, ln1pz in enumerate(nodes):
        z = float(np.expm1(ln1pz))
        terms = comp.emission_terms({**fixed, "redshift": jnp.asarray(z)}, ctx.wave, **probe)
        for k, shape in enumerate(terms.values()):
            expected = np.asarray(project_bands(shape, ctx.wave, ctx.bands, z))
            got = np.asarray(table["R"][i, k])
            np.testing.assert_allclose(got, expected, rtol=NODE_RTOL, atol=0.0)
            live |= bool(np.any(expected != 0.0))
    assert live, f"{name}: every term projects to zero, so the check is vacuous"
    assert (nodes.shape[0] == 1) == isinstance(redshift, Fixed)


@pytest.mark.parametrize("convention", CONVENTIONS, ids=lambda c: c.value)
def test_shock_line_coefficients_equal_project_bands(convention):
    """Each line's observed and rest band coefficient is ``project_bands`` of its unit line."""
    from tengri.components.nebular._shared import render_nebular_lines
    from tengri.components.nebular.mappings_shock_precompute import _load_mappings_grids

    model = _wavepre_model("shock", convention, None)
    entry = model.band_table("shock", "band_table")
    assert entry is not None
    coeff = np.asarray(entry.arrays)
    ctx = _table_context(model)
    line_waves = jnp.asarray(_load_mappings_grids()["mappings5"]["line_wavelengths_aa"])
    wave = jnp.asarray(ctx.wave)
    n_lines = int(line_waves.shape[0])
    lit = [i for i in range(n_lines) if np.any(coeff[i] != 0.0)]
    assert lit, "no line reaches a band"
    for i in lit[:: max(1, len(lit) // 8)]:
        unit = jnp.zeros(n_lines).at[i].set(1.0)
        sed = render_nebular_lines(line_waves, unit, wave, 0.0, 0.0)
        for j, z in enumerate((0.1, 0.0)):
            expected = np.asarray(project_bands(sed, wave, ctx.bands, z))
            np.testing.assert_allclose(coeff[i, j], expected, rtol=NODE_RTOL, atol=0.0)
