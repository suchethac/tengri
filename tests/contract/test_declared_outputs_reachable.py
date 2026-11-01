# SPDX-License-Identifier: BSD-3-Clause
"""Every user-facing output a component declares is reachable by its own name (#3144).

A component's ``outputs()`` is the orchestrator contract: it says what the component puts into
``state.derived``. The public property catalog (``pred.properties[...]``,
``model.available_properties``, ``predict_properties``) is a separate list, and before #3144
nothing connected the two, so a physical quantity an AGN, dust-emission or nebular component
computed was silently absent from the menu. A component now declares what it publishes in a
class-level ``published`` mapping, lowered into the same registry as ``properties``. This file
pins the three promises that follow:

* A. every published name a built model provides is in ``available_properties``, and
  ``predict_properties`` returns a finite value for it at a sampled point. The issue's named keys
  are required by name, so the check cannot pass vacuously on a component that publishes nothing.
* B. every name in a built chain's declared outputs is either served (``available_properties``),
  a composable ledger key (``COMPOSABLE_LEDGER_OUTPUTS``), or listed in ``_NOT_PUBLISHED`` with a
  reason. A new declared output therefore cannot slip past the menu unnoticed.
* C. for a composable AGN, the AGN entries of the catalog are exactly the AGN ``published`` names
  plus the ledger keys, with no name served twice.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.agn.blocks._ledger import COMPOSABLE_LEDGER_OUTPUTS

pytestmark = pytest.mark.contract

_SFH = {"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0)}
_DUST = {"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)}
_COMPOSABLE = {
    "type": "composable",
    "norm": "independent",
    "all_params": Fixed(DEFAULT),
    "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
}

#: One model per family the contract covers: stellar + nebular + dust (+ emission), each AGN
#: family (monolithic, composable kubota_done, composable ADAF), radio and X-ray.
MODELS: dict[str, dict] = {
    "stellar_nebular_dust_ir": {
        "neb": {"type": "cue", "all_params": Fixed(DEFAULT)},
        "dust_emission": {"type": "dale2014", "all_params": Fixed(DEFAULT)},
    },
    "agn_skirtor": {"agn": {"type": "skirtor", "all_params": Fixed(DEFAULT)}},
    "agn_composable_kubota_done": {
        "agn": {**_COMPOSABLE, "disc": {"type": "kubota_done", "all_params": Fixed(DEFAULT)}},
    },
    "agn_composable_adaf": {
        "agn": {**_COMPOSABLE, "disc": {"type": "adaf", "all_params": Fixed(DEFAULT)}},
    },
    "radio_xray": {
        "agn": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "radio": {"all_params": Fixed(DEFAULT)},
        "xray": {"type": "simple", "all_params": Fixed(DEFAULT)},
    },
}

#: Published names each model must serve, by name: the issue's keys and the dust/nebular
#: quantities a user reads. A model that drops one of these fails here, whatever the mechanism.
_REQUIRED: dict[str, frozenset[str]] = {
    "stellar_nebular_dust_ir": frozenset(
        {"log_L_ir_emergent", "L_ir", "log_L_ir", "sfr", "log_L_lyc", "dust_umean"}
    ),
    "agn_skirtor": frozenset(
        {"L_agn_bol", "log_L_agn_bol", "L_agn_bol_lsun", "log_L_6um", "log_L_12um"}
    ),
    "agn_composable_kubota_done": frozenset(
        {
            "L_agn_bol",
            "log_L_agn_bol",
            "L_agn_bol_lsun",
            "log_L_agn_direct_los",
            "L_agn_direct_los_lsun",
        }
    ),
    "agn_composable_adaf": frozenset(
        {
            "agn_adaf_mdot",
            "log_L_agn_adaf_accretion",
            "L_agn_adaf_accretion_lsun",
            "log_L_agn_direct_los",
        }
    ),
    "radio_xray": frozenset({"L_agn_bol", "log_L_agn_bol", "log_L_6um"}),
}

#: Declared outputs that stay off the menu on purpose. Each entry gives the reason. A name is
#: either a spectrum or array, an echo of an input, plumbing read by another component, a value
#: served under a different catalog name, or a retired name.
_NOT_PUBLISHED: dict[str, str] = {
    # Spectral arrays on the pipeline grid (reached through pred.components / pred.sed).
    "sed_agn": "spectral array on the pipeline grid",
    "sed_agn_disc": "spectral array (composable sub-block SED)",
    "sed_agn_torus": "spectral array (composable sub-block SED)",
    "sed_agn_lines": "spectral array (composable sub-block SED)",
    "sed_agn_polar": "spectral array (composable sub-block SED)",
    "sed_agn_lines_attenuated": "spectral array (line SED after the AGN screen)",
    "sed_dust_ir": "spectral array (dust emission SED)",
    "sed_dust_attenuated": "spectral array (attenuated stellar SED)",
    "sed_nebular": "spectral array (nebular continuum)",
    "sed_xray": "spectral array (X-ray SED)",
    "sed_radio": "spectral array (radio SED)",
    "sed_shock": "spectral array (shock SED)",
    "igm_transmission": "wavelength-grid transmission",
    "line_lums": "per-line array; read through pred.lines",
    "line_waves": "wavelength array of the line catalog",
    "log_line_lums": "per-line array; read through pred.lines",
    "log_line_lums_attenuated": "per-line array; read through pred.lines",
    "L_age": "per-SSP-age array read only inside the stellar functions",
    "log_L_age": "per-SSP-age array read only inside the stellar functions",
    # Echoes of an input the user already sets.
    "agn_cos_inc": "echo of the agn_cos_inc parameter",
    "radio_L_ir_input": "echo of the L_ir input to radio",
    "radio_log_L_ir_input": "echo of the L_ir input to radio",
    # Plumbing read by another component.
    "L_2500_intrinsic": "read by the radio and X-ray components (alpha_ox, radio loudness)",
    "L_4400_intrinsic": "read by the radio component (radio loudness)",
    "agn_torus_frac_effective": "torus fraction the composable AGN was evaluated at; read by the "
    "energy ledger on request",
    "dust_attenuation_factor": "wavelength-grid factor of the attenuation",
    "lyc_fdust": "read by the dust attenuators",
    "lyc_fesc": "read by the dust attenuators",
    "lyc_transmission": "read by the dust attenuators",
    "nebular_line_frac": "read by the dust attenuators",
    "age_boundary_younger_fraction": "age-bin plumbing of the stellar grid",
    "age_weights": "age weights read by the X-ray component",
    "age_weights_surviving": "age weights read by the X-ray component",
    "joint_weights": "age x metallicity weight grid",
    "lnu_age": "required input of the nebular component",
    "lnu_interval": "stellar grid plumbing",
    "log_metallicity_history": "history read by the X-ray component",
    "log_mstar": "formed-mass input read by the radio and X-ray components (menu: stellar_mass)",
    "log_stellar_mass_scale": "normalization plumbing",
    "sfh_grid_lbt_yr": "SFH grid read by the catalog SFH functions",
    "sfr_history": "SFH history array",
    "ssp_ages_yr": "SSP grid read by the nebular and X-ray components",
    "stellar_mass_scale": "normalization plumbing",
    "stellar_phot_lnu_precomp": "precompute array",
    # Same value under a different catalog name.
    "L_absorbed": "catalog l_dust_absorbed reads it",
    "log_L_absorbed": "catalog l_dust_absorbed reads it (log form)",
    "log_L_x_agn_2_10": "catalog log_l_x_agn reads it (lower-case l)",
    "log_L_x_agn_total": "catalog log_l_x_agn_total reads it",
    "log_L_x_hmxb_2_10": "catalog log_l_x_hmxb_2_10 reads it",
    "log_L_x_hotgas_0p5_2": "catalog log_l_x_hotgas_0p5_2 reads it",
    "log_L_x_lmxb_2_10": "catalog log_l_x_lmxb_2_10 reads it",
    "log_L_x_xrb_2_10": "catalog log_l_x_xrb reads it",
    "log_Lnu_2keV_30deg": "catalog log_lnu_2kev_30deg reads it (case-only name split)",
    "log_mstar_formed": "catalog stellar_mass reads it",
    "log_mstar_surviving": "catalog stellar_mass_surviving reads it",
    "log_nion": "catalog log_q_h reads it",
    # Retired on purpose: the linear value overflows float32 (#1206).
    "nion": "retired with no alias (#1206); the linear photon rate overflows float32",
    # Introspection value, -inf at the default neb_fdust_frac = 0.
    "log_L_lyc_dust": "introspection value; -inf at the default neb_fdust_frac = 0",
}


@pytest.fixture(scope="module")
def built(synthetic_ssp_wide, synthetic_tophat_obs):
    """``label -> SEDModel``, each model built once per module."""
    cache: dict[str, SEDModel] = {}

    def get(label: str) -> SEDModel:
        if label not in cache:
            kwargs = dict(MODELS[label])
            neb = kwargs.pop("neb", {"type": "none"})
            cache[label] = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh=_SFH,
                dust_attenuation=_DUST,
                neb=neb,
                redshift=Fixed(0.1),
                **kwargs,
            )
        return cache[label]

    return get


def _published_names(
    model: SEDModel, component: str | None = None, *, conditional: bool = False
) -> set[str]:
    """Names the chain's components publish.

    By default only the names whose ``available`` predicate holds for the configured component.
    ``conditional=True`` adds the names a component declares but its configuration does not
    produce (a composable-ADAF key on a monolithic AGN, say).
    """
    names: set[str] = set()
    for comp in model._full_state_chain():
        if component is not None and comp.name != component:
            continue
        for key, spec in getattr(type(comp), "published", {}).items():
            predicate = spec[3] if len(spec) > 3 else None
            if conditional or predicate is None or predicate(comp):
                names.add(key)
    return names


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=float).ravel()[0])


@pytest.mark.parametrize("label", sorted(MODELS))
def test_published_names_are_served_and_finite(label, built):
    """A: every published name the model provides is listed and computes a finite value."""
    model = built(label)
    available = set(model.available_properties)
    published = _published_names(model)
    assert _REQUIRED[label] <= published, f"not published: {sorted(_REQUIRED[label] - published)}"
    assert published <= available, f"published but not listed: {sorted(published - available)}"
    params = {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()}
    values = model.predict_properties(params, names=tuple(sorted(published)))
    not_finite = {k: _scalar(v) for k, v in values.items() if not np.isfinite(_scalar(v))}
    assert not not_finite, f"non-finite at the sampled point: {not_finite}"


@pytest.mark.parametrize("label", sorted(MODELS))
def test_every_declared_output_is_accounted_for(label, built):
    """B: each declared output is listed, published, a ledger key, or allowlisted."""
    model = built(label)
    available = set(model.available_properties) | _published_names(model, conditional=True)
    unaccounted = {}
    for comp in model._full_state_chain():
        for key in comp.outputs():
            name = key[0]
            if name in available or name in COMPOSABLE_LEDGER_OUTPUTS or name in _NOT_PUBLISHED:
                continue
            unaccounted.setdefault(name, comp.name)
    assert not unaccounted, f"declared but neither listed nor allowlisted: {unaccounted}"


def test_not_published_allowlist_has_a_reason_for_every_entry():
    empty = sorted(name for name, reason in _NOT_PUBLISHED.items() if not reason.strip())
    assert not empty, f"allowlist entries without a reason: {empty}"


def test_composable_agn_catalog_is_published_plus_ledger(built):
    """C: AGN catalog entries are its published names plus the ledger keys, once each."""
    model = built("agn_composable_kubota_done")
    catalog = model._ensure_property_catalog()
    agn_entries = {name for name, entry in catalog.items() if entry.component_name == "agn"}
    published = _published_names(model, component="agn")
    assert _REQUIRED["agn_composable_kubota_done"] <= published, "the AGN publishes nothing"
    assert not published & set(COMPOSABLE_LEDGER_OUTPUTS), (
        "a name is both published and a ledger key"
    )
    assert agn_entries == published | set(COMPOSABLE_LEDGER_OUTPUTS), (
        f"missing {sorted(published | set(COMPOSABLE_LEDGER_OUTPUTS) - agn_entries)}; "
        f"extra {sorted(agn_entries - published - set(COMPOSABLE_LEDGER_OUTPUTS))}"
    )
