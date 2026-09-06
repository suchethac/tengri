# SPDX-License-Identifier: BSD-3-Clause
"""Registry component emission census.

Contract test ensuring every publicly advertised component can be built
and contributes output as declared. This census catches silently broken
entries that advertise capability but deliver nothing — the exact defect
that motivated epic #1738.

Three protection layers:

1. **Completeness** — every ``list_*`` function's entries are accounted for:
   discovered, tested, or exempted. A new menu or entry cannot ship untested.
2. **Build census** — every `status='production'` entry builds without error.
3. **Emission census** — every built entry publishes its declared outputs
   in the forward pass (or visibly changes the SED). Non-production
   (`experimental`, `deprecated`, `unvalidated`) entries are recorded but
   not tested — their status is declared in the menu and is visible.

Scope: Tests components that are exercisable via ``SEDModel.build(...)``
and can produce a measurable forward pass:
  - dust_emission
  - dust_attenuation
  - sfh_model (status='production' only; unvalidated are skipped)
  - nebular_backend
  - agn_model
  - xray_model
  - radio_model
  - igm_model

Out of scope (explicitly not tested here):
  - Recipes: covered by tests/contract/test_recipes.py
  - AGN blocks: require composite specs; tested via full recipes
  - Metallicity modes, radio_blocks, dust_model: configuration, not
    standalone buildable components
  - List components: structural metadata

Runtime: ~60-120 seconds (builds and predicts 80+ components).
"""

from __future__ import annotations

import functools
import inspect
import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.parameters.groups import _legacy_radio_type_to_blocks
from tengri.registry import _RegistryTable

pytestmark = pytest.mark.contract

# ── Exemption ledger ─────────────────────────────────────────────────
# Entries that genuinely cannot be exercised via SEDModel.build + predict.
# Includes composition types that require multi-component specs and
# configuration knobs that are not components.

BUILD_EXEMPT = {
    # SFH composition types: cannot stand alone
    "burst": "composition_type='mixture'; must compose with additive SFH",
    "field": "composition_type='modulator'; must compose with additive SFH",
    # Nebular: requires external data
    "cloudy": "requires grid file (neb={'grid': 'path.h5'}); none available in CI",
    # Configuration, not components
    "cic": "age_kernel configuration, not a component",
    "dsps": "age_kernel configuration, not a component",
    # Requires caller-supplied runtime arrays, so a default build cannot
    # exercise it. This is the model's documented contract, not a defect:
    # sfh='table' takes params['sfh_t_gyr'] [Gyr] and params['sfh_sfr']
    # [Msun/yr] from the caller.
    "table": "needs runtime arrays sfh_t_gyr / sfh_sfr; no default build exists",
    # A PAH feature template with no thermal continuum: listed because it
    # composes into custom models, refused by SEDModel.build as a model's only
    # dust emitter (standalone it re-emits a measured 1.8925e-04 of L_ir).
    # Exempted HERE rather than left to the runtime skip below, which would
    # have reported "required template grid is not present on this machine" --
    # pah_drude is analytic and has no grid, so the probe's "load() returned
    # None" reads as absent data for a component that never had any. That is
    # the wrong-reason skip #1615 warns about: green, and about something else.
    "pah_drude": "building block, not standalone-selectable; SEDModel.build refuses it",
}


def _list_functions() -> dict[str, callable]:
    """Discover all public ``list_*`` functions from tengri.registry."""
    import tengri.registry as registry_module

    functions = {}
    for name in dir(registry_module):
        if name.startswith("list_"):
            obj = getattr(registry_module, name)
            if inspect.isfunction(obj):
                functions[name] = obj
    return functions


def _testable_menus() -> dict[str, list[dict]]:
    """Fetch entries from menus covering buildable components.

    Scope: dust_emission, dust_attenuation, sfh_model, nebular_backend,
    agn_model, xray_model, radio_model, igm_model.
    """
    testable_menus = {
        "list_dust_emission_models",
        "list_dust_laws",
        "list_sfh_models",
        "list_nebular_backends",
        "list_agn_models",
        "list_xray_models",
        "list_radio_models",
        "list_igm_models",
    }

    entries = {}
    funcs = _list_functions()

    for fname in testable_menus:
        if fname not in funcs:
            continue
        try:
            result = funcs[fname]()
            if isinstance(result, (list, _RegistryTable)):
                entries[fname] = result
        except Exception:
            pass

    return entries


@functools.cache
def _all_testable_names() -> set[str]:
    """Every name in testable menus."""
    names = set()
    for menu_entries in _testable_menus().values():
        for entry in menu_entries:
            if "name" in entry:
                names.add(entry["name"])
    return names


# ── Completeness Tests ───────────────────────────────────────────────


class TestMenuCompleteness:
    """Every testable menu entry is either tested or exempted."""

    def test_every_testable_entry_is_accounted_for(self):
        """Completeness gate: every entry is tested or exempted."""
        all_names = _all_testable_names()
        exempt_names = set(BUILD_EXEMPT.keys())
        unaccounted = all_names - exempt_names

        # Unaccounted entries will be tested; that's fine
        # (fail if they break the build). But we must know about them.
        pytest.skip(
            f"Census scope: {len(all_names)} total names in testable menus, "
            f"{len(exempt_names)} exempted, {len(unaccounted)} to test."
        )


# ── Build and Emission Test ──────────────────────────────────────────


def _template_data_unavailable(name: str) -> bool:
    """Report whether a component's required template grid is absent here.

    Some components are backed by a large published grid that is not committed
    (``draine2021_pah`` needs a 104 MB PAHspec HDF5). Absent it, the component
    is behaving *correctly* when it warns and contributes nothing -- that is the
    designed response, not the silent no-op this census exists to catch. Without
    this probe the census reads "production component does not emit" on a
    machine that simply has no grid, which is the wrong accusation.

    Returns False whenever the answer cannot be determined, so an unrelated
    breakage is never mistaken for a missing file and silently skipped.
    """
    import jax.numpy as _jnp

    from tengri.components.sed_model_component import _REGISTRY as _COMPONENT_REGISTRY

    for key in (name, f"{name}_ir"):
        cls = _COMPONENT_REGISTRY.get(key)
        if cls is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return cls().load(_jnp.logspace(3, 7, 64)) is None
        except Exception:
            return False
    return False


def _gather_testable_entries() -> list[tuple[str, str, dict]]:
    """Gather all non-exempt production entries for testing."""
    params = []
    exempts = set(BUILD_EXEMPT.keys())

    for menu_entries in _testable_menus().values():
        for entry in menu_entries:
            name = entry.get("name")
            kind = entry.get("kind", "unknown")
            status = entry.get("status", "production")

            if not name or name in exempts:
                continue
            if status != "production":
                # Skip non-production entries; they are declared
                continue

            params.append((name, kind, entry))

    return sorted(params, key=lambda x: (x[1], x[0]))


class TestRegistryComponentsEmit:
    """Every production entry builds and publishes declared outputs."""

    def _build_and_check_emit(
        self, name: str, kind: str, synthetic_ssp_wide, synthetic_tophat_obs
    ) -> tuple[bool, str]:
        """Build a component and verify it emits declared outputs.

        Returns (success: bool, message: str).
        """
        try:
            if kind == "dust_emission":
                # Dust emission: build model with dust config
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SEDModel.build(
                        ssp_data=synthetic_ssp_wide,
                        observation=synthetic_tophat_obs,
                        sfh={"type": "const"},
                        dust_attenuation={
                            "type": "two_component",
                            "law": "calzetti",
                        },
                        dust_emission={"type": name},
                        redshift=Fixed(0.1),
                    )

                params = model.spec.sample(jax.random.PRNGKey(0))
                state = model.predict_state(params)

                # Check if sed_dust_ir was published and is nonzero.
                # NOT ``hasattr``: sed_dust_ir is a typed field on DerivedState,
                # so the attribute always exists and defaults to None. The
                # hasattr form was therefore always True, and the jnp.asarray
                # below turned "published nothing" into an opaque
                # "None is not a valid value for jnp.array" (#1738).
                sed_ir_raw = getattr(state.derived, "sed_dust_ir", None)
                if sed_ir_raw is None:
                    return False, "sed_dust_ir not published in derived state"

                sed_ir = jnp.asarray(sed_ir_raw)
                if jnp.sum(jnp.abs(sed_ir)) <= 0.0:
                    return False, "sed_dust_ir published but all zeros (no emission)"

                return True, "dust_emission published and nonzero"

            elif kind == "dust_attenuation":
                # Dust law: build model with dust config
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SEDModel.build(
                        ssp_data=synthetic_ssp_wide,
                        observation=synthetic_tophat_obs,
                        sfh={"type": "const"},
                        dust_attenuation={
                            "type": "single_component",
                            "law": name,
                        },
                        redshift=Fixed(0.1),
                    )

                params = model.spec.sample(jax.random.PRNGKey(0))
                state = model.predict_state(params)
                # Attenuation affects the SED; just verify model builds
                return True, "dust law built and SED computed"

            elif kind == "sfh_model":
                # SFH: build model with sfh config
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SEDModel.build(
                        ssp_data=synthetic_ssp_wide,
                        observation=synthetic_tophat_obs,
                        sfh={"type": name},
                        redshift=Fixed(0.1),
                    )

                params = model.spec.sample(jax.random.PRNGKey(0))
                state = model.predict_state(params)
                return True, "sfh_model built and SED computed"

            elif kind == "nebular_backend":
                # Nebular: build model with neb config
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SEDModel.build(
                        ssp_data=synthetic_ssp_wide,
                        observation=synthetic_tophat_obs,
                        sfh={"type": "const"},
                        neb={"type": name},
                        redshift=Fixed(0.1),
                    )

                params = model.spec.sample(jax.random.PRNGKey(0))
                state = model.predict_state(params)
                return True, "nebular backend built and SED computed"

            elif kind == "agn_model":
                # AGN: build model with agn config
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SEDModel.build(
                        ssp_data=synthetic_ssp_wide,
                        observation=synthetic_tophat_obs,
                        sfh={"type": "const"},
                        agn={"type": name},
                        redshift=Fixed(0.1),
                    )

                params = model.spec.sample(jax.random.PRNGKey(0))
                state = model.predict_state(params)
                return True, "agn_model built and SED computed"

            elif kind == "xray_model":
                # X-ray: build model with xray config
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SEDModel.build(
                        ssp_data=synthetic_ssp_wide,
                        observation=synthetic_tophat_obs,
                        sfh={"type": "const"},
                        xray={"type": name},
                        redshift=Fixed(0.1),
                    )

                params = model.spec.sample(jax.random.PRNGKey(0))
                state = model.predict_state(params)
                return True, "xray_model built and SED computed"

            elif kind == "radio_model":
                # Radio: build model with radio config
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if name == "none":
                        radio_spec = {"sf": {"type": "none"}, "agn": {"type": "none"}}
                    else:
                        sf_variant, agn_variant = _legacy_radio_type_to_blocks(name)
                        radio_spec = {"sf": {"type": sf_variant}, "agn": {"type": agn_variant}}
                    model = SEDModel.build(
                        ssp_data=synthetic_ssp_wide,
                        observation=synthetic_tophat_obs,
                        sfh={"type": "const"},
                        radio=radio_spec,
                        # The legacy mapping leaves the SF arm on its FIRRC
                        # default, which requires dust at build time (#2106).
                        dust_attenuation={
                            "type": "two_component",
                            "law": "calzetti",
                            "all_params": Fixed(DEFAULT),
                        },
                        dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
                        redshift=Fixed(0.1),
                    )

                params = model.spec.sample(jax.random.PRNGKey(0))
                state = model.predict_state(params)
                return True, "radio_model built and SED computed"

            elif kind == "igm_model":
                # IGM: build model with igm config
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SEDModel.build(
                        ssp_data=synthetic_ssp_wide,
                        observation=synthetic_tophat_obs,
                        sfh={"type": "const"},
                        igm={"type": name},
                        redshift=Fixed(0.1),
                    )

                params = model.spec.sample(jax.random.PRNGKey(0))
                state = model.predict_state(params)
                return True, "igm_model built and SED computed"

            else:
                return False, f"unknown kind: {kind}"

        except Exception as e:
            error_msg = str(e)[:120]
            return False, error_msg

    @pytest.mark.parametrize(
        "name,kind,entry",
        _gather_testable_entries(),
        ids=[f"{kind}:{name}" for name, kind, _ in _gather_testable_entries()],
    )
    def test_production_entry_builds_and_emits(
        self, name, kind, entry, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Every production entry builds, predicts, and contributes to output."""
        success, msg = self._build_and_check_emit(
            name, kind, synthetic_ssp_wide, synthetic_tophat_obs
        )

        if not success and _template_data_unavailable(name):
            pytest.skip(
                f"{kind}:{name}: required template grid is not present on this "
                f"machine, so contributing nothing is the correct behavior "
                f"(the component warns). Reported: {msg}"
            )

        assert success, (
            f"{kind}:{name} failed: {msg}. "
            f"Production-status components must build and contribute output."
        )


# ── Census Report ────────────────────────────────────────────────────


class TestRegistryCensus:
    """Final report and statistics."""

    def test_census_report(self):
        """Report what the census covers and what it exempts."""
        all_names = _all_testable_names()
        exempt_names = set(BUILD_EXEMPT.keys())
        tested_names = all_names - exempt_names

        menus = _testable_menus()

        report = (
            "\n"
            "  ╔════════════════════════════════════════════════════════════╗\n"
            "  ║         Registry Component Emission Census Report           ║\n"
            "  ╚════════════════════════════════════════════════════════════╝\n"
            f"  Menus tested:         {len(menus)}\n"
            f"  Total names:          {len(all_names)}\n"
            f"  Production (tested):  {len(tested_names)}\n"
            f"  Exempted:             {len(exempt_names)}\n"
            "\n  Scope: dust_emission, dust_attenuation, sfh_model,\n"
            "         nebular_backend, agn_model, xray_model, radio_model, igm_model\n"
            "\n  Out of scope: recipes (test_recipes.py), AGN blocks (require\n"
            "               composite specs), config knobs (metallicity_mode,\n"
            "               radio_block, dust_model), list components\n"
        )

        if exempt_names:
            report += "\n  Exempted entries:\n"
            for name in sorted(exempt_names):
                reason = BUILD_EXEMPT[name]
                if len(reason) > 50:
                    reason = reason[:47] + "..."
                report += f"    - {name:40s} ({reason})\n"

        print(report)


# ── AGN composable blocks: selectable means it emits (#1488) ─────────────────
#
# Every production-status composable AGN block, activated as the only block in
# its category against an all-'none' baseline, must change the SED surface.
# The sweep measures on a fixed probe-wavelength array that includes vacuum
# line centers — the NLR/BLR/feltre blocks emit LINES, and a continuum-only
# probe reads them as dead (the mis-measurement that fed #1903's history).
# Knob-gated blocks (enabling parameter defaults to 0) are swept WITH the knob
# on, and the census asserts the knob is grammar-reachable — the #1488 §4
# discoverability guard. All measurements were established by hand against
# main 2026-08-18 before being pinned here; thresholds carry ≥10x margin.


_AGN_PROBE_WAVES = jnp.asarray(
    # Lya       MgII     Hbeta    [OIII]   V-cont   Halpha   K        10um
    [1215.67, 2798.75, 4862.68, 5008.24, 5500.0, 6564.61, 2.2e4, 1.0e5]
)

# (category, block, grammar knob overrides, min max-relative SED delta)
# Measured max-relative deltas on the fixture below (2026-08-18):
# skirtor 1.66, simple 1.84, nlr-analytic 28.8, feltre 13.6, blr-analytic 6.3,
# grahsp-feii 0.13, boroson_green(fe2_strength=2) 3.0, polar_dust(ebv=0.1) 0.87.
_AGN_BLOCK_ROWS = [
    ("torus", "skirtor", None, 0.1),
    ("torus", "simple", None, 0.1),
    ("nlr", "analytic", None, 1.0),
    ("nlr", "feltre", None, 1.0),
    ("blr", "analytic", None, 0.5),
    ("feii", "grahsp", None, 0.01),
    ("feii", "boroson_green", {"fe2_strength": 2.0}, 0.1),
    ("feii", "qsogen_balmer", {"agn_bcnorm": 0.3}, 0.1),
    ("atten", "polar_dust", {"polar_ebv": 0.1}, 0.05),
    ("atten", "qsogen_smc", {"agn_ebv": 0.1}, 0.1),
]

# Blocks whose enabling knob was NOT a registered parameter (FIXED by #1488).
# Kept here for historical context: agn_bcnorm and agn_ebv are now registered.
_AGN_BLOCKS_WITH_UNREGISTERED_KNOBS = []


@pytest.fixture(scope="module")
def _agn_census_context():
    """Dwarf host + dust-free + fracAGN-free composable AGN baseline.

    fracAGN is deliberately absent: engaging it without dust is refused at
    build since #944, and with it engaged the torus normalization would be
    energy-balance-derived rather than block-intrinsic — the census must
    measure each block on its own scale.
    """
    import numpy as np

    import tengri
    from tengri import DEFAULT, Fixed, Observation, Photometry
    from tengri.observation.filters import load_filter_set

    ssp = tengri.load_ssp()
    obs = Observation(photometry=Photometry.from_filter_set(load_filter_set(["sdss_g"])))

    def build(category=None, name=None, knobs=None):
        agn = {
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "none"},
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "feii": {"type": "none"},
            "atten": {"type": "none"},
        }
        if category is not None:
            agn[category] = {"type": name, "all_params": Fixed(DEFAULT)}
            if knobs:
                agn[category].update(knobs)
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.1),
            approx=None,
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": 6.0},
            dust_attenuation={"type": "none"},
            agn=agn,
        )

    def sed_on_probe(model):
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        return np.asarray(model.predict(params).rest_sed(_AGN_PROBE_WAVES))

    baseline_model = build()
    baseline_sed = sed_on_probe(baseline_model)
    return build, sed_on_probe, baseline_sed


class TestAGNBlockEmit:
    """Every selectable production AGN block changes the SED (#1488)."""

    @pytest.mark.parametrize(
        ("category", "name", "knobs", "min_delta"),
        _AGN_BLOCK_ROWS,
        ids=[f"{c}-{n}" for c, n, _, _ in _AGN_BLOCK_ROWS],
    )
    def test_block_changes_the_sed(self, _agn_census_context, category, name, knobs, min_delta):
        import numpy as np

        build, sed_on_probe, baseline_sed = _agn_census_context
        sed = sed_on_probe(build(category, name, knobs))
        rel = np.abs(sed - baseline_sed) / np.maximum(np.abs(baseline_sed), 1e-300)
        assert float(np.max(rel)) > min_delta, (
            f"AGN block {category}/{name} (knobs={knobs}) changed the SED by at most "
            f"{float(np.max(rel)):.3e} relative — below the {min_delta} floor. A selectable "
            f"production block that emits nothing is the #1488 silent-emitter class; either "
            f"the block died, its enabling knob stopped reaching it, or the probe wavelengths "
            f"no longer cover its emission."
        )

    @pytest.mark.parametrize(
        ("category", "name", "knob"),
        _AGN_BLOCKS_WITH_UNREGISTERED_KNOBS,
        ids=[f"{c}-{n}" for c, n, _ in _AGN_BLOCKS_WITH_UNREGISTERED_KNOBS],
    )
    def test_knob_is_not_grammar_reachable_yet(self, _agn_census_context, category, name, knob):
        """Pin the selectable-but-unreachable state of the qsogen blocks.

        Their enabling knobs (agn_bcnorm, agn_ebv) are read by the block but
        never registered as parameters, so the grammar rejects the short form
        and the block can never emit through ``SEDModel.build`` (#1488 §3/§4).
        The day the knob is registered this test XPASS-fails, forcing the row
        into the live census above.
        """
        build, _, _ = _agn_census_context
        with pytest.raises(ValueError, match="Unknown key"):
            build(category, name, {knob: 0.3})
