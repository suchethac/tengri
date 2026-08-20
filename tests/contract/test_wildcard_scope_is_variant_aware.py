# SPDX-License-Identifier: BSD-3-Clause
r"""Contract: no group's ``all_params: FREE`` frees a parameter its variant ignores.

A group's declared parameters span every structural variant it can dispatch to,
but a build selects one. Freeing the whole superset hands the sampler dimensions
the selected variant never reads — flat directions explored at full cost whose
posterior comes back equal to the prior, with nothing in the fit saying why.

That defect has now been found in six places, each time only because someone
looked at that group:

===================  ==============================================
group                what an unscoped wildcard freed
===================  ==============================================
``agn``              inactive blocks' params (e.g. GRAHSP under SKIRTOR)
``radio.sf/agn``     the unselected radio model's coefficients
``dust.emission``    6 inert dims under ``schreiber2016`` (#1482)
``dust``             4 shape modifiers ``calzetti`` discards
``shock``            the unselected normalization's luminosity scale
``xray``             ``alpha_irx``, read only by the ``lopez24`` corona
===================  ==============================================

The first three were fixed one at a time behind three branches in the resolver,
so a fourth group needing it failed silently until someone wrote a fourth
branch. This test is the general form: it enumerates variants *from the live
registries* — all 22 attenuation laws come from ``DUST_LAWS``, not a list here —
and asserts the property for every one.

A guard, though, is only as wide as the set of *groups* it enumerates, which is
the same failure one level up: green about what it covers and silent about the
rest, with nothing distinguishing "swept and clean" from "never looked at". So
the group list is not written here either. It is derived from
``_GROUP_STRUCTURAL_KEYS`` — the map ``parse_groups`` validates against, so a
group cannot be accepted by the grammar without an entry — and every group that
takes a wildcard must be either swept below or named in :data:`_EXCUSED` with
its reason. See :func:`test_every_wildcard_group_is_swept_or_excused`.

Two measurement traps this file exists downstream of, both of which produced a
wrong answer before being caught:

**The observable must be able to see the parameter.** An earlier version probed
``predict_photometry`` and reported four live nebular parameters as inert:
broadband photometry genuinely cannot see line broadening
(``neb_eline_sigma_kms`` redistributes flux *within* a band), so "photometry did
not move" conflates *the model ignores this* with *this observable cannot
resolve it*. Photometry is a linear functional of the SED — if the SED is
unchanged the photometry is too — so the SED is strictly more sensitive and is
what these probes use.

**The arm must be live.** ``shock_velocity`` measured as perfectly inert under
``norm='lhalpha'`` while being correctly wired: the shock is anchored to an
*absolute* Hα luminosity capped at 1e46 erg/s, and the synthetic galaxy's
L_bol is 5e59, so the shock contributed ~1e-11 of the SED. Sensitivity ramps
smoothly with the anchor (2e-16, 2e-14, 2e-12, 2e-10 at log L_Hα = 40, 42, 44,
46) — a dead arm, not a dead parameter. The shock cases therefore run on a
low-mass galaxy where the same sweep moves the SED by 16%. Because that failure
mode is invisible from the inside, the assertion below distinguishes it: an
inert parameter *alongside* a live one proves the arm works and indicts the
scope, while an all-inert group is reported as a fixture failure instead.

Values are swept across each parameter's own declared support rather than drawn
randomly: two random draws can coincide, which reads as "inert" and would make
this test pass for the wrong reason.

**Not covered, stated rather than left silent.** :data:`_EXCUSED` carries one
entry per unswept group with the measured reason, and the census test above
fails if any group is missing from both it and the sweep.

**A correction, recorded because the wrong answer was written down first.** This
docstring previously stated that ``xray`` could not be asserted because "all
five variants report an identical five-parameter inert set", which was read as a
fixture artifact — the synthetic SSP being a ``(5000/lambda)^2`` power law that
swamps the X-ray component. Both halves were wrong, and the second explained
away the first:

* The inert set looked variant-independent only because the *probe* was. With an
  X-ray-capable SSP and a luminous AGN, ``xray_alpha_irx`` moves the SED by 62%
  under ``lopez24`` and by **exactly zero** under every other corona — the
  textbook scoping defect, which ``xray_alpha_irx``'s own declaration had
  described in prose all along.
* "Swamped by a bright fixture" predicts small-but-nonzero, and predicts it for
  every X-ray parameter. The measurements were *exactly* 0.0 while
  ``xray_E_cut`` moved the SED by 85% on the same build. A magnitude argument
  cannot explain a set of exact zeros sitting beside a live parameter.

The remaining two, ``xray_det_hmxb`` / ``xray_det_lmxb``, were inert for a third
reason that is not scoping at all: the component never passed them to the
physics. Fixed in #1706, so they are no longer narrowed away and
:data:`_XRAY_UNREACHABLE_PARAMS` is now empty. Their own guard lives in
``tests/regression/bug/test_xray_xrb_offsets_wired.py``, which sweeps them on
both X-ray models and on both accelerated paths — this suite cannot see them,
because a wildcard test only reaches parameters the wildcard frees.
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, FREE, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract

#: Relative SED change below which a parameter counts as unread. A wired
#: parameter that the fixture makes weak still lands orders above this; an
#: unwired one is bit-identical, so the measured gap is not marginal.
_INERT_TOL = 1e-12

#: Attenuation deep enough that the dust terms are real rather than rounding.
#: Applied to the *base draw* rather than passed as explicit priors: a
#: user-supplied value is excluded from the wildcard's accounting, and pinning
#: the only two parameters ``calzetti`` leaves in scope makes the build refuse
#: the wildcard outright ("freed 0 of 1"). The curve-shape parameters are what
#: this test is about, and they are multiplied by tau — at tau = 0 every one of
#: them is inert for an honest reason that would read as a scoping failure.
_DEEP_ATTENUATION = {"dust_tau_bc": 2.0, "dust_tau_diff": 1.5}

#: Hα anchor for the absolute shock normalization, near the top of its declared
#: range. Paired with :data:`_FAINT_GALAXY` — see the module docstring.
_LUMINOUS_SHOCK = {"shock_log_lhalpha": 45.0}

#: Stellar mass low enough that an absolute shock luminosity is a real term in
#: the SED rather than 11 orders below it.
_FAINT_GALAXY = 0.0


@pytest.fixture(scope="module")
def panchromatic_obs():
    """Six top-hats from 1500 A to 500 um — UV through far-IR.

    Optical-only bands would report every dust-emission parameter as inert for
    the honest reason that the emission falls outside the bandpasses, which is
    filter coverage rather than a wiring defect.
    """

    def _tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (1500.0, 5000.0, 2.0e4, 2.4e5, 1.0e6, 5.0e6)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


@pytest.fixture(scope="module")
def xray_capable_ssp():
    """Like ``synthetic_ssp_wide`` but starting at 1 A, so the X-ray band exists.

    ``synthetic_ssp_wide`` starts at 100 A. The X-ray corona lives at ~1-100 A,
    i.e. off the end of that grid, which makes every X-ray parameter report as
    inert for a reason that is the fixture's rather than the model's.
    """
    n_age = 20
    wave = jnp.logspace(0.0, 7.0, 2000)  # 1 A - 1 mm
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


class Case(dict):
    """One (group, structural variant) build, with the fixture it needs.

    ``group`` is the grammar group this case covers, and is what
    :func:`test_every_wildcard_group_is_swept_or_excused` counts. It is stated
    rather than parsed back out of ``label`` so that renaming a label for
    readability cannot silently drop a group out of the census.
    """

    def __init__(
        self, label, group, prefix, groups, *, overrides=None, log_total_mass=None, ssp="wide"
    ):
        super().__init__(
            label=label,
            group=group,
            prefix=prefix,
            groups=groups,
            overrides=overrides or {},
            log_total_mass=log_total_mass,
            ssp=ssp,
        )


def _dust_law_cases():
    """Every registered attenuation law, read from the registry not a list."""
    from tengri.components.dust.laws._registry import DUST_LAWS

    for law in sorted(DUST_LAWS):
        yield Case(
            f"dust[law={law}]",
            "dust",
            "dust_",
            {
                "dust": {
                    "type": "two_component",
                    "law_bc": law,
                    "law_diff": law,
                    "*": FREE,
                    "emission": {"type": "themis", "*": FIXED},
                },
                "neb": {"type": "none"},
            },
            overrides=_DEEP_ATTENUATION,
        )


def _shock_cases():
    """Both shock normalizations, on a galaxy faint enough to see the shock."""
    for norm in ("frac", "lhalpha"):
        yield Case(
            f"shock[norm={norm}]",
            "shock",
            "shock_",
            {
                "dust": {
                    "type": "two_component",
                    "law": "calzetti",
                    "*": FIXED,
                },
                "neb": {"type": "none"},
                "shock": {"norm": norm, "*": FREE},
            },
            overrides=_LUMINOUS_SHOCK,
            log_total_mass=_FAINT_GALAXY,
        )


def _neb_cases():
    """Photoionization backends — none, on the fixtures available here.

    Deliberately empty, and the reason is worth more than the case would be.

    * ``cue`` refuses a bare-stellar SSP fixture (``CueWNESSPError``).
    * ``ssp`` (baked-in) declares no free parameters, so there is no superset.
    * ``cloudy`` needs a grid file that is not shipped and has no synthetic
      stand-in, so the build raises outright in CI.
    * ``cb19`` builds, because ``conftest`` synthesizes a stand-in grid when the
      real one is absent — but that stand-in is ``np.broadcast_to`` of a single
      line-ratio vector, so it is **constant along every interpolation axis**,
      including the three this suite would test. On it, moving ``neb_log_nH``
      changes the prediction by ~1e-9 of floating-point interpolation noise
      whether or not the value reaches the backend. A case that passes on noise
      is a case that passes for the wrong reason, and it would keep passing if
      the threading regressed.

    The threading those cases would have covered is asserted directly instead,
    at the wiring, by :func:`test_backend_declared_nebular_params_are_threaded`
    — which does not depend on how rich the grid is.
    """
    return ()


def _xray_cases():
    """Every X-ray corona model, on a fixture where the X-ray band exists.

    Two fixture conditions are load-bearing, and each was established by
    measurement rather than assumed:

    * the SSP must reach the X-ray band (see :func:`xray_capable_ssp`);
    * an AGN must be present. ``gamma_agn`` and ``log_nh`` shape the *AGN*
      corona, so with no AGN they are inert for an honest reason — a dead arm,
      not a dead parameter, the distinction the ``shock_velocity`` measurement
      forced. With a luminous AGN they move the SED by 10.9 and 89.4.
    """
    from tengri.parameters.groups import _valid_xray_types

    for model in sorted(t for t in _valid_xray_types() if t != "none"):
        yield Case(
            f"xray[model={model}]",
            "xray",
            "xray_",
            {
                "dust": {
                    "type": "two_component",
                    "law": "calzetti",
                    "*": FIXED,
                },
                "neb": {"type": "none"},
                "xray": {"type": model, "*": FREE},
                # The AGN needs a DISC, not just a luminosity. The corona
                # models anchor to the disc's L_2500 through alpha_ox, and a
                # bare ``composable`` AGN with no disc block publishes no
                # L_2500 -- measured: it is bit-identical to no AGN at all.
                # This went unnoticed while every xray name resolved to the
                # shared component, whose XRB channel is driven by star
                # formation and so emits with no AGN; once ``agn_xray_corona``
                # began building its own component (#1684) the same fixture
                # made it contribute nothing, and this suite's own
                # discriminator correctly indicted the fixture rather than the
                # scope.
                "agn": {
                    "type": "composable",
                    "*": FIXED,
                    "log_lbol": Fixed(13.0),
                    "disc": {"type": "multicolor"},
                    "torus": {"type": "skirtor"},
                },
            },
            ssp="xray",
        )


def _simple_group_cases():
    """Groups whose wildcard needs no special fixture, swept for completeness."""
    yield Case(
        "sfh[dpl]",
        "sfh",
        "sfh_",
        {
            "sfh": {"type": "dpl", "*": FREE},
            "neb": {"type": "none"},
            "dust": {"law": "power_law", "type": "two_component", "*": FIXED},
        },
    )
    yield Case(
        "met",
        "met",
        "met_",
        {
            "neb": {"type": "none"},
            "dust": {"law": "power_law", "type": "two_component", "*": FIXED},
            "met": {"*": FREE},
        },
    )
    yield Case(
        "radio[condon92]",
        "radio",
        "radio_",
        {
            "neb": {"type": "none"},
            "dust": {"law": "power_law", "type": "two_component", "*": FIXED},
            "radio": {"type": "condon92", "*": FREE},
        },
    )


CASES = [
    *_dust_law_cases(),
    *_shock_cases(),
    *_neb_cases(),
    *_xray_cases(),
    *_simple_group_cases(),
]

#: Groups this suite cannot sweep, and the measured reason for each. Consulted
#: by :func:`test_every_wildcard_group_is_swept_or_excused`; every entry is a
#: statement that someone looked, not that nobody did. Where another suite
#: already owns the group, the entry names it — the point is that no group is
#: unaccounted for, not that this file must own all of them.
_EXCUSED: dict[str, str] = {
    "dust.emission": (
        "owned by tests/contract/test_dust_emission_wildcard.py, the guard "
        "written for #1482 — it narrows the outcome to the selected engine's "
        "own declared_parameters(), which is finer than this sweep."
    ),
    "agn": (
        "owned by tests/contract/test_agn_block_consumes.py, which pins "
        "agn_active_param_set against the CONSUMES tables and asserts every "
        "free AGN param in recipes.agn_panchromatic() moves predict()."
    ),
    "agn.disc": "owned by tests/contract/test_agn_block_consumes.py (block-scoped wildcard).",
    "agn.torus": "owned by tests/contract/test_agn_block_consumes.py (block-scoped wildcard).",
    "agn.nlr": "owned by tests/contract/test_agn_block_consumes.py (block-scoped wildcard).",
    "agn.blr": "owned by tests/contract/test_agn_block_consumes.py (block-scoped wildcard).",
    "agn.feii": "owned by tests/contract/test_agn_block_consumes.py (block-scoped wildcard).",
    "agn.atten": "owned by tests/contract/test_agn_block_consumes.py (block-scoped wildcard).",
    "agn.lines": (
        "deprecated alias, expanded to an (agn.nlr, agn.blr) pair by "
        "expand_lines_alias before any wildcard is resolved, so it never "
        "reaches the resolver as a group of its own."
    ),
    "neb": (
        "no usable fixture — see _neb_cases for the measurement. Both CB19 "
        "grids reachable from this checkout are constant along all seven "
        "physical axes (max relative variation exactly 0.0), so a sweep would "
        "measure interpolation noise; cue refuses the synthetic SSP; cloudy "
        "ships no grid. The threading is asserted at the wiring instead."
    ),
    "igm": (
        "measured: the wildcard frees nothing under any igm model, so there is "
        "no superset to over-free and the sweep would be vacuous."
    ),
    "igm.dla": "measured: frees nothing, as igm — no superset to over-free.",
    "radio.sf": "measured: frees nothing under bell2003 — no superset to over-free.",
    "radio.agn": (
        "its two params (radio_loudness, radio_alpha_agn) are inert with no "
        "AGN present and every param freed is inert, which by this suite's own "
        "discriminator indicts the fixture rather than the scope. Sweeping it "
        "needs a radio-loud AGN fixture this module does not have."
    ),
}


def _build(ssp, obs, case):
    # A case may carry its own sfh group (the sfh case frees it); otherwise the
    # sfh is pinned so it contributes no free params to the prefix under test.
    groups = dict(case["groups"])
    sfh = dict(groups.pop("sfh", None) or {"type": "dpl", "*": FIXED})
    if case["log_total_mass"] is not None:
        sfh["log_total_mass"] = case["log_total_mass"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=sfh,
            redshift=Fixed(0.5),
            **groups,
        )


def _predict_sed(model, params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(params).rest_sed())


def _partition_freed(model, prefix, overrides):
    """Split freed params into (live, inert) by sweeping their declared support."""
    freed = sorted(p for p in model.spec.free_params if p.startswith(prefix))
    base = dict(model.spec.sample(jax.random.PRNGKey(0)))
    base.update({k: np.float64(v) for k, v in overrides.items() if k in base})

    sed0 = _predict_sed(model, base)
    denom = np.where(np.abs(sed0) > 0, np.abs(sed0), 1.0)

    live, inert = [], []
    for name in freed:
        lo, hi = model.spec.get_distribution(name).bounds
        worst = 0.0
        for value in np.linspace(float(lo), float(hi), 5):
            sed = _predict_sed(model, {**base, name: np.float64(value)})
            worst = max(worst, float(np.max(np.abs(sed - sed0) / denom)))
        (live if worst > _INERT_TOL else inert).append(name)
    return freed, live, inert


@pytest.mark.parametrize("case", CASES, ids=[c["label"] for c in CASES])
def test_no_wildcard_freed_parameter_is_inert(
    synthetic_ssp_wide, xray_capable_ssp, panchromatic_obs, case
):
    """Every parameter the wildcard hands the sampler must move the prediction."""
    ssp = xray_capable_ssp if case["ssp"] == "xray" else synthetic_ssp_wide
    model = _build(ssp, panchromatic_obs, case)
    freed, live, inert = _partition_freed(model, case["prefix"], case["overrides"])

    if not inert:
        return

    assert live, (
        f"{case['label']}: none of {freed} moves the SED. That indicts the "
        f"fixture, not the scope — this component contributes nothing "
        f"measurable here, so the probe cannot tell an unread parameter from an "
        f"invisible one. Make the component a real term in the SED (see the "
        f"module docstring on the shock arm) before reading anything into this."
    )
    assert not inert, (
        f"{case['label']}: 'all_params: FREE' freed {inert}, which cannot move "
        f"the SED anywhere in their own declared support, while {live} can — so "
        f"the component is live and these are genuinely unread. The wildcard is "
        f"not scoped to the selected variant (freed={freed})."
    )


def test_the_dust_freed_set_depends_on_the_selected_law(synthetic_ssp_wide, panchromatic_obs):
    """Scoping must discriminate, not just avoid inert params.

    A wildcard that froze all four shape modifiers unconditionally would pass
    the sweep above for the same reason ``calzetti`` does — by freeing nothing
    inert — while making ``dust_Rv`` unreachable under ``cardelli``. This pins
    the other side: the freed set is a function of the law.
    """
    freed = {}
    for law in ("calzetti", "cardelli", "power_law", "kriek_conroy"):
        case = Case(
            law,
            "dust",
            "dust_",
            {
                "dust": {
                    "type": "two_component",
                    "law_bc": law,
                    "law_diff": law,
                    "*": FREE,
                    "emission": {"type": "themis", "*": FIXED},
                },
                "neb": {"type": "none"},
            },
        )
        model = _build(synthetic_ssp_wide, panchromatic_obs, case)
        freed[law] = {p for p in model.spec.free_params if p.startswith("dust_")}

    assert "dust_Rv" in freed["cardelli"], (
        "cardelli reads dust_Rv but the wildcard no longer frees it — scoping "
        "has become a blanket freeze, which is the opposite failure"
    )
    assert "dust_Rv" not in freed["calzetti"], (
        "calzetti fixes R_V = 4.05 internally, so freeing dust_Rv gives the "
        "sampler a dimension the curve contradicts"
    )
    assert "dust_slope" in freed["power_law"]
    assert freed["kriek_conroy"] >= {"dust_delta", "dust_bump_strength"}

    # Every law must still free the optical depths: they are not shape
    # arguments to any curve, so no law selection may narrow them away.
    for law, names in freed.items():
        assert {"dust_tau_bc", "dust_tau_diff"} <= names, (
            f"{law} lost the Charlot & Fall optical depths — the scope is "
            f"subtracting parameters no law claims (got {sorted(names)})"
        )


def test_an_omitted_law_scopes_as_its_resolved_default():
    """Not naming a law must scope identically to naming the default one.

    ``law_bc``/``law_diff`` both default to ``power_law``, which reads
    ``dust_slope`` — but an unnamed slot is simply *absent* from the translated
    structural kwargs. Reading the slots from the raw kwargs therefore saw no
    law, narrowed ``dust_slope`` away, and silently pinned it for every caller
    who did not name a law explicitly, which is the common case. The scope must
    come from the resolved ``Parameters``, where the default is present.
    """
    import tengri

    def freed(dust):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = tengri.parse_groups(
                sfh={"type": "dpl", "*": FIXED},
                dust=dust,
                neb={"type": "none"},
                redshift=Fixed(0.1),
            )
        return {p for p in spec.free_params if p.startswith("dust_")}

    omitted = freed({"law": "power_law", "type": "two_component", "*": FREE})
    explicit = freed(
        {"type": "two_component", "law_bc": "power_law", "law_diff": "power_law", "*": FREE}
    )
    assert omitted == explicit, (
        f"omitting the law scopes differently from naming its default: "
        f"omitted={sorted(omitted)} explicit={sorted(explicit)}"
    )
    assert "dust_slope" in omitted, (
        "the default law is power_law, which reads dust_slope — narrowing it "
        "away pins a parameter the law in force does read"
    )


def test_backend_declared_nebular_params_are_threaded():
    """A backend that names an axis must receive it; one that does not, must not.

    The component built one shared kwargs dict that never contained
    ``neb_log_nH``, ``neb_co`` or ``neb_dno``, so ``CB19Backend`` — which names
    all three in both methods the dict is splatted into — received its signature
    defaults on every call while the sampler proposed values freely. Sweeping
    each across its declared support moved the SED by exactly 0.0.

    Asserted at the wiring rather than through a prediction on purpose. The only
    CB19 grid available in CI is ``conftest``'s synthetic stand-in, which is
    constant along these very axes, so an SED-level probe there measures
    interpolation noise and would stay green if this regressed.

    Passing them unconditionally is not the alternative: every backend ends in
    ``**kwargs``, so an unmodeled parameter is silently dropped rather than
    rejected — indistinguishable from being read.
    """
    from tengri.components.nebular.baked_in import BakedInBackend
    from tengri.components.nebular.cloudy_cb19 import CB19Backend
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend
    from tengri.components.nebular.component import (
        _BACKEND_OPTIONAL_PARAMS,
        _backend_accepted_params,
    )

    assert _backend_accepted_params(CB19Backend) == frozenset(_BACKEND_OPTIONAL_PARAMS), (
        "CB19 names log_nH / log_CO / dNO in both predict_nebular_sed and "
        "predict_nebular_line_luminosities; all three must be threaded to it"
    )
    for backend in (CloudyGridBackend, BakedInBackend):
        assert _backend_accepted_params(backend) == frozenset(), (
            f"{backend.__name__} models none of these axes, so threading them "
            f"would rely on **kwargs swallowing the value silently"
        )


def test_the_threaded_values_actually_reach_the_backend(synthetic_ssp_wide, panchromatic_obs):
    """The sampler's value, not the signature default, must arrive at the call.

    The acceptance filter being correct is not the same as the component using
    it: the defect was a missing dict entry, and a filter nothing consults would
    restore it exactly. Spy on the call and compare against the *default*, which
    is what the backend received for as long as the bug existed.
    """
    backend_cls = pytest.importorskip("tengri.components.nebular.cloudy_cb19").CB19Backend

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=panchromatic_obs,
            sfh={"type": "dpl", "*": FIXED},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
            },
            neb={"type": "cb19", "*": FREE},
            redshift=Fixed(0.5),
        )

    from tengri.components.nebular.component import _backend_accepted_params

    seen = {}
    original = backend_cls.predict_nebular_sed

    # ``functools.wraps`` is load-bearing, not tidiness. The threading filter
    # decides what to pass by introspecting this very method's signature, so a
    # bare ``spy(self, *args, **kwargs)`` names none of the parameters and the
    # filter returns the empty set — the probe would then measure the spy and
    # report the defect whether or not it was present. ``wraps`` sets
    # ``__wrapped__``, which ``inspect.signature`` follows back to the original.
    @functools.wraps(original)
    def spy(self, *args, **kwargs):
        seen.update({k: v for k, v in kwargs.items() if k.startswith("neb_")})
        return original(self, *args, **kwargs)

    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    # A value the signature default is not, so "arrived" cannot be confused
    # with "defaulted": the defaults are 2.0 / -0.36 / 0.0.
    probe_values = {"neb_log_nH": 3.5, "neb_co": -0.9, "neb_dno": 0.2}
    params.update({k: np.float64(v) for k, v in probe_values.items() if k in params})

    # The filter is cached per class, so a warm entry from another test would
    # mask a spy that had perturbed the signature. Clear on both sides so this
    # test neither inherits nor leaves a cached answer — without it the result
    # depends on test order, which is how the signature problem above first
    # showed up as "passes in the file, fails alone".
    _backend_accepted_params.cache_clear()
    backend_cls.predict_nebular_sed = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.predict(params).rest_sed()
    finally:
        backend_cls.predict_nebular_sed = original
        _backend_accepted_params.cache_clear()

    assert seen, "the spy never fired — predict_nebular_sed was not called"
    for name, expected in probe_values.items():
        assert name in seen, (
            f"{name} never reached predict_nebular_sed; the backend fell back to "
            f"its signature default, which is the defect this guards"
        )
        assert float(seen[name]) == pytest.approx(expected), (
            f"{name} arrived as {float(seen[name])}, not the {expected} that was "
            f"passed — a stale or defaulted value is reaching the interpolator"
        )


def test_law_scope_is_read_from_the_signature_not_a_table():
    """A law registered later is scoped without editing the resolver.

    The three earlier scopings each shipped a maintained table, which is why
    each new group needed a code change. Deriving the scope from the law's own
    signature is what makes this general; assert that property directly rather
    than trusting it.
    """
    from tengri.parameters.groups import _law_shape_params

    assert _law_shape_params("calzetti") == frozenset()
    assert _law_shape_params("cardelli") == frozenset({"dust_Rv"})
    assert _law_shape_params("power_law") == frozenset({"dust_slope"})
    assert _law_shape_params("conroy2010") == frozenset({"dust_Rv", "dust_slope"})
    # An unregistered name must not raise — it resolves to "reads nothing", and
    # the caller leaves such a group unnarrowed.
    assert _law_shape_params("no_such_law") == frozenset()


def test_every_registered_law_is_covered_by_the_sweep():
    """The census must be the registry, not a list that drifts behind it.

    A scoping guard is only as wide as the set of variants it enumerates; a
    hand-maintained list silently stops covering laws added after it was
    written.
    """
    from tengri.components.dust.laws._registry import DUST_LAWS

    swept = {
        c["label"].removeprefix("dust[law=").rstrip("]")
        for c in CASES
        if c["label"].startswith("dust[law=")
    }
    assert swept == set(DUST_LAWS), (
        f"the law sweep and DUST_LAWS have diverged: "
        f"missing={sorted(set(DUST_LAWS) - swept)}, extra={sorted(swept - set(DUST_LAWS))}"
    )


def _wildcard_groups() -> frozenset[str]:
    """Every grammar group that accepts a wildcard, read from the grammar itself.

    ``_GROUP_STRUCTURAL_KEYS`` is what :func:`parse_groups` validates user keys
    against, so a group cannot be *accepted* without an entry here. Deriving the
    census denominator from it is what makes the census widen by itself.

    Relevance is derived, not listed: a group can only over-free if it takes a
    wildcard at all. ``foreground`` declares no ``'*'`` key and so drops out on
    its own merits rather than by being remembered.
    """
    from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS

    return frozenset(g for g, keys in _GROUP_STRUCTURAL_KEYS.items() if "*" in keys)


def test_every_wildcard_group_is_swept_or_excused():
    """Every group that takes a wildcard is swept here, or excused with a reason.

    The defect this file guards recurred three times because each group was
    scoped where it was noticed. A guard that enumerates only the groups someone
    thought of has the same failure mode one level up: it is green about the
    groups it covers and silent about the rest, and nothing distinguishes
    "swept and clean" from "never looked at".

    So the denominator comes from the grammar and every group must be accounted
    for explicitly — swept by a case, or named in :data:`_EXCUSED` with the
    measured reason it cannot be. Adding a group to the grammar without doing
    either fails here.
    """
    swept = {c["group"] for c in CASES}
    uncovered = _wildcard_groups() - swept - set(_EXCUSED)
    assert not uncovered, (
        f"{len(uncovered)} wildcard-accepting group(s) are neither swept nor excused: "
        f"{sorted(uncovered)}. Add a case, or an _EXCUSED entry stating why not."
    )


def test_no_group_is_excused_without_a_reason():
    """An excuse must carry its reason, and must not outlive the group it names.

    An empty or stale excuse is worse than none: it reads as a considered
    decision while covering nothing.
    """
    for group, reason in _EXCUSED.items():
        assert group in _wildcard_groups(), (
            f"_EXCUSED names '{group}', which the grammar no longer accepts a "
            f"wildcard for — delete the entry."
        )
        assert len(reason.split()) >= 5, f"_EXCUSED['{group}'] does not state a reason"
