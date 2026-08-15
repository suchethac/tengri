# SPDX-License-Identifier: BSD-3-Clause
"""Registry distinctness census -- does choosing a name change anything?

The emit census (``test_registry_components_emit.py``, #1738 step 3) asks
"does this name build, and does it emit?".  For ``dust_emission`` it really
does check emission; for the other six kinds it asserts only that the model
builds and an SED was computed.  So a name that builds, emits, and is
*bit-identical to a different name* passes it.

That is the #1738 fail-open one level up.  The component is not silent -- the
user's **choice** is.  Three shapes, all of which the emit census is blind to:

============  ==================================================  ============
shape         symptom                                             example
============  ==================================================  ============
``inert``     output bit-identical to the group-absent baseline    #1488
``twin``      two different names produce bit-identical output     #1684
``dead knob`` a free parameter has an exactly-zero gradient        this file
============  ==================================================  ============

The third is the sharpest, because a zero gradient is not merely cosmetic: the
fit cannot move that parameter, so the posterior returns the prior and reports
convergence.  Nothing warns.

**Instrument correctness.**  A census like this is worthless measured with the
wrong instrument, and every wrong answer here looks like a bug:

* IGM is an *observed-frame* transform.  ``rest_sed()`` cannot see it, and at
  z=0 there is nothing to see -- so IGM is measured on ``obs_sed()``, and at a
  redshift where the models under comparison actually differ (``_IGM_TEST_Z``).
* X-ray emits below 100 A and radio beyond 1 mm.  On the shared 100 A - 1 mm
  fixture grid both fall off the end and every model looks identical.  Each
  kind therefore gets a grid that spans its own band, and the scaffolding its
  own physics needs (``_KIND_SCAFFOLD``) -- an X-ray corona with no disc
  beneath it has no ``L_2500`` to act on.

Every one of those four rules was added after a wrong answer that looked like a
defect: two were published as issue comments before being caught, and one
(#1809) was filed as an issue and withdrawn.

**The ledgers are exact, not permissive.**  A declared coincidence must name a
knob that separates the pair, and the test *verifies the knob separates them*
-- so a declaration cannot quietly rot into a real alias.  A known defect must
cite its issue, and the test asserts the defect is still present: fixing it
turns the ledger entry red with "remove this entry", so the ledger cannot go
stale in either direction.

Measured on origin/main @ 57baf39ed, 2026-08-15.
"""

from __future__ import annotations

import gc
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FREE, Fixed, SEDModel, WavePrecomp
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract


# ── Instrument ───────────────────────────────────────────────────────

#: Puts the synthetic SSP on a real galaxy's luminosity scale (L_ir ~ 6e44
#: erg/s). A synthetic continuum is normally normalization-free, because every
#: comparison here is relative -- but not when one component's amplitude is
#: derived from another's. See :func:`_ssp`.
_LUMINOSITY_SCALE = 1e-15


def _ssp(lo_dex: float, hi_dex: float, n_wave: int = 2000) -> SSPData:
    """Synthetic SSP spanning a configurable decade range.

    Mirrors the ``synthetic_ssp_wide`` fixture but with a caller-chosen span,
    because a component can only be measured on a grid that contains its band.

    The continuum dies below the Lyman limit, which a plain ``(5000/lambda)**2``
    power law does not. That is not cosmetic on the wide grids here: continued
    to 0.1 A it makes the *stellar* flux in the X-ray band enormous, swamping
    any X-ray component and canceling to **negative** photometry. Measured on
    such a grid, ``lopez24`` appeared bit-identical to ``yang20`` under
    WavePrecomp -- a defect that does not exist. With the break in place the two
    separate by 14.3 on the exact path and 13.9 under WavePrecomp.

    The break is at 912 A rather than 1216 A on purpose: the 912-1216 A window
    is where the Lyman forest acts, and IGM prescriptions are compared through
    it (see ``_IGM_TEST_Z``), so it must keep flux.

    ``_LUMINOSITY_SCALE`` matters for the same reason the break does. Unscaled,
    this SSP gives ``L_ir = 6.2e59`` erg/s -- fifteen orders above a real galaxy
    -- and the radio free-free term, which tracks the ionizing output, inflates
    with it: measured 1.95e43 against an AGN synchrotron term of 1.02e30. Every
    AGN radio knob is then invisible underneath, and ``radio_dpl`` measures
    identical to ``radio_powerlaw``. Both are artifacts of the normalization.
    Scaled to ``L_ir ~ 6e44``, ``radio_loudness`` moves the 21 cm flux by 9761x
    and dpl separates from powerlaw by 0.69.
    """
    n_age = 25
    wave = jnp.logspace(lo_dex, hi_dex, n_wave)
    ages_gyr = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    lyman_break = 1.0 / (1.0 + jnp.exp(-(wave - 912.0) / 40.0))
    base = (5000.0 / wave) ** 2 * lyman_break * _LUMINOSITY_SCALE
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages_gyr - ages_gyr.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-30,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


def _observation() -> Observation:
    """5-band synthetic top-hat photometry; no filter-data files needed."""

    def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    curves = tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0, 7600.0, 9000.0))
    return Observation(photometry=Photometry(filters=curves))


# A luminous AGN. The X-ray corona models tie their emission to the disc's
# L_2500 via alpha_ox, so without a disc there is nothing for the corona
# prescription to act on and every X-ray name measures identical -- for an
# honest reason. This fixture is the difference between reporting four aliased
# X-ray models and reporting five: an earlier revision of this file omitted it
# and wrongly recorded ``lopez24`` as a fifth member of the class.
_LUMINOUS_AGN: dict = {
    "type": "composable",
    "disc": {"type": "multicolor"},
    "torus": {"type": "skirtor"},
    "log_lbol": Fixed(13.0),
}

# Grid span, output surface, and required scaffolding per kind. None of these
# are cosmetic. On the shared 100 A - 1 mm grid X-ray and radio emission falls
# off the end and every model in the menu measures identical, which is a false
# positive for exactly the defect this file exists to detect; and a corona
# model with no disc beneath it is the same trap one level up.
_KIND_INSTRUMENT: dict[str, tuple[float, float, str]] = {
    "xray": (-1.0, 7.0, "rest"),  # 0.1 A - 1 mm
    "radio": (2.0, 10.0, "rest"),  # 100 A - 1 m
    "igm": (2.0, 7.0, "obs"),  # observed frame; IGM is invisible in rest
    "dust": (2.0, 7.0, "rest"),
    "agn": (1.0, 7.0, "rest"),
}

#: Groups a kind needs present before its own choice can matter.
_KIND_SCAFFOLD: dict[str, dict] = {
    "xray": {"agn": _LUMINOUS_AGN},
    "radio": {"agn": _LUMINOUS_AGN},
}

#: Wavelength window [A] a kind's physics actually occupies, when it is narrower
#: than the grid. Comparing over the whole grid is not the same as comparing
#: where the component lives, and the difference is not academic: radio_dpl vs
#: radio_powerlaw measures 1.45e-09 at the 1 mm IR/mm boundary -- just above the
#: inertness tolerance, over 0.69% of the grid -- while in the radio band itself
#: it is 9.1e-16. Judged grid-wide, that boundary artifact reads as "the double
#: power-law works"; judged in the radio band, the truth is that it changes
#: nothing where radio physics lives.
_KIND_BAND: dict[str, tuple[float, float]] = {
    "radio": (1.0e9, 1.0e11),  # ~10 cm - 10 m
    "xray": (0.1, 1.0e2),  # 0.1 - 100 A
}

#: Redshift at which IGM prescriptions are compared.
#:
#: Not a free choice. ``asada25`` is Inoue+2014 plus the Asada+2025 proximate-CGM
#: damping wing, whose column follows
#: ``log10 N_HI(z) = 3.592 / (1 + exp(-1.841(z - 6))) + 18.001``. At z=3 that is
#: ~1e18 cm^-2, and its effect on the transmission is exactly zero: blueward of
#: Lyman-alpha the mean IGM is already saturated, so extra absorption takes 0 to
#: 0, and redward a 1e18 column's wing is below numerical significance. Measured:
#: asada25 and inoue14 are bit-identical at z=3 and z=5, and differ by up to 1.0
#: at z=7 and z=9.
#:
#: So comparing IGM models at z=3 says only that the CGM term is negligible
#: there, which is correct physics. An earlier revision of this file did exactly
#: that and recorded the pair as a defect; #1809 was filed on that basis and
#: withdrawn.
_IGM_TEST_Z = 7.0


def _sed(group: str, cfg: dict | None, *, extra: dict | None = None) -> jnp.ndarray:
    """Build one configuration and return its SED where the physics lives."""
    lo, hi, surface = _KIND_INSTRUMENT[group]
    groups: dict = {"sfh": {"type": "const"}}
    groups.update(_KIND_SCAFFOLD.get(group, {}))
    if cfg is not None:
        groups[group] = cfg
    groups.update(extra or {})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(ssp_data=_ssp(lo, hi), observation=_observation(), **groups)
        params = model.spec.sample(jax.random.PRNGKey(0))
        pred = model.predict(params)
        sed = jnp.asarray(pred.obs_sed() if surface == "obs" else pred.rest_sed())
        band = _KIND_BAND.get(group)
        if band is None:
            return sed
        axis = jnp.asarray(pred.wave_obs if surface == "obs" else pred.wave_rest)
        return sed[(axis >= band[0]) & (axis <= band[1])]


#: Two outputs closer than this are not a distinction a user could act on.
#: Matches ``_INERT_TOL`` in ``test_wildcard_scope_is_variant_aware.py`` so both
#: suites answer "does this choice matter?" against the same bar.
_INERT_TOL = 1e-9


def _indistinguishable(a: jnp.ndarray, b: jnp.ndarray) -> bool:
    """Is the difference between two choices below the point of mattering?

    Bit-identity is the usual signature here -- most classes in this file
    measure exactly 0.0, which is the fingerprint of one model reached by two
    names. But it is the wrong bar on its own: ``radio_dpl`` differs from the
    single power-law by ~3e-9 relative, which is not bit-identical and is also
    not a double power-law. Judging such a pair by ``array_equal`` alone
    decides on floating-point noise, in whichever direction the noise happens
    to fall.
    """
    if a.shape != b.shape:
        return False
    denom = jnp.where(jnp.abs(b) > 0, jnp.abs(b), 1.0)
    return bool(jnp.max(jnp.abs(a - b) / denom) <= _INERT_TOL)


# ── Ledger 1: coincidences that are correct by construction ──────────
# Each entry MUST name a knob that separates the pair. The knob is exercised
# in ``test_declared_coincidence_has_a_working_separator``, so an entry cannot
# decay into a real alias without turning red.

DECLARED_COINCIDENT: list[dict] = [
    {
        "group": "dust",
        "names": {"leitherer02", "noll09"},
        "reason": (
            "noll09 is Calzetti+L02 with a UV bump and a slope modification "
            "applied as (base + bump) * power_law, and BOTH modifications "
            "default to zero (dust_bump_strength=0.0, dust_delta=0.0). At "
            "default it is therefore the unmodified L02 base by construction, "
            "which is what leitherer02 selects. On the two_component path the "
            "slope is live (dust_delta gradient rel 0.265), so the separator "
            "below genuinely distinguishes them."
        ),
        "separator": (
            {"type": "two_component", "law_bc": "leitherer02", "law_diff": "leitherer02"},
            {
                "type": "two_component",
                "law_bc": "noll09",
                "law_diff": "noll09",
                "dust_delta": -0.8,
            },
        ),
    },
    {
        "group": "dust",
        "names": {"leitherer02", "noll09", "salim_sbl18"},
        "reason": (
            "noll09 applies a UV bump and a slope modification to the "
            "Calzetti+L02 base as (base + bump) * power_law; salim_sbl18 "
            "applies the same two as base * power_law + bump. Both "
            "modifications default to zero in each law's own signature, so at "
            "default each reduces to the unmodified L02 base -- which is what "
            "leitherer02 selects. Three parameterized generalizations meeting "
            "at their common origin.\n\n"
            "This was a KNOWN_UNDISTINCT class until #1808 was fixed, on the "
            "grounds that the separating knobs were dead on single_component. "
            "They are live now (dust_delta rel 0.152), so the claim is "
            "verifiable and belongs here."
        ),
        "separator": (
            {"type": "single_component", "law_bc": "leitherer02"},
            {"type": "single_component", "law_bc": "noll09", "dust_delta": -0.8},
        ),
    },
    {
        "group": "dust",
        "names": {"power_law", "vw07_diff"},
        "reason": (
            "vw07_diff is a fixed curve taking no parameters -- the Wild+2007 "
            "diffuse slope -- and power_law's n_slope defaults to -0.7, the "
            "same slope. One curve reached two ways, at default."
        ),
        "separator": (
            {"type": "single_component", "law_bc": "vw07_diff"},
            {"type": "single_component", "law_bc": "power_law", "dust_slope": -2.5},
        ),
    },
    {
        "group": "radio",
        "names": {"condon92", "radio_powerlaw"},
        "reason": (
            "condon92 predates the SF/AGN split and names the composite, so "
            "_legacy_radio_type_to_blocks resolves it to (bell2003, powerlaw) "
            "-- exactly what radio_powerlaw selects. Documented in that "
            "function's own docstring. One pair of names, one resolved model."
        ),
        "separator": (
            {"type": "condon92"},
            {"agn": {"type": "dpl"}},
        ),
    },
]
# This entry has a history worth keeping. An earlier revision could not declare
# it, because the separator below measured as inert and the pair was recorded as
# a #1461 defect instead. The separator was not inert; this file's SSP was
# unnormalized, so free-free swamped the AGN radio arm (see _LUMINOSITY_SCALE).
# The ledger refusing an unverifiable claim was right; what it was refusing was
# a bad measurement rather than a bad declaration.


# ── Ledger 2: defects this census detects and that are still open ────
# Recorded as equivalence CLASSES, not pairs: five names that are mutually
# bit-identical are one model under five names, and spelling that as ten pairs
# obscures it. Each class is asserted to be STILL mutually identical, so
# fixing one turns the entry red with "remove this entry" and the ledger
# cannot go stale in either direction.

KNOWN_UNDISTINCT: list[dict] = [
    {
        "group": "xray",
        "names": {"simple", "yang20"},
        "reason": (
            "A declared alias, not a defect: groups.py:1663 states that "
            "'simple' is 'yang20' physics, and both resolve to the same branch "
            "of the shared XRaySEDComponent. One model under two names by "
            "intent, so no knob can separate them -- which is why this is not "
            "a DECLARED_COINCIDENT entry.\n\n"
            "#1684 is otherwise closed. 'xray_aird' and 'agn_xray_corona' were "
            "both in this class; each now builds its own registered component "
            "and separates from yang20. That issue was the unfinished half of "
            "#1120, which closed after adding the names to the grammar "
            "allowlist but not to component_factory.\n\n"
            "'lopez24' was never a member. It separates once a disc is present "
            "-- its alpha_ox branch needs an L_2500 to act on -- and an early "
            "revision of this file recorded it here only because the fixture "
            "had no AGN. See _LUMINOUS_AGN."
        ),
    },
    # No radio entry. An earlier revision recorded {condon92, radio_powerlaw,
    # radio_dpl} here as a #1461 regression. That was wrong, and the cause was
    # this file's own SSP normalization -- see _LUMINOSITY_SCALE. radio_dpl
    # separates from radio_powerlaw by 0.69 (composable) and 0.54 (legacy
    # spelling), on both prediction paths, beside a control of 6.4e10.
    # condon92 == radio_powerlaw remains, and is legitimate: see
    # DECLARED_COINCIDENT.
    {
        "group": "igm",
        "names": {"inoue", "inoue14"},
        "reason": (
            "'inoue' is a documented back-compat alias of 'inoue14' -- both "
            "resolve to the same function through _IGM_ALIASES, and it was the "
            "internal default before 2026-05 while the dict grammar used "
            "'inoue14'. One model under two names by intent, so no knob can "
            "separate them and this is not a DECLARED_COINCIDENT entry.\n\n"
            "asada25 was in this class in an earlier revision and has been "
            "removed: it is bit-identical to inoue14 only below z~6, which is "
            "correct physics rather than a defect. See _IGM_TEST_Z. #1809 was "
            "filed on the earlier measurement and withdrawn."
        ),
    },
    # No dust entries. Two classes lived here, both attributed to the
    # single_component shape parameters being dead (#1808). They are live now,
    # and the coincidences that remain are coincidences at DEFAULT values --
    # which is a DECLARED_COINCIDENT claim, because a knob separates each pair.
]


def _covered(group: str, a: str, b: str) -> bool:
    """Is this identical pair accounted for by either ledger?"""
    for entry in (*DECLARED_COINCIDENT, *KNOWN_UNDISTINCT):
        if entry["group"] == group and {a, b} <= entry["names"]:
            return True
    return False


def _menu_names(group: str) -> list[str]:
    """Production entry names for one group's menu."""
    import tengri.registry as registry

    menu = {
        "xray": "list_xray_models",
        "radio": "list_radio_models",
        "igm": "list_igm_models",
        "dust": "list_dust_laws",
    }[group]
    fn = getattr(registry, menu, None)
    if fn is None:  # pragma: no cover - menu renamed
        pytest.skip(f"{menu} is not exported by tengri.registry")
    return [
        e["name"] for e in fn() if e.get("name") and e.get("status", "production") == "production"
    ]


def _type_cfg(group: str, name: str) -> dict:
    """The group config that selects ``name``.

    Most groups select by ``'type'``; attenuation laws are selected by
    ``law_bc`` inside a dust config, so the dust group needs its own spelling.
    ``single_component`` is used deliberately: it is the path on which the
    law shape parameters are dead, and that is what this census must see.
    """
    if group == "dust":
        return {"type": "single_component", "law_bc": name}
    return {"type": name}


@pytest.mark.parametrize("group", ["xray", "radio", "igm", "dust"])
def test_no_undeclared_twins(group: str) -> None:
    """No two production names may produce bit-identical output undeclared.

    A name that builds, emits, and is indistinguishable from a different name
    has delivered the user a silent substitution. Either it is correct by
    construction (DECLARED_COINCIDENT, with a knob that proves the models
    differ) or it is a defect (KNOWN_UNDISTINCT, with an issue).
    """
    names = [n for n in _menu_names(group) if n != "none"]
    seds: dict[str, jnp.ndarray] = {}
    for name in names:
        extra = {"redshift": Fixed(_IGM_TEST_Z)} if group == "igm" else None
        try:
            seds[name] = _sed(group, _type_cfg(group, name), extra=extra)
        except Exception as exc:
            pytest.fail(f"{group}={name!r} is advertised as production but failed to build: {exc}")

    undeclared: list[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if not _indistinguishable(seds[a], seds[b]):
                continue
            if _covered(group, a, b):
                continue
            undeclared.append(f"{group}: {a!r} and {b!r} are bit-identical")

    assert not undeclared, (
        "Undeclared silent substitution -- a user selecting one of these names "
        "receives another model's output, with no warning:\n  "
        + "\n  ".join(undeclared)
        + "\n\nIf the coincidence is correct by construction, add it to "
        "DECLARED_COINCIDENT with a knob that separates the pair. If it is a "
        "defect, file it and add it to KNOWN_UNDISTINCT."
    )


@pytest.mark.parametrize("entry", KNOWN_UNDISTINCT, ids=[e["group"] for e in KNOWN_UNDISTINCT])
def test_known_undistinct_ledger_is_current(entry: dict) -> None:
    """A ledgered defect must still be present -- else delete the entry.

    Without this, a fix leaves a stale entry behind that silently re-permits
    the defect if it ever regresses. The ledger is a statement of measured
    fact about the tree, so it is asserted in both directions.
    """
    group = entry["group"]
    names = sorted(entry["names"])
    extra = {"redshift": Fixed(_IGM_TEST_Z)} if group == "igm" else None
    seds = {n: _sed(group, _type_cfg(group, n), extra=extra) for n in names}

    reference = names[0]
    separated = [n for n in names[1:] if not _indistinguishable(seds[reference], seds[n])]
    assert not separated, (
        f"{group}: {separated} no longer match {reference!r} -- the defect "
        f"recorded as:\n    {entry['reason']}\nappears to be FIXED for those "
        "names.\nRemove them from this KNOWN_UNDISTINCT entry so they are "
        "protected by test_no_undeclared_twins from here on."
    )


@pytest.mark.parametrize(
    "entry", DECLARED_COINCIDENT, ids=[e["group"] for e in DECLARED_COINCIDENT]
)
def test_declared_coincidence_has_a_working_separator(entry: dict) -> None:
    """A declared coincidence must be separable by the knob it names.

    This is what stops the ledger becoming a place to hide real aliases: the
    claim "these are the same only at default settings" is not taken on trust,
    it is exercised.
    """
    cfg_a, cfg_b = entry["separator"]
    sed_a = _sed(entry["group"], cfg_a)
    sed_b = _sed(entry["group"], cfg_b)

    assert not _indistinguishable(sed_a, sed_b), (
        f"{entry['group']}: the separator declared for {sorted(entry['names'])} "
        "does not separate the models -- they are bit-identical even with it "
        "applied. The coincidence is therefore not 'correct at default "
        "settings'; it is a silent alias. Move this entry to KNOWN_UNDISTINCT "
        "and file an issue."
    )


# ── Layer 3: a free parameter the fit cannot move ────────────────────

# Parameters measured to have an exactly-zero gradient. A zero gradient means
# the fit cannot move the parameter at all: the posterior returns the prior
# and reports convergence, with no warning. Each entry is (dust config label,
# parameter) -> issue/reason.
#: Free parameters measured to have an exactly-zero gradient.
#:
#: **Empty since #1808 was fixed.** It held every attenuation-law shape
#: parameter on the single_component path -- dust_slope, dust_delta and
#: dust_bump_strength across six laws -- unreachable because the screen called
#: each law with no arguments and preferred a k(lambda) cached before any
#: parameter value existed. Measured after the fix: dust_slope rel 0.173,
#: dust_delta rel 0.152, beside dust_tau_v rel 1.43 on the same build, with the
#: two_component control unchanged to every digit.
#:
#: The fix passes a shape parameter only when spec provenance says a caller
#: asked for it, so each law's published default still stands when nobody did.
#: A first attempt passed them unconditionally and collapsed kriek_conroy,
#: narayanan_z and salim onto one curve; the guard against that lives in
#: tests/regression/bug/test_single_component_law_shape_params.py.
#:
#: Kept (empty) rather than deleted: a genuinely unfittable free parameter is a
#: real defect class, and the next one should land here rather than re-deriving
#: the mechanism.
KNOWN_DEAD_PARAMS: dict[tuple[str, str], str] = {}

# two_component builds carry no ledger entries on purpose: they are the
# control. If one ever acquires a dead parameter, the census should say so
# loudly rather than have it absorbed into an exemption.
_DUST_BUILDS: dict[str, dict] = {
    "single_component/power_law": {
        "type": "single_component",
        "law_bc": "power_law",
        "all_params": FREE,
    },
    "single_component/noll09": {
        "type": "single_component",
        "law_bc": "noll09",
        "all_params": FREE,
    },
    "single_component/salim_sbl18": {
        "type": "single_component",
        "law_bc": "salim_sbl18",
        "all_params": FREE,
    },
    "single_component/kriek_conroy": {
        "type": "single_component",
        "law_bc": "kriek_conroy",
        "all_params": FREE,
    },
    "single_component/calzetti": {
        "type": "single_component",
        "law_bc": "calzetti",
        "all_params": FREE,
    },
    "two_component/noll09": {
        "type": "two_component",
        "law_bc": "noll09",
        "law_diff": "noll09",
        "all_params": FREE,
    },
    "two_component/kriek_conroy": {
        "type": "two_component",
        "law_bc": "kriek_conroy",
        "law_diff": "kriek_conroy",
        "all_params": FREE,
    },
}


def _gradients(label: str) -> tuple[dict[str, float], float]:
    """d(sum photometry)/d(param) for every free parameter, plus the scale.

    The objective is ~1e-12 in these units, so an absolute gradient must be
    read against that scale -- 1e-12 is a healthy gradient here, not a dead
    one. Only an exactly-zero gradient is scale-free evidence.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=_ssp(2.0, 7.0),
            observation=_observation(),
            sfh={"type": "const"},
            dust=_DUST_BUILDS[label],
        )
        params = model.spec.sample(jax.random.PRNGKey(0))
        scale = float(jnp.sum(model.predict_photometry(params)))

        grads = {}
        for name in sorted(model.spec.free_params):

            def objective(value, _n=name):
                probe = dict(params)
                probe[_n] = value
                return jnp.sum(model.predict_photometry(probe))

            grads[name] = float(jax.grad(objective)(jnp.asarray(params[name], dtype=jnp.float64)))
    return grads, scale


@pytest.mark.parametrize("label", sorted(_DUST_BUILDS))
def test_no_undeclared_dead_free_parameter(label: str) -> None:
    """Every free parameter must have a nonzero gradient.

    A free parameter is a promise that a fit can constrain it. An exactly-zero
    gradient breaks that promise silently -- the sampler explores the prior and
    the posterior looks converged.
    """
    grads, scale = _gradients(label)
    assert scale != 0.0, f"{label}: the objective itself is zero; nothing can be measured"

    dead = [
        name for name, g in grads.items() if g == 0.0 and (label, name) not in KNOWN_DEAD_PARAMS
    ]
    assert not dead, (
        f"{label}: free parameter(s) with an exactly-zero gradient: {dead}.\n"
        "A fit cannot move these, so the posterior returns the prior and "
        "reports convergence. Either wire the parameter through to the "
        "physics, or stop advertising it as free."
    )


@pytest.mark.parametrize("key", sorted(KNOWN_DEAD_PARAMS))
def test_dead_param_ledger_is_current(key: tuple[str, str]) -> None:
    """A ledgered dead parameter must still be dead -- else delete the entry."""
    label, name = key
    grads, _ = _gradients(label)
    assert name in grads, (
        f"{label}: {name!r} is no longer a free parameter of this build; the "
        "KNOWN_DEAD_PARAMS entry is stale and should be removed."
    )
    assert grads[name] == 0.0, (
        f"{label}: {name!r} now has a nonzero gradient ({grads[name]:.3e}) -- "
        "it appears to be FIXED. Remove this entry from KNOWN_DEAD_PARAMS so "
        "the parameter is protected by test_no_undeclared_dead_free_parameter."
    )


def test_control_a_live_parameter_is_not_flagged() -> None:
    """Positive control: the census must be able to see a working parameter.

    Without this, a census that flagged nothing would be indistinguishable
    from a census whose instrument was broken -- which is the failure mode
    that let every defect above survive its own tests.
    """
    grads, scale = _gradients("two_component/noll09")
    assert grads.get("dust_tau_diff", 0.0) != 0.0, (
        "dust_tau_diff has a zero gradient, which contradicts the measurement "
        "this census was built on. The instrument is broken, not the code "
        "under test -- fix the harness before trusting any result in this file."
    )
    assert abs(grads["dust_tau_diff"]) / abs(scale) > 1e-3, (
        "dust_tau_diff's gradient is vanishing relative to the objective; the "
        "harness can no longer resolve a live parameter from a dead one."
    )


# ── Axis 3: the census must ask its question on the path fits take ───
#
# Everything above measures ``model.predict()`` -- the exact path. That is
# only half the surface a user meets. Every fit resolves ``approx="auto"`` to
# ``WavePrecomp`` for photometry (``Fitter``, ``PopulationFitter``,
# ``CatalogFitter``), so a name can be perfectly well behaved under
# ``predict()`` and still be unusable, or indistinguishable, in an actual fit.
#
# That gap is not hypothetical. ``smc`` and ``lmc`` -- both production, both
# among the most used curves in SED fitting -- built and predicted fine on the
# exact path and raised ``ValueError: Incompatible shapes for broadcasting``
# under ``WavePrecomp``, so selecting them failed at *fit* time, after the user
# had chosen the law and started fitting. No test in this file could see it,
# because no test in this file crossed the seam.

#: Filter centers [A] per kind, placed where that kind's physics lives.
#:
#: The optical set used by :func:`_observation` is right for ``dust`` and
#: ``igm`` and useless for ``xray`` and ``radio``: an X-ray corona moves no
#: optical band, so every corona name would measure identical and the axis
#: would report a clean bill of health it had not earned. This is the same
#: fixture-adequacy trap as :data:`_KIND_BAND`, one surface further out --
#: and it is the specific error that produced the withdrawn ``lopez24``
#: finding, which was measured against an optical-only objective.
_KIND_FILTERS: dict[str, tuple[float, ...]] = {
    "xray": (1.0, 5.0, 20.0, 60.0),  # 0.1 - 100 A
    "radio": (1.0e9, 5.0e9, 2.0e10, 8.0e10),  # ~10 cm - 10 m
    "igm": (3500.0, 4800.0, 6200.0, 9000.0),
    "dust": (3500.0, 4800.0, 6200.0, 9000.0),
}


def _observation_for(group: str) -> Observation:
    """Top-hat photometry in the band where ``group``'s physics lives."""

    def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.3g}")

    return Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in _KIND_FILTERS[group]))
    )


#: Wavelength samples for the two path tests.
#:
#: Deliberately coarser than the 1200 used by :func:`_sed`. These two tests
#: build every production name twice -- once per path -- and a ``WavePrecomp``
#: LUT is (sub-bands x ages x filters x z) per model. At the census's default
#: grid that is enough live arrays to be OOM-killed partway through (measured:
#: SIGKILL at 22 tests), which reads as a flaky suite rather than as a result.
#: Neither test needs spectral resolution: one asks "is this finite", the other
#: compares two models on the *same* grid, so a coarse grid shifts both sides
#: equally.
_PATH_N_WAVE = 400

#: z-table span for the precompute LUT. Must cover :data:`_IGM_TEST_Z`; the
#: sampling is coarse for the same memory reason, and is not load-bearing
#: because every comparison is between two models at one redshift.
_PATH_PRECOMP = WavePrecomp(n_z=8, z_min=0.0, z_max=8.0)


def _photometry(group: str, cfg: dict, approx) -> np.ndarray:
    """Band fluxes for one configuration, on the exact or the precompute path.

    Uses ``predict_photometry`` rather than ``predict`` deliberately: it is the
    surface ``WavePrecomp`` actually serves, and the one every fit calls.

    Drops every reference to the model before returning -- see
    :data:`_PATH_N_WAVE` for why holding them is not affordable here.
    """
    lo, hi, _ = _KIND_INSTRUMENT[group]
    groups: dict = {"sfh": {"type": "const"}}
    groups.update(_KIND_SCAFFOLD.get(group, {}))
    groups[group] = cfg
    if group == "igm":
        groups["redshift"] = Fixed(_IGM_TEST_Z)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=_ssp(lo, hi, n_wave=_PATH_N_WAVE),
            observation=_observation_for(group),
            approx=approx,
            **groups,
        )
        params = model.spec.sample(jax.random.PRNGKey(0))
        out = np.asarray(model.predict_photometry(params))

    del model, params
    gc.collect()
    return out


_PATHS: tuple[tuple[str, object], ...] = (("exact", None), ("WavePrecomp", _PATH_PRECOMP))

#: ``group -> {name: {path: photometry-or-exception}}``, built once per session.
#:
#: The two tests below ask different questions of the *same* measurement, and
#: computing it twice is what made this pair too heavy to run: each pass builds
#: every production name on both paths, and the models -- an SSP cube, a
#: SKIRTOR template library, and a ``WavePrecomp`` LUT apiece -- dominate peak
#: RSS. Sharing the pass halves it, and what stays resident afterwards is a
#: handful of 4-element arrays.
_PATH_CACHE: dict[str, dict[str, dict[str, np.ndarray | Exception]]] = {}


def _path_measurements(group: str) -> dict[str, dict[str, np.ndarray | Exception]]:
    """Photometry for every production name in ``group``, on both paths.

    Exceptions are captured rather than raised so one unbuildable name is
    reported as a finding by the reachability test instead of aborting the
    whole census before the other names are measured.
    """
    if group not in _PATH_CACHE:
        out: dict[str, dict[str, np.ndarray | Exception]] = {}
        for name in (n for n in _menu_names(group) if n != "none"):
            row: dict[str, np.ndarray | Exception] = {}
            for label, approx in _PATHS:
                try:
                    row[label] = _photometry(group, _type_cfg(group, name), approx)
                except Exception as exc:
                    # Captured, not swallowed: the reachability test reports it.
                    row[label] = exc
            out[name] = row
        _PATH_CACHE[group] = out
    return _PATH_CACHE[group]


@pytest.mark.parametrize("group", ["xray", "radio", "igm", "dust"])
def test_every_production_name_predicts_on_both_paths(group: str) -> None:
    """A production name must be usable in a fit, not merely in ``predict()``.

    Reachability, asserted for all 36 production names across the four menus.
    Non-vacuous: ``smc`` and ``lmc`` raise here before the ``_pei92_curve``
    rank fix, which is the defect this axis was added to generalize.

    "Finite" is the bar rather than "distinct" on purpose -- distinctness is
    the job of :func:`test_no_undeclared_twins` and of the path-agreement test
    below. This one only asserts that choosing the name does not detonate, or
    silently produce garbage, on the path a fit takes.
    """
    failures: list[str] = []
    for name, row in _path_measurements(group).items():
        for label, _ in _PATHS:
            phot = row[label]
            if isinstance(phot, Exception):
                failures.append(f"{group}={name!r} on {label}: {type(phot).__name__}: {phot}")
            elif not np.all(np.isfinite(phot)):
                failures.append(f"{group}={name!r} on {label}: non-finite photometry")
            elif not np.all(phot > 0.0):
                failures.append(f"{group}={name!r} on {label}: photometry is not positive")

    assert not failures, (
        "A name advertised as production is not usable on a path a user reaches.\n  "
        + "\n  ".join(failures)
        + "\n\nEvery fitter resolves approx='auto' to WavePrecomp for photometry, so a "
        "failure on that path is a failure at fit time -- after the user has chosen "
        "the model and started fitting."
    )


@pytest.mark.parametrize("group", ["xray", "radio", "igm", "dust"])
def test_a_distinction_survives_the_precompute_path(group: str) -> None:
    """A choice that matters exactly must still matter under the LUT.

    The failure this guards against is subtler than a crash: two names that are
    genuinely different models on the exact path collapsing to bit-identical
    under ``WavePrecomp``. The user picks a model, the fit silently runs a
    different one, and nothing raises. ``predict()`` would show the
    distinction, so no test above this line could catch it.

    Currently green for every pair in all four menus. It is a regression guard,
    not a live defect -- and it is stated that way rather than skipped, because
    the pair that made this axis worth writing (``lopez24`` vs the other
    coronae) turned out to be an artifact of measuring X-ray models against an
    optical-only objective. Measured in the X-ray band with a disc beneath the
    corona, the distinction survives precompute intact.
    """
    rows = _path_measurements(group)
    # A name that failed to build is the reachability test's finding, not this
    # one's; including it here would report the same defect twice in different
    # words.
    names = [
        n
        for n, row in rows.items()
        if not any(isinstance(row[label], Exception) for label, _ in _PATHS)
    ]
    exact = {n: rows[n]["exact"] for n in names}
    precomp = {n: rows[n]["WavePrecomp"] for n in names}

    def _rel(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
        return float(np.max(np.abs(a - b) / denom))

    erased: list[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r_exact, r_precomp = _rel(exact[a], exact[b]), _rel(precomp[a], precomp[b])
            if r_exact > _INERT_TOL and r_precomp <= _INERT_TOL:
                erased.append(
                    f"{group}: {a!r} vs {b!r} differ by {r_exact:.3e} exactly "
                    f"but {r_precomp:.3e} under WavePrecomp"
                )

    assert not erased, (
        "The precompute path erased a distinction the exact path makes. In a fit "
        "-- which is always the precompute path for photometry -- these names are "
        "the same model, with no warning:\n  " + "\n  ".join(erased)
    )
