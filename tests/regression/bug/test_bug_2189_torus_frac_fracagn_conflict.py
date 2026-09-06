# SPDX-License-Identifier: BSD-3-Clause
"""#2189 (RULING R15): agn_torus_frac silently discarded under fracAGN.

``AGNSEDComponent.apply`` (``components/agn/component.py``) overrides
whatever ``agn_torus_frac`` a caller supplies with a value derived from the
dust-absorbed stellar luminosity (the CIGALE skirtor2016
``agn_power = L_absorbed x fracAGN/(1-fracAGN)`` coupling) whenever fracAGN
(``agn_ir_frac``) is active. The override computation reads only
``agn_ir_frac``/``agn_torus_frac``/the absorbed luminosity -- it runs BEFORE
the runner's separate ``agn_norm`` branch and never references it -- so it
applies regardless of ``agn_norm``. Measured on this branch (5 torus types,
``sfh=delayed``, ``dust_attenuation=two_component``, ``dust_emission=dl07``,
WISE/2MASS/Spitzer bands, ``norm='independent'`` explicit): varying
``agn_torus_frac`` 0.05 -> 0.95 with fracAGN left at its registry default
(0.0, inactive) changes photometry by 8.7x-17.1x (max relative diff); with
fracAGN explicitly active (measured under the DEFAULT ``agn_norm``,
``'cigale_joint'``) it changes photometry by EXACTLY 0.0 -- bit-identical,
for every torus type.

Two guards close this silent-override trap:

1. A build-time ``ConfigError`` (``sed_model.py::
   _validate_torus_frac_fracagn_conflict``) when a caller explicitly names
   BOTH ``agn_torus_frac`` (a prior or ``Fixed`` value) and an active fracAGN
   -- a contradiction the caller should be told about, not silently resolved
   in fracAGN's favor.
2. The ``agn.torus`` sub-block's own ``'all_params': FREE`` wildcard never
   frees ``agn_torus_frac`` when fracAGN is active
   (``parameters.groups._agn_ir_frac_explicit_and_active``, consulted while
   computing wildcard scopes) -- silently narrower, not an error, since a
   wildcard asks for "whatever varies", not a specific name.

This file asserts BOTH the refusal (guard 1) and the measured LIVE case
(fracAGN inactive) -- the refusal alone would not prove agn_torus_frac
actually works when it is safe to use it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel
from tengri.config.exceptions import ConfigError

pytestmark = [pytest.mark.regression_bug, pytest.mark.contract]

#: Torus types measured; F1's own probe covered these five.
_TORUS_TYPES = ("fritz", "cat3d_wind", "nenkova", "nenkova_agnfitter", "simple", "skirtor")

#: Measured (this branch) max relative photometry diff for agn_torus_frac
#: 0.05 -> 0.95, fracAGN inactive, norm='independent' explicit. Every value
#: is well above the _LIVE_FLOOR the regression test asserts, with margin.
_MEASURED_LIVE_REL_DIFF = {
    "fritz": 8.7394,
    "cat3d_wind": 17.130,
    "nenkova": 16.948,
    # R32: this type declares exactly ONE parameter of its own,
    # agn_torus_frac, so the #2189 narrowing empties its whole wildcard scope
    # when fracAGN is active. That is correct physics, not a scoping defect --
    # and the parameter is emphatically live when fracAGN is inactive, which is
    # what this row measures.
    "nenkova_agnfitter": 16.885,
    "simple": 11.460,
    "skirtor": 16.448,
}
_LIVE_FLOOR = 5.0
assert all(v > _LIVE_FLOOR for v in _MEASURED_LIVE_REL_DIFF.values()), (
    "measured numbers must exceed the floor the regression test asserts"
)

_BANDS = (
    "SLOAN_SDSS_g",
    "2MASS_2MASS_Ks",
    "WISE_WISE_W1",
    "WISE_WISE_W2",
    "WISE_WISE_W3",
    "WISE_WISE_W4",
    "Spitzer_IRAC_I4",
    "Spitzer_MIPS_24mu",
)


@pytest.fixture(scope="module")
def ssp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tengri.load_ssp()


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(list(_BANDS)))


def _build(ssp, obs, torus_type: str, *, torus_frac, ir_frac=None):
    """One composable-AGN build with the given torus type, 'norm':
    'independent' explicit (no cross-block energy coupling to confound the
    measurement), and an explicit ``torus_frac``. ``ir_frac`` is left unset
    (registry default 0.0, inactive) unless given.
    """
    agn = {
        "type": "composable",
        "disc": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "torus": {"type": torus_type, "all_params": Fixed(DEFAULT), "torus_frac": torus_frac},
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
    }
    if ir_frac is not None:
        agn["ir_frac"] = ir_frac
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dl07", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn=agn,
            redshift=Fixed(0.1),
        )


def _photometry(model) -> np.ndarray:
    p = model.spec.sample_batch(__import__("jax").random.PRNGKey(0), 1)
    p = {k: v[0] for k, v in p.items()}
    return np.asarray(model.predict_photometry(p))


@pytest.mark.parametrize("torus_type", _TORUS_TYPES)
def test_explicit_torus_frac_and_active_fracagn_raises(ssp, obs, torus_type):
    """Guard 1: naming both agn_torus_frac AND an active fracAGN explicitly
    must raise ConfigError naming both keys and the two ways out."""
    with pytest.raises(ConfigError) as excinfo:
        _build(ssp, obs, torus_type, torus_frac=Fixed(0.3), ir_frac=Fixed(0.5))
    msg = str(excinfo.value)
    assert "agn_torus_frac" in msg
    assert "fracAGN" in msg or "agn_ir_frac" in msg
    assert "#2189" in msg


@pytest.mark.parametrize("torus_type", _TORUS_TYPES)
def test_explicit_torus_frac_is_live_when_fracagn_inactive(ssp, obs, torus_type):
    """The LIVE case (not merely the refusal): with fracAGN left at its
    registry default, agn_torus_frac 0.05 -> 0.95 must change photometry by
    at least _LIVE_FLOOR (max relative diff) -- well below every measured
    value in _MEASURED_LIVE_REL_DIFF, so this is a floor, not a repeat of
    the exact measurement."""
    a = _photometry(_build(ssp, obs, torus_type, torus_frac=Fixed(0.05)))
    b = _photometry(_build(ssp, obs, torus_type, torus_frac=Fixed(0.95)))
    rel = float(np.max(np.abs(b - a) / np.maximum(np.abs(a), 1e-300)))
    assert rel > _LIVE_FLOOR, (
        f"{torus_type}: agn_torus_frac 0.05->0.95 changed photometry by only "
        f"{rel:.4e} (floor {_LIVE_FLOOR}) with fracAGN inactive -- expected "
        f"live (measured {_MEASURED_LIVE_REL_DIFF[torus_type]:.4e} on this branch)"
    )


@pytest.mark.parametrize("torus_type", _TORUS_TYPES)
def test_wildcard_never_frees_torus_frac_when_fracagn_active(ssp, obs, torus_type):
    """Guard 2: agn.torus={'all_params': FREE} must NOT free agn_torus_frac
    when fracAGN is explicitly active -- it silently narrows (no error,
    unlike guard 1's explicit-vs-explicit case), since freeing a parameter
    AGNSEDComponent.apply() is about to override would hand the sampler a
    dead dimension. Every OTHER declared torus parameter must still be
    freed normally (the narrowing is targeted, not a wholesale wildcard
    failure)."""
    agn = {
        "type": "composable",
        "disc": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "torus": {"type": torus_type, "all_params": FREE},
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
        "ir_frac": Fixed(0.5),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dl07", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn=agn,
            redshift=Fixed(0.1),
        )
    free = set(model.spec.free_params)
    assert "agn_torus_frac" not in free, (
        f"{torus_type}: agn.torus={{'all_params': FREE}} froze agn_torus_frac "
        f"as free even though fracAGN is explicitly active -- it should be "
        f"silently narrowed out (guard 2)"
    )
    from tengri.parameters.groups import _agn_subblock_declared_params

    declared = _agn_subblock_declared_params("torus", torus_type) or frozenset()
    other_declared = declared - {"agn_torus_frac"}
    if other_declared:
        assert other_declared <= free, (
            f"{torus_type}: narrowing agn_torus_frac out of the wildcard scope "
            f"also dropped OTHER declared torus parameters: "
            f"{sorted(other_declared - free)}"
        )


# ──────────────────────────────────────────────────────────────────────────
# The same guard, through the OTHER placements the grammar honors.
# ──────────────────────────────────────────────────────────────────────────

#: Every (location, key) a caller can legally write fracAGN under. The
#: grammar accepts a shared parameter inside a sub-block as well as at the
#: agn top level (``_build_agn_search_view``'s documented cross-level
#: acceptance), and both the canonical and the legacy spelling. The guard
#: read only the top-level dict, so the sub-block placements slipped past it:
#: measured with ``torus={'type': 'fritz', 'all_params': FREE,
#: 'ir_frac': Fixed(0.5)}``, ``agn_torus_frac`` was freed, its gradient was
#: 0.0 across five seeds, and sweeping it 0.05 -> 0.95 moved photometry by
#: 0.0 -- the exact dead dimension #2189 exists to prevent, reachable through
#: a spelling the grammar advertises.
_FRACAGN_PLACEMENTS = (
    ("<top>", "ir_frac"),
    ("<top>", "agn_ir_frac"),
    ("<top>", "fracAGN"),
    ("<top>", "agn_fracAGN"),
)

#: The same four spellings written inside a sub-block. fracAGN is a top-level
#: key: it drives the cross-block normalization stage, not one block's physics.
#: Every one of these is now refused (R38) rather than half-honored -- see
#: ``test_a_sub_block_fracagn_key_is_refused``.
_FRACAGN_SUBBLOCK_PLACEMENTS = tuple(
    (location, key)
    for location in ("torus", "disc", "nlr", "blr", "feii", "atten")
    for key in ("ir_frac", "agn_ir_frac", "fracAGN", "agn_fracAGN")
)


def _build_with_placement(
    ssp,
    obs,
    *,
    location,
    key,
    torus_frac=None,
    torus_wildcard=None,
    torus_type="fritz",
    mute=True,
):
    """One build with fracAGN written at ``location`` under ``key``.

    ``mute=False`` lets the build's own advisories through, for the callers
    whose subject IS one of those warnings -- a blanket suppression there
    would make the assertion pass on silence.
    """
    torus: dict = {"type": torus_type}
    if torus_wildcard is not None:
        torus["all_params"] = torus_wildcard
    else:
        torus["all_params"] = Fixed(DEFAULT)
    if torus_frac is not None:
        torus["torus_frac"] = torus_frac
    agn = {
        "type": "composable",
        "disc": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "torus": torus,
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
    }
    if location == "<top>":
        agn[key] = Fixed(0.5)
    else:
        sub = dict(agn.get(location) or {"type": "none"})
        sub[key] = Fixed(0.5)
        agn[location] = sub
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if mute else "always")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dl07", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn=agn,
            redshift=Fixed(0.1),
        )


@pytest.mark.parametrize(
    ("location", "key"), _FRACAGN_PLACEMENTS, ids=[f"{loc}:{k}" for loc, k in _FRACAGN_PLACEMENTS]
)
def test_explicit_torus_frac_raises_for_every_fracagn_placement(ssp, obs, location, key):
    """Guard 1 must see fracAGN wherever the builder would resolve it."""
    with pytest.raises(ConfigError) as excinfo:
        _build_with_placement(ssp, obs, location=location, key=key, torus_frac=Fixed(0.3))
    msg = str(excinfo.value)
    assert "agn_torus_frac" in msg
    assert "#2189" in msg


@pytest.mark.parametrize(
    ("location", "key"), _FRACAGN_PLACEMENTS, ids=[f"{loc}:{k}" for loc, k in _FRACAGN_PLACEMENTS]
)
def test_wildcard_narrows_for_every_fracagn_placement(ssp, obs, location, key):
    """Guard 2 likewise: the torus wildcard must not free a dead dimension
    because fracAGN was written one level down."""
    model = _build_with_placement(ssp, obs, location=location, key=key, torus_wildcard=FREE)
    assert "agn_torus_frac" not in set(model.spec.free_params), (
        f"fracAGN written as {key!r} at {location!r} escaped the #2189 "
        f"narrowing: agn_torus_frac was freed and is dead there"
    )


def test_the_narrowing_never_manufactures_an_empty_scope():
    """R32, re-measured: no registered torus type is left with nothing.

    The review found ``torus='nenkova_agnfitter'`` emptied by this narrowing
    -- its declared set was ``{'agn_torus_frac'}`` alone -- which would make
    the post-#2207 refusal read *"covers no parameters"* to a user whose real
    problem is fracAGN. That premise no longer holds: R34's partition gave
    ``agn_theta_torus`` (the gray Type-1/2 mask's own opening angle, which this
    block reads) its torus owner, so the scope narrows to ``{agn_theta_torus}``
    rather than to nothing.

    This is the guard rather than the message, because a message for a
    configuration nothing can reach is dead code. If a torus type ever does
    declare ``agn_torus_frac`` and nothing else, this fires and the question
    of what to say comes back with it.
    """
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS
    from tengri.parameters.groups import _agn_subblock_declared_params

    emptied = []
    for torus_type in sorted(AGN_BLOCKS["torus"]):
        if torus_type == "none":
            continue
        declared = _agn_subblock_declared_params("torus", torus_type) or frozenset()
        if declared and not (declared - {"agn_torus_frac"}):
            emptied.append(torus_type)
    assert not emptied, (
        f"the #2189 narrowing empties the whole agn.torus wildcard scope for "
        f"{emptied}: those builds need a message naming fracAGN as the cause, "
        f"not a bare 'covers no parameters'"
    )


def test_nenkova_agnfitter_keeps_a_nonempty_scope_under_active_fracagn(ssp, obs):
    """The measured half of the guard above, end to end."""
    model = _build_with_placement(
        ssp,
        obs,
        location="<top>",
        key="ir_frac",
        torus_wildcard=FREE,
        torus_type="nenkova_agnfitter",
    )
    free = {p for p in model.spec.free_params if p.startswith("agn_")}
    assert "agn_torus_frac" not in free, sorted(free)
    assert "agn_theta_torus" in free, sorted(free)


# ──────────────────────────────────────────────────────────────────────────
# R38: fracAGN is a top-level key, and saying so is better than half-honoring
# it. Written inside a sub-block, the four spellings used to split three ways.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("location", "key"),
    _FRACAGN_SUBBLOCK_PLACEMENTS,
    ids=[f"{loc}:{k}" for loc, k in _FRACAGN_SUBBLOCK_PLACEMENTS],
)
def test_a_sub_block_fracagn_key_is_refused_with_placement_advice(ssp, obs, location, key):
    """Measured before this guard, the four spellings behaved three ways.

    With ``agn={'torus': {'type': 'fritz', 'all_params': FREE,
    'fracAGN': Fixed(0.5)}}``:

    * the builder silently ignored the key -- ``agn_ir_frac`` stayed 0.0, i.e.
      fracAGN INACTIVE;
    * the #2189 detector's legacy scan saw it anyway and narrowed
      ``agn_torus_frac`` out of the torus wildcard -- a live dimension
      (measured 8.74 relative photometry change for ``fritz`` with fracAGN
      inactive) lost to a key with no effect;
    * and the #2189 conflict guard, which reads provenance, did not fire, so
      the two halves of one guard disagreed.

    The canonical spellings were honored by the builder but placed a key that
    governs the runner's cross-block normalization stage inside one block's
    dict, where it reads as that block's parameter. One rule for all four:
    fracAGN belongs at the agn top level, and writing it elsewhere says so.
    """
    with pytest.raises(ValueError) as excinfo:
        _build_with_placement(ssp, obs, location=location, key=key, torus_wildcard=FREE)
    message = str(excinfo.value)
    assert key in message, message
    assert location in message, message
    assert "agn_ir_frac" in message or "fracAGN" in message, message


def test_the_legacy_scan_reads_only_the_top_level(ssp, obs):
    """The detector sees exactly what the builder honors.

    Unit-level, because the build now raises: a legacy spelling that appears
    only inside a sub-block must not make the detector call fracAGN active,
    which is what silently narrowed ``agn_torus_frac`` away.
    """
    from tengri.parameters.groups import _agn_ir_frac_explicit_and_active

    assert not _agn_ir_frac_explicit_and_active(
        {"type": "composable", "torus": {"type": "fritz", "fracAGN": Fixed(0.5)}}
    )
    assert not _agn_ir_frac_explicit_and_active(
        {"type": "composable", "disc": {"type": "skirtor", "agn_fracAGN": Fixed(0.5)}}
    )
    # The top level still activates it, by either spelling.
    assert _agn_ir_frac_explicit_and_active({"type": "composable", "fracAGN": Fixed(0.5)})
    assert _agn_ir_frac_explicit_and_active({"type": "composable", "agn_ir_frac": Fixed(0.5)})


def test_the_torus_wildcard_keeps_torus_frac_when_no_top_level_fracagn(ssp, obs):
    """The control: with fracAGN nowhere at the top level, nothing narrows."""
    agn = {
        "type": "composable",
        "disc": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "fritz", "all_params": FREE},
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dl07", "all_params": Fixed(DEFAULT)},
            neb={"type": "none"},
            agn=agn,
            redshift=Fixed(0.1),
        )
    assert "agn_torus_frac" in set(model.spec.free_params)
