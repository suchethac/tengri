# SPDX-License-Identifier: BSD-3-Clause
r"""``neb={'type': 'cb19', 'grid': <path>}`` reaches the backend (#2220).

https://github.com/suchethac/tengri/issues/2220

Before this fix, ``grid`` was a legal structural key of every ``neb`` variant
(``_GROUP_STRUCTURAL_KEYS["neb"]`` carried it unconditionally), and the
``cloudy``, ``mappings`` and ``mappings_agn`` branches of ``_translate_neb``
already read it -- but the ``cb19`` branch did not, and nothing downstream
refused the key either: ``parse_groups(neb={'type': 'cb19', 'grid':
'/bogus.h5'})`` returned a ``Parameters`` holding the path nowhere, so
``CB19Backend`` silently fell back to its own module-level default. Every
existing cb19 regression test reached a non-default grid by monkeypatching
``cloudy_cb19._DEFAULT_PATH`` -- the only route that worked, and the evidence
that no grammar route existed.

This file pins the grammar route directly, **without** that monkeypatch:
the key now lowers to ``nebular_cb19_grid_path``, survives ``Parameters``
construction and the cache-key policy, round-trips through
``spec.to_groups()``, and reaches ``CB19Backend(grid_path=...)`` so
``model._nebular_backend.grid_path`` is exactly the path given, and the
resulting predictions are bit-identical to the pre-existing monkeypatch
route on the same synthetic grid. ``grid`` is now legal only for the four
``neb`` types that actually read it (``cloudy``, ``cb19``, ``mappings``,
``mappings_agn``); every other type (``cue``, ``ssp``, ``none``) refuses it
by name instead of silently accepting and dropping it, which was the same
disease one layer over -- see test (f).
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tests._cb19_grid import write_synthetic_cb19_grid

pytestmark = pytest.mark.regression_bug

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SSP_FILE = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"

#: Headline lines (see ``EmissionLines``) that fall inside the synthetic
#: grid's ten-line catalog (``tests/_cb19_grid.py::CB19_LINE_WAVES_AA``):
#: H-beta and H-alpha are hydrogen recombination lines (fixed Case B ratio),
#: [OIII] 5007 and [NII] 6584 each carry a metal-line axis factor.
_HEADLINE_LINES = ("hbeta", "oiii_5007", "halpha", "nii_6584")


@pytest.fixture(scope="module")
def ssp():
    """Bare-stellar SSP: CB_19 adds nebular emission, so a wNE grid would double it."""
    if not _SSP_FILE.is_file():
        pytest.skip(f"bare-stellar SSP not present: {_SSP_FILE}")
    return tengri.load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def observation():
    """Five optical top-hats spanning the strong lines at z = 0.1."""
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    def _tophat(center, frac=0.12, n=40):
        wave = np.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = np.sin(np.linspace(0.0, np.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    curves = tuple(_tophat(c) for c in (3550.0, 4700.0, 6200.0, 7500.0, 8900.0))
    return Observation(photometry=Photometry(filters=curves))


def _write_grid(path):
    """A synthetic cb19 grid, perturbed so it cannot coincide with the default.

    ``tests/conftest.py::_create_cb19_fixture_if_missing`` writes a grid via
    this exact same generator (:func:`write_synthetic_cb19_grid`, deterministic,
    no RNG) at the packaged default path whenever nothing is there yet -- which
    on a bare worktree is every run. Without this scale factor, a build that
    silently falls back to the default (the pre-#2220 defect) would load
    content bit-identical to the one this test asks for by name, and the
    equality assertions in (b) below would pass for the wrong reason. The 2x
    scale keeps the grid non-degenerate (``check_cb19_free_params`` still
    sees every axis vary) while guaranteeing the two grids disagree.
    """
    grid_path = write_synthetic_cb19_grid(path)
    import h5py

    with h5py.File(grid_path, "r+") as f:
        ratios = f["grids/SSP/Kroupa01/mu100/line_ratios"]
        ratios[...] = np.asarray(ratios) * 2.0
    return grid_path


#: Every cb19 grid-axis parameter pinned to a fixed value, so both builds
#: below carry zero free nebular parameters and the comparison in (b) is not
#: also a random-sample comparison.
_NEB_AXIS_PARAMS = {
    "logU": Fixed(-3.0),
    "logZ_gas": Fixed(0.0),
    "log_nH": Fixed(2.0),
    "co": Fixed(-0.36),
    "dno": Fixed(0.0),
    "fesc": Fixed(0.0),
}


def _build_common_kwargs(ssp_data, obs, neb_dict):
    return dict(
        ssp_data=ssp_data,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Fixed(10.0),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "other_params": Fixed(DEFAULT),
        },
        neb=neb_dict,
        redshift=Fixed(0.1),
    )


def _build_via_key(ssp_data, obs, grid_path):
    """Build a cb19 model naming ``grid_path`` through the grammar's ``grid`` key.

    No ``_DEFAULT_PATH`` monkeypatch anywhere in this helper -- the whole
    point of #2220 is that this route exists at all.
    """
    neb_dict = {"type": "cb19", "grid": str(grid_path), "other_params": Fixed(DEFAULT)}
    neb_dict.update(_NEB_AXIS_PARAMS)
    return SEDModel.build(**_build_common_kwargs(ssp_data, obs, neb_dict))


def _build_via_monkeypatch(ssp_data, obs, grid_path, monkeypatch):
    """Build the same model reaching ``grid_path`` via the pre-#2220 route.

    Mirrors ``test_bug_2181_cb19_placeholder_grid.py::_build``: every cb19
    test before this one reached a non-default grid this way. Used only as
    the (b) comparison baseline, never as a substitute for the key route.
    """
    monkeypatch.setattr("tengri.components.nebular.cloudy_cb19._DEFAULT_PATH", Path(grid_path))
    neb_dict = {"type": "cb19", "other_params": Fixed(DEFAULT)}
    neb_dict.update(_NEB_AXIS_PARAMS)
    return SEDModel.build(**_build_common_kwargs(ssp_data, obs, neb_dict))


def test_grammar_grid_key_reaches_the_backend(ssp, observation, tmp_path):
    """(a) The grammar accepts ``grid`` for cb19 and the backend gets it exactly.

    Fails on origin/main: the ``cb19`` branch of ``_translate_neb`` drops the
    key, so ``model._nebular_backend.grid_path`` ends up at the module
    default instead of the synthetic path.
    """
    grid_path = _write_grid(tmp_path / "cb19_templates.h5")
    model = _build_via_key(ssp, observation, grid_path)
    assert model._nebular_backend.grid_path == Path(grid_path)


def test_grid_key_route_matches_the_monkeypatch_route(ssp, observation, tmp_path, monkeypatch):
    """(b) End to end through ``predict``, the key route reproduces the old one.

    Fails on origin/main: ``_build_via_key`` silently loads whatever grid
    ``_DEFAULT_PATH`` resolves to (not the synthetic one), so the two
    predictions are built from different data and finite/nonzero and
    bit-identical assertions below have no reason to hold.
    """
    grid_path = _write_grid(tmp_path / "cb19_templates.h5")

    model_key = _build_via_key(ssp, observation, grid_path)
    params_key = dict(model_key.spec.sample(jax.random.PRNGKey(0)))
    flux_key = np.asarray(model_key.predict_photometry(params_key))
    lines_key = model_key.predict(params_key).lines

    model_patched = _build_via_monkeypatch(ssp, observation, grid_path, monkeypatch)
    params_patched = dict(model_patched.spec.sample(jax.random.PRNGKey(0)))
    flux_patched = np.asarray(model_patched.predict_photometry(params_patched))
    lines_patched = model_patched.predict(params_patched).lines

    assert np.isfinite(flux_key).all(), flux_key
    assert np.any(flux_key != 0.0), "photometry is bit-exactly zero -- the grid never loaded"

    for name in _HEADLINE_LINES:
        value = float(getattr(lines_key, name))
        assert np.isfinite(value) and value != 0.0, f"{name} is not finite/nonzero: {value}"

    np.testing.assert_allclose(flux_key, flux_patched, rtol=1e-12, atol=0.0)
    for name in _HEADLINE_LINES:
        np.testing.assert_allclose(
            float(getattr(lines_key, name)),
            float(getattr(lines_patched, name)),
            rtol=1e-12,
            atol=0.0,
            err_msg=f"line {name} differs between the grid-key route and the monkeypatch route",
        )


def test_misspelled_grid_key_still_did_you_means(tmp_path):
    """(c) ``'gird'`` (typo) must still raise with the did-you-mean hint.

    ``grid`` is legal for the ``cb19`` type both before and after this fix
    (only its per-type legality changed -- see test (f); ``cb19`` stayed
    accepted throughout), so the did-you-mean hint held pre-fix too. Pinned
    here so the fix in (a)/(b) cannot be read as also being what makes the
    typo detectable, and so a later change that makes ``grid`` type-specific
    cannot silently drop the suggestion for a type where it is legal.
    """
    from tengri.parameters.groups import parse_groups

    bogus = str(tmp_path / "unused.h5")
    with pytest.raises(ValueError, match=r"Did you mean: grid\?"):
        parse_groups(neb={"type": "cb19", "gird": bogus}, redshift=Fixed(0.1))


def test_missing_grid_key_path_names_that_path_not_the_default(ssp, observation, tmp_path):
    """(d) A nonexistent ``grid`` path raises ``FileNotFoundError`` naming itself.

    Must not silently fall back to ``_DEFAULT_PATH`` and either succeed
    against an unrelated file or name the wrong path in the error.
    """
    from tengri.components.nebular.cloudy_cb19 import _DEFAULT_PATH

    missing = tmp_path / "does_not_exist_cb19.h5"
    assert str(_DEFAULT_PATH) != str(missing)

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))) as excinfo:
        _build_via_key(ssp, observation, missing)

    message = str(excinfo.value)
    assert str(_DEFAULT_PATH) not in message, (
        f"the refusal must name the given path, not the default: {message}"
    )


def test_grid_key_survives_to_groups_roundtrip(tmp_path):
    """(e) ``spec.to_groups()`` must not silently drop the cb19 ``grid`` key.

    ``_STRUCTURAL_ROUNDTRIP["neb"]`` held only the ``cloudy`` entry for
    ``grid`` (``only_types=("cloudy",)``): a cb19 spec's ``to_groups()``
    dropped the key entirely and ``parse_groups(**spec.to_groups())`` silently
    reverted to the packaged default grid with no error -- the same
    silent-drop disease #2220 fixed one layer over, in the direction back out
    of the spec rather than into it. This test builds no SED model (no SSP,
    no observation, no HDF5 file needs to exist): ``parse_groups`` only
    produces a ``Parameters``, and the round-trip is exercised purely through
    the grammar/spec layer.
    """
    from tengri.parameters.groups import parse_groups

    grid_path = str(tmp_path / "cb19_templates.h5")
    spec = parse_groups(
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "cb19", "grid": grid_path, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    assert spec.nebular_cb19_grid_path == grid_path

    groups = spec.to_groups()
    assert groups["neb"].get("grid") == grid_path, (
        f"to_groups() dropped the cb19 grid key: {groups['neb']}"
    )

    reparsed = parse_groups(**groups)
    assert reparsed.nebular_cb19_grid_path == grid_path, (
        f"re-parsing to_groups() output lost the grid path: "
        f"got {reparsed.nebular_cb19_grid_path!r}, expected {grid_path!r}"
    )


def test_grid_key_refused_for_types_that_never_read_it():
    """(f) ``grid`` used to be legal for *every* ``neb`` type (M2 of the #2220
    review): ``neb={'type': 'cue', 'grid': p}`` and
    ``neb={'type': 'ssp', 'grid': p}`` were accepted and the path silently
    dropped, never reaching any backend -- the identical disease the cb19
    branch had, just for types that never read the key at all. ``grid`` is
    now legal only for ``cloudy``, ``cb19``, ``mappings`` and
    ``mappings_agn``; every other type must refuse it by name, naming both
    the key and the type.
    """
    from tengri.parameters.groups import parse_groups

    for neb_type in ("cue", "ssp"):
        with pytest.raises(ValueError) as excinfo:
            parse_groups(
                neb={"type": neb_type, "grid": "/tmp/unused_cb19_grid.h5"},
                redshift=Fixed(0.1),
            )
        message = str(excinfo.value)
        assert "grid" in message, f"type={neb_type!r}: {message}"
        assert neb_type in message, f"type={neb_type!r}: {message}"


def test_unknown_key_message_is_type_aware_for_grid():
    """(g) The unknown-key message's displayed/suggested keys track the type.

    Making ``grid`` type-specific in (f) means it is no longer in the base
    ``_GROUP_STRUCTURAL_KEYS["neb"]`` set that ``_check_dict_keys`` used to
    read its "Valid structural keys for this group are" list from. Left
    unfixed, that list is computed from the base set alone regardless of
    type: a cb19 typo (``'gird'``) would suggest ``grid`` via the did-you-mean
    hint (which reads the type-aware ``allowed`` union, already correct) and
    then immediately contradict itself by listing the valid keys as NOT
    including ``grid`` -- the exact self-contradiction this test refuses to
    let back in. The fix must not simply restore ``grid`` to the base set
    (that would tell a ``cue`` user the key is valid and then refuse it, the
    same disease from the other direction); the displayed list must instead
    union in the resolved type's own specific keys, so cb19's list includes
    ``grid`` and cue's does not.
    """
    from tengri.parameters.groups import parse_groups

    with pytest.raises(ValueError) as excinfo:
        parse_groups(neb={"type": "cb19", "gird": "/tmp/unused.h5"}, redshift=Fixed(0.1))
    cb19_message = str(excinfo.value)
    assert "Did you mean: grid?" in cb19_message, cb19_message
    assert "'grid'" in cb19_message, (
        f"cb19's displayed valid-keys list must include 'grid': {cb19_message}"
    )

    with pytest.raises(ValueError) as excinfo:
        parse_groups(neb={"type": "cue", "gird": "/tmp/unused.h5"}, redshift=Fixed(0.1))
    cue_message = str(excinfo.value)
    assert "grid" not in cue_message, (
        f"cue does not read 'grid'; it must be neither suggested nor listed: {cue_message}"
    )
    assert "'all_params'" in cue_message and "'full_catalog'" in cue_message, (
        f"cue's own valid structural keys must still be named: {cue_message}"
    )
