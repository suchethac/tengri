# SPDX-License-Identifier: BSD-3-Clause
"""The ``Galaxy`` facade, ``doctor()``, and name-based preset resolution.

What changed and why
--------------------

Seven tests built the same four-line un-fitted ``Galaxy`` before asserting one
thing about it; that setup is a fixture now. Three ``resolve_preset`` tests
asserted ``params is not None`` and ``config is not None``, which is true of
almost any return value -- they are one table that asserts *which* model each
preset selected, since a preset that silently resolved to the wrong SFH family
passed all three.

Two vacuity holes are closed rather than restated:

* The citation tests each asserted ``"nifty" in keys`` or ``"blackjax" in
  keys``. A ``_infer_citation_keys`` that returned every key it knows would
  have passed all three. Measured base set (no backend) is ``{calzetti2000,
  charlot_fall2000, cue, dsps, jax, tengri}``, so each backend case now also
  asserts the *other* backend's key is absent.
* ``test_galaxy_fit_requires_flux_data`` did not test that. It asserted
  ``pytest.raises(AttributeError)`` and the error raised is ``'MockSSP' object
  has no attribute 'ssp_lg_age_gyr'`` -- the stand-in SSP, not the missing
  flux. It would pass with flux data present. Renamed to what it pins, and the
  message is matched so it cannot drift back into meaning something else.

Removed
-------

``test_galaxy_class_exists`` and ``test_presets_are_importable`` asserted
``X is not None`` on an import; six tests below fail outright if either import
breaks.

``test_galaxy_from_arrays_smoke`` carried an unconditional
``@pytest.mark.skip``, so it had never executed, and its body was ``assert g is
not None``. Its conditional inner ``pytest.skip`` on ``TENGRI_SSP_PATH`` was
unreachable behind the marker. ``Galaxy.from_arrays`` has no other happy-path
test in the tree; adding one means loading a real SSP grid, which is new
coverage rather than consolidation and belongs in its own change.

The module-level ``skipif`` on ``hasattr(_presets, "resolve_preset")`` is gone.
Its reason read "not yet implemented (only synthesizer_default is registered
today)"; ``resolve_preset`` is a public part of ``tengri.presets``, so the
guard could not fire and would have hidden its removal if it ever did.
"""

from __future__ import annotations

import platform

import jax
import pytest

import tengri
from tengri import Galaxy, Observation, Photometry, doctor
from tengri.presets import resolve_preset

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# doctor()
# ---------------------------------------------------------------------------

#: (label printed by doctor, the live value it must be reporting).
_DOCTOR_FACTS = [
    ("Python", platform.python_version()),
    ("tengri", tengri.__version__),
    ("JAX", jax.__version__),
    ("JAX backend", jax.default_backend()),
]


@pytest.mark.parametrize(("label", "value"), _DOCTOR_FACTS, ids=[c[0] for c in _DOCTOR_FACTS])
def test_doctor_reports_the_live_environment(label, value):
    """Each fact is read from the running interpreter, not printed as a banner.

    The three tests this replaces asserted that the word "jax" and the word
    "Python" appeared somewhere in the output -- satisfied by a hardcoded
    header, and by the ASCII-art logo in the case-insensitive comparison. What
    matters is that the reported version is the one actually imported.
    """
    out = doctor()
    assert label in out, out
    assert value in out, f"doctor() does not report the live {label} ({value}):\n{out}"


# ---------------------------------------------------------------------------
# Constructors that need SSP data
# ---------------------------------------------------------------------------


def test_from_arrays_without_an_ssp_raises():
    """``Galaxy.from_arrays`` cannot build a model without SSP data."""
    with pytest.raises((ValueError, TypeError, FileNotFoundError, RuntimeError)):
        Galaxy.from_arrays(
            filters=["sdss_u", "sdss_g"],
            flux=[1e-28, 2e-28],
            flux_err=[1e-29, 1e-29],
            redshift=0.1,
        )


def test_from_observation_without_an_ssp_raises():
    """``Galaxy.from_observation`` likewise."""
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i"]))
    with pytest.raises((ValueError, TypeError, FileNotFoundError)):
        Galaxy.from_observation(obs)


# ---------------------------------------------------------------------------
# Name-based preset resolution
# ---------------------------------------------------------------------------

#: (name, redshift, the SFH family it must select, the birth-cloud law).
#: Each preset's identity is its SFH family; the three differ, which is the
#: property `params is not None` could not see.
_PRESET_CASES = [
    ("starforming", 0.1, "dpl", "calzetti"),
    ("quiescent", 0.5, "dexp", "power_law"),
    ("high_z", 4.0, "tsnorm", "calzetti"),
]


@pytest.mark.parametrize(
    ("name", "redshift", "sfh_type", "law_bc"), _PRESET_CASES, ids=[c[0] for c in _PRESET_CASES]
)
def test_resolve_preset_selects_the_right_model(name, redshift, sfh_type, law_bc):
    """The resolved config names this preset's SFH family and dust law.

    And the redshift is fixed to the value passed, not merely fixed to
    something -- a preset that ignored the argument and pinned ``Fixed(0.0)``
    satisfies "is_fixed".
    """
    params, config = resolve_preset(name, redshift=redshift)

    assert config.sfh.mean_type == (sfh_type,)
    assert config.dust.law_bc == law_bc

    z = params._distributions.get("redshift")
    assert z is not None and z.is_fixed, f"{name} left redshift free"
    assert z.value == pytest.approx(redshift)

    # The free set is the preset's own; nothing here is a placeholder.
    assert any(p.startswith(f"sfh_{sfh_type}_") for p in params.free_params), sorted(
        params.free_params
    )


def test_the_presets_are_not_the_same_model():
    """Non-vacuity for the table above: no two presets resolve alike.

    Three separate ``is not None`` tests would have passed unchanged if every
    name resolved to one shared default.
    """
    resolved = {name: resolve_preset(name, redshift=z)[1] for name, z, _s, _law in _PRESET_CASES}
    families = {name: cfg.sfh.mean_type for name, cfg in resolved.items()}
    assert len(set(families.values())) == len(families), families


def test_resolve_preset_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        resolve_preset("invalid_preset")


# ---------------------------------------------------------------------------
# A Galaxy that has been constructed but never fitted
# ---------------------------------------------------------------------------


class _MockSSP:
    """A stand-in with none of the SSP grid attributes the model reads."""


@pytest.fixture
def unfitted():
    """A constructed, never-fitted ``Galaxy``.

    Function-scoped deliberately: two of the tests below assign to
    ``_last_backend``, and a shared instance would carry that into whichever
    test ran next.
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
    params, config = resolve_preset("starforming", redshift=0.1)
    return Galaxy(ssp=_MockSSP(), observation=obs, parameters=params, model_config=config)


_UNFITTED_CALLS = [
    ("summary", lambda g, _tmp: g.summary(), "No fit result available"),
    ("plot", lambda g, _tmp: g.plot(), "No fit result available"),
    ("save", lambda g, tmp: g.save(str(tmp / "unused.h5")), "has not been fitted"),
]


@pytest.mark.parametrize(
    ("call", "match"),
    [(c[1], c[2]) for c in _UNFITTED_CALLS],
    ids=[c[0] for c in _UNFITTED_CALLS],
)
def test_result_accessors_refuse_before_a_fit(unfitted, tmp_path, call, match):
    """Each accessor says a fit is missing, rather than returning empty.

    The message is matched: three tests asserting a bare ``RuntimeError`` would
    equally accept one raised for an unrelated reason, which is exactly the
    defect ``test_fit_fails_on_a_stand_in_ssp`` below was carrying.
    """
    pytest.importorskip("h5py")
    with pytest.raises(RuntimeError, match=match):
        call(unfitted, tmp_path)


def test_fit_fails_on_a_stand_in_ssp(unfitted):
    """A Galaxy built with a stand-in SSP fails at the first grid attribute.

    Formerly ``test_galaxy_fit_requires_flux_data``, which is not what it
    asserted: the ``AttributeError`` it caught comes from ``_MockSSP``, not
    from absent flux, and it would pass with flux data supplied. Matching the
    message keeps the two apart.
    """
    with pytest.raises(AttributeError, match="ssp_lg_age_gyr"):
        unfitted.fit(method="map", verbose=False)


# ---------------------------------------------------------------------------
# Citation inference
# ---------------------------------------------------------------------------

#: Keys the starforming preset yields with no backend recorded. Pinned so the
#: backend cases below can assert what each backend *adds*.
_BASE_CITATIONS = {"tengri", "dsps", "jax", "calzetti2000", "charlot_fall2000", "cue"}

#: (backend recorded on the Galaxy, keys it must add).
_BACKEND_CITATIONS = [
    ("vi", {"nifty"}),
    ("mcmc_nuts", {"blackjax"}),
    ("map", set()),
]


def test_citation_keys_without_a_backend(unfitted):
    """The core and physics citations, exactly -- not a superset.

    Asserted as equality. Three separate ``"x" in keys`` tests are all
    satisfied by a function that returns every key it knows about, which is the
    failure mode worth excluding here.
    """
    assert set(unfitted._infer_citation_keys()) == _BASE_CITATIONS


@pytest.mark.parametrize(
    ("backend", "added"), _BACKEND_CITATIONS, ids=[c[0] for c in _BACKEND_CITATIONS]
)
def test_backend_adds_only_its_own_citation(unfitted, backend, added):
    """Recording a backend adds that backend's key and no other's."""
    unfitted._last_backend = backend
    keys = set(unfitted._infer_citation_keys())

    expected = _BASE_CITATIONS | added
    assert keys == expected, f"{backend}: unexpected {sorted(keys ^ expected)}"


# ---------------------------------------------------------------------------
# Result round-trip and the flux unit ladder
# ---------------------------------------------------------------------------


def test_load_result_roundtrip(tmp_path):
    """``Galaxy.load_result`` restores a ``FitResult`` from HDF5."""
    pytest.importorskip("h5py")
    from tengri.results import FitRecord, FitResult

    fr = FitResult(
        inner={"samples": {"x": [1.0, 2.0, 3.0]}},
        record=FitRecord.capture(),
        citation_keys=["jax", "dsps"],
        backend="map",
        preset="starforming",
    )
    path = tmp_path / "roundtrip.h5"
    fr.save(str(path))

    loaded = Galaxy.load_result(str(path))

    assert loaded.backend == "map"
    assert loaded.preset == "starforming"
    assert set(loaded.citation_keys) == {"jax", "dsps"}


def test_flux_unit_table_internally_consistent():
    """Regression: the Jansky ladder in _FLUX_UNIT_TO_CGS must be decades apart.

    Found 2026-07-08 while fact-checking the docs: ``uJy`` carried the Jy
    factor (1e-23, a 1e6 silent flux error) and ``maggies`` carried
    3631x the mJy factor (a 1e3 error). 1 Jy = 1e-23 erg/s/cm2/Hz and
    1 maggie = 3631 Jy define the whole table.
    """
    from tengri.facade import _FLUX_UNIT_TO_CGS as units

    jy = units["Jy"]
    assert jy == 1e-23
    assert units["mJy"] == pytest.approx(jy * 1e-3)
    assert units["uJy"] == pytest.approx(jy * 1e-6)
    assert units["nJy"] == pytest.approx(jy * 1e-9)
    assert units["maggies"] == pytest.approx(3631 * jy)
