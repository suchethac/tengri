# SPDX-License-Identifier: BSD-3-Clause
r"""#2239: a headline line the selected catalog cannot carry must say so, not NaN silently.

https://github.com/suchethac/tengri/issues/2239

Cue's legacy 128-line CLOUDY/FSPS-matched subset (``full_catalog: False``)
carries no entry within ``_LINE_MATCH_TOL_AA`` of C IV 1549 (nearest is He II
1640, ~90 A away), so ``civ_1549`` returns NaN there while every other
headline accessor is unaffected. Before this fix nothing said so:
``warn_if_lines_are_unavailable`` only checked whether the active backend
publishes *a* line catalog at all, not whether the *requested* line is
covered by it, so this one NaN reached a likelihood or a catalog column in
silence.

Two independent changes land here, tested together:

1. The warning seam (``_warn_if_headline_line_uncovered`` in
   ``forward/properties.py``): fires when a headline line's ``KEY_LINES``
   wavelength has no catalog match within tolerance, naming the property,
   the backend, the nearest catalog line, and the remedy.
2. Cue's default catalog flips from the 128-line subset to the full ~138-line
   one, so ``civ_1549`` is finite (and silent) with no ``full_catalog`` key at
   all; ``full_catalog: False`` remains the explicit opt-out for cross-code
   comparisons, and is exactly what now triggers the new warning.

At HEAD (before this fix): (a) has no warning (the seam does not exist yet);
(b) and (c) fail because the default is ``False`` (128 rows, not 138); only
(d) already passes, since ``halpha`` was never affected by the subset choice.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel
from tengri.utils.batching import vmap_chunked
from tests._data_skip import CUE_WEIGHTS, DATA_DIR, requires_cue_weights

pytestmark = pytest.mark.regression_bug

_SSP_PATH = DATA_DIR / "fsps_prsc_miles_chabrier.h5"

#: Cue's trained species count and the legacy CLOUDY/FSPS-matched subset size
#: (r12 research, measured on the packaged ``data/cue_weights.npz``).
_N_FULL_CATALOG = 138
_N_LEGACY_SUBSET = 128


@pytest.fixture(scope="module")
def _cue_fixture_available():
    if not _SSP_PATH.is_file() or not CUE_WEIGHTS.is_file():
        pytest.skip(f"needs {_SSP_PATH} and {CUE_WEIGHTS}")


def _build(full_catalog: bool | None):
    """A cue model, with an explicit ``full_catalog`` or the bare default.

    Parameters
    ----------
    full_catalog : bool or None
        ``True``/``False`` sets the ``neb`` group's ``full_catalog`` key
        explicitly; ``None`` omits the key entirely, so the model resolves
        whatever the grammar's own default currently is.
    """
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    neb = {"type": "cue", "all_params": Fixed(DEFAULT)}
    if full_catalog is not None:
        neb["full_catalog"] = full_catalog
    return SEDModel.build(
        ssp_data=load_ssp_data(str(_SSP_PATH)),
        observation=Observation(photometry=Photometry.from_names(["sdss_g"])),
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb=neb,
        redshift=Fixed(0.1),
    )


@requires_cue_weights
def test_civ_1549_warns_and_is_nan_on_the_legacy_subset(_cue_fixture_available):
    """``full_catalog: False`` -- civ_1549 warns, names the remedy, and is NaN."""
    model = _build(full_catalog=False)
    params = dict(model.spec.get_fixed_values())
    with pytest.warns(UserWarning, match=r"'civ_1549'.*full_catalog"):
        value = float(model.predict(params).properties["civ_1549"])
    assert np.isnan(value), f"civ_1549 should be NaN on the legacy subset, got {value}"


@requires_cue_weights
def test_civ_1549_is_silent_and_finite_on_the_default_catalog(_cue_fixture_available):
    """No ``full_catalog`` key at all -- civ_1549 is finite and raises no warning."""
    model = _build(full_catalog=None)
    params = dict(model.spec.get_fixed_values())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", category=UserWarning)
        value = float(model.predict(params).properties["civ_1549"])
    assert not caught, f"the default catalog should carry civ_1549 silently: {caught}"
    assert np.isfinite(value) and value > 0.0, (
        f"civ_1549 should be finite and positive on the default (full) catalog, got {value}"
    )


@requires_cue_weights
def test_published_catalog_size_tracks_full_catalog(_cue_fixture_available):
    """138 rows by default, 128 under the explicit legacy-subset opt-out."""
    default_model = _build(full_catalog=None)
    subset_model = _build(full_catalog=False)
    params_default = dict(default_model.spec.get_fixed_values())
    params_subset = dict(subset_model.spec.get_fixed_values())

    # No warning filter here, deliberately: reading ``.lines.all_waves`` does
    # not route through the per-line warning seam at all (that seam fires
    # only from ``PropertyCatalog.__getitem__`` / ``predict_properties``, on
    # a named property, not on the raw catalog array), so this builds and
    # reads with zero warnings on both models -- measured.
    n_default = int(default_model.predict(params_default).lines.all_waves.size)
    n_subset = int(subset_model.predict(params_subset).lines.all_waves.size)

    assert n_default == _N_FULL_CATALOG, (
        f"default cue catalog carries {n_default} lines, expected {_N_FULL_CATALOG}"
    )
    assert n_subset == _N_LEGACY_SUBSET, (
        f"full_catalog=False cue catalog carries {n_subset} lines, expected {_N_LEGACY_SUBSET}"
    )


@requires_cue_weights
def test_halpha_does_not_warn_on_the_legacy_subset(_cue_fixture_available):
    """The seam must not fire for a line the 128-line subset already carries."""
    model = _build(full_catalog=False)
    params = dict(model.spec.get_fixed_values())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", category=UserWarning)
        value = float(model.predict(params).properties["halpha"])
    assert not caught, f"halpha should not warn on the legacy subset: {caught}"
    assert np.isfinite(value) and value > 0.0, f"halpha should be finite, got {value}"


# ── C1 regression: the seam must be jit/vmap-safe (review round) ──────────
#
# The first version of the seam read ``state.derived["line_waves"]`` inside
# ``jax.jit``/``jax.vmap``, guarded only by
# ``except jax.errors.ConcretizationTypeError``. Under jax 0.11.1 that never
# catches ``jax.errors.TracerArrayConversionError`` -- a *sibling* under
# ``JAXTypeError``, not a subclass -- so every line property raised under
# jit, including ``predict_properties`` (NAMING_CONTRACT's "the ONE jit/vmap
# surface for derived quantities") and ``Posterior.properties[...]`` (which
# uses ``vmap_chunked``). The fix reads the published catalog wavelengths
# from static (never-traced) backend state instead of ``state`` at all
# (``CueBackend.published_line_wavelengths`` / ``grid.line_wavelengths``),
# so the seam cannot see a tracer and needs no ``try``/``except``.


@requires_cue_weights
def test_predict_properties_is_jit_safe_for_line_properties(_cue_fixture_available):
    """``jax.jit(predict_properties(names=(...)))`` must not raise, either catalog.

    No blanket ignore: exactly one of the four (full_catalog, name) cases is
    expected to warn (civ_1549 on the legacy subset); the other three must
    stay silent, asserted explicitly rather than swallowed.
    """
    for full_catalog in (None, False):
        model = _build(full_catalog=full_catalog)
        params = dict(model.spec.get_fixed_values())
        for name in ("halpha", "civ_1549"):

            @jax.jit
            def _compute(p, _model=model, _name=name):
                return _model.predict_properties(p, names=(_name,))[_name]

            expect_warning = full_catalog is False and name == "civ_1549"
            if expect_warning:
                with pytest.warns(UserWarning, match=r"'civ_1549'.*full_catalog"):
                    value = float(_compute(params))
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    value = float(_compute(params))
                assert not caught, (
                    f"{name!r} under full_catalog={full_catalog} warned "
                    f"unexpectedly under jit: {caught}"
                )
            if name == "halpha":
                assert np.isfinite(value), (
                    f"halpha under full_catalog={full_catalog} should be finite "
                    f"under jit, got {value}"
                )
            # civ_1549 under the legacy subset is legitimately NaN -- the
            # point of this test is that the call above does not raise, not
            # that the value is finite in every case.


@requires_cue_weights
def test_predict_properties_is_vmap_safe_for_a_line_property(_cue_fixture_available):
    """``jax.vmap`` and the repository's ``vmap_chunked`` must not raise.

    No warning filter: measured, this body raises none (the default catalog
    carries ``halpha`` regardless of the subset choice), so a blanket ignore
    would silence nothing today and hide a future regression -- e.g. if
    ``vmap_chunked``'s own jittability probe ever falls back to its eager
    loop, which warns (``utils/batching.py``).
    """
    model = _build(full_catalog=None)
    params_batch = model.spec.sample_batch(jax.random.PRNGKey(0), n=4)

    def _single(p):
        return model.predict_properties(p, names=("halpha",))["halpha"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vmapped = jax.vmap(_single)(params_batch)
        chunked = vmap_chunked(_single, chunk_size=4)(params_batch)
    assert not caught, f"halpha under vmap/vmap_chunked warned unexpectedly: {caught}"

    assert vmapped.shape == (4,)
    assert np.all(np.isfinite(np.asarray(vmapped)))
    assert chunked.shape == (4,)
    assert np.all(np.isfinite(np.asarray(chunked)))


@requires_cue_weights
def test_civ_1549_warns_at_trace_time_under_jit_on_the_legacy_subset(_cue_fixture_available):
    """The warning still fires under ``jax.jit``, at trace (compile) time."""
    model = _build(full_catalog=False)
    params = dict(model.spec.get_fixed_values())

    @jax.jit
    def _compute(p):
        return model.predict_properties(p, names=("civ_1549",))["civ_1549"]

    with pytest.warns(UserWarning, match=r"'civ_1549'.*full_catalog"):
        value = float(_compute(params))
    assert np.isnan(value), f"civ_1549 should be NaN on the legacy subset, got {value}"


@requires_cue_weights
def test_published_line_wavelengths_matches_state_derived_line_waves(_cue_fixture_available):
    """I5: the seam's static catalog must equal the array the component publishes.

    ``NebularSEDComponent.apply`` applies tengri's vacuum-wavelength contract
    (``nebular_line_waves_to_vacuum``) to the backend's raw catalog before
    publishing ``state.derived["line_waves"]``; cue's raw catalog is air.
    ``_published_line_wavelengths_static`` must apply the exact same
    conversion, not compare against the pre-conversion array -- otherwise
    the seam's NaN decision (made against the vacuum array) and its own
    tolerance check (made against whatever this function returns) can
    disagree. RED before this fix: up to 2.70 Angstrom off, 70 of 138
    entries on the default catalog differing by more than 0.5 Angstrom.
    """
    from tengri.forward.properties import _published_line_wavelengths_static

    for full_catalog in (None, False):
        model = _build(full_catalog=full_catalog)
        params = dict(model.spec.get_fixed_values())
        state = model.predict_state(params)
        published = np.asarray(state.derived["line_waves"])
        static = _published_line_wavelengths_static(model, model._nebular_backend)
        np.testing.assert_allclose(
            static,
            published,
            rtol=1e-12,
            atol=0.0,
            err_msg=(
                f"_published_line_wavelengths_static (full_catalog={full_catalog}) "
                f"diverges from the published state.derived['line_waves']"
            ),
        )
