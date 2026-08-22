# SPDX-License-Identifier: BSD-3-Clause
"""First-contact error-quality contracts.

Each case here reproduces something a new user actually did in QoL testing
and pins the friendly failure mode that replaced a crash or a misleading
message:

- ``predict_*({})`` with free params → ``MissingParameterError`` naming the
  params (was: bare ``KeyError: 'dust_tau_bc'`` from inside the dust
  component).
- Typo'd filter name → did-you-mean + the real public API name (was: a
  pointer to non-exported ``list_available_filters()``).
- Non-string SFH ``type`` → ``TypeError`` with an example (was:
  ``unhashable type: 'dict'``).
- ``'all_params': FIXED`` on the delayed SFH → silent (was: three internal-sounding
  midpoint warnings; registry-wide sweep tracked in #1007).
- Missing SSP file → points at ``tengri.download_ssp()`` (was: raw h5py
  OSError).
- wNE SSP detection is two-sided: real wNE grids fail HIGH (log Q_H ≈ 62,
  nebular continuum corrupting the ionizing fit), not just low.
"""

import os
import warnings

import pytest

from tengri import FIXED, Fixed, Photometry, SEDModel, load_ssp_data
from tengri.config.exceptions import MissingParameterError

pytestmark = pytest.mark.contract

_WEIGHTS = os.path.join("data", "cue_weights.npz")


@pytest.fixture(scope="module")
def dustless_model(synthetic_ssp_wide, synthetic_tophat_obs):
    """Minimal build with explicit dust to test missing-params error handling.

    The dust group must be explicitly enabled with free params since PR-B changed
    the default: omitted dust_attenuation now means dust_model='off' (parity with
    other optional groups). This fixture needs dust params to be free to test the
    MissingParameterError contract.
    """
    from tengri import FREE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "delayed", "all_params": FIXED},
            dust_attenuation={"type": "two_component", "law": "power_law", "all_params": FREE},
            redshift=Fixed(0.1),
        )


def test_missing_free_params_raises_helpfully(dustless_model):
    with pytest.raises(MissingParameterError, match="dust_tau_bc"):
        dustless_model.predict_photometry({})


def test_missing_free_params_message_names_the_fix(dustless_model):
    with pytest.raises(MissingParameterError, match=r"spec\.sample"):
        dustless_model.mock({})


def test_full_params_still_predict(dustless_model):
    import jax
    import numpy as np

    params = dustless_model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(dustless_model.predict_photometry(params))
    assert np.isfinite(flux).all()


def test_predict_validates_eagerly_like_predict_photometry(dustless_model):
    """``model.predict`` must reject bad params up front, not defer to accessor.

    The lean ``predict_photometry({})`` raised a helpful ``MissingParameterError``,
    but the rich ``predict({})`` returned a lazy ``Prediction`` that only crashed
    with a bare ``KeyError`` when the first quantity was accessed — and a typo'd
    key was silently dropped the same way. Both paths now validate at the call
    site with the same checks.
    """
    import jax

    from tengri.config.exceptions import UnknownParameterError

    good = dustless_model.spec.sample(jax.random.PRNGKey(0))

    # A valid full params dict still returns a (lazy) Prediction.
    assert dustless_model.predict(good) is not None

    # Missing free params → eager MissingParameterError (not a deferred KeyError).
    with pytest.raises(MissingParameterError, match="dust_tau_bc"):
        dustless_model.predict({})

    # A typo'd key is caught (was silently dropped, then KeyError on accessor).
    typo = dict(good)
    a_key = next(iter(dustless_model.spec.free_params))
    typo[f"{a_key}_typo"] = typo.pop(a_key)
    with pytest.raises(UnknownParameterError):
        dustless_model.predict(typo)

    # A non-dict (e.g. a bare array) names the expected type instead of an
    # opaque "cannot convert dictionary update sequence" error.
    import numpy as np

    with pytest.raises(TypeError, match="expects a params dict"):
        dustless_model.predict(np.array([0.1, 0.2, 0.3]))


def test_no_observation_model_accessors_name_the_fix(synthetic_ssp_wide):
    """A rest-frame-only model must fail *helpfully* on observable accessors.

    Building without ``observation=`` is a valid mode — ``predict``,
    ``rest_sed()`` and ``stellar_mass`` all work. But ``pred.photometry()`` /
    ``.magnitudes()`` / ``.spectrum()`` used to crash with a bare
    ``'NoneType' object has no attribute 'photometry'`` because the accessor
    dereferenced ``model.observation`` (None) directly — while the lean
    ``model.predict_photometry`` already raised a helpful ValueError. The rich
    accessors now match.
    """
    import jax

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "all_params": FIXED},
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )
    assert model.observation is None
    pred = model.predict(model.spec.sample(jax.random.PRNGKey(0)))

    # Rest-frame quantities still work without an observation.
    assert pred.rest_sed().shape[0] > 0

    with pytest.raises(ValueError, match="No observation is configured"):
        pred.photometry()
    with pytest.raises(ValueError, match="No observation is configured"):
        pred.magnitudes()
    with pytest.raises(ValueError, match="No spectroscopy is configured"):
        pred.spectrum()


def test_unknown_filter_did_you_mean():
    with pytest.raises(KeyError, match="sdss_u"):
        Photometry.from_names(["sdss_q"])
    with pytest.raises(KeyError, match="list_filters"):
        Photometry.from_names(["sdss_q"])


def test_sfh_type_must_be_string():
    from tengri.parameters.groups import parse_groups

    with pytest.raises(TypeError, match="must be a string"):
        parse_groups(sfh={"type": {"oops": 1}}, redshift=Fixed(0.1))


def test_delayed_wildcard_is_warning_free(synthetic_ssp_wide, synthetic_tophat_obs):
    from tengri.parameters.groups import parse_groups

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_groups(sfh={"type": "delayed", "all_params": FIXED}, redshift=Fixed(0.1))
    midpoint_warnings = [w for w in caught if "no curated default" in str(w.message)]
    assert not midpoint_warnings, [str(w.message) for w in midpoint_warnings]


def test_missing_ssp_file_points_at_download():
    with pytest.raises(FileNotFoundError, match="download_ssp"):
        load_ssp_data("data/definitely_not_here.h5")


@pytest.mark.skipif(not os.path.exists(_WEIGHTS), reason="cue weights absent (data-gated)")
@pytest.mark.parametrize(
    ("fake_logqion", "match"),
    [(62.0, "far above"), (30.0, "well below")],
)
def test_wne_guard_is_two_sided(synthetic_ssp_wide, monkeypatch, fake_logqion, match):
    """Real wNE grids fail HIGH (observed log Q_H ≈ 62) — both sides must raise."""
    import numpy as np

    import tengri.components.nebular.ionizing_spectrum as ionspec
    from tengri.components.nebular.cue import CueBackend, CueWNESSPError

    orig = ionspec.precompute_ionizing_params_table

    def rigged(*args, **kwargs):
        result = orig(*args, **kwargs)
        result["logqion_table"] = np.full_like(np.asarray(result["logqion_table"]), fake_logqion)
        return result

    monkeypatch.setattr(ionspec, "precompute_ionizing_params_table", rigged)
    monkeypatch.delenv("TENGRI_ALLOW_WNE_CUE", raising=False)
    with pytest.raises(CueWNESSPError, match=match):
        CueBackend(_WEIGHTS, ssp_data=synthetic_ssp_wide)


# ── Filter discovery round-trip (list_filters() names must actually load) ──


def test_list_filters_names_round_trip_through_from_names():
    """Every name ``tengri.list_filters()`` advertises must load via from_names.

    Regression for the two-registry mismatch (fresh-user audit 2026-07):
    ``list_filters()`` displayed SVO-style names (e.g. ``'2MASS_2MASS_H'``)
    that ``Photometry.from_names`` rejected with ``KeyError`` — even though
    each row's own ``use`` hint told the user to call from_names with that
    exact name. The menu and the loader read different registries.
    """
    import tengri

    names = [row["name"] for row in tengri.list_filters()]
    assert names, "list_filters() returned nothing — is data/filters present?"
    # Spot-check a spread across the registry (loading all 249 is slow, though
    # offline); a single KeyError here is the regression.
    for name in names[::20]:
        Photometry.from_names([name])  # must not raise


def test_svo_display_name_and_short_alias_load_the_same_curve():
    """An SVO display name (not itself a registry key) resolves like its alias."""
    import numpy as np

    from tengri.observation.filters import (
        FILTER_REGISTRY,
        _svo_name_to_key,
        load_filter,
    )

    # Pick a display name that is NOT already a short key, so we exercise the
    # SVO-alias branch of load_filter rather than the direct lookup.
    svo_name, short_key = next(
        (s, k) for s, k in _svo_name_to_key().items() if s not in FILTER_REGISTRY
    )
    a = load_filter(svo_name)
    b = load_filter(short_key)
    np.testing.assert_allclose(np.asarray(a.wave), np.asarray(b.wave))
    np.testing.assert_allclose(np.asarray(a.trans), np.asarray(b.trans))


# --------------------------------------------------------------------------
# Fresh-user audit (2026-07): three discovery/loader mismatches — each is a
# case where a menu/constructor the discovery API advertises fed the user
# straight into a TypeError/ValueError.
# --------------------------------------------------------------------------


def test_observation_accepts_bare_photometry(synthetic_ssp_wide, synthetic_tophat_obs):
    """A bare ``Photometry`` passed as ``observation=`` is auto-wrapped.

    ``list_filters()`` advertises ``Photometry.from_names([...])``, which
    returns a ``Photometry`` — not an ``Observation``. Passing it straight to
    ``SEDModel.build(observation=...)`` used to raise ``TypeError`` with no
    fix. It now wraps like the ``filters=`` path already does.
    """
    photometry = synthetic_tophat_obs.photometry  # a bare Photometry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=photometry,
            sfh={"type": "delayed", "all_params": FIXED},
            redshift=Fixed(0.1),
        )
    assert model.observation.photometry is photometry


def test_observation_wrong_type_names_the_fix(synthetic_ssp_wide):
    """A genuinely wrong ``observation=`` type errors with the explicit wrap."""
    with pytest.raises(TypeError, match=r"Observation\(photometry="):
        SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=42,
            sfh={"type": "delayed", "all_params": FIXED},
            redshift=Fixed(0.1),
        )


def test_list_sfh_models_does_not_advertise_unbuildable_as_production():
    """``list_sfh_models()`` must not call a builder-rejected SFH 'production'.

    Types in ``UNVALIDATED_SFH_TYPES`` raise ``ValueError`` at
    ``SEDModel.build(sfh={'type': ...})``; advertising them as production
    sent a fresh user into a build-time crash. They now report
    ``status='unvalidated'`` and drop out of the ``status='production'``
    filter.
    """
    import tengri
    from tengri.components.stellar.sfh.registry import UNVALIDATED_SFH_TYPES

    rows = {r["name"]: r for r in tengri.list_sfh_models()}
    for name in UNVALIDATED_SFH_TYPES:
        assert rows[name]["status"] == "unvalidated", name

    production = {r["name"] for r in tengri.list_sfh_models(status="production")}
    assert not (production & UNVALIDATED_SFH_TYPES)
    # A validated staple stays production.
    assert rows["dpl"]["status"] == "production"


def test_list_agn_blocks_use_strings_name_valid_grammar_keys(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """Every AGN-block ``use:`` string must name a real ``agn`` group key.

    The advertised strings used to be ``agn_<cat>_block='<name>'`` — a kwarg
    ``SEDModel.build`` does not accept (``TypeError`` for all 42 blocks). They
    now use the nested grammar ``agn={'<key>': {'type': '<name>'}}``, and the
    'attenuation' category maps to its terser grammar key ``'atten'``.
    """
    import re

    import tengri
    from tengri.parameters.groups import _AGN_SUBBLOCK_KEYS

    rows = tengri.list_agn_blocks()
    for r in rows:
        m = re.search(r"agn=\{'(\w+)':", r["use"])
        assert m, f"unparseable use string: {r['use']!r}"
        key = m.group(1)
        assert key in _AGN_SUBBLOCK_KEYS, f"{r['category']}/{r['name']}: bad key {key!r}"
        if r["category"] == "attenuation":
            assert key == "atten", r["use"]

    # The exact regression: the corrected attenuation key is actually accepted
    # by the builder (the old 'attenuation' key raised "Unknown key").
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "delayed", "all_params": FIXED},
            neb={"type": "none"},
            agn={"disc": {"type": "powerlaw"}, "atten": {"type": "qsogen"}},
            redshift=Fixed(0.1),
        )
    assert model is not None


def test_model_menu_use_strings_teach_sedmodel_build():
    """Every model-menu ``use:`` hint teaches the recommended build path.

    The discovery menus used to split idioms: ``sfh``/``dust``/``nebular``/
    ``agn_models`` advertised the flat ``Parameters(..., mean_sfh_type=...)``
    expert escape hatch, while ``radio``/``xray``/``igm`` advertised
    ``SEDModel.build(...)``. A fresh user was pointed at two different APIs
    (and, per the naming contract, the non-recommended one). All model menus
    now teach ``SEDModel.build(...)`` uniformly. (Filters and inference keep
    their own idioms — ``Photometry.from_names`` / ``fitter.run``.)
    """
    import tengri

    menus = [
        tengri.list_sfh_models,
        tengri.list_dust_laws,
        tengri.list_dust_emission_models,
        tengri.list_nebular_backends,
        tengri.list_agn_models,
        tengri.list_agn_blocks,
        tengri.list_radio_models,
        tengri.list_xray_models,
        tengri.list_igm_models,
    ]
    # Status-aware, deliberately. The unconditional form of this check asserted
    # that EVERY row starts with ``SEDModel.build(`` — a test of the hint's
    # SHAPE, not of whether it runs. The eight ``UNVALIDATED_SFH_TYPES`` passed
    # it while ``SEDModel.build(..., sfh={'type': 'top_hat'})`` raised
    # ValueError, which is why the "advice that raises" class (#1275) read as
    # holding across 153 hints. A row the builder rejects must not advertise a
    # call at all; see ``test_unbuildable_rows_do_not_advertise_a_build_call``.
    offenders = [
        (fn.__name__, row["name"], row["use"])
        for fn in menus
        for row in fn()
        if row.get("status") != "unvalidated" and not row["use"].startswith("SEDModel.build(")
    ]
    assert not offenders, offenders


def test_unbuildable_rows_do_not_advertise_a_build_call():
    """A row the builder rejects must not hand the user a call that raises.

    The complement of the check above: ``status='unvalidated'`` rows are kept
    in the menu on purpose — delisting them recreates the invisibility of #1120
    — but their ``use:`` field must carry the reason and the next step rather
    than a copy-pasteable ``SEDModel.build(...)`` that fails.
    """
    import tengri

    offenders = [
        (row["name"], row["use"])
        for fn in (tengri.list_sfh_models,)
        for row in fn()
        if row.get("status") == "unvalidated" and row["use"].startswith("SEDModel.build(")
    ]
    assert not offenders, (
        f"unbuildable rows advertise a build call that raises: {offenders}. "
        "Put the reason in `use:`, not a call the builder refuses."
    )


def test_every_advertised_sfh_use_hint_actually_builds(synthetic_ssp_wide, synthetic_tophat_obs):
    """Execute the hints rather than pattern-matching them.

    The shape check above cannot tell a working call from a failing one, which
    is the gap that let the unvalidated eight through. This runs every ``use:``
    string the SFH menu advertises. Scoped to SFH because these models build on
    the synthetic grid — component menus whose libraries are data-gated cannot
    be executed in CI without turning a contract test into a data-gated one
    that silently skips.
    """
    import warnings

    import tengri
    from tengri import FIXED, Fixed, SEDModel

    failures = []
    for row in tengri.list_sfh_models(status="production"):
        assert row["use"].startswith("SEDModel.build("), row["use"]
        # Mixture/modulator rows advertise the composed ``['const', name]``
        # form; parse whichever shape the row carries rather than assuming one.
        spec = ["const", row["name"]] if "['const', " in row["use"] else row["name"]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                SEDModel.build(
                    ssp_data=synthetic_ssp_wide,
                    observation=synthetic_tophat_obs,
                    sfh={"type": spec, "all_params": FIXED},
                    redshift=Fixed(0.1),
                )
        except Exception as exc:
            failures.append((row["name"], row["use"], f"{type(exc).__name__}: {exc}"))

    assert not failures, "menu rows advertise `use:` hints that do not build:\n" + "\n".join(
        f"  {n}: {u}\n    -> {e}" for n, u, e in failures
    )


def test_list_components_use_hints_reference_real_functions():
    """``list_components()`` may only point at ``list_*`` functions that exist.

    The component ``use:`` hint was built by formula —
    ``f"list_{name}_models / list_{name}_laws"`` — but the real menus are
    irregular (``list_sfh_models``, ``list_dust_laws``,
    ``list_nebular_backends``, ...). So every component row advertised
    ``list_stellar_models`` / ``list_dust_models`` / ``list_agn_laws`` and
    other functions that raise ``AttributeError``. Now a lookup, not a formula.
    """
    import re

    import tengri

    referenced = set()
    for row in tengri.list_components():
        referenced.update(re.findall(r"list_[a-z_]+", row["use"]))
    missing = sorted(fn for fn in referenced if not hasattr(tengri, fn))
    assert not missing, f"list_components() points at non-existent functions: {missing}"
    # And it should still be pointing at *something* useful (regression against
    # dropping the hint entirely).
    assert referenced, "list_components() hints reference no discovery menus at all"


def test_sfh_mixture_modulator_use_strings_build(synthetic_ssp_wide, synthetic_tophat_obs):
    """Non-additive SFH ``use:`` hints must be a call the builder accepts.

    ``burst`` (composition_type ``mixture``) and ``field`` (``modulator``)
    cannot stand alone — ``sfh={'type': 'burst'}`` raises "At least one
    additive (smooth) SFH component required". Both are ``status='production'``,
    so their advertised ``use:`` string used to send a fresh user straight into
    that ``ValueError``. The hint now shows the composed list form, which builds.
    """
    import tengri
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    rows = {r["name"]: r for r in tengri.list_sfh_models()}
    non_additive = [
        n
        for n, e in SFH_REGISTRY.items()
        if getattr(e, "composition_type", "additive") in ("mixture", "modulator")
    ]
    assert non_additive, "expected at least burst/field to be non-additive"
    for name in non_additive:
        use = rows[name]["use"]
        # The advertised composed form must actually build.
        assert "['const', " in use, f"{name} use hint not composed: {use!r}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=synthetic_tophat_obs,
                sfh={"type": ["const", name], "all_params": FIXED},
                neb={"type": "none"},
                redshift=Fixed(0.1),
            )
        assert model is not None


def test_cloudy_nebular_use_string_names_the_grid_key():
    """The generic CLOUDY backend hint must name its required grid key.

    ``neb={'type': 'cloudy'}`` raises "The CLOUDY nebular backend needs a grid
    file"; the ``use:`` hint used to omit it. ``cb19`` ships its own grid and
    keeps the bare form.

    This asserted the literal substring ``"gridfile"`` — and the grammar's key
    is ``grid``, so it pinned a hint that raises ``Unknown key 'gridfile' in
    group 'neb'``. A hint written *because* the previous one failed failed
    differently, and a substring assertion cannot tell the two apart. Both
    halves are now checked by **parsing** the advertised dict.
    """
    import ast

    import tengri
    from tengri.parameters.groups import parse_groups

    rows = {r["name"]: r for r in tengri.list_nebular_backends()}
    cloudy = ast.literal_eval(rows["cloudy"]["use"].split("neb=", 1)[1].rstrip(")"))
    assert "grid" in cloudy, f"the cloudy hint names no grid key: {rows['cloudy']['use']}"
    parse_groups(neb=cloudy, redshift=Fixed(0.1))

    cb19 = ast.literal_eval(rows["cb19"]["use"].split("neb=", 1)[1].rstrip(")"))
    assert "grid" not in cb19, f"cb19 ships its own grid: {rows['cb19']['use']}"
    parse_groups(neb=cb19, redshift=Fixed(0.1))


def test_line_list_select_positional_list_names_the_fix():
    """``LineList.select(["Halpha", ...])`` must name the keyword fix.

    ``select``'s primary discovery use — keep specific lines by name — sits
    behind two positional wavelength bounds (``wave_min``, ``wave_max``). A
    fresh user reasoning by analogy with ``Photometry.from_names([...])``
    writes ``cat.select(["Halpha", "Hbeta"])``; the list bound to ``wave_min``
    used to crash 70 lines later with an opaque ``'>' not supported between
    instances of 'list' and 'float'``. It now raises ``TypeError`` pointing at
    the ``names=`` / ``species=`` keyword form.
    """
    from tengri import LineList

    cat = LineList.default_optical()
    with pytest.raises(TypeError, match=r"select\(names="):
        cat.select(["Halpha", "Hbeta"])
    # A string in wave_max is caught the same way.
    with pytest.raises(TypeError, match="wave_max must be a wavelength"):
        cat.select(0.0, "7000")

    # The correct calls are untouched — keyword names, and numeric windows
    # (Python int/float and numpy floats) all still filter.
    import numpy as np

    assert cat.select(names=["Halpha", "Hbeta"]).n_lines == 2
    assert cat.select(wave_min=3700.0, wave_max=7000.0).n_lines > 0
    assert cat.select(np.float32(3700), np.float32(7000)).n_lines > 0
    assert cat.select(species=["H1"]).n_lines > 0
