# SPDX-License-Identifier: BSD-3-Clause
"""#1677: metallicity histories reach the SED, and a bad one is refused at ingest.

``Catalog.from_histories`` already accepted ``met=`` and the forward model already
consumed it — the exact and ``WavePrecomp`` paths were both measured working. What
was missing was every guard around it, and each gap was measured on the pre-fix
code:

* ``logzsol = -6`` produced **byte-identical** photometry to the grid edge at
  ``-2.152``, because the metallicity lookup ``jnp.clip``s onto ``ssp_lgmet``.
  A simulation reaches primordial values at early times as a matter of course, so
  this is the ordinary case rather than the pathological one.
* A metal mass fraction — how a snapshot actually stores metallicity — was
  accepted as ``log10(Z/Zsun)``, which is a factor of ~70 for
  ``Z = 2e-4``, and *in-grid*, so no range check could ever see it.
* A ``NaN`` metallicity node left the flux bit-identical: silently dropped. A
  ``NaN`` SFR passed the ``sfr < 0`` guard, which every NaN passes.
* A model built ``met={'type': 'table'}`` and given no ``met=`` was
  accepted at construction and failed inside the first ``predict()``, which is
  the fail-late ``from_histories`` exists to prevent.

Flux equality is checked as a **ratio** throughout, never with bare
``np.allclose``: these fluxes are ~1e-11, far below ``allclose``'s default
``atol=1e-8``, which would call any two of them equal.

The ingest-level unit tests are in ``tests/unit/inference/test_history_ingest.py``;
the advice-message regression that #1678 fixed is in
``tests/regression/bug/test_catalog_met_table_advice.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from tengri.config.exceptions import (
    GasStellarMetallicityWarning,
    MetallicityUnitWarning,
    OutOfSSPGridWarning,
)
from tengri.parameters.translate import LOG10_ZSUN

pytestmark = [pytest.mark.regression_bug]

# The gas-phase arm needs a live nebular backend, and every CLOUDY grid is
# gitignored (data/.gitignore), so CI has none and these skip there. Cue is not
# an option: it refuses the wNE-shaped synthetic fixture by design.
#
# Resolved by walking parents for a ``data/`` holding the file, the same way
# ``load_ssp`` resolves grids — a git worktree is a fresh checkout without the
# ignored data, so a path fixed to the repo root would skip locally too and the
# arm would never be seen to run at all.
_CLOUDY_GRID_NAME = "cloudy_grid_prsc.h5"


def _find_data_file(name):
    """The nearest ``data/<name>`` walking up from this file, or None."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / name
        if candidate.is_file():
            return candidate
    return None


requires_cloudy = pytest.mark.skipif(
    _find_data_file(_CLOUDY_GRID_NAME) is None,
    reason=f"CLOUDY nebular grid {_CLOUDY_GRID_NAME} not found in any parent data/",
)

_Z_OBS = 0.05
_T_GYR = np.concatenate([np.array([0.0]), np.linspace(1.0, 13.0, 39)])
_N_T = _T_GYR.shape[0]
_N = 3

# ``synthetic_ssp_wide`` ships absolute log10(Z) = [-4.0, -2.65, -1.3]; with
# LOG10_ZSUN = -1.848 that is logzsol [-2.152, +0.548].
_GRID_LO = -4.0 - LOG10_ZSUN
_GRID_HI = -1.3 - LOG10_ZSUN


def _build(ssp, obs, *, met_mode=None, met_logzsol=None, approx=None, neb="none", gas_free=False):
    """A 5-band table-SFH ForwardModel, optionally with a tabulated metallicity."""
    from tengri import FIXED, ForwardModel, SEDModel
    from tengri.parameters.priors import Fixed, Uniform

    met_group = {}
    if met_mode is not None:
        met_group["type"] = met_mode
    if met_logzsol is not None:
        met_group["logzsol"] = Fixed(met_logzsol)

    neb_group = {"type": neb} if isinstance(neb, str) else dict(neb)
    if neb_group["type"] != "none":
        neb_group["all_params"] = FIXED
        if gas_free:
            neb_group["logZ_gas"] = Uniform(-1.5, 0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sed = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "table"},
            dust={
                "type": "two_component",
                "all_params": FIXED,
                "tau_bc": 0.5,
                "tau_diff": Uniform(0.0, 2.0),
            },
            neb=neb_group,
            redshift=Fixed(_Z_OBS),
            **({"met": met_group} if met_group else {}),
            **({"approx": approx} if approx is not None else {}),
        )
        return ForwardModel.build(sed=sed, observation=obs)


@pytest.fixture
def fwd_table_met(synthetic_ssp_wide, synthetic_tophat_obs):
    """Tabulated SFH **and** tabulated metallicity — both arrive at runtime."""
    return _build(synthetic_ssp_wide, synthetic_tophat_obs, met_mode="table")


@pytest.fixture
def histories():
    """(t, sfr, met, params) for three galaxies sharing one SFH shape."""
    shape = np.ones(_N_T)
    shape[0] = 0.0  # anchor at the Big Bang so nothing extrapolates past it
    t = np.broadcast_to(_T_GYR, (_N, _N_T)).copy()
    sfr = np.stack([shape * s for s in (1.0, 5.0, 20.0)])
    met = np.broadcast_to(np.linspace(-1.5, -0.1, _N_T), (_N, _N_T)).copy()
    return t, sfr, met, {"dust_tau_diff": np.full(_N, 0.2)}


# ── the silent clamp this all exists to stop ─────────────────────────


def test_the_clamp_really_is_invisible(fwd_table_met, histories):
    """Pin the defect itself, so the guard cannot be removed as over-cautious.

    Under ``on_out_of_grid='ignore'`` a wildly sub-primordial history and one
    sitting exactly on the grid edge return the **same photometry to machine
    precision**. Nothing about the resulting SED says a third of a galaxy's
    history was pinned to a template it never reached. That is the whole case
    for refusing it at ingest.
    """
    from tengri import Catalog

    t, sfr, _, params = histories

    def _predict(level):
        return np.asarray(
            Catalog.from_histories(
                fwd_table_met,
                t_gyr=t,
                sfr=sfr,
                met=np.full((_N, _N_T), level),
                params=params,
                on_out_of_grid="ignore",
            ).predict()
        )

    off_grid, at_edge = _predict(-6.0), _predict(_GRID_LO)
    assert np.allclose(off_grid / at_edge, 1.0, rtol=1e-12), (
        "the premise of this whole fix: logzsol=-6 must be indistinguishable "
        "from the grid edge, which is why it has to be refused at ingest"
    )


def test_out_of_grid_metallicity_is_refused_by_default(fwd_table_met, histories):
    from tengri import Catalog

    t, sfr, _, params = histories
    with pytest.raises(ValueError, match="fall outside the SSP metallicity grid") as exc:
        Catalog.from_histories(
            fwd_table_met, t_gyr=t, sfr=sfr, met=np.full((_N, _N_T), -6.0), params=params
        )
    message = str(exc.value)
    assert "of the stellar mass formed" in message, "the report must be mass-weighted"
    assert "met_unit=" in message, (
        "a mis-declared unit lands out of grid exactly like this, so the advice "
        "has to name it before the reader starts clipping good data"
    )


def test_warn_downgrades_and_still_predicts(fwd_table_met, histories):
    from tengri import Catalog

    t, sfr, _, params = histories
    with pytest.warns(OutOfSSPGridWarning):
        cat = Catalog.from_histories(
            fwd_table_met,
            t_gyr=t,
            sfr=sfr,
            met=np.full((_N, _N_T), -6.0),
            params=params,
            on_out_of_grid="warn",
        )
    assert np.isfinite(np.asarray(cat.predict())).all()


# ── units ────────────────────────────────────────────────────────────


def test_a_mass_fraction_history_predicts_the_same_sed(fwd_table_met, histories):
    """The conversion has to agree with the forward model's own solar offset.

    Ingest converts ``z_mass_fraction`` with ``LOG10_ZSUN``, and
    ``_tabulated_lgmet_on_ssp_ages`` adds the same constant back when it lifts
    ``met_history`` onto the SSP grid. If the two ever disagree, the SED is
    computed at a metallicity nobody asked for — and it stays finite, smooth and
    unremarkable, so only an equality test like this one would notice.
    """
    from tengri import Catalog

    t, sfr, met, params = histories
    z_mass = 10.0 ** (met + LOG10_ZSUN)

    as_logzsol = np.asarray(
        Catalog.from_histories(fwd_table_met, t_gyr=t, sfr=sfr, met=met, params=params).predict()
    )
    as_mass_fraction = np.asarray(
        Catalog.from_histories(
            fwd_table_met,
            t_gyr=t,
            sfr=sfr,
            met=z_mass,
            met_unit="z_mass_fraction",
            params=params,
        ).predict()
    )
    assert np.allclose(as_mass_fraction / as_logzsol, 1.0, rtol=1e-10)


def test_log_z_abs_history_predicts_the_same_sed(fwd_table_met, histories):
    from tengri import Catalog

    t, sfr, met, params = histories
    a = np.asarray(
        Catalog.from_histories(fwd_table_met, t_gyr=t, sfr=sfr, met=met, params=params).predict()
    )
    b = np.asarray(
        Catalog.from_histories(
            fwd_table_met,
            t_gyr=t,
            sfr=sfr,
            met=met + LOG10_ZSUN,
            met_unit="log_z_abs",
            params=params,
        ).predict()
    )
    assert np.allclose(b / a, 1.0, rtol=1e-10)


def test_an_undeclared_mass_fraction_warns(fwd_table_met, histories):
    """In-grid, finite, plausible — and wrong by ~70x. Only the flatness gives it away."""
    from tengri import Catalog

    t, sfr, met, params = histories
    z_mass = 10.0 ** (met + LOG10_ZSUN)
    assert np.all((z_mass > _GRID_LO) & (z_mass < _GRID_HI)), (
        "the premise: read as logzsol these are in-grid, so no range check sees them"
    )
    with pytest.warns(MetallicityUnitWarning):
        Catalog.from_histories(fwd_table_met, t_gyr=t, sfr=sfr, met=z_mass, params=params)


# ── fail fast, not deep in the forward pass ──────────────────────────


def test_a_tabulated_metallicity_model_demands_a_history(fwd_table_met, histories):
    """Pre-#1677 this was accepted here and blew up on the first predict().

    The point of ``from_histories`` is that everything wrong is wrong before
    anything compiles. Both halves are asserted: that construction refuses, and
    that the refusal names the two ways out.
    """
    from tengri import Catalog

    t, sfr, _, params = histories
    with pytest.raises(ValueError, match="no met= history was given") as exc:
        Catalog.from_histories(fwd_table_met, t_gyr=t, sfr=sfr, params=params)
    message = str(exc.value)
    assert "met_mode='table'" in message or "'met_mode': 'table'" in message
    assert "met_logzsol" in message, "name the parametric alternative, not just the fault"


def test_a_nan_metallicity_node_is_refused_end_to_end(fwd_table_met, histories):
    from tengri import Catalog

    t, sfr, met, params = histories
    met[1, 3] = np.nan
    with pytest.raises(ValueError, match=r"galaxy 1, node 3"):
        Catalog.from_histories(fwd_table_met, t_gyr=t, sfr=sfr, met=met, params=params)


# ── stellar Z and gas-phase Z are two knobs, not one ─────────────────
#
# ``met=`` is the metallicity the stars formed from; it picks the SSP templates.
# ``met_gas=`` is the metallicity of the ionized gas; it drives nebular
# emission. The four photoionized backends carry an
# ``if neb_logZ_gas is None: neb_logZ_gas = log_z`` inheritance, but the build
# grammar always supplies ``neb_logZ_gas``'s declared default, so that branch
# never runs: leaving it unset was measured bit-identical to passing -0.3, and
# the gas therefore never follows the stellar history.


def test_a_history_catalog_says_plainly_that_it_cannot_fit(fwd_table_met, histories):
    """The docstring claimed ``.fit()`` was "still meaningful" here. It raises.

    A history-built catalog carries no observed fluxes to fit *to*, and the
    data-table constructor has no channel to attach histories, so fitting at a
    known simulation SFH is unreachable through this class. Pinned because the
    claim was in the published API reference — ``docs/api/*.rst`` are autodoc
    stubs — and that is the #1276 failure class this issue is an instance of.
    """
    import jax

    from tengri import Catalog

    t, sfr, met, params = histories
    cat = Catalog.from_histories(fwd_table_met, t_gyr=t, sfr=sfr, met=met, params=params)
    with pytest.raises(ValueError, match="No table provided at construction"):
        cat.fit(key=jax.random.PRNGKey(0), method="map")


def test_gas_metallicity_without_a_nebular_backend_is_refused(fwd_table_met, histories):
    """Nothing consumes it, so accepting it would be accepting a no-op."""
    from tengri import Catalog

    t, sfr, met, params = histories
    with pytest.raises(ValueError, match="no active nebular backend"):
        Catalog.from_histories(
            fwd_table_met,
            t_gyr=t,
            sfr=sfr,
            met=met,
            met_gas=np.full(_N, -0.3),
            params=params,
        )


@requires_cloudy
def test_gas_and_stellar_metallicity_move_the_sed_independently(
    synthetic_ssp_wide, synthetic_tophat_obs, histories
):
    """Each must move the prediction with the other held fixed.

    One direction alone is not enough: a wiring that fed ``met_gas`` into the
    stellar templates would also make both "work".
    """
    from tengri import Catalog

    t, sfr, met, params = histories
    fwd = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        met_mode="table",
        neb="cloudy",
        gas_free=True,
    )

    def _predict(stellar, gas):
        return np.asarray(
            Catalog.from_histories(
                fwd,
                t_gyr=t,
                sfr=sfr,
                met=stellar,
                met_gas=np.full(_N, gas),
                params=params,
            ).predict()
        )

    gas_only = _predict(met, 0.25) / _predict(met, -1.2)
    stellar_only = _predict(np.full_like(met, 0.3), -0.3) / _predict(np.full_like(met, -2.0), -0.3)
    assert not np.allclose(gas_only, 1.0, rtol=1e-9), (
        f"gas-phase Z did not move the SED at fixed stellar Z: {gas_only[0]}"
    )
    assert not np.allclose(stellar_only, 1.0, rtol=1e-9), (
        f"stellar Z did not move the SED at fixed gas Z: {stellar_only[0]}"
    )


#: A stellar history ending well above the ``neb_logZ_gas`` default of -0.3.
#: The ``histories`` fixture ends at -0.1, only 0.2 dex away and so deliberately
#: inside the tolerance — the guard is about a *default left behind*, not about
#: any offset at all, and a fixture that tripped it at 0.2 dex would be pinning
#: the wrong threshold.
_ENRICHED_ENDPOINT = 0.4


@pytest.fixture
def enriched_history():
    """(N, n_t) stellar Z rising to well above the gas-phase default."""
    return np.broadcast_to(np.linspace(-1.5, _ENRICHED_ENDPOINT, _N_T), (_N, _N_T)).copy()


@requires_cloudy
def test_enriched_stars_in_default_gas_do_not_pass_quietly(
    synthetic_ssp_wide, synthetic_tophat_obs, histories, enriched_history
):
    """The measured trap: the stars enrich, the gas stays at 0.5 Zsun, silently.

    Decoupling the two is legitimate physics — pristine inflow does exactly
    that — so this warns rather than raising, and only when the gas value was
    left at its declaration.
    """
    from tengri import Catalog
    from tengri.config.exceptions import measurements_of

    t, sfr, _, params = histories
    fwd = _build(synthetic_ssp_wide, synthetic_tophat_obs, met_mode="table", neb="cloudy")

    with pytest.warns(GasStellarMetallicityWarning, match="never enriched") as rec:
        Catalog.from_histories(fwd, t_gyr=t, sfr=sfr, met=enriched_history, params=params)
    m = measurements_of(rec[0].message)
    assert m["gas_logzsol"] == pytest.approx(-0.3), "the declared default"
    assert m["stellar_present_day_max"] == pytest.approx(_ENRICHED_ENDPOINT)
    assert m["offset_dex"] == pytest.approx(_ENRICHED_ENDPOINT + 0.3)


@requires_cloudy
def test_a_modest_gas_stellar_offset_is_left_alone(
    synthetic_ssp_wide, synthetic_tophat_obs, histories
):
    """0.2 dex apart is ordinary, not a mistake — the guard must not cry wolf.

    The mass-metallicity relation scatters by more than this, so a threshold
    tight enough to fire here would fire on most real catalogs.
    """
    from tengri import Catalog

    t, sfr, met, params = histories  # this history ends at -0.1
    fwd = _build(synthetic_ssp_wide, synthetic_tophat_obs, met_mode="table", neb="cloudy")

    with warnings.catch_warnings():
        warnings.simplefilter("error", GasStellarMetallicityWarning)
        Catalog.from_histories(fwd, t_gyr=t, sfr=sfr, met=met, params=params)


@requires_cloudy
def test_the_gas_warning_is_silent_once_the_choice_is_made(
    synthetic_ssp_wide, synthetic_tophat_obs, histories, enriched_history
):
    """Three ways of deciding, none of which should nag.

    All three run on the *enriched* history, the one that does warn when nothing
    is decided — otherwise each case would pass for the trivial reason that the
    guard was never going to fire.
    """
    from tengri import Catalog
    from tengri.parameters.priors import Fixed

    t, sfr, _, params = histories
    fwd = _build(synthetic_ssp_wide, synthetic_tophat_obs, met_mode="table", neb="cloudy")
    fwd_free = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        met_mode="table",
        neb="cloudy",
        gas_free=True,
    )
    fwd_matched = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        met_mode="table",
        neb={"type": "cloudy", "logZ_gas": Fixed(_ENRICHED_ENDPOINT)},
    )

    cases = (
        ("met_gas= supplied", fwd, {"met_gas": np.full(_N, 0.1), "params": params}),
        (
            "neb_logZ_gas as a column",
            fwd_free,
            {"params": {**params, "neb_logZ_gas": np.full(_N, -0.2)}},
        ),
        ("gas pinned at the stellar endpoint", fwd_matched, {"params": params}),
    )
    for label, model, kwargs in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("error", GasStellarMetallicityWarning)
            try:
                Catalog.from_histories(model, t_gyr=t, sfr=sfr, met=enriched_history, **kwargs)
            except GasStellarMetallicityWarning as exc:  # pragma: no cover
                pytest.fail(f"{label} still warned: {exc}")


# ── the precompute paths serve a tabulated metallicity ───────────────


def test_wave_precomp_serves_a_metallicity_history(
    synthetic_ssp_wide, synthetic_tophat_obs, fwd_table_met, histories
):
    """The LUT path must carry Z(t), and must not quietly fall back to the exact one.

    Both halves matter. Agreement alone would also be produced by a silent
    fallback, so the tolerance is two-sided: close enough that the LUT is
    serving the same physics, but *not* bit-identical, which is what a fallback
    would give. Measured on the real PRSC/MILES grid the deviation is 5.5e-5
    with a 19.6x warm speedup and 17.6x lower peak memory, which is why a
    simulation catalog should be built this way.
    """
    from tengri import Catalog, WavePrecomp

    t, sfr, met, params = histories
    fwd_lut = _build(
        synthetic_ssp_wide, synthetic_tophat_obs, met_mode="table", approx=WavePrecomp()
    )

    def _predict(fwd):
        return np.asarray(
            Catalog.from_histories(fwd, t_gyr=t, sfr=sfr, met=met, params=params).predict()
        )

    ratio = _predict(fwd_lut) / _predict(fwd_table_met)
    deviation = np.abs(ratio - 1.0).max()
    assert deviation < 5e-3, f"LUT disagrees with the exact path by {deviation:.2e}"
    assert deviation > 0.0, (
        "LUT output is bit-identical to the exact path — the precompute route "
        "was not taken, so this test would pass on a silent fallback"
    )


def test_the_catalog_costs_one_compile_with_a_metallicity_history(fwd_table_met, histories):
    """Threading, not baking: the histories are arguments to one compiled program.

    ``jit(vmap(...))`` over a padded galaxy axis is what makes a catalog one
    compile rather than one per galaxy (``_map_chunks``: 236 -> 1 when the vmap
    was first jitted). Adding a Z history adds a third ``(N, n_t)`` argument, and
    the risk is that it arrives closed-over instead — constant-folded into the
    HLO, which recompiles for every new catalog and inflates the program.

    Asserted on the lowered signature rather than on a timing, so it cannot go
    flaky: three ``(N, n_t)`` arguments must appear at the entry point.
    """
    from tengri import Catalog

    t, sfr, met, params = histories
    cat = Catalog.from_histories(fwd_table_met, t_gyr=t, sfr=sfr, met=met, params=params)
    cat.predict()

    columns = {
        "dust_tau_diff": params["dust_tau_diff"],
        "sfh_t_gyr": t,
        "sfh_sfr": sfr,
        "met_history": met,
    }
    text = cat._batched_cache["photometry"].lower(columns).as_text()
    entry = next(ln for ln in text.splitlines() if "func public @main" in ln)
    assert entry.count(f"tensor<{_N}x{_N_T}x") == 3, (
        f"expected t_gyr, sfr and met_history as three ({_N}, {_N_T}) arguments; "
        f"a missing one has been baked into the program as a constant:\n{entry}"
    )


# ── the history is genuinely wired to the SED ────────────────────────


def test_metallicity_history_moves_the_photometry(fwd_table_met, histories):
    """A history that changed nothing would satisfy every guard above."""
    from tengri import Catalog

    t, sfr, _, params = histories

    def _predict(level):
        return np.asarray(
            Catalog.from_histories(
                fwd_table_met,
                t_gyr=t,
                sfr=sfr,
                met=np.full((_N, _N_T), level),
                params=params,
            ).predict()
        )

    ratio = _predict(0.4) / _predict(-2.0)
    assert np.all(ratio > 1.0), (
        f"the synthetic SSP brightens monotonically with metallicity, so a "
        f"metal-rich history must be brighter in every band; got {ratio[0]}"
    )


def test_a_shared_1d_history_matches_the_explicit_one(fwd_table_met, histories):
    """``met`` broadcasts like ``t_gyr`` — one track, many mass scalings."""
    from tengri import Catalog

    t, sfr, met, params = histories
    shared = np.asarray(
        Catalog.from_histories(
            fwd_table_met, t_gyr=_T_GYR, sfr=sfr, met=met[0], params=params
        ).predict()
    )
    explicit = np.asarray(
        Catalog.from_histories(fwd_table_met, t_gyr=t, sfr=sfr, met=met, params=params).predict()
    )
    assert np.allclose(shared / explicit, 1.0, rtol=1e-12)


def test_a_constant_history_reduces_to_the_delta_model(
    synthetic_ssp_wide, synthetic_tophat_obs, fwd_table_met, histories
):
    """The degenerate limit: Z(t) = const must equal a single met_logzsol.

    This is the test that would catch an off-by-``LOG10_ZSUN`` in either
    direction, a transposed table, or an interpolation that lands on the wrong
    age nodes — none of which produce anything but a smooth, plausible SED.
    """
    from tengri import Catalog

    t, sfr, _, params = histories
    fwd_delta = _build(synthetic_ssp_wide, synthetic_tophat_obs, met_logzsol=-0.3)

    parametric = np.asarray(
        Catalog.from_histories(fwd_delta, t_gyr=t, sfr=sfr, params=params).predict()
    )
    tabulated = np.asarray(
        Catalog.from_histories(
            fwd_table_met,
            t_gyr=t,
            sfr=sfr,
            met=np.full((_N, _N_T), -0.3),
            params=params,
        ).predict()
    )
    assert np.allclose(tabulated / parametric, 1.0, rtol=2e-3), (
        f"constant Z(t) diverges from the delta model by "
        f"{np.abs(tabulated / parametric - 1).max():.2e}"
    )


def test_simulate_carries_the_metallicity_history(fwd_table_met, histories):
    """``simulate`` is the mock-catalog verb, and must see the same Z the SED did."""
    from tengri import Catalog

    t, sfr, met, params = histories
    mock = Catalog.from_histories(
        fwd_table_met, t_gyr=t, sfr=sfr, met=met, params=params
    ).simulate(properties=("stellar_mass", "mass_weighted_metallicity"))

    assert np.asarray(mock.photometry).shape == (_N, 5)
    mwz = np.asarray(mock.properties["mass_weighted_metallicity"])
    assert np.all(np.isfinite(mwz))
    assert np.all((mwz > met.min() + LOG10_ZSUN) & (mwz < met.max() + LOG10_ZSUN)), (
        f"a mass-weighted metallicity must lie inside the history it averages; "
        f"got {mwz} against absolute log10(Z) range "
        f"[{met.min() + LOG10_ZSUN:.3f}, {met.max() + LOG10_ZSUN:.3f}]"
    )
