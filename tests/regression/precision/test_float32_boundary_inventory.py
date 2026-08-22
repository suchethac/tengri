# SPDX-License-Identifier: BSD-3-Clause
r"""The float32 boundary, pinned in both directions (#1206).

Tier B moves the forward model to pure float32 (JAX-Metal, no float64
fallback). Some published quantities are genuinely outside the float32 window
``[1.18e-38, 3.40e38]`` no matter how the arithmetic is arranged — an absorbed
luminosity really is ~1e43 erg/s — so each one gets a ``log_``-prefixed
companion that *is* representable, and the linear key stays as a float64
convenience.

This file is the inventory of that boundary, and it is **two-way**:

* a key in :data:`MUST_BE_FINITE` that goes non-finite is a regression — the
  log-domain contract it depends on has broken;
* a key in :data:`KNOWN_NOT_FLOAT32` that becomes finite is *also* a failure,
  because the inventory is now stale and the reader is being misinformed about
  what pure float32 delivers.

The second direction is the one that rots silently. A one-way test would let
the inventory drift into a lie while staying green.

Fixture note
------------
``synthetic_ssp_wide`` cannot host this test. Its ``ssp_flux`` is
``(5000/wave)**2`` (~1e-12 to 2.5e3) against a real grid's ~1.4e-70 to 9.4e-11,
so a 1e10 Msun galaxy lands at ``sed_intrinsic ~ 2.8e47`` — seventeen decades
brighter than reality and nine past the float32 ceiling. Under pure float32 the
*SED itself* overflows there, which drives ``log_L_ir`` to ``-inf`` and
``log_nion`` to ``nan``: the log contracts look broken when what is actually
broken is the input.

So the flux is rescaled to physical magnitudes (:data:`_SSP_FLUX_SCALE`). The
rescaled grid reproduces a real SSP's regime closely — ``L_ir`` 1.5e43 against a
real 2.4e43, ``lnu_age`` 5.7e29 against 2.8e28 — while staying a pure fixture,
so this guard runs on CI instead of skipping with the gitignored
``data/ssp_*.h5`` grids (data-gated tests are invisible, which is how #629/#617
reached main).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug

#: Keys whose float32-safe log form is the whole point of Tier B. Each must be
#: finite in a pure-float32 pass and agree with float64 to within ``ATOL_DEX``.
MUST_BE_FINITE = {
    "log_stellar_mass_scale": "log10(total_mass x L_sun); linear form ~1e43",
    "log_nion": "log10(Q_H); linear form ~1e49-1e56",
    "log_L_ir": "log10(L_ir); linear form ~1e43",
    "log_mstar": "log10 stellar mass; always in range",
}

#: Linear keys that are NOT float32-representable and are kept only as a
#: float64 convenience. Each MUST have a finite log companion — a linear key
#: with no log form would leave a pure-float32 consumer with nothing to read.
KNOWN_NOT_FLOAT32 = {
    "stellar_mass_scale": "log_stellar_mass_scale",
    "nion": "log_nion",
    "L_ir": "log_L_ir",
    "L_absorbed": "log_L_ir",
    # Added by #1534, which is also what emptied NOT_FLOAT32_AND_NO_COMPANION.
    "L_age": "log_L_age",
}

#: Keys that overflow float32 and have **no** log companion — i.e. they violate
#: the rule this file states, and are recorded here rather than left invisible.
#:
#: **Currently empty, and that is the goal state.** It held ``L_age`` until
#: #1534: 92% non-finite in pure float32 (float64 max 5.7e45), named by nothing,
#: and measurable with this file's own fixture the whole time — no test ever
#: looked, because every test iterated a hand-maintained list.
#: :func:`test_no_unlisted_key_overflows_float32` is the sweep that closed that,
#: and it is what will populate this dict again if a new key appears.
#:
#: A pure-float32 consumer reaching for an entry here finds ``inf`` and has
#: nowhere to go, so an entry is a defect with a deadline, not a category.
NOT_FLOAT32_AND_NO_COMPANION: dict[str, str] = {}

#: ``line_lums`` (~1e41 erg/s) also gained ``log_line_lums`` in #1534, but cannot
#: be exercised here: this fixture's model has no nebular block, so neither key is
#: published.
#:
#: **An entry here is only acceptable if another test covers it**, and that is now
#: enforced by convention rather than hope: ``test_log_companion_is_upstream.py``
#: builds a Cue model precisely so this pair is swept. It had to — the companion
#: shipped broken (``log10_magnitude`` applied *after* the value had already
#: overflowed, so 100% of the overflowing entries stayed ``+inf``) and this tuple
#: was the entire reason nothing noticed. Do not add a name here without naming
#: the test that covers it on the same line.
UNMEASURABLE_HERE = ("line_lums",)  # covered by test_log_companion_is_upstream.py

#: log10 tolerance between the float64 and pure-float32 evaluations. float32
#: carries ~7 decimal digits, so a value near 1e43 resolves to ~1e36 absolute —
#: in log space that is ~1e-6 dex; 1e-4 leaves headroom for accumulated error.
ATOL_DEX = 1.0e-4


#: Brings ``synthetic_ssp_wide``'s per-Msun flux down to a real grid's regime.
#: Chosen by measurement, not theory: it puts ``L_ir`` at 1.5e43 against a real
#: grid's 2.4e43. Anything within an order of magnitude would do — what matters
#: is that the SED sits inside the float32 window and the ~1e43 scalars do not.
_SSP_FLUX_SCALE = 1.0e-17


def _physical_ssp(ssp):
    """``ssp`` with its per-Msun flux rescaled to physical magnitudes."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    return SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * _SSP_FLUX_SCALE,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )


def _model(ssp):
    """Stellar + two-component dust + Dale IR emission, everything pinned."""
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": FIXED,
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(0.7),
            "all_params": FIXED,
        },
        dust_emission={"type": "dale2014", "all_params": FIXED},
        redshift=Fixed(0.5),
    )


@pytest.fixture(scope="module")
def derived_pair(synthetic_ssp_wide):
    """``(float64, pure float32)`` derived states from the same model spec."""
    ssp = _physical_ssp(synthetic_ssp_wide)
    f64 = _model(ssp).predict_state({}).derived
    # Precondition: the SED must be inside float32 range, or this measures the
    # fixture's brightness rather than the log contracts (see the module docstring).
    sed_max = float(np.abs(np.asarray(f64["lnu_age"], dtype=np.float64)).max())
    assert sed_max < 3.4e38, f"setup: rescaled SED still overflows float32 ({sed_max:.3e})"
    with jax.enable_x64(False):
        state32 = _model(ssp).predict_state({})
        # Materialize inside the context: the catalog is lazy, so reading a key
        # outside it would silently evaluate in float64 and prove nothing.
        f32 = {k: np.asarray(state32.derived[k]) for k in state32.derived}
        dtypes = {k: state32.derived[k].dtype for k in MUST_BE_FINITE if k in state32.derived}
    assert dtypes, "setup: no log keys published at all"
    assert all(dt == jnp.float32 for dt in dtypes.values()), (
        f"setup: not a pure-float32 evaluation, got {dtypes}"
    )
    return f64, f32


@pytest.mark.parametrize("key", sorted(MUST_BE_FINITE))
def test_log_key_survives_pure_float32(derived_pair, key):
    """Each log-domain key is published, finite in float32, and matches float64."""
    f64, f32 = derived_pair
    assert key in f64, f"{key} is not published ({MUST_BE_FINITE[key]})"
    assert key in f32, f"{key} vanished under pure float32"

    value32 = float(np.asarray(f32[key]))
    assert np.isfinite(value32), f"{key} non-finite in pure float32: {value32}"
    np.testing.assert_allclose(
        value32, float(np.asarray(f64[key])), atol=ATOL_DEX, err_msg=f"{key} float32 vs float64"
    )


@pytest.mark.parametrize("key", sorted(KNOWN_NOT_FLOAT32))
def test_linear_key_is_still_outside_float32(derived_pair, key):
    """The inventory must not go stale in the *other* direction.

    If one of these becomes float32-representable the fix is to update this
    inventory and the boundary doc — not to leave a green test asserting a
    boundary that has moved.
    """
    f64, f32 = derived_pair
    assert key in f64, f"{key} is no longer published; update the inventory"
    assert not np.all(np.isfinite(np.asarray(f32[key]))), (
        f"{key} is now finite in pure float32. That is good news, but this "
        "inventory and docs/dev/float32-tier-b-boundary.md now misdescribe the "
        "boundary — move the key out of KNOWN_NOT_FLOAT32."
    )


@pytest.mark.parametrize(("linear", "log_key"), sorted(KNOWN_NOT_FLOAT32.items()))
def test_every_unrepresentable_key_has_a_log_companion(derived_pair, linear, log_key):
    """A linear key past the float32 ceiling must leave a usable alternative.

    Without this, a pure-float32 consumer reaching for ``linear`` finds ``inf``
    and has nowhere else to go — which is exactly how the energy balance came
    to fail open to zero.
    """
    f64, f32 = derived_pair
    assert log_key in f32, f"{linear} has no float32-safe companion {log_key!r}"
    # Companions are not all scalars — ``log_L_age`` is per age bin — so this checks
    # the whole array. A companion that is itself non-finite is not a companion: it
    # is the same overflow one log deeper, which is what happens when it is computed
    # from the already-overflowed linear value instead of peak-factored at the source.
    # ``-inf`` is a value here, not a failure: it is the log-domain spelling of
    # "this bin emits exactly nothing", and it powers back through ``pow10`` to 0.0.
    # Only ``+inf``/NaN mean no answer exists (#1527). ``log_L_age`` carries two
    # -inf entries for the two genuinely dark age bins, matching float64 exactly.
    companion = np.asarray(f32[log_key], dtype=np.float64)
    reference = np.asarray(f64[log_key], dtype=np.float64)
    unusable = np.isposinf(companion) | np.isnan(companion)
    assert not unusable.any(), (
        f"{linear}'s companion {log_key} is {unusable.mean():.0%} +inf/NaN in float32, so "
        "it leaves a float32 consumer exactly where it started — a companion computed "
        "from the already-overflowed linear value does this"
    )
    lost = np.isneginf(companion) & np.isfinite(reference)
    assert not lost.any(), (
        f"{log_key} is -inf in float32 at {int(lost.sum())} positions where float64 is "
        "finite — float32 underflowed a real value into the zero sentinel"
    )


def test_no_unlisted_key_overflows_float32(derived_pair):
    """The rule, swept — rather than the handful of keys someone remembered.

    Every other test in this file iterates over :data:`MUST_BE_FINITE` or
    :data:`KNOWN_NOT_FLOAT32`, so all of them are green by construction on a key
    that appears in neither. That is how ``L_age`` sat at 92% non-finite,
    unnamed, while the file claimed to be "the inventory of that boundary".

    This walks what the model actually publishes and requires every key with a
    non-finite float32 value to be *declared* somewhere. Declaring it is cheap;
    the point is that adding a new overflowing published key must be a deliberate
    act, not something that happens quietly.

    **The comparison is against float64, not against finiteness alone.** A key
    that is non-finite in *both* precisions is not a float32 problem — it is
    saying something else, and this file is not the place to relitigate it.
    ``log_mstar_surviving`` is NaN in float64 whenever the SSP grid carries no
    mass-remaining table, deliberately: the honest answer to "how much mass is
    left" is unknown, and returning the formed mass would silently assert zero
    mass loss (#1131). A first version of this sweep tested float32 finiteness
    alone and flagged it, which would have recorded a considered sentinel as an
    overflow.
    """
    f64, f32 = derived_pair
    declared = set(MUST_BE_FINITE) | set(KNOWN_NOT_FLOAT32) | set(NOT_FLOAT32_AND_NO_COMPANION)

    undeclared = {}
    for key, value in f32.items():
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0 or key in declared:
            continue
        reference = np.asarray(f64.get(key, np.nan), dtype=np.float64)
        if reference.shape != array.shape:
            continue  # cannot compare elementwise; the pair is not the same quantity
        # The rule is "float32 lost something float64 had" — elementwise, not in
        # aggregate. A key can be legitimately non-finite in both (``-inf`` for a
        # dark age bin, NaN for the #1131 unknown-mass sentinel) without that being
        # a precision boundary; only positions where float64 is finite count.
        lost = ~np.isfinite(array) & np.isfinite(reference)
        if not lost.any():
            continue
        finite = reference[np.isfinite(reference)]
        undeclared[key] = (float(lost.mean()), float(np.max(np.abs(finite))))

    assert not undeclared, (
        "published keys go non-finite in pure float32 and are named by no category "
        "in this file:\n"
        + "\n".join(
            f"  {k}: {frac:.0%} non-finite, float64 max {peak:.2e}"
            for k, (frac, peak) in sorted(undeclared.items())
        )
        + "\n\nAdd each to KNOWN_NOT_FLOAT32 (with a log_ companion) or to "
        "NOT_FLOAT32_AND_NO_COMPANION (and file the companion as work). Do not "
        "delete this assertion — an undeclared overflow is exactly what it exists to "
        "catch."
    )


def test_the_no_companion_category_is_not_quietly_growing(derived_pair):
    """Negative control on the category above: it must stay a *known gap*, not a bin.

    ``NOT_FLOAT32_AND_NO_COMPANION`` is an escape hatch, and escape hatches
    accumulate. This pins its contents so adding to it is a visible diff, and
    checks each entry still genuinely overflows — an entry that became finite
    would mean the gap closed and the record is now misinformation.
    """
    _, f32 = derived_pair
    assert NOT_FLOAT32_AND_NO_COMPANION == {}, (
        "the no-companion gap list is no longer empty. An entry means a published key "
        "overflows float32 with nothing a float32 consumer can read instead — allowed "
        "only as a tracked defect: update this assertion and #1534 together"
    )
    for key in NOT_FLOAT32_AND_NO_COMPANION:
        if key not in f32:
            continue
        array = np.asarray(f32[key], dtype=np.float64)
        assert not np.all(np.isfinite(array)), (
            f"{key} is now fully finite in pure float32 — the gap closed, so remove it "
            "from NOT_FLOAT32_AND_NO_COMPANION rather than leaving a stale record"
        )


def test_stellar_sed_cube_is_representable(derived_pair):
    """``lnu_age`` must stay in range — it is the SED every observable is built from.

    Unlike the scalars above this is not a log contract: a real galaxy's L_nu is
    ~1e28 erg/s/Hz and comfortably inside float32. It is pinned because the
    published mass scale sits right next to it in the producer, and wiring the
    ~1e43 scale into the cube would poison every downstream flux at once.
    """
    _, f32 = derived_pair
    assert "lnu_age" in f32, "setup: the stellar cube is not published"
    finite_fraction = float(np.isfinite(np.asarray(f32["lnu_age"])).mean())
    assert finite_fraction == 1.0, (
        f"only {finite_fraction:.4%} of the float32 stellar cube is finite; "
        "an out-of-range scale has been wired into the per-age SED"
    )
