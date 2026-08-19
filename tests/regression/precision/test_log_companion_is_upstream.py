# SPDX-License-Identifier: BSD-3-Clause
r"""A log companion must be computed *upstream* of the overflow it exists to survive.

``#1534`` gave two float32-unrepresentable derived keys a ``log_`` companion.
One of them worked and one of them did nothing, and the difference is entirely
*where* the log was taken::

    # works — the log is taken on the per-Msun cube, before total_mass scales it
    log_L_age = log10_magnitude(per_msun) + log10(total_mass)

    # does nothing — line_lums is ALREADY inf by the time log10 sees it
    log_line_lums = log10_magnitude(line_lums)

``log10_magnitude(inf)`` is ``+inf`` by contract (#1527: ``+inf`` means "no
answer exists", as distinct from ``-inf`` meaning "exactly zero"). So the second
spelling faithfully reports that it has nothing — which is correct behavior and
a useless key. It advertises a float32-safe route that does not exist, which is
worse than no key at all: a consumer that finds ``log_line_lums`` in
``state.derived`` has every reason to believe it can use it.

**The one-line test that separates the two** is the sweep below::

    log_X is decorative  <=>  log_X == log10_magnitude(X_float32)

The presence of a ``log_`` key proves nothing. Only its *finiteness where the
linear sibling overflowed* does. That is asserted here for every published pair,
discovered by iteration rather than by a hand-maintained list — the same
structural move as :func:`test_no_unlisted_key_overflows_float32` in
``test_float32_boundary_inventory.py``, and for the same reason: every guard
this repository has keyed on a hand-written list has been green on the one item
nobody thought to add (#1534, #1482, #1276).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.utils.scale import log10_magnitude

pytestmark = pytest.mark.regression_bug

_PARAMS = {"sfh_delayed_log_total_mass": 10.0, "sfh_delayed_tau_gyr": 1.0}

#: Brings the synthetic per-Msun flux down to a real grid's regime, so the line
#: luminosities and ``L_age`` are physical rather than an artifact of an
#: unnormalized fixture.
#:
#: **Deliberately smaller than ``_SSP_FLUX_SCALE`` in
#: ``test_float32_boundary_inventory.py``**, which this used to match at 1e-17.
#: That file has no nebular block; this one does, and a rendered emission line
#: puts its whole luminosity into roughly one grid cell, so ``L_nu = L / dnu``
#: on the 0.72 %-spaced ``synthetic_ssp_wide`` grid is enormous. At 1e-17 the
#: nebular SED peaks at 1.099e45 and **four pixels go to inf in float32** — and
#: then the energy balance reports ``+inf`` by the #1527 corrupt contract
#: ("no answer exists"), which is correct but makes *every* companion below look
#: decorative no matter where its log was taken. The sweep would be asserting on
#: an un-integrable input rather than on where the log was taken, which is the
#: one thing it exists to measure.
#:
#: 1e-26 keeps the SED at 1.099e36 — 300x under the float32 ceiling — while
#: ``L_ir`` and 6 of 128 ``line_lums`` still overflow, so the sweep is exercised
#: exactly as before. Measured on this fixture: ``log_L_ir`` is then 45.0941
#: while ``L_ir`` is ``inf``, i.e. the companion demonstrably works, which is the
#: result 1e-17 was hiding once nebular lines stopped being silently dropped
#: (#1836).
_SSP_FLUX_SCALE = 1.0e-26


def _model(ssp):
    """A model with a **nebular block**, so ``line_lums`` is actually published.

    The float32 inventory fixture has no nebular block, which is exactly why
    ``log_line_lums`` shipped broken: it was listed in that file's
    ``UNMEASURABLE_HERE`` and therefore swept by nothing.
    """
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    scaled = SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * _SSP_FLUX_SCALE,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )
    return SEDModel.build(
        ssp_data=scaled,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        redshift=Fixed(0.1),
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": Uniform(0.5, 3.0),
            "age_gyr": Fixed(5.0),
        },
        dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
        neb={"type": "cue", "all_params": FIXED},
    )


def _derived(ssp, *, x64):
    """Derived dict, plus ``log10_magnitude(linear)`` recomputed **in the same dtype**.

    The recomputation has to happen inside the ``enable_x64`` block. Comparing a
    float32-computed ``log_X`` against a float64 recomputation differs in the
    last bits at every *finite* entry, so an exact-equality test done afterwards
    can never fire — which is how the first draft of
    :func:`test_a_working_companion_is_not_just_the_log_of_its_linear_sibling`
    passed against code it was written to fail.
    """
    with jax.enable_x64(x64):
        derived = _model(ssp).predict_state(_PARAMS).derived
        recomputed = {
            k: log10_magnitude(jnp.asarray(derived[k[4:]]))
            for k in derived
            if k.startswith("log_") and k[4:] in derived
        }
        as_np = {k: np.asarray(v, dtype=np.float64) for k, v in derived.items()}
        recomputed_np = {k: np.asarray(v, dtype=np.float64) for k, v in recomputed.items()}
    return as_np, recomputed_np


@pytest.fixture(scope="module")
def derived_pair(synthetic_ssp_wide):
    """``(float64, pure_float32, float32-recomputed-logs)`` from one model, one param set."""
    d64, _ = _derived(synthetic_ssp_wide, x64=True)
    d32, recomputed32 = _derived(synthetic_ssp_wide, x64=False)
    return d64, d32, recomputed32


def _pairs(derived):
    """Every published ``(log_X, X)`` companion pair, by naming convention."""
    return [(k, k[4:]) for k in sorted(derived) if k.startswith("log_") and k[4:] in derived]


def test_the_sweep_is_not_vacuous(derived_pair):
    """Guard the guard: a sweep over an empty set passes without testing anything."""
    d64, _, _ = derived_pair
    pairs = _pairs(d64)
    assert pairs, (
        "no (log_X, X) companion pairs are published by this model, so every sweep "
        "below is vacuously true. Either the naming convention changed or the model "
        "stopped publishing — fix the fixture, do not delete the sweep"
    )
    assert "log_line_lums" in dict(pairs), (
        f"log_line_lums is not published — this fixture no longer exercises the key "
        f"#1534 got wrong. Published pairs: {pairs}"
    )


def test_log_companions_survive_float32_where_the_linear_overflows(derived_pair):
    """The rule, swept over every pair. A companion taken after the overflow fails here."""
    d64, d32, _ = derived_pair
    broken = {}

    for log_key, lin_key in _pairs(d64):
        lin64, lin32 = d64[lin_key], d32[lin_key]
        log64, log32 = d64[log_key], d32[log_key]
        if lin64.shape != lin32.shape or log64.shape != log32.shape:
            continue

        # Entries the linear key genuinely lost to float32 range: finite in
        # float64, non-finite in float32. This is the whole reason the companion
        # exists, so it is exactly where the companion must still work.
        overflowed = ~np.isfinite(lin32) & np.isfinite(lin64)
        if not overflowed.any():
            continue  # companion not exercised by this fixture; nothing to prove

        # ``-inf`` is the legitimate "exactly zero" sentinel and is NOT a loss;
        # only ``+inf``/NaN are (#1527). Conflating them is the fail-open this
        # whole precision tree exists to prevent.
        lost = overflowed & ~np.isfinite(log32) & ~np.isneginf(log32) & np.isfinite(log64)
        if lost.any():
            i = int(np.argmax(lost))
            broken[log_key] = (
                f"{int(lost.sum())}/{int(overflowed.sum())} overflowed entries have no "
                f"log value either (e.g. index {i}: float64 {log64[i]:.3f} dex -> "
                f"float32 {log32[i]})"
            )

    assert not broken, (
        "log companion(s) computed DOWNSTREAM of the overflow they exist to survive: "
        f"{broken}. log10_magnitude(inf) is +inf, so taking the log after the linear "
        "value has already overflowed carries no information the linear key did not. "
        "Take the log where the offending scale factor ENTERS — see log_L_age in "
        "components/stellar/component.py, which factors total_mass out first"
    )


def test_log_companions_carry_the_right_value_not_merely_a_finite_one(derived_pair):
    r"""Finiteness is half the contract; the other half is agreeing with float64.

    Moving a ``log10`` upstream of a scale factor means the factor has to be
    added back as an offset — ``log10_magnitude(lum) + log10(L_sun)``, not
    ``log10_magnitude(lum)``. Drop the offset and the companion is *perfectly
    finite in float32 and wrong by 33.58 dex*, which the two sweeps above both
    accept: one asks only whether a value exists, the other only whether it was
    derived from the overflowed sibling.

    This is the same asymmetry as the Fisher matrix in #1542 — a loud failure
    is cheap, a plausible-looking wrong number is expensive.

    **The reference has to be independent.** Comparing the float32 companion
    against the float64 *companion* cannot catch a dropped offset, because both
    arms are the same expression and are wrong by the same 33.58 dex: that
    version of this test passed against a deliberately broken
    ``log10_magnitude(lum)``. The reference used here is the **linear sibling**
    in float64, which is untouched by the log-domain work and is by definition
    what ``log_X`` claims to be the log of.
    """
    d64, d32, _ = derived_pair
    wrong = {}

    for log_key, lin_key in _pairs(d64):
        lin64, log64, log32 = d64[lin_key], d64[log_key], d32[log_key]
        if lin64.shape != log64.shape or log64.shape != log32.shape:
            continue

        # The contract: log_X == log10(|X|). Checked in float64, where the
        # linear key is representable and so can serve as ground truth.
        usable = np.isfinite(lin64) & (np.abs(lin64) > 0) & np.isfinite(log64)
        if usable.any():
            # Absolute in dex: these are logs, so a dex offset IS the error
            # scale. 1e-6 is ~2 ppm in linear — far above float64 round-off on
            # a log10, far below the smallest real mistake (a dropped
            # log10(L_sun) = 33.58).
            delta = np.abs(log64[usable] - np.log10(np.abs(lin64[usable]))).max()
            if delta > 1.0e-6:
                wrong[f"{log_key} (vs log10 of its own linear key, float64)"] = f"{delta:.4g} dex"

        # Separately, float32 must track float64 — a precision claim, not a
        # correctness one, and it needs the looser bound: a Cue forward in
        # float32 accumulates ~1e-4 dex (measured).
        comparable = np.isfinite(log64) & np.isfinite(log32)
        if comparable.any():
            spread = np.abs(log32[comparable] - log64[comparable]).max()
            if spread > 1.0e-3:
                wrong[f"{log_key} (float32 vs float64)"] = f"{spread:.4g} dex"

    assert not wrong, (
        f"log companion(s) do not equal the log of what they name: {wrong}. A log moved "
        "upstream of a scale factor must add that factor back as an offset; without it "
        "the key is finite, plausible, and wrong by exactly log10(scale)"
    )


def test_a_working_companion_is_not_just_the_log_of_its_linear_sibling(derived_pair):
    """The direct form of the rule — the check that told #1534 apart from itself.

    Stated as its own test because it is the cheap diagnostic: one array
    comparison, no float64 reference needed. If ``log_X`` is elementwise equal
    to ``log10_magnitude(X_float32)`` *and* the linear overflowed anywhere, the
    companion was derived from the corpse.
    """
    _, d32, recomputed32 = derived_pair
    decorative = []

    for log_key, lin_key in _pairs(d32):
        lin32, log32 = d32[lin_key], d32[log_key]
        if lin32.shape != log32.shape or np.isfinite(lin32).all():
            continue
        if np.array_equal(log32, recomputed32[log_key], equal_nan=True):
            decorative.append(log_key)

    assert not decorative, (
        f"{decorative} is exactly log10_magnitude(<its own overflowed linear sibling>), "
        "so it fixes nothing. The log must be taken before the value goes out of range, "
        "not after"
    )
