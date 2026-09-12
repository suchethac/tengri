# SPDX-License-Identifier: BSD-3-Clause
r"""Float32 status of every composable-AGN disc block (#1206).

A durable inventory: for each registered disc block, build a composable AGN
(disc + SKIRTOR torus, CIGALE-joint norm) and compare ``sed_agn`` between
float64 and pure float32.

The **shape-class** discs — whose spectral shape depends on L_bol (temperature),
so the float32 path must take the TRUE L_bol for the shape while normalizing
MAGNITUDE to a reference — are all fixed: ``multicolor``, ``kubota_done`` and
``adaf`` carry log-space (or L_sun-unit) internals plus the
``agn_log_lbol_shape`` split. The shape-invariant discs (``powerlaw``,
``richards2006``, ``skirtor``, ``qsogen``, ``schartmann2005``) are exact under
the plain reference-evaluation + rescale.

``adaf_lopez2024`` is exact too: its CIGALE piecewise power law formed
``wavelength**coef * norm`` where, on a steep segment at long wavelength, the two
factors leave the float32 window in OPPOSITE directions (``wavelength**-4``
~1e-40 flushes to 0 while the continuity ``norm`` ~1e40 overflows) — ``0 * inf =
nan``. It is now built as one log10 sum, peak-factored before exponentiating.

``grahsp_sbpl`` is fixed too (#1206 §D), and it was the last disc in this
inventory that was not: it was blocked on a linear erg/s *parameter*
(``agn_grahsp_l5100``, ``LogUniform(1e42, 1e47)`` — the value itself was
``inf`` in float32, before any physics ran), not a kernel overflow, so a
log-space parameter (``agn_grahsp_log_l5100``) was the fix rather than a kernel
change. The disc block pre-shifts an *explicit* ``log_l5100`` by the same
``agn_log_lbol_shape - agn_log_lbol`` offset the reference-evaluation scheme
already applies to every other block (the auto-normalized case, ``log_l5100``
left unset, was already consistent with that scheme without any change), so the
existing final rescale restores the true l5100 -- no second mechanism. Measured
end-to-end (composable ``grahsp_sbpl`` + SKIRTOR torus, ``agn_log_lbol=11``):
float32 finite, max relative deviation from float64 1.6e-5 above the noise floor.

``_SHAPE_CLASS_XFAIL`` / ``_GRID_CLASS_XFAIL`` are both empty as a result:
every registered composable disc block is float32-exact. Kept (rather than
deleted) as the named place a future non-float32-safe disc joins, with its own
entry and rationale.

This test pins the exact discs (regression guard) and ``xfail``\ s the rest
(progress tracker: fixing one turns its ``xfail`` into an unexpected pass). It is
the enforced record of "checked every AGN disc component".

Coverage note: every build here carries no ``agn_ir_frac`` (fracAGN), so
``compose_l_nu`` selects the ``_disc_debited`` branch of the disc luminosity
(the plain ``agn_log_lbol``-normalized shape) for the value this file
verdicts; the ``agn_power x R`` SKIRTOR-R-tie branch -- active only under a
non-zero fracAGN with ``cigale_joint`` -- is still traced on every build here
but never the selected branch, so a float32 fault confined to it would not
surface in this inventory.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

# Discs whose spectral shape is invariant under L_bol (template / power-law) OR
# whose L_bol-dependent shape is now handled on the float32 path (multicolor,
# kubota_done — log-space internals + shape/normalization split).
#: Discs run on every PR — one per *fix mechanism*, so a regression in any of the
#: float32 techniques still fails fast:
#:
#: * ``multicolor``     — log-space disc internals + the shape/normalization split
#:                        (and one of the two science defaults)
#: * ``powerlaw``       — shape-invariant control: plain reference-eval + rescale
#:                        (the other science default)
#: * ``slone_netzer``   — peak-factored template normalization + a float32-scale floor
#: * ``adaf_lopez2024`` — log-space piecewise power law, the kernel also used by
#:                        ``skirtor`` and ``schartmann2005``
#:
#: The rest are marked ``slow``: they re-test the *same* mechanisms on further
#: disc impls, and each costs a fresh JAX compile in both precisions (~96% of this
#: file's runtime), so they earn their keep nightly rather than on every push.
#: ``relagn`` is here deliberately — it is the most expensive case (26 MB grid,
#: enough to OOM a parallel worker) and its mechanism is covered by slone_netzer.
_PR_DISCS = ["multicolor", "powerlaw", "slone_netzer", "adaf_lopez2024"]
_NIGHTLY_DISCS = [
    "kubota_done",
    "adaf",
    "richards2006",
    "skirtor",
    "qsogen",
    "schartmann2005",
    "relagn",
    # Grid-backed (template + bolometric-integral precompute), same OOM/compile
    # cost profile as relagn/slone_netzer -- nightly rather than every-PR
    # (#1206 §D: fixed, was the sole _GRID_CLASS_XFAIL member).
    "grahsp_sbpl",
]
_EXACT_DISCS = _PR_DISCS + _NIGHTLY_DISCS

#: Full matrix for the parametrization: PR discs plain, the rest ``slow``-marked so
#: the default run deselects them (pyproject ``addopts`` carries ``-m 'not slow'``)
#: and the scheduled / ``run-slow-tests`` job picks them up.
_EXACT_DISC_PARAMS = [*_PR_DISCS] + [
    pytest.param(d, marks=pytest.mark.slow) for d in _NIGHTLY_DISCS
]

# Shape depends on L_bol; float32 reference evaluation gives the wrong shape.
_SHAPE_CLASS_XFAIL = []

# Non-finite in float32 even at the reference L_bol — a distinct internal
# overflow. Empty (#1206 §D): grahsp_sbpl, the last member, is fixed.
_GRID_CLASS_XFAIL = []


def _sed_agn(ssp, disc, dtype):
    obs = Observation(photometry=Photometry.from_names(["sdss_r", "wise_w3", "wise_w4"]))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": 0.3,
            "tau_bc": 0.0,
        },
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "disc": {"type": disc, "all_params": Fixed(DEFAULT)},
            "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
            "norm": "cigale_joint",
            # The float32 verdict below is measured AT this luminosity, so it
            # has to be the one the forward uses. This build used to carry
            # ``fracAGN: 0.1`` too, which derives the AGN power from the
            # dust-absorbed stellar luminosity and discards the stated 11.0
            # (R55 refuses that pair); the inventory was then silently
            # measuring each disc at the fracAGN-derived scale instead.
            "log_lbol": Fixed(11.0),  # #2069: pinned to break flat direction
        },
        redshift=Fixed(0.1),
    )
    p = {
        k: jnp.asarray(v, dtype=dtype)
        for k, v in {"sfh_delayed_log_total_mass": 10.0, "agn_log_lbol": 11.0}.items()
    }
    return np.asarray(model.predict_state(p).derived["sed_agn"])


#: Disc blocks backed by a large HDF5 template grid. Holding the float64 and
#: float32 copies at once (plus their compiled executables) is enough to OOM a
#: parallel pytest worker — the RELAGN grid alone is 26 MB — so the float64 side is
#: released before the float32 model is built.
_LARGE_GRID_DISCS = frozenset({"relagn", "slone_netzer"})


#: A step this large between adjacent grid points is a model cutoff, not physics.
#: The panchromatic grid samples ~0.9 % in wavelength here, so no smooth SED — not
#: even a Wien tail — moves by 10x from one point to the next.
_DISCONTINUITY_STEP = 10.0


def _locally_continuous(ref):
    """Mask out reference points that sit on a step, not on a curve.

    The AGN SED has a hard cutoff at 1 mm: the SKIRTOR torus stops there and
    ``sed_agn`` falls by 2700x between two adjacent grid points, both of which
    are well above the ``live`` floor. Comparing float32 against float64 *at* a
    step like that is ill-posed — the two dtypes land on opposite sides of a
    boundary one ulp wide, so the measured "error" is the size of the step
    rather than of any rounding, and no amount of float32 hardening can close
    it. (Measured: 3.3e-2 for ``powerlaw``, against 7e-6 everywhere else.)

    The mask is derived from the **float64 reference only**. That is what makes
    this a narrowing of scope rather than a weakening of the guard: a float32
    value that blows up where float64 is smooth is still compared in full.

    Parameters
    ----------
    ref : ndarray, shape (n_wave,)
        Float64 reference SED [erg/s/Hz].

    Returns
    -------
    ndarray of bool, shape (n_wave,)
        True where ``ref`` is within a factor ``_DISCONTINUITY_STEP`` of both
        neighbors.
    """
    a = np.abs(ref)
    prev = np.concatenate([a[:1], a[:-1]])
    nxt = np.concatenate([a[1:], a[-1:]])
    lo, hi = 1.0 / _DISCONTINUITY_STEP, _DISCONTINUITY_STEP
    with np.errstate(divide="ignore", invalid="ignore"):
        r_prev = np.where(prev > 0, a / prev, 1.0)
        r_next = np.where(nxt > 0, a / nxt, 1.0)
    return (r_prev > lo) & (r_prev < hi) & (r_next > lo) & (r_next < hi)


def _f32_matches_f64(ssp, disc):
    with jax.enable_x64(True):
        ref = _sed_agn(ssp, disc, jnp.float64)
    if disc in _LARGE_GRID_DISCS:
        import gc

        jax.clear_caches()
        gc.collect()
    with jax.enable_x64(False):
        f32 = _sed_agn(ssp, disc, jnp.float32)
    if not np.all(np.isfinite(f32)):
        return False, "non-finite in float32"
    peak = np.abs(ref).max()
    bright = np.abs(ref) > 1e-6 * peak
    live = bright & _locally_continuous(ref)
    # The exclusion is meant to drop a couple of cutoff points, not to quietly
    # empty the comparison. If a change ever makes most of the SED look like a
    # step, this test must fail loudly rather than pass on three survivors.
    dropped = int(bright.sum() - live.sum())
    if dropped > 0.01 * bright.sum():
        return False, f"{dropped} of {bright.sum()} bright points are discontinuous"
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    return rel.max() < 1e-3, f"max_rel={rel.max():.2e} (excluded {dropped} cutoff points)"


@pytest.mark.parametrize("disc", _EXACT_DISC_PARAMS)
def test_disc_is_exact_in_float32(ssp_bare, disc):
    """These disc blocks must match float64 to float32 eps (regression guard)."""
    ok, detail = _f32_matches_f64(ssp_bare, disc)
    assert ok, f"disc '{disc}' regressed on the pure-float32 path: {detail}"


@pytest.mark.parametrize("disc", _SHAPE_CLASS_XFAIL + _GRID_CLASS_XFAIL)
@pytest.mark.xfail(reason="#1206 follow-up: disc not yet float32-hardened", strict=True)
def test_disc_float32_pending(ssp_bare, disc):
    """Progress tracker — fixing one of these flips its xfail to an unexpected pass."""
    ok, detail = _f32_matches_f64(ssp_bare, disc)
    assert ok, f"disc '{disc}' still float32-broken: {detail}"


# ``test_grid_class_disc_warns_in_float32`` and
# ``test_float32_safe_disc_does_not_warn`` are removed (#1206 §D):
# ``_GRID_CLASS_XFAIL`` is empty (``grahsp_sbpl`` fixed, the last member),
# so there is no longer a non-float32-safe disc to warn about, and
# ``Float32UnsafeAGNWarning`` -- the class both tests exercised -- is
# removed from ``tengri.components.agn.component`` rather than kept
# dormant. ``test_disc_is_exact_in_float32`` (parametrized over
# ``_EXACT_DISC_PARAMS``, which now includes ``grahsp_sbpl``) is the
# regression guard that replaces them.
