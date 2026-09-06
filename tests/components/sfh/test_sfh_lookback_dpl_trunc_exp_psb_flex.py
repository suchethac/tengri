# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for ``dpl_lookback``, ``trunc_exp`` and ``psb_flex``.

Each of the three is a sibling of an SFH tengri already carried, differing in
one stated way:

``dpl_lookback``
    The double power law of Carnall+2018 written against the **stellar age**
    rather than cosmic time since formation. ``dpl`` (BAGPIPES ``dblplaw``)
    applies the same algebra to ``T = age - t_lookback``, and the form is not
    symmetric under that reflection, so the two are different models: measured
    here, 41-96 % of the stellar mass lands in different age bins on matched
    parameters.

``trunc_exp``
    ``declining_exp`` — the FSPS ``sfh=1`` / BAGPIPES ``exponential`` tau model
    — plus a young-end cutoff and a signed tau. Pinned here to be bit-identical
    to ``declining_exp`` when the cutoff is at zero, so the new content is
    exactly the cutoff.

``psb_flex``
    The Suess+2022 post-starburst SFH with its flexible quenching zone cut into
    five equal-width bins instead of one. ``psb_suess2022`` is the one-bin
    special case; the shared shape function must return a *bit-identical* SFH
    when no ``flex_*`` ratio is supplied, which is what makes that entry a case
    of this one rather than a second implementation of the same physics.

These tests need no external package and run in the default tier; the numerical
comparison against Synthesizer lives in
``tests/crossval/test_synthesizer_crossval.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel, builders, describe, list_sfh_models
from tengri.components.stellar.sfh.mean_sfh import (
    declining_exponential,
    dpl,
    dpl_lookback,
    trunc_exp,
)
from tengri.components.stellar.sfh.nonparametric import (
    PSB_FLEX_DEFAULT_MAX_AGE_GYR,
    PSB_FLEX_DEFAULT_N_FIXED,
    psb_continuity,
    psb_continuity_flex,
)
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, validate_bin_edges_gyr

pytestmark = pytest.mark.contract


NEW_TYPES = ("dpl_lookback", "trunc_exp", "psb_flex")

# A dense lookback grid; fine enough that the two truncation boundaries each
# fall inside one narrow cell.
T_FINE_YR = jnp.linspace(0.0, 13.8e9, 20_001)
T_LOG_YR = jnp.logspace(6.0, 10.14, 256)

# An ascending fixed-bin ladder for the ``psb_continuity`` tests. The shipped
# default ladder is NOT ascending once ``tflex_gyr`` exceeds 0.3 Gyr, which its
# own prior always does; that is why ``psb_flex`` derives its fixed bins from
# ``tflex_gyr`` instead. See
# test_psb_flex_ladder_is_ascending_for_every_tflex_in_its_prior.
PSB_FIXED_EDGES_GYR = jnp.array([2.0, 5.0, 9.0, 13.7])


# ── 1. psb_continuity: the flex ratios must be a strict extension ──


PSB_LEGACY_CASES = [
    dict(log_total_mass=10.0, tlast_gyr=0.1, tflex_gyr=1.5),
    dict(
        log_total_mass=10.5,
        tlast_gyr=0.3,
        tflex_gyr=1.8,
        ratio_young=1.0,
        ratio_old_0=0.2,
        ratio_old_1=-0.3,
    ),
    dict(log_total_mass=9.0, tlast_gyr=0.02, tflex_gyr=0.9, ratio_young=-1.5, ratio_old_1=0.9),
]


def _legacy_psb(age_yr, log_total_mass, tlast_gyr, tflex_gyr, fixed_edges_gyr, **ratio_kwargs):
    """The one-flex-bin PSB history, written out independently of the code.

    Written out from the Suess+2022 / Prospector construction rather than
    delegating to :func:`psb_continuity`: a reference that called the function
    under test would agree with any implementation, including a broken one.
    """
    fixed = np.asarray(fixed_edges_gyr, dtype=float)
    edges = np.concatenate([np.array([0.0, tlast_gyr, tflex_gyr]), fixed[1:]])
    n_fixed = fixed.shape[0] - 1
    ratio_old = [ratio_kwargs.get(f"ratio_old_{i}", 0.0) for i in range(n_fixed - 1)]
    # log SFR of each old bin, oldest = reference 0.
    log_old = np.zeros(n_fixed)
    for i in range(n_fixed - 2, -1, -1):
        log_old[i] = log_old[i + 1] + ratio_old[i]
    log_flex = log_old[0]
    log_young = log_flex + ratio_kwargs.get("ratio_young", 0.0)
    log_bins = np.concatenate([[log_young, log_flex], log_old])

    sfr = 10.0**log_bins
    mass = np.sum(sfr * np.diff(edges) * 1e9)
    sfr = sfr * (10.0**log_total_mass) / mass

    out = np.zeros_like(np.asarray(age_yr, dtype=float))
    ages = np.asarray(age_yr, dtype=float) / 1e9
    for i in range(len(sfr)):
        lo, hi = edges[i], edges[i + 1]
        sel = (ages >= lo) & (ages < hi) if i else (ages < hi)
        out[sel] = sfr[i]
    return out


@pytest.mark.parametrize("case", PSB_LEGACY_CASES)
def test_psb_continuity_without_flex_ratios_is_the_one_bin_model(case):
    """No ``flex_*`` kwarg means one flexible bin, exactly as before.

    This is what makes ``psb_suess2022`` the ``nflex = 1`` case of the same
    function rather than a second implementation of the same physics.
    """
    got = np.asarray(psb_continuity(T_FINE_YR, **case, bin_edges_gyr=PSB_FIXED_EDGES_GYR))
    want = _legacy_psb(np.asarray(T_FINE_YR), **case, fixed_edges_gyr=PSB_FIXED_EDGES_GYR)
    inside = np.asarray(T_FINE_YR) < 13.7e9
    assert np.max(np.abs(got[inside] / want[inside] - 1.0)) < 1e-12


def test_psb_zero_flex_ratios_collapse_onto_the_one_bin_history():
    """Five flex bins at equal SFR are the single flex bin, to float precision.

    The equal-width flex bins carry no structure of their own: only the
    ``flex_*`` ratios do. If they did, adding bins would move mass around even
    at zero ratios and the ``psb_suess2022`` special case would be a fiction.
    """
    case = dict(
        log_total_mass=10.5,
        tlast_gyr=0.3,
        tflex_gyr=1.8,
        ratio_young=1.0,
        ratio_old_0=0.2,
        ratio_old_1=-0.3,
        bin_edges_gyr=PSB_FIXED_EDGES_GYR,
    )
    one = np.asarray(psb_continuity(T_FINE_YR, **case))
    five = np.asarray(
        psb_continuity(T_FINE_YR, **case, flex_0=0.0, flex_1=0.0, flex_2=0.0, flex_3=0.0)
    )
    live = one > 0
    assert np.max(np.abs(five[live] / one[live] - 1.0)) < 1e-12
    assert np.max(np.abs(five[~live])) == 0.0


def test_psb_flex_ratio_moves_mass_within_the_flexible_zone():
    """A non-zero ``flex_i`` must change the history, and only inside [tlast, tflex]."""
    case = dict(log_total_mass=10.0, tlast_gyr=0.2, tflex_gyr=2.0, ratio_young=0.3)
    flat = np.asarray(psb_continuity_flex(T_FINE_YR, **case, flex_0=0.0, flex_1=0.0, flex_2=0.0))
    tilted = np.asarray(
        psb_continuity_flex(T_FINE_YR, **case, flex_0=0.8, flex_1=-0.6, flex_2=0.3)
    )
    t_gyr = np.asarray(T_FINE_YR) / 1e9

    zone = (t_gyr > 0.2) & (t_gyr < 2.0)
    assert np.max(np.abs(tilted[zone] / flat[zone] - 1.0)) > 0.2, (
        "flex ratios left the quenching zone unchanged"
    )
    # Outside the zone the *shape* is untouched; only the shared normalization
    # moves, so the two curves stay proportional there.
    old = (t_gyr > 2.5) & (t_gyr < 13.0) & (flat > 0)
    ratios = tilted[old] / flat[old]
    assert np.max(np.abs(ratios / np.median(ratios) - 1.0)) < 1e-12


def test_psb_flex_conserves_the_declared_total_mass():
    """``sum(SFR_i * dt_i)`` over the bins is ``10**log_total_mass``."""
    sfr = psb_continuity_flex(
        T_FINE_YR,
        log_total_mass=10.25,
        tlast_gyr=0.2,
        tflex_gyr=2.0,
        ratio_young=0.6,
        flex_0=0.4,
        flex_1=-0.3,
        flex_2=0.2,
        flex_3=0.1,
        ratio_old_0=-0.2,
        ratio_old_1=0.3,
    )
    # Riemann sum on the uniform grid: the history is piecewise constant, so
    # trapezoid would clip half a cell at each jump.
    dt = float(T_FINE_YR[1] - T_FINE_YR[0])
    mass = float(jnp.sum(sfr) * dt)
    assert mass == pytest.approx(10.0**10.25, rel=2e-3, abs=0.0)
    assert bool(jnp.all(sfr >= 0.0))


@pytest.mark.parametrize("tflex_gyr", [0.5, 0.9, 2.0, 5.0])
def test_psb_flex_ladder_is_ascending_for_every_tflex_in_its_prior(tflex_gyr):
    """``psb_flex`` lays its fixed bins out FROM ``tflex``, so they never cross.

    ``psb_continuity``'s ladder splices ``tflex_gyr`` in ahead of the caller's
    fixed edges, so it is the caller's job to keep ``tflex_gyr`` below the
    first of them. That is checked here on the derived ladder, which is what
    makes ``jnp.searchsorted`` well defined and the mass sum exact across the
    whole Uniform(0.5, 5.0) prior on ``tflex_gyr``.
    """
    fixed = np.linspace(tflex_gyr, PSB_FLEX_DEFAULT_MAX_AGE_GYR, PSB_FLEX_DEFAULT_N_FIXED + 1)
    edges = np.concatenate([[0.0, 0.2, tflex_gyr], fixed[1:]])
    assert np.all(np.diff(edges) > 0), edges

    sfr = psb_continuity_flex(
        T_FINE_YR,
        log_total_mass=10.0,
        tlast_gyr=0.2,
        tflex_gyr=tflex_gyr,
        ratio_young=0.5,
        flex_0=0.2,
        flex_1=-0.3,
        flex_2=0.1,
        flex_3=0.05,
        ratio_old_0=-0.2,
        ratio_old_1=0.3,
    )
    dt = float(T_FINE_YR[1] - T_FINE_YR[0])
    assert float(jnp.sum(sfr) * dt) == pytest.approx(1e10, rel=2e-3, abs=0.0)


def test_psb_flex_refuses_a_bin_edge_count_its_ratios_cannot_fill():
    """A surplus/short ``bin_edges_gyr`` would be swallowed silently (#1975)."""
    with pytest.raises(ValueError, match="ratio_old parameters"):
        validate_bin_edges_gyr("psb_flex", np.array([2.0, 5.0, 9.0, 11.0, 13.7]))
    # The count the registry actually declares is accepted.
    validate_bin_edges_gyr("psb_flex", np.array([2.0, 6.0, 10.0, 13.7]))


def test_every_psb_flex_ratio_lands_on_the_pair_of_bins_it_names():
    """Each declared ratio is ``log10(SFR_i / SFR_{i+1})`` for the pair it names.

    Read off the returned history bin by bin rather than inferred from a
    global norm: a ratio wired to the wrong end of the flexible zone (say
    ``ratio_young`` anchored to the oldest flex bin instead of the youngest)
    leaves the total mass, the positivity and the zone-changed check all
    intact, and only this comparison sees it.
    """
    tlast_gyr, tflex_gyr = 0.2, 2.0
    n_flex = 5
    flex = [0.7, -0.5, 0.4, -0.3]
    sfr = np.asarray(
        psb_continuity_flex(
            T_FINE_YR,
            log_total_mass=10.0,
            tlast_gyr=tlast_gyr,
            tflex_gyr=tflex_gyr,
            ratio_young=0.6,
            **{f"flex_{i}": flex[i] for i in range(len(flex))},
            ratio_old_0=-0.25,
            ratio_old_1=0.35,
        )
    )
    t_gyr = np.asarray(T_FINE_YR) / 1e9

    edges = np.concatenate(
        [
            [0.0, tlast_gyr],
            np.linspace(tlast_gyr, tflex_gyr, n_flex + 1)[1:],
            np.linspace(tflex_gyr, PSB_FLEX_DEFAULT_MAX_AGE_GYR, PSB_FLEX_DEFAULT_N_FIXED + 1)[1:],
        ]
    )
    # SFR sampled at each bin's midpoint: piecewise constant, so one sample is
    # the whole bin.
    mid = 0.5 * (edges[:-1] + edges[1:])
    per_bin = np.array([sfr[np.searchsorted(t_gyr, m)] for m in mid])
    assert np.all(per_bin > 0)

    got = np.log10(per_bin[:-1] / per_bin[1:])
    # youngest -> oldest: young/flex_0, the four flex steps, the pinned
    # flex_last/fixed_0 step, then the two fixed steps.
    want = np.array([0.6, *flex, 0.0, -0.25, 0.35])
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


# ── 2. trunc_exp: a strict extension of declining_exp ──────────────


@pytest.mark.parametrize(
    ("tau_yr", "age_yr"), [(1e9, 8e9), (2e9, 10e9), (5e8, 13.0e9), (7e9, 5e9)]
)
def test_trunc_exp_with_no_cutoff_is_declining_exp(tau_yr, age_yr):
    """``end = 0`` must reproduce ``declining_exp`` bit for bit.

    The exponent stabilization added for the signed tau is a constant shift
    under a normalization that is invariant to it, so for positive tau it has
    to be exactly zero -- not merely small.
    """
    a = declining_exponential(T_LOG_YR, log_total_mass=10.0, tau=tau_yr, age=age_yr)
    b = trunc_exp(T_LOG_YR, log_total_mass=10.0, tau=tau_yr, age=age_yr, end=0.0)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_trunc_exp_cutoff_empties_the_young_end():
    """SFR is exactly zero below ``end`` and the mass moves to the old side."""
    end_yr = 2e9
    sfr = np.asarray(trunc_exp(T_FINE_YR, log_total_mass=10.0, tau=2e9, age=9e9, end=end_yr))
    t = np.asarray(T_FINE_YR)
    # One cell of slack at the boundary: the window is cell-averaged, by design.
    h = float(T_FINE_YR[1] - T_FINE_YR[0])
    assert np.all(sfr[t < end_yr - h] == 0.0)
    assert np.all(sfr[t > 9e9 + h] == 0.0)
    assert np.any(sfr[(t > end_yr + h) & (t < 9e9 - h)] > 0.0)


def test_trunc_exp_negative_tau_rises_toward_the_present():
    """Sign convention: a positive tau declines in cosmic time.

    The exponent is ``-(age - t_lookback) / tau``, so a positive tau makes the
    SFR *increase* with lookback age (highest at formation); a negative tau
    does the reverse.
    """
    pos = np.asarray(trunc_exp(T_FINE_YR, log_total_mass=10.0, tau=2e9, age=10e9, end=0.0))
    neg = np.asarray(trunc_exp(T_FINE_YR, log_total_mass=10.0, tau=-2e9, age=10e9, end=0.0))
    t = np.asarray(T_FINE_YR)
    young = (t > 1e9) & (t < 2e9)
    old = (t > 8e9) & (t < 9e9)
    assert pos[old].mean() > pos[young].mean()
    assert neg[old].mean() < neg[young].mean()


@pytest.mark.parametrize("tau_yr", [2e7, -2e7, 1e11, -1e6])
@pytest.mark.parametrize("end_yr", [0.0, 4e9])
def test_trunc_exp_stays_finite_for_extreme_tau(tau_yr, end_yr):
    """The exponent shift keeps a very small or negative tau from over/underflowing.

    ``end_yr = 4e9`` also exercises the masked region ahead of the cutoff,
    where the *unshifted* exponent for a small negative tau reaches ``exp(4000)``
    and a plain ``exp(...) * 0`` would be NaN rather than zero.
    """
    sfr = trunc_exp(T_FINE_YR, log_total_mass=10.0, tau=tau_yr, age=13.0e9, end=end_yr)
    assert bool(jnp.all(jnp.isfinite(sfr)))
    assert bool(jnp.all(sfr >= 0.0))
    assert float(jnp.trapezoid(sfr, T_FINE_YR)) == pytest.approx(1e10, rel=1e-6, abs=0.0)


def test_trunc_exp_degenerate_window_returns_zeros_not_nan():
    """``end >= age`` is an empty window: zeros, never NaN."""
    sfr = trunc_exp(T_FINE_YR, log_total_mass=10.0, tau=-1e6, age=3e9, end=5e9)
    assert bool(jnp.all(jnp.isfinite(sfr)))
    assert float(jnp.max(jnp.abs(sfr))) == 0.0


# ── 3. dpl_lookback is a different model from dpl ──────────────────


@pytest.mark.parametrize(
    ("alpha", "beta", "scale_yr"), [(2.0, 1.0, 2e9), (1.5, 1.0, 3e9), (3.0, 0.5, 1.5e9)]
)
def test_dpl_lookback_is_not_a_reparameterization_of_dpl(alpha, beta, scale_yr):
    """The two place a large fraction of the mass in different age bins.

    Quantified as the total-variation distance L1/2 between the two unit-mass
    histories, i.e. the fraction of stellar mass that would have to be moved to
    turn one into the other.
    """
    t = np.asarray(T_FINE_YR)
    a = np.asarray(
        dpl(T_FINE_YR, alpha=alpha, beta=beta, tau=scale_yr, age=13.8e9, log_total_mass=0.0)
    )
    b = np.asarray(
        dpl_lookback(
            T_FINE_YR,
            peak=scale_yr,
            alpha=alpha,
            beta=beta,
            age=13.8e9,
            end=0.0,
            log_total_mass=0.0,
        )
    )
    p = a / np.trapezoid(a, t)
    q = b / np.trapezoid(b, t)
    l1_half = 0.5 * np.trapezoid(np.abs(p - q), t)
    assert l1_half > 0.35, (
        f"dpl and dpl_lookback differ by only L1/2={l1_half:.3f} at "
        f"alpha={alpha}, beta={beta} -- if they agreed, one of them is wrong"
    )


def test_dpl_lookback_peaks_where_the_algebra_says_it_does():
    """SFR turns over at ``peak * (beta/alpha)**(1/(alpha+beta))``."""
    alpha, beta, peak = 2.0, 1.0, 2e9
    sfr = np.asarray(
        dpl_lookback(
            T_FINE_YR,
            peak=peak,
            alpha=alpha,
            beta=beta,
            age=13.0e9,
            end=0.0,
            log_total_mass=0.0,
        )
    )
    t_peak = float(np.asarray(T_FINE_YR)[int(np.argmax(sfr))])
    expected = peak * (beta / alpha) ** (1.0 / (alpha + beta))
    assert t_peak == pytest.approx(expected, rel=2e-3, abs=0.0)


def test_dpl_lookback_truncations_bracket_the_history():
    """Zero outside ``[end, age]``, positive inside, and mass-normalized."""
    sfr = np.asarray(
        dpl_lookback(
            T_FINE_YR,
            peak=1.5e9,
            alpha=3.0,
            beta=0.5,
            age=12e9,
            end=0.3e9,
            log_total_mass=10.0,
        )
    )
    t = np.asarray(T_FINE_YR)
    h = float(T_FINE_YR[1] - T_FINE_YR[0])
    assert np.all(sfr[t < 0.3e9 - h] == 0.0)
    assert np.all(sfr[t > 12e9 + h] == 0.0)
    assert np.all(sfr >= 0.0)
    assert float(np.trapezoid(sfr, t)) == pytest.approx(1e10, rel=1e-6, abs=0.0)


# ── 4. JIT / gradient health ───────────────────────────────────────


@pytest.mark.parametrize(
    ("fn", "kwargs", "wrt"),
    [
        (
            dpl_lookback,
            dict(peak=2e9, alpha=2.0, beta=1.0, age=10e9, end=5e8, log_total_mass=10.0),
            "alpha",
        ),
        (
            dpl_lookback,
            dict(peak=2e9, alpha=2.0, beta=1.0, age=10e9, end=5e8, log_total_mass=10.0),
            "beta",
        ),
        (
            dpl_lookback,
            dict(peak=2e9, alpha=2.0, beta=1.0, age=10e9, end=5e8, log_total_mass=10.0),
            "peak",
        ),
        (
            dpl_lookback,
            dict(peak=2e9, alpha=2.0, beta=1.0, age=10e9, end=5e8, log_total_mass=10.0),
            "end",
        ),
        (
            dpl_lookback,
            dict(peak=2e9, alpha=2.0, beta=1.0, age=10e9, end=5e8, log_total_mass=10.0),
            "age",
        ),
        (trunc_exp, dict(log_total_mass=10.0, tau=1e9, age=8e9, end=5e8), "tau"),
        (trunc_exp, dict(log_total_mass=10.0, tau=1e9, age=8e9, end=5e8), "end"),
        (trunc_exp, dict(log_total_mass=10.0, tau=1e9, age=8e9, end=5e8), "age"),
        (trunc_exp, dict(log_total_mass=10.0, tau=-1e9, age=8e9, end=5e8), "tau"),
    ],
)
def test_every_parameter_carries_a_finite_nonzero_gradient(fn, kwargs, wrt):
    """A truncation boundary with a zero gradient gives a sampler no signal (#1374).

    The objective is the mass-weighted mean lookback time, not ``sum(sfr)``:
    every one of these histories is renormalized to a fixed total mass, so on a
    uniform grid ``sum(sfr)`` is that mass divided by the spacing and its
    gradient is exactly zero for *any* implementation. A test built on it would
    pass whether or not the parameter reached the model.
    """

    def objective(value):
        sfr = fn(T_FINE_YR, **{**kwargs, wrt: value})
        return jnp.sum(sfr * T_FINE_YR) / jnp.sum(sfr)

    grad = jax.grad(objective)(kwargs[wrt])
    assert bool(jnp.isfinite(grad)), f"non-finite gradient w.r.t. {wrt}"
    assert float(grad) != 0.0, f"zero gradient w.r.t. {wrt}"


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (
            dpl_lookback,
            dict(peak=2e9, alpha=2.0, beta=1.0, age=10e9, end=5e8, log_total_mass=10.0),
        ),
        (trunc_exp, dict(log_total_mass=10.0, tau=-1e9, age=8e9, end=5e8)),
    ],
)
def test_jit_matches_eager(fn, kwargs):
    eager = fn(T_LOG_YR, **kwargs)
    jitted = jax.jit(lambda **kw: fn(T_LOG_YR, **kw))(**kwargs)
    assert float(jnp.max(jnp.abs(eager - jitted))) < 1e-9 * float(jnp.max(eager))


# ── 5. the public grammar ──────────────────────────────────────────


@pytest.mark.parametrize("sfh_type", NEW_TYPES)
def test_registered_on_every_discovery_surface(sfh_type):
    assert sfh_type in SFH_REGISTRY
    assert sfh_type in {row["name"] for row in list_sfh_models()}
    assert sfh_type in builders.sfh.available()
    assert describe(sfh_type) is not None


@pytest.mark.parametrize("sfh_type", NEW_TYPES)
def test_every_declared_parameter_carries_the_type_prefix(sfh_type):
    """NAMING_CONTRACT 3.2: ``sfh_<type>_<name>`` for every declared parameter."""
    for name in SFH_REGISTRY[sfh_type].params:
        assert name.startswith(f"sfh_{sfh_type}_"), name


@pytest.mark.parametrize("sfh_type", NEW_TYPES)
def test_every_declared_parameter_is_wired_to_the_shape_function(sfh_type):
    """A declared parameter with no ``internal_param_map`` entry is a silent no-op."""
    spec = SFH_REGISTRY[sfh_type]
    assert set(spec.params) == set(spec.internal_param_map)


# (type, short key, value that must visibly change the history)
PUBLIC_CASES = [
    ("dpl_lookback", "peak_gyr", 3.5),
    ("dpl_lookback", "alpha", 4.0),
    ("trunc_exp", "end_gyr", 0.8),
    ("trunc_exp", "tau_gyr", 8.0),
    ("psb_flex", "ratio_flex_1", 0.7),
    ("psb_flex", "ratio_young", 1.2),
]


def _sfr_history(ssp, group):
    model = SEDModel.build(ssp_data=ssp, sfh=group, redshift=Fixed(0.05))
    return np.asarray(model.predict_state({}).derived["sfr_history"])


@pytest.mark.parametrize(("sfh_type", "short", "value"), PUBLIC_CASES)
def test_short_key_reaches_the_forward_model(ssp_data_fsps, sfh_type, short, value):
    """``SEDModel.build(sfh={'type': t, <short>: ...}).predict_state`` sees the change."""
    base = _sfr_history(ssp_data_fsps, {"type": sfh_type, "all_params": Fixed(DEFAULT)})
    assert np.all(np.isfinite(base))
    assert base.max() > 0.0
    got = _sfr_history(
        ssp_data_fsps,
        {"type": sfh_type, "all_params": Fixed(DEFAULT), short: Fixed(value)},
    )
    frac = np.max(np.abs(got - base) / np.maximum(np.abs(base), 1e-30))
    assert frac > 0.01, f"short key {short!r} was ignored (max rel change {frac:.3g})"


@pytest.mark.parametrize("sfh_type", NEW_TYPES)
def test_wildcard_free_builds_a_fittable_model(ssp_data_fsps, sfh_type):
    """``all_params: FREE`` must expand to a non-empty free-parameter set."""
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        sfh={"type": sfh_type, "all_params": FREE},
        redshift=Fixed(0.05),
    )
    free = [p for p in model.spec.free_params if p.startswith("sfh_")]
    assert free, f"{sfh_type!r} freed no SFH parameter under all_params=FREE"
    assert all(p.startswith(f"sfh_{sfh_type}_") for p in free), free


def test_psb_flex_adds_only_the_flex_ratios_to_the_psb_suess2022_parameter_set():
    """The two entries differ by the ``ratio_flex_*`` block and by nothing else.

    ``psb_suess2022`` is the ``nflex = 1`` case of the same shape function, so
    its parameters are ``psb_flex``'s with the flexible-zone ratios removed and
    the fixed section (three equal-width old bins, two adjacent-step ratios)
    shared name for name.

    This pinned a seven-name set until #2184, ending in a ``ratio_old_2`` that
    the corrected three-bin fixed section has no step for: it sampled a prior,
    cost a dimension, and reached no bin.
    """
    assert list(SFH_REGISTRY["psb_suess2022"].params) == [
        "sfh_psb2022_log_total_mass",
        "sfh_psb2022_tlast_gyr",
        "sfh_psb2022_tflex_gyr",
        "sfh_psb2022_ratio_young",
        "sfh_psb2022_ratio_old_0",
        "sfh_psb2022_ratio_old_1",
    ]

    def _suffixes(prefix, name):
        return [p[len(prefix) :] for p in SFH_REGISTRY[name].params]

    flex = _suffixes("sfh_psb_flex_", "psb_flex")
    assert [s for s in flex if not s.startswith("ratio_flex_")] == _suffixes(
        "sfh_psb2022_", "psb_suess2022"
    )
