# SPDX-License-Identifier: BSD-3-Clause
r"""The geoVI curvature primitives must survive pure float32 (#1588; #1206, after #1535).

#1535 made the Gaussian *energy* float32-safe by standardizing the residual
before squaring. It did not touch the *metric*. At a real photometric
:math:`\sigma \sim 3\times10^{-30}` the two quantities the engine derives from
the noise sit on opposite sides of the float32 ceiling (3.4e38):

===========================  ==========  ==================================
quantity                     magnitude   float32
===========================  ==========  ==================================
:math:`1/\sigma`             3.3e29      representable
:math:`1/\sigma^2`           1.1e59      **inf**
===========================  ==========  ==================================

``data_args["sqrt_noise_inv"]`` wants the first and was spelled
``jnp.sqrt(1.0 / noise**2)`` — routing a representable destination through an
unrepresentable intermediate, so it arrived as ``sqrt(inf) = inf``. Every
geoVI/MGVI sqrt-metric primitive reads it: ``transformation_flat``,
``left_sqrt_metric_flat``, ``right_sqrt_metric_flat``, ``draw_residuals``,
``draw_metric_sample``.

``metric_vec`` is a **second, independent** defect needing a **different**
remedy: it reads ``noise_inv`` directly, and :math:`1/\sigma^2` is genuinely
outside float32 however it is spelled. It has to be restructured as
:math:`(J/\sigma)^\mathsf{T}(J/\sigma)` — the same measure already proven in
``analysis/diagnostics/fisher.py`` (#1542). Fixing only the spelling leaves
``metric_vec`` NaN; that is asserted below, so the two fixes cannot be
conflated.

Why this was invisible: the Hamiltonian stays finite because
``standardized_residual`` protects it. geoVI therefore reports a healthy
energy while ``metric_vec`` — the operator CG inverts to draw posterior
samples — returns NaN. A converging fit with NaN posterior draws, and nothing
in the output connecting the two.
"""

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

#: A photometric uncertainty at the magnitude a real fit sees [erg/s/cm^2/Hz].
_SIGMA = 3.0e-30
_FLUX = 1.9e-28

_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src" / "tengri"

#: Roots the class guard sweeps. `src/` is the code; `examples/` and
#: `docs/dev/` are the surfaces that *teach* a spelling, and a doc showing the
#: removed form is how it comes back — `docs/dev/inference_methods.md` still
#: printed `vjp_fn(noise_inv * Jv)` after the code stopped doing it.
#:
#: Deliberately NOT swept: `tests/` (several legitimately build a dense
#: reference covariance, `jnp.diag(1.0/err**2)`, or freeze the pre-fix
#: arithmetic as a comparison arm — that is their job), `docs/auto_examples/`
#: (generated from `examples/`), and `docs/dev/archive/` (historical record).
_SWEPT_ROOTS = (
    _REPO / "src" / "tengri",
    _REPO / "examples",
    _REPO / "docs" / "dev",
)
_SWEPT_SKIP = ("archive/",)


def _f32(fn):
    """Evaluate ``fn`` under genuine float32 (no x64 upcast)."""
    with jax.enable_x64(False):
        return fn()


def _f64(fn):
    """Evaluate ``fn`` under float64."""
    with jax.enable_x64(True):
        return fn()


# --------------------------------------------------------------------------
# 1. The primitive
# --------------------------------------------------------------------------


def test_inverse_variance_really_is_unrepresentable_in_float32():
    """Precondition: 1/sigma**2 overflows and 1/sigma does not.

    Without this the rest of the module proves nothing — it is the reason the
    two halves of the defect need two different remedies.
    """

    def run():
        noise = jnp.full((4,), _SIGMA)
        return jnp.asarray([jnp.max(1.0 / noise**2), jnp.max(1.0 / noise)])

    got = np.asarray(_f32(run), dtype=np.float64)
    assert np.isinf(got[0]), (
        f"1/sigma**2 = {got[0]:.3e} is finite in float32 — the premise of this "
        "module is wrong and the metric restructure may be unnecessary"
    )
    assert np.isfinite(got[1]) and got[1] > 0.0, (
        f"1/sigma = {got[1]:.3e} is NOT representable in float32 — then no "
        "spelling saves the sqrt-metric primitives and a deeper fix is needed"
    )


def test_inv_noise_std_is_finite_in_float32():
    """``inv_noise_std`` must reach 1/sigma without an inf intermediate."""
    from tengri.inference.likelihoods.gaussian import inv_noise_std

    def run():
        return inv_noise_std(jnp.full((4,), _SIGMA))

    f32 = np.asarray(_f32(run), dtype=np.float64)
    f64 = np.asarray(_f64(run), dtype=np.float64)
    assert np.all(np.isfinite(f32)), (
        f"inv_noise_std is {f32[0]:.3e} in float32 — it was computed as "
        "sqrt(1/sigma**2), whose intermediate is inf"
    )
    np.testing.assert_allclose(f32, f64, rtol=1e-6)


def test_whiten_matches_manual_division_in_float64():
    """The barrier must be semantically the identity (bit-exact in float64)."""
    from tengri.inference.likelihoods.gaussian import whiten

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(16) * _FLUX)
    s = jnp.full((16,), _SIGMA)

    got = np.asarray(_f64(lambda: whiten(x, s)), dtype=np.float64)
    want = np.asarray(_f64(lambda: x / s), dtype=np.float64)
    np.testing.assert_array_equal(got, want)


# --------------------------------------------------------------------------
# 2. The metric itself — the restructure, measured against the old spelling
# --------------------------------------------------------------------------


def _linear_response(rng):
    """A linear signal response at realistic flux magnitudes.

    Row 3 is exactly zero so that ``0 * inf = NaN`` is reachable — an
    all-nonzero Jacobian would report ``inf`` and hide the NaN.
    """
    a = rng.standard_normal((12, 5)) * _FLUX
    a[3, :] = 0.0
    return a


def _metric_old(a, v, noise):
    """``J^T N^{-1} J v + v`` as shipped before this fix."""
    noise_inv = 1.0 / noise**2
    return a.T @ (noise_inv * (a @ v)) + v


def _metric_new(a, v, noise):
    """``(J/sigma)^T (J/sigma) v + v`` — the restructure under test."""
    from tengri.inference.likelihoods.gaussian import whiten

    return a.T @ whiten(whiten(a @ v, noise), noise) + v


def test_metric_vec_restructure_preserves_float64():
    """The restructure must not move the float64 answer (bar: rtol 1e-12)."""
    rng = np.random.default_rng(0)
    a = jnp.asarray(_linear_response(rng))
    v = jnp.asarray(rng.standard_normal(5))
    noise = jnp.full((12,), _SIGMA)

    old = np.asarray(_f64(lambda: _metric_old(a, v, noise)), dtype=np.float64)
    new = np.asarray(_f64(lambda: _metric_new(a, v, noise)), dtype=np.float64)
    np.testing.assert_allclose(new, old, rtol=1e-12)


def test_metric_vec_is_finite_in_float32():
    """The restructured metric must be finite where the old spelling is NaN."""
    rng = np.random.default_rng(0)
    a = jnp.asarray(_linear_response(rng))
    v = jnp.asarray(rng.standard_normal(5))
    noise = jnp.full((12,), _SIGMA)

    old = np.asarray(_f32(lambda: _metric_old(a, v, noise)), dtype=np.float64)
    new = np.asarray(_f32(lambda: _metric_new(a, v, noise)), dtype=np.float64)
    f64 = np.asarray(_f64(lambda: _metric_new(a, v, noise)), dtype=np.float64)

    assert not np.all(np.isfinite(old)), (
        "the OLD metric spelling is finite in float32 — the arm this test "
        "compares against no longer fails, so a pass proves nothing"
    )
    assert np.all(np.isfinite(new)), (
        f"restructured metric_vec still has {np.sum(~np.isfinite(new))} non-finite "
        "entries in float32"
    )
    np.testing.assert_allclose(new, f64, rtol=2e-5)


def test_metric_vec_survives_jit_constant_folding():
    """Under jit with constant noise, XLA must not re-associate back to 1/sigma**2.

    This is the #1535 lesson: source-level grouping is a suggestion, a data
    dependency is binding. Without the barrier XLA folds
    ``(x/s)/s`` into ``x * (1/s**2)`` and the constant becomes inf.
    """
    rng = np.random.default_rng(0)
    a = jnp.asarray(_linear_response(rng))
    v = jnp.asarray(rng.standard_normal(5))

    def run():
        # noise as a compile-time constant is what enables the folding
        noise = jnp.full((12,), _SIGMA)
        return jax.jit(lambda vv: _metric_new(a, vv, noise))(v)

    got = np.asarray(_f32(run), dtype=np.float64)
    assert np.all(np.isfinite(got)), (
        "metric_vec is non-finite under jit but finite eagerly — XLA "
        "re-associated the double division into 1/sigma**2"
    )


# --------------------------------------------------------------------------
# 3. The real shipped closures, not a replica of their arithmetic
# --------------------------------------------------------------------------

#: Pinned data/noise at real AB magnitudes, identical in both precisions.
#: Deriving them from each dtype's own forward pass gives the two arms
#: different data (measured: f32 truth 2.7e-32 vs f64 1.9e-28 on this SSP) and
#: makes any cross-precision statement meaningless.
_DATA = (1.9e-28, 2.4e-28, 3.1e-28, 2.0e-28, 1.5e-28)


def _build_engine(ssp_wide, obs):
    """Return ``(engine, data_args, flat_position)`` for a real Fitter."""
    from tengri import FIXED, FREE, SEDModel
    from tengri.components.stellar.sps.dsps_wrapper import SSPData
    from tengri.inference.fitter import Fitter
    from tengri.inference.jit_engine import build_jit_engine

    # Rescale so predicted photometry lands at real AB magnitudes. The bare
    # synthetic SSP predicts ~1e-9, for which 1/sigma**2 ~ 1e20 is comfortably
    # representable — a test on it would pass on a build that still has the bug.
    ssp = SSPData(
        ssp_wave=ssp_wide.ssp_wave,
        ssp_flux=ssp_wide.ssp_flux * 1e-19,
        ssp_lg_age_gyr=ssp_wide.ssp_lg_age_gyr,
        ssp_lgmet=ssp_wide.ssp_lgmet,
    )
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust={"type": "two_component", "all_params": FIXED},
        redshift=0.05,
        approx=None,
    )
    n_band = model.observation.n_data
    assert n_band <= len(_DATA), f"fixture has {n_band} bands; extend _DATA"
    truth = np.asarray(_DATA[:n_band])
    fitter = Fitter(model, jnp.asarray(truth), jnp.asarray(truth * 0.05))
    pos = fitter._initialize_unbounded(jax.random.PRNGKey(1))
    engine = build_jit_engine(fitter, pos)
    return engine, fitter._data_args, engine["flatten"](pos)


def test_real_geovi_closures_are_finite_in_float32(synthetic_ssp_wide, synthetic_tophat_obs):
    """Every geoVI primitive the engine actually ships must survive float32.

    The arithmetic tests above use a replica; this one calls the closures
    ``build_jit_engine`` returns, through a real ``Fitter``, so a fix that lands
    in the replica but misses the shipped path cannot pass.
    """
    with jax.enable_x64(False):
        engine, data_args, pos_f = _build_engine(synthetic_ssp_wide, synthetic_tophat_obs)
        v = jnp.asarray(np.random.default_rng(0).standard_normal(pos_f.shape[0]) * 0.1)
        v_data = jnp.ones_like(data_args["data"])

        results = {
            "sqrt_noise_inv": data_args["sqrt_noise_inv"],
            "transformation_flat": engine["transformation_flat"](pos_f, data_args),
            "left_sqrt_metric_flat": engine["left_sqrt_metric_flat"](pos_f, v_data, data_args),
            "right_sqrt_metric_flat": engine["right_sqrt_metric_flat"](pos_f, v, data_args),
            "metric_vec": engine["metric_vec"](pos_f, v, data_args),
            "hamiltonian": engine["hamiltonian"](pos_f, data_args),
        }
        broken = {
            k: int(np.sum(~np.isfinite(np.asarray(x, dtype=np.float64))))
            for k, x in results.items()
        }

    sigma = float(np.min(np.asarray(data_args["noise"], dtype=np.float64)))
    assert 1.0 / sigma**2 > np.finfo(np.float32).max, (
        f"sigma={sigma:.3e} gives a representable 1/sigma**2 — this fixture cannot "
        "exercise the defect; rescale the SSP"
    )
    assert not any(broken.values()), (
        f"non-finite entries in the shipped geoVI primitives: "
        f"{ {k: n for k, n in broken.items() if n} }"
    )


def test_noise_inv_is_not_published_in_data_args(synthetic_ssp_wide, synthetic_tophat_obs):
    """``data_args`` must not carry 1/sigma**2 — it is ``inf`` in float32.

    Nothing reads it once ``metric_vec`` whitens twice, and publishing an
    all-``inf`` array invites the next backend author to use it.
    """
    with jax.enable_x64(False):
        _engine, data_args, _pos = _build_engine(synthetic_ssp_wide, synthetic_tophat_obs)
        keys = set(data_args)
    assert "noise_inv" not in keys, (
        "data_args still publishes 'noise_inv' (1/sigma**2 = inf in float32); "
        "backends should apply sqrt_noise_inv twice instead"
    )
    assert "sqrt_noise_inv" in keys, "sqrt_noise_inv must remain — backends read it"


# --------------------------------------------------------------------------
# 4. Defect B — analytic emission-line marginalization
# --------------------------------------------------------------------------


def _eline_inputs():
    """Residual, noise and design matrix at real spectroscopic magnitudes."""
    rng = np.random.default_rng(1)
    n_pix, n_lines = 64, 3
    wave = np.linspace(6500.0, 6620.0, n_pix)
    design = np.stack(
        [np.exp(-0.5 * ((wave - c) / 3.0) ** 2) for c in (6548.0, 6564.6, 6584.0)],
        axis=1,
    )
    residual = _FLUX * (design @ np.array([1.0, 3.0, 1.5]) + 0.05 * rng.standard_normal(n_pix))
    return (
        jnp.asarray(residual),
        jnp.full((n_pix,), _SIGMA),
        jnp.asarray(design * _FLUX),
    )


def test_marginalize_emission_lines_is_finite_in_float32():
    """All three outputs must be finite at a real spectroscopic sigma."""
    from tengri.observation.eline_marginalization import marginalize_emission_lines

    residual, noise, design = _eline_inputs()

    def run():
        ln_l, a_hat, a_cov = marginalize_emission_lines(residual, noise, design)
        return jnp.concatenate([jnp.atleast_1d(ln_l), a_hat, a_cov.ravel()])

    got = np.asarray(_f32(run), dtype=np.float64)
    assert np.all(np.isfinite(got)), (
        f"{np.sum(~np.isfinite(got))}/{got.size} outputs of "
        "marginalize_emission_lines are non-finite in float32 — n_inv = 1/sigma**2 "
        "is inf, so G^T N^-1 G is inf/NaN and the whole solve collapses"
    )


def test_marginalize_emission_lines_gradient_is_finite_in_float32():
    """Its docstring promises gradient-safety; hold it to that in float32."""
    from tengri.observation.eline_marginalization import marginalize_emission_lines

    residual, noise, design = _eline_inputs()

    def run():
        return jax.grad(lambda r: marginalize_emission_lines(r, noise, design)[0])(residual)

    got = np.asarray(_f32(run), dtype=np.float64)
    assert np.all(np.isfinite(got)), (
        "d ln_L_marg / d residual is non-finite in float32, contradicting the "
        "documented 'Gradient-safe: yes'"
    )


def test_marginalize_emission_lines_preserves_float64():
    """Whitening must not move the float64 answer (bar: rtol 1e-10)."""
    from tengri.observation.eline_marginalization import marginalize_emission_lines

    residual, noise, design = _eline_inputs()

    def run_new():
        ln_l, a_hat, _ = marginalize_emission_lines(residual, noise, design)
        return jnp.concatenate([jnp.atleast_1d(ln_l), a_hat])

    def run_old():
        # the pre-fix arithmetic, frozen here so the comparison is independent
        n_inv = 1.0 / noise**2
        gw = design * n_inv[:, None]
        gtg = gw.T @ design
        gtr = gw.T @ residual
        pv = jnp.full((design.shape[1],), 1e10)
        a_cov = jnp.linalg.inv(gtg + jnp.diag(1.0 / pv))
        a_hat = a_cov @ gtr
        chi2 = jnp.sum(residual**2 * n_inv) - a_hat @ gtg @ a_hat
        _s, logdet = jnp.linalg.slogdet(a_cov)
        ln_l = -0.5 * chi2 - 0.5 * jnp.sum(a_hat**2 / pv) + 0.5 * (logdet - jnp.sum(jnp.log(pv)))
        return jnp.concatenate([jnp.atleast_1d(ln_l), a_hat])

    new = np.asarray(_f64(run_new), dtype=np.float64)
    old = np.asarray(_f64(run_old), dtype=np.float64)
    np.testing.assert_allclose(new, old, rtol=1e-10)


# --------------------------------------------------------------------------
# 5. The class guard — no site may reintroduce the spelling
# --------------------------------------------------------------------------

#: ``1.0 / <anything> ** 2`` in any spacing. Deliberately name-blind: an
#: earlier version keyed on the variable containing "noise"/"sigma", and missed
#: ``1.0 / n**2`` in ``hierarchical.py`` — a guard that cannot see the site it
#: exists for. Everything it catches that is *not* an inverse variance goes in
#: ``_ALLOWED`` with a stated reason, so a new site forces a decision.
_INV_VARIANCE = re.compile(r"1(?:\.0)?\s*/\s*[\w\.\[\]\(\)]+\s*\*\*\s*2")

#: ``sqrt(<anything>_inv)`` — the representable destination via an inf route.
_SQRT_OF_INV_VARIANCE = re.compile(r"sqrt\(\s*[A-Za-z_][\w\.\[\]]*_inv\w*\s*\)", re.IGNORECASE)

#: reST inline literals — prose *quoting* the banned spelling to explain it is
#: documentation, not an instance of it. Stripped before matching so the
#: allowlist below holds only genuine code decisions.
#: Matches both reST ``double`` and Markdown `single` backtick spans — the
#: sweep now covers .md, where prose warning *against* a spelling quotes it.
_RST_LITERAL = re.compile(r"`+[^`]*`+")

#: ``(file, exact source text)`` pairs exempt from :data:`_INV_VARIANCE`, each
#: with the reason it is not an inverse variance. Editing an exempt line
#: re-arms the guard on it, which is the intended behavior.
_ALLOWED = {
    # Not a variance: 1/lambda**2 is the ENERGY filter convention.
    ("utils/conversions.py", "inv_lambda_sq = 1.0 / (wavelength_aa**2)"),
    # Not a variance: dust emission line-profile denominator, O(1) quantities.
    (
        "components/dust/emission/analytic/_closures.py",
        "denominator = (ratio - 1.0 / ratio) ** 2 + gamma**2",
    ),
    # Not a variance: Lyman-series index n, an integer >= 2.
    (
        "components/igm/meiksin06.py",
        "lambda_n = lambda_limit / (1.0 - 1.0 / (n_idx.astype(jnp.float64) ** 2))",
    ),
    # KNOWN OPEN, deliberately not fixed here (reported separately):
    # tau = 1/sigma_eff ~ 3e29, so tau**2 ~ 1e59 overflows and 1/tau**2
    # underflows to 0 in float32. The variable-noise Hessian needs the whole
    # metric rescaled, not a spelling change -- a design task, not this fix.
    ("observation/noise.py", "H_tt = residual**2 + 1.0 / tau**2"),
    # (The calibration `inv_var = 1.0/max(obs_err**2, 1e-30)` entry used to sit
    # here, filed as a Tier-B float32 known-open. That was wrong: the floor
    # bound in float64 too and collapsed the polynomial. Fixed, not exempted —
    # see tests/regression/bug/test_calibration_variance_floor.py.)
    # Calibration-parameter prior width, an O(0.1) dimensionless quantity.
    ("observation/calibration.py", "prior_precision = 1.0 / (prior_sigma**2)"),
}


def _scan(pattern, allowed=frozenset(), roots=None, suffixes=(".py",)):
    """Return ``[(relpath, lineno, text)]`` for every match under ``roots``."""
    hits = []
    for root in roots or (_SRC,):
        if not root.exists():
            continue
        for suffix in suffixes:
            for path in sorted(root.rglob(f"*{suffix}")):
                rel = path.relative_to(_REPO).as_posix()
                if any(skip in rel for skip in _SWEPT_SKIP):
                    continue
                short = path.relative_to(root).as_posix()
                for i, line in enumerate(path.read_text().splitlines(), start=1):
                    stripped = line.strip()
                    if (short, stripped) in allowed or (rel, stripped) in allowed:
                        continue
                    code = _RST_LITERAL.sub("", line.split("#", 1)[0])
                    if pattern.search(code):
                        hits.append((rel, i, stripped))
    return hits


def test_no_doc_or_example_teaches_the_removed_metric_spelling():
    """A doc that prints ``noise_inv * Jv`` is how the form comes back.

    The code guard only ever scanned ``src/``, so
    ``docs/dev/inference_methods.md`` went on teaching the exact expression the
    fix removed, and a published gallery example hand-rolled it too.
    """
    hits = _scan(
        re.compile(r"noise_inv\s*\*"),
        roots=(_REPO / "examples", _REPO / "docs" / "dev"),
        suffixes=(".py", ".md"),
    )
    assert not hits, (
        "these TEACH the removed spelling; whiten twice instead "
        "(tengri.utils.scale.whiten):\n" + "\n".join(f"  {p}:{n}: {t}" for p, n, t in hits)
    )


def test_inverse_variance_allowlist_has_no_stale_entries():
    """Every allowlisted line must still exist verbatim.

    Without this the allowlist silently grows exemptions for code that has
    moved on, and the guard weakens without anyone deciding to weaken it.
    """
    stale = []
    for rel, text in sorted(_ALLOWED):
        path = _SRC / rel
        if not path.is_file() or text not in path.read_text():
            stale.append(f"{rel}: {text}")
    assert not stale, "allowlist entries no longer present in src — remove them:\n" + "\n".join(
        stale
    )


def test_no_inverse_variance_construction_in_src():
    """No live site may build 1/sigma**2 — it is inf at real photometric sigma."""
    hits = _scan(_INV_VARIANCE, allowed=_ALLOWED, roots=_SWEPT_ROOTS)
    assert not hits, "1/sigma**2 is inf in float32; whiten twice instead:\n" + "\n".join(
        f"  {p}:{n}: {t}" for p, n, t in hits
    )


def test_no_sqrt_of_inverse_variance_in_src():
    """No site may reach 1/sigma by square-rooting the overflowing 1/sigma**2."""
    hits = _scan(_SQRT_OF_INV_VARIANCE)
    assert not hits, "use inv_noise_std(noise) = 1/sigma directly:\n" + "\n".join(
        f"  {p}:{n}: {t}" for p, n, t in hits
    )
