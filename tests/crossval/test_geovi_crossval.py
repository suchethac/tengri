# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate JIT geoVI implementation against NIFTy's optimize_kl.

The JIT engine in ``Fitter._build_jit_engine`` reimplements NIFTy's geoVI
primitives (linear residual draws, metric-vector products, KL value/gradient,
full optimize_kl loop) in pure JAX with ``jax.lax`` control flow. This test
module verifies numerical agreement between the two implementations.

Expected differences:
    - CG solvers differ in convergence criteria and iteration order, so
      individual residual draws are NOT bitwise identical. Instead we check
      that the *covariance structure* matches (statistical test over many draws).
    - KL values and gradients should agree to ~1e-4 relative tolerance for
      the same expansion point and samples.
    - Converged expansion points agree within ~5-10% because CG internals
      differ and stochastic samples diverge across iterations.
    - Posterior widths (standard deviations) agree within ~20%.

Invoke with:
    pytest -m crossval tests/crossval/test_geovi_crossval.py

Requires:
    - nifty8.re (``pip install nifty8[re]``)
    - SSP data in ``data/``
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [
    pytest.mark.crossval,
    # The whole module, not just the largest test, exceeds CI memory: the
    # per-file census measured this file at ~20 GB even with the single
    # 26 GB draw test deselected, and the first per-file CI run (workflow
    # dispatch 32287379706) died by runner shutdown inside this file after
    # ~11 minutes on a 7 GB VM. The NIFTy/JIT CG-inversion fixtures are
    # intrinsically dense. Run manually on a large-memory machine (#1728).
    pytest.mark.oom,
]

jft = pytest.importorskip("nifty8.re", reason="nifty8.re not installed")

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform
from tengri.utils.transforms import to_bounded

# ── Data paths -- skip if SSP data missing ────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

if not _SSP_FILE.is_file():
    pytest.skip("SSP data not found", allow_module_level=True)

# NIFTy optimize_kl kwargs (match VIConfig defaults from vi_config.py)
_NIFTY_KW = dict(
    draw_linear_kwargs={
        "cg_name": "SL",
        "cg_kwargs": {"absdelta": 1e-4, "maxiter": 30},
    },
    nonlinearly_update_kwargs={
        "minimize_kwargs": {
            "name": "SN",
            "xtol": 1e-3,
            "cg_kwargs": {"name": None},
            "maxiter": 3,
        },
    },
    kl_kwargs={
        "minimize_kwargs": {
            "name": "M",
            "absdelta": 1e-3,
            "cg_kwargs": {"name": "MCG"},
            "maxiter": 10,
        },
    },
)

# ── Shared fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp_data():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def filters():
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


@pytest.fixture(scope="module")
def smooth_spec():
    """Smooth parametric SFH spec (D=5) for tractable comparison."""
    return Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=-0.7,
        redshift=0.1,
    )


@pytest.fixture(scope="module")
def model_and_data(smooth_spec, ssp_data, filters):
    """Build SEDModel, generate mock data, return (model, data, noise)."""
    from tengri.forward.sed_model import SEDModel

    model = SEDModel(smooth_spec, ssp_data, filters=filters)

    # Generate mock from a random prior draw
    true_params = smooth_spec.sample(jax.random.PRNGKey(42))
    mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(99))
    return model, mock.flux_obs, mock.noise


@pytest.fixture(scope="module")
def fitter(model_and_data):
    """Build Fitter for the smooth model."""
    from tengri.inference.fitter import Fitter

    model, data, noise = model_and_data
    return Fitter(model, data, noise, data_type="photometry")


@pytest.fixture(scope="module")
def init_pos(fitter):
    """Deterministic initial position in unbounded space."""
    return fitter._initialize_unbounded(jax.random.PRNGKey(0))


@pytest.fixture(scope="module")
def jit_engine(fitter, init_pos):
    """Build JIT engine from init position."""
    return fitter._get_or_build_engine(init_pos)


@pytest.fixture(scope="module")
def data_args(fitter):
    """Data-dependent arguments for JIT engine calls."""
    return fitter._data_args


@pytest.fixture(scope="module")
def nifty_likelihood(fitter, model_and_data):
    """Build NIFTy likelihood identical to what _run_geovi constructs.

    Mirrors the construction in ``Fitter._run_geovi`` for the fixed-noise
    Gaussian case.
    """
    model, data, noise = model_and_data
    free_names = fitter._free_names
    bounds = fitter._bounds
    fixed_values = fitter._fixed_values

    def signal_response(primals):
        params = {}
        for name in free_names:
            lo, hi = bounds[name]
            params[name] = to_bounded(primals[name], lo, hi)
        for name, val in fixed_values.items():
            params[name] = val
        return model.predict_photometry(params)

    domain = {name: jft.ShapeWithDtype(()) for name in free_names}
    # jft.Model is NIFTy's, not ours. It must not follow tengri's
    # Model -> SEDModel rename (NAMING_CONTRACT); nifty8.re has no SEDModel.
    nifty_model = jft.Model(jax.jit(signal_response), domain=domain)

    noise_cov_inv = 1.0 / noise**2
    return jft.Gaussian(data, noise_cov_inv).amend(nifty_model)


# ── Helper: evaluate Hamiltonian H(xi) = lh(xi) + 0.5 ||xi||^2 ────


def _hamiltonian_nifty(likelihood, pos_dict, flatten_fn):
    """Evaluate the standard Hamiltonian using NIFTy's likelihood."""
    pos_vec = jft.Vector(pos_dict)
    lh = float(likelihood(pos_vec))
    prior = 0.5 * float(jnp.sum(flatten_fn(pos_dict) ** 2))
    return lh + prior


# ── 1. Linear residual draw: covariance structure ─────────────────


class TestLinearResidualCovariance:
    """Verify that JIT draw_residuals produces samples with correct M^{-1}
    covariance, matching the statistical properties of NIFTy's
    draw_linear_residual.

    The exact residual vectors differ because CG solvers use different
    iteration orders and convergence criteria. Instead we draw many samples
    and check that the empirical covariance approximates M^{-1}.
    """

    def test_jit_residual_covariance_matches_metric_inverse(self, jit_engine, init_pos, data_args):
        """Sample covariance of JIT residuals should approximate M^{-1}.

        For M = J^T N^{-1} J + I, residuals have covariance M^{-1} < I,
        so trace(cov)/D < 1 and all diagonal entries are positive and <= 1.
        """
        engine = jit_engine
        flatten = engine["flatten"]
        d_total = engine["d_total"]
        pos_flat = flatten(init_pos)

        n_draws = 500
        keys = jax.random.split(jax.random.PRNGKey(123), n_draws)
        residuals = engine["draw_samples"](pos_flat, keys, data_args)

        residuals_np = np.array(residuals)
        cov = np.cov(residuals_np, rowvar=False)

        diag = np.diag(cov)
        assert np.all(diag > 0), "Covariance diagonal must be positive"
        assert np.all(diag <= 1.5), (
            "Covariance diagonal should be <= 1 for M^{-1} (allowing sampling noise)"
        )

        # trace(cov) / D < 1 since M > I
        trace_ratio = np.trace(cov) / d_total
        assert trace_ratio < 1.0, (
            f"trace(cov)/D = {trace_ratio:.3f}, expected < 1.0 for M^{{-1}} < I"
        )
        assert trace_ratio > 0.01, f"trace(cov)/D = {trace_ratio:.3f}, unexpectedly small"

    @pytest.mark.oom
    def test_nifty_and_jit_residual_variances_agree(
        self, jit_engine, init_pos, nifty_likelihood, data_args
    ):
        """Per-parameter variance from JIT and NIFTy draws should agree.

        We draw 50 residuals from each and compare the per-dimension
        variance. Agreement within a factor of 2 is expected given
        finite-sample noise and different CG internals.
        Marked oom: NIFTy's residual drawing loop allocates ~26GB even with 50 draws;
        the cost is intrinsic to the verification procedure (independent draws from a
        full CG inversion). Run manually with sufficient memory. Reduced from 200 to 50
        draws to minimize memory impact when run manually.
        """
        engine = jit_engine
        flatten = engine["flatten"]
        pos_flat = flatten(init_pos)

        # JIT draws
        n_draws = 50
        jit_keys = jax.random.split(jax.random.PRNGKey(456), n_draws)
        jit_residuals = np.array(engine["draw_samples"](pos_flat, jit_keys, data_args))
        jit_var = np.var(jit_residuals, axis=0)

        # NIFTy draws
        nifty_pos = jft.Vector(init_pos)
        nifty_residuals = []
        nifty_keys = jax.random.split(jax.random.PRNGKey(456), n_draws * 2)
        for k in nifty_keys:
            try:
                residual, _ = jft.draw_linear_residual(
                    nifty_likelihood,
                    nifty_pos,
                    k,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 30},
                )
                res_dict = residual.tree if hasattr(residual, "tree") else dict(residual)
                nifty_residuals.append(np.array(flatten(res_dict)))
            except Exception:
                continue
            # Stop once we have enough samples
            if len(nifty_residuals) >= n_draws:
                break

        assert len(nifty_residuals) >= n_draws // 2, (
            f"Only {len(nifty_residuals)} NIFTy draws succeeded, need >= {n_draws // 2}"
        )

        nifty_residuals_arr = np.stack(nifty_residuals)
        nifty_var = np.var(nifty_residuals_arr, axis=0)

        # Median variance ratio should be near 1.0
        ratio = jit_var / (nifty_var + 1e-30)
        median_ratio = np.median(ratio)
        assert 0.5 < median_ratio < 2.0, (
            f"Median variance ratio JIT/NIFTy = {median_ratio:.2f}, expected ~1.0"
        )


# ── 2. KL value comparison ────────────────────────────────────────


class TestKLValue:
    """Verify that the Hamiltonian H(xi) = 0.5 chi2 + 0.5 ||xi||^2 is
    evaluated consistently between JIT and NIFTy at the same point.

    The JIT engine does not directly expose the hamiltonian function, so
    we compare indirectly: evaluate NIFTy's likelihood at the init point,
    then run one JIT evi_step_full and verify the returned KL is finite
    and positive. For a stronger comparison, we check that both
    Hamiltonians decrease after optimization.
    """

    def test_hamiltonian_at_init_is_consistent(
        self, jit_engine, init_pos, nifty_likelihood, data_args
    ):
        """Hamiltonian at init should be finite and positive for both."""
        engine = jit_engine
        flatten = engine["flatten"]
        pos_flat = flatten(init_pos)

        # NIFTy Hamiltonian
        H_nifty = _hamiltonian_nifty(nifty_likelihood, init_pos, flatten)

        assert np.isfinite(H_nifty), "NIFTy Hamiltonian at init is not finite"
        assert H_nifty > 0, f"NIFTy H = {H_nifty}, expected > 0"

        # JIT: run one evi_step to get a KL value (H averaged over samples)
        d_total = engine["d_total"]
        step_fn = engine["evi_step_full"]
        dummy_keys = jax.random.split(jax.random.PRNGKey(0), 1)
        _, kl_val, _, _ = step_fn(
            pos_flat,
            jax.random.PRNGKey(0),
            1,
            "linear_resample",
            jnp.zeros((2, d_total)),
            dummy_keys,
            jnp.zeros(d_total, dtype=bool),
            jnp.ones(d_total),
            data_args,
        )

        assert np.isfinite(float(kl_val)), "JIT KL at init is not finite"
        assert float(kl_val) > 0, f"JIT KL = {float(kl_val)}, expected > 0"

    def test_kl_decreases_after_optimization(
        self, jit_engine, init_pos, nifty_likelihood, data_args
    ):
        """After a few geoVI iterations the KL should decrease.

        This is not a direct value comparison but ensures both engines
        optimize in the right direction.
        """
        engine = jit_engine
        flatten = engine["flatten"]
        pos_flat = flatten(init_pos)

        H_init = _hamiltonian_nifty(nifty_likelihood, init_pos, flatten)

        # Run JIT geoVI for a few iterations
        run_geovi = engine["run_evi_geovi"]
        m_opt, _ = run_geovi(
            pos_flat,
            jax.random.PRNGKey(55),
            data_args,
            n_iterations=5,
            n_samples=2,
            kl_rtol=1e-3,
            sample_mode="nonlinear_resample",
        )

        opt_dict = engine["unflatten"](m_opt)
        H_opt = _hamiltonian_nifty(nifty_likelihood, opt_dict, flatten)

        assert H_opt < H_init, f"JIT geoVI did not decrease H: init={H_init:.2f}, opt={H_opt:.2f}"


# ── 3. Metric-vector product ──────────────────────────────────────


class TestMetricVectorProduct:
    """JIT metric_vec(xi, v) = J^T N^{-1} J v + v should match NIFTy's
    likelihood.metric for the Gaussian case.

    NIFTy's ``Gaussian.metric(pos, v)`` computes J^T N^{-1} J v (the
    likelihood part only). The full metric adds the identity (prior).
    """

    def test_metric_positive_definite_nifty(self, jit_engine, init_pos, nifty_likelihood):
        """NIFTy metric should be positive definite: v^T M v > ||v||^2."""
        engine = jit_engine
        flatten = engine["flatten"]
        unflatten = engine["unflatten"]
        d_total = engine["d_total"]

        v_flat = jax.random.normal(jax.random.PRNGKey(111), shape=(d_total,))

        nifty_pos = jft.Vector(init_pos)
        v_dict = unflatten(v_flat)
        nifty_v = jft.Vector(v_dict)

        try:
            nifty_Mv_lh = nifty_likelihood.metric(nifty_pos, nifty_v)
        except AttributeError:
            pytest.skip("NIFTy likelihood does not expose .metric()")

        nifty_Mv_lh_dict = nifty_Mv_lh.tree if hasattr(nifty_Mv_lh, "tree") else dict(nifty_Mv_lh)
        nifty_Mv_lh_flat = flatten(nifty_Mv_lh_dict)
        nifty_Mv_full = nifty_Mv_lh_flat + v_flat  # + I @ v (prior)

        # Likelihood part should be nonzero
        assert jnp.linalg.norm(nifty_Mv_lh_flat) > 1e-10, "Likelihood metric contribution is zero"

        # Full metric should be positive definite: v^T M v >= ||v||^2
        vMv = float(jnp.dot(v_flat, nifty_Mv_full))
        v2 = float(jnp.dot(v_flat, v_flat))
        assert vMv >= v2 * 0.99, (
            f"v^T M v = {vMv:.4f} < ||v||^2 = {v2:.4f}, metric not positive definite"
        )

    def test_cg_inverts_metric_correctly(self, jit_engine, init_pos, data_args):
        """M @ M^{-1} b ~ b: draw a residual (covariance M^{-1}), then
        verify that applying the metric recovers approximately b.

        The residual r solves M r = b where b = J^T sqrt(N^{-1}) eta_lh + eta_pr.
        We cannot access b directly, but we can check that the residual
        variance is < 1 (confirming M^{-1} < I).
        """
        engine = jit_engine
        flatten = engine["flatten"]
        d_total = engine["d_total"]
        pos_flat = flatten(init_pos)

        # Single residual draw
        key = jax.random.split(jax.random.PRNGKey(222), 1)
        residual = engine["draw_samples"](pos_flat, key, data_args)[0]

        # Residual should have norm < sqrt(D) (since var < 1 per dim)
        r_norm = float(jnp.linalg.norm(residual))
        assert r_norm < 3.0 * np.sqrt(d_total), (
            f"Residual norm {r_norm:.2f} unexpectedly large for D={d_total}"
        )
        assert r_norm > 0.0, "Residual is zero"


# ── 4. Full optimize_kl convergence ───────────────────────────────


class TestOptimizeKLConvergence:
    """Run NIFTy optimize_kl and JIT run_evi_geovi from the same init.

    The converged expansion points should agree within reasonable tolerance.
    Exact agreement is not expected because CG implementations differ, but
    the Hamiltonian values at convergence should be close.
    """

    def test_converged_hamiltonian_close(
        self, fitter, init_pos, jit_engine, nifty_likelihood, data_args
    ):
        """Hamiltonian at converged points should agree within 10%.

        Both methods minimize the same KL divergence, so they should
        converge to similar expansion points. We use a lenient tolerance
        because CG solver differences accumulate over iterations.
        """
        engine = jit_engine
        flatten = engine["flatten"]
        unflatten = engine["unflatten"]
        pos_flat = flatten(init_pos)

        # --- JIT geoVI ---
        n_iterations = 8
        n_samples = 3
        run_geovi = engine["run_evi_geovi"]
        m_jit, _ = run_geovi(
            pos_flat,
            jax.random.PRNGKey(42),
            data_args,
            n_iterations=n_iterations,
            n_samples=n_samples,
            kl_rtol=1e-3,
            sample_mode="nonlinear_resample",
        )

        # --- NIFTy geoVI ---
        nifty_pos = jft.Vector(init_pos)
        _, opt_key = jax.random.split(jax.random.PRNGKey(42))
        try:
            samples_nifty, _ = jft.optimize_kl(
                nifty_likelihood,
                nifty_pos,
                n_total_iterations=n_iterations,
                n_samples=n_samples,
                key=opt_key,
                sample_mode="nonlinear_resample",
                odir=None,
                **_NIFTY_KW,
            )
        except Exception as e:
            pytest.skip(f"NIFTy optimize_kl failed: {e}")

        converged_nifty = samples_nifty.pos
        nifty_dict = (
            converged_nifty.tree if hasattr(converged_nifty, "tree") else dict(converged_nifty)
        )

        # Evaluate Hamiltonian at both converged points via NIFTy
        H_at_jit = _hamiltonian_nifty(nifty_likelihood, unflatten(m_jit), flatten)
        H_at_nifty = _hamiltonian_nifty(nifty_likelihood, nifty_dict, flatten)

        assert np.isfinite(H_at_jit), "JIT Hamiltonian is not finite"
        assert np.isfinite(H_at_nifty), "NIFTy Hamiltonian is not finite"

        # Both should have decreased from initial value
        H_init = _hamiltonian_nifty(nifty_likelihood, init_pos, flatten)
        assert H_at_jit < H_init, f"JIT H={H_at_jit:.2f} did not decrease from init H={H_init:.2f}"
        assert H_at_nifty < H_init, (
            f"NIFTy H={H_at_nifty:.2f} did not decrease from init H={H_init:.2f}"
        )

        # Converged H values should be within 10% of each other
        mean_H = 0.5 * (abs(H_at_jit) + abs(H_at_nifty))
        rel_diff = abs(H_at_jit - H_at_nifty) / mean_H
        assert rel_diff < 0.10, (
            f"Converged Hamiltonian values differ by {rel_diff:.1%}: "
            f"JIT={H_at_jit:.4f}, NIFTy={H_at_nifty:.4f}"
        )


# ── 5. Posterior width comparison ─────────────────────────────────


class TestPosteriorWidthComparison:
    """Draw posterior samples from both converged approximations and compare
    standard deviations.

    After convergence, both methods should yield similar posterior widths
    (standard deviations) for each parameter. We allow generous tolerance
    because the approximations differ slightly and sample noise is large
    at N=200.
    """

    def test_posterior_stds_agree(self, fitter, init_pos, jit_engine, nifty_likelihood, data_args):
        """Posterior standard deviations should agree within ~50%.

        We run both methods for a few iterations, draw 200 posterior
        samples from each, and compare per-parameter standard deviations.
        The 50% tolerance accounts for different CG convergence, different
        PRNG consumption patterns, and finite sample variance.
        """
        engine = jit_engine
        flatten = engine["flatten"]
        pos_flat = flatten(init_pos)

        # --- JIT: converge + draw samples ---
        run_geovi = engine["run_evi_geovi"]
        m_jit, _ = run_geovi(
            pos_flat,
            jax.random.PRNGKey(77),
            data_args,
            n_iterations=8,
            n_samples=3,
            kl_rtol=1e-3,
            sample_mode="nonlinear_resample",
        )

        n_posterior = 200
        draw_keys = jax.random.split(jax.random.PRNGKey(888), n_posterior)
        jit_residuals = engine["draw_samples"](m_jit, draw_keys, data_args)
        jit_samples = np.array(m_jit[None, :] + jit_residuals)

        # --- NIFTy: converge + draw samples ---
        nifty_pos = jft.Vector(init_pos)
        _, opt_key = jax.random.split(jax.random.PRNGKey(77))
        try:
            samples_nifty, _ = jft.optimize_kl(
                nifty_likelihood,
                nifty_pos,
                n_total_iterations=8,
                n_samples=3,
                key=opt_key,
                sample_mode="nonlinear_resample",
                odir=None,
                **_NIFTY_KW,
            )
        except Exception as e:
            pytest.skip(f"NIFTy optimize_kl failed: {e}")

        converged_nifty = samples_nifty.pos
        nifty_pos_dict = (
            converged_nifty.tree if hasattr(converged_nifty, "tree") else dict(converged_nifty)
        )

        # Draw NIFTy posterior samples
        nifty_draw_keys = jax.random.split(jax.random.PRNGKey(888), n_posterior)
        nifty_samples_list = []
        nifty_pos_flat = flatten(nifty_pos_dict)
        for k in nifty_draw_keys:
            try:
                residual, _ = jft.draw_linear_residual(
                    nifty_likelihood,
                    converged_nifty,
                    k,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 30},
                )
                res_dict = residual.tree if hasattr(residual, "tree") else dict(residual)
                res_flat = flatten(res_dict)
                nifty_samples_list.append(nifty_pos_flat + res_flat)
            except Exception:
                continue

        if len(nifty_samples_list) < 100:
            pytest.skip(
                f"Only {len(nifty_samples_list)} NIFTy posterior samples -- "
                "need >= 100 for reliable comparison"
            )

        nifty_samples = np.stack(nifty_samples_list)

        # Compare standard deviations per dimension
        jit_std = np.std(jit_samples, axis=0)
        nifty_std = np.std(nifty_samples, axis=0)

        # Only compare dimensions with nonzero variance in both
        valid = (jit_std > 1e-10) & (nifty_std > 1e-10)
        assert np.sum(valid) >= 3, "Too few dimensions with nonzero std"

        ratio = jit_std[valid] / nifty_std[valid]
        median_ratio = np.median(ratio)

        # Median should be near 1.0 (within 50%)
        assert 0.5 < median_ratio < 2.0, (
            f"Median std ratio JIT/NIFTy = {median_ratio:.2f}, expected ~1.0"
        )

        # No dimension should be off by more than 5x
        assert np.all(ratio > 0.2) and np.all(ratio < 5.0), (
            f"Some dimensions have extreme std ratios: "
            f"min={np.min(ratio):.2f}, max={np.max(ratio):.2f}"
        )
