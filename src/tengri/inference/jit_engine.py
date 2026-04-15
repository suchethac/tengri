"""JIT engine builder extracted from Fitter._build_jit_engine.

Module-level function that accepts a Fitter instance and a position dict,
returning the compiled-function dict used by all geoVI/MGVI/EVI inference
paths.  Extracted here to keep fitter.py under the 800-line project limit
and to make the JIT-compilation logic independently readable.

The geoVI path is an exact JAX port of NIFTy's CG, Newton-CG, sample
drawing, and nonlinear curving algorithms.  Mathematical equivalence with
``jft.optimize_kl`` is verified by the cross-validation tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.utils.transforms import to_bounded


def build_jit_engine(fitter, pos_dict):
    """Build JIT-compiled inference engine: optimizer + posterior sampler.

    Returns a dict with compiled functions for the full EVI pipeline.
    All functions operate on flat arrays and use jax.lax.while_loop
    for zero Python overhead.

    The geoVI path uses NIFTy's actual implementations of CG,
    Newton-CG, sample drawing, and nonlinear curving — imported
    directly and called within the JIT boundary. This ensures
    mathematical equivalence with ``jft.optimize_kl``.

    Parameters
    ----------
    fitter : Fitter
        Configured Fitter instance (read-only; only attributes are accessed).
    pos_dict : dict
        Position dict mapping parameter names to initial JAX arrays.
        Used only to compute static shapes for flatten/unflatten.

    Returns
    -------
    dict
        Compiled functions: run_evi, run_evi_geovi, run_nifty_jit,
        draw_samples, draw_nonlinear_samples, flatten, unflatten, etc.
    """
    from tengri.observation.noise import (
        compute_std_inv,
        get_noise_dof,
        has_noise_model,
        uses_student_t,
        variable_noise_hamiltonian,
        variable_noise_metric_vec,
    )

    # Import NIFTy for the exact geoVI path
    try:
        from nifty8.re.evi import Samples as NiftySamples
        from nifty8.re.optimize_kl import OptimizeVI

        _has_nifty = True
    except ImportError:
        _has_nifty = False

    model = fitter.model
    data_type = fitter.data_type
    free_names = fitter._free_names
    bounds = fitter._bounds
    fixed_values = fitter._fixed_values
    stochastic = fitter.spec.stochastic
    # data/noise are NO LONGER captured here as local variables.
    # Instead they are passed at call-time via the ``data_args`` dict
    # so that the compiled engine can be reused across galaxies.
    use_variable_noise = has_noise_model(fitter.spec)
    noise_dof = get_noise_dof(fitter.spec) if uses_student_t(fitter.spec) else None

    # --- Signal response (physics only) ---
    # NOT JIT'd — must remain traceable so that jax.jvp/vjp (in metric_vec)
    # and jax.value_and_grad (in hamiltonian) can differentiate through it.
    def signal_response(primals):
        params = {}
        for name in free_names:
            lo, hi = bounds[name]
            params[name] = to_bounded(primals[name], lo, hi)
        for name, val in fixed_values.items():
            params[name] = val
        if stochastic and "psd_xi" in primals:
            params["psd_xi"] = primals["psd_xi"]
        if data_type == "photometry":
            return model.predict_photometry(params, mode="_traceable")
        elif data_type == "spectroscopy":
            return model.predict_spectrum(params, model._wave_obs, mode="_traceable")
        elif data_type == "joint":
            p = model.predict_photometry(params, mode="_traceable")
            s = model.predict_spectrum(params, model._wave_obs, mode="_traceable")
            return jnp.concatenate([p, s])
        raise ValueError(f"Unknown data_type: {data_type}")

    # --- Signal + noise response for variable noise ---
    if use_variable_noise:

        def signal_noise_response(primals, data_args):
            """Return (predicted, std_inv) tuple for variable noise metric."""
            params = {}
            for name in free_names:
                lo, hi = bounds[name]
                params[name] = to_bounded(primals[name], lo, hi)
            for name, val in fixed_values.items():
                params[name] = val
            if stochastic and "psd_xi" in primals:
                params["psd_xi"] = primals["psd_xi"]
            if data_type == "photometry":
                predicted = model.predict_photometry(params, mode="_traceable")
            elif data_type == "spectroscopy":
                predicted = model.predict_spectrum(params, model._wave_obs, mode="_traceable")
            elif data_type == "joint":
                p = model.predict_photometry(params, mode="_traceable")
                s = model.predict_spectrum(params, model._wave_obs, mode="_traceable")
                predicted = jnp.concatenate([p, s])
            else:
                raise ValueError(f"Unknown data_type: {data_type}")
            f_cal = params.get("noise_frac_cal", 0.0)
            noise = data_args["noise"]
            std_inv = compute_std_inv(noise, predicted, f_cal)
            return predicted, std_inv

    # --- Flatten/unflatten (static shapes) ---
    param_keys = sorted(pos_dict.keys())
    slices = []
    idx = 0
    for k in param_keys:
        arr = jnp.atleast_1d(pos_dict[k]).ravel()
        shape = jnp.atleast_1d(pos_dict[k]).shape
        slices.append((idx, idx + arr.shape[0], shape))
        idx += arr.shape[0]
    d_total = idx
    n_data = len(fitter.data)  # static shape — same for all galaxies with same obs

    def flatten(d):
        return jnp.concatenate([jnp.atleast_1d(d[k]).ravel() for k in param_keys])

    def unflatten(x):
        d = {}
        for i_k, k in enumerate(param_keys):
            start, end, shape = slices[i_k]
            val = jax.lax.dynamic_slice(x, (start,), (end - start,)).reshape(shape)
            if shape == (1,):
                val = val[0]
            d[k] = val
        return d

    # --- Core primitives ---
    _eps = 6.0 * jnp.finfo(jnp.float64).eps

    if use_variable_noise:

        def metric_vec(xi, v, data_args):
            """GGN metric for VariableCovarianceGaussian likelihood."""
            data = data_args["data"]

            def _snr(primals):
                return signal_noise_response(primals, data_args)

            return variable_noise_metric_vec(xi, v, _snr, data, unflatten, flatten)

        def hamiltonian(xi, data_args):
            """E_lh + 0.5 ||xi||^2 with variable noise (includes logdet)."""
            data = data_args["data"]
            noise = data_args["noise"]
            pred = signal_response(unflatten(xi))
            primals = unflatten(xi)
            params = {}
            for name in free_names:
                lo, hi = bounds[name]
                params[name] = to_bounded(primals[name], lo, hi)
            for name, val in fixed_values.items():
                params[name] = val
            f_cal = params.get("noise_frac_cal", 0.0)
            return variable_noise_hamiltonian(
                data, noise, pred, f_cal, dof=noise_dof
            ) + 0.5 * jnp.sum(xi**2)

    else:

        def metric_vec(xi, v, data_args):
            """M(xi) @ v = J^T N^{-1} J v + v."""
            noise_inv = data_args["noise_inv"]
            xi_d, v_d = unflatten(xi), unflatten(v)
            _, Jv = jax.jvp(signal_response, (xi_d,), (v_d,))
            _, vjp_fn = jax.vjp(signal_response, xi_d)
            return flatten(vjp_fn(noise_inv * Jv)[0]) + v

        def hamiltonian(xi, data_args):
            """H(xi) = 0.5 chi2 + 0.5 ||xi||^2."""
            data = data_args["data"]
            noise = data_args["noise"]
            pred = signal_response(unflatten(xi))
            chi2 = jnp.sum(((data - pred) / noise) ** 2)
            return 0.5 * chi2 + 0.5 * jnp.sum(xi**2)

    def H_vg(xi, data_args):
        """Hamiltonian value and gradient w.r.t. xi only."""
        return jax.value_and_grad(lambda x: hamiltonian(x, data_args))(xi)

    _tiny = 6.0 * jnp.finfo(jnp.float64).tiny
    _n_reset = 20

    def cg_solve(mat_fn, b, x0, maxiter=30, miniter=6, absdelta=0.0, resnorm=0.0):
        """CG solve: mat_fn(x) = b.

        Exact port of NIFTy ``_static_cg`` (conjugate_gradient.py:217-388)
        for flat arrays.  Residual-norm (L2) is the primary convergence
        criterion; energy-based absdelta is secondary.  Negative curvature
        on the first CG iteration triggers a steepest-descent fallback.
        """
        r = mat_fn(x0) - b
        d = r
        gamma = jnp.dot(r, r)
        energy = jnp.dot((r - b) / 2, x0)
        init_info = jnp.where(gamma == 0.0, jnp.int32(0), jnp.int32(-2))
        init = (x0, r, d, gamma, energy, init_info, jnp.int32(0))

        def cond(s):
            return s[5] < -1

        def body(s):
            pos, r, d, prev_gamma, prev_energy, info, i = s
            i = i + 1

            q = mat_fn(d)
            curv = jnp.dot(d, q)
            alpha = prev_gamma / curv

            # Negative / zero curvature (NIFTy cg:278-286)
            info = jnp.where(curv <= 0.0, jnp.int32(0), info)
            alpha = jnp.where(curv <= 0.0, 0.0, alpha)
            pos = pos - alpha * d
            # First iter + negative curvature: steepest-descent fallback
            pos = jnp.where(
                (curv < 0.0) & (i <= 1),
                prev_energy / (-curv) * (-b),
                pos,
            )

            # Periodic residual reset (NIFTy cg:287-291)
            r_reset = mat_fn(pos) - b
            r_step = r - q * alpha
            r = jnp.where((i % _n_reset == 0) & (info < -1), r_reset, r_step)

            gamma = jnp.dot(r, r)

            # Tiny gamma (NIFTy cg:295)
            info = jnp.where(
                (gamma >= 0.0) & (gamma <= _tiny) & (info != -1),
                jnp.int32(0),
                info,
            )

            # Residual norm -- PRIMARY (NIFTy cg:296-298, norm_ord=2)
            r_norm = jnp.sqrt(gamma)
            info = jnp.where(
                (resnorm > 0.0) & (r_norm < resnorm) & (i >= miniter) & (info != -1),
                jnp.int32(0),
                info,
            )

            # Energy -- SECONDARY (NIFTy cg:301-313)
            energy = jnp.dot((r - b) / 2, pos)
            energy_diff = prev_energy - energy
            neg_energy_eps = -_eps * jnp.abs(energy)
            info = jnp.where(
                energy_diff < neg_energy_eps,
                jnp.where(info < -1, i, info),
                info,
            )
            info = jnp.where(
                (absdelta > 0.0) & (energy_diff < absdelta) & (i >= miniter) & (info != -1),
                jnp.int32(0),
                info,
            )

            # Maxiter (NIFTy cg:314)
            info = jnp.where((i >= maxiter) & (info != -1), i, info)

            # Update search direction (NIFTy cg:316)
            d = d * jnp.maximum(0.0, gamma / prev_gamma) + r

            return (pos, r, d, gamma, energy, info, i)

        return jax.lax.while_loop(cond, body, init)[0]

    # --- Posterior sampler: draw linear residuals ---
    def draw_residuals(pos_f, subkeys, data_args):
        """Draw n linear residual samples (vmapped)."""
        sqrt_ni = data_args["sqrt_noise_inv"]
        n_d = n_data  # static, captured at engine-build time

        def draw_one(subkey):
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=(d_total,))
            eta_lh = jax.random.normal(k2, shape=(n_d,))
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            jt = flatten(vjp_fn(sqrt_ni * eta_lh)[0])
            return cg_solve(
                lambda v: metric_vec(pos_f, v, data_args),
                jt + eta_pr,
                eta_pr,
                maxiter=30,
                miniter=6,
                absdelta=1e-4,
            )

        return jax.vmap(draw_one)(subkeys)

    def _draw_batch_fn(pos_f, k, data_args):
        return draw_residuals(pos_f, k, data_args)

    draw_batch = jax.jit(jax.vmap(_draw_batch_fn, in_axes=(None, 0, None)))

    # --- geoVI: nonlinear coordinate transform primitives ---

    def transformation_flat(pos_f, data_args):
        """t(x) = sqrt(N^{-1}) @ f(x). Maps to whitened data-space."""
        sqrt_ni = data_args["sqrt_noise_inv"]
        return sqrt_ni * signal_response(unflatten(pos_f))

    def left_sqrt_metric_flat(pos_f, v_data, data_args):
        """L^T(pos) @ v = J^T(pos) @ sqrt(N^{-1}) @ v.

        Maps whitened data-space vector to parameter-space.
        Matches NIFTy's ``likelihood.left_sqrt_metric(pos, v)``
        for the Gaussian case.
        """
        sqrt_ni = data_args["sqrt_noise_inv"]
        _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
        return flatten(vjp_fn(sqrt_ni * v_data)[0])

    def right_sqrt_metric_flat(pos_f, v_param, data_args):
        """L(pos) @ v = sqrt(N^{-1}) @ J(pos) @ v.

        Maps parameter-space vector to whitened data-space.
        Matches NIFTy's ``likelihood.right_sqrt_metric(pos, v)``
        for the Gaussian case.
        """
        sqrt_ni = data_args["sqrt_noise_inv"]
        _, Jv = jax.jvp(signal_response, (unflatten(pos_f),), (unflatten(v_param),))
        return sqrt_ni * Jv

    def draw_metric_sample(pos_f, subkey, data_args):
        """Draw one sample with covariance M = J^T N^{-1} J + I.

        This is ``draw_linear_residual(..., from_inverse=False)``
        in NIFTy. The metric sample is NOT CG-inverted.
        """
        sqrt_ni = data_args["sqrt_noise_inv"]
        n_d = n_data  # static, captured at engine-build time
        k1, k2 = jax.random.split(subkey)
        eta_pr = jax.random.normal(k1, shape=(d_total,))
        eta_lh = jax.random.normal(k2, shape=(n_d,))
        _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
        jt = flatten(vjp_fn(sqrt_ni * eta_lh)[0])
        return jt + eta_pr

    def _newton_cg_flat(
        fun_and_grad,
        hessp,
        x0,
        custom_gradnorm=None,
        maxiter=10,
        miniter=0,
        xtol=1e-3,
        energy_reduction_factor=0.1,
    ):
        """Newton-CG with successive-halving line search.

        Exact port of NIFTy ``_static_newton_cg`` (optimize.py:285-449)
        for flat arrays.  Includes adaptive CG tolerance, steepest-descent
        reset after 5 line-search halvings, and custom gradient norm.
        """
        ncg_xtol = xtol * d_total  # NIFTy: xtol * size(x0)

        def gradnorm(v):
            if custom_gradnorm is not None:
                return custom_gradnorm(v)
            return jnp.sum(jnp.abs(v))  # L1 norm (NIFTy default)

        energy, g = fun_and_grad(x0)
        init_state = (
            x0,
            energy,
            jnp.array(jnp.inf),
            g,
            jnp.where(maxiter == 0, jnp.int32(0), jnp.int32(-2)),
            jnp.int32(0),
        )

        def ncg_cond(state):
            return state[4] < -1

        def ncg_body(state):
            pos, energy, old_energy, g, status, i = state
            i = i + 1

            # Adaptive CG tolerance (NIFTy optimize.py:351-358)
            cg_abd_fallback = jnp.array(0.0, dtype=energy.dtype)
            cg_absdelta = jnp.where(
                ~jnp.isinf(old_energy),
                energy_reduction_factor * (old_energy - energy),
                cg_abd_fallback,
            )
            cg_absdelta = jnp.array(cg_absdelta, dtype=energy.dtype)

            # CG resnorm (NIFTy optimize.py:359-360, norm_ord=1)
            mag_g = jnp.sum(jnp.abs(g))
            cg_resnorm = jnp.minimum(0.5, jnp.sqrt(mag_g)) * mag_g

            # CG solve (NIFTy: norm_ord=1, _raise_nonposdef=False)
            nat_g = cg_solve(
                lambda v: hessp(pos, v),
                g,
                jnp.zeros_like(pos),
                maxiter=min(200, 20 * d_total),
                miniter=min(6, min(200, 20 * d_total)),
                absdelta=cg_absdelta,
                resnorm=cg_resnorm,
            )

            # Line search: successive halving (NIFTy optimize.py:452-523)
            # State: (status, iter, new_pos, new_energy, new_g,
            #         dd, grad_scaling, reset, nhev)
            ls_init = (
                jnp.int32(-2),
                jnp.int32(0),
                pos,
                jnp.array(jnp.inf),
                g,
                nat_g,
                1.0,
                jnp.bool_(False),
                jnp.int32(0),
            )

            def ls_cond(ls):
                return ls[0] < -1

            def ls_body(ls):
                (
                    ls_st,
                    ls_i,
                    _np,
                    _ne,
                    _ng,
                    dd,
                    gs,
                    reset,
                    nhev,
                ) = ls
                new_pos = pos - gs * dd
                new_e, new_g = fun_and_grad(new_pos)
                ls_st = jnp.where(new_e <= energy, jnp.int32(0), ls_st)
                gs = jnp.where(ls_st < -1, gs / 2.0, gs)
                # Steepest descent reset at iteration 5
                do_reset = (ls_i == 5) & (ls_st < -1)
                reset = jnp.where(do_reset, jnp.bool_(True), reset)
                gs = jnp.where(do_reset, 1.0, gs)
                gam = jnp.dot(g, g)
                curv = jnp.dot(g, hessp(pos, g))
                sd_dd = gam / curv * g
                dd = jnp.where(do_reset, sd_dd, dd)
                nhev = nhev + do_reset.astype(jnp.int32)
                # Abort after 8 iterations
                do_abort = (ls_i == 8) & (ls_st < -1)
                ls_st = jnp.where(do_abort, jnp.int32(-1), ls_st)
                return (
                    ls_st,
                    ls_i + 1,
                    new_pos,
                    new_e,
                    new_g,
                    dd,
                    gs,
                    reset,
                    nhev,
                )

            ls_result = jax.lax.while_loop(ls_cond, ls_body, ls_init)
            (
                ls_status,
                ls_iter,
                new_pos,
                new_energy,
                new_g,
                dd,
                gs,
                _reset,
                _nhev,
            ) = ls_result

            status = jnp.where(ls_status != 0, jnp.int32(-1), status)

            # Update only if line search succeeded (NIFTy opt:381-385)
            success = status < -1
            old_energy = jnp.where(success, energy, old_energy)
            energy_out = jnp.where(success, new_energy, energy)
            energy_diff = jnp.where(success, old_energy - energy_out, 0.0)
            pos_out = jnp.where(success, new_pos, pos)
            g_out = jnp.where(success, new_g, g)
            gs_out = jnp.where(success, gs, 0.0)

            descent_norm = gs_out * gradnorm(dd)

            # absdelta convergence (NIFTy optimize.py:407-414)
            min_cond = (ls_iter < 2) & (i > miniter)
            status = jnp.where(
                (energy_diff >= 0.0) & (energy_diff < 1e-3) & min_cond & (status != -1),
                jnp.int32(0),
                status,
            )
            # xtol convergence (NIFTy optimize.py:415-417)
            status = jnp.where(
                (descent_norm <= ncg_xtol) & (i > miniter) & (status != -1),
                jnp.int32(0),
                status,
            )
            # maxiter (NIFTy optimize.py:418)
            status = jnp.where((i == maxiter) & (status < -1), i, status)

            return (pos_out, energy_out, old_energy, g_out, status, i)

        result = jax.lax.while_loop(ncg_cond, ncg_body, init_state)
        return result[0], result[1]

    def curve_residual(m, r_linear, metric_key, sign, data_args):
        """Nonlinearly update a linear residual to a geoVI curved residual.

        Exact port of NIFTy ``nonlinearly_update_residual``
        (evi.py:136-217) using ``_newton_cg_flat`` for the inner
        Newton-CG optimization.

        Parameters
        ----------
        m : flat array, expansion point
        r_linear : flat array, linear residual (covariance M^{-1})
        metric_key : PRNG key (same as used for draw_residuals)
        sign : +1.0 or -1.0 (for mirrored samples)
        data_args : dict, data-dependent arguments

        Returns
        -------
        flat array : curved residual (x_opt - m)
        """
        x0 = m + r_linear
        ms = sign * draw_metric_sample(m, metric_key, data_args)
        trafo_at_m = transformation_flat(m, data_args)

        def phi_vg(x):
            trafo_x = transformation_flat(x, data_args)
            delta_trafo = trafo_x - trafo_at_m
            g_x = (x - m) + left_sqrt_metric_flat(m, delta_trafo, data_args)
            r = ms - g_x
            val = 0.5 * jnp.dot(r, r)
            ngrad = r + left_sqrt_metric_flat(
                x, right_sqrt_metric_flat(m, r, data_args), data_args
            )
            return val, -ngrad

        def phi_metric(x, v):
            tm = left_sqrt_metric_flat(m, right_sqrt_metric_flat(x, v, data_args), data_args) + v
            return (
                left_sqrt_metric_flat(x, right_sqrt_metric_flat(m, tm, data_args), data_args) + tm
            )

        # sampnorm (evi.py:178-181)
        def sampnorm(natgrad):
            fpp = right_sqrt_metric_flat(m, natgrad, data_args)
            return jnp.sqrt(jnp.dot(natgrad, natgrad) + jnp.dot(fpp, fpp))

        x_opt, _ = _newton_cg_flat(
            phi_vg,
            phi_metric,
            x0,
            custom_gradnorm=sampnorm,
            maxiter=3,
            miniter=0,
            xtol=1e-3,
            energy_reduction_factor=0.1,
        )
        return x_opt - m

    def draw_nonlinear_residuals(m, subkeys, data_args):
        """Draw geoVI nonlinear residuals: linear draw + curving + mirror.

        Returns (2*n_samples, D) array: curved residuals with mirrored pairs.
        Matches NIFTy's ``nonlinear_resample`` sample mode.
        """
        # First draw linear residuals
        linear_residuals = draw_residuals(m, subkeys, data_args)

        # Curve each residual and its mirror
        def curve_pair(r, subkey):
            r_pos = curve_residual(m, r, subkey, sign=1.0, data_args=data_args)
            r_neg = curve_residual(m, -r, subkey, sign=-1.0, data_args=data_args)
            return r_pos, r_neg

        pos_curved, neg_curved = jax.vmap(curve_pair)(linear_residuals, subkeys)
        return jnp.concatenate([pos_curved, neg_curved], axis=0)

    def update_nonlinear_residuals(m, prev_residuals, subkeys, data_args):
        """Re-curve existing residuals at updated expansion point.

        Takes 2*n_samples residuals (first half positive, second half
        negative mirrors) and re-applies geoVI curving at the new m.
        Matches NIFTy's ``nonlinear_update`` sample mode.
        """
        n_half = prev_residuals.shape[0] // 2
        r_pos = prev_residuals[:n_half]
        r_neg = prev_residuals[n_half:]

        def recurve_pair(r_p, r_n, subkey):
            new_p = curve_residual(m, r_p, subkey, sign=1.0, data_args=data_args)
            new_n = curve_residual(m, r_n, subkey, sign=-1.0, data_args=data_args)
            return new_p, new_n

        new_pos, new_neg = jax.vmap(recurve_pair)(r_pos, r_neg, subkeys)
        return jnp.concatenate([new_pos, new_neg], axis=0)

    # --- EVI optimizer: fully JIT'd optimize_kl ---
    def kl_vg(m, residuals, data_args):
        """KL value and gradient averaged over samples."""

        def single_vg(r):
            return H_vg(m + r, data_args)

        vals, grads = jax.vmap(single_vg)(residuals)
        return jnp.mean(vals), jnp.mean(grads, axis=0)

    def kl_metric(m, residuals, v, data_args):
        """KL metric-vector product averaged over samples."""

        def single_met(r):
            return metric_vec(m + r, v, data_args)

        return jnp.mean(jax.vmap(single_met)(residuals), axis=0)

    def evi_step(m, subkey, n_samples, data_args):
        """One EVI iteration: draw samples + Newton-CG KL minimize.

        Returns (m_new, kl_value).
        """
        # Draw linear residual samples + mirror
        sample_keys = jax.random.split(subkey, n_samples)
        residuals = draw_residuals(m, sample_keys, data_args)
        residuals = jnp.concatenate([residuals, -residuals], axis=0)

        # Newton-CG KL minimization (same path as evi_step_full)
        def _evi_kl_vg(m_cur):
            return kl_vg(m_cur, residuals, data_args)

        def _evi_kl_hessp(m_cur, v):
            return kl_metric(m_cur, residuals, v, data_args)

        m_opt, kl_val = _newton_cg_flat(
            _evi_kl_vg,
            _evi_kl_hessp,
            m,
            maxiter=10,
            miniter=0,
            xtol=1e-3,  # match NIFTy default (vi_config.py)
            energy_reduction_factor=0.1,
        )
        return m_opt, kl_val

    def run_evi(init_pos, key, data_args, n_iterations, n_samples, kl_rtol):
        """Run EVI with automatic convergence detection.

        ``n_iterations`` is dynamic — uses ``jax.random.fold_in``
        for per-iteration keys so no pre-split is needed.
        """

        # State: (m, prev_kl, iteration, converged)
        def cond_fn(state):
            _m, _prev_kl, i, converged = state
            return (~converged) & (i < n_iterations)

        def body_fn(state):
            m, prev_kl, i, converged = state
            subkey = jax.random.fold_in(key, i)
            m_new, kl_val = evi_step(m, subkey, n_samples, data_args)
            # Relative KL change
            rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
            # Converge if relative change < rtol and at least 5 iterations done
            converged = (rel_change < kl_rtol) & (i >= 5)
            return (m_new, kl_val, i + 1, converged)

        # First iteration (no convergence check)
        first_key = jax.random.fold_in(key, 0)
        m0, kl0 = evi_step(init_pos, first_key, n_samples, data_args)
        init_state = (m0, kl0, jnp.int32(1), jnp.bool_(False))

        m_final, _kl_final, n_iters, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return m_final, n_iters

    # --- geoVI optimizer: per-mode functions (no lax.switch) ---
    #
    # Each sample mode gets its own evi_step function so that JAX
    # compiles ONLY the code path actually used.  This avoids the
    # 56s compilation cost of tracing all three branches via
    # ``jax.lax.switch``.
    #
    # ``sample_mode`` is a **static** string argument: JAX caches
    # a separate compiled version for each mode.
    SAMPLE_LINEAR = jnp.int32(0)
    SAMPLE_NONLINEAR_RESAMPLE = jnp.int32(1)
    SAMPLE_NONLINEAR_UPDATE = jnp.int32(2)

    def _kl_minimize(m, residuals, constants_mask, data_args):
        """Newton-CG KL minimization with constants mask."""

        def _masked_kl_vg(m_cur, res):
            val, grad = kl_vg(m_cur, res, data_args)
            grad = jnp.where(constants_mask, 0.0, grad)
            return val, grad

        def _masked_kl_metric(m_cur, res, v):
            v_masked = jnp.where(constants_mask, 0.0, v)
            mv = kl_metric(m_cur, res, v_masked, data_args)
            return jnp.where(constants_mask, 0.0, mv)

        def _fun_and_grad(m_cur):
            return _masked_kl_vg(m_cur, residuals)

        def _hessp(m_cur, v):
            return _masked_kl_metric(m_cur, residuals, v)

        return _newton_cg_flat(
            _fun_and_grad,
            _hessp,
            m,
            maxiter=10,
            miniter=0,
            xtol=1e-3,  # match NIFTy default (vi_config.py)
            energy_reduction_factor=0.1,
        )

    _RESAMPLE_EVERY = 5  # refresh stale samples every N iterations

    def evi_step_full(
        m,
        subkey,
        n_samples,
        sample_mode,
        prev_residuals,
        prev_keys,
        constants_mask,
        pe_mask,
        data_args,
        iteration=0,
    ):
        """One geoVI iteration — ``sample_mode`` must be a static string.

        When used inside ``run_evi_geovi`` (which marks ``sample_mode``
        as static), JAX compiles a separate version per mode.  The
        unused branches are never traced, so ``"linear"`` compiles in
        ~0.03s while ``"nonlinear_resample"`` compiles in ~56s.

        Parameters
        ----------
        sample_mode : str  (STATIC — triggers recompilation per value)
            ``"linear_resample"`` — fresh MGVI samples (standard MGVI)
            ``"linear_sample"`` — reuse keys from prev iter (deterministic MGVI)
            ``"nonlinear_resample"`` — fresh geoVI samples (standard geoVI)
            ``"nonlinear_sample"`` — reuse keys + curve (deterministic geoVI)
            ``"nonlinear_update"`` — re-curve existing residuals at new m
        data_args : dict
            Data-dependent arguments (data, noise, noise_inv, etc.).

        Returns
        -------
        m_new, kl_value, new_residuals, used_keys
        """
        # Key handling: _resample = fresh keys, _sample = reuse prev keys
        if sample_mode.endswith("_resample") or sample_mode == "vi":
            sample_keys = jax.random.split(subkey, n_samples)
        elif sample_mode == "nonlinear_update":
            sample_keys = prev_keys
        else:  # _sample modes: reuse
            sample_keys = prev_keys

        # Python if — only the used branch is traced by JAX
        if sample_mode == "vi":
            # Optimal schedule: resample at iter 0 and every
            # _RESAMPLE_EVERY, nonlinear_update in between.
            # Uses jax.lax.cond (traces both branches, executes one).
            do_resample = (iteration == 0) | (iteration % _RESAMPLE_EVERY == 0)

            def _do_resample(_):
                return draw_nonlinear_residuals(m, sample_keys, data_args)

            def _do_update(_):
                return update_nonlinear_residuals(m, prev_residuals, prev_keys, data_args)

            residuals = jax.lax.cond(do_resample, _do_resample, _do_update, None)
        elif sample_mode in ("nonlinear_resample", "nonlinear_sample"):
            residuals = draw_nonlinear_residuals(m, sample_keys, data_args)
        elif sample_mode == "nonlinear_update":
            residuals = update_nonlinear_residuals(m, prev_residuals, sample_keys, data_args)
        else:  # linear_resample, linear_sample
            res = draw_residuals(m, sample_keys, data_args)
            residuals = jnp.concatenate([res, -res], axis=0)

        # Apply point estimates mask
        residuals = residuals * pe_mask[None, :]

        # KL minimization
        m_opt, kl_val = _kl_minimize(m, residuals, constants_mask, data_args)
        return m_opt, kl_val, residuals, sample_keys

    def run_evi_geovi(init_pos, key, data_args, n_iterations, n_samples, kl_rtol, sample_mode):
        """Run geoVI with automatic convergence detection.

        ``n_iterations`` is a **dynamic** traced value — changing it
        does NOT trigger recompilation.  Keys are generated on-the-fly
        via ``jax.random.fold_in`` instead of pre-splitting.

        ``sample_mode`` is a **static** string — JAX compiles a
        separate XLA program per mode.  All 5 NIFTy modes supported:

        - ``"linear_resample"`` — fresh MGVI samples each iteration
        - ``"linear_sample"`` — reuse PRNG keys (deterministic MGVI)
        - ``"nonlinear_resample"`` — fresh geoVI samples
        - ``"nonlinear_sample"`` — reuse keys + curve (deterministic geoVI)
        - ``"nonlinear_update"`` — re-curve existing residuals at new m
        """
        # Generate per-iteration keys on-the-fly via fold_in (no
        # pre-split needed, so n_iterations can be dynamic).
        dummy_residuals = jnp.zeros((2 * n_samples, d_total))
        dummy_keys = jax.random.split(jax.random.fold_in(key, 0), n_samples)
        no_constants = jnp.zeros(d_total, dtype=bool)
        all_sampled = jnp.ones(d_total)

        # State: (m, prev_kl, residuals, prev_keys, iter, converged)
        def cond_fn(state):
            _m, _prev_kl, _res, _pk, i, converged = state
            return (~converged) & (i < n_iterations)

        def body_fn(state):
            m, prev_kl, prev_res, prev_k, i, converged = state
            subkey = jax.random.fold_in(key, i)
            m_new, kl_val, new_res, new_k = evi_step_full(
                m,
                subkey,
                n_samples,
                sample_mode,
                prev_res,
                prev_k,
                no_constants,
                all_sampled,
                data_args,
                iteration=i,
            )
            rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
            converged = (rel_change < kl_rtol) & (i >= 5)
            return (m_new, kl_val, new_res, new_k, i + 1, converged)

        # First iteration (always resample to establish initial keys)
        first_key = jax.random.fold_in(key, 0)
        m0, kl0, res0, keys0 = evi_step_full(
            init_pos,
            first_key,
            n_samples,
            sample_mode,
            dummy_residuals,
            dummy_keys,
            no_constants,
            all_sampled,
            data_args,
        )
        init_state = (
            m0,
            kl0,
            res0,
            keys0,
            jnp.int32(1),
            jnp.bool_(False),
        )

        result = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return result[0], result[4]  # m_final, n_iters

    # --- Parameter range mapping for mask construction ---
    param_ranges = {}
    for i_k, k in enumerate(param_keys):
        start, end, _shape = slices[i_k]
        param_ranges[k] = (start, end)

    def make_mask(param_names):
        """Create boolean mask: True for named params, False otherwise."""
        mask = jnp.zeros(d_total, dtype=bool)
        for name in param_names:
            if name in param_ranges:
                start, end = param_ranges[name]
                mask = mask.at[start:end].set(True)
        return mask

    def make_pe_mask(param_names):
        """Create point-estimate mask: 0.0 for PE params, 1.0 for sampled."""
        mask = jnp.ones(d_total)
        for name in param_names:
            if name in param_ranges:
                start, end = param_ranges[name]
                mask = mask.at[start:end].set(0.0)
        return mask

    # --- NIFTy-backed geoVI: exact NIFTy math, minimal Python overhead ---
    # Uses NIFTy's OptimizeVI.update directly (already JIT'd internally)
    # but skips logging, pickling, and callbacks for speed.
    nifty_likelihood = None
    nifty_opt_vi = None
    if _has_nifty:
        try:
            import nifty8.re as jft

            # Build the NIFTy likelihood (same as _run_nifty_vi)
            _nifty_domain = {}
            for name in fitter._free_names:
                _nifty_domain[name] = jft.ShapeWithDtype(())
            if fitter.spec.stochastic:
                _nifty_domain["psd_xi"] = jft.ShapeWithDtype((fitter.spec.n_grid,))
            _nifty_model = jft.Model(jax.jit(signal_response), domain=_nifty_domain)
            if not use_variable_noise:
                nifty_likelihood = jft.Gaussian(fitter.data, fitter._data_args["noise_inv"]).amend(
                    _nifty_model
                )
            # Build OptimizeVI with vmap and JIT.
            nifty_opt_vi = OptimizeVI(
                nifty_likelihood,
                n_total_iterations=50,  # max, actual controlled by caller
                kl_jit=True,
                residual_jit=True,
                kl_map=jax.vmap,
                residual_map=jax.vmap,
            )
        except Exception:
            _has_nifty = False
            nifty_likelihood = None
            nifty_opt_vi = None

    def run_nifty_jit(
        init_pos_flat,
        key,
        n_iterations,
        n_samples,
        sample_mode_str,
        draw_linear_kwargs,
        nonlinearly_update_kwargs,
        kl_kwargs,
    ):
        """Run NIFTy's exact optimize_kl with minimal Python overhead.

        Uses NIFTy's OptimizeVI.update (already JIT'd) in a tight
        Python loop — no logging, no pickling, no callbacks.
        Exact same math as ``jft.optimize_kl``.

        Returns (converged_flat, n_iters).
        """
        import nifty8.re as jft

        pos_dict_local = unflatten(init_pos_flat)
        samples = NiftySamples(pos=jft.Vector(pos_dict_local), samples=None, keys=None)
        state = nifty_opt_vi.init_state(
            key,
            n_samples=n_samples,
            sample_mode=sample_mode_str,
            draw_linear_kwargs=draw_linear_kwargs,
            nonlinearly_update_kwargs=nonlinearly_update_kwargs,
            kl_kwargs=kl_kwargs,
        )
        for _i in range(n_iterations):
            samples, state = nifty_opt_vi.update(samples, state)
        converged = samples.pos
        pos_d = converged.tree if hasattr(converged, "tree") else dict(converged)
        return flatten(pos_d), samples

    # Wrap core functions in JIT but do NOT pre-compile (no dummy calls).
    # Compilation happens lazily on first real call — avoids the 2+ GB
    # protobuf size limit that eager compilation can hit when the forward
    # model is large.  signal_response is already JIT'd above so it
    # won't be re-traced into these scopes.
    draw_samples_jit = jax.jit(draw_residuals)

    run_evi_jit = jax.jit(run_evi, static_argnames=("n_samples",))

    run_evi_geovi_jit = jax.jit(
        run_evi_geovi,
        static_argnames=("n_samples", "sample_mode"),
    )

    return {
        "run_evi": run_evi_jit,
        "run_evi_geovi": run_evi_geovi_jit,
        "run_nifty_jit": run_nifty_jit if _has_nifty else None,
        "nifty_likelihood": nifty_likelihood,
        "draw_samples": draw_samples_jit,
        "draw_nonlinear_samples": jax.jit(draw_nonlinear_residuals),
        "draw_batch": draw_batch,
        "flatten": flatten,
        "unflatten": unflatten,
        "param_keys": param_keys,
        "param_ranges": param_ranges,
        "make_mask": make_mask,
        "make_pe_mask": make_pe_mask,
        "d_total": d_total,
        "SAMPLE_LINEAR": SAMPLE_LINEAR,
        "SAMPLE_NONLINEAR_RESAMPLE": SAMPLE_NONLINEAR_RESAMPLE,
        "SAMPLE_NONLINEAR_UPDATE": SAMPLE_NONLINEAR_UPDATE,
        "evi_step_full": evi_step_full,
        # geoVI-NUTS primitives (coordinate transform + metric)
        "transformation_flat": transformation_flat,
        "left_sqrt_metric_flat": left_sqrt_metric_flat,
        "right_sqrt_metric_flat": right_sqrt_metric_flat,
        "metric_vec": metric_vec,
        "cg_solve": cg_solve,
        "hamiltonian": hamiltonian,
    }
