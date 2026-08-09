# SPDX-License-Identifier: BSD-3-Clause
"""Pipeline profiler for tengri forward model.

Instruments the ``predict_sed`` / ``predict_photometry`` pipeline to
produce a step-by-step timing breakdown, automatically detecting the
execution path (fused vs exact) and model configuration.

Usage
-----
>>> from tengri.profiling.pipeline import profile_pipeline
>>> report = profile_pipeline(model, params)
>>> print(report.summary())
>>> report.to_csv("profiling/outputs/pipeline_timing.csv")

>>> # Compare fused vs exact
>>> from tengri.profiling.pipeline import compare_paths
>>> compare_paths(spec, ssp, filters, params)
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Any

import jax
import jax.numpy as jnp

from tengri.profiling.timers import _sync, bench

# ── Data containers ───────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class StepTiming:
    """Timing result for a single pipeline step."""

    name: str
    mean_us: float
    grad_us: float | None = None
    array_mb: float | None = None
    pct: float = 0.0


@dataclasses.dataclass
class PipelineReport:
    """Complete pipeline profiling report."""

    steps: list[StepTiming]
    total_us: float
    gradient_us: float | None
    compile_us: float | None
    path: str  # "FUSED" or "EXACT"
    n_free: int
    config_name: str = ""

    def summary(self) -> str:
        """Human-readable summary table."""
        lines = []
        lines.append(f"PIPELINE BREAKDOWN ({self.config_name}, D={self.n_free}, {self.path} path)")
        lines.append("=" * 72)
        lines.append(f"{'Step':<40s} {'Time (μs)':>10s} {'% Total':>8s} {'Grad (μs)':>10s}")
        lines.append("-" * 72)

        for step in self.steps:
            grad_str = f"{step.grad_us:.1f}" if step.grad_us is not None else "—"
            lines.append(
                f"  {step.name:<38s} {step.mean_us:>8.1f} {step.pct:>7.1f}% {grad_str:>10s}"
            )

        lines.append("-" * 72)
        grad_total = f"{self.gradient_us:.1f}" if self.gradient_us is not None else "—"
        lines.append(f"  {'TOTAL':<38s} {self.total_us:>8.1f} {'100.0%':>8s} {grad_total:>10s}")

        if self.compile_us is not None:
            lines.append(
                f"\n  Compilation (first call):  {self.compile_us:>10.0f} μs"
                f" ({self.compile_us / 1e6:.1f}s)"
            )

        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """Write timing breakdown to CSV."""
        import csv

        fieldnames = [
            "step",
            "mean_us",
            "pct",
            "grad_us",
            "array_mb",
            "path",
            "n_free",
            "config",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for step in self.steps:
                writer.writerow(
                    {
                        "step": step.name,
                        "mean_us": f"{step.mean_us:.1f}",
                        "pct": f"{step.pct:.1f}",
                        "grad_us": f"{step.grad_us:.1f}" if step.grad_us is not None else "",
                        "array_mb": f"{step.array_mb:.3f}" if step.array_mb is not None else "",
                        "path": self.path,
                        "n_free": self.n_free,
                        "config": self.config_name,
                    }
                )

    def __repr__(self) -> str:
        return self.summary()


# ── Array memory helper ───────────────────────────────────────────


def _array_mb(arr: Any) -> float:
    """Memory of a JAX/numpy array in MB."""
    if arr is None:
        return 0.0
    if hasattr(arr, "nbytes"):
        return arr.nbytes / 1e6
    return 0.0


# ── Step-level timing ─────────────────────────────────────────────


def _time_step(name: str, fn, n: int = 200, grad_fn=None) -> StepTiming:
    """Time a single pipeline step."""
    t_fwd, result = bench(fn, n=n, warmup=3)
    mem = _array_mb(result)

    t_grad = None
    if grad_fn is not None:
        # Broad by design: a step whose gradient does not compile is a real
        # result for a profiler, and should not abort the rest of the report.
        # ``t_grad`` staying None is already visible in the output, so the log
        # only has to say *why* — otherwise a missing timing reads as "not
        # measured" when it means "measurement failed".
        try:
            t_grad, _ = bench(grad_fn, n=n, warmup=3)
        except Exception as exc:  # noqa: BLE001 - profiling must not abort
            import logging

            logging.getLogger(__name__).debug(
                "gradient timing for step %r unavailable (%s: %s)",
                name,
                type(exc).__name__,
                exc,
            )

    return StepTiming(name=name, mean_us=t_fwd, grad_us=t_grad, array_mb=mem)


# ── Pipeline profiling: exact path ────────────────────────────────


def _profile_exact_path(model, params, n: int = 200) -> PipelineReport:
    """Profile the exact (non-fused) predict_sed pipeline step by step."""
    from tengri.components.dust.attenuation import two_component_dust
    from tengri.components.stellar.sps.dsps_wrapper import (
        compute_csp_sed,
        compute_csp_weights,
        interpolate_metallicity,
    )
    from tengri.observation.photometry import compute_flux_density

    p = model._get_internal_params(params)
    steps = []

    # 1. Param conversion
    steps.append(
        _time_step(
            "param_conversion",
            lambda: model._get_internal_params(params),
            n=min(n, 500),
        )
    )

    # 2. SFH computation
    step_sfh = _time_step("sfh_computation", lambda: model._compute_sfr(p), n=min(n, 500))
    steps.append(step_sfh)
    sfr = model._compute_sfr(p)

    # 3. SFR interpolation to SSP ages
    step_interp = _time_step(
        "sfr_interpolation",
        lambda: jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr),
        n=min(n, 500),
    )
    steps.append(step_interp)
    sfr_on_ssp = jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr)

    # 4. CSP weights
    steps.append(
        _time_step(
            "csp_weights",
            lambda: compute_csp_weights(sfr_on_ssp, model.ssp_ages_yr),
            n=min(n, 500),
        )
    )
    weights = compute_csp_weights(sfr_on_ssp, model.ssp_ages_yr)

    # 5. Metallicity interpolation
    steps.append(
        _time_step(
            "metallicity_interp",
            lambda: interpolate_metallicity(
                model.ssp_data.ssp_flux,
                model.ssp_data.ssp_lgmet,
                p.get("log_z_abs", p.get("log_z_abs_final", -1.8477)),
            ),
            n=n,
        )
    )
    ssp_at_z = interpolate_metallicity(
        model.ssp_data.ssp_flux,
        model.ssp_data.ssp_lgmet,
        p.get("log_z_abs", p.get("log_z_abs_final", -1.8477)),
    )

    # 6. Dust attenuation
    dust_kw = {}
    if hasattr(model, "_dust_law_bc"):
        dust_kw["law_bc"] = model._dust_law_bc
        dust_kw["law_diff"] = model._dust_law_diff
    dust_kw["n_slope"] = p.get("dust_slope", -0.7)

    steps.append(
        _time_step(
            "dust_attenuation",
            lambda: two_component_dust(
                model.ssp_data.ssp_wave,
                model.ssp_ages_yr,
                p["tau_bc"],
                p["tau_diff"],
                **dust_kw,
            ),
            n=n,
        )
    )
    dust = two_component_dust(
        model.ssp_data.ssp_wave,
        model.ssp_ages_yr,
        p["tau_bc"],
        p["tau_diff"],
        **dust_kw,
    )

    # 7. CSP SED (einsum)
    steps.append(
        _time_step(
            "csp_sed_einsum",
            lambda: compute_csp_sed(weights, ssp_at_z, dust),
            n=n,
        )
    )
    sed = compute_csp_sed(weights, ssp_at_z, dust)

    # 8. Photometric integration
    if model.filter_waves is not None:
        from tengri.cosmology import luminosity_distance

        z = model._get_redshift(params)
        dl_cm = luminosity_distance(z)
        fw_list = model.filter_waves
        ft_list = model.filter_trans

        def phot_loop():
            """Integrate SED through all filter curves."""
            fluxes = []
            for fwi, fti in zip(fw_list, ft_list):
                f = compute_flux_density(sed, model.ssp_data.ssp_wave, fwi, fti, z, dl_cm)
                fluxes.append(f)
            return jnp.array(fluxes)

        steps.append(_time_step("photometric_integration", phot_loop, n=min(n, 100)))

    # Compute percentages
    total_us = sum(s.mean_us for s in steps)
    steps_with_pct = [
        dataclasses.replace(s, pct=(s.mean_us / total_us * 100) if total_us > 0 else 0)
        for s in steps
    ]

    # Full-model gradient timing
    grad_us = None
    try:
        grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_photometry(p))))
        _ = grad_fn(params)
        t_grad, _ = bench(lambda: grad_fn(params), n=min(n, 100))
        grad_us = t_grad
    except (AttributeError, TypeError, RuntimeError):
        # AttributeError: predict_photometry doesn't exist
        # TypeError: grad/jit compilation failed
        # RuntimeError: JAX compilation error
        pass

    return PipelineReport(
        steps=steps_with_pct,
        total_us=total_us,
        gradient_us=grad_us,
        compile_us=None,
        path="EXACT",
        n_free=len(model._spec.free_params),
        config_name="predict_photometry",
    )


# ── Pipeline profiling: fused path ────────────────────────────────


def _profile_fused_path(model, params, n: int = 200) -> PipelineReport:
    """Profile the fused (precomputed) predict_photometry pipeline."""
    steps = []

    p = model._get_internal_params(params)

    # 1. Param conversion
    steps.append(
        _time_step(
            "param_conversion",
            lambda: model._get_internal_params(params),
            n=min(n, 500),
        )
    )

    # 2. SFH computation
    steps.append(_time_step("sfh_computation", lambda: model._compute_sfr(p), n=min(n, 500)))
    sfr = model._compute_sfr(p)

    # 3. SFR interpolation
    steps.append(
        _time_step(
            "sfr_interpolation",
            lambda: jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr),
            n=min(n, 500),
        )
    )

    # 4. Fused kernel (everything else in one JIT scope)
    # Time the full predict_photometry which internally calls the fused kernel
    steps.append(
        _time_step(
            "fused_kernel",
            lambda: model.predict_photometry(params),
            n=n,
        )
    )

    # Compute percentages
    total_us = sum(s.mean_us for s in steps)
    steps_with_pct = [
        dataclasses.replace(s, pct=(s.mean_us / total_us * 100) if total_us > 0 else 0)
        for s in steps
    ]

    # Compilation time
    grad_us = None
    try:
        grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_photometry(p))))
        _ = grad_fn(params)
        _sync(grad_fn(params))
        # Now the compiled gradient
        t_grad, _ = bench(lambda: grad_fn(params), n=n)
        grad_us = t_grad
    except (AttributeError, TypeError, RuntimeError):
        # AttributeError: predict_photometry doesn't exist
        # TypeError: grad/jit compilation failed
        # RuntimeError: JAX compilation error
        pass

    # Forward compilation time
    fwd_compile = None
    try:
        pred_fn = jax.jit(model.predict_photometry)
        t0 = time.perf_counter()
        r = pred_fn(params)
        _sync(r)
        fwd_compile = (time.perf_counter() - t0) * 1e6
    except (AttributeError, TypeError, RuntimeError):
        # AttributeError: predict_photometry doesn't exist
        # TypeError: jit compilation failed
        # RuntimeError: JAX compilation error
        pass

    return PipelineReport(
        steps=steps_with_pct,
        total_us=total_us,
        gradient_us=grad_us,
        compile_us=fwd_compile,
        path="FUSED",
        n_free=len(model._spec.free_params),
        config_name="predict_photometry",
    )


# ── Public API ────────────────────────────────────────────────────


def profile_pipeline(
    model,
    params: dict,
    n: int = 200,
    config_name: str = "",
) -> PipelineReport:
    """Profile the forward model pipeline.

    Automatically detects whether the model uses the fused or exact path
    and instruments accordingly.

    Parameters
    ----------
    model : SEDModel
        A tengri SEDModel instance.
    params : dict
        Parameter values (public names).
    n : int
        Number of timed iterations per step.
    config_name : str
        Label for this configuration (e.g., "stellar+dust, D=7").

    Returns
    -------
    PipelineReport
        Timing breakdown with summary and CSV export.

    Examples
    --------
    >>> report = profile_pipeline(model, params, config_name="smooth D=7")
    >>> print(report)
    >>> report.to_csv("pipeline_timing.csv")
    """
    # Warmup: ensure everything is compiled
    _ = model.predict_photometry(params)
    _sync(model.predict_photometry(params))

    has_fast_path = model.has_fixedz_photometry_precompute and (
        getattr(getattr(model, "hybrid", None), "photometry", None) is not None
        or getattr(getattr(model, "_compositional", None), "photometry", None) is not None
    )

    if has_fast_path:
        report = _profile_fused_path(model, params, n=n)
    else:
        report = _profile_exact_path(model, params, n=n)

    if config_name:
        report.config_name = config_name

    return report




