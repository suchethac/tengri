"""Display / introspection helpers extracted from core/model.py.

Each function takes an SEDModel instance as its first argument so that model.py
delegates to a one-liner stub and stays focused on the forward model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tengri.core.model import SEDModel


# ---------------------------------------------------------------------------
# Inference method recommendation
# ---------------------------------------------------------------------------


def method_recommendation(model: SEDModel) -> tuple[str, str]:
    """Return (method_name, reason) for the recommended inference method.

    Parameters
    ----------
    model : SEDModel
        Configured model instance.

    Returns
    -------
    tuple of (str, str)
        ``(method_name, reason)`` where *method_name* is a canonical string
        accepted by ``Fitter.run()`` and *reason* is a human-readable
        explanation.
    """
    d = model.spec.n_free
    if model.spec.stochastic:
        d_total = d + model.spec.n_grid
        return "vi", f"D={d_total}, stochastic, geoVI default"
    elif d <= 15:
        return "laplace", f"D={d}, smooth, instant Gaussian approximation"
    elif d <= 50:
        return "vi_linear", f"D={d}, smooth, fast VI"
    else:
        return "vi", f"D={d}, moderate-high, geoVI default"


# ---------------------------------------------------------------------------
# tree()
# ---------------------------------------------------------------------------


def tree(model: SEDModel) -> str:
    """Return a human-readable physics tree showing the model hierarchy.

    Shows the active sub-models at each physical layer (SFH, SPS, Dust,
    Nebular, AGN, Observation), the free parameters at each layer, and
    the recommended inference method.

    Parameters
    ----------
    model : SEDModel
        Configured model instance.

    Returns
    -------
    str
        Multi-line formatted tree string.
    """
    sep = "│"
    branch = "├──"
    last = "└──"
    lines: list[str] = []

    d = model.spec.n_free
    stoch = "True" if model.spec.stochastic else "False"
    n_grid = model.spec.n_grid if model.spec.stochastic else 0
    d_total = d + n_grid
    lines.append(f"Model  [D={d_total}, stochastic={stoch}]")
    lines.append(sep)

    # SFH layer
    sfh_type = getattr(model.spec, "mean_sfh_type", ["unknown"])
    sfh_name = "+".join(sfh_type) if isinstance(sfh_type, (list, tuple)) else str(sfh_type)
    lines.append(f"{branch} SFH: {sfh_name}")
    sfh_params = [
        p
        for p in model.spec.free_params
        if p.startswith("sfh_") or p in ("psd_sigma", "psd_tau_myr")
    ]
    for i, name in enumerate(sfh_params):
        prefix = last if i == len(sfh_params) - 1 else branch
        try:
            dist = model.spec.get_distribution(name)
            lines.append(f"{sep}   {prefix} {name:<30s} ~ {dist!r}")
        except Exception:
            lines.append(f"{sep}   {prefix} {name}")
    if model.spec.stochastic:
        lines.append(f"{sep}   {last} sfh_field_xi  [{n_grid}-dim GP latent, xi ~ N(0,I)]")
    lines.append(sep)

    # SPS layer
    try:
        ssp = model.ssp_data
        n_met, n_age, n_wave = ssp.ssp_flux.shape
        lines.append(f"{branch} SPS: DSPS  [{n_met} Z x {n_age} ages x {n_wave} lambda]")
    except Exception:
        lines.append(f"{branch} SPS: DSPS")
    lines.append(sep)

    # Dust layer
    lines.append(f"{branch} Dust: Charlot & Fall")
    dust_params = [p for p in model.spec.free_params if p.startswith("dust_")]
    for name in dust_params:
        try:
            dist = model.spec.get_distribution(name)
            lines.append(f"{sep}   {branch} {name:<30s} ~ {dist!r}")
        except Exception:
            lines.append(f"{sep}   {branch} {name}")
    lines.append(sep)

    # Nebular layer
    neb_mode = getattr(model.spec, "nebular_mode", None)
    if neb_mode and neb_mode != "off":
        lines.append(f"{branch} Nebular: {neb_mode}")
        neb_params = [p for p in model.spec.free_params if p.startswith("neb_")]
        for name in neb_params:
            try:
                dist = model.spec.get_distribution(name)
                lines.append(f"{sep}   {branch} {name:<30s} ~ {dist!r}")
            except Exception:
                lines.append(f"{sep}   {branch} {name}")
        lines.append(sep)

    # AGN layer
    agn_model = getattr(model, "_agn_model", None) or getattr(model.spec, "agn_model", None)
    if agn_model:
        lines.append(f"{branch} AGN: {agn_model}")
        agn_params = [p for p in model.spec.free_params if p.startswith("agn_")]
        for name in agn_params:
            try:
                dist = model.spec.get_distribution(name)
                lines.append(f"{sep}   {branch} {name:<30s} ~ {dist!r}")
            except Exception:
                lines.append(f"{sep}   {branch} {name}")
        lines.append(sep)

    # Observation layer
    filter_waves = getattr(model, "filter_waves", None)
    z_fixed = getattr(model, "_z_fixed", None)
    z_info = f"z={z_fixed:.4f} [fixed]" if z_fixed is not None else "z [free]"
    if filter_waves is not None:
        n_filt = len(filter_waves)
        precomp = model._precomputed.photometry
        precomp_str = "YES (21.6x speedup)" if precomp is not None else "NO"
        lines.append(f"{last} Observation: Photometry [{n_filt} bands] at {z_info}")
        lines.append(f"    Precomputed: {precomp_str}")
    else:
        wave_obs = getattr(model, "_wave_obs", None)
        if wave_obs is not None:
            lines.append(f"{last} Observation: Spectroscopy at {z_info}")
        else:
            lines.append(f"{last} Observation: {z_info}")

    lines.append("")
    rec_method, reason = method_recommendation(model)
    lines.append("Recommended inference:")
    lines.append(f"  -> model.fit(data, noise, method={rec_method!r})   [{reason}]")
    if not model.spec.stochastic and d <= 30:
        lines.append("  -> model.fit(data, noise, method='evidence')  [Bayesian evidence, D<=30]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def summary(model: SEDModel) -> str:
    """Return a human-readable summary of the model configuration.

    Parameters
    ----------
    model : SEDModel
        Configured model instance.

    Returns
    -------
    str
        Formatted summary showing SSP grid, filters, precomputation,
        fused kernel status, and enabled components.
    """
    sep = "─" * 66
    lines: list[str] = [f"Model  SFH: {'+'.join(model.spec.mean_sfh_type)}", sep]

    # SSP grid
    n_met, n_age, n_wave = model.ssp_data.ssp_flux.shape
    wave = model.ssp_data.ssp_wave
    lines.append(
        f"  SSP grid:    {n_met} Z × {n_age} ages × {n_wave} λ "
        f"[{float(wave[0]):.0f}–{float(wave[-1]):.0f} Å]"
    )

    # Filters
    if model.filter_waves is not None:
        n_filt = len(model.filter_waves)
        lines.append(f"  Filters:     {n_filt} bands")
    else:
        lines.append("  Filters:     none")

    # Redshift
    if model._z_fixed is not None:
        lines.append(f"  Redshift:    {model._z_fixed:.4f} (fixed)")
    else:
        lines.append("  Redshift:    free")

    # Dtype and precomputation
    lines.append(f"  Dtype:       {model._forward_dtype}")
    precomp_parts: list[str] = []
    if model._precomputed.photometry is not None:
        precomp_parts.append("photometry")
    if model._precomputed.spectroscopy is not None:
        precomp_parts.append("spectroscopy")
    if model._precomputed.photometry_ztable is not None:
        precomp_parts.append("z-table")
    lines.append(f"  Precomputed: {', '.join(precomp_parts) if precomp_parts else 'none'}")

    # Fused kernel status
    fused = "active" if model._precomputed_kernels.photometry is not None else "off"
    lines.append(f"  Fused kernel: {fused}")

    # Enabled components
    components: list[str] = []
    if model.spec.nebular_mode != "off":
        components.append(f"nebular={model.spec.nebular_mode}")
    if model._dust_emission_model:
        components.append(f"dust_emission={model._dust_emission_model}")
    if model._agn_model:
        components.append(f"agn={model._agn_model}")
    if model._apply_igm:
        components.append("igm")
    if model._radio_enabled:
        components.append("radio")
    if model._xray_enabled:
        components.append("xray")
    if model._shock_enabled:
        components.append("shock")
    if components:
        lines.append(f"  Components:  {', '.join(components)}")

    # Dimensionality
    n_free = model.spec.n_free
    n_grid = model._n_grid if model._has_field else 0
    lines.append(f"  Parameters:  {n_free} free" + (f" + {n_grid} latent (ξ)" if n_grid else ""))

    lines.append(sep)
    return "\n".join(lines)
