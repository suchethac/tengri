# SPDX-License-Identifier: BSD-3-Clause
"""Display / introspection helpers extracted from core/model.py.

Each function takes an SEDModel instance as its first argument so that model.py
delegates to a one-liner stub and stays focused on the forward model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tengri.forward.sed_model import SEDModel


# ── Inference method recommendation ───────────────────────────────


def method_recommendation(model: SEDModel) -> tuple[str, str]:
    """Return (method_name, reason) for the recommended inference method.

    Parameters
    ----------
    model: SEDModel
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


# ── tree() ────────────────────────────────────────────────────────


def tree(model: SEDModel) -> str:
    """Return a human-readable physics tree showing the model hierarchy.

    Shows the active sub-models at each physical layer (SFH, SPS, Dust,
    Nebular, AGN, Observation), the free parameters at each layer, and
    the recommended inference method.

    Parameters
    ----------
    model: SEDModel
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
    sfh_params = [p for p in model.spec.free_params if p.startswith("sfh_")]
    for i, name in enumerate(sfh_params):
        prefix = last if i == len(sfh_params) - 1 else branch
        try:
            dist = model.spec.get_distribution(name)
            lines.append(f"{sep}   {prefix} {name:<30s} ~ {dist!r}")
        except (KeyError, AttributeError, ValueError):
            lines.append(f"{sep}   {prefix} {name}")
    if model.spec.stochastic:
        lines.append(f"{sep}   {last} sfh_field_xi  [{n_grid}-dim GP latent, xi ~ N(0,I)]")
    lines.append(sep)

    # SPS layer
    try:
        ssp = model.ssp_data
        n_met, n_age, n_wave = ssp.ssp_flux.shape
        lines.append(f"{branch} SPS: DSPS  [{n_met} Z x {n_age} ages x {n_wave} lambda]")
    except (AttributeError, ValueError):
        lines.append(f"{branch} SPS: DSPS")
    lines.append(sep)

    # Dust layer
    lines.append(f"{branch} Dust: Charlot & Fall")
    dust_params = [p for p in model.spec.free_params if p.startswith("dust_")]
    for name in dust_params:
        try:
            dist = model.spec.get_distribution(name)
            lines.append(f"{sep}   {branch} {name:<30s} ~ {dist!r}")
        except (KeyError, AttributeError, ValueError):
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
            except (KeyError, AttributeError, ValueError):
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
            except (KeyError, AttributeError, ValueError):
                lines.append(f"{sep}   {branch} {name}")
        lines.append(sep)

    # Observation layer
    filter_waves = getattr(model, "filter_waves", None)
    z_fixed = getattr(model, "_z_fixed", None)
    z_info = f"z={z_fixed:.4f} [fixed]" if z_fixed is not None else "z [free]"
    if filter_waves is not None:
        n_filt = len(filter_waves)
        fast = getattr(model, "has_fixedz_photometry_precompute", False)
        precomp_str = "eligible (build with approx=WavePrecomp())" if fast else "NO"
        lines.append(f"{last} Observation: Photometry [{n_filt} bands] at {z_info}")
        lines.append(f"    Fast path: {precomp_str}")
    else:
        # Probe the observation itself: ``model._wave_obs`` is a cache slot
        # that is never populated, so it mislabeled every spectroscopy model
        # with the generic line (same fail-open class as #1222).
        spec_obs = getattr(getattr(model, "observation", None), "spectroscopy", None)
        if spec_obs is not None:
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


# ── summary() ─────────────────────────────────────────────────────


def summary(model: SEDModel) -> str:
    """Return a human-readable summary of the model configuration.

    Parameters
    ----------
    model: SEDModel
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
    if model.z_fixed is not None:
        lines.append(f"  Redshift:    {model.z_fixed:.4f} (fixed)")
    else:
        lines.append("  Redshift:    free")

    # Dtype and precompute fast-path eligibility
    lines.append(f"  Dtype:       {model._forward_dtype}")
    approx = getattr(model, "_approx", {}) or {}
    fast_parts: list[str] = []
    if approx.get("wave_precomp"):
        fast_parts.append("photometry LUT")
    if approx.get("spectrum_precomp"):
        fast_parts.append("spectrum LUT")
    lines.append(
        f"  Fast path:   {', '.join(fast_parts) if fast_parts else 'exact (approx=None)'}"
    )

    # Enabled components
    components: list[str] = []
    if model.spec.nebular_mode != "off":
        components.append(f"nebular={model.spec.nebular_mode}")
    if model._dust_emission_model:
        components.append(f"dust_emission={model._dust_emission_model}")
    if model._agn_model:
        components.append(f"agn={model._agn_model}")
    if model._uses_igm:
        components.append("igm")
    if model._uses_radio:
        components.append("radio")
    if model._uses_xray:
        components.append("xray")
    if model._uses_shock:
        components.append("shock")
    if components:
        lines.append(f"  Components:  {', '.join(components)}")

    # Pipeline chain (the SEDComponent contract: order, publishes, requires).
    # Skipped quietly on any failure so summary() never blocks a debug session.
    try:
        chain = model._build_component_chain()
    except Exception as exc:
        lines.append(f"  Pipeline:    <unable to build chain: {type(exc).__name__}>")
        chain = []

    if chain:
        lines.append("")
        lines.append(f"  Pipeline ({len(chain)} components):")
        for i, c in enumerate(chain, 1):
            cfg = _component_config_summary(c)
            head = f"    {i}. {getattr(c, 'name', type(c).__name__):14s}"
            lines.append(f"{head} {cfg}".rstrip())
            reqs = [k.name for k in _safe_call(c, "requires")]
            opts = [k.name for k in _safe_call(c, "requires_optional")]
            pubs = [k.name for k in _safe_call(c, "publishes")]
            if reqs:
                lines.append(f"       reads:     {', '.join(reqs)}")
            if opts:
                lines.append(f"       reads*:    {', '.join(opts)}  (optional, with fallback)")
            if pubs:
                lines.append(f"       publishes: {', '.join(pubs)}")

    # Dimensionality
    n_free = model.spec.n_free
    n_grid = model.n_grid if model.uses_stochastic_sfh else 0
    lines.append("")
    lines.append(f"  Parameters:  {n_free} free" + (f" + {n_grid} latent (ξ)" if n_grid else ""))

    lines.append(sep)
    return "\n".join(lines)


def _safe_call(component, attr: str) -> tuple:
    """Call ``component.<attr>()`` if it exists; return ``()`` otherwise.

    Used by :func:`summary` to render publishes / requires / requires_optional
    without crashing on partial components that don't declare all three.
    """
    fn = getattr(component, attr, None)
    if fn is None:
        return ()
    try:
        return tuple(fn())
    except Exception:
        return ()


def _component_config_summary(c) -> str:
    """Render a short ``[k=v, k=v]`` config snippet for one chain component.

    Picks a small whitelist of well-known config keys. Returns an empty
    string if the component has no config or none of the keys are set.
    """
    cfg = getattr(c, "config", None)
    if cfg is None:
        return ""
    candidate_keys = (
        "sfh_model",
        "metallicity_model",
        "n_grid",
        "backend",
        "model",
        "law",
        "law_bc",
        "law_diff",
        "emission_model",
    )
    parts = []
    for k in candidate_keys:
        v = getattr(cfg, k, None)
        if v is None:
            continue
        # Agreeing screens read back as ONE law, matching what the grammar
        # accepts and what to_groups() emits. Rendering
        # "law_bc=calzetti, law_diff=calzetti" for a config written as
        # law='calzetti' shows a key the user did not write, and spends two of
        # the three displayed slots saying it twice.
        if k == "law_bc" and getattr(cfg, "law_diff", None) == v:
            parts.append(f"law={v}")
            continue
        if k == "law_diff" and getattr(cfg, "law_bc", None) == v:
            continue  # already shown as the shared law
        parts.append(f"{k}={v}")
    if not parts:
        return ""
    return f"[{', '.join(parts[:3])}]"
