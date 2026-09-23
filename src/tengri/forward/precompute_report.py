# SPDX-License-Identifier: BSD-3-Clause
"""
Engagement report for build-time precompute mechanisms.

Provides introspection into which precompute optimizations engaged during
SEDModel construction. Each mechanism can be in three states: engaged
(caching the result), declined (attempted but conditions not met), or
never attempted.

References
----------
.. [1] Silent precompute forfeits observable. GitHub Issue #2485.
"""

from typing import Any, NamedTuple


class PrecomputeEngagementReport(NamedTuple):
    """Structured report of precompute mechanism engagement.

    Attributes
    ----------
    energy_balance_lut : PrecomputeState
        Energy-balance luminosity lookup table status.
    dust_band_response : PrecomputeState
        Dust emission band-integrated response status.
    emitter_term_responses : dict[str, PrecomputeState]
        Per-emitter term-response cache status (radio, xray, etc.).
    observed_facts : dict[str, str | list]
        Context for why mechanisms declined (e.g., free parameters that
        trigger gates).
    """

    energy_balance_lut: "PrecomputeState"
    dust_band_response: "PrecomputeState"
    emitter_term_responses: dict[str, "PrecomputeState"]
    observed_facts: dict[str, Any]

    def summary_text(self) -> str:
        """Human-readable summary of engagement.

        Returns
        -------
        str
            Narrative description of which mechanisms engaged and (where
            determinable) why others declined.
        """
        lines = ["Precompute Engagement Summary", "=" * 27]

        # Energy balance LUT
        eb_state = self.energy_balance_lut.state
        lines.append(f"\n• Energy-balance LUT: {eb_state.upper()}")
        if eb_state == "declined" and self.energy_balance_lut.reason:
            lines.append(f"  Reason: {self.energy_balance_lut.reason}")

        # Dust band response
        db_state = self.dust_band_response.state
        lines.append(f"• Dust band response: {db_state.upper()}")
        if db_state == "declined" and self.dust_band_response.reason:
            lines.append(f"  Reason: {self.dust_band_response.reason}")

        # Emitter term responses
        if self.emitter_term_responses:
            lines.append("\n• Emitter term responses:")
            for name, state_obj in sorted(self.emitter_term_responses.items()):
                st = state_obj.state
                lines.append(f"  - {name}: {st.upper()}")
                if st == "declined" and state_obj.reason:
                    lines.append(f"    Reason: {state_obj.reason}")

        # Observed facts
        if self.observed_facts:
            lines.append("\n• Observed facts:")
            for key, value in sorted(self.observed_facts.items()):
                if isinstance(value, list):
                    if value:
                        lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class PrecomputeState(NamedTuple):
    """State of a single precompute mechanism.

    Attributes
    ----------
    state : str
        One of: 'engaged' (object stored), 'declined' (None stored),
        'never_attempted' (attribute unset).
    reason : str | None
        Human-readable explanation of why the mechanism declined (if state
        is 'declined'). None for 'engaged' or 'never_attempted'.
    """

    state: str
    reason: str | None = None


def precompute_engagement_report(model: Any) -> PrecomputeEngagementReport:
    """Inspect a built SEDModel for precompute mechanism engagement.

    Reads the model's cache attributes without recomputing or duplicating
    any gate logic. Distinguishes three states: engaged (object stored),
    declined (None stored), and never attempted (attribute unset).

    Parameters
    ----------
    model : SEDModel
        Built SED model instance.

    Returns
    -------
    PrecomputeEngagementReport
        Structured report including engagement state, reasons for decline,
        and context facts.
    """
    # Check each mechanism's cache attribute
    eb_lut_cached = getattr(model, "_energy_balance_lut_cache", "unset")
    dust_response_cached = getattr(model, "_dust_band_response_cache", "unset")

    # Determine state and reason for energy balance LUT
    if eb_lut_cached == "unset":
        eb_state_obj = PrecomputeState(state="never_attempted")
    elif eb_lut_cached is None:
        reason = _infer_lut_decline_reason(model)
        eb_state_obj = PrecomputeState(state="declined", reason=reason)
    else:
        eb_state_obj = PrecomputeState(state="engaged")

    # Determine state and reason for dust band response
    if dust_response_cached == "unset":
        dust_state_obj = PrecomputeState(state="never_attempted")
    elif dust_response_cached is None:
        reason = _infer_dust_response_decline_reason(model)
        dust_state_obj = PrecomputeState(state="declined", reason=reason)
    else:
        dust_state_obj = PrecomputeState(state="engaged")

    # Discover and inspect emitter term responses by pattern
    emitter_caches = {}
    for attr_name in dir(model):
        if attr_name.startswith("_") and attr_name.endswith("_term_response_cache"):
            # Extract emitter name: _<name>_term_response_cache -> <name>
            emitter_name = attr_name[1 : -len("_term_response_cache")]
            cached = getattr(model, attr_name, "unset")

            if cached == "unset":
                emitter_caches[emitter_name] = PrecomputeState(state="never_attempted")
            elif cached is None:
                reason = _infer_term_response_decline_reason(model, emitter_name)
                emitter_caches[emitter_name] = PrecomputeState(state="declined", reason=reason)
            else:
                emitter_caches[emitter_name] = PrecomputeState(state="engaged")

    # Gather observed facts that may explain decline decisions
    observed_facts = _gather_observed_facts(model)

    return PrecomputeEngagementReport(
        energy_balance_lut=eb_state_obj,
        dust_band_response=dust_state_obj,
        emitter_term_responses=emitter_caches,
        observed_facts=observed_facts,
    )


def _infer_lut_decline_reason(model: Any) -> str | None:
    """Infer reason why energy-balance LUT declined if possible.

    Checks observable conditions that gate the LUT.

    Returns
    -------
    str | None
        Reason string, or None if unable to determine.
    """
    from tengri.components.dust.two_component import DustSEDComponent

    free = set(model.spec.free_params)

    # Check for two-component dust component
    dust = None
    if hasattr(model, "_cached_component_chain") and model._cached_component_chain:
        dust = next(
            (c for c in model._cached_component_chain if isinstance(c, DustSEDComponent)),
            None,
        )

    if dust is None:
        return "no two-component dust attenuation component"

    # Check for unsafe free parameters
    unsafe_free = {p for p in free if p.startswith("dust_") and p not in model._EB_ATTEN_FREE_OK}

    if unsafe_free:
        return f"free dust parameters outside allowlist: {sorted(unsafe_free)}"

    # Check for free redshift on a law that reads it
    if "redshift" in free:
        from tengri.components.dust.laws._registry import law_kwarg_names

        laws_in_play = (dust.config.law_bc, dust.config.law_diff)
        if any(law and "redshift" in law_kwarg_names(law) for law in laws_in_play):
            return "free redshift with law that reads it"

    # Check for WavePrecomp
    if not (hasattr(model, "_approx") and model._approx.get("wave_precomp")):
        return "approx=WavePrecomp() not enabled"

    # Check for alpha_fe_evolving
    if getattr(model.spec, "alpha_fe_evolving", False):
        return "alpha_fe_evolving is True"

    # Check for SSP data
    if model.ssp_data is None:
        return "ssp_data is None"

    return "unknown reason (gate conditions appear satisfied)"


def _infer_dust_response_decline_reason(model: Any) -> str | None:
    """Infer reason why dust emission band response declined if possible.

    Returns
    -------
    str | None
        Reason string, or None if unable to determine.
    """
    free = set(model.spec.free_params)
    free_dust = {p for p in free if p.startswith("dust_")}

    # Check for free shape parameters
    shape_free = bool(free_dust - model._EB_ATTEN_FREE_OK) or ("redshift" in free)
    if shape_free:
        extra_params = free_dust - model._EB_ATTEN_FREE_OK
        return f"free shape parameters: {sorted(extra_params) if extra_params else 'redshift'}"

    # Check for WavePrecomp
    if not (hasattr(model, "_approx") and model._approx.get("wave_precomp")):
        return "approx=WavePrecomp() not enabled"

    # Check for dust emission component
    if model._dust_emission_model is None:
        return "no dust emission component configured"

    return "unknown reason (gate conditions appear satisfied)"


def _infer_term_response_decline_reason(model: Any, emitter_name: str) -> str | None:
    """Infer reason why an emitter term response declined if possible.

    Parameters
    ----------
    model : SEDModel
        Built model instance.
    emitter_name : str
        Name of the emitter (e.g., 'radio', 'xray').

    Returns
    -------
    str | None
        Reason string, or None if unable to determine.
    """
    free = set(model.spec.free_params)
    prefix = f"{emitter_name}_"

    # Check for free emitter parameters
    free_emitter = {p for p in free if p.startswith(prefix)}
    if free_emitter:
        return f"free {emitter_name} parameters: {sorted(free_emitter)}"

    # Check for free redshift
    if "redshift" in free:
        return "free redshift"

    # Check for WavePrecomp
    if not (hasattr(model, "_approx") and model._approx.get("wave_precomp")):
        return "approx=WavePrecomp() not enabled"

    return "unknown reason (gate conditions appear satisfied)"


def _gather_observed_facts(model: Any) -> dict[str, Any]:
    """Collect observable facts that may explain precompute decisions.

    Returns
    -------
    Dict[str, Any]
        Dictionary of observed conditions (e.g., free parameters,
        approximation mode, component types).
    """
    facts = {}

    # Approximation mode
    if hasattr(model, "_approx"):
        facts["wave_precomp_enabled"] = model._approx.get("wave_precomp", False)
        facts["ztable_enabled"] = model._approx.get("ztable", False)

    # Free parameters
    if hasattr(model, "spec"):
        free = model.spec.free_params
        if free:
            facts["free_params"] = list(free)
        else:
            facts["free_params"] = []

    # Component presence
    components = []
    if hasattr(model, "_cached_component_chain") and model._cached_component_chain:
        for c in model._cached_component_chain:
            if hasattr(c, "name"):
                components.append(c.name)
    if components:
        facts["components"] = components

    # Dust configuration
    if hasattr(model, "_dust_emission_model"):
        facts["dust_emission_configured"] = model._dust_emission_model is not None

    # Redshift mode
    if hasattr(model, "spec") and hasattr(model.spec, "get_fixed_values"):
        fixed = model.spec.get_fixed_values()
        if "redshift" in fixed:
            facts["redshift_fixed"] = fixed.get("redshift")
        elif "redshift" in set(model.spec.free_params):
            facts["redshift_free"] = True

    return facts
