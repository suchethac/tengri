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

from tengri.forward.precompute.band_table import BandTable, Decline


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
                if isinstance(value, dict):
                    pairs = ", ".join(f"{k}={v}" for k, v in value.items())
                    lines.append(f"  {key}: {pairs}")
                elif isinstance(value, list):
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

    # Determine state and reason for energy balance LUT
    if eb_lut_cached == "unset":
        eb_state_obj = PrecomputeState(state="never_attempted")
    elif eb_lut_cached is None:
        # The reason is the one the gate recorded in the memo; nothing is inferred from outside.
        eb_state_obj = PrecomputeState(
            state="declined", reason=_recorded_reason(model, "energy_balance", "lut")
        )
    else:
        eb_state_obj = PrecomputeState(state="engaged")

    # Determine state and reason for dust band response. The memo of the band tables is read
    # first; the old attribute answers only for a slot the memo has not seen.
    dust_state = _state_from_memo(_memo_entry(model, "dust_emission", "emission_band_response"))
    if dust_state == "never_attempted":
        dust_state_obj = PrecomputeState(state="never_attempted")
    elif dust_state == "declined":
        dust_state_obj = PrecomputeState(
            state="declined",
            reason=_recorded_reason(model, "dust_emission", "emission_band_response"),
        )
    else:
        dust_state_obj = PrecomputeState(state="engaged")

    # Every emitter whose term response the memo holds, in name order.
    emitter_caches = {}
    for emitter_name in sorted(getattr(model, "_band_tables", {})):
        entry = _memo_entry(model, emitter_name, "term_band_response")
        if entry is None:
            continue
        if _state_from_memo(entry) == "declined":
            emitter_caches[emitter_name] = PrecomputeState(
                state="declined",
                reason=_recorded_reason(model, emitter_name, "term_band_response"),
            )
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


def _memo_entry(model: Any, name: str, slot: str) -> Any:
    """Return the band-table memo entry of one component slot, or None when absent."""
    return getattr(model, "_band_tables", {}).get(name, {}).get(slot)


def _recorded_reason(model: Any, name: str, slot: str) -> str | None:
    """Return the reason the gate recorded for one slot, or None when it holds no ``Decline``."""
    entry = _memo_entry(model, name, slot)
    return entry.reason if isinstance(entry, Decline) else None


def _state_from_memo(entry: Any) -> str:
    """Return the engagement state of a mechanism from its memo entry.

    Parameters
    ----------
    entry : BandTable, Decline or None
        The memo entry of the slot; None when the memo has not seen the slot.

    Returns
    -------
    str
        One of ``"never_attempted"``, ``"declined"`` or ``"engaged"``.
    """
    if entry is None:
        return "never_attempted"
    return "engaged" if isinstance(entry, BandTable) else "declined"


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

    # Declines recorded by the approx='auto' resolvers: table attempted, refused, with a reason.
    declines = {
        slot: entry.reason
        for slot, entry in getattr(model, "_band_tables", {}).get("approx", {}).items()
        if isinstance(entry, Decline)
    }
    if declines:
        facts["approx_declines"] = declines

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

    # IGM fold: the declared mode beside the one the build actually applied
    # (None when no fold was built), so "auto" is never left unresolved (#2445).
    from tengri.forward.sed_model import WavePrecomp

    cfg = getattr(model, "_approx_config_wave", None)
    if cfg is not None:
        declared = cfg.igm_fold
    elif facts.get("wave_precomp_enabled"):
        declared = WavePrecomp.igm_fold
    else:
        declared = None
    chain = getattr(model, "_cached_component_chain", None) or []
    stellar_state = next(
        (getattr(c, "_state", None) for c in chain if getattr(c, "name", None) == "stellar"),
        None,
    )
    facts["igm_fold"] = {
        "declared": declared,
        "resolved": getattr(stellar_state, "igm_fold_resolved", None),
    }

    return facts
