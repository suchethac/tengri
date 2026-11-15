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
    feature_precomp : PrecomputeState
        The ``FeaturePrecomp`` top-up of ``approx='auto'``: engaged when live on the model,
        declined with the recorded reason when its table refused.
    declines : tuple[tuple[str, str], ...]
        Every ``Decline`` the model recorded, as ``("<name>.<slot>", reason)`` pairs in key order,
        so a gate that records through ``record_decline`` is readable without a field of its own.
    """

    energy_balance_lut: "PrecomputeState"
    dust_band_response: "PrecomputeState"
    emitter_term_responses: dict[str, "PrecomputeState"]
    observed_facts: dict[str, Any]
    feature_precomp: "PrecomputeState" = PrecomputeState(state="never_attempted")
    declines: tuple[tuple[str, str], ...] = ()

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

        # FeaturePrecomp top-up of approx='auto'; only a refusal is worth a line
        if self.feature_precomp.state == "declined":
            lines.append("\n• FeaturePrecomp top-up: DECLINED")
            lines.append(f"  Reason: {self.feature_precomp.reason}")

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

    def state_of(self, what: str) -> "PrecomputeState":
        """Return the engagement state of one mechanism by its key.

        Parameters
        ----------
        what : str
            ``"energy_balance_lut"``, ``"dust_band_response"`` or ``"dust_emission"`` (the same
            mechanism), an emitter name in ``emitter_term_responses``, or ``"feature_precomp"``.

        Returns
        -------
        PrecomputeState
            The state, with the recorded reason when the mechanism declined.

        Raises
        ------
        KeyError
            If ``what`` is not a known mechanism; the message lists the known keys.

        """
        if what == "energy_balance_lut":
            return self.energy_balance_lut
        if what in ("dust_band_response", "dust_emission"):
            return self.dust_band_response
        if what in self.emitter_term_responses:
            return self.emitter_term_responses[what]
        if what == "feature_precomp":
            return self.feature_precomp
        recorded = dict(self.declines)
        if what in recorded:
            return PrecomputeState(state="declined", reason=recorded[what])
        known = ["dust_band_response", "dust_emission", "energy_balance_lut", "feature_precomp"]
        known += sorted(self.emitter_term_responses) + sorted(recorded)
        raise KeyError(f"unknown precompute mechanism {what!r}; known: {known}")


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

    # The FeaturePrecomp top-up: live on the model, or the refusal its gate recorded.
    refusal = _recorded_reason(model, "approx", "feature_precomp")
    if refusal is not None:
        feature_state_obj = PrecomputeState(state="declined", reason=refusal)
    elif getattr(getattr(model, "approx", None), "feature_precomp", False):
        feature_state_obj = PrecomputeState(state="engaged")
    else:
        feature_state_obj = PrecomputeState(state="never_attempted")

    # Gather observed facts that may explain decline decisions
    observed_facts = _gather_observed_facts(model)

    return PrecomputeEngagementReport(
        energy_balance_lut=eb_state_obj,
        dust_band_response=dust_state_obj,
        emitter_term_responses=emitter_caches,
        observed_facts=observed_facts,
        feature_precomp=feature_state_obj,
        declines=_recorded_declines(model),
    )


def _memo_entry(model: Any, name: str, slot: str) -> Any:
    """Return the band-table memo entry of one component slot, or None when absent."""
    return getattr(model, "_band_tables", {}).get(name, {}).get(slot)


def _recorded_declines(model: Any) -> tuple[tuple[str, str], ...]:
    """Return every ``Decline`` in the model's memo as ``("<name>.<slot>", reason)`` pairs."""
    memo = getattr(model, "_band_tables", {})
    return tuple(
        (f"{name}.{slot}", entry.reason)
        for name in sorted(memo)
        for slot, entry in sorted(memo[name].items())
        if isinstance(entry, Decline)
    )


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


def engagement_reports(
    target: object, *, regime: str | None = None
) -> tuple[PrecomputeEngagementReport, ...]:
    """Return the engagement report of each model the target evaluates, in galaxy order.

    Parameters
    ----------
    target : SEDModel, ForwardModel, Fitter, CatalogFitter, Posterior or CatalogPosterior
        What the report is about. A catalog target gives one report per galaxy.
    regime : {"single", "catalog"} or None
        Optional kind check. ``None`` accepts any target.

    Returns
    -------
    tuple of PrecomputeEngagementReport
        One report per galaxy; a single-model target gives a one-element tuple.

    Raises
    ------
    TypeError
        If the target is not a recognised kind.
    ValueError
        If ``regime`` is not a known value, does not match the target, or a posterior carries
        no model reference.
    """
    kind, models = _targets(target)
    if regime is not None:
        if regime not in ("single", "catalog"):
            raise ValueError(f"regime must be None, 'single' or 'catalog', got {regime!r}")
        if regime != kind:
            raise ValueError(
                f"regime mismatch: {type(target).__name__} is a {kind} target, "
                f"but the check asked for a {regime} one"
            )
    return tuple(precompute_engagement_report(_evaluated_model(model)) for model in models)


def _targets(target: object) -> tuple[str, tuple[Any, ...]]:
    """Return the regime of a target and the models it covers, in galaxy order."""
    from tengri.forward.forward_model import ForwardModel
    from tengri.forward.sed_model import SEDModel
    from tengri.inference.catalog_fitter import CatalogPosterior, _CatalogFitterOriginal
    from tengri.inference.fitter import Fitter
    from tengri.inference.posterior import Posterior

    if isinstance(target, CatalogPosterior):
        return "catalog", tuple(_posterior_model(post) for post in target.posteriors)
    if isinstance(target, _CatalogFitterOriginal):
        return "catalog", (target.model,)
    if isinstance(target, Posterior):
        return "single", (_posterior_model(target),)
    if isinstance(target, Fitter):
        return "single", (target.model,)
    if isinstance(target, (SEDModel, ForwardModel)):
        return "single", (target,)
    raise TypeError(f"no engagement target for {type(target).__name__}")


def _posterior_model(posterior: Any) -> Any:
    """Return the model a posterior was fitted with, or refuse when it carries none."""
    if posterior._model is None:
        raise ValueError(
            "posterior has no model reference, so its engagement cannot be read; "
            "Fitter.run() attaches one"
        )
    return posterior._model


def _evaluated_model(model: Any) -> Any:
    """Return the model the fit actually evaluates.

    Today this is the model itself. It switches to the runtime-promoted model when #3531 lands;
    that promotion has no API yet, so nothing is invented here.
    """
    return model
