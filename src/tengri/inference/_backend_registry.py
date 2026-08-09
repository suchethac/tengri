# SPDX-License-Identifier: BSD-3-Clause
"""Inference backend registry — single source of truth for fitter.run dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BackendEntry:
    """Registry entry for an inference backend.

    Parameters
    ----------
    name : str
        Canonical method name (e.g., ``"map"``, ``"mcmc_nuts"``).
    runner : Callable
        Backend entry point. Signature depends on ``legacy_fitter``:

        - ``legacy_fitter=True``  → ``runner(fitter, *, key, **kwargs)``
        - ``legacy_fitter=False`` → ``runner(context, *, key, **kwargs)``

        where ``context`` is an :class:`InferenceContext`.
    tier : str
        ``"primary"`` for promoted methods, ``"experimental"`` for backends
        that work but are not yet validated, ``"broken"`` for backends known
        to produce wrong answers or crash. ``"broken"`` is hidden from the
        default :func:`tengri.list_inference_methods` listing and refused by
        ``Fitter.run`` unless the caller passes ``allow_unvalidated=True``.
    short_doc : str
        Brief description.
    requires : tuple[str, ...]
        Optional dependency import names (e.g. ``("blackjax",)``).
    legacy_fitter : bool
        If ``True`` (default), ``Fitter.run`` passes the full Fitter to
        the runner. Set to ``False`` for backends migrated to the
        :class:`InferenceContext` Protocol (ADR-0010).

        Every in-tree backend is migrated — ``test_backend_conformance``
        asserts ``legacy_fitter is False`` for all of them, so the ``True``
        branch in ``Fitter.run`` is reachable only from an out-of-tree
        backend that registers without passing the flag. That is what the
        default is *for*, which is why the flag outlived the migration it
        was named after. It is a compatibility shim for third-party
        backends, not unfinished work.
    accepts_precondition : bool
        Whether the runner understands ``precondition=`` — metric preconditioning
        of the standardized latent space (see
        :mod:`tengri.inference.preconditioning`). True for the Hamiltonian samplers,
        whose integrator has a metric to whiten. Declared here rather than inferred,
        so dispatch can refuse the kwarg at one seam instead of letting it raise
        ``TypeError`` deep inside a backend — and so the ``mcmc`` auto-dispatcher
        can ask the registry about whichever backend it picked rather than naming
        one in an ``if``.
    """

    name: str
    runner: Callable
    tier: str = "experimental"  # "primary" | "experimental"
    short_doc: str = ""
    requires: tuple[str, ...] = field(default_factory=tuple)  # optional dep names
    legacy_fitter: bool = True
    accepts_precondition: bool = False
    # Predicate called with whatever ``runner`` receives (Fitter or InferenceContext).
    # Returns True if this backend can run for the given target's spec/dims/dtypes.
    # Default ``None`` means "no compatibility constraint" (always usable).
    is_compatible: Callable[[Any], bool] | None = None


#: The tiers a backend may declare.
#:
#: ``"broken"`` is not a softer ``"experimental"``. Experimental means "works,
#: not yet validated"; broken means the backend's own ``short_doc`` says it
#: returns wrong answers (``[POOR MIXING]``) or crashes the process
#: (``[UNSTABLE]``). Five backends carried such a warning while sitting in the
#: experimental tier, indistinguishable from ones that work (#1287).
TIERS: frozenset[str] = frozenset({"primary", "experimental", "broken"})

#: The one default inference method, shared by every surface that starts a fit.
#:
#: Five surfaces used to answer this question differently (#1289):
#:
#:     ForwardModel.fit   'vi'                  <- canonical
#:     Fitter.run         'vi_nonlinear_fast'   <- the engine ForwardModel.fit calls
#:     SEDModel.fit       'vi'                  <- deprecated
#:     fit_batch          'vi'
#:     Galaxy.fit         'map'                 <- and a different kwarg name
#:
#: So ``forward.fit(d, n)`` and ``Fitter(forward, d, n).run()`` -- same objects,
#: same data -- ran different backends with no warning. ``'vi'`` and
#: ``'vi_nonlinear_fast'`` are in fact the same geoVI algorithm (both pass
#: ``sample_mode="nonlinear_resample"``); they differ only in Python logging,
#: so aligning them is posterior-preserving.
#:
#: ``Galaxy.fit`` deliberately keeps ``"map"`` -- see the note in its docstring.
DEFAULT_METHOD: str = "vi"

_BACKENDS: dict[str, BackendEntry] = {}


def register_backend(
    name: str,
    *,
    tier: str = "experimental",
    short_doc: str = "",
    requires: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    legacy_fitter: bool = True,
    is_compatible: Callable[[Any], bool] | None = None,
    accepts_precondition: bool = False,
):
    """Decorator to register an inference backend.

    Parameters
    ----------
    name : str
        Canonical method name (e.g., "map", "mcmc_nuts").
    tier : str
        One of :data:`TIERS`. ``"primary"`` for promoted methods,
        ``"experimental"`` for working-but-unvalidated ones, ``"broken"``
        for backends known to return wrong answers or crash.
    short_doc : str
        Brief description of the method.
    requires : tuple[str, ...]
        Optional dependency import names (e.g., ("blackjax",)).
    aliases : tuple[str, ...]
        Additional names that map to this backend.
    accepts_precondition : bool
        Declare that the runner takes ``precondition=``. See
        :class:`BackendEntry`. Kept honest against the runner's real signature by
        ``tests/contract/test_preconditioning_capability.py``.

    Raises
    ------
    ValueError
        If ``tier`` is not a recognized tier. A typo would otherwise create a
        silent third tier that no filter matches.
    """
    if tier not in TIERS:
        raise ValueError(
            f"register_backend({name!r}): unknown tier {tier!r}. Valid tiers: {sorted(TIERS)}."
        )

    def deco(fn):
        entry = BackendEntry(
            name=name,
            runner=fn,
            tier=tier,
            short_doc=short_doc,
            requires=requires,
            legacy_fitter=legacy_fitter,
            is_compatible=is_compatible,
            accepts_precondition=accepts_precondition,
        )
        _BACKENDS[name] = entry
        for a in aliases:
            _BACKENDS[a] = entry
        return fn

    return deco


def get_backend(name: str) -> BackendEntry:
    """Retrieve a backend by name.

    Parameters
    ----------
    name : str
        Method name.

    Returns
    -------
    BackendEntry
        The backend registry entry.

    Raises
    ------
    ValueError
        If the method is not registered.
    """
    if name not in _BACKENDS:
        # Group available methods by tier so the error is digestible at
        # 19 backends. Suggest the discovery API rather than dumping a flat
        # list of every experimental sampler.
        primary = sorted({e.name for e in _BACKENDS.values() if e.tier == "primary"})
        raise ValueError(
            f"Unknown inference method '{name}'.  "
            f"Recommended (tier=primary): {primary}.  "
            "Run `tengri.list_inference_methods()` for the full list including "
            "experimental backends."
        )
    return _BACKENDS[name]


def check_requires(entry: BackendEntry) -> None:
    """Verify the backend's optional dependencies are importable.

    Raises a friendly ImportError before the runner crashes deep in a
    third-party package. Called by ``Fitter.run`` just before dispatch.

    Parameters
    ----------
    entry : BackendEntry
        The backend whose ``requires`` tuple should be checked.

    Raises
    ------
    ImportError
        If any required dependency cannot be imported. The error message
        names the offending package and gives the recommended pip extra.
    """
    import importlib

    _PIP_EXTRA: dict[str, str] = {
        "blackjax": 'pip install "tengri[blackjax]"  (or:  pip install blackjax)',
        "nifty8": "pip install nifty8.re",
        "optax": "pip install optax",
        "jaxopt": "pip install jaxopt",
        "dynesty": "pip install dynesty",
    }
    for pkg in entry.requires:
        try:
            importlib.import_module(pkg)
        except ImportError as exc:
            hint = _PIP_EXTRA.get(pkg, f"pip install {pkg}")
            raise ImportError(
                f"Inference method '{entry.name}' requires {pkg!r}, "
                f"which is not installed.  Install it with:\n    {hint}"
            ) from exc


def check_usable(entry: BackendEntry, *, allow_unvalidated: bool = False) -> None:
    """Refuse to run a backend that is known to give wrong answers (#1287).

    Five backends declared ``[POOR MIXING]`` or ``[UNSTABLE]`` in their own
    ``short_doc`` while sitting at ``tier="experimental"`` — the same tier as
    backends that work. A user who picked ``mcmc_ghmc`` because it is "fast
    (cold ~17s)" got R-hat ~ 2.5-3.1 and no runtime signal that the chains
    had not converged.

    Wrongness that only a doc string mentions is wrongness that ships. This
    makes the caller say out loud that they accept it.

    Parameters
    ----------
    entry : BackendEntry
        The backend about to be dispatched.
    allow_unvalidated : bool, optional
        Escape hatch for benchmarking and backend development. Default False.

    Raises
    ------
    BackendError
        If ``entry.tier == "broken"`` and ``allow_unvalidated`` is False. The
        message carries the backend's own diagnosis verbatim.
    """
    if entry.tier != "broken" or allow_unvalidated:
        return

    from tengri.config.exceptions import BackendError

    primary = sorted({e.name for e in _BACKENDS.values() if e.tier == "primary"})
    raise BackendError(
        f"Inference method '{entry.name}' is registered as tier='broken' and "
        f"will not run by default.\n\n"
        f"  {entry.short_doc}\n\n"
        f"Working alternatives (tier=primary): {primary}.\n"
        f"To run it anyway -- for benchmarking or backend development, not for "
        f"science -- pass allow_unvalidated=True."
    )


def refuse_if_broken(method: str, *, allow_unvalidated: bool = False) -> None:
    """Apply the :func:`check_usable` tier gate to a method *name* (#1394).

    :func:`check_usable` takes a :class:`BackendEntry`, so every caller that
    holds only a method string has to look the entry up first. Two batched
    entry points never did — ``CatalogFitter.run`` and ``PopulationFitter.run``
    validated the name with
    :func:`~tengri.inference.fitter.resolve_method` and dispatched straight
    into the backend module, so a ``tier="broken"`` method ran with no refusal
    and no ``allow_unvalidated`` prompt. Both then *defaulted* to one.

    A gate that every path must remember to call is a gate that some path will
    forget. This is the name-keyed form so the lookup is not the caller's job.

    Unknown names return silently: name validation belongs to
    ``resolve_method``, and a hierarchical method can legitimately have no
    registry entry — ``evi_nifty`` is dispatched by
    :func:`~tengri.inference.hierarchical.PopulationFitter` but registered
    nowhere. Raising here would turn a missing registration into a broken
    user call.

    (This paragraph long cited ``vi_nonlinear`` as the unregistered example.
    It is registered — a ``tier="primary"`` alias of ``vi`` — so the
    justification named a case that never reaches this branch; ``evi_nifty``
    is the one that does.)

    Parameters
    ----------
    method : str
        Canonical method name, already resolved.
    allow_unvalidated : bool, optional
        Escape hatch, forwarded to :func:`check_usable`. Default False.

    Raises
    ------
    BackendError
        If ``method`` is registered ``tier="broken"`` and ``allow_unvalidated``
        is False.

    Notes
    -----
    Not JIT-compatible; a Python-level dispatch guard called once per fit.
    """
    entry = _BACKENDS.get(method)
    if entry is not None:
        check_usable(entry, allow_unvalidated=allow_unvalidated)


#: Capability-gated keyword arguments: kwarg name -> :class:`BackendEntry` field.
#:
#: A kwarg belongs here when it names a *sampler capability* rather than a tuning
#: knob — something a backend either implements or cannot. Gating at one seam keeps
#: the option out of the dispatcher's control flow: ``mcmc``'s auto-pick asks the
#: registry about the backend it chose instead of naming one in an ``if``.
_CAPABILITY_FIELDS: dict[str, str] = {"precondition": "accepts_precondition"}


def check_capabilities(entry: BackendEntry, kwargs: dict) -> None:
    """Refuse a capability kwarg the backend does not implement.

    Without this the kwarg travels until something rejects it: ``run_nifty_vi`` and
    ``run_map`` take no ``**kwargs``, so ``precondition=True`` on ``method='vi'``
    surfaces as ``TypeError: run_nifty_vi() got an unexpected keyword argument`` from
    inside a backend the caller never named.

    ``ValueError``, not ``TypeError``: the caller never called that runner, so this is
    an unsupported *combination* of method and option rather than a malformed call.
    ``_mcmc_auto_pick`` already made that choice for the raytrace branch (#1359); this
    extends the same answer to every backend instead of leaving the rest on a deep
    ``TypeError``.

    Only truthy values are refused. ``precondition=False`` asks for the behavior every
    backend already has, so rejecting it would be pedantic.

    Parameters
    ----------
    entry : BackendEntry
        The backend about to be dispatched.
    kwargs : dict
        Keyword arguments destined for ``entry.runner``.

    Raises
    ------
    ValueError
        If ``kwargs`` carries a truthy capability the backend does not declare.
    """
    for kwarg, field_name in _CAPABILITY_FIELDS.items():
        if not kwargs.get(kwarg):
            continue
        if getattr(entry, field_name):
            continue
        capable = sorted({e.name for e in all_backends() if getattr(e, field_name)})
        raise ValueError(
            f"Inference method '{entry.name}' does not support {kwarg}={kwargs[kwarg]!r}. "
            f"Backends that do: {capable}. "
            f"Drop the argument, or choose one of those methods."
        )


def check_unknown_kwargs(
    entry: BackendEntry, kwargs: dict, also_accepted: frozenset[str] = frozenset()
) -> None:
    """Refuse a kwarg the backend's runner does not declare.

    :func:`check_capabilities` gives this answer for the handful of *declared
    capability* kwargs. This gives it for every other unknown name, which
    otherwise traveled all the way into the runner and surfaced as
    ``TypeError: run_map() got an unexpected keyword argument 'lines'`` --
    naming a function the caller never mentioned, inside a backend they did
    not choose (#1469).

    The check sits at the dispatch seam, so every fit surface that forwards
    ``**kwargs`` -- ``SEDModel.fit``, ``ForwardModel.fit``, ``Catalog.fit`` --
    is covered by one rule rather than each growing its own validation.

    Parameters
    ----------
    entry : BackendEntry
        The backend about to be dispatched.
    kwargs : dict
        Keyword arguments destined for ``entry.runner``.
    also_accepted : frozenset of str, optional
        Names the calling *surface* accepts but routes elsewhere -- the
        ``Fitter.__init__`` parameters ``split_fitter_kwargs`` sends to
        construction. Used for suggestions only, never to widen the
        rejection, since a correctly spelled one never reaches this check.

    Raises
    ------
    TypeError
        If ``kwargs`` carries a name the runner cannot accept.

        This raised ``ValueError`` when the check landed, on the grounds that
        "the caller never called that runner". The concrete harm that argued
        against ``TypeError`` was the *message* — ``run_map() got an
        unexpected keyword argument 'lines'`` names an internal function the
        caller never chose (#1469) — and rewriting the message already fixed
        that; the exception *type* was never what caused the confusion.

        Meanwhile the type is what callers catch, and the same user mistake
        raises ``TypeError`` everywhere else: from Python itself, and from
        ``SEDModel.build`` for a kwarg it does not take. Two types for one
        mistake is the inconsistency, so this is ``TypeError`` and #1378's
        regression test — which pins exactly that — passes again.

    Notes
    -----
    Runners that declare ``**kwargs`` are skipped -- they accept anything, so
    there is nothing to reject. Capability names are let through because
    :func:`check_capabilities` has already vetted them.

    The accepted list is read off the live signature, so it cannot drift from
    what the backend actually takes. Advice that its own caller would refuse
    is the failure in #1576.
    """
    import difflib
    import inspect

    try:
        params = inspect.signature(entry.runner).parameters
    except (TypeError, ValueError):  # pragma: no cover - unintrospectable runner
        return

    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return

    # The first parameter is the dispatch target (``context`` or ``fitter``,
    # depending on ``legacy_fitter``) and is always passed positionally.
    names = list(params)
    accepted = {
        name for name, p in params.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    accepted -= {names[0]} if names else set()
    accepted -= {"key", "init_from"}
    accepted |= set(_CAPABILITY_FIELDS)

    unknown = sorted(k for k in kwargs if k not in accepted)
    if not unknown:
        return

    offered = sorted(accepted - set(_CAPABILITY_FIELDS))
    # Suggest against everything the *surface* takes, not just this runner.
    # Constructor-routed options are documented fit() kwargs, so a typo'd one
    # must be correctable even though it is not a runner parameter -- without
    # this, ``calibration_marginalze`` was rejected with a list that could not
    # contain ``calibration_marginalize``, and no correction offered.
    #
    # Suggestions only: the rejection set above is untouched. A correctly
    # spelled constructor kwarg is routed away by ``split_fitter_kwargs`` and
    # never arrives here, so accepting one would only move its failure deeper
    # into the backend.
    suggestable = sorted(set(offered) | set(also_accepted))
    hints = []
    for name in unknown:
        close = difflib.get_close_matches(name, suggestable, n=1, cutoff=0.6)
        if close:
            hints.append(f"{name} -> {close[0]}")
    hint_str = f" Did you mean: {', '.join(hints)}?" if hints else ""

    # TypeError, not ValueError: this is "the callable does not take that
    # keyword", which is what Python raises for the same mistake, and what
    # the rest of tengri already raises for a kwarg `SEDModel.build` refuses.
    # #1378's regression test pins it — a misspelled fit option must fail the
    # same way whether the rejection comes from Python or from this check.
    raise TypeError(
        f"Inference method '{entry.name}' does not accept {unknown}. "
        f"It takes: {offered}.{hint_str} "
        "Arguments that are not fit options belong to the model or the data, "
        "not to fit() -- emission lines, for example, are supplied when the "
        "problem is built (Data(lines=...) for one galaxy, "
        "Catalog(line_cols=...) for a catalog)."
    )


def lookup_backend(name: str) -> BackendEntry | None:
    """Return the entry ``name`` dispatches to, or ``None`` if unregistered.

    Resolves aliases (``"vi_nonlinear"`` -> the ``"vi"`` entry) and every
    tier, including ``"broken"``. This is the *identification* question —
    "what does this name run?" — as opposed to :func:`all_backends`, which
    answers the *curation* question and is filtered for presentation.

    Answering identification through a curated listing is what made
    ``describe_inference_method`` report six dispatchable names as unknown
    (#1560): five ``tier="broken"`` backends, hidden from the menu by
    design, plus ``"vi_nonlinear"``, a ``tier="primary"`` alias that the
    public ``fit()`` docstring teaches.

    Parameters
    ----------
    name : str
        A method name or alias, as passed to ``fit(method=...)``.

    Returns
    -------
    BackendEntry or None
        The backend, or ``None`` if no such name is registered.

    Notes
    -----
    Not JIT-compatible; a Python-level registry lookup.
    """
    return _BACKENDS.get(name)


def all_backends(*, include_broken: bool = True) -> list[BackendEntry]:
    """Return all registered backends, deduplicated and sorted.

    Parameters
    ----------
    include_broken : bool, optional
        Include ``tier="broken"`` entries. Default True, so internal callers
        that need the complete registry (dispatch, conformance tests) keep
        seeing everything; the user-facing listing opts out.

    Returns
    -------
    list[BackendEntry]
        Backends sorted by (tier != "primary", name).
    """
    seen, out = set(), []
    for entry in _BACKENDS.values():
        if id(entry) in seen:
            continue
        seen.add(id(entry))
        if not include_broken and entry.tier == "broken":
            continue
        out.append(entry)
    return sorted(out, key=lambda e: (e.tier != "primary", e.name))
