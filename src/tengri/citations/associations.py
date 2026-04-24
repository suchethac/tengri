"""Static associations between tengri model configuration and citation keys.

Module layout:

* ``CORE_CITATIONS``          always-applicable citations for any tengri run.
* ``DUST_LAW_CITATIONS``      map of dust-law name → citation keys.
* ``NEBULAR_BACKEND_CITATIONS`` map of nebular-backend name → citation keys.
* ``IGM_CITATIONS``           map of IGM-model name → citation keys.
* ``BACKEND_CITATIONS``       map of inference-backend name → citation keys.
* ``FUNCTION_CITATIONS``      optional map of ``"module.function"`` → citation
  keys, populated by the :func:`cites` decorator or by direct registration.

Downstream logic (:func:`tengri.collect_citations`) reads these tables to
assemble a per-Galaxy citation set. If you add a new dust law, nebular
backend, or inference path, add an entry here so the citation machinery
can surface the right paper.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


# Citations that apply to every tengri run.
CORE_CITATIONS: list[str] = ["tengri", "jax", "dsps"]


# Dust attenuation laws. Keys match tengri.config.settings.DustConfig.law_*
# and DustConfig.model values.
DUST_LAW_CITATIONS: dict[str, list[str]] = {
    "calzetti": ["calzetti2000"],
    "power_law": [],
    "smc": [],
    "mw": [],
    "cf00": ["charlot_fall2000"],
}

# Additional citation(s) triggered by the dust *model* wrapper, independent
# of the per-component law. Two-component = Charlot & Fall 2000.
DUST_MODEL_CITATIONS: dict[str, list[str]] = {
    "two_component": ["charlot_fall2000"],
    "single_screen": [],
}

# Nebular-emission backends. Keys match tengri.config.settings.NebularConfig.backend.
NEBULAR_BACKEND_CITATIONS: dict[str, list[str]] = {
    "cue": ["cue"],
    "baked_in": [],
    "off": [],
    None: [],
}

# IGM-attenuation models.
IGM_CITATIONS: dict[str, list[str]] = {
    "inoue": ["inoue2014"],
    "inoue2014": ["inoue2014"],
    "madau": ["madau1995"],
    "madau1995": ["madau1995"],
    None: [],
}

# Inference backends (values passed to ``Fitter.run(backend=...)``).
BACKEND_CITATIONS: dict[str, list[str]] = {
    "map": [],
    "laplace": [],
    "pathfinder": [],
    "mcmc_nuts": ["blackjax"],
    "nuts": ["blackjax"],
    "mcmc_raytrace": [],
    "raytrace": [],
    "evidence": [],
    "vi": ["nifty", "ift"],
    "vi_native": ["nifty", "ift"],
}


# Optional function-level annotations. Populated by the @cites decorator.
# Key: "module.qualname" string. Value: list of citation registry keys.
FUNCTION_CITATIONS: dict[str, list[str]] = {}


def cites(*keys: str) -> Callable[[F], F]:
    """Decorator that records citations for a function or class.

    Use it to tie a scientific module to its upstream paper(s) at definition
    time — the citation is then visible to :func:`tengri.collect_citations`
    whenever that function is called on the object graph of a Galaxy.

    Parameters
    ----------
    *keys : str
        Citation registry keys (see ``references.bib``).

    Examples
    --------
    >>> from tengri.citations import cites
    >>> @cites("calzetti2000")
    ... def calzetti_law(wave, av):
    ...     '''Calzetti et al. (2000) starburst attenuation law.'''
    ...     ...

    The decorator is transparent — it returns ``func`` unchanged — and
    simply registers the function's fully-qualified name in
    ``FUNCTION_CITATIONS``.
    """

    def _decorate(func: F) -> F:
        qual = f"{func.__module__}.{func.__qualname__}"
        FUNCTION_CITATIONS.setdefault(qual, []).extend(keys)
        # Also expose on the function object for introspection.
        existing = list(getattr(func, "_tengri_cites", ()))
        existing.extend(keys)
        import contextlib

        with contextlib.suppress(AttributeError, TypeError):
            # Some callables (builtins, slot wrappers) reject attributes.
            func._tengri_cites = tuple(existing)  # type: ignore[attr-defined]
        return func

    return _decorate


def register_function_citations(qualname: str, keys: list[str]) -> None:
    """Register citations for a function by fully-qualified name.

    Equivalent to ``@cites`` but usable without touching the function
    definition (e.g. annotating third-party code or JIT-wrapped callables).
    """
    FUNCTION_CITATIONS.setdefault(qualname, []).extend(keys)


__all__ = [
    "BACKEND_CITATIONS",
    "CORE_CITATIONS",
    "DUST_LAW_CITATIONS",
    "DUST_MODEL_CITATIONS",
    "FUNCTION_CITATIONS",
    "IGM_CITATIONS",
    "NEBULAR_BACKEND_CITATIONS",
    "cites",
    "register_function_citations",
]
