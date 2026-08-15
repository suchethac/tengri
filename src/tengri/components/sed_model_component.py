# SPDX-License-Identifier: BSD-3-Clause
"""SEDModelComponent: astronomer-facing base class for SED physics blocks.

Provides a convenient superclass over the :class:`SEDComponent` Protocol,
with automatic discovery of free parameters, cross-component inputs/outputs,
and a sensible default :meth:`apply` orchestration.

Subclasses declare :class:`Distribution`-typed class attributes to define
free parameters (auto-discovered via ``__init_subclass__``), and optionally
define ``inputs`` and ``outputs`` dicts to declare the cross-component contract.
The base class auto-derives :class:`ParamDeclaration` and :class:`DerivedKey`
tuples from these class-level declarations.

Concrete adapters like :class:`tengri.components.radio.component.RadioSEDComponent`
and :class:`tengri.components.igm.component.IGMSEDComponent` follow
this pattern. See :doc:`docs/dev/archive/forward-model-architecture.md` §3.1 and
:doc:`docs/dev/sed-model-components.md` for the full authoring guide.

Examples
--------
A bare-minimum custom component::

    from tengri import SEDModelComponent, Uniform, Fixed
    from tengri.protocols.component import DerivedKey


    class MyComponent(SEDModelComponent):
        name = "my_model"
        parameter_prefix = "my_"

        # Free parameters — auto-discovered
        T = Uniform(20.0, 80.0, description="Temperature", units="K")

        # Cross-component contract (optional)
        inputs = {"L_in": "erg/s"}
        outputs = {"L_out": "erg/s"}

        def predict(self, p, sed_in, wave, *, L_in):
            '''Pure JAX prediction step.

            Parameters
            ----------
            p : dict
                Sliced parameter dict (prefix already stripped).
                p["T"] = temperature value.
            sed_in : ndarray
                Input SED in erg/s/Hz.
            wave : ndarray
                Rest-frame wavelength grid in Angstrom.
            L_in : ndarray
                Published quantity from upstream component.

            Returns
            -------
            tuple[ndarray, dict]
                (sed_out, published) where sed_out is the updated SED
                and published is a dict of keys this component publishes
                (e.g., {"L_out": L_out_value}).
            '''
            # Compute modified SED and publish derived quantities
            sed_out = sed_in + my_physics_fn(wave, p["T"])
            L_out = jnp.sum(sed_out)
            return sed_out, {"L_out": L_out}

Registries
----------
:data:`_REGISTRY` holds a mapping of component names to their classes,
populated automatically by :meth:`__init_subclass__`. Name collisions
are detected at class-definition time and raise :class:`ValueError`.

Notes
-----
**JIT-compatible**: :meth:`predict` MUST be pure JAX so it can flow
through :meth:`apply` and be JIT-compiled by the orchestrator.

**Immutability**: components are frozen dataclasses. Mutable state is
discouraged; cached tensors should be held in a frozen :class:`SEDComponentState`.

**Parameter discovery**: Subclasses declare priors as class attributes typed with
:class:`Distribution` subclasses (e.g., ``T = Uniform(...)``). The base class
walks ``vars(cls)`` at class-definition time to extract these into a tuple,
stripping them from the class dict so :meth:`inputs()`/:meth:`outputs()` method
resolution finds the base class's method implementations rather than dicts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from tengri.components.template_threading import TemplateThreading
from tengri.parameters.priors import Distribution
from tengri.protocols.component import (
    BARE_NAME_ALLOWLIST,
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = [
    "SEDModelComponent",
]

# Module-level registry: name → class
_REGISTRY: dict[str, type[SEDModelComponent]] = {}


class SEDModelComponent(TemplateThreading):
    """Astronomer-friendly base class for SED physics components.

    Implements the :class:`SEDComponent` Protocol with automatic parameter
    discovery, cross-component contract validation, and a default :meth:`apply`
    orchestration that calls the user's :meth:`predict` method.

    Required attributes (set by subclass)
    ------------------------------------
    name : str
        Stable identifier (e.g., ``"dust_ir"``, ``"radio"``).
    parameter_prefix : str
        Domain prefix for all free parameters this component owns
        (e.g., ``"dust_"``, ``"radio_"``).
    config : SEDComponentConfig
        Frozen structural knobs (e.g., which dust law, which radio model).
        Defaults to base :class:`SEDComponentConfig` if not overridden.
    taylor_order : int, default 0
        Taylor expansion order for WavePrecomp refinement. Set to 1 to enable
        first-order derivative publishing (``{name}_phot_lnu_slope_precomp``).

    Optional class-level declarations (auto-processed by __init_subclass__)
    ----------
    inputs : dict[str, str]
        Cross-component inputs: dict mapping key name to units string.
        Example: ``{"L_absorbed": "erg/s"}``.
        Default: ``{}``.
    outputs : dict[str, str]
        Cross-component outputs: dict mapping key name to units string.
        Example: ``{"L_ir": "erg/s"}``.
        Default: ``{}``.

    Free parameters (auto-discovered)
    ---------------------------------
    Any class attribute typed with a :class:`Distribution` subclass is
    treated as a free parameter. Examples::

        T = Uniform(20.0, 80.0, description="Temperature", units="K")
        beta = Gaussian(1.5, 0.3, description="Index", units="dimensionless")
        tau = Fixed(0.5)

    The base class extracts these at class-definition time and stores them
    in ``_priors: dict[str, Distribution]``. These are NOT visible as class
    attributes after the class is created (they are deleted to avoid shadowing
    the instance method :meth:`inputs`/:meth:`outputs`).

    Methods
    -------
    load(wave) -> object | None
        Optional precomputation. Override to return cached static tensors
        (template grids, precomputed age masks). Default returns None.
        Called by :meth:`precompute`, which sets result on ``self.data``.

    predict(p, sed_in, wave, **inputs) -> tuple[ndarray, dict]
        Pure JAX prediction step. MUST be implemented by subclass.
        Called by the default :meth:`apply` orchestration.

        Parameters:
            p (dict): Parameters with prefix already stripped.
            sed_in (ndarray): Input rest-frame L_nu in erg/s/Hz.
            wave (ndarray): Rest-frame wavelength grid in Angstrom.
            **inputs: Unpacked cross-component inputs from state.derived.

        Returns:
            (sed_out, published) where sed_out is updated SED and published
            is a dict of keys this component publishes.

    declared_parameters() -> list[ParamDeclaration]
        Returns :class:`ParamDeclaration` for each discovered prior.
        Automatically constructed; generally no need to override.

    inputs() -> tuple[DerivedKey, ...]
        Cross-component inputs as :class:`DerivedKey` tuples.
        Automatically constructed from ``inputs`` dict; do not override.

    outputs() -> tuple[DerivedKey, ...]
        Cross-component outputs as :class:`DerivedKey` tuples.
        Automatically constructed from ``outputs`` dict; do not override.

    precompute(ssp_data=None, wave_grid=None, approx=None, filters=None)
        Calls :meth:`load(wave_grid)`, caches result on ``self.data``,
        and returns an :class:`SEDComponentState`. Override :meth:`load`
        instead of this method in most cases.

    apply(state, params) -> ForwardState
        Default orchestration: slices params to prefix, looks up inputs from
        state.derived, calls :meth:`predict` (or :meth:`_apply_precomp` if
        WavePrecomp is active), and returns new state with updated sed_intrinsic
        and published keys. Generally no need to override.

    _apply_precomp(p, sed_in, filter_eff_waves, **inputs) -> dict
        Helper called by :meth:`apply` when ``filter_eff_waves`` is in
        state.derived (WavePrecomp active). Computes photometric LUTs via
        :meth:`predict_precomp`, optionally with Taylor first-order refinement.

    predict_precomp(p, filter_eff_waves, **inputs) -> tuple[ndarray, dict]
        Compute photometric LUT at effective filter wavelengths. Default
        falls back to :meth:`predict`; subclasses with direct photometric
        paths should override for specialized LUT generation.

    Vocabulary: **precompute** (verb) names build-time work — the
    :meth:`precompute` hook and ``*_precompute.py`` modules. **precomp**
    (noun) names the resulting LUT path and its artifacts —
    ``predict_precomp`` / ``_apply_precomp``, the ``*_lnu_precomp`` keys,
    ``predict_via_precomp``. The two spellings are deliberate, not drift.

    Raises
    ------
    ValueError
        At class-definition time if ``name`` collides with another registered
        component, or if ``config`` is not an :class:`SEDComponentConfig`.

    Examples
    --------
    Minimal concrete component::

        class DustIR(SEDModelComponent):
            name = "dust_ir"
            parameter_prefix = "dust_"
            T = Uniform(20.0, 80.0, "temperature", units="K")
            outputs = {"L_ir": "erg/s"}

            def predict(self, p, sed_in, wave):
                L_ir = mbb_lnu(wave, p["T"])
                return sed_in + L_ir, {"L_ir": jnp.sum(L_ir)}

    Notes
    -----
    **Naming**: Input/output dicts are declared as class attributes ``inputs``
    and ``outputs`` (user-friendly names). The class methods :meth:`inputs()`
    and :meth:`outputs()` (Protocol-style accessors returning tuples) have the
    same names. This shadowing is intentional: the base class removes the dict
    attributes after processing so method resolution finds the methods.

    **Instance data**: To cache static tensors from :meth:`load`, use
    setting as a normal object attribute or storing on a frozen
    :class:`SEDComponentState`.

    **Parameter discovery**: Subclasses declare priors as class attributes typed with
    :class:`Distribution` subclasses (e.g., ``T = Uniform(...)``). The base class
    walks ``vars(cls)`` at class-definition time to extract these into a tuple,
    stripping them from the class dict so :meth:`inputs()`/:meth:`outputs()` method
    resolution finds the base class's method implementations rather than dicts.

    See Also
    --------
    SEDComponent : the Protocol this class implements.
    ParamDeclaration, DerivedKey : contract types.
    docs/dev/archive/forward-model-architecture.md : the architecture design.
    docs/dev/sed-model-components.md : detailed authoring guide.
    """

    # Class attributes populated by __init_subclass__
    _priors: ClassVar[dict[str, Distribution]] = {}
    _inputs_tuple: ClassVar[tuple[DerivedKey, ...]] = ()
    _outputs_tuple: ClassVar[tuple[DerivedKey, ...]] = ()
    _optional_inputs_tuple: ClassVar[tuple[DerivedKey, ...]] = ()
    _citations_tuple: ClassVar[tuple[str, ...]] = ()

    #: Set to ``True`` by a component that genuinely reads no parameters.
    #:
    #: An empty ``_priors`` is ambiguous and cannot be inferred from: it means
    #: either "this component reads nothing" (``pah_drude``, a pure template
    #: shape) or "this component's parameters are declared somewhere other than
    #: the class" (``energy_balance_split`` reads six knobs declared in
    #: ``components/dust/_params.py``, because re-declaring them beside the
    #: attenuator's would raise a duplicate declaration).
    #:
    #: ``_declared_param_names`` therefore refuses to guess: it narrows a group
    #: wildcard to the empty set only when a component *says* it reads nothing,
    #: and leaves the wildcard alone otherwise. Inferring instead of asking
    #: would pin all six of the latter's parameters, which is the failure that
    #: narrowing exists to prevent; not asking at all leaves the former freeing
    #: the whole static union, which is #1482.
    declares_no_parameters: ClassVar[bool] = False

    #: Domain to publish precompute keys under, when it differs from ``name``.
    #:
    #: ``name`` is the registry key, and the precompute keys are derived from it
    #: (``{name}_phot_lnu_precomp`` and siblings). Those keys are typed fields on
    #: ``DerivedState``, so a component whose registry key is not itself a
    #: declared domain spills into ``_extras`` and trips the ADR-0007 guard.
    #:
    #: Several components share one domain by construction -- only one X-ray
    #: component is ever built, so ``xray_aird`` and the shared ``xray``
    #: component publish the same ``xray_*`` fields and are never in a state
    #: together. Setting this lets the registry key and the published domain
    #: differ, instead of adding a parallel set of ``DerivedState`` fields per
    #: registry name (which would also leave each new name outside the
    #: ``sed_xray`` accounting until someone remembered to add it).
    publish_name: ClassVar[str] = ""

    @property
    def _derived_prefix(self) -> str:
        """Prefix for this component's published derived keys."""
        return self.publish_name or self.name

    # Instance attributes (set by subclass, but with class defaults)
    name: str = "component"
    parameter_prefix: str = "component_"
    config: SEDComponentConfig = SEDComponentConfig()
    taylor_order: int = 0  # Taylor expansion order: 0 (zeroth-order), 1 (+ first-order derivative)

    # The template-threading seam (``accepts_threaded_templates``,
    # ``template_namespace``, ``threaded_templates``, ``templates_for_threading``)
    # is inherited from :class:`TemplateThreading`, which the bare-Protocol
    # component family inherits too — see that class for why it does not live
    # here.

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-discover free parameters, inputs, outputs; register by name.

        Called when a subclass is defined. Walks ``vars(cls)`` to extract
        :class:`Distribution`-typed attributes into ``_priors``, and parses
        ``inputs``/``outputs`` dicts into :class:`DerivedKey` tuples.
        Then deletes the dicts so method resolution finds the instance methods.

        Raises ValueError if ``name`` is already registered.
        """
        super().__init_subclass__(**kwargs)

        # Extract free parameters (Distribution-typed class attributes)
        priors: dict[str, Distribution] = {}
        for attr_name, attr_value in list(vars(cls).items()):
            if isinstance(attr_value, Distribution):
                priors[attr_name] = attr_value

        # Store on the class
        cls._priors = priors

        # Extract and parse inputs / outputs / optional_inputs dicts.
        # Build DerivedKey tuples from the full inheritance chain.
        inputs_dict = {}
        outputs_dict = {}
        optional_inputs_dict = {}

        # Collect dicts from this class and inherited classes (traverse MRO).
        for base in cls.__mro__[::-1]:  # Start from most distant base
            base_inputs = vars(base).get("inputs")
            if isinstance(base_inputs, dict):
                inputs_dict.update(base_inputs)
            base_outputs = vars(base).get("outputs")
            if isinstance(base_outputs, dict):
                outputs_dict.update(base_outputs)
            base_opt_inputs = vars(base).get("optional_inputs")
            if isinstance(base_opt_inputs, dict):
                optional_inputs_dict.update(base_opt_inputs)

        # Build DerivedKey tuples
        inputs_tuple = tuple(DerivedKey(name, units, "") for name, units in inputs_dict.items())
        outputs_tuple = tuple(DerivedKey(name, units, "") for name, units in outputs_dict.items())
        optional_inputs_tuple = tuple(
            DerivedKey(name, units, "") for name, units in optional_inputs_dict.items()
        )

        cls._inputs_tuple = inputs_tuple
        cls._outputs_tuple = outputs_tuple
        cls._optional_inputs_tuple = optional_inputs_tuple

        # Delete the dicts so method resolution finds the base methods.
        # BUT: only delete if this is a concrete class (has a 'name' attribute
        # defined in its own vars()). For abstract base classes like EmissionComponent,
        # keep the dicts so subclasses can inherit and process them.
        is_concrete = "name" in vars(cls)
        if is_concrete:
            # The declared inputs/outputs/optional_inputs dicts must not shadow the
            # same-named accessor methods once the tuples are collected. Three cases:
            #  (a) the dict is on THIS concrete class -> delete it so the base method resolves;
            #  (b) the dict is inherited from an intermediate ABSTRACT base (e.g.
            #      EmissionComponent, sharing the emission I/O contract) -> that dict shadows the
            #      method for this subclass, so rebind the base accessor onto the concrete class.
            #  (c) BOTH: this concrete class OVERRODE the dict *and* an abstract base also
            #      declares one -> deleting our dict (a) merely re-exposes the base's dict, so
            #      the rebind (b) must run afterwards, not as an ``elif``. The tuple was already
            #      built from the MRO union, so it carries the override either way.
            for _attr in ("inputs", "outputs", "optional_inputs"):
                if isinstance(vars(cls).get(_attr), dict):
                    delattr(cls, _attr)
                if isinstance(getattr(cls, _attr, None), dict):
                    setattr(cls, _attr, getattr(SEDModelComponent, _attr))

        # Citations: read class attribute (tuple of bib keys), store on class.
        # Subclasses declare ``citations = ("calzetti2000", ...)``; the
        # citations() method synthesized below returns the tuple. Default ().
        #
        # ``_citations_tuple`` written directly in the class body is honored as
        # well. It has to be: the assignment below is unconditional, so a
        # subclass spelling it that way had its keys overwritten with ``()`` at
        # class creation — silently, because the attribute it wrote is exactly
        # the one this line clobbers. THIRTEEN of the fifteen components that
        # declare citations used that spelling, so `component.citations()`
        # returned nothing for every dust emission backend in the library
        # (#1777). Accepting both spellings means neither can be dropped;
        # ``tests/contract/test_component_citations_are_not_dropped.py``
        # pins it.
        citations_attr = vars(cls).get("citations")
        if citations_attr is None:
            citations_attr = vars(cls).get("_citations_tuple", ())
        if not isinstance(citations_attr, tuple):
            citations_attr = tuple(citations_attr)
        cls._citations_tuple = citations_attr
        # Delete the class attribute so the method takes precedence on lookup.
        if "citations" in vars(cls):
            delattr(cls, "citations")

        # Properties: read class attribute (dict of Property objects).
        # Subclasses declare ``properties = {"name": Property(...), ...}``
        # to publish derived quantities. ONLY process for concrete classes.
        if is_concrete:
            properties_dict = vars(cls).get("properties")
            if isinstance(properties_dict, dict):
                # Lazy import to avoid cycles at module load time
                from tengri.forward.properties import register_properties

                component_name = vars(cls)["name"]
                register_properties(component_name, properties_dict)
                # Delete the class attribute so direct access doesn't shadow methods.
                # ``properties`` came from ``vars(cls)``, so it is on this class and
                # ``delattr`` cannot raise.
                delattr(cls, "properties")

        # Register by name — ONLY concrete classes that define their OWN ``name``.
        # Abstract authoring bases (e.g. EmissionComponent) inherit the default name and
        # must NOT register: they are scaffolds, not dispatchable components.
        if "name" in vars(cls):
            component_name = vars(cls)["name"]
            if component_name in _REGISTRY and _REGISTRY[component_name] is not cls:
                existing_cls = _REGISTRY[component_name]
                raise ValueError(
                    f"Component name {component_name!r} already registered by "
                    f"{existing_cls.__module__}.{existing_cls.__qualname__} — "
                    f"collision with {cls.__module__}.{cls.__qualname__}"
                )
            _REGISTRY[component_name] = cls

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Optional precomputation hook. Override to cache static tensors.

        Called by :meth:`precompute` before tracing. May perform eager
        computation, file I/O, or return None if no caching is needed.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid in Angstrom. Components that don't
            need a grid (radio, IGM, X-ray) may ignore this.

        Returns
        -------
        object or None
            Cached static tensors (e.g., template grid, precomputed masks).
            Stored on ``self.data`` if not None. Return None for no-op.
        """
        return None

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Declare free parameters this component owns.

        Returns
        -------
        list[ParamDeclaration]
            One entry per discovered :class:`Distribution`-typed class
            attribute, with name prefixed by :attr:`parameter_prefix`.
            Units and description are extracted from the Distribution
            if available (attributes ``units`` and ``description``).
        """
        result = []
        for name, prior in self._priors.items():
            full_name = self.parameter_prefix + name
            description = prior.description
            units = prior.units
            result.append(
                ParamDeclaration(name=full_name, prior=prior, description=description, units=units)
            )
        return result

    def inputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component inputs this component reads.

        Returns
        -------
        tuple[DerivedKey, ...]
            Derived keys required from upstream components.
            Constructed from the ``inputs`` class dict at class-definition time.
        """
        return self._inputs_tuple

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component outputs this component publishes.

        Returns
        -------
        tuple[DerivedKey, ...]
            Derived keys published to downstream components.
            Constructed from the ``outputs`` class dict at class-definition time.
        """
        return self._outputs_tuple

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component reads with a documented fallback (zero by default).

        Returns
        -------
        tuple[DerivedKey, ...]
            Derived keys this component reads *opportunistically* — if an
            upstream component publishes the key, its value is passed to
            :meth:`predict` as a keyword argument; if not, the framework
            substitutes ``jnp.asarray(0.0)``. This is how a downstream
            component like radio can read ``L_ir`` from dust if dust is
            in the chain but still produce a sensible output if not.

            Constructed from the ``optional_inputs`` class dict at
            class-definition time, mirroring the ``inputs`` shape.
        """
        return self._optional_inputs_tuple

    def citations(self) -> tuple[str, ...]:
        """Bib keys for papers this component implements or solves for.

        Returns
        -------
        tuple of str
            Keys into :data:`tengri.citations.registry.REGISTRY`.
            Subclasses declare the tuple via a ``citations`` class
            attribute; this method returns it. Default is ``()``.
        """
        return self._citations_tuple

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> SEDComponentState:
        """Cache static tensors before tracing. Calls :meth:`load` and stores result.

        Parameters
        ----------
        ssp_data : object, optional
            SSP stellar population synthesis grid. Some components (stellar,
            nebular) need this; others (radio, IGM) ignore it.
        wave_grid : ndarray, optional
            Rest-frame wavelength grid in Angstrom.
        approx : mapping, optional
            Build-time approximation flags (e.g., ``wave_precomp=True``).
        filters : tuple, optional
            Photometric filter curves for precomputation.

        Returns
        -------
        SEDComponentState
            Cached state for use in :meth:`apply`. Holds any result from
            :meth:`load` in the ``data`` attribute if not None.
        """
        # Call user's optional load() hook
        data = self.load(wave_grid)

        # Stash on self for use in apply()
        if data is not None:
            self.data = data

        return SEDComponentState(name=self.name)

    def slice_params(self, params):
        """Strip :attr:`parameter_prefix` to the form :meth:`predict` expects.

        The single definition of the runtime slicing rule. Build-time precomputes
        (e.g. the additive-emitter band response) MUST call this rather than
        re-deriving it: a precompute that slices differently from ``apply`` builds
        its table from the *wrong* parameter values and returns confidently wrong
        fluxes with no error — the failure mode this codebase keeps rediscovering.

        Parameters
        ----------
        params : mapping[str, ndarray]
            Full parameter dict, prefixed names (``dust_alpha_dale``, …).

        Returns
        -------
        dict
            Prefix-stripped params (``alpha_dale``), plus bare-name allowlist
            entries (``redshift``) passed through unstripped.

        Notes
        -----
        **JIT-compatible**: yes — pure dict manipulation, no tracing.
        """
        prefix_len = len(self.parameter_prefix)
        p_sliced = {
            k[prefix_len:]: v for k, v in params.items() if k.startswith(self.parameter_prefix)
        }
        # Bare-name allowlist params (e.g. redshift) have no domain prefix; the
        # orchestrator threads them to every component
        # (protocols.component.BARE_NAME_ALLOWLIST). Expose them to predict()/LUT
        # paths unstripped, honoring the documented contract.
        for _bare in BARE_NAME_ALLOWLIST:
            if _bare in params:
                p_sliced[_bare] = params[_bare]
        return p_sliced

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Mapping[str, Any] | None = None,
    ) -> ForwardState:
        """Default orchestration: slice params, look up inputs, call predict.

        Strips :attr:`parameter_prefix` from param names, looks up required
        inputs from :attr:`state.derived`, and calls :meth:`predict`. Updates
        :attr:`state.sed_intrinsic` with the returned SED and publishes
        returned keys to :attr:`state.derived`.

        Bare-name allowlist parameters (e.g., ``redshift``) are passed through
        unstripped to :meth:`predict` and precomp paths to enable cross-component
        access.

        When WavePrecomp is active (``filter_eff_waves`` in state.derived),
        automatically routes through the LUT path via :meth:`predict_precomp`
        to compute effective-wavelength contributions, optionally with Taylor
        first-order refinement if ``taylor_order >= 1``.

        Parameters
        ----------
        state : ForwardState
            Current state with wave, sed_intrinsic, and derived keys.
        params : mapping
            Full parameter dict (this method slices by prefix).
        ssp_data : object, optional
            SSP data (ignored for dust emission components; available for
            subclasses that need it).
        template_data : mapping, optional
            Threaded template grids and precomputed data, keyed
            ``[namespace][component_name]``. Forwarded to :meth:`predict` as
            ``templates=`` when :attr:`accepts_threaded_templates` is set;
            ignored otherwise.

        Returns
        -------
        ForwardState
            New state with sed_intrinsic and derived keys updated.
        """
        p_sliced = self.slice_params(params)

        # Look up required inputs from derived
        input_kwargs = {}
        for input_key in self.inputs():
            key_name = input_key.name
            if key_name in state.derived:
                input_kwargs[key_name] = state.derived[key_name]
            else:
                # Required input missing — fail loudly
                raise KeyError(
                    f"Component {self.name!r} declares required input {key_name!r} "
                    f"but it was not published by any upstream component. "
                    f"Available derived keys: {list(state.derived.keys())}"
                )

        # Look up optional inputs from derived — fallback to 0.0 when missing.
        # This is the "documented-fallback" cross-component read pattern
        # used by radio (L_ir / L_agn_bol / log_mstar) and X-ray.
        for opt_key in self.optional_inputs():
            key_name = opt_key.name
            if key_name in state.derived:
                input_kwargs[key_name] = state.derived[key_name]
            else:
                input_kwargs[key_name] = jnp.asarray(0.0)

        # Initialize SED if not yet done
        if state.sed_intrinsic is None:
            sed_in = jnp.zeros_like(state.wave)
        else:
            sed_in = state.sed_intrinsic

        # LUT mode(s). A joint photometry+spectroscopy model publishes BOTH
        # ``spec_eff_waves`` (SpectrumPrecomp) and ``filter_eff_waves`` (WavePrecomp);
        # photometry- or spectroscopy-only models publish just one. Run every
        # LUT branch whose grid is present and UNION the published dicts, so a
        # single pass emits both ``*_spec_lnu_precomp`` and ``*_phot_lnu_precomp``
        # families (Part A — joint precompute). The full-grid SED
        # (``sed_intrinsic``) is intentionally NOT updated on any LUT path.
        spec_eff_waves = state.derived.get("spec_eff_waves")
        filter_eff_waves = state.derived.get("filter_eff_waves")
        if spec_eff_waves is not None or filter_eff_waves is not None:
            published: dict[str, Any] = {}
            if spec_eff_waves is not None:
                published.update(
                    self._apply_spec_precomp(p_sliced, sed_in, spec_eff_waves, **input_kwargs)
                )
            if filter_eff_waves is not None:
                published.update(
                    self._apply_precomp(p_sliced, sed_in, filter_eff_waves, **input_kwargs)
                )
            # ...and STILL add to sed_intrinsic. The LUT families are what
            # ``predict_via_precomp`` consumes, but ``sed_intrinsic`` is the panchromatic
            # model SED that ``Prediction.photometry()`` / ``rest_sed`` / ``obs_sed`` and
            # every best-fit overlay project directly. Leaving this component out of it
            # made a WavePrecomp model's "exact" photometry read ~6x low in W3/W4 —
            # bit-identical to a model built with no dust emission at all — while the
            # likelihood (which reads the LUT) was fine. Silent, and invisible to a fit.
            #
            # Free on the fit path: ``predict_via_precomp`` never READS ``sed_intrinsic``,
            # so XLA dead-code-eliminates the full-grid chain and the compiled kernel is
            # unchanged (358,180 FLOPs, ~130 us). It is NOT free in eager mode, which is
            # what ``predict_state`` and the test suite run — hence the single shared
            # evaluation here rather than one per LUT branch.
            # Build a SEPARATE dict for predict. Mutating ``input_kwargs``
            # would also inject ``templates`` into the ``_apply_*precomp``
            # helpers above, which forward it with ``**inputs`` and do not
            # accept it.
            predict_kwargs = dict(input_kwargs)
            if self.accepts_threaded_templates:
                predict_kwargs["templates"] = self.threaded_templates(template_data)
            sed_out, published_full = self.predict(p_sliced, sed_in, state.wave, **predict_kwargs)
            new_derived = self._merge_published(state.derived, {**published_full, **published})
            return state.with_(sed_intrinsic=sed_out, derived=new_derived)
        else:
            # Default full-grid path
            predict_kwargs = dict(input_kwargs)
            if self.accepts_threaded_templates:
                predict_kwargs["templates"] = self.threaded_templates(template_data)
            sed_out, published = self.predict(p_sliced, sed_in, state.wave, **predict_kwargs)
            # Update state with new SED and published keys
            new_derived = self._merge_published(state.derived, published)
            return state.with_(sed_intrinsic=sed_out, derived=new_derived)

    @staticmethod
    def _merge_published(derived: Any, published: Mapping[str, Any]) -> Any:
        """Route published keys to typed fields when defined, else into ``_extras``.

        A component declares output names in its ``outputs`` dict (e.g.
        ``{"L_ir": "erg/s"}``). If a name matches a typed field on
        :class:`DerivedState`, write through to that field so the value
        is observable via attribute / ``__contains__`` / mapping access.
        Otherwise drop the value into ``_extras``, the documented
        escape hatch for keys not yet promoted to typed fields.
        """
        known = set(derived.field_names())
        typed_updates = {k: v for k, v in published.items() if k in known}
        extras_updates = {k: v for k, v in published.items() if k not in known}
        if extras_updates:
            merged_extras = {**derived._extras, **extras_updates}
            return derived.with_(_extras=merged_extras, **typed_updates)
        return derived.with_(**typed_updates) if typed_updates else derived

    def _apply_precomp(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        filter_eff_waves: jnp.ndarray,
        **inputs: Any,
    ) -> Mapping[str, jnp.ndarray]:
        """Compute WavePrecomp photometric LUTs for this component.

        Called by :meth:`apply` when WavePrecomp is active. Evaluates
        :meth:`predict_precomp` at filter effective wavelengths to build
        a photometric LUT, and optionally computes first-order Taylor
        refinement if ``taylor_order >= 1``.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped.
        sed_in : ndarray
            Input SED (used for shape/dtype, but not consumed on LUT path).
        filter_eff_waves : ndarray, shape (n_filter,)
            Rest-frame effective wavelengths of filters in Angstrom.
        **inputs : ndarray
            Cross-component inputs from state.derived.

        Returns
        -------
        mapping[str, ndarray]
            Published keys for this component, keyed as
            ``{self.name}_phot_lnu_precomp`` and optionally
            ``{self.name}_phot_lnu_slope_precomp`` if Taylor mode is enabled.
        """
        # Compute zeroth-order LUT
        phot_lnu_precomp, _ = self.predict_precomp(p, filter_eff_waves, **inputs)

        published = {f"{self._derived_prefix}_phot_lnu_precomp": phot_lnu_precomp}

        # Optionally compute first-order Taylor slope
        if self.taylor_order >= 1:
            # Compute ∂predict_precomp/∂wave at each filter wavelength via vmap
            def predict_precomp_scalar(wave_scalar):
                # Reshape to (1,) for predict_precomp, extract scalar result
                result, _ = self.predict_precomp(p, jnp.asarray([wave_scalar]), **inputs)
                return result[0]

            # Element-wise gradient using vmap
            slope = jax.vmap(jax.grad(predict_precomp_scalar))(filter_eff_waves)
            published[f"{self._derived_prefix}_phot_lnu_slope_precomp"] = slope

        return published

    def _apply_spec_precomp(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        spec_eff_waves: jnp.ndarray,
        **inputs: Any,
    ) -> Mapping[str, jnp.ndarray]:
        """Compute spectrum LUT at pixel centers (SpectrumPrecomp).

        Called by :meth:`apply` when approx=SpectrumPrecomp() is active.
        Evaluates :meth:`predict` at spectrum pixel effective wavelengths
        (in the galaxy rest frame) to build a per-pixel spectrum LUT.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped.
        sed_in : ndarray
            Input SED (used for shape/dtype, but not consumed on LUT path).
        spec_eff_waves : ndarray, shape (n_spec_pixel,)
            Rest-frame effective wavelengths of spectrum pixels in Angstrom.
        **inputs : ndarray
            Cross-component inputs from state.derived.

        Returns
        -------
        mapping[str, ndarray]
            ``{self.name}_spec_lnu_precomp`` (the per-pixel contribution)
            plus every key the component's :meth:`predict` published. The
            published dict is preserved — **not discarded** — so grid-
            independent derived quantities survive the LUT path. This is
            what lets a line-publishing nebular backend (Cue, CloudyGrid)
            still surface ``line_waves`` / ``line_lums`` under
            ``SpectrumPrecomp``, so ``predict_line_fluxes``, line ratios,
            and the ``pred.lines.*`` diagnostics keep working.
        """
        # Evaluate predict at spectrum pixel centers. ``published`` carries
        # grid-independent derived quantities (e.g. nebular line_waves /
        # line_lums) that must reach state.derived exactly as on the
        # full-grid path.
        spec_lnu_precomp, published = self.predict(
            p, jnp.zeros_like(spec_eff_waves), spec_eff_waves, **inputs
        )

        out = dict(published)
        out[f"{self._derived_prefix}_spec_lnu_precomp"] = spec_lnu_precomp
        return out

    def predict_precomp(
        self,
        p: Mapping[str, jnp.ndarray],
        filter_eff_waves: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Compute photometric LUT at effective filter wavelengths.

        Called by :meth:`_apply_precomp` when WavePrecomp is active.
        Default implementation calls :meth:`predict` with a dummy SED
        (zeros) and extracts the relevant LUT. Subclasses that consume
        photometric data directly (stellar, nebular, AGN, dust) should
        override to return specialized LUTs without computing the full SED.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped.
        filter_eff_waves : ndarray, shape (n_filter,)
            Rest-frame effective filter wavelengths in Angstrom.
        **inputs : ndarray
            Cross-component inputs from state.derived.

        Returns
        -------
        tuple[ndarray, mapping[str, ndarray]]
            (phot_lnu, published) where phot_lnu is shape (n_filter,) in
            erg/s/Hz, and published is a dict of derived keys (may be empty).
        """
        # Default: evaluate the full-grid predict at filter effective wavelengths
        # This is a fallback; subclasses that have direct photometric paths
        # should override with specialized implementations.
        sed_dummy = jnp.zeros_like(filter_eff_waves)
        sed_out, published = self.predict(p, sed_dummy, filter_eff_waves, **inputs)
        return sed_out, published

    def predict(
        self, p: Mapping[str, jnp.ndarray], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX prediction step. MUST be implemented by subclass.

        Called by the default :meth:`apply` with sliced parameters and
        unpacked inputs from state.derived.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped. Access via p["T"], p["beta"], etc.
        sed_in : ndarray
            Input rest-frame L_nu in erg/s/Hz.
        wave : ndarray
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Cross-component inputs, keyed by the names declared in
            the ``inputs`` dict. Example: ``L_absorbed=1e45``.

        Returns
        -------
        tuple[ndarray, mapping[str, ndarray]]
            (sed_out, published) where:

            - sed_out: Updated rest-frame L_nu in erg/s/Hz.
            - published: Dict of keys declared in ``outputs``, e.g.,
              ``{"L_ir": 1e45}``.

        Raises
        ------
        NotImplementedError
            Default implementation raises. Subclasses MUST override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.predict() must be implemented by subclass"
        )
