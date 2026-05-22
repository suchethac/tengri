"""SEDModelComponent: astronomer-facing base class for SED physics blocks.

Provides a convenient superclass over the :class:`SEDComponent` Protocol,
with automatic discovery of free parameters, cross-component inputs/outputs,
and a sensible default :meth:`apply` orchestration.

Subclasses declare :class:`Distribution`-typed class attributes to define
free parameters (auto-discovered via ``__init_subclass__``), and optionally
define ``inputs`` and ``outputs`` dicts to declare the cross-component contract.
The base class auto-derives :class:`ParamDeclaration` and :class:`DerivedKey`
tuples from these class-level declarations.

Concrete adapters like :class:`tengri.components.radio.RadioSEDComponent`
and :class:`tengri.components.igm.IGMSEDComponent` (Phase II-2 onward) follow
this pattern. See :doc:`docs/dev/forward-model-architecture.md` §3.1 and
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

from tengri.parameters.priors import Distribution
from tengri.protocols.component import (
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


class SEDModelComponent:
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
        beta = Gaussian(1.5, 0.3, description="Index", units="")
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
    docs/dev/forward-model-architecture.md : the architecture design.
    docs/dev/sed-model-components.md : detailed authoring guide.
    """

    # Class attributes populated by __init_subclass__
    _priors: ClassVar[dict[str, Distribution]] = {}
    _inputs_tuple: ClassVar[tuple[DerivedKey, ...]] = ()
    _outputs_tuple: ClassVar[tuple[DerivedKey, ...]] = ()

    # Instance attributes (set by subclass, but with class defaults)
    name: str = "component"
    parameter_prefix: str = "component_"
    config: SEDComponentConfig = SEDComponentConfig()
    taylor_order: int = 0  # Taylor expansion order: 0 (zeroth-order), 1 (+ first-order derivative)

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

        # Extract and parse inputs / outputs dicts
        inputs_dict = vars(cls).get("inputs", {})
        outputs_dict = vars(cls).get("outputs", {})

        # Build DerivedKey tuples
        inputs_tuple = tuple(DerivedKey(name, units, "") for name, units in inputs_dict.items())
        outputs_tuple = tuple(DerivedKey(name, units, "") for name, units in outputs_dict.items())

        cls._inputs_tuple = inputs_tuple
        cls._outputs_tuple = outputs_tuple

        # Delete the dicts so method resolution finds the base methods
        if "inputs" in vars(cls):
            delattr(cls, "inputs")
        if "outputs" in vars(cls):
            delattr(cls, "outputs")

        # Register by name
        component_name = vars(cls).get("name", "component")
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
            description = getattr(prior, "description", "")
            units = getattr(prior, "units", "")
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

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Default orchestration: slice params, look up inputs, call predict.

        Strips :attr:`parameter_prefix` from param names, looks up required
        inputs from :attr:`state.derived`, and calls :meth:`predict`. Updates
        :attr:`state.sed_intrinsic` with the returned SED and publishes
        returned keys to :attr:`state.derived`.

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

        Returns
        -------
        ForwardState
            New state with sed_intrinsic and derived keys updated.
        """
        # Slice parameters: strip prefix
        prefix_len = len(self.parameter_prefix)
        p_sliced = {
            k[prefix_len:]: v for k, v in params.items() if k.startswith(self.parameter_prefix)
        }

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

        # Initialize SED if not yet done
        if state.sed_intrinsic is None:
            sed_in = jnp.zeros_like(state.wave)
        else:
            sed_in = state.sed_intrinsic

        # Check for WavePrecomp mode
        filter_eff_waves = state.derived.get("filter_eff_waves")
        if filter_eff_waves is not None:
            # WavePrecomp path: compute photometry LUTs
            published = self._apply_precomp(p_sliced, sed_in, filter_eff_waves, **input_kwargs)
            # Do NOT update sed_intrinsic on the LUT path (only publish LUTs)
            # Use _extras for component-specific precomp LUTs that aren't typed fields
            new_extras = {**state.derived._extras, **published}
            return state.with_(derived=state.derived.with_(_extras=new_extras))
        else:
            # Default full-grid path
            sed_out, published = self.predict(p_sliced, sed_in, state.wave, **input_kwargs)
            # Update state with new SED and published keys
            new_extras = {**state.derived._extras, **published}
            return state.with_(
                sed_intrinsic=sed_out,
                derived=state.derived.with_(_extras=new_extras),
            )

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

        published = {f"{self.name}_phot_lnu_precomp": phot_lnu_precomp}

        # Optionally compute first-order Taylor slope
        if self.taylor_order >= 1:
            # Compute ∂predict_precomp/∂wave at each filter wavelength via vmap
            def predict_precomp_scalar(wave_scalar):
                # Reshape to (1,) for predict_precomp, extract scalar result
                result, _ = self.predict_precomp(p, jnp.asarray([wave_scalar]), **inputs)
                return result[0]

            # Element-wise gradient using vmap
            slope = jax.vmap(jax.grad(predict_precomp_scalar))(filter_eff_waves)
            published[f"{self.name}_phot_lnu_slope_precomp"] = slope

        return published

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
