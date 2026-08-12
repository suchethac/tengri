# SPDX-License-Identifier: BSD-3-Clause
"""PopulationSEDModel — SubModel for hierarchical galaxy populations.

A single class that holds an :class:`tengri.SEDModel` template + a list
of per-galaxy data dicts + the list of parameters that are tied across
the population. Used inside :class:`tengri.ForwardModel` exactly like
:class:`tengri.SEDModel` or :class:`tengri.SpatialSEDModel` — the
inference path stays uniform.

Example
-------

.. code-block:: python

    from tengri import (
        Fitter, ForwardModel, PopulationSEDModel, SEDModel,
    )

    template = SEDModel.build(ssp_data=ssp, observation=obs, ...)
    pop = PopulationSEDModel(
        sed=template,
        galaxies=[{'flux_obs': ..., 'noise': ...}, ...],  # N dicts
        shared=('sfh_field_psd_sigma', 'sfh_field_psd_tau_myr'),
        priors={
            'sfh_field_psd_sigma': (0.1, 4.0),
            'sfh_field_psd_tau_myr': (1.0, 300.0),
        },
    )
    forward = ForwardModel.build(population=pop, observation=obs)

    fitter = Fitter(forward, batched_flux, batched_noise)
    posterior = fitter.run('vi')        # or 'mcmc_raytrace'

    posterior.shared_samples['sfh_field_psd_sigma']
    posterior.shared_samples['sfh_field_psd_tau_myr']

How it works
------------
1. ``PopulationSEDModel`` satisfies :class:`tengri.protocols.SubModel` — it
   has ``name`` (``"population_sed_model"``), ``declared_parameters``
   (population-wide, from the SED template), and ``run`` (vmaps the
   template's ``run`` over the galaxy axis to produce a batched
   :class:`ForwardState`).

2. ``ForwardModel.build(population=pop, observation=obs)`` slots the
   PopulationSEDModel into ``Population(name="default", sed=pop)`` so the
   outer shell stays uniform.

3. ``Fitter(forward).run('vi')`` works exactly as for a single-galaxy
   fit. The standard inference path (NUTS / VI / MAP / Pathfinder /
   Ray Tracing / nested) consumes the batched ``(N_gal, n_filters)``
   prediction directly; the spec view publishes ``(N_gal,)`` shapes
   for per-galaxy latents; the prior penalty sums over all latent
   axes. **One information-Hamiltonian path** for every fit shape.
   See ``docs/forward_model.md`` 'Hierarchical population fits'.

The legacy ``PopulationFitter`` / ``HierarchicalFitter`` direct API
remains importable but emits a one-shot ``DeprecationWarning``
pointing at the canonical path above; it will be removed in v1.0.

See ``docs/dev/archive/forward-model-architecture.md`` §6 and issue #211.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.protocols.component import ForwardState, ParamDeclaration

__all__ = ["PopulationSEDModel"]


_DEFAULT_SHARED: tuple[str, ...] = (
    "sfh_field_psd_sigma",
    "sfh_field_psd_tau_myr",
)
_DEFAULT_PRIORS: dict[str, tuple[float, float]] = {
    "sfh_field_psd_sigma": (0.1, 4.0),
    "sfh_field_psd_tau_myr": (1.0, 300.0),
}


def _validate_homogeneous_galaxies(galaxies: Sequence[Mapping[str, Any]]) -> None:
    """Validate the shared-observation contract: galaxies share one data grid.

    The batched forward vmaps the SED template over the population and stacks
    per-galaxy data into a rectangular ``(N_gal, n_data)`` array
    (:meth:`PopulationSEDModel.batched_data`), so every galaxy must carry
    ``flux_obs``/``noise`` of the **same** shape — i.e. the same measurement grid
    (same filters or the same observed-frame spectroscopy pixels). Per-galaxy
    *values* (flux, noise, redshift and other free parameters) still vary freely;
    only the grid is shared. Truly heterogeneous per-galaxy grids would need
    ragged/padded batching and are out of scope for the batched Hamiltonian path.

    Raising here turns the otherwise cryptic ``jnp.stack`` broadcast error into a
    message that names the offending galaxy and the contract it violates.
    """
    ref_shapes: dict[str, tuple[int, ...]] = {}
    for i, gal in enumerate(galaxies):
        for key in ("flux_obs", "noise"):
            if key not in gal:
                raise ValueError(
                    f"PopulationSEDModel: galaxy {i} is missing required '{key}'. "
                    f"Each galaxy dict needs 'flux_obs' and 'noise'."
                )
            shape = jnp.asarray(gal[key]).shape
            if i == 0:
                ref_shapes[key] = shape
            elif shape != ref_shapes[key]:
                raise ValueError(
                    f"PopulationSEDModel: galaxy {i} '{key}' has shape {shape}, but "
                    f"galaxy 0 has {ref_shapes[key]}. All galaxies must share one "
                    f"measurement grid (same filters / spectroscopy pixels) — the "
                    f"batched forward stacks them into a rectangular array. "
                    f"Per-galaxy redshift and other parameters may still differ."
                )


@dataclass(frozen=True)
class PopulationSEDModel:
    """SubModel for a hierarchical galaxy population with shared parameters.

    Parameters
    ----------
    sed : SEDModel
        Template SED chain. Its physics (SFH family, dust law,
        nebular backend, …) is shared across the population.
    galaxies : sequence of mapping
        Per-galaxy data. Each dict must contain ``flux_obs`` and
        ``noise``; optionally ``spec_obs``, ``spec_noise``, ``wave_spec``.
    shared : sequence of str, optional
        Parameter names tied across the population. Default is the
        two canonical PSD hyperparameters.
    priors : mapping of str -> (lo, hi), optional
        Uniform prior bounds for the shared parameters. Must contain
        an entry for every name in ``shared``. Defaults to
        PSD-typical ranges.
    data_type : str, default ``"photometry"``
        Inference channel; passed to the underlying
        :class:`tengri.PopulationFitter`.

    Attributes
    ----------
    name : str
        The :class:`tengri.protocols.SubModel` identifier,
        ``"population_sed_model"``. Read-only — the protocol calls it a
        *stable* identifier, and this constructor has never accepted it.

    Notes
    -----
    Satisfies the :class:`tengri.protocols.SubModel` Protocol. The
    forward-time ``.run`` path is a stub today — the batched-vmap
    implementation lands when ``ForwardModel.predict_observables`` learns the
    PopulationSEDModel case (issue #211). Inference is routed at the
    :class:`tengri.Fitter` layer.
    """

    sed: Any
    galaxies: tuple[Mapping[str, Any], ...]
    shared: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_SHARED)
    priors: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_PRIORS)
    )
    data_type: str = "photometry"
    #: The :class:`SubModel` identifier. ``init=False`` because the protocol
    #: calls it a *stable* identifier: this constructor never accepted it, and
    #: an ordinary field declaration advertised otherwise to
    #: ``dataclasses.fields()`` and to the docstring above.
    name: str = field(default="population_sed_model", init=False)

    def __init__(
        self,
        sed: Any,
        galaxies: Sequence[Mapping[str, Any]],
        shared: Sequence[str] = _DEFAULT_SHARED,
        priors: Mapping[str, tuple[float, float]] | None = None,
        data_type: str = "photometry",
    ) -> None:
        if not galaxies:
            raise ValueError("PopulationSEDModel needs at least one galaxy.")
        if priors is None:
            priors = dict(_DEFAULT_PRIORS)
        missing = [n for n in shared if n not in priors]
        if missing:
            raise ValueError(
                f"PopulationSEDModel.priors is missing entries for shared parameters: {missing}. "
                f"Pass priors={{name: (lo, hi), ...}} covering every shared name."
            )
        _validate_homogeneous_galaxies(galaxies)
        object.__setattr__(self, "sed", sed)
        object.__setattr__(self, "galaxies", tuple(galaxies))
        object.__setattr__(self, "shared", tuple(shared))
        object.__setattr__(self, "priors", dict(priors))
        object.__setattr__(self, "data_type", data_type)

    def __hash__(self) -> int:
        """Identity-based hash.

        Frozen-dataclass auto-generated ``__hash__`` would hash the
        field tuple, but ``galaxies`` is a tuple of ``Mapping``s
        whose contents are JAX arrays / dicts — unhashable. The cache
        machinery (:class:`weakref.WeakKeyDictionary` in
        :mod:`tengri.inference._model_cache`) uses object identity
        anyway, so identity-hash is functionally equivalent and
        sidesteps the unhashable-field problem.

        Equality semantics retained: two PopulationSEDModel instances
        with identical contents are still ``__eq__``, but they hash
        differently. This matches the spirit of weak-keyed caches.
        """
        return id(self)

    @property
    def n_galaxies(self) -> int:
        return len(self.galaxies)

    @property
    def spec(self):
        """The batched-sample :class:`Parameters`-shaped view.

        Returns a :class:`PopulationSpecView` wrapping the SED
        template's scalar spec. Implements the same implicit Protocol
        :class:`Fitter` consumes (``free_params``, ``get_fixed_values``,
        ``sample(key)``, ``_distributions``, …) — but :meth:`sample`
        returns per-galaxy free parameters with shape ``(N_galaxies,)``,
        keeping shared parameters scalar.

        Fitter has zero special-case code for hierarchical fits: it
        just consumes whatever shapes the spec returns.

        Notes
        -----
        Construction is cheap (no JAX work). Use ``self.sed.spec`` to
        access the underlying scalar spec when you need it directly.
        """
        from tengri.parameters._population_view import PopulationSpecView

        return PopulationSpecView(
            template=self.sed.spec,
            n_galaxies=self.n_galaxies,
            shared=self.shared,
        )

    def batched_data(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(flux_obs, noise)`` arrays stacked across the population.

        Each array has shape ``(N_galaxies, n_filters)`` — the
        canonical batched shape that :class:`Fitter` accepts as the
        ``data`` and ``noise`` arguments for hierarchical fits.

        The likelihood's :math:`\\chi^2` broadcasts naturally over
        the leading galaxy axis against the prediction dict's
        ``(N_galaxies, n_filters)`` arrays returned by
        :meth:`ForwardModel.predict_observables`.

        Returns
        -------
        flux_obs : ndarray, shape ``(N_galaxies, n_filters)``
            Per-galaxy observed flux.
        noise : ndarray, shape ``(N_galaxies, n_filters)``
            Per-galaxy 1-sigma uncertainty.

        Raises
        ------
        KeyError
            If any galaxy dict is missing ``flux_obs`` or ``noise``.
        """
        flux_obs = jnp.stack([gal["flux_obs"] for gal in self.galaxies])
        noise = jnp.stack([gal["noise"] for gal in self.galaxies])
        return flux_obs, noise

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free-parameter declarations seen at the population level.

        Returns the shared population-level parameters. Per-galaxy
        free parameters are managed by the inference layer, not the
        population-level declared_parameters view (which is consumed
        by introspection helpers like
        :class:`tengri.Parameters.summary`).
        """
        return [
            ParamDeclaration(
                name=name,
                prior=None,  # bounds live in self.priors; expanded by inference
                description=f"Shared {name} (population-level)",
                units="",
            )
            for name in self.shared
        ]

    @property
    def batched_axes(self) -> dict[str, int]:
        """Named batched axes this SubModel introduces.

        Returns ``{"galaxy": 0}`` — PopulationSEDModel publishes a single
        named axis (``"galaxy"``) at position 0 of every per-galaxy
        array. Single-galaxy SubModels (``SEDModel``, ``SpatialModel``,
        …) return ``{}``.

        Used by :meth:`ForwardModel.predict_observables` and by the inference layer
        to compose batching (vmap / pmap / shard_map) without hidden
        nested vmaps. See :meth:`predict_one` for the un-batched
        primitive.

        Returns
        -------
        dict[str, int]
            Map from named axis to its integer position in arrays.
        """
        return {"galaxy": 0}

    def predict_one(
        self,
        state: ForwardState,
        params_one: Mapping[str, Any],
    ) -> ForwardState:
        """Un-batched primitive: run the SED template for a single galaxy.

        Corresponds to one term in the hierarchical information
        Hamiltonian's data sum (paper §4):

        .. math::

            \\mathcal{H}_{\\rm hier} = \\tfrac{1}{2}\\sum_{j=1}^{N_{\\rm gal}}
                \\chi^2\\!\\bigl(\\mathbf{d}^{(j)},\\,
                    \\mathbf{f}\\!\\bigl(\\mathbf{h}(\\boldsymbol{\\xi}^{(j)},\\,
                                              \\boldsymbol{\\xi}^{(\\rm hyp)})\\bigr)\\bigr)
              + \\tfrac{1}{2}\\,\\boldsymbol{\\xi}^{\\!\\top}\\boldsymbol{\\xi}

        where :math:`\\mathbf{f}\\circ\\mathbf{h}` is the per-galaxy
        forward model from latent space to predicted observables.
        ``predict_one`` evaluates that pipeline for one galaxy;
        :meth:`run` vmaps over the population to evaluate the full sum.

        Composable with outer ``jax.vmap`` / ``jax.pmap`` / ``shard_map``.
        Use this entry point when you need control over the batching
        strategy — e.g. multi-device sharding, catalog × hierarchical
        fits, or posterior-predictive sweeps that already have an
        outer ``vmap``. Use :meth:`run` for the default batched path.

        Parameters
        ----------
        state : ForwardState
            Initial state. Wavelength grid + any upstream state.
        params_one : Mapping
            **Un-batched** parameters for a single galaxy. Every
            value is a scalar (no leading ``N_galaxies`` axis).

        Returns
        -------
        ForwardState
            Single-galaxy state. No leading galaxy axis.

        Notes
        -----
        Pure JAX. JIT/grad/vmap/pmap-compatible. Forwarding to
        ``self.sed.run`` is a no-op layer that exists so external
        callers can compose their own batching without nesting
        :meth:`run`'s internal vmap.
        """
        return self.sed.run(state, params_one)

    def parameter_axes(self, params: Mapping[str, Any]) -> dict[str, int | None]:
        """Per-parameter vmap axis: 0 for per-galaxy, None for shared / scalar.

        Three cases, in order:

        1. Names in :attr:`shared` always broadcast — axis ``None``.
        2. Rank-0 / scalar values (e.g. fixed values like ``redshift``
           merged from :attr:`spec`) also broadcast — axis ``None``.
        3. Everything else is per-galaxy — axis ``0`` (the value is a
           1-D array of length N_galaxies).

        The scalar-detection branch is what lets fixed values
        (``redshift``, calibration coefficients, …) flow through
        without being treated as per-galaxy data.

        Parameters
        ----------
        params : Mapping[str, Any]
            The parameters dict that ``run`` will be called with.

        Returns
        -------
        dict[str, int | None]
            One entry per key in ``params``.
        """
        shared_set = set(self.shared)
        out: dict[str, int | None] = {}
        for name, value in params.items():
            if name in shared_set:
                out[name] = None
            else:
                shape = getattr(value, "shape", ())
                out[name] = 0 if (shape and len(shape) > 0) else None
        return out

    def run(
        self,
        state: ForwardState,
        params: Mapping[str, Any],
    ) -> ForwardState:
        """Forward-time batched SED across the population (via ``jax.vmap``).

        Each population-level parameter is either ``shared`` (one value,
        broadcast across the N galaxies via vmap ``in_axes=None``) or
        per-galaxy (an array of length N, vmap ``in_axes=0``). The
        returned :class:`ForwardState` has a leading ``N_galaxies``
        axis on every per-galaxy quantity (``sed_intrinsic``,
        ``sed_observed``, ``sed_attenuated``, etc.).

        See ``docs/superpowers/plans/2026-05-22-population-sed-batched-forward.md``
        for the design rationale.

        Parameters
        ----------
        state : ForwardState
            Initial state. The wavelength grid is shared across the
            population (broadcast) — the SED template owns the grid.
        params : Mapping
            Per-parameter values. Shared params are scalars; per-galaxy
            params are length-N arrays. ``parameter_axes(params)``
            returns the vmap-axis dict.

        Returns
        -------
        ForwardState
            Batched state: every per-galaxy quantity has a leading
            ``N_galaxies`` axis.

        Notes
        -----
        JIT/grad/vmap-compatible.

        Notes (composability)
        ---------------------
        This method is the default convenience path; it vmaps
        :meth:`predict_one` over the galaxy axis. For multi-device
        sharding, catalog × hierarchical fits, or any case where you
        need to compose outer batching, call :meth:`predict_one`
        directly with your own ``vmap`` / ``pmap`` / ``shard_map``.
        The :attr:`batched_axes` property publishes the axis layout
        so external code can stay axis-aware.
        """
        import jax

        in_axes_params = self.parameter_axes(params)
        # State is broadcast across the population (one wavelength grid,
        # one upstream state); params follow their per-parameter axes.
        return jax.vmap(self.predict_one, in_axes=(None, in_axes_params))(state, params)
