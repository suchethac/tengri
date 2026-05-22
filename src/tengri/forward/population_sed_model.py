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
   (population-wide, from the SED template), and ``run`` (currently a
   stub raising ``NotImplementedError``; the forward-time batched-vmap
   path lands in a follow-up refactor — see issue #211).

2. ``ForwardModel.build(population=pop, observation=obs)`` slots the
   PopulationSEDModel into ``Population(name="default", sed=pop)`` so the
   outer shell stays uniform.

3. ``Fitter(forward, ...)`` detects that ``forward.populations[0].sed``
   is a :class:`PopulationSEDModel` and routes inference to the existing
   :class:`tengri.PopulationFitter` machinery. The user calls
   ``fitter.run('vi')`` exactly as for a single-galaxy fit. The two
   shared PSD parameters' priors and the per-galaxy data come from the
   ``PopulationSEDModel`` construction.

The legacy ``PopulationFitter`` / ``HierarchicalFitter`` direct API
keeps working unchanged. New code should reach for ``PopulationSEDModel``.

See ``docs/dev/forward-model-architecture.md`` §6 and issue #211.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
    name : str, default ``"population_sed_model"``
        SubModel identifier.

    Notes
    -----
    Satisfies the :class:`tengri.protocols.SubModel` Protocol. The
    forward-time ``.run`` path is a stub today — the batched-vmap
    implementation lands when ``ForwardModel.predict`` learns the
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
    name: str = "population_sed_model"

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
        object.__setattr__(self, "sed", sed)
        object.__setattr__(self, "galaxies", tuple(galaxies))
        object.__setattr__(self, "shared", tuple(shared))
        object.__setattr__(self, "priors", dict(priors))
        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "name", "population_sed_model")

    @property
    def n_galaxies(self) -> int:
        return len(self.galaxies)

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

    def run(
        self,
        state: ForwardState,
        params: Mapping[str, Any],
    ) -> ForwardState:
        """Forward-time batched SED across the population.

        **Not yet implemented.** Today's hierarchical inference path
        bypasses ``ForwardModel.predict`` entirely (it routes at the
        :class:`tengri.Fitter` layer through
        :class:`tengri.PopulationFitter`). When
        ``ForwardModel.predict`` learns to return batched arrays
        (issue #211), this method will run the SED template under
        ``jax.vmap`` across the population and return a ForwardState
        whose ``sed_intrinsic`` has shape ``(N_galaxies, n_wave)``.

        Raises
        ------
        NotImplementedError
            Always, until the batched-vmap forward path lands. Use
            ``Fitter(forward, ...).run('vi')`` — the inference path
            handles hierarchical population fits without going through
            ``forward.predict``.
        """
        raise NotImplementedError(
            "PopulationSEDModel.run (batched forward via vmap) is not yet wired up. "
            "Use the inference path: "
            "Fitter(forward, batched_flux, batched_noise).run('vi') — "
            "the Fitter detects PopulationSEDModel and routes through "
            "tengri.PopulationFitter under the hood. "
            "See issue #211 for the forward-path refactor."
        )
