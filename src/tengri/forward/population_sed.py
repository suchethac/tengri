"""PopulationSED — a SubModel-shaped wrapper for hierarchical population fits.

A common scientific use case in tengri is the hierarchical PSD fit:
many galaxies share an underlying PSD (σ_PSD, τ_PSD) that you want to
constrain at the population level. The legacy machine for this is
:class:`tengri.PopulationFitter`, which requires the user to write a
``model_factory(psd_sigma, psd_tau_myr) -> SEDModel`` closure.

``PopulationSED`` is a single class that bundles the population
description (one SED template + a list of galaxies + the parameters
shared across the population) into the same shape as the rest of the
forward-model architecture. The user writes one
:class:`tengri.SEDModel` template, hands it a list of galaxy data,
and calls ``.fit("vi")`` — no model-factory closure required.

Example
-------

.. code-block:: python

    from tengri import SEDModel, PopulationSED

    template = SEDModel.build(ssp_data=ssp, observation=obs, ...)

    pop = PopulationSED(
        sed=template,
        galaxies=[{"flux_obs": ..., "noise": ...}, ...],  # N dicts
        shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
        priors={
            "sfh_field_psd_sigma": (0.1, 4.0),
            "sfh_field_psd_tau_myr": (1.0, 300.0),
        },
    )

    result = pop.fit("vi")
    # result.shared_samples["sfh_field_psd_sigma"]
    # result.shared_samples["sfh_field_psd_tau_myr"]

Notes
-----
This is a *convenience wrapper*. Under the hood it constructs a
:class:`tengri.PopulationFitter` with a model factory derived from the
SED template; existing ``PopulationFitter`` callers continue to work
unchanged. The wrapper exists to make the common hierarchical-PSD
case a one-liner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PopulationSED"]


@dataclass(frozen=True)
class PopulationSED:
    """Hierarchical population of SEDs with shared parameters.

    Holds an :class:`tengri.SEDModel` template plus N galaxy data
    dicts. Some parameters are tied across the population (the
    canonical case is the PSD hyperparameters ``σ_PSD``, ``τ_PSD``);
    everything else is fit per galaxy.

    Parameters
    ----------
    sed : SEDModel
        Template SED chain. Its physics (SFH family, dust law,
        nebular backend, …) is shared across the population; the
        priors on the *shared* parameters are kept on the template,
        and per-galaxy parameter values are drawn from the priors at
        fit time.
    galaxies : sequence of mapping
        Per-galaxy observations. Each dict must have ``flux_obs``
        and ``noise`` (photometry); optionally ``spec_obs``,
        ``spec_noise``, ``wave_spec`` for spectroscopy.
    shared : sequence of str
        Names of parameters that are tied across the population.
        Defaults to the two canonical PSD parameters.
    priors : mapping of str -> (lo, hi), optional
        Uniform-prior bounds for the shared parameters. Must contain
        an entry for every name in ``shared``. Defaults to PSD-typical
        ranges.
    data_type : str, default "photometry"
        Channel name; passed through to
        :class:`tengri.PopulationFitter`.

    Notes
    -----
    This is a SubModel-shaped *convenience* wrapper. It does not yet
    integrate with :class:`tengri.ForwardModel.predict`; calling
    ``.fit(method)`` delegates to :class:`tengri.PopulationFitter`,
    which carries the existing hierarchical-VI / EVI / raytrace
    machinery. A future refactor will absorb ``PopulationFitter``
    into the forward-model pipeline; this class will keep the same
    construction API.

    JIT/grad compatibility is inherited from :class:`PopulationFitter`.
    """

    sed: Any
    galaxies: tuple[Mapping[str, Any], ...]
    shared: tuple[str, ...] = ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr")
    priors: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "sfh_field_psd_sigma": (0.1, 4.0),
            "sfh_field_psd_tau_myr": (1.0, 300.0),
        }
    )
    data_type: str = "photometry"

    name: str = "population_sed"

    def __init__(
        self,
        sed: Any,
        galaxies: Sequence[Mapping[str, Any]],
        shared: Sequence[str] = ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
        priors: Mapping[str, tuple[float, float]] | None = None,
        data_type: str = "photometry",
    ) -> None:
        if not galaxies:
            raise ValueError("PopulationSED needs at least one galaxy.")
        object.__setattr__(self, "sed", sed)
        object.__setattr__(self, "galaxies", tuple(galaxies))
        object.__setattr__(self, "shared", tuple(shared))
        # Default PSD priors; override per shared name as the user wishes.
        default_priors = {
            "sfh_field_psd_sigma": (0.1, 4.0),
            "sfh_field_psd_tau_myr": (1.0, 300.0),
        }
        if priors is None:
            priors = default_priors
        missing = [n for n in shared if n not in priors]
        if missing:
            raise ValueError(
                f"PopulationSED.priors is missing entries for shared parameters: {missing}. "
                f"Pass priors={{name: (lo, hi), ...}} covering every shared name."
            )
        object.__setattr__(self, "priors", dict(priors))
        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "name", "population_sed")

    @property
    def n_galaxies(self) -> int:
        return len(self.galaxies)

    def fit(
        self,
        method: str = "native_vi_linear",
        *,
        key: Any = None,
        **kwargs: Any,
    ):
        """Run hierarchical inference. Delegates to :class:`PopulationFitter`.

        Parameters
        ----------
        method : str, default ``"native_vi_linear"``
            Inference method. Recognised values match
            :meth:`tengri.PopulationFitter.run`: ``"native_vi_linear"``,
            ``"native_vi_nonlinear"``, ``"vi"``, ``"mcmc_raytrace"``,
            ``"evi_nifty"``, ``"geovi"``, etc.
        key : jax.random.PRNGKey, optional
            Inference seed. If ``None``, an arbitrary seed is used.
        **kwargs : Any
            Method-specific options (e.g. ``n_burnin``, ``n_steps``,
            ``n_leapfrog_steps``, ``step_size`` for ``mcmc_raytrace``;
            ``n_iterations``, ``n_posterior_samples`` for ``vi``).

        Returns
        -------
        PopulationPosterior
            Same shape as :meth:`tengri.PopulationFitter.run`'s return.
            ``.shared_samples`` contains the per-population shared
            params; per-galaxy posteriors live on the per-galaxy
            entries.

        Notes
        -----
        See :class:`tengri.PopulationFitter` for the legacy direct API
        (factory + galaxies + per-prior bounds).
        """
        from tengri.inference.hierarchical import PopulationFitter

        # Build the model_factory that PopulationFitter expects.
        # The factory plugs the shared parameter values into the SED
        # template's spec as Fixed values, then returns the SEDModel.
        factory = self._make_model_factory()

        # PopulationFitter's API only knows about (psd_sigma, psd_tau_myr)
        # priors directly. For now, we feed it those two; future revs
        # will generalise once PopulationFitter's signature widens.
        psd_sigma_prior = self.priors.get("sfh_field_psd_sigma", (0.1, 4.0))
        psd_tau_prior = self.priors.get("sfh_field_psd_tau_myr", (1.0, 300.0))

        fitter = PopulationFitter(
            factory,
            list(self.galaxies),
            psd_sigma_prior=psd_sigma_prior,
            psd_tau_prior=psd_tau_prior,
            data_type=self.data_type,
        )
        return fitter.run(method, key=key, **kwargs)

    def _make_model_factory(self):
        """Build a ``(psd_sigma, psd_tau_myr) -> SEDModel`` closure."""
        sed = self.sed

        def factory(psd_sigma, psd_tau_myr):
            # The factory is called per-evaluation with current shared
            # values. We need a fresh SEDModel with those values set.
            # The simplest path is to clone the spec and override the
            # shared params as Fixed, then rebuild the SEDModel.
            return _rebuild_sed_with_fixed_shared(
                sed, {"sfh_field_psd_sigma": psd_sigma, "sfh_field_psd_tau_myr": psd_tau_myr}
            )

        return factory


def _rebuild_sed_with_fixed_shared(sed: Any, overrides: Mapping[str, Any]) -> Any:
    """Return a copy of ``sed`` with shared parameters fixed to override values.

    The model-factory contract used by :class:`PopulationFitter` is
    ``(psd_sigma, psd_tau_myr) -> SEDModel``; per-call, the population
    sampler picks values for the shared params and expects a fresh
    model wired to those values. This helper produces that model.

    The default implementation tries two paths:

    1. If the SED has a ``with_fixed(...)`` method, call it. This is
       the preferred future surface.
    2. Otherwise reach into ``sed.spec`` and rebuild via
       ``SEDModel(spec_with_overrides, ssp_data, observation=...)``.
    """
    # Preferred path: a ``with_fixed`` method on SEDModel.
    if hasattr(sed, "with_fixed"):
        return sed.with_fixed(**overrides)

    # Fallback: mutate spec and rebuild.
    from tengri.forward.sed_model import SEDModel
    from tengri.parameters.priors import Fixed

    spec = sed.spec
    spec_kwargs = dict(spec.kwargs) if hasattr(spec, "kwargs") else {}
    for name, value in overrides.items():
        spec_kwargs[name] = Fixed(value)
    new_spec = type(spec)(**spec_kwargs)
    ssp_data = getattr(sed, "ssp_data", None)
    observation = getattr(sed, "observation", None)
    if ssp_data is None or observation is None:
        raise RuntimeError(
            "PopulationSED could not rebuild the SEDModel: the template lacks "
            "an ssp_data or observation attribute, and no with_fixed method. "
            "Add SEDModel.with_fixed(**overrides) for a clean path."
        )
    return SEDModel(new_spec, ssp_data, observation=observation)
