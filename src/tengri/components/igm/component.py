# SPDX-License-Identifier: BSD-3-Clause
"""IGMSEDComponent: intergalactic-medium transmission as a SEDComponent.

A thin wrapper around :func:`tengri.components.igm.igm.igm_transmission`
(Inoue et al. 2014) that satisfies the
:class:`tengri.protocols.SEDComponent` contract.

Design choices (mirrored in :mod:`tengri.components.radio.component`):

- ``parameter_prefix = "igm_"`` for the patchy-reionization extras
  (``igm_x_HI``, ``igm_bubble_mpc``, ``igm_z_mid``, ``igm_dz``,
  ``igm_log_nhi``). The bare ``redshift`` is read via
  :data:`tengri.protocols.component.BARE_NAME_ALLOWLIST` — IGM is the
  canonical reason that allowlist exists.
- IGM is *transmissive*: it multiplies :attr:`ForwardState.sed_observed`
  in place. If ``sed_observed`` is ``None`` the component is a no-op
  (useful in unit tests run before the rest→observed redshifting step).
- :meth:`precompute` is a no-op (Inoue's piecewise-power-law fit is
  evaluated lazily inside :func:`igm_transmission`; no redshift-dependent
  grid needs caching).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.igm._params import PARAMS as _IGM_PARAMS
from tengri.components.igm.igm import igm_absorption
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = ["IGMSEDComponent", "IGMSEDComponentConfig"]


@dataclass(frozen=True)
class IGMSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`IGMSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"igm"``.
    igm_model : str
        Mean-IGM transmission model resolved from the registry: ``"inoue"``
        (Inoue+2014, default), ``"madau"`` (Madau+1995), ``"meiksin06"``
        (Meiksin 2006), or ``"asada25"`` (Inoue + Asada+2025 CGM damping
        wing). Threaded from ``spec.igm_model`` so the observed-frame
        photometry and spectroscopy honor the configured model rather than
        always falling back to Inoue.
    igm_patchy : bool
        Use the patchy-reionization damping-wing model instead of the mean
        IGM. Default ``False``.
    use_dla : bool
        Multiply by a damped-Lyman-α absorber (params read at apply time),
        so photometry/spectroscopy see the DLA — not only
        ``predict_obs_sed``. Default ``False``.
    """

    name: str = "igm"
    igm_model: str = "inoue"
    igm_patchy: bool = False
    use_dla: bool = False


@dataclass(frozen=True)
class IGMSEDComponentState(SEDComponentState):
    r"""IGM has no precomputed tensors — this is just a typed marker."""

    name: str = "igm"


@dataclass(frozen=True)
class IGMSEDComponent:
    r"""SEDComponent adapter around :func:`igm_transmission`.

    Parameters
    ----------
    config : IGMSEDComponentConfig, optional
        Frozen structural settings. Default :class:`IGMSEDComponentConfig`.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX.
    **Transmissive**: writes ``sed_observed = sed_observed * T(λ)``.
    Components are no-ops when ``sed_observed is None``.
    """

    config: IGMSEDComponentConfig = field(default_factory=IGMSEDComponentConfig)
    name: str = "igm"
    parameter_prefix: str = "igm_"

    def citations(self) -> tuple[str, ...]:
        """IGM transmission backend (Inoue+2014 / Madau+1995) is config-driven;
        see :data:`tengri.citations.associations.IGM_CITATIONS`."""
        return ()

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        :mod:`tengri.components.igm._params` — the CGM damping-wing
        knobs read by :func:`igm_transmission`. The bare ``redshift``
        parameter is read via :data:`BARE_NAME_ALLOWLIST` and not
        declared here.
        """
        return list(_IGM_PARAMS)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this IGM component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        return (
            DerivedKey(
                "igm_transmission",
                "",
                "Inoue+2014 transmission T(lambda) on observed-frame grid",
            ),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> IGMSEDComponentState:
        r"""No-op precompute. IGM transmission is evaluated at apply time."""
        del ssp_data, wave_grid, filters
        return IGMSEDComponentState(name=self.name)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        r"""Multiply ``state.sed_observed`` by the Inoue+2014 transmission.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_observed`` is
            ``None`` this returns ``state`` unchanged.
        params : mapping
            Receives ``igm_*`` keys plus the bare ``redshift`` from the
            allowlist.

        Returns
        -------
        ForwardState
            New state with ``sed_observed *= T_IGM(λ_obs, z)``.

        Notes
        -----
        Observed-frame wavelength is :math:`\lambda_{\rm obs} = (1 + z)\,\lambda_{\rm rest}`.
        The transmission curve from Inoue+2014 takes observed-frame Å.
        """
        if state.sed_observed is None:
            return state

        z = jnp.asarray(params["redshift"])
        wave_obs = state.wave * (1.0 + z)

        # Single flat dispatch honoring the configured mean-IGM model and DLA
        # (was hardcoded to Inoue with no DLA, so the observed-frame
        # photometry/spectroscopy projection silently ignored both — #932).
        dla_z = params.get("dla_z", 0.0)
        T = igm_absorption(
            wave_obs,
            z,
            igm_x_HI=params.get("igm_x_HI", 0.0),
            igm_bubble_mpc=params.get("igm_bubble_mpc", 10.0),
            igm_patchy=self.config.igm_patchy,
            igm_model=self.config.igm_model,
            use_dla=self.config.use_dla,
            dla_z=dla_z,
            dla_log_n_hi=params.get("dla_log_n_hi", 20.0),
            dla_temp=params.get("dla_temp", 1e4),
            dla_b_turb=params.get("dla_b_turb", 0.0),
        )

        return state.with_(
            sed_observed=state.sed_observed * T,
            derived=state.derived.with_(igm_transmission=T),
        )


# Register in the unified component dispatch table so build_components resolves
# the IGM component via _resolve_registry_component (single dispatch, #845)
# instead of importing the class directly.
from tengri.components.sed_model_component import _REGISTRY

_REGISTRY["igm"] = IGMSEDComponent
