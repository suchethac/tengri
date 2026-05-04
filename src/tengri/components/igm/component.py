# SPDX-License-Identifier: BSD-3-Clause
"""IGMSEDComponent: SEDComponent adapter around :func:`igm_transmission`.

Phase II-1 first-adapter exercise. The physics function
:func:`tengri.components.igm.igm.igm_transmission` (Inoue et al. 2014)
is unchanged — this module is a thin wrapper that satisfies the
:class:`tengri.core.SEDComponent` Protocol so the orchestrator can run
IGM transmission alongside other Phase-II adapters.

Design choices made here (and mirrored in
:mod:`tengri.components.radio.component`):

- ``parameter_prefix = "igm_"`` for the patchy-reionization extras
  (``igm_x_HI``, ``igm_bubble_mpc``, ``igm_z_mid``, ``igm_dz``,
  ``igm_log_nhi``). The bare ``redshift`` parameter is read via
  :data:`tengri.core.component.BARE_NAME_ALLOWLIST` — IGM is the canonical
  reason that allowlist exists.
- IGM is *transmissive*: it multiplies :attr:`PipelineState.sed_observed`
  in place. If ``sed_observed`` is ``None`` the component is a no-op
  (useful when the orchestrator is being driven through unit tests
  before any rest→observed redshifting has happened).
- :meth:`precompute` is a no-op (Inoue's piecewise-power-law fit is
  evaluated lazily inside :func:`igm_transmission`; there is no
  redshift-dependent grid to precompute).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.igm.igm import igm_transmission
from tengri.core.component import (
    ParamDeclaration,
    PipelineState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.parameters.priors import Fixed

__all__ = ["IGMSEDComponent", "IGMSEDComponentConfig"]


@dataclass(frozen=True)
class IGMSEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`IGMSEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"igm"``.
    add_cgm : bool
        Add Asada+2025 CGM damping wing at z > 5. Default ``False``
        (matches :func:`igm_transmission`).
    """

    name: str = "igm"
    add_cgm: bool = False


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

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        IGM declares the CGM damping-wing knobs from
        :func:`igm_transmission`. The bare ``redshift`` parameter is
        read via :data:`BARE_NAME_ALLOWLIST` and not declared here.
        """
        return [
            ParamDeclaration(
                "igm_z_mid",
                Fixed(7.0),
                "CGM damping-wing sigmoid midpoint redshift [dimensionless]",
            ),
            ParamDeclaration(
                "igm_dz",
                Fixed(0.5),
                "CGM damping-wing sigmoid width [dimensionless]",
            ),
            ParamDeclaration(
                "igm_log_nhi",
                Fixed(20.0),
                "CGM plateau log10(N_HI / cm^-2) [dimensionless]",
            ),
        ]

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> IGMSEDComponentState:
        r"""No-op precompute. IGM transmission is evaluated at apply time."""
        del ssp_data, wave_grid
        return IGMSEDComponentState(name=self.name)

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
    ) -> PipelineState:
        r"""Multiply ``state.sed_observed`` by the Inoue+2014 transmission.

        Parameters
        ----------
        state : PipelineState
            Must carry rest-frame ``wave`` (Å). If ``sed_observed`` is
            ``None`` this returns ``state`` unchanged.
        params : mapping
            Receives ``igm_*`` keys plus the bare ``redshift`` from the
            allowlist.

        Returns
        -------
        PipelineState
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

        T = igm_transmission(
            wave_obs,
            z,
            add_cgm=self.config.add_cgm,
            cgm_z_mid=jnp.asarray(params.get("igm_z_mid", 7.0)),
            cgm_dz=jnp.asarray(params.get("igm_dz", 0.5)),
            cgm_log_nhi=jnp.asarray(params.get("igm_log_nhi", 20.0)),
        )

        new_derived = dict(state.derived)
        new_derived["igm_transmission"] = T
        return state.with_(
            sed_observed=state.sed_observed * T,
            derived=new_derived,
        )
