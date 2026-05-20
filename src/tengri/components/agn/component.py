# SPDX-License-Identifier: BSD-3-Clause
"""AGNSEDComponent: unified AGN emission (disc + torus + lines + jets/corona).

Dispatches to any registered AGN model (``"simple"``, ``"standard"``,
``"kubota_done_full"``, ``"adaf"``, ``"unified_nlr_blr"``, etc.) via
:func:`tengri.components.agn.unified.resolve_agn_model`.

Cross-component reads
---------------------
Nothing required — AGN is a self-contained additive emitter. ``redshift``
is read from :data:`BARE_NAME_ALLOWLIST`.

Cross-component publications
----------------------------
- ``state.derived["L_agn_bol"]`` (scalar, erg/s) — bolometric AGN
  luminosity. Consumed by
  :class:`tengri.components.xray.XRaySEDComponent` and
  :class:`tengri.components.radio.RadioSEDComponent` via their
  documented fallback (``state.derived.get("L_agn_bol", 0.0)``).
- ``state.derived["sed_agn"]`` — the AGN SED contribution
  (erg/s/Hz, shape n_wave) for diagnostics.

Architectural notes
-------------------
``agn_torus_frac`` is **never** auto-derived from ``agn_cos_inc`` /
``theta_torus`` (CLAUDE.md gotcha — gradient discontinuity). It is
read directly from ``params`` as an independent free parameter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.agn._params import PARAMS as _AGN_PARAMS
from tengri.components.agn.unified import resolve_agn_model
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.utils.physics_constants import L_SUN

__all__ = ["AGNSEDComponent", "AGNSEDComponentConfig"]


@dataclass(frozen=True)
class AGNSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for :class:`AGNSEDComponent`.

    Parameters
    ----------
    name : str
        Diagnostic identifier. Default ``"agn"``.
    model : str
        AGN model registry key. One of ``"simple"`` (powerlaw disc +
        single-temp torus), ``"standard"`` (multicolor disc +
        two-temp torus), ``"kubota_done_full"``, ``"adaf"``,
        ``"unified_nlr_blr"``, etc. Default ``"simple"``.
    """

    name: str = "agn"
    model: str = "simple"


@dataclass(frozen=True)
class AGNSEDComponentState(SEDComponentState):
    """Marker state — AGN models are analytic; no precomputed tensors."""

    name: str = "agn"


@dataclass(frozen=True)
class AGNSEDComponent:
    """SEDComponent adapter for the unified AGN model registry.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` delegates to a registered
    AGN function which is pure JAX.
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_AGN(λ)``,
    matching the convention used by :class:`RadioSEDComponent` and
    :class:`XRaySEDComponent`.
    """

    config: AGNSEDComponentConfig = field(default_factory=AGNSEDComponentConfig)
    name: str = "agn"
    parameter_prefix: str = "agn_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        :mod:`tengri.components.agn._params`. The legacy ``_AGN_PARAMS``
        bucket in :mod:`tengri.parameters._param_defs` is a derived view
        of the same tuple (plus the ``neb_xid`` orphan kept in the
        registry for the Feltre NLR backend).

        The full ``agn_*`` parameter superset is declared so that any
        registered model can run without missing keys. Users freely
        ``Fixed`` whatever a particular model does not consume; the
        AGN registry functions accept ``**kwargs`` and ignore unused
        names.
        """
        return list(_AGN_PARAMS)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this AGN component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.
        """
        return (
            DerivedKey("L_agn_bol", "erg/s", "AGN bolometric luminosity"),
            DerivedKey("sed_agn", "erg/s/Hz", "AGN SED contribution on pipeline wave grid"),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> AGNSEDComponentState:
        """No-op marker — all AGN templates are evaluated at runtime."""
        del ssp_data, wave_grid, filters
        return AGNSEDComponentState(name=self.name)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Add AGN emission to ``state.sed_intrinsic`` and publish ``L_agn_bol``.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialised to zeros of the same shape.
        params : mapping
            Receives ``agn_*`` keys plus the bare ``redshift``.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` updated and ``L_agn_bol``
            published to ``derived``.
        """
        wave = state.wave

        # Convert log10(L_bol/Lsun) → erg/s once for both the model
        # call and the L_agn_bol publication.
        agn_log_lbol = jnp.asarray(params["agn_log_lbol"])
        L_agn_bol = jnp.power(10.0, agn_log_lbol) * L_SUN

        # Resolve the AGN model from the registry. resolve_agn_model is
        # a factory-time lookup but the returned callable is pure JAX
        # so it folds into a JIT trace cleanly.
        agn_fn = resolve_agn_model(self.config.model)
        L_agn = agn_fn(
            wave,
            agn_log_lbol=agn_log_lbol,
            agn_frac=jnp.asarray(params.get("agn_frac", 1.0)),
            agn_alpha=jnp.asarray(params.get("agn_alpha", -1.0)),
            agn_log_mbh=jnp.asarray(params.get("agn_log_mbh", 8.0)),
            agn_log_ledd=jnp.asarray(params.get("agn_log_ledd", -1.0)),
            agn_a_spin=jnp.asarray(params.get("agn_a_spin", 0.0)),
            agn_torus_frac=jnp.asarray(params.get("agn_torus_frac", 0.5)),
            agn_T_torus=jnp.asarray(params.get("agn_T_torus", 1000.0)),
            agn_tau_torus=jnp.asarray(params.get("agn_tau_torus", 3.0)),
            agn_T_hot=jnp.asarray(params.get("agn_T_hot", 1500.0)),
            agn_T_warm=jnp.asarray(params.get("agn_T_warm", 300.0)),
            agn_frac_hot=jnp.asarray(params.get("agn_frac_hot", 0.5)),
            agn_cos_inc=jnp.asarray(params.get("agn_cos_inc", 0.5)),
            agn_polar_ebv=jnp.asarray(params.get("agn_polar_ebv", 0.0)),
            agn_polar_oa=jnp.asarray(params.get("agn_polar_oa", 40.0)),
            agn_ebv_disc=jnp.asarray(params.get("agn_ebv_disc", 0.0)),
        )

        return state.add_intrinsic(L_agn).with_(
            derived=state.derived.with_(L_agn_bol=L_agn_bol, sed_agn=L_agn),
        )
