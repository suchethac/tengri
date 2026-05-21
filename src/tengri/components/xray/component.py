# SPDX-License-Identifier: BSD-3-Clause
"""XRaySEDComponent: SEDComponent adapter around :func:`xray_total`.

X-ray coronal and AGN emission. Implements the SEDComponent protocol
over the X-ray bands (0.1 keV to 100 keV). Reads AGN bolometric
luminosity, stellar mass, and SFR from upstream components to compute
accretion-driven and star-formation-driven X-ray fluxes.

Cross-component reads
---------------------
X-ray depends on quantities owned by other components:

- ``sfr`` (M_⊙/yr) — produced by the stellar component as the
  current-time SFR. Read from ``state.derived["sfr"]`` with a
  fallback to 1.0.
- ``log_mstar`` (log10 M_⊙) — produced by the stellar component.
  Read from ``state.derived["log_mstar"]`` with a fallback to 10.0
  (i.e. 10¹⁰ M_⊙). X-ray's ``xray_total`` consumes the linear stellar
  mass in M_⊙, so the adapter exponentiates: ``M_* = 10**log_mstar``.
- ``L_agn_bol`` (erg/s) — produced by the AGN component. Read from
  ``state.derived["L_agn_bol"]`` with a fallback to 0.0 (no AGN).
- ``redshift`` — bare parameter from :data:`BARE_NAME_ALLOWLIST`,
  passed through but consumed by the observation model rather than
  by ``xray_total`` itself.

The contract publishes ``log_mstar`` (not ``stellar_mass``) to match
:class:`tengri.components.radio.component.RadioSEDComponent` and the
:class:`StellarSEDComponent` planned in
:doc:`/dev/phase_ii_2_stellar_migration`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.components.xray._params import PARAMS as _XRAY_PARAMS
from tengri.components.xray.xray import xray_total
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = ["XRaySEDComponent", "XRaySEDComponentConfig"]


@dataclass(frozen=True)
class XRaySEDComponentConfig(SEDComponentConfig):
    r"""Frozen knobs for :class:`XRaySEDComponent`.

    Attributes
    ----------
    name : str
        Diagnostic identifier. Default ``"xray"``.
    """

    name: str = "xray"


@dataclass(frozen=True)
class XRaySEDComponentState(SEDComponentState):
    r"""X-ray has no precomputed tensors — typed marker only."""

    name: str = "xray"


@dataclass(frozen=True)
class XRaySEDComponent:
    r"""SEDComponent adapter around :func:`xray_total`.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`apply` is pure JAX.
    **Additive**: writes ``sed_intrinsic = sed_intrinsic + L_xray(λ)``.
    Initialises ``sed_intrinsic`` from zeros if upstream did not.

    The physics covers two channels combined inside :func:`xray_total`:

    - X-ray binaries (HMXB + LMXB) scaling with SFR and M_*
      (Lehmer et al. 2010, 2016).
    - AGN corona via the alpha_ox–L_2500 relation (Lusso & Risaliti 2016).

    Both default to small (or zero) contributions when the cross-component
    reads fall back to defaults — i.e. a galaxy without an AGN gets
    XRB-only X-rays automatically.
    """

    config: XRaySEDComponentConfig = field(default_factory=XRaySEDComponentConfig)
    name: str = "xray"
    parameter_prefix: str = "xray_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this component owns.

        Returns the canonical :data:`PARAMS` tuple from
        :mod:`tengri.components.xray._params`. The legacy ``_XRAY_PARAMS``
        bucket in :mod:`tengri.parameters._param_defs` is a derived view
        of the same tuple, so the two registration paths agree by
        construction.
        """
        return list(_XRAY_PARAMS)

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys this X-ray component publishes.

        See :func:`tengri.forward.orchestrator.validate_pipeline`.

        """
        return (
            DerivedKey(
                "sed_xray",
                "erg/s/Hz",
                "X-ray luminosity contribution on pipeline wave grid",
            ),
        )

    def optional_inputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component derived keys X-ray reads *opportunistically*.

        Read from ``state.derived`` with documented fallbacks. The
        validator does NOT require an upstream publisher, but it WILL
        check that if one is present, its units match. Catches a future
        publisher rename or unit drift without forcing every pipeline
        to instantiate stellar + AGN. Phase B of #21 — see ADR-0004.
        """
        return (
            DerivedKey("sfr", "Msun/yr", "Read from stellar if present; falls back to 1.0"),
            DerivedKey(
                "log_mstar",
                "dex",
                "Read from stellar if present; falls back to 10.0",
            ),
            DerivedKey(
                "L_agn_bol",
                "erg/s",
                "Read from AGN if present; falls back to 0.0",
            ),
        )

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
        approx: Mapping[str, bool] | None = None,
        filters: tuple[tuple[jnp.ndarray, jnp.ndarray], ...] | None = None,
    ) -> XRaySEDComponentState:
        r"""No-op precompute. X-ray is a closed-form function of (λ, params)."""
        del ssp_data, wave_grid, filters
        return XRaySEDComponentState(name=self.name)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Any | None = None,
    ) -> ForwardState:
        r"""Add X-ray emission to ``state.sed_intrinsic``.

        Parameters
        ----------
        state : ForwardState
            Must carry rest-frame ``wave`` (Å). If ``sed_intrinsic`` is
            ``None`` it is initialised to zeros of the same shape.
        params : mapping
            Receives ``xray_*`` keys plus the bare ``redshift`` from
            the allowlist. Cross-component scalars (``sfr``,
            ``log_mstar``, ``L_agn_bol``) are read from
            ``state.derived`` with documented fallbacks.

        Returns
        -------
        ForwardState
            New state with ``sed_intrinsic`` updated and
            ``derived["sed_xray"]`` published for downstream readers.
        """
        wave = state.wave

        sfr = jnp.asarray(state.derived.get("sfr", 1.0))
        # Contract: stellar publishes log_mstar (log10 M_⊙). xray_total
        # takes M_* in M_⊙; exponentiate at the boundary.
        log_mstar = jnp.asarray(state.derived.get("log_mstar", 10.0))
        stellar_mass = 10.0**log_mstar
        L_agn_bol = jnp.asarray(state.derived.get("L_agn_bol", 0.0))

        L_xray = xray_total(
            wave,
            sfr=sfr,
            stellar_mass=stellar_mass,
            L_agn_bol=L_agn_bol,
            gamma_hmxb=jnp.asarray(params["xray_gamma_hmxb"]),
            gamma_lmxb=jnp.asarray(params["xray_gamma_lmxb"]),
            gamma_agn=jnp.asarray(params["xray_gamma_agn"]),
            E_cut=jnp.asarray(params["xray_E_cut"]),
            alpha_ox=jnp.asarray(params["xray_alpha_ox"]),
        )

        return state.add_intrinsic(L_xray).with_(
            derived=state.derived.with_(sed_xray=L_xray),
        )
