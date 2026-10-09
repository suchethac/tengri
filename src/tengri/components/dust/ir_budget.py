# SPDX-License-Identifier: BSD-3-Clause
"""Publish an explicit dust luminosity when attenuation is disabled."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.sed_model_component import SEDModelComponent
from tengri.protocols.component import ForwardState

__all__ = ["DustIRBudgetComponent"]


class DustIRBudgetComponent(SEDModelComponent):
    """Publish the requested IR luminosity for a standalone dust emitter."""

    name: str = "dust_ir_budget"
    parameter_prefix: str = "dust_"
    requires_template_data: ClassVar[bool] = False
    reads_parameters: ClassVar[frozenset[str]] = frozenset({"dust_log_L_ir"})
    outputs: ClassVar[dict[str, str]] = {"L_ir": "erg/s", "log_L_ir": "dex"}

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Mapping[str, Any] | None = None,
        ztable_data: Any | None = None,
    ) -> ForwardState:
        """Publish the independent luminosity without creating projection LUTs."""
        del ssp_data, template_data, ztable_data
        sed_in = state.sed_intrinsic
        if sed_in is None:
            sed_in = jnp.zeros_like(state.wave)
        sed_out, published = self.predict(self.slice_params(params), sed_in, state.wave)
        derived = self._merge_published(state.derived, published)
        return state.with_(sed_intrinsic=sed_out, derived=derived)

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        r"""Publish the explicit dust luminosity without changing the SED.

        Parameters
        ----------
        p : mapping[str, array_like]
            Prefix-stripped dust parameters. ``log_L_ir`` is
            :math:`\log_{10}(L_{\rm IR}/L_\odot)` [dex].
        sed_in : array_like, shape (n_wave,)
            Input spectral luminosity density [erg/s/Hz].
        wave : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Å].

        Returns
        -------
        tuple[ndarray, dict[str, ndarray]]
            The unchanged input SED and the ``L_ir`` [erg/s] and
            ``log_L_ir`` [dex] values.

        Notes
        -----
        **JIT-compatible**: yes. The component uses JAX array operations only.

        The conversion is

        .. math::

            \log_{10}(L_{\rm IR}/[\mathrm{erg\,s^{-1}}]) =
            \texttt{dust\_log\_L\_ir} + \log_{10}(L_\odot/[\mathrm{erg\,s^{-1}}]).

        Here :math:`L_{\rm IR}` is the requested total IR luminosity [erg/s],
        :math:`L_\odot` is Tengri's solar luminosity conversion [erg/s], and
        ``dust_log_L_ir`` is the user parameter [dex].
        """
        del wave
        from tengri.utils.scale import pow10
        from tengri.utils.sed_quantities import LOG10_L_SUN

        log_l_ir = jnp.asarray(p["log_L_ir"]) + LOG10_L_SUN
        return sed_in, {"L_ir": pow10(log_l_ir), "log_L_ir": log_l_ir}
