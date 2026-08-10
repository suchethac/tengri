# SPDX-License-Identifier: BSD-3-Clause
"""EmissionComponent: Abstract base class for dust IR emission models.

Centralizes the WavePrecomp photometry-LUT projection logic for all dust
emission templates (analytic and grid-based), recovering the pre-slim
DustSEDComponent.apply() orchestration for the three dispatch branches:

1. Photometry LUT (WavePrecomp): project full-wave emission onto filters
2. Spectrum LUT (SpectrumPrecomp): project onto spectrum pixels
3. Exact full-wave: compute the full SED

Concrete components (ModifiedBlackbodyIRSEDComponent, DaleCaseyIRSEDComponent,
...) inherit from EmissionComponent and focus ONLY on their predict() physics,
with no need to override apply().
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.resolve import require_redshift
from tengri.protocols.component import BARE_NAME_ALLOWLIST, ForwardState

__all__ = ["EmissionComponent"]


class EmissionComponent(SEDModelComponent):
    """Abstract base for all dust IR emission models.

    Subclasses declare free parameters and define predict() physics.
    apply() is inherited and handles the three dispatch branches automatically.

    Notes
    -----
    **Registry**: Concrete subclasses (e.g., modified_blackbody, dale2014)
    register in _REGISTRY under their ``name``. EmissionComponent itself does NOT
    register — it is abstract (name not in vars(cls) at class-definition time).

    **JIT-compatible**: yes — all orchestration is pure JAX.
    """

    # Shared class attributes for all emission components
    parameter_prefix: str = "dust_"

    #: Opt-in: when True, :meth:`apply` passes this component's template bundle
    #: to :meth:`predict` as ``templates=``, taken from
    #: ``template_data["dust_ir"][self.name]`` and falling back to ``self.data``.
    #:
    #: It is opt-in rather than automatic because ``predict`` signatures are
    #: written per backend; passing an unexpected keyword to every one of them
    #: would break the analytic components, which have no library at all. A
    #: template-backed backend sets this and accepts ``templates=None``.
    #:
    #: Without threading, a backend that loads its grid inside ``predict``
    #: freezes the whole library into the graph as ``Constant`` ops — measured
    #: at 66.6 MB for Draine & Li 2014 (#1649).
    accepts_threaded_templates: ClassVar[bool] = False

    def threaded_templates(self, template_data: Mapping[str, Any] | None) -> Any | None:
        """Return this component's pre-loaded template bundle, if any.

        Parameters
        ----------
        template_data : mapping, optional
            The nested threading dict; the ``"dust_ir"`` slot is namespaced by
            component ``name``.

        Returns
        -------
        object or None
            The threaded bundle, else ``self.data`` (set by ``precompute`` via
            :meth:`load`), else ``None`` — in which case the backend falls back
            to its own module-level load and bakes.
        """
        if isinstance(template_data, dict):
            dust_ir = template_data.get("dust_ir")
            if isinstance(dust_ir, dict):
                found = dust_ir.get(self.name)
                if found is not None:
                    return found
        return getattr(self, "data", None)

    # Cross-component contract: all components consume L_ir and produce dust IR SED.
    # SEDModelComponent.__init_subclass__ collects these dicts into the DerivedKey
    # tuples and rebinds the shadowed accessor methods onto concrete subclasses.
    # EmissionComponent itself is abstract (defines no own ``name``) so it does not register.
    optional_inputs: ClassVar[dict[str, str]] = {"L_ir": "erg/s"}
    outputs: ClassVar[dict[str, str]] = {"sed_dust_ir": "erg/s/Hz"}

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
        ssp_data: Any | None = None,
        template_data: Mapping[str, Any] | None = None,
    ) -> ForwardState:
        """Apply dust IR emission with WavePrecomp support.

        Orchestrates three branches:
        1. Photometry LUT (filter_eff_waves in state.derived) — project
           full-wave emission onto filters via integral (exact) or effective
           wavelength sample (approximate).
        2. Spectrum LUT (spec_eff_waves in state.derived) — project onto
           spectrum pixels.
        3. Exact full-wave — compute full SED and add to state.sed_intrinsic.

        Parameters
        ----------
        state : ForwardState
            Current pipeline state with wave, sed_intrinsic, derived keys.
        params : mapping
            Full parameter dict (sliced by prefix inside).
        ssp_data : object, optional
            SSP data (ignored for emission components).
        template_data : mapping, optional
            Cached templates and LUTs (e.g., dust_ir["energy_balance_lut"],
            dust_ir["emission_band_response"]).

        Returns
        -------
        ForwardState
            Updated state with sed_intrinsic and derived keys.
        """
        # Slice parameters: strip prefix
        prefix_len = len(self.parameter_prefix)
        p_sliced = {
            k[prefix_len:]: v for k, v in params.items() if k.startswith(self.parameter_prefix)
        }

        # Bare-name allowlist (e.g. redshift) — pass through unstripped
        for _bare in BARE_NAME_ALLOWLIST:
            if _bare in params:
                p_sliced[_bare] = params[_bare]

        # Look up optional inputs (using the method from base class, not the dict attribute)
        input_kwargs = {}
        for opt_key in super().optional_inputs():
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

        # Detect WavePrecomp and SpectrumPrecomp branches
        spec_eff_waves = state.derived.get("spec_eff_waves")
        filter_eff_waves = state.derived.get("filter_eff_waves")

        if spec_eff_waves is not None or filter_eff_waves is not None:
            # ONE full-grid evaluation, shared by every consumer below. Each LUT branch
            # used to recompute it, which jit made free (CSE) but eager execution did not
            # — and predict_state, Prediction, and most of the test suite run eager.
            # Build a SEPARATE dict for predict. Mutating ``input_kwargs`` would
            # also inject ``templates`` into the two ``_apply_*_precomp`` helpers
            # below, which forward it with ``**input_kwargs`` and do not accept
            # it — that leaked through to the nebular line-catalog path and broke
            # tests/contract/test_line_ratio_data.py.
            predict_kwargs = dict(input_kwargs)
            if self.accepts_threaded_templates:
                predict_kwargs["templates"] = self.threaded_templates(template_data)
            sed_ir, published_full = self.predict(
                p_sliced, jnp.zeros_like(state.wave), state.wave, **predict_kwargs
            )

            # LUT path: publish the precomp families the LUT projectors consume...
            published: dict[str, Any] = {}

            if filter_eff_waves is not None:
                published.update(
                    self._apply_photometry_precomp(
                        p_sliced, state, filter_eff_waves, template_data, sed_ir, **input_kwargs
                    )
                )

            if spec_eff_waves is not None:
                published.update(
                    self._apply_spectrum_precomp(
                        p_sliced, state, spec_eff_waves, sed_ir, **input_kwargs
                    )
                )

            # ...and STILL add to sed_intrinsic. This used to return without touching it,
            # which left the panchromatic model SED of every WavePrecomp model with NO
            # dust IR at all: ``Prediction.photometry()`` (exact by default, #1097) read
            # 5.8x low in W3 and 6x low in W4 — bit-identical to a model built with no
            # dust emission — while the likelihood, which reads the LUT families, was
            # correct. So fits were fine and every best-fit overlay, residual plot, and
            # mid-IR diagnostic drawn from one was silently missing the IR bump.
            #
            # It costs nothing: the fast path is fast because ``predict_via_precomp``
            # never READS ``sed_intrinsic``, so XLA prunes the full-grid chain outright.
            # Writing an array nobody reads is still dead code. Radio and X-ray have
            # always added unconditionally and still compile to ~143 us.
            # An emission component is additive by contract, so sed_in + sed_ir is exactly
            # what predict(p, sed_in, wave) returns — pinned by
            # tests/regression/bug/test_precomp_sed_intrinsic_completeness.py, which
            # compares this against the exact path.
            new_derived = self._merge_published(state.derived, {**published_full, **published})
            return state.with_(sed_intrinsic=sed_in + sed_ir, derived=new_derived)
        else:
            # Exact full-wave path
            predict_kwargs = dict(input_kwargs)
            if self.accepts_threaded_templates:
                predict_kwargs["templates"] = self.threaded_templates(template_data)
            sed_out, published = self.predict(p_sliced, sed_in, state.wave, **predict_kwargs)
            new_derived = self._merge_published(state.derived, published)
            return state.with_(sed_intrinsic=sed_out, derived=new_derived)

    def _apply_photometry_precomp(
        self,
        p: Mapping[str, jnp.ndarray],
        state: ForwardState,
        filter_eff_waves: jnp.ndarray,
        template_data: Mapping[str, Any] | None,
        sed_ir: jnp.ndarray,
        **inputs: Any,
    ) -> Mapping[str, jnp.ndarray]:
        """Project dust IR emission onto photometry filters.

        Recovers pre-slim DustSEDComponent.apply() photometry-LUT logic with
        four branches:
        1. band_response (linear models like Dale2014) — exact, fast
        2. fast_emission — effective-wavelength sample (approximate)
        3. padded_curves — full filter integral (exact)
        4. fallback — effective-wavelength sample (default)

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped.
        state : ForwardState
            Pipeline state (provides wave, derived keys).
        filter_eff_waves : ndarray, shape (n_filter,)
            Rest-frame filter effective wavelengths in Angstrom.
        template_data : mapping, optional
            Cached LUTs and responses (dust_ir dict).
        **inputs : ndarray
            Cross-component inputs (L_ir, etc.).

        Returns
        -------
        mapping[str, ndarray]
            Published dict with dust_emission_phot_lnu_precomp key.
        """
        # ``sed_ir`` is the full-grid emission, computed ONCE by apply() and shared with
        # the spectrum branch and with sed_intrinsic. Recomputing it per branch was three
        # identical full-grid evaluations: free under jit (CSE + DCE), but real work in
        # eager mode, which is what predict_state and the test suite actually run.
        L_ir = jnp.asarray(inputs.get("L_ir", 0.0))

        # Try to get band response (exact for linear models)
        band_response = None
        if isinstance(template_data, dict):
            _dir = template_data.get("dust_ir")
            if isinstance(_dir, dict):
                band_response = _dir.get("emission_band_response")

        # Project emission onto filters
        if band_response is not None:
            # Exact fast path: template is linear in L_ir, so integral = L_ir * R
            phot_lnu = L_ir * band_response
        elif getattr(self, "fast_emission", False):
            # Approximate path: sample at effective wavelength
            phot_lnu = jnp.interp(filter_eff_waves, state.wave, sed_ir)
        else:
            # Exact path: full filter integral (when padded curves are available)
            fw_pad = state.derived.get("phot_filter_waves_padded")
            ft_pad = state.derived.get("phot_filter_trans_padded")
            if fw_pad is not None:
                from tengri.observation.photometry import lnu_filter_integral_batch

                redshift = jnp.asarray(
                    require_redshift(
                        p, "components.dust.emission._component_base._apply_photometry_precomp"
                    )
                )
                phot_lnu = lnu_filter_integral_batch(sed_ir, state.wave, fw_pad, ft_pad, redshift)
            else:
                # Fallback: effective wavelength sample (no padded curves)
                phot_lnu = jnp.interp(filter_eff_waves, state.wave, sed_ir)

        return {"dust_emission_phot_lnu_precomp": phot_lnu}

    def _apply_spectrum_precomp(
        self,
        p: Mapping[str, jnp.ndarray],
        state: ForwardState,
        spec_eff_waves: jnp.ndarray,
        sed_ir: jnp.ndarray,
        **inputs: Any,
    ) -> Mapping[str, jnp.ndarray]:
        """Project dust IR emission onto spectrum pixels.

        Similar to photometry LUT but for spectrum pixels: sample the
        full-wave emission at spectrum effective wavelengths.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped.
        state : ForwardState
            Pipeline state.
        spec_eff_waves : ndarray, shape (n_pixel,)
            Rest-frame spectrum effective wavelengths in Angstrom.
        **inputs : ndarray
            Cross-component inputs (L_ir, etc.).

        Returns
        -------
        mapping[str, ndarray]
            Published dict with dust_emission_spec_lnu_precomp key.
        """
        # ``sed_ir`` is the shared full-grid emission from apply() (see the photometry
        # branch). Sample at spectrum pixels.
        spec_lnu = jnp.interp(spec_eff_waves, state.wave, sed_ir)

        return {"dust_emission_spec_lnu_precomp": spec_lnu}

    def predict(
        self, p: Mapping[str, jnp.ndarray], sed_in: jnp.ndarray, wave: jnp.ndarray, **inputs: Any
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX emission prediction. MUST be implemented by concrete subclass.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped.
        sed_in : ndarray, shape (n_wave,)
            Input SED (ignored for emission — typically zeros).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            L_ir (scalar) and other inputs.

        Returns
        -------
        tuple[ndarray, mapping]
            (sed_ir, {"sed_dust_ir": sed_ir}) where sed_ir is the emission
            in erg/s/Hz.
        """
        raise NotImplementedError(f"{type(self).__name__}.predict() must be implemented")
