# SPDX-License-Identifier: BSD-3-Clause
"""Unified AGN SED: combined disc + torus + NLR + BLR emission.

Combines accretion disc and dust torus models into a single AGN SED.
The disc emission is partly absorbed by the torus (via covering factor)
and re-emitted in the IR.

Architecture overview
---------------------
The component tree and geometric masking design in this module is inspired
by the ``UnifiedAGN`` class in the Synthesizer package
(Lovell et al. 2025, Open J. Astrophys. 8, doi:10.33232/001c.145766;
Roper et al. 2026, JOSS 11, 9436, doi:10.21105/joss.09436;
https://github.com/synthesizer-project/synthesizer).
Per the Synthesizer citation policy, BOTH papers must be cited together.

Synthesizer's model defines:

- An accretion disc whose inclination-dependent emission is extracted from
  precomputed photoionization **grids** (CLOUDY-based).
- NLR emission as grid-extracted nebular spectra, *not* masked by the torus
  (isotropic: always visible at all inclinations).
- BLR emission as grid-extracted nebular spectra, masked by the torus using a
  **hard binary condition**: ``inclination + theta_torus > 90°`` → zeroed.
- A torus that reprocesses the isotropic disc emission, scaled by
  ``torus_fraction = theta_torus / 90°`` (geometric fraction of sky covered).
- Optional diffuse dust attenuation layered on top.

Differences in this implementation
-----------------------------------
tengri adopts the same conceptual decomposition but diverges in several ways
for compatibility with gradient-based inference (VI, HMC):

1. **Analytic disc models, not grids**: disc emission uses the closed-form
   multicolor/Shakura-Sunyaev or power-law models from ``disc.py`` rather
   than a photoionization grid. This makes the disc JIT/grad-compatible.

2. **Analytic NLR/BLR templates, not grids**: NLR and BLR use empirically
   calibrated Gaussian line templates (Vanden Berk et al. 2001; Groves et al.
   2004) rather than CLOUDY-generated photoionization grids. Grids would
   require non-differentiable table look-ups over (U, n_H) axes.

3. **Smooth sigmoid geometric mask, not a hard binary**: Synthesizer zeros
   the disc/BLR whenever ``inclination + theta_torus > 90°``. tengri replaces
   this with a smooth sigmoid centered at the critical angle (see
   ``_sigmoid_mask``). The sigmoid preserves the gradient through the
   inclination parameter, which would otherwise be zero almost everywhere
   under the hard mask.

4. **Explicit ``agn_torus_frac`` parameter, not derived from theta_torus**:
   Synthesizer internally computes ``torus_fraction = theta_torus / 90°``,
   coupling the torus covering factor to the opening angle. tengri keeps
   ``agn_torus_frac`` as an independent free parameter. This decoupling is
   intentional: deriving the covering factor from ``cos(theta_torus)`` in
   the forward pass introduces a gradient discontinuity at ``theta_torus=0``
   and at ``theta_torus=90°`` (see CLAUDE.md gotcha).

5. **Polar dust reddening (SMC law)**: an optional ``agn_polar_ebv`` parameter
   applies SMC-law extinction to the disc and BLR for Type 1 sightlines.
   This is absent from Synthesizer's basic UnifiedAGN but present in the
   CIGALE ``skirtor2016`` module (Yang et al. 2020, MNRAS, 491, 740).

Pre-registered configurations
------------------------------

- **multicolor_agn** (= deprecated alias ``kubota_done``): multi-color disc with
  BH physics + 2-T torus (8+ params).
- **kubota_done_full**: full Kubota & Done 3-zone disc + 2-T torus (13+ params).
- **adaf**: faithful Mahadevan 1997 ADAF + Silva+04 IR torus for LLAGN.
- **unified_nlr_blr**: full Synthesizer-inspired model with NLR/BLR + polar dust.
- **skirtor**: power-law disc + SKIRTOR clumpy torus (Stalevski+2012, 2016).
- **silva04**: power-law disc + Silva+04 smooth torus.
- **cat3d_wind**: power-law disc + CAT3D-Wind clumpy torus.
- **relagn**: RELAGN relativistic disc + 2-T torus.

.. warning::

   ``agn_log_lbol`` is **log10(L_bol / L_sun)**, not log10(L_bol / [erg/s]).
   See the convention note below before setting this parameter.

Convention for ``agn_log_lbol``
--------------------------------
``agn_log_lbol`` is :math:`\\log_{10}(L_{\\rm bol} / L_\\odot)`: the
bolometric luminosity expressed **in solar luminosities**, not erg/s.
This matches the internal computation in ``components/agn/_phys.py``
(``l_bol_erg = 10**agn_log_lbol * L_SUN``).

* Typical bright Seyfert: :math:`L_{\\rm bol}\\!\\sim\\!10^{44}` erg/s
  :math:`\\Rightarrow` ``agn_log_lbol = 10.5``.
* Bright quasar: :math:`L_{\\rm bol}\\!\\sim\\!10^{46}` erg/s
  :math:`\\Rightarrow` ``agn_log_lbol = 12.5``.
* Synthesizer's ``bolometric_luminosity`` is in **erg/s**; convert with
  ``agn_log_lbol = log10(L_bol_erg) - log10(L_SUN_erg)``,
  i.e. subtract :math:`\\approx 33.58` from synthesizer's value.

Default ``agn_log_lbol=11.0`` in the function signatures of this module
corresponds to a bolometric luminosity of ~10^44 erg/s,
a typical bright Seyfert nucleus. This is a physically reasonable default for AGN
fitting; always set this parameter explicitly in production to match your target.

Usage::

    from tengri.components.agn.unified import unified_agn, resolve_agn_model

    # Use a named configuration. agn_log_lbol = 11 → L_bol ≈ 4e44 erg/s
    # (a typical bright Seyfert nucleus).
    model_fn = resolve_agn_model("multicolor_agn")
    l_nu = model_fn(wavelength, agn_log_lbol=11.0, agn_lum_ratio=0.1, ...)

    # Or use the generic combiner directly.
    l_nu = unified_agn(wavelength, agn_log_lbol=11.0, disc_model="multicolor", ...)

References
----------
.. [1] Lovell C. C. et al. 2025, Open Journal of Astrophysics, 8,
       "Synthesizer: a Software Package for Synthetic Astronomical Observables",
       https://doi.org/10.33232/001c.145766
.. [2] Roper W. J. et al. 2026, Journal of Open Source Software, 11, 9436,
       "Synthesizer: Synthetic Observables for Modern Astronomy",
       https://doi.org/10.21105/joss.09436
       (Both Synthesizer papers [1]_ [2]_ must be cited together.)
.. [3] synthesizer source: https://github.com/synthesizer-project/synthesizer
       (src/synthesizer/emission_models/agn/unified_agn.py)
"""

import functools
import warnings
from collections.abc import Callable

import jax.numpy as jnp

from tengri.components.agn._params import (
    DEFAULT_AGN_COS_INC,
    DEFAULT_AGN_LOG_LBOL,
    DEFAULT_AGN_LOG_MBH,
    DEFAULT_AGN_LUM_RATIO,
)
from tengri.components.agn.adaf import adaf_spectrum
from tengri.components.agn.blr import compute_blr_sed
from tengri.components.agn.cat3d_wind import cat3d_wind_sed
from tengri.components.agn.disc import (
    create_relagn_disc_from_grid,
    kubota_done_disc,
    multicolor_disc,
    powerlaw_disc,
)
from tengri.components.agn.nlr import compute_nlr_sed
from tengri.components.agn.reddening import redden_disc as _redden_disc
from tengri.components.agn.silva04 import silva04_sed
from tengri.components.agn.skirtor import _find_skirtor_grid, create_skirtor_from_grid
from tengri.components.radio.radio import radio_total
from tengri.components.xray.xray import (
    _xray_agn_corona_bolometric as _xray_agn_corona_legacy,
)
from tengri.utils.physics_constants import C_AA as _C_AA, L_SUN as _LSUN_ERG


@functools.cache
def _load_skirtor_fn():
    """Load SKIRTOR template grid from file.

    Delegates to ``skirtor._find_skirtor_grid()`` which searches v3, v2,
    and npz formats in priority order.
    """
    return create_skirtor_from_grid(_find_skirtor_grid())


@functools.cache
def _load_skirtor_raw_total_fn():
    """Load the faithful raw-Stalevski SKIRTOR total interpolator (cached).

    Prefers the v4 grid (``scripts/build_skirtor_raw_grid.py``): full
    ``ta,p,q,oa,R,i`` axes with the published radiative-transfer total: and
    returns ``(fn, has_radius_ratio=True)``. If the v4 grid is absent, falls
    back to the v3 component interpolator's reconstructed ``.total`` (no R axis),
    returning ``(fn, has_radius_ratio=False)``. Used by ``skirtor_stalevski``.
    """
    from tengri.components.agn.skirtor import (
        _find_skirtor_raw_grid,
        create_skirtor_components_from_grid,
        create_skirtor_raw_total_from_grid,
    )

    v4 = _find_skirtor_raw_grid()
    if v4 is not None:
        return create_skirtor_raw_total_from_grid(v4), True
    comp_fn = create_skirtor_components_from_grid(_find_skirtor_grid())
    return (lambda *a, **k: comp_fn(*a, **k).total), False


def _find_relagn_grid() -> str:
    """Locate the RELAGN outer-disc grid file.

    Searches ``data/relagn_disc_grid.h5`` relative to the repository root.

    Raises
    ------
    FileNotFoundError
        If no grid file is found. Run ``scripts/build_relagn_disc_grid.py``.
    """
    from tengri._data_setup import find_data

    # Honors $TENGRI_DATA_DIR as well as the repo root (#1431).
    found = find_data("relagn_disc_grid.h5")
    if found is not None:
        return str(found)
    raise FileNotFoundError(
        "RELAGN disc grid not found. Run: "
        "conda run -n henv python scripts/build_relagn_disc_grid.py"
    )


@functools.cache
def _load_relagn_fn():
    """Load RELAGN outer-disc grid from HDF5 (cached)."""
    return create_relagn_disc_from_grid(_find_relagn_grid())


# ── AGN model registry ────────────────────────────────────────────


class AGNRegistryEntry:
    """Registry entry for an AGN model with optional metadata.

    Attributes
    ----------
    callable : Callable
        The AGN model function.
    citation : str
        Optional academic citation. Default empty string.
    status : str
        Model status: "production", "experimental", "demo", or "deprecated".
        Default "production".
    short_doc : str
        Optional one-line description. Default empty string.

    Notes
    -----
    **JIT-compatible**: no, class for registry initialization.
    """

    def __init__(
        self,
        callable: Callable,
        citation: str = "",
        status: str = "production",
        short_doc: str = "",
    ) -> None:
        self.callable = callable
        self.citation = citation
        self.status = status
        self.short_doc = short_doc

    def __call__(self, *args, **kwargs):
        """Forward calls to the wrapped callable."""
        return self.callable(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward attribute access to wrapped callable."""
        return getattr(self.callable, name)


AGN_MODELS: dict[str, Callable] = {}


def register_agn_model(
    name: str,
    *,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
) -> Callable:
    """Register an AGN model function (decorator factory).

    Parameters
    ----------
    name : str
        Unique model name for the registry.
    citation : str, optional
        Academic citation for the model. Default empty string.
    status : str, optional
        Model status ("production", "experimental", "demo", "deprecated").
        Default "production".
    short_doc : str, optional
        One-line description. Default empty string.

    Returns
    -------
    callable
        Decorator that registers the decorated function in AGN_MODELS.

    Notes
    -----
    **JIT-compatible**: no, registers at module load time (not JIT-compilable).

    The registered function must have signature::

        fn(wavelength, agn_log_lbol, **kwargs) -> L_nu [erg/s/Hz]

    Metadata is stored in an AGNRegistryEntry for introspection via the
    registry.list_agn_models() façade.
    """

    def decorator(fn: Callable) -> Callable:
        """Inner decorator that registers function in AGN_MODELS dict."""
        entry = AGNRegistryEntry(
            callable=fn,
            citation=citation,
            status=status,
            short_doc=short_doc,
        )
        AGN_MODELS[name] = entry
        return fn

    return decorator


def _resolve_monolithic_model(name: str) -> Callable | None:
    """Return the monolithic forward function for a self-contained model.

    Some deprecated model names are backed by self-contained forward functions
    that carry structural variant selectors (``torus_model``, ``disc_model``,
    raw radiative-transfer templates) which have no composable-block
    equivalent. For those the deprecated name resolves to the monolithic
    function directly rather than a composable preset, so every parameter still
    reaches the physics. Returns ``None`` for all other names.

    Parameters
    ----------
    name : str
        Deprecated AGN model name.

    Returns
    -------
    callable or None
        The monolithic forward function (with a deprecation warning already
        emitted), or ``None`` if ``name`` is not a self-contained model.
    """
    if name == "skirtor_stalevski":
        warnings.warn(
            "AGN model 'skirtor_stalevski' is deprecated. It returns the raw "
            "Stalevski (2016) SKIRTOR radiative-transfer total SED, which is not "
            "reproducible as a composable disc+torus recipe. For the CIGALE-style "
            "tunable-disc variant use agn_model='composable', "
            "agn_disc_block='skirtor', agn_torus_block='skirtor'.",
            DeprecationWarning,
            stacklevel=3,
        )
        return skirtor_stalevski_agn
    if name == "grahsp":
        # Lazy import avoids a grahsp → unified import cycle at module load.
        from tengri.components.agn.grahsp.registry import grahsp

        warnings.warn(
            "AGN model 'grahsp' is deprecated. It routes to the self-contained "
            "GRAHSP forward model (Kauffmann et al.), whose torus_model/disc_model "
            "variant selectors are not composable-block kwargs. For the block "
            "grammar use agn_model='composable', agn_disc_block='grahsp_sbpl', "
            "agn_torus_block='grahsp', etc.",
            DeprecationWarning,
            stacklevel=3,
        )
        return grahsp
    return None


def resolve_agn_model(name: str) -> Callable:
    """Retrieve a registered AGN model by name.

    All monolithic AGN models have been migrated to composable presets. This
    function accepts old model names (with deprecation warning) and routes them
    through the composable runner with the appropriate block selectors.

    Parameters
    ----------
    name : str
        Model name. Both old monolithic names (deprecated) and ``"composable"``
        are supported.

    Returns
    -------
    callable
        Either the composable runner (for new code) or a preset-routed wrapper
        for deprecated names.

    Notes
    -----
    **JIT-compatible**: no, performs dictionary lookup at initialization time.
    Old monolithic model names still work but emit DeprecationWarning.
    """
    if name == "composable":
        return AGN_MODELS["composable"]

    # Self-contained / un-composable models: the deprecated name returns the
    # monolithic forward function *directly*, not a composable preset, because
    # it carries structural variant selectors that do not map to the composable
    # disc/torus/lines block grammar:
    #   * skirtor_stalevski, the raw Stalevski (2016) SKIRTOR radiative-transfer
    #     *total* (disc + torus + scattering computed jointly), physically NOT a
    #     disc-block + torus-block sum (see test_skirtor_stalevski.py).
    #   * grahsp, a self-contained parity implementation whose ``torus_model``
    #     / ``disc_model`` variant selectors are GrahspConfig structural choices,
    #     not forwardable block kwargs (see grahsp/test_parity_integration.py).
    # Routing them here preserves full param forwarding; the composable blocks
    # (disc='skirtor'/'grahsp_sbpl', torus='skirtor'/'grahsp', …) remain
    # available for callers who opt into the block grammar explicitly.
    monolithic = _resolve_monolithic_model(name)
    if monolithic is not None:
        return monolithic

    if name not in _AGN_PRESETS:
        deprecated = [*_AGN_PRESETS.keys(), "skirtor_stalevski", "grahsp"]
        raise ValueError(
            f"Unknown AGN model '{name}'. Available: 'composable', "
            f"or any of the deprecated monolithic names: {deprecated}"
        )

    # Route deprecated monolithic names through composable with presets
    _preset_args = ", ".join(
        f"{k}={v}" for k, v in _AGN_PRESETS[name].items() if k != "_description"
    )
    warnings.warn(
        f"AGN model '{name}' is deprecated. It routes through composable blocks: "
        f"{_AGN_PRESETS[name]['_description']}. "
        f"Update your code to use: agn_model='composable', {_preset_args}",
        DeprecationWarning,
        stacklevel=2,
    )

    # Return a wrapper that applies the preset
    preset = _AGN_PRESETS[name]

    def preset_wrapper(wavelength, agn_log_lbol, **kwargs):
        """Apply preset block selectors and route through composable runner."""
        # Preset block-selectors + norm are AUTHORITATIVE and must win over
        # kwargs. The build path (agn/component.py) injects the component's
        # *default* block selectors ("none") into kwargs; if those clobbered
        # the preset, a monolithic model like ``richards2006`` collapsed to
        # ``disc=none`` → identically-zero SED (the #941 regression). The
        # preset carries only selector/norm keys, so physics params in kwargs
        # still flow through unchanged.
        merged = dict(kwargs)
        merged.update({k: v for k, v in preset.items() if k != "_description"})
        return AGN_MODELS["composable"](wavelength, agn_log_lbol, **merged)

    return preset_wrapper


# AGN_PRESETS: Maps deprecated monolithic model names to composable block recipes.
# Each preset specifies block selectors + normalization policy to reproduce the
# monolithic model behavior in the composable architecture.
_AGN_PRESETS = {
    # ``agn_norm='conserving'`` reproduces the monolithic disc+torus functions'
    # energy-conserving normalization bit-exactly (verified 3e-7 for
    # multicolor/silva04/cat3d_wind); ``'independent'`` was a #941 regression
    # that mis-scaled the disc/torus by an ``agn_lum_ratio`` factor (~2x, 99% off).
    "multicolor_agn": {
        "agn_disc_block": "multicolor",
        "agn_torus_block": "silva04",
        "agn_norm": "conserving",
        "_description": "disc=multicolor + torus=silva04",
    },
    "kubota_done": {  # Kubota & Done 2018 3-zone disc (full model)
        "agn_disc_block": "kubota_done",
        "agn_torus_block": "silva04",
        "agn_norm": "conserving",
        "_description": "disc=kubota_done + torus=silva04",
    },
    "kubota_done_full": {
        "agn_disc_block": "kubota_done",
        "agn_torus_block": "silva04",
        "agn_norm": "conserving",
        "_description": "disc=kubota_done + torus=silva04",
    },
    "silva04": {
        "agn_disc_block": "powerlaw",
        "agn_torus_block": "silva04",
        "agn_norm": "conserving",
        "_description": "disc=powerlaw + torus=silva04",
    },
    "cat3d_wind": {
        "agn_disc_block": "powerlaw",
        "agn_torus_block": "cat3d_wind",
        "agn_norm": "conserving",
        "_description": "disc=powerlaw + torus=cat3d_wind",
    },
    "adaf": {
        "agn_disc_block": "adaf",
        "agn_torus_block": "silva04",
        "agn_norm": "conserving",
        "_description": "disc=adaf + torus=silva04",
    },
    "relagn": {
        "agn_disc_block": "relagn",
        "agn_torus_block": "silva04",
        "agn_norm": "conserving",
        "_description": "disc=relagn + torus=silva04",
    },
    "skirtor": {
        # The monolithic ``skirtor_agn`` pairs a *power-law* disc with the
        # SKIRTOR clumpy torus (CIGALE ``skirtor2016``); ``disc='skirtor'``
        # (the raw Stalevski disc) belongs to ``skirtor_stalevski`` only.
        "agn_disc_block": "powerlaw",
        "agn_torus_block": "skirtor",
        "agn_norm": "cigale_joint",
        "_description": "disc=powerlaw + torus=skirtor",
    },
    # NOTE: ``skirtor_stalevski`` is intentionally absent; it is an
    # un-composable raw radiative-transfer template routed directly to the
    # monolithic ``skirtor_stalevski_agn`` in ``resolve_agn_model``.
    "qsogen": {
        "agn_disc_block": "qsogen",
        "agn_nlr_block": "none",
        "agn_blr_block": "qsogen",
        "agn_feii_block": "qsogen_balmer",
        "agn_torus_block": "qsogen",
        "agn_attenuation_block": "qsogen_smc",
        "agn_norm": "independent",
        "_description": (
            "disc=qsogen + blr=qsogen + feii=qsogen_balmer + torus=qsogen + atten=qsogen_smc"
        ),
    },
    # NOTE: ``grahsp`` is intentionally absent; it is a self-contained parity
    # model whose torus_model/disc_model variant selectors are not composable
    # kwargs, so it routes directly to the monolithic GRAHSP function in
    # ``resolve_agn_model`` (via ``_resolve_monolithic_model``). The block
    # decomposition (disc='grahsp_sbpl' + torus='grahsp' + …) is still available
    # explicitly through the composable grammar.
    "richards2006": {
        "agn_disc_block": "richards2006",
        "agn_norm": "independent",
        "_description": "disc=richards2006",
    },
    "unified_nlr_blr": {
        "agn_disc_block": "multicolor",
        "agn_nlr_block": "analytic",
        "agn_blr_block": "analytic",
        "agn_torus_block": "silva04",
        "agn_norm": "conserving",
        "_description": "disc=multicolor + nlr=analytic + blr=analytic + torus=silva04",
    },
}


# ── Generic unified AGN combiner ──────────────────────────────────


def unified_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    disc_model: str = "powerlaw",
    torus_model: str = "silva04",
    agn_torus_frac: float = 0.5,
    agn_ebv_disc: float = 0.0,
    **kwargs,
) -> jnp.ndarray:
    """Compute unified AGN SED: disc + torus.

    The torus re-emits a fraction ``agn_torus_frac`` of the bolometric
    luminosity, while the disc emits the remaining ``(1 - agn_torus_frac)``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    disc_model : str, optional
        Disc model name: "powerlaw", "multicolor", "kubota_done_3zone", "adaf".
        Default "powerlaw".
    torus_model : str, optional
        Torus model name: "silva04" (production, default), "skirtor".
        Default "silva04" (radiative-transfer based Silva+04 smooth torus).
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless], range [0, 1]. Disc gets
        (1 - agn_torus_frac). Default 0.5.
    **kwargs
        Additional parameters passed to disc and torus functions
        (they ignore unrecognized kwargs).

    Returns
    -------
    ndarray, shape (n_wave,)
        Total AGN L_nu [erg/s/Hz] = L_disc + L_torus.

    References
    ----------
    .. [1] See :mod:`tengri.components.agn.disc` and
           :mod:`tengri.components.agn.torus` for individual model references.

    Notes
    -----
    **JIT-compatible**: depends on disc and torus models (mostly yes for analytic).

    **Gradient-safe**: yes when using analytic disc and torus models.
    """
    from tengri.components.agn.skirtor import skirtor_sed

    disc_fns = {
        "powerlaw": powerlaw_disc,
        "multicolor": multicolor_disc,
        "kubota_done_3zone": kubota_done_disc,
        "adaf": adaf_spectrum,
    }
    torus_fns = {
        "silva04": silva04_sed,
        "skirtor": skirtor_sed,
    }

    disc_fn = disc_fns[disc_model]
    torus_fn = torus_fns[torus_model]

    # Disc gets (1 - covering_factor) of L_bol
    l_disc = disc_fn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=1.0 - agn_torus_frac,
        **kwargs,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)

    # Torus re-emits covering_factor of L_bol
    l_torus = torus_fn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
        **kwargs,
    )

    return l_disc + l_torus


# ── Deprecated monolithic AGN models ──────────────────────────────
# These functions are deprecated and no longer registered in AGN_MODELS.
# Use composable blocks instead (see resolve_agn_model for migration paths).
# Functions are retained for backward compatibility if imported directly.


def multicolor_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_log_nh_silva: float = 23.0,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    """Multicolor Shakura-Sunyaev disc + Silva+04 smooth AGN torus.

    Standard thin-disc SED with spin-dependent ISCO and Novikov-Thorne
    radiative efficiency. This is the outer standard disc only: no warm
    Comptonization or hot corona (for the full 3-zone model, see kubota_done_full).
    The torus uses the Silva+04 radiative-transfer solution for smooth dust geometry.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_lum_ratio : float, optional
        AGN luminosity fraction [dimensionless]. Default 0.1.
    agn_log_mbh : float, optional
        log10 of black hole mass [Msun]. Default 8.0.
    agn_log_ledd : float, optional
        log10 of Eddington ratio [dimensionless]. Default -1.0.
    agn_a_spin : float, optional
        Black hole spin [dimensionless], range [0, 0.998]. Default 0.0.
    agn_cos_inc : float, optional
        cos(inclination) [dimensionless], range [0, 1]. Default 0.5.
    agn_log_nh_silva : float, optional
        Torus hydrogen column density, log10(N_H / cm^-2). Default 23.0.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless]. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, uses :func:`multicolor_disc` and :func:`silva04_sed`.

    Also registered as "kubota_done" (deprecated alias).

    References
    ----------
    .. [1] Silva, L., Maiolino, R., & Granato, G. L. (2004). MNRAS, 355, 973.
       arXiv:astro-ph/0403425. AGN torus radiative transfer.
    .. [2] Kubota, A. & Done, C. (2018). MNRAS, 480, 1247. Disc model.
    """
    l_nu = unified_agn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        disc_model="multicolor",
        torus_model="silva04",
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
    )
    return l_nu * agn_lum_ratio


# kubota_done alias removed; use composable blocks instead:
# agn_model="composable", agn_disc_block="multicolor", agn_torus_block="silva04"


def kubota_done_full_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_f_hard: float = 0.02,
    agn_gamma_warm: float = 2.5,
    agn_kt_warm: float = 0.2,
    agn_gamma_hard: float = 1.8,
    agn_kt_hot: float = 100.0,
    agn_r_warm_ratio: float = 2.0,
    agn_log_nh_silva: float = 23.0,
    agn_torus_frac: float = 0.5,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Full Kubota & Done (2018) 3-zone disc + Silva+04 torus.

    Extends ``multicolor_agn`` with the full K&D 3-zone disc model:
    outer standard disc, warm Comptonization (soft X-ray excess), and
    hot corona (hard X-ray power law). Combined with a Silva+04 smooth
    dust torus using radiative-transfer geometry.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_lum_ratio : float, optional
        AGN luminosity fraction [dimensionless]. Default 0.1.
    agn_log_mbh : float, optional
        log10 of black hole mass [Msun]. Default 8.0.
    agn_log_ledd : float, optional
        log10 of Eddington ratio [dimensionless]. Default -1.0.
    agn_a_spin : float, optional
        Black hole spin [dimensionless], range [0, 0.998]. Default 0.0.
    agn_cos_inc : float, optional
        cos(inclination) [dimensionless]. Default 0.5.
    agn_f_hard : float, optional
        Fraction of L_Edd in corona [dimensionless]. Default 0.02.
    agn_gamma_warm : float, optional
        Warm Comptonization photon index [dimensionless]. Default 2.5.
    agn_kt_warm : float, optional
        Warm electron temperature [keV]. Default 0.2.
    agn_gamma_hard : float, optional
        Hard X-ray photon index [dimensionless]. Default 1.8.
    agn_kt_hot : float, optional
        Hot corona temperature [keV]. Default 100.0.
    agn_r_warm_ratio : float, optional
        R_warm / R_hot ratio [dimensionless]. Default 2.0.
    agn_log_nh_silva : float, optional
        Torus hydrogen column density, log10(N_H / cm^-2). Default 23.0.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless]. Default 0.5.
    agn_ebv_disc : float, optional
        SMC-law color excess on disc [mag]. Default 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, uses :func:`kubota_done_disc` and :func:`silva04_sed`.

    References
    ----------
    .. [1] Silva, L., Maiolino, R., & Granato, G. L. (2004). MNRAS, 355, 973.
       arXiv:astro-ph/0403425. AGN torus radiative transfer.
    .. [2] Kubota, A. & Done, C. (2018). MNRAS, 480, 1247. Disc model.
    """
    # 3-zone disc gets (1 - covering_factor) of L_bol
    l_disc = kubota_done_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=1.0 - agn_torus_frac,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
        agn_f_hard=agn_f_hard,
        agn_gamma_warm=agn_gamma_warm,
        agn_kt_warm=agn_kt_warm,
        agn_gamma_hard=agn_gamma_hard,
        agn_kt_hot=agn_kt_hot,
        agn_r_warm_ratio=agn_r_warm_ratio,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)

    # Torus re-emits covering_factor of L_bol
    l_torus = silva04_sed(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
    )

    return (l_disc + l_torus) * agn_lum_ratio


def skirtor_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_tau_skirtor: float = 7.0,
    agn_p_skirtor: float = 1.0,
    agn_q_skirtor: float = 1.0,
    agn_oa_skirtor: float = 40.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_torus_frac: float = 0.5,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """SKIRTOR clumpy torus AGN: power-law disc + SKIRTOR torus (analytic).

    Uses the analytic SKIRTOR approximation (Stalevski et al. 2012, 2016)
    for the torus emission, combined with a power-law accretion disc.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float, optional
        log10 of bolometric luminosity [Lsun]. Default 11.0.
    agn_lum_ratio : float, optional
        AGN luminosity fraction [dimensionless]. Default 0.1.
    agn_tau_skirtor : float, optional
        Optical depth at 9.7 um [dimensionless], range [3, 11]. Default 7.0.
    agn_p_skirtor : float, optional
        Radial density gradient [dimensionless], range [0, 1.5]. Default 1.0.
    agn_q_skirtor : float, optional
        Polar density gradient [dimensionless], range [0, 1.5]. Default 1.0.
    agn_oa_skirtor : float, optional
        Opening angle [degrees], range [20, 60]. Default 40.0.
    agn_cos_inc : float, optional
        cos(inclination) [dimensionless], 0=edge-on, 1=face-on. Default 0.5.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless]. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: no, requires SKIRTOR template interpolation (non-differentiable).
    """
    # Disc gets (1 - covering_factor) of L_bol
    l_disc = powerlaw_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=1.0 - agn_torus_frac,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)

    # SKIRTOR torus re-emits covering_factor of L_bol
    # Auto-load tabulated templates on first call
    l_torus = _load_skirtor_fn()(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_tau_skirtor=agn_tau_skirtor,
        agn_p_skirtor=agn_p_skirtor,
        agn_q_skirtor=agn_q_skirtor,
        agn_oa_skirtor=agn_oa_skirtor,
        agn_cos_inc=agn_cos_inc,
        agn_torus_frac=agn_torus_frac,
    )

    return (l_disc + l_torus) * agn_lum_ratio


def skirtor_stalevski_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_tau_skirtor: float = 7.0,
    agn_p_skirtor: float = 1.0,
    agn_q_skirtor: float = 1.0,
    agn_oa_skirtor: float = 40.0,
    agn_radius_ratio: float = 20.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    **_kwargs,
) -> jnp.ndarray:
    r"""Raw Stalevski (2016) SKIRTOR SED, the published radiative-transfer output.

    Returns the SKIRTOR template's **total** SED (accretion disc + clumpy torus
    + scattered light) exactly as the Stalevski et al. (2016) radiative-transfer
    grid computed it at the requested viewing angle, scaled to the requested
    bolometric luminosity. Unlike the ``skirtor`` model (power-law disc) and the
    composable ``disc.skirtor`` block (CIGALE's analytic disc + ``norm=1/∫dust``
    energy balance), this applies **no analytic-disc substitution and no
    re-normalization**: it is the faithful SKIRTOR template, matching codes that
    read SKIRTOR directly (e.g. ProSpect's ``SKIRTOR_interp``) rather than CIGALE's
    reconstruction.

    Use this model to reproduce the raw SKIRTOR SED; use the composable
    ``disc.skirtor`` + ``torus.skirtor`` blocks (``norm='cigale_joint'``) to
    reproduce CIGALE's ``skirtor2016`` instead. The two answer different
    questions, the raw template vs CIGALE's tunable-disc variant.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float, optional
        log10 of bolometric luminosity [Lsun]. Default 11.0.
    agn_lum_ratio : float, optional
        AGN luminosity fraction (host scaling) [dimensionless]. Default 0.1.
    agn_tau_skirtor : float, optional
        Optical depth at 9.7 um [dimensionless], grid range [3, 11]. Default 7.0.
    agn_p_skirtor, agn_q_skirtor : float, optional
        Radial / polar dust-density gradients [dimensionless]. Default 1.0.
    agn_oa_skirtor : float, optional
        Half-opening angle of the dust-free cone [degrees]. Default 40.0.
    agn_cos_inc : float, optional
        cos(inclination); 1 = face-on (Type 1), 0 = edge-on. Default cos(30 deg).

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz], total SKIRTOR SED scaled to
        ``10**agn_log_lbol * L_sun * agn_lum_ratio``.

    Notes
    -----
    **JIT-compatible**: no, requires SKIRTOR grid interpolation.

    **Grid caveat**: tengri ships the SKIRTOR v3 grid (tau in {3,5,7,9,11}); a
    requested ``agn_tau_skirtor`` outside that range is clamped. Bit-for-bit
    agreement with another code additionally requires the same SKIRTOR grid
    version.

    References
    ----------
    .. [S16] Stalevski, M. et al. 2016, MNRAS, 458, 2288. arXiv:1602.06954.
       https://doi.org/10.1093/mnras/stw444
    """
    fn, has_radius_ratio = _load_skirtor_raw_total_fn()
    kw = dict(
        agn_log_lbol=agn_log_lbol,
        agn_tau_skirtor=agn_tau_skirtor,
        agn_p_skirtor=agn_p_skirtor,
        agn_q_skirtor=agn_q_skirtor,
        agn_oa_skirtor=agn_oa_skirtor,
        agn_cos_inc=agn_cos_inc,
        frac_agn=1.0,
    )
    if has_radius_ratio:
        kw["agn_radius_ratio"] = agn_radius_ratio
    return fn(wavelength, **kw) * agn_lum_ratio


def silva04_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_log_nh_silva: float = 23.0,
    agn_torus_frac: float = 0.5,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Silva+04 smooth-torus AGN: power-law disc + Silva+04 torus.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 11.0.
    agn_lum_ratio : float, optional
        Overall AGN luminosity fraction applied on top of the
        disc-plus-torus sum. Default 0.1.
    agn_log_nh_silva : float, optional
        Hydrogen column density, ``log10(N_H / cm^-2)``. Default 23.0.
    agn_torus_frac : float, optional
        Fraction of L_bol reprocessed by the torus. Disc receives
        ``1 - agn_torus_frac``. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        Combined AGN SED. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes, both the power-law disc and the Silva+04
    grid interpolation are pure JAX.

    Grid templates published with AGNfitter (Calistro Rivera et al. 2016);
    see :mod:`tengri.components.agn.silva04` and
    ``scripts/build_silva04_grid.py``.
    """
    l_disc = powerlaw_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=1.0 - agn_torus_frac,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)
    l_torus = silva04_sed(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
    )
    return (l_disc + l_torus) * agn_lum_ratio


def cat3d_wind_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_a_cat3d: float = -2.0,
    agn_fwd_cat3d: float = 1.0,
    agn_torus_frac: float = 0.5,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """CAT3D-Wind AGN: power-law disc + clumpy-disc + polar-wind torus.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 11.0.
    agn_lum_ratio : float, optional
        Overall AGN luminosity fraction applied on top of the
        disc-plus-torus sum. Default 0.1.
    agn_cos_inc : float, optional
        Cosine of inclination. Default 0.5.
    agn_a_cat3d : float, optional
        Radial power-law index of the clumpy-cloud distribution. Default
        −2.0.
    agn_fwd_cat3d : float, optional
        Polar-wind mass fraction. Default 1.0.
    agn_torus_frac : float, optional
        Fraction of L_bol reprocessed by the torus. Disc receives
        ``1 - agn_torus_frac``. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        Combined AGN SED. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes.

    Grid templates published with AGNfitter-rX (Martínez-Ramírez
    et al. 2024, A&A 688, A46, arXiv:2405.12111), a Hönig & Kishimoto 2017
    CAT3D-Wind three-parameter projection. See
    :mod:`tengri.components.agn.cat3d_wind` and
    ``scripts/build_cat3d_wind_grid.py``.
    """
    l_disc = powerlaw_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=1.0 - agn_torus_frac,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)
    l_torus = cat3d_wind_sed(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_cos_inc=agn_cos_inc,
        agn_a_cat3d=agn_a_cat3d,
        agn_fwd_cat3d=agn_fwd_cat3d,
        agn_torus_frac=agn_torus_frac,
    )
    return (l_disc + l_torus) * agn_lum_ratio


def adaf_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_adaf_alpha: float = 0.3,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.1,
    agn_torus_frac: float = 0.5,
    agn_log_nh_silva: float = 23.0,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Faithful ADAF (Mahadevan 1997) + Silva+04 torus for low-luminosity AGN.

    The inner flow is an advection-dominated accretion flow; a Silva+04 smooth
    torus re-emits a fraction of the bolometric luminosity in the IR. As of #898
    the disc is the *faithful* Mahadevan 1997 model
    (:func:`~tengri.components.agn.adaf.adaf_spectrum`): ``agn_log_lbol`` is the
    canonical luminosity and the accretion rate is derived from it (Eq. 49), so
    ``agn_log_ledd`` is retired; the ad-hoc truncated outer disc is dropped.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_lum_ratio : float, optional
        AGN luminosity fraction [dimensionless]. Default 0.1.
    agn_log_mbh : float, optional
        log10 of black hole mass [Msun]. Default 8.0.
    agn_adaf_alpha : float, optional
        ADAF viscosity parameter alpha. Default 0.3.
    agn_adaf_beta : float, optional
        Gas-to-total pressure ratio (magnetic fraction is 1-beta). Default 0.5.
    agn_adaf_delta : float, optional
        Electron viscous-heating fraction (default 0.1 = modern preference;
        Mahadevan 1997 fiducial is 1/2000). Default 0.1.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless]. Default 0.5.
    agn_log_nh_silva : float, optional
        Torus hydrogen column density, log10(N_H / cm^-2). Default 23.0.
    agn_ebv_disc : float, optional
        SMC-law color excess on disc [mag]. Default 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, uses :func:`~tengri.components.agn.adaf.adaf_spectrum`
    and :func:`silva04_sed`.

    References
    ----------
    .. [1] Mahadevan, R. 1997, ApJ, 477, 585. arXiv:astro-ph/9609107.
    .. [2] Silva, L., Maiolino, R., & Granato, G. L. (2004). MNRAS, 355, 973.
       arXiv:astro-ph/0403425. AGN torus radiative transfer.
    """
    # ADAF disc gets (1 - torus_frac) of L_bol (energy-conserving debit).
    l_disc = adaf_spectrum(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=1.0 - agn_torus_frac,
        agn_log_mbh=agn_log_mbh,
        agn_adaf_alpha=agn_adaf_alpha,
        agn_adaf_beta=agn_adaf_beta,
        agn_adaf_delta=agn_adaf_delta,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)

    # Silva+04 torus re-emits torus_frac of L_bol
    l_torus = silva04_sed(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
    )

    return (l_disc + l_torus) * agn_lum_ratio


def relagn_agn(
    wavelength: jnp.ndarray,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_mdot: float = -1.0,
    agn_astar: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_torus_frac: float = 0.5,
    agn_log_nh_silva: float = 23.0,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """RELAGN relativistic outer disc + Silva+04 dust torus.

    Uses the precomputed RELAGN grid (Hagen & Done 2023) with KYCONV
    (Dovciak, Karas & Yaqoob 2004) per-annulus Kerr ray-tracing for the
    disc, and a Silva+04 smooth dust torus for radiative-transfer reprocessing.

    The disc luminosity is self-consistent with BH mass and accretion rate;
    the torus re-emits ``agn_torus_frac`` of the disc bolometric output.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_mbh : float, optional
        ``log10(M_BH / M_sun)``, range [7, 10]. Default 8.0.
    agn_log_mdot : float, optional
        ``log10(Ṁ / Ṁ_Edd)``, range [−1.5, 0.3]. Default −1.0.
    agn_astar : float, optional
        Dimensionless BH spin, prograde only, range [0, 0.998]. Default 0.0.
    agn_cos_inc : float, optional
        Cosine of inclination (1 = face-on). Default 0.5.
    agn_torus_frac : float, optional
        Torus covering factor; torus re-emits this fraction of disc L_bol.
        Disc is attenuated by ``(1 − agn_torus_frac)``. Default 0.5.
    agn_log_nh_silva : float, optional
        Torus hydrogen column density, log10(N_H / cm^-2). Default 23.0.
    agn_ebv_disc : float, optional
        SMC-law color excess applied to disc [mag]. Default 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Total AGN L_ν. [erg s⁻¹ Hz⁻¹]

    Notes
    -----
    **JIT-compatible**: yes, disc interpolation is pure JAX triweight kernel.

    **Gradient-safe**: yes, C²-continuous triweight kernel on all grid axes.

    **Grid required**: ``data/relagn_disc_grid.h5`` built by
    ``scripts/build_relagn_disc_grid.py`` (requires HEASOFT/XSPEC + KYCONV).

    **Torus normalization**: derived by integrating the disc L_ν over the
    output wavelength grid via ``jnp.trapezoid``: no separate ``agn_log_lbol``
    parameter needed.

    References
    ----------
    .. [1] Dovciak, M., Karas, V., & Yaqoob, T. (2004).
       ApJS, 153, 205. doi:10.1086/421115  [KYCONV]

    .. [2] Hagen, S. & Done, C. (2023).
       MNRAS, 521, 251. doi:10.1093/mnras/stad478  [RELAGN disc]

    .. [3] Silva, L., Maiolino, R., & Granato, G. L. (2004). MNRAS, 355, 973.
       arXiv:astro-ph/0403425. AGN torus radiative transfer.
    """
    disc_fn = _load_relagn_fn()

    # Full disc at reference inclination from precomputed KYCONV grid
    l_disc_full = disc_fn(
        wavelength,
        agn_log_mbh=agn_log_mbh,
        agn_log_mdot=agn_log_mdot,
        agn_astar=agn_astar,
        agn_cos_inc=agn_cos_inc,
    )
    l_disc_full = _redden_disc(wavelength, l_disc_full, agn_ebv_disc)

    # Attenuate disc by torus covering factor
    l_disc = l_disc_full * (1.0 - agn_torus_frac)

    # Derive disc L_bol by integrating L_ν over ν (trapezoid in JAX)
    nu = _C_AA / wavelength  # decreasing
    # Sort ascending for trapezoid
    lbol_disc_erg = jnp.trapezoid(jnp.flip(l_disc_full), jnp.flip(nu))
    log_lbol_lsun = jnp.log10(jnp.maximum(lbol_disc_erg, 1e30)) - jnp.log10(_LSUN_ERG)

    # Torus re-emits agn_torus_frac of disc L_bol
    l_torus = silva04_sed(
        wavelength,
        agn_log_lbol=log_lbol_lsun,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
    )

    return l_disc + l_torus


# ── Geometric masking (smooth sigmoid for differentiability) ──────


def _sigmoid_mask(
    cos_inc: float,
    theta_torus: float,
    width: float = 2.0,
) -> float:
    """Smooth geometric mask for disc/BLR visibility.

    Returns ~1 (visible) for face-on orientations and ~0 (obscured)
    when the line of sight passes through the torus.

    The critical angle is ``inc_crit = 90° - theta_torus``: above this
    the torus intercepts the sightline. In Synthesizer [1]_ this condition
    is implemented as a hard binary step (``inclination + theta_torus > 90°``
    → zeroed). tengri replaces the hard step with a smooth sigmoid to
    preserve differentiability for gradient-based inference.

    .. math::

        \\sigma(i, \\theta_t) = \\mathrm{sigmoid}\\!\\left(
            -\\frac{\\arccos(\\cos i) - (90^\\circ - \\theta_t)}{w}
        \\right)

    where :math:`w` is the transition half-width in degrees (default 2°).
    At :math:`w \\to 0` this converges to Synthesizer's hard binary mask.

    Parameters
    ----------
    cos_inc : float
        Cosine of inclination (0 = edge-on, 1 = face-on).
    theta_torus : float
        Torus half-opening angle [degrees].
    width : float
        Sigmoid transition width [degrees]. Default 2.0.

    Returns
    -------
    float
        Visibility fraction in [0, 1].

    Notes
    -----
    JIT/grad/vmap compatible.

    References
    ----------
    .. [1] Synthesizer ``torus_edgeon_condition``:
           https://github.com/synthesizer-project/synthesizer/blob/main/src/synthesizer/emission_models/agn/unified_agn.py
    """
    # Canonical implementation lives in the shared masking module so the
    # composable runner and this monolithic model apply the identical mask.
    from tengri.components.agn.blocks.masking import sigmoid_visibility_mask

    return sigmoid_visibility_mask(cos_inc, theta_torus, width)


# ── Unified AGN with NLR + BLR decomposition ──────────────────────


def unified_nlr_blr(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_theta_torus: float = 30.0,
    agn_nlr_cf: float = 0.1,
    agn_blr_cf: float = 0.1,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_log_nh_silva: float = 23.0,
    agn_torus_frac: float = 0.5,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_blr_fwhm: float = 5000.0,
    agn_nlr_fwhm: float = 500.0,
    # Differs from the declared agn_polar_ebv default (0.03) on purpose: polar
    # reddening here is opt-in, so the default is the no-op.
    agn_polar_ebv: float = 0.0,
    nlr_fn: "Callable | None" = None,
    blr_fn: "Callable | None" = None,
    include_xray: bool = False,
    xray_gamma_agn: float = 1.8,
    xray_alpha_ox: float = 0.0,
    xray_E_cut: float = 300.0,
    include_radio: bool = False,
    radio_q_ir: float = 2.64,
    radio_alpha_sf: float = 0.8,
    radio_loudness: float = 0.0,
    radio_alpha_agn: float = 0.7,
    **_kwargs,
) -> jnp.ndarray:
    """Unified AGN SED with NLR/BLR decomposition and geometric masking.

    Implements the same conceptual decomposition as the ``UnifiedAGN`` class
    in Synthesizer (Lovell et al. 2025 [1]_, Roper et al. 2026 [2]_):
    an accretion disc, dusty torus, narrow line region (NLR), and broad line
    region (BLR), combined with inclination-dependent geometric masking.
    Polar dust reddening (SMC law) additionally follows CIGALE's
    ``skirtor2016`` module [3]_.

    .. rubric:: Correspondence with Synthesizer's UnifiedAGN

    The following table maps tengri parameters to Synthesizer attributes
    (``synthesizer.components.blackhole.BlackholeComponent``):

    ==================  =====================================  ============================
    tengri              Synthesizer                            Note
    ==================  =====================================  ============================
    agn_cos_inc         ``cos(inclination)``                   Synth uses inclination [deg]
    agn_theta_torus     ``theta_torus`` [deg]                  Same meaning
    agn_nlr_cf          ``covering_fraction_nlr``              Identical semantics
    agn_blr_cf          ``covering_fraction_blr``              Identical semantics
    agn_torus_frac      ``torus_fraction = theta_torus/90°``   **Different**: independent param
    agn_blr_fwhm        ``velocity_dispersion_blr`` [km/s]     tengri uses FWHM directly
    agn_nlr_fwhm        ``velocity_dispersion_nlr`` [km/s]     Same
    ==================  =====================================  ============================

    .. rubric:: Deliberate differences from Synthesizer

    1. **Analytic disc, not a grid**: Synthesizer extracts disc emission from
       precomputed CLOUDY photoionization grids. tengri uses the closed-form
       ``multicolor_disc`` (Shakura-Sunyaev / Novikov-Thorne) from
       ``disc.py``. Rationale: grid look-ups are not JAX-jittable under
       gradient tape; analytic models are differentiable by construction.

    2. **Analytic NLR/BLR templates, not grids**: Synthesizer uses CLOUDY
       grids over (logU, log n_H) for both NLR and BLR emission. tengri uses
       empirically calibrated Gaussian line templates (Vanden Berk et al.
       2001 for BLR; Groves et al. 2004 for NLR). Rationale: same as above.

    3. **Smooth sigmoid mask, not a hard binary**: Synthesizer zeros the disc
       and BLR when ``inclination + theta_torus > 90°``, a hard step
       function. tengri replaces this with a smooth sigmoid (see
       ``_sigmoid_mask``) centered at the same critical angle with a ~2°
       transition width. Rationale: the hard mask has zero gradient with
       respect to inclination almost everywhere, making gradient-based VI
       and HMC blind to inclination information near the critical angle.

    4. **``agn_torus_frac`` decoupled from ``theta_torus``**: Synthesizer
       internally sets ``torus_fraction = theta_torus / 90°``, coupling the
       reprocessed fraction to the geometric opening angle. tengri keeps
       ``agn_torus_frac`` as an independent free parameter. Rationale:
       auto-deriving from ``cos(theta_torus)`` in the forward pass introduces
       a gradient discontinuity at the poles (CLAUDE.md gotcha). The coupling
       can be enforced at the ``Parameters`` level via a fixed/derived param.

    5. **Polar dust reddening (SMC law)**: ``agn_polar_ebv`` applies SMC
       extinction to the disc and BLR for Type 1 sightlines (mask > 0).
       Absent from Synthesizer's basic ``UnifiedAGNIntrinsic``; follows
       CIGALE ``skirtor2016`` (Yang et al. 2020 [3]_).

    .. rubric:: Geometry summary

    .. math::

        L_{\\rm AGN}(\\lambda) =
            \\sigma(i, \\theta_t)\\, A_{\\rm pol,eff}(\\lambda)\\, L_{\\rm disc}(\\lambda)
            + L_{\\rm torus}(\\lambda)
            + f_{\\rm NLR}\\, \\eta_{\\rm NLR}(\\lambda)\\, L_{\\rm bol,disc}
            + \\sigma(i, \\theta_t)\\, A_{\\rm pol,eff}(\\lambda)\\,
              f_{\\rm BLR}\\, \\eta_{\\rm BLR}(\\lambda)\\, L_{\\rm bol,disc}
            + \\mathbb{1}_{\\rm X-ray} \\, L_{\\rm X-ray}(\\lambda)
            + \\mathbb{1}_{\\rm radio} \\, L_{\\rm radio}(\\lambda)

    where :math:`\\sigma(i, \\theta_t)` is the smooth sigmoid mask
    (1 = visible, 0 = obscured), :math:`A_{\\rm pol,eff}(\\lambda) = 1 + \\sigma(i, \\theta_t) \\,
    (A_{\\rm pol}(\\lambda) - 1)` is the visibility-weighted SMC polar dust transmission
    (SMC law; only reddens when disc is visible to the observer), :math:`f_{\\rm NLR/BLR}`
    are the covering fractions, :math:`\\eta_{\\rm NLR/BLR}` are the normalized
    line templates, and :math:`\\mathbb{1}_{\\rm X-ray/radio}` are indicator functions
    controlling whether X-ray and radio components are included.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        :math:`\\log_{10}(L_{\\rm bol} / L_\\odot)`: total AGN bolometric
        luminosity expressed in **solar luminosities** (not erg/s).
        Convert from synthesizer's ``bolometric_luminosity`` [erg/s] via
        ``agn_log_lbol = log10(L_bol_erg) - log10(L_SUN_erg)``,
        i.e. subtract :math:`\\approx 33.58`. Typical bright Seyfert: 10.5;
        bright quasar: 12.5. The default ``44.0`` is a legacy test fixture
        that is **not a physical AGN luminosity**: set this parameter
        explicitly. See module-level "Convention" note. Default 11.0.
    agn_cos_inc : float
        Cosine of inclination angle (0 = edge-on/Type 2,
        1 = face-on/Type 1). Synthesizer uses inclination in degrees;
        tengri uses cos(inclination). Default 0.5.
    agn_theta_torus : float
        Torus half-opening angle [degrees]. Controls the critical inclination
        above which the disc and BLR are obscured. Same meaning as
        ``theta_torus`` in Synthesizer. Default 30.0.
    agn_nlr_cf : float
        NLR covering fraction (0 to 1). Fraction of disc luminosity
        reprocessed into NLR emission. Synthesizer: ``covering_fraction_nlr``.
        Default 0.1.
    agn_blr_cf : float
        BLR covering fraction (0 to 1). Fraction of disc luminosity
        reprocessed into BLR emission. Synthesizer: ``covering_fraction_blr``.
        Default 0.1.
    agn_log_mbh : float
        log10(M_BH / Msun). Default 7.0.
    agn_log_ledd : float
        log10(L/L_Edd). Default -1.0.
    agn_a_spin : float
        BH spin (0 to 0.998). Default 0.0.
    agn_T_hot : float
        Hot dust temperature [K]. Default 1200.
    agn_T_warm : float
        Warm dust temperature [K]. Default 300.
    agn_frac_hot : float
        Hot-to-warm dust fraction. Default 0.3.
    agn_tau_torus : float
        Torus optical depth at 9.7 um. Default 5.
    agn_torus_frac : float
        Torus covering factor (fraction of L_bol intercepted by torus).
        In Synthesizer this is derived as ``theta_torus / 90°``; here it is
        an independent free parameter. Default 0.5.
    agn_lum_ratio : float
        Overall AGN fraction scaling. Default 0.1.
    agn_blr_fwhm : float
        BLR line FWHM [km/s]. Synthesizer uses ``velocity_dispersion_blr``;
        tengri uses FWHM directly. Default 5000.
    agn_nlr_fwhm : float
        NLR line FWHM [km/s]. Default 500.
    agn_polar_ebv : float
        Polar dust E(B-V) applied to disc + BLR for Type 1 views (SMC law).
        Follows CIGALE skirtor2016 [3]_. Not in Synthesizer's basic
        UnifiedAGN. Default 0.0 (no reddening).
    nlr_fn : callable or None
        NLR emission backend. Signature::

            nlr_fn(wavelength, l_disc_bol_erg, covering_fraction,
                   fwhm_kms=500.0, **kwargs) -> ndarray, shape (n_wave,)

        where the return value is L_nu [erg/s/Hz]. Default ``None`` uses the
        built-in analytic template from :func:`~tengri.components.agn.nlr.compute_nlr_sed`.
        To use the Cue neural-net emulator, pass the result of
        :func:`make_cue_nlr_fn`.
    blr_fn : callable or None
        BLR emission backend. Same signature as ``nlr_fn``. Default ``None``
        uses the built-in analytic template from
        :func:`~tengri.components.agn.blr.compute_blr_sed`.
    include_xray : bool
        If True, include X-ray corona emission. Default False (UV-optical-IR only).
    xray_gamma_agn : float
        X-ray photon index (power-law slope). Default 1.8. Valid range: 1.4–2.4.
    xray_alpha_ox : float
        Offset [dex] to the empirical alpha_OX (Just+2007). Default 0.0 (pure
        empirical). Negative values harden the corona, positive soften it.
        Valid range: -2.0 to -1.0 for typical AGN.
    xray_E_cut : float
        X-ray exponential cutoff energy [keV]. Default 300.0.
    include_radio : bool
        If True, include radio synchrotron + AGN radio jet emission. Default False.
    radio_q_ir : float
        FIR–radio correlation parameter (Bell+2003 mode). Default 2.64.
    radio_alpha_sf : float
        Star-forming synchrotron spectral index. Default 0.8.
    radio_loudness : float
        AGN radio-loudness log10(L_5GHz/L_B). Default 0.0 (no radio AGN).
    radio_alpha_agn : float
        AGN radio spectral index. Default 0.7.

    Returns
    -------
    array, shape (n_wave,)
        Total AGN L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    JIT/grad/vmap compatible when ``nlr_fn`` and ``blr_fn`` are JIT-compatible
    (the default analytic templates are; Cue-backed closures are also JIT-safe
    since the weights are closed over as static pytree leaves).

    **X-ray and radio components** (when included):

    - X-ray: Power-law corona emission normalized via the alpha_OX relation
      (Tananbaum+1979). Wavelength range: λ < 124 Å (E > ~100 eV). Computed
      via :func:`~tengri.components.xray.xray_agn_corona` [4]_.
    - Radio: Synchrotron + optional free-free + AGN jet emission. Wavelength
      range: λ > 1 mm (ν < 300 GHz). Computed via
      :func:`~tengri.components.radio.radio_total` [5]_.

    When both ``include_xray=False`` and ``include_radio=False``, the function
    returns the UV-optical-FIR unified AGN SED (original behavior). The new
    components are additive and do not affect existing parameters or output
    ranges.

    References
    ----------
    .. [1] Lovell C. C. et al. 2025, Open Journal of Astrophysics, 8,
           "Synthesizer: a Software Package for Synthetic Astronomical Observables",
           https://doi.org/10.33232/001c.145766
    .. [2] Roper W. J. et al. 2026, Journal of Open Source Software, 11, 9436,
           "Synthesizer: Synthetic Observables for Modern Astronomy",
           https://doi.org/10.21105/joss.09436
           (Both Synthesizer papers [1]_ [2]_ must be cited together.)
    .. [3] G. Yang et al. 2020, MNRAS, 491, 740, "X-CIGALE: Fitting AGN/galaxy
           SEDs from X-ray to infrared" (skirtor2016 module),
           https://doi.org/10.1093/mnras/stz3001
    .. [4] C. Ricci et al. 2017, ApJS, 233, 17, "Swift/BAT and XMM-Newton
           observations of the hard X-ray selected active galactic nuclei
           sample" (X-ray AGN sample), https://doi.org/10.3847/1538-4365/aa96ad
    .. [5] Murphy E.~J. et al. 2011, ApJ, 737, 67, "The Radio Flux and
           Infrared Properties of the GOODS-North Radio Sources",
           https://doi.org/10.1088/0004-637X/737/2/67

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.agn.unified import unified_nlr_blr
    >>> wave = jnp.linspace(1000.0, 30000.0, 512)
    >>> sed = unified_nlr_blr(wave, agn_log_lbol=11.42, agn_cos_inc=0.8)
    >>> sed.shape
    (512,)
    >>> bool(jnp.all(sed >= 0))
    True
    """
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG

    # Derive torus covering factor from opening angle for consistency:
    # covering_factor ~ cos(theta_torus). If agn_torus_frac is at default
    # (0.5), use the geometric value; otherwise honor the explicit setting.
    geom_cf = jnp.cos(jnp.radians(jnp.clip(agn_theta_torus, 0.0, 90.0)))
    # When agn_torus_frac is a free parameter, use it directly.
    # The jnp.where(|x-0.5| < 1e-6, geom_cf, x) creates a likelihood discontinuity
    # at torus_frac=0.5±1e-6 that corrupts gradient-based inference (VI, MAP).
    # Auto-derivation from theta_torus is done at the Parameters level via fixed values,
    # not in the forward pass.
    _ = geom_cf  # retained for reference; used when agn_torus_frac is fixed by Parameters

    # --- Geometric masks ---
    mask_disc = _sigmoid_mask(agn_cos_inc, agn_theta_torus)
    mask_blr = _sigmoid_mask(agn_cos_inc, agn_theta_torus)

    # --- Disc emission (intrinsic, before masking) ---
    # Disc gets (1 - torus_frac) of L_bol
    l_disc = multicolor_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=1.0 - agn_torus_frac,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
    )
    # --- Polar dust extinction (Type 1 reddening, SMC law) ---
    # Applied to disc and BLR when visible (mask > 0).
    # Uses SMC law since AGN sightlines typically lack a 2175 A bump.
    # Inclination-conditional: polar dust effect scales with visibility mask,
    # so edge-on (Type 2) views have no polar-dust reddening since the disc/BLR
    # are already torus-obscured.
    from tengri.components.dust.attenuation import smc as _smc_law

    k_polar = _smc_law(wavelength)  # k(λ)/k(V), normalized at 5500 A
    # R_V(SMC) = 2.93; A(λ) = E(B-V) * R_V * k(λ)
    _RV_SMC = 2.93
    polar_trans_base = jnp.exp(-0.921 * agn_polar_ebv * _RV_SMC * k_polar)
    # Effective polar transmission: visibility-weighted.
    # When visibility ≈ 0 (edge-on/Type-2), polar_trans_eff → 1 (no effect).
    # When visibility ≈ 1 (face-on/Type-1), polar_trans_eff → polar_trans_base.
    # Smooth interpolation: 1 + visibility * (polar_trans - 1)
    polar_trans = 1.0 + mask_disc * (polar_trans_base - 1.0)

    # Apply geometric masking + visibility-weighted polar dust to disc
    l_disc_masked = mask_disc * l_disc * polar_trans

    # --- Torus emission (always visible) ---
    l_torus = silva04_sed(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
    )

    # --- NLR emission (isotropic, always visible) ---
    # NLR is illuminated by the disc bolometric luminosity.
    # Pluggable: use nlr_fn if provided, else the built-in analytic template.
    l_disc_bol_erg = (1.0 - agn_torus_frac) * l_bol_erg
    _nlr_fn = nlr_fn if nlr_fn is not None else compute_nlr_sed
    l_nlr = _nlr_fn(
        wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm,
        **_kwargs,
    )

    # --- BLR emission (masked by torus + visibility-weighted polar dust) ---
    # Pluggable: use blr_fn if provided, else the built-in analytic template.
    _blr_fn = blr_fn if blr_fn is not None else compute_blr_sed
    l_blr_raw = _blr_fn(
        wavelength,
        l_disc_bol_erg=l_bol_erg,  # Full intrinsic L_bol; BLR is interior to torus
        covering_fraction=agn_blr_cf,
        fwhm_kms=agn_blr_fwhm,
        **_kwargs,
    )
    l_blr = mask_blr * l_blr_raw * polar_trans

    # --- X-ray corona (optional) ---
    # ``unified_nlr_blr`` exposes ``xray_alpha_ox`` as a delta offset
    # (default 0.0 = pure empirical), consistent with the composable
    # component path. The legacy ``_xray_agn_corona_legacy`` expects an
    # absolute alpha_ox; we map the offset to the historical base of
    # -1.4 to preserve backwards compatibility (0.0 delta → -1.4 absolute).
    l_xray_contrib = jnp.zeros_like(wavelength)
    if include_xray:
        l_xray_contrib = _xray_agn_corona_legacy(
            wavelength,
            L_agn_bol=l_bol_erg,
            gamma=xray_gamma_agn,
            E_cut=xray_E_cut,
            alpha_ox=-1.4 + xray_alpha_ox,
        )

    # --- Radio emission (optional) ---
    l_radio_contrib = jnp.zeros_like(wavelength)
    if include_radio:
        l_radio_contrib = radio_total(
            wavelength,
            L_ir=0.0,  # No IR contribution to radio for AGN-only model
            L_agn_bol=l_bol_erg,
            q_ir=radio_q_ir,
            alpha_sf=radio_alpha_sf,
            radio_loudness=radio_loudness,
            alpha_agn=radio_alpha_agn,
            sfr_mode="bell2003",
            **_kwargs,
        )

    # --- Total ---
    l_total = l_disc_masked + l_torus + l_nlr + l_blr + l_xray_contrib + l_radio_contrib
    return l_total * agn_lum_ratio


# ── Pluggable NLR/BLR backend factories ──────────────────────────
