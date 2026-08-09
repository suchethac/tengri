# SPDX-License-Identifier: BSD-3-Clause
"""Dust emission models for tengri.

This module implements IR re-emission of UV/optical light absorbed by dust.
All models are pure JAX (JIT-compatible, fully differentiable) and follow
the energy-balance constraint: total IR luminosity equals total absorbed
luminosity from the attenuation step.

Available Emission Models
-------------------------

- **modified_blackbody**: Optically-thin modified blackbody (2-3 params)
- **casey2012**: Casey (2012) modified blackbody + mid-IR power law (3 params)
- **pah_drude**: Smith et al. (2007) PAH Drude profiles (0 params, pure template)
- **dale2014**: Dale et al. (2014) 1-parameter IR template family (tabulated)
- **draine_li2007**: Draine & Li (2007) 3-parameter model (tabulated)
- **draine_li2014**: Draine & Li (2014 update) 4-parameter model (tabulated)
- **astrodust**: Hensley & Draine (2023) Astrodust+PAH model (tabulated)
- **bosa**: Boquien & Salim (2021) (L_TIR, sSFR)-parameterized model (tabulated)
- **schreiber2018**: Schreiber et al. (2018) cold-dust template library (tabulated)
- **dh02_ce01**: Dale & Helou (2002) + Chary & Elbaz (2001) cold-dust model (tabulated)
- **themis**: Jones et al. (2017) THEMIS/DustEM model (tabulated)

Template Auto-Loading
---------------------
The ``"draine_li2007"``, ``"dale2014"``, ``"draine_li2014"``,
``"astrodust"``, ``"bosa"``, ``"schreiber2018"``, ``"dh02_ce01"``,
and ``"themis"`` models auto-load tabulated templates from the ``data/``
directory on first use.  If templates are not found, they raise
``FileNotFoundError`` (no analytic fallback available).

Energy Balance
--------------
The normalization for every model is set by::

    L_dust_emission = L_dust_absorbed
                    = integral[(1 - transmission) * L_stellar_intrinsic * dlambda]

This is computed from the attenuation step and passed to each model as
``L_absorbed`` (scalar, in Lsun).

References
----------

- Casey 2012, MNRAS, 425, 3094
- Chary, R. & Elbaz, D. 2001, ApJ, 556, 562 (CE01)
- Dale, D. A. & Helou, G. 2002, ApJ, 576, 159 (DH02)
- Dale et al. 2014, ApJ, 784, 83
- Draine & Li 2007, ApJ, 657, 810
- Draine & Li 2014 update (CIGALE implementation, Boquien+2019)
- Aniano et al. 2012, ApJ, 756, 138
- da Cunha et al. 2013, ApJ, 766, 13
- Hildebrand 1983, QJRAS, 24, 267
- Hensley & Draine 2023, ApJ, 948, 55 (Astrodust+PAH)
- Boquien & Salim 2021, A&A, 653, A149 (BOSA templates)
- Schreiber, C. et al. 2018, A&A, 609, A30 (S17)
- Jones et al. 2017, A&A, 602, A46 (THEMIS dust model)

"""

import contextlib
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

# ── Template search paths (resolved once, reused for all models) ──


def _find_data_file(filename: str) -> str | None:
    """Search standard data directories for a template file.

    Walks the CWD's parents so notebooks and scripts running from a
    subdirectory (e.g. ``reproduction/prospector/``) still resolve
    ``<repo>/data/``.

    This previously tried a static package-anchored list *before* falling back
    to the shared locator, which put ``$TENGRI_DATA_DIR`` last: a stale copy in
    the source tree outranked the directory the user had configured. The shared
    locator is a superset of that list and orders it correctly (#1431).
    """
    from tengri._data_setup import find_data

    found = find_data(filename)
    return str(found) if found is not None else None


# ── Dust emission model catalog ──────────────────────────────────

DUST_EMISSION_MODELS: dict[str, Callable] = {}

# Track which lazy loaders have been resolved to avoid duplicate loading
_resolved: set[str] = set()


def register_emission_model(name: str) -> Callable:
    """Decorator factory that registers a dust emission model under a name.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"dale2014"``, ``"draine_li2007"``).

    Returns
    -------
    Callable
        Decorator that registers the decorated function and returns it unchanged.

    Notes
    -----
    **JIT-compatible**: no — registration happens at factory time before JIT.

    Decorated functions must implement the ``DustEmissionTemplate`` protocol:
    accept a wavelength array, absorbed luminosity, and keyword arguments,
    returning an emission SED ``L_ν`` [erg/s/Hz].

    """

    def decorator(fn: Callable) -> Callable:
        """Inner decorator that registers function in DUST_EMISSION_MODELS dict.

        Parameters
        ----------
        fn : Callable
            Dust emission model function matching ``DustEmissionTemplate`` protocol.

        Returns
        -------
        Callable
            The input function unchanged (enables use as a decorator).

        Notes
        -----
        **JIT-compatible**: depends on the decorated function.

        """
        DUST_EMISSION_MODELS[name] = fn
        return fn

    return decorator


def preload_emission_model(name: str) -> Callable:
    """Force lazy template loading outside any JAX JIT scope.

    Template-based emission models use lazy loaders that fire on first call.
    If the first call happens inside a ``@jax.jit`` scope, ``jnp.array()``
    inside the loader creates ``DynamicJaxprTracer`` objects that escape into
    closures, causing ``UnexpectedTracerError`` on subsequent non-JIT calls.

    Call this function at factory time (outside JIT) so templates are loaded
    into ``DUST_EMISSION_MODELS[name]`` as regular ``DeviceArray`` objects.
    Dynamic JAX indexing inside JIT then works correctly.

    Parameters
    ----------
    name : str
        Registry name (e.g. ``"draine_li2007"``).

    Returns
    -------
    Callable
        The loaded (real) emission function — NOT a lazy wrapper.

    Notes
    -----
    **JIT-compatible**: no — template loading happens at factory time.

    Safe to call at ``SEDModel.__init__`` time to prevent tracer leaks
    when models are first called inside a ``@jax.jit`` scope.

    """
    if name not in DUST_EMISSION_MODELS:
        raise ValueError(
            f"Unknown emission model '{name}'. Available: {list(DUST_EMISSION_MODELS.keys())}"
        )
    if name not in _resolved:
        # Trigger lazy loading with dummy inputs; ignore computation output —
        # we only want the side effect of loading templates into the registry.
        import numpy as _np

        _dummy_wave = _np.linspace(1e3, 1e7, 5, dtype=_np.float64)
        with contextlib.suppress(Exception):
            DUST_EMISSION_MODELS[name](_dummy_wave, 1.0)
    return DUST_EMISSION_MODELS[name]


# ── Shared physics utilities ──────────────────────────────────────
# Moved to _physics.py (leaf: physics_constants + jnp) so closure modules and
# this facade can import them without an import cycle (#843).
from tengri.components.dust.emission._physics import (
    cmb_contrast_factor as cmb_contrast_factor,
    cmb_corrected_temperature as cmb_corrected_temperature,
    compute_absorbed_luminosity as compute_absorbed_luminosity,
    compute_absorbed_luminosity_from_tau as compute_absorbed_luminosity_from_tau,
    planck_bnu as planck_bnu,
)

# ── Model 1: Modified blackbody (2-3 parameters) ──────────────────
from tengri.components.dust.emission.analytic._closures import (
    casey2012 as casey2012,
    energy_balance_split as energy_balance_split,
    modified_blackbody as modified_blackbody,
    pah_drude as pah_drude,
    schreiber2016 as schreiber2016,
)

# Register the grammar-dispatchable analytic closures (defined in
# analytic/_closures.py; #843). energy_balance_split is intentionally NOT
# registered here — it dispatches via its _REGISTRY component only (single
# dispatch, #850).
DUST_EMISSION_MODELS["modified_blackbody"] = modified_blackbody
DUST_EMISSION_MODELS["casey2012"] = casey2012
DUST_EMISSION_MODELS["pah_drude"] = pah_drude
# Deprecated alias: draine2021_pah resolves to the canonical pah_drude (#693).
DUST_EMISSION_MODELS["draine2021_pah"] = pah_drude
DUST_EMISSION_MODELS["schreiber2016"] = schreiber2016


def draine_li2007(*args, **kwargs):
    """Draine & Li (2007) — dispatches to the registry (auto-loads templates).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters (alpha, U_min, gamma, q_pah). See registered
        function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded tabulated DL07 model. Auto-loads HDF5 templates
    on first call.

    """
    return DUST_EMISSION_MODELS["draine_li2007"](*args, **kwargs)


def dale2014(*args, **kwargs):
    """Dale et al. (2014) — dispatches to the registry (auto-loads templates).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters (alpha). See registered function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded Dale2014 template model. Auto-loads HDF5
    templates on first call.

    """
    return DUST_EMISSION_MODELS["dale2014"](*args, **kwargs)


def draine_li2014(*args, **kwargs):
    """Draine & Li (2014) — dispatches to the registry (auto-loads templates).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters (alpha, U_min, gamma, q_pah). See registered
        function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded DL14 template model. Auto-loads HDF5 templates
    on first call. DL14 is the 2014 update to Draine & Li 2007 with additional
    alpha (radiation field) dependence.

    """
    return DUST_EMISSION_MODELS["draine_li2014"](*args, **kwargs)


def astrodust(*args, **kwargs):
    """Astrodust+PAH (Hensley & Draine 2023) — dispatches to the registry.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters. See registered function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded Astrodust+PAH template model. Auto-loads
    HDF5 templates on first call.

    """
    return DUST_EMISSION_MODELS["astrodust"](*args, **kwargs)


def bosa(*args, **kwargs):
    """BOSA (Boquien & Salim 2021) — dispatches to the registry.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters. See registered function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded BOSA template model. Auto-loads HDF5
    templates on first call. BOSA parameterizes dust by (L_TIR, sSFR).

    """
    return DUST_EMISSION_MODELS["bosa"](*args, **kwargs)


def themis(*args, **kwargs):
    """THEMIS (Jones+2017) — dispatches to the registry.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters. See registered function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded THEMIS/DustEM template model. Auto-loads
    HDF5 templates on first call. Uses a-C(:H) aromatic carbon composition
    rather than PAH fraction.

    """
    return DUST_EMISSION_MODELS["themis"](*args, **kwargs)


# ── Energy-balance decomposition models ───────────────────────────


def apply_dust_emission(
    model_name: str,
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    **params,
) -> jnp.ndarray:
    r"""Apply a named dust emission model.

    Dispatches to a registered model function by name.

    Parameters
    ----------
    model_name : str
        Registered model name (e.g. "modified_blackbody", "draine_li2007").
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    L_absorbed : float
        Absorbed luminosity. [Lsun]
    **params
        Model-specific keyword arguments (e.g., dust_T, dust_umin, dust_gamma_dl).

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν. [Lsun Hz⁻¹]

    Raises
    ------
    ValueError
        If model_name is not registered.

    Notes
    -----
    **JIT-compatible**: yes if the underlying model is JIT-compatible.

    """
    if model_name not in DUST_EMISSION_MODELS:
        raise ValueError(
            f"Unknown dust emission model '{model_name}'. "
            f"Available: {list(DUST_EMISSION_MODELS.keys())}"
        )
    return DUST_EMISSION_MODELS[model_name](wavelength_aa, L_absorbed, **params)


# ── Lazy loading infrastructure for template-based models ─────────


def _make_lazy_loader(
    name: str,
    template_filename: str,
    loader_fn_name: str,
) -> Callable:
    """Create a lazy-loading wrapper that auto-loads templates on first call.

    Parameters
    ----------
    name : str
        Registry name (e.g. ``"dale2014"``).
    template_filename : str
        Canonical HDF5 filename to search for in data/ (e.g. ``"dale2014_templates.h5"``).
        The v2 variant (``"*_v2.h5"``) is tried first if present.
    loader_fn_name : str
        Name of the ``create_*_from_grid`` function in this module.

    """

    def _lazy_wrapper(*args, **kwargs):
        """Resolve and cache the dust emission template on first call, then delegate to it."""
        if name not in _resolved:
            _resolved.add(name)
            # Try v2 HDF5 first (improved grid), then canonical HDF5
            stem = template_filename.rsplit(".", 1)[0]
            v2_name = stem + "_v2.h5"
            path = _find_data_file(v2_name) or _find_data_file(template_filename)
            if path is not None:
                loader = globals()[loader_fn_name]
                try:
                    # First call typically happens INSIDE a jit trace (the
                    # model's first predict). Without this guard the template
                    # arrays are created as trace-staged constants, cached in
                    # the module registry, and every LATER model touching the
                    # slot dies with UnexpectedTracerError (or worse).
                    with jax.ensure_compile_time_eval():
                        tabulated = loader(path)
                except (KeyError, ValueError) as exc:
                    # Schema mismatch — file exists but doesn't match the
                    # legacy (qpah, umin) layout this loader expects.  This
                    # happens for astrodust / bosa / themis after the
                    # synthetic placeholder was replaced with real published
                    # data that has a different schema.
                    raise RuntimeError(
                        f"Template file '{path}' has an incompatible schema "
                        f"for the legacy {name!r} registry path: {exc!r}. "
                        f"Use the modern emission-component dispatch "
                        f"(dust={{'emission': {{'type': {name!r}}}}}) or the "
                        f"model-specific loader "
                        f"(load_astrodust_hd23_or_raise / load_bosa_*) "
                        f"instead."
                    ) from exc
                DUST_EMISSION_MODELS[name] = tabulated
                return tabulated(*args, **kwargs)
            else:
                raise FileNotFoundError(
                    f"Template file '{template_filename}' not found in data/. "
                    f"The analytic fallback for {name} has been removed because it "
                    f"produced scientifically incorrect results. Download templates "
                    f"or register manually via register_*_tabulated()."
                )
        # Already resolved.  If the slot still holds *this* lazy wrapper the
        # earlier resolution failed silently — fail loudly instead of
        # recursing forever.
        if DUST_EMISSION_MODELS[name] is _lazy_wrapper:
            raise RuntimeError(
                f"{name!r} lazy loader is in an inconsistent state — the "
                f"first resolution did not replace the registry entry. "
                f"Check {template_filename} or use the modern SEDComponent path."
            )
        return DUST_EMISSION_MODELS[name](*args, **kwargs)

    _lazy_wrapper.__name__ = name
    _lazy_wrapper.__doc__ = (
        f"Lazy-loading wrapper for {name}. Auto-loads tabulated templates "
        f"from data/{template_filename} on first call (v2 grid preferred if present)."
    )
    return _lazy_wrapper


def _find_dl07_templates() -> str | None:
    """Find DL07 template files, preferring v2 grid."""
    for fn in ("dl07_templates_v2.h5", "dl07_templates.h5"):
        path = _find_data_file(fn)
        if path is not None:
            return path
    return None


def _dl07_lazy_wrapper(*args, **kwargs):
    """Draine & Li (2007) — auto-loads tabulated templates on first call."""
    if "draine_li2007" not in _resolved:
        _resolved.add("draine_li2007")
        path = _find_dl07_templates()
        if path is not None:
            from tengri.components.dust.emission_templates import create_dl07_from_grid

            # Same trace-escape guard as _make_lazy_loader: the first call
            # usually runs inside a jit trace, and this wrapper resolves
            # THREE registry slots (draine_li2007 / dl07_tabulated / dl07
            # aliases) shared by later models.
            with jax.ensure_compile_time_eval():
                tabulated = create_dl07_from_grid(path)
            DUST_EMISSION_MODELS["draine_li2007"] = tabulated
            DUST_EMISSION_MODELS["dl07_tabulated"] = tabulated
            return tabulated(*args, **kwargs)
        else:
            raise FileNotFoundError(
                "DL07 template files (dl07_templates_v2.h5 / dl07_templates.h5) "
                "not found in data/. "
                "The analytic fallback has been removed because it produced "
                "scientifically incorrect results (single-Gaussian PAH approximation). "
                "Run: python scripts/convert_dl07_templates.py"
            )
    # Already resolved. If the slot still holds *this* wrapper the first
    # resolution failed (and its exception was swallowed upstream) — fail
    # loudly instead of recursing forever, same guard as _make_lazy_loader.
    if DUST_EMISSION_MODELS["draine_li2007"] is _dl07_lazy_wrapper:
        raise RuntimeError(
            "'draine_li2007' lazy loader is in an inconsistent state — the "
            "first resolution did not replace the registry entry (template "
            "files missing?). Check data/dl07_templates_v2.h5 or use the "
            "modern SEDComponent path."
        )
    return DUST_EMISSION_MODELS["draine_li2007"](*args, **kwargs)


# ── Import emission template functions ───────────────────────────

from tengri.components.dust.emission_templates import (
    create_astrodust_from_grid as create_astrodust_from_grid,
    create_bosa_from_grid as create_bosa_from_grid,
    create_dale2014_from_grid as create_dale2014_from_grid,
    create_dh02_ce01_from_grid as create_dh02_ce01_from_grid,
    create_dl07_from_grid as create_dl07_from_grid,
    create_dl14_from_grid as create_dl14_from_grid,
    create_schreiber2018_from_grid as create_schreiber2018_from_grid,
    create_themis_from_grid as create_themis_from_grid,
    load_astrodust_templates as load_astrodust_templates,
    load_bosa_templates as load_bosa_templates,
    load_dale2014_templates as load_dale2014_templates,
    load_dl14_templates as load_dl14_templates,
    load_draine_li_templates as load_draine_li_templates,
    load_schreiber2018_templates as load_schreiber2018_templates,
    load_themis_templates as load_themis_templates,
    register_astrodust_tabulated as register_astrodust_tabulated,
    register_bosa_tabulated as register_bosa_tabulated,
    register_dale2014_tabulated as register_dale2014_tabulated,
    register_dl07_tabulated as register_dl07_tabulated,
    register_dl14_tabulated as register_dl14_tabulated,
    register_themis_tabulated as register_themis_tabulated,
)

# NOTE: ``energy_balance_split`` is NOT registered in ``DUST_EMISSION_MODELS``.
# Its canonical dispatch is the ``EnergyBalanceSplitIRSEDComponent`` component in
# ``_REGISTRY`` (analytic/energy_balance_split.py), which calls the pure
# :func:`energy_balance_split` closure directly with ``eta_balance=1.0``.
# A second registration here would create a divergent dispatch path
# (``apply_dust_emission("energy_balance_split")`` -> closure with the raw
# ``eta_balance`` default), defeating the single-dispatch invariant (#850).

# Register lazy loaders at module load time
# These will auto-load templates on first call
DUST_EMISSION_MODELS["draine_li2007"] = _dl07_lazy_wrapper

DUST_EMISSION_MODELS["dale2014"] = _make_lazy_loader(
    "dale2014",
    "dale2014_templates.h5",
    "create_dale2014_from_grid",
)

# CIGALE-sourced Dale2014 variant: same SF templates regenerated from CIGALE's
# database AND the pure-AGN quasar template, so the ``dust_frac_agn`` AGN-heated
# mixing (L_AGN = L_dust*f/(1-f); CIGALE dale2014.py) actually has a quasar SED
# to add. The default ``dale2014`` stays the Wyoming-source SF-only release that
# the contract tests pin. See scripts/regenerate_dale2014_from_cigale.py.
DUST_EMISSION_MODELS["dale2014_cigale"] = _make_lazy_loader(
    "dale2014_cigale",
    "dale2014_templates_cigale.h5",
    "create_dale2014_from_grid",
)

# Schreiber et al. (2018) "S17" cold-dust library — the tabulated, real-PAH
# counterpart of the analytic ``schreiber2016`` model. Same
# (dust_T, dust_f_pah) interface; grid data published with AGNfitter-rX
# (scripts/build_schreiber2018_grid.py).
DUST_EMISSION_MODELS["schreiber2018"] = _make_lazy_loader(
    "schreiber2018",
    "schreiber2018_templates.h5",
    "create_schreiber2018_from_grid",
)


@functools.cache
def _load_dl14_fn():
    """Load DL14 template grid from file."""
    from tengri.components.dust.emission_templates import create_dl14_from_grid

    for fname in ("dl14_templates_v2.h5", "dl14_templates.h5"):
        path = _find_data_file(fname)
        if path is not None:
            return create_dl14_from_grid(path)
    raise FileNotFoundError(
        "DL14 template files not found (dl14_templates_v2.h5 or dl14_templates.h5). "
        "The analytic fallback has been removed because it produced scientifically "
        "incorrect results. Run: python scripts/download_dl14_templates.py"
    )


def _dl14_lazy_wrapper(*args, **kwargs):
    """Lazy loader for DL14: prioritizes v2 grid, falls back to legacy grid."""
    fn = _load_dl14_fn()
    return fn(*args, **kwargs)


DUST_EMISSION_MODELS["draine_li2014"] = _dl14_lazy_wrapper

DUST_EMISSION_MODELS["astrodust"] = _make_lazy_loader(
    "astrodust",
    "astrodust_templates.h5",
    "create_astrodust_from_grid",
)

DUST_EMISSION_MODELS["bosa"] = _make_lazy_loader(
    "bosa",
    "bosa_templates.h5",
    "create_bosa_from_grid",
)

DUST_EMISSION_MODELS["themis"] = _make_lazy_loader(
    "themis",
    "themis_templates.h5",
    "create_themis_from_grid",
)

DUST_EMISSION_MODELS["dh02_ce01"] = _make_lazy_loader(
    "dh02_ce01",
    "dh02_ce01_grid.h5",
    "create_dh02_ce01_from_grid",
)

# ── Friendly aliases ─────────────────────────────────────────────
# Short names commonly used in the literature and surfaced by
# ``tengri.list_dust_emission_models()``. Without these entries the
# ``SEDModel.build(..., dust={"emission": {"type": "dl07"}})`` validator
# (which derives accepted types from this dict) rejected the names that
# the public introspection helper advertised. Closes #495.
DUST_EMISSION_MODELS["dl07"] = DUST_EMISSION_MODELS["draine_li2007"]
DUST_EMISSION_MODELS["dl14"] = DUST_EMISSION_MODELS["draine_li2014"]
DUST_EMISSION_MODELS["mbb"] = modified_blackbody
