# SPDX-License-Identifier: BSD-3-Clause
"""Supporting types for SEDModel: MockData, PriorPredictive, and internal kernel containers."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp

from tengri.config.exceptions import warn_measured

# ── MockData container ────────────────────────────────────────────


class MockData(NamedTuple):
    """Container for mock galaxy observations.

    Parameters
    ----------
    flux_true: ndarray, shape (n_filters,)
        Noiseless model photometry. [erg/s/cm²/Hz]
    flux_obs: ndarray, shape (n_filters,)
        Noisy photometry with Gaussian scatter added. [erg/s/cm²/Hz]
    noise: ndarray, shape (n_filters,)
        1-sigma photometric uncertainties used to draw the noise. [erg/s/cm²/Hz]
    params: dict
        Input physical parameters used to generate the mock.

    Returns
    -------
    MockData
        Named tuple containing noiseless and noisy photometry.

    Attributes
    ----------
    flux_true: ndarray, shape (n_filters,)
        Noiseless model photometry. [erg/s/cm²/Hz]
    flux_obs: ndarray, shape (n_filters,)
        Noisy photometry with Gaussian scatter added. [erg/s/cm²/Hz]
    noise: ndarray, shape (n_filters,)
        1-sigma photometric uncertainties used to draw the noise. [erg/s/cm²/Hz]
    params: dict
        Input physical parameters used to generate the mock.

    Notes
    -----
    **JIT-compatible**: yes, NamedTuple is a JAX pytree.

    **Immutable**: All fields are read-only by design. To create a modified
    version, use the ``_replace()`` method inherited from NamedTuple.

    Examples
    --------
    .. code-block:: python

        from tengri import SEDModel, Parameters, Uniform
        model = SEDModel(spec, ssp_data, filter_names=["hst_acs_f606w", "hst_acs_f814w"])
        params = {"sfh_dpl_alpha": 2.0, "sfh_dpl_beta": 1.5, ...}
        mock = model.make_mock(params, snr=20.0)
        mock.flux_obs.shape    # (n_filters,)
        mock.plot()            # matplotlib Figure
    """

    flux_true: jnp.ndarray  # noiseless photometry (erg/s/cm²/Hz)
    flux_obs: jnp.ndarray  # noisy photometry
    noise: jnp.ndarray  # 1-sigma uncertainties
    params: dict  # input parameters

    # ── Mapping-style access ──────────────────────────────────────────
    # MockData is the object returned by ``SEDModel.mock()``; the standalone
    # ``generate_mock()`` returns a mapping. Supporting both ``mock.flux_obs``
    # and ``mock["flux_obs"]`` here means notebook/user code written against
    # either surface works against both, see ``src/tengri/analysis/mock.py``.

    def __getitem__(self, key):
        """Field access by name (``mock["flux_obs"]``) or position (``mock[0]``).

        Parameters
        ----------
        key: str or int or slice
            Field name for dict-style access, or an index/slice for the
            NamedTuple's positional access.

        Returns
        -------
        object
            The requested field value (or tuple slice).

        Raises
        ------
        KeyError
            If ``key`` is a string that is not a field name.
        """
        if isinstance(key, str):
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError(key) from None
        # Explicit base (not zero-arg super()): a typing.NamedTuple metaclass
        # does not set the __class__ cell, so super() fails at class creation.
        return tuple.__getitem__(self, key)

    def __contains__(self, key) -> bool:
        """``"flux_obs" in mock`` tests field names; other values fall back to tuple membership."""
        if isinstance(key, str):
            return key in self._fields
        return tuple.__contains__(self, key)

    def get(self, key: str, default=None):
        """Return field ``key`` if present, else ``default`` (mirrors :meth:`dict.get`)."""
        return getattr(self, key) if isinstance(key, str) and key in self._fields else default

    def keys(self):
        """Return the field names (``flux_true``, ``flux_obs``, ``noise``, ``params``)."""
        return list(self._fields)

    def values(self):
        """Return the field values, in field order."""
        return list(self)

    def items(self):
        """Return ``(name, value)`` pairs, in field order."""
        return list(zip(self._fields, self))

    def plot(self, filter_names=None, ax=None):
        """Plot mock photometry with errorbars.

        Parameters
        ----------
        filter_names: list of str, optional
            Filter labels for the x-axis. Falls back to integer indices if None.
        ax: matplotlib Axes, optional
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        fig: matplotlib Figure
            Matplotlib figure with photometry plotted as error bars
            (observed with noise) and markers (true noiseless).

        Notes
        -----
        **JIT-compatible**: no, uses matplotlib for visualization.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.get_figure()

        n = len(self.flux_true)
        x = np.arange(n)
        labels = filter_names if filter_names is not None else [str(i) for i in x]

        ax.errorbar(
            x,
            np.array(self.flux_obs),
            yerr=np.array(self.noise),
            fmt="o",
            color="C0",
            label="observed (noisy)",
            capsize=3,
            zorder=3,
        )
        ax.plot(x, np.array(self.flux_true), "s--", color="C1", label="true (noiseless)", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]", fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        ax.set_title("Mock Photometry", fontsize=11)
        fig.tight_layout()
        return fig


# ── PriorPredictive container ─────────────────────────────────────


@dataclasses.dataclass
class PriorPredictive:
    """Results of a prior predictive check.

    Parameters
    ----------
    flux: jnp.ndarray or None
        Predicted photometry draws [erg/s/cm²/Hz], shape ``(n, n_filters)``.
        None if the model has no filters.
    sfh: jnp.ndarray
        SFR on log-age grid [Msun/yr], shape ``(n, n_grid)``.
    params: dict
        Drawn parameter samples, each of shape ``(n,)``.

    Returns
    -------
    PriorPredictive
        Results container with draws from the prior predictive distribution.

    Attributes
    ----------
    flux: jnp.ndarray or None
        Predicted photometry draws [erg/s/cm²/Hz], shape ``(n, n_filters)``.
        None if the model has no filters.
    sfh: jnp.ndarray
        SFR on log-age grid [Msun/yr], shape ``(n, n_grid)``.
    params: dict
        Drawn parameter samples, each of shape ``(n,)``.
    _model: object
        Back-reference to the parent model.

    Notes
    -----
    Use :meth:`check_finite` to diagnose NaN/Inf in prior draws before inference.

    Examples
    --------
    .. code-block:: python

        import jax
        from tengri import SEDModel, Parameters, Uniform

        spec = Parameters(sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.5, 4.0))
        model = SEDModel(spec, ssp_data, filter_names=["sdss_r", "sdss_i"])
        ppc = model.prior_predictive(n=200, seed=0)
        ppc.flux.shape  # (200, 2)
        ppc.sfh.shape  # (200, n_grid)
        check = ppc.check_finite()
        check["ok"]  # True if no NaN/Inf
    """

    flux: jnp.ndarray | None
    sfh: jnp.ndarray
    params: dict
    _model: object = dataclasses.field(default=None, repr=False)

    def check_finite(self) -> dict:
        """Check for NaN/Inf in flux draws.

        Parameters
        ----------
        self: PriorPredictive
            The prior predictive result container.

        Returns
        -------
        dict
            Diagnostic dict with keys:

            - ``"n_nan"``: count of NaN values [dimensionless]
            - ``"n_inf"``: count of Inf values [dimensionless]
            - ``"frac_bad"``: fraction of bad (NaN or Inf) values [dimensionless]
            - ``"ok"``: bool, True if no NaN/Inf found

        Notes
        -----
        **JIT-compatible**: no, uses Python-level checking and warnings.
        """
        import numpy as np

        if self.flux is None:
            return {"n_nan": 0, "n_inf": 0, "frac_bad": 0.0, "ok": True}

        flux_np = np.array(self.flux)
        n_nan = int(np.sum(np.isnan(flux_np)))
        n_inf = int(np.sum(np.isinf(flux_np)))
        total = flux_np.size
        frac_bad = (n_nan + n_inf) / max(total, 1)
        if n_nan + n_inf > 0:
            warn_measured(
                f"prior_predictive: {n_nan} NaN and {n_inf} Inf values in flux draws "
                f"({frac_bad:.1%} of total). Check priors for extreme parameter combinations.",
                UserWarning,
                stacklevel=2,
                frac_bad=frac_bad,
                n_nan=n_nan,
                n_inf=n_inf,
            )
        return {"n_nan": n_nan, "n_inf": n_inf, "frac_bad": frac_bad, "ok": (n_nan + n_inf == 0)}


# ── Forward-kernel runtime bundle ─────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SEDModelState:
    """Frozen runtime bundle of everything the forward kernels need.

    This is the kernel-layer seam: ``hybrid``, ``compositional``, and ``exact``
    builders accept ``state: SEDModelState`` instead of the full ``SEDModel``
    instance. Built once at ``SEDModel.__init__`` and updated immutably via
    ``dataclasses.replace`` when ``precompute_spectroscopy`` /
    ``precompute_ztable`` are called.

    Future ``SpatialModelState`` and composed ``SpatialSEDModelState`` will
    follow the same pattern for spatially-resolved SED fitting.

    Notes
    -----
    Frozen dataclass; share by reference. To update a field, use
    ``dataclasses.replace(state, field=new_value)``.
    """

    spec: object  # Parameters
    ssp_data: object  # SSPData
    filter_waves: object | None  # list of jnp.ndarray
    filter_trans: object | None  # list of jnp.ndarray
    rest_wavelength: jnp.ndarray
    log_age_grid: jnp.ndarray
    age_yr: jnp.ndarray
    d_log_age: float
    n_grid: int
    ssp_log_ages_yr: jnp.ndarray
    ssp_ages_yr: jnp.ndarray
    csp_matrix: jnp.ndarray | None
    csp_age_dt: jnp.ndarray | None
    csp_integration: str
    forward_dtype: object  # jnp.dtype
    met_interp: str
    met_mode: str
    z_interp: str
    lgmet_scatter: float
    sfh_fn: object
    sfh_internal_names: object  # set[str]
    uses_stochastic_sfh: bool
    gp_kernel: str
    dust_model: str
    dust_law_bc: str
    dust_law_diff: str
    dust_law_bc_fn: object
    dust_law_diff_fn: object
    dust_emission_model: str | None
    nebular_backend: object
    agn_model: str | None
    agn_config: object | None
    agn_luminosity_mode: bool
    uses_igm: bool
    uses_radio: bool
    uses_xray: bool
    radio_include_freefree: bool | None
    radio_sfr_mode: str | None
    radio_agn_model: str | None
    z_fixed: float | None
    dl_cm_fixed: float | None
    param_map: object
    igm_fn: object | None
    wave_obs: jnp.ndarray | None = None
