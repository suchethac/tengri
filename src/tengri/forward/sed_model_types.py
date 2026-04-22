"""Supporting types for SEDModel: MockData, PriorPredictive, and internal kernel containers."""

from __future__ import annotations

import dataclasses
import warnings
from typing import NamedTuple

import jax.numpy as jnp

# ── MockData container ────────────────────────────────────────────


class MockData(NamedTuple):
    """Container for mock galaxy observations.

    Attributes
    ----------
    flux_true : ndarray, shape (n_filters,)
        Noiseless model photometry. [erg/s/cm²/Hz]
    flux_obs : ndarray, shape (n_filters,)
        Noisy photometry with Gaussian scatter added. [erg/s/cm²/Hz]
    noise : ndarray, shape (n_filters,)
        1-sigma photometric uncertainties used to draw the noise. [erg/s/cm²/Hz]
    params : dict
        Input physical parameters used to generate the mock.

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

    def plot(self, filter_names=None, ax=None):
        """Plot mock photometry with errorbars.

        Parameters
        ----------
        filter_names : list of str, optional
            Filter labels for the x-axis. Falls back to integer indices if None.
        ax : matplotlib Axes, optional
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        fig : matplotlib Figure
            Matplotlib figure with photometry plotted as error bars
            (observed with noise) and markers (true noiseless).

        Notes
        -----
        **JIT-compatible**: no — uses matplotlib for visualization.
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

    Attributes
    ----------
    flux : jnp.ndarray or None
        Predicted photometry draws [erg/s/cm²/Hz], shape ``(n, n_filters)``.
        None if the model has no filters.
    sfh : jnp.ndarray
        SFR on log-age grid [Msun/yr], shape ``(n, n_grid)``.
    params : dict
        Drawn parameter samples, each of shape ``(n,)``.
    _model : object
        Back-reference to the parent model.

    Returns
    -------
    This is a dataclass returned by :func:`prior_predictive`.

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
        **JIT-compatible**: no — uses Python-level checking and warnings.
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
            warnings.warn(
                f"prior_predictive: {n_nan} NaN and {n_inf} Inf values in flux draws "
                f"({frac_bad:.1%} of total). Check priors for extreme parameter combinations.",
                UserWarning,
                stacklevel=2,
            )
        return {"n_nan": n_nan, "n_inf": n_inf, "frac_bad": frac_bad, "ok": (n_nan + n_inf == 0)}


# ── Kernel hierarchy dataclasses ──────────────────────────────────


@dataclasses.dataclass
class PrecomputedData:
    """Level 1: Precomputed SSP tensors pre-integrated through filters.

    Data only — no JIT kernels. Built once at ``SEDModel.__init__`` and
    updated by ``precompute_spectroscopy()`` / ``precompute_ztable()``.

    Attributes
    ----------
    photometry : object or None
        PhotometricPrecomputation for fixed redshift.
    photometry_ztable : object or None
        PhotometricZTable for free redshift mode.
    spectroscopy : object or None
        SpectroscopicPrecomputation for spectrum prediction.
    dust_age_weights : ndarray or None
        Sigmoid weights [dimensionless] for two-component dust attenuation, shape (n_age,).
    igm_at_effective_wavelengths : ndarray or None
        IGM transmission T(λ_eff) at filter effective wavelengths, shape (n_filters,).
    effective_bandwidths_hz : ndarray or None
        Effective bandwidth per filter [Hz], shape (n_filters,).
    dust_ir_lookup : object or None
        Preintegrated template-based dust IR photometry lookup.
    kd_preintegrated : object or None
        KDPreintegratedData for K&D AGN disc model.
    skirtor_preintegrated : object or None
        Preintegrated SKIRTOR torus photometry lookup.

    Notes
    -----
    This is an internal container used by SEDModel. Users do not construct this directly.
    """

    photometry: object | None = None  # PhotometricPrecomputation (fixed z)
    photometry_ztable: object | None = None  # PhotometricZTable (free z)
    spectroscopy: object | None = None  # SpectroscopicPrecomputation
    dust_age_weights: jnp.ndarray | None = None  # sigmoid weights for two-component dust
    igm_at_effective_wavelengths: jnp.ndarray | None = None  # IGM T(λ_eff) for fixed z
    effective_bandwidths_hz: jnp.ndarray | None = None  # Voronoi Δν per filter (Hz)
    dust_ir_lookup: object | None = None  # Preintegrated template-based dust IR photometry
    kd_preintegrated: object | None = None  # KDPreintegratedData for K&D AGN disc
    skirtor_preintegrated: object | None = None  # Preintegrated SKIRTOR torus photometry lookup


@dataclasses.dataclass
class CompositionalKernels:
    """Level 2: Full-resolution JIT-compiled kernels.

    These compute entire SEDs at full wavelength resolution. The ``rest_sed``
    kernel is the compositional engine; ``photometry`` and ``spectrum`` wrap
    it with params translation + filter/wavelength integration.

    Attributes
    ----------
    rest_sed : object or None
        JIT-compiled kernel for full rest-frame SED [erg/s/Hz].
    photometry : object or None
        JIT-compiled kernel for photometric predictions [erg/s/cm²/Hz].
    spectrum : object or None
        JIT-compiled kernel for spectroscopic predictions [erg/s/cm²/Hz].
    exact_sed : object or None
        JIT-compiled exact SED kernel with fused dust+stellar.

    Notes
    -----
    This is an internal container used by SEDModel. Users do not construct this directly.
    All kernels are callables (built via :func:`build_fused_rest_sed`, etc.).
    """

    rest_sed: object | None = None  # build_fused_rest_sed
    photometry: object | None = None  # build_fused_tier2_photometry (renamed)
    spectrum: object | None = None  # build_fused_tier2_spectrum (renamed)
    exact_sed: object | None = None  # build_exact_sed


@dataclasses.dataclass
class HybridKernels:
    """Mode 3: Precomputed SSP stellar + compositional non-stellar.

    Stellar photometry uses the precomputed SSP×filter einsum (fast).
    Non-stellar components use emission_helpers.py at full wavelength
    resolution, then integrate through filters. Populated in PR 2.

    Attributes
    ----------
    photometry : object or None
        Hybrid kernel for fixed-redshift photometry [erg/s/cm²/Hz].
    photometry_ztable : object or None
        Hybrid kernel for free-redshift photometry [erg/s/cm²/Hz].
    spectrum : object or None
        Hybrid kernel for spectroscopy [erg/s/cm²/Hz].

    Notes
    -----
    This is an internal container used by SEDModel. Users do not construct this directly.
    Hybrid kernels trade speed (precomputed stellar) for accuracy (compositional non-stellar).
    """

    photometry: object | None = None
    photometry_ztable: object | None = None
    spectrum: object | None = None
