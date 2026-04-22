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
            Filter labels for the x-axis. Falls back to integer indices.
        ax : matplotlib Axes, optional
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        fig : matplotlib Figure
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
        Predicted photometry draws, shape ``(n, n_filters)``.
        None if the model has no filters.
    sfh : jnp.ndarray
        SFH draws, shape ``(n, n_grid)``.
    params : dict
        Drawn parameter samples, each of shape ``(n,)``.
    _model : object
        Back-reference to the parent model.

    Examples
    --------
    .. code-block:: python

        import jax
        from tengri import SEDModel, Parameters, Uniform
        spec = Parameters(sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.5, 4.0))
        model = SEDModel(spec, ssp_data, filter_names=["sdss_r", "sdss_i"])
        ppc = model.prior_predictive(n=200, key=jax.random.PRNGKey(0))
        ppc.flux.shape    # (200, 2)
        ppc.sfh.shape     # (200, n_grid)
        check = ppc.check_finite()
        check["ok"]       # True if no NaN/Inf
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
            ``{"n_nan": int, "n_inf": int, "frac_bad": float, "ok": bool}``
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
    """

    photometry: object | None = None
    photometry_ztable: object | None = None
    spectrum: object | None = None
