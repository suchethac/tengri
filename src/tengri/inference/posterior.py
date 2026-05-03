"""Posterior inference results with sampling and diagnostics.

The Posterior object stores parameter samples (or point estimates for MAP),
provides summary statistics, derived quantities, and can convert to ArviZ
format or back to a Parameters for mock generation.

Usage:
    result = model.fit(data, noise, method="mcmc_nuts")
    print(result.stats())
    sfh_draws = [model.predict_sfh(result.resample(key)) for ...]
    idata = result.to_arviz()
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field

__all__ = ["Posterior"]

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def _stack_or_nan(values: list, n: int) -> jnp.ndarray:
    """Stack array values or return NaN-filled array if any value is None.

    Handles the case where a derived quantity is unavailable (None) for all
    samples, as occurs when predict_derived() returns None for a field
    (e.g., stellar_mass_surviving when SSP mass-remaining table is absent).

    Parameters
    ----------
    values : list
        List of values (jnp.ndarray, scalar, or None).
    n : int
        Expected number of samples. Used to shape the NaN array if needed.

    Returns
    -------
    ndarray, shape (n,)
        If all values are not-None: stacked array with jnp.asarray defensiveness.
        If any value is None: NaN-filled array of shape (n,).
    """
    if any(x is None for x in values):
        return jnp.full(n, jnp.nan)
    return jnp.stack([jnp.asarray(x) for x in values])


@dataclass
class Posterior:
    """Inference results with samples, diagnostics, and derived quantities.

    Stores posterior samples (or point estimate for MAP), best-fit parameters,
    convergence diagnostics, and provides methods for summary statistics,
    derived physical quantities, ArviZ conversion, and refinement via resampling
    or additional fitting iterations.

    Parameters
    ----------
    samples : dict or None
        Posterior samples in physical parameter space (optional, set by inference).
    params : dict
        Best-fit or posterior mean parameters.
    method : str
        Inference method name (e.g., ``"vi"``, ``"mcmc_nuts"``, ``"map"``).
    wall_time_s : float
        Total wall-clock runtime in seconds.
    diagnostics : dict
        Method-specific convergence metrics.
    loss_history : ndarray or None
        Optimization loss values (optimization methods only).
    log_evidence : float or None
        Log Bayesian evidence (NSS only).
    _model : SEDModel, optional
        Forward model reference.
    _fitter : Fitter, optional
        Fitter reference for refinement methods.
    eline_fluxes : ndarray or None
        Emission line fluxes [erg/s/cm²].
    eline_flux_cov : ndarray or None
        Emission line flux covariance.
    eline_names : tuple or None
        Emission line identifiers.
    eline_wavelengths : ndarray or None
        Rest-frame vacuum wavelengths [Angstrom].

    Returns
    -------
    Posterior
        Posterior instance with results populated.

    Attributes
    ----------
    samples : dict or None
        Posterior samples in physical parameter space. Each value has shape
        (n_samples, ...). Keys are parameter names (e.g., ``"stellar_mass"``,
        ``"age_gyr"``, ``"psd_xi"``). ``None`` for point estimates (MAP, Laplace,
        Pathfinder).

    params : dict
        Best-fit (MAP for point estimation) or posterior mean parameters in
        physical space. Same keys as ``samples`` (without ``"psd_xi"`` latent field).

    method : str
        Inference method name (e.g., ``"vi"``, ``"mcmc_nuts"``, ``"map"``).

    wall_time_s : float
        Total wall-clock runtime in seconds, including compilation and sampling.

    diagnostics : dict
        Method-specific convergence and quality metrics. Contents vary by method:

        - **VI methods**: ``{"kl_iter": int, "kl_final": float}``, etc.
        - **NUTS**: ``{"n_divergent": int, "accept_rate": float}``, etc.
        - **Ray Tracing**: ``{"accept_rate": float, "step_size": float}``, etc.
        - **MAP**: ``{"final_loss": float, "n_steps": int}``, etc.
        - **NSS**: ``{"n_live": int, "log_evidence_err": float}``, etc.

    loss_history : ndarray or None
        Optimization loss values over iterations (MAP/Laplace/Pathfinder only).
        Shape (n_iterations,). ``None`` for sampling methods.

    log_evidence : float or None
        Bayesian evidence log(Z) integral (NSS only). ``None`` for other methods.
        Used for model comparison via Bayes factors.

    _model : SEDModel, optional
        Reference to the forward model. Required for computing derived quantities
        (stellar mass, SFR, sSFR, etc.). Set by ``Fitter.run()`` automatically.

    _fitter : Fitter, optional
        Reference to the Fitter instance. Enables ``refine()`` and other
        refinement methods. Set by ``Fitter.run()`` automatically.

    eline_fluxes : ndarray or None
        Emission line fluxes. Shape (n_lines,) for MAP, (n_samples, n_lines) for
        sampling. ``None`` if no emission line fitting/marginalization was enabled.
        Flux units match input data [erg/s/cm²].

    eline_flux_cov : ndarray or None
        Posterior covariance of emission line fluxes. Shape (n_lines, n_lines) for
        MAP, (n_samples, n_lines, n_lines) for sampling. ``None`` if unavailable.

    eline_names : tuple or None
        Emission line identifiers (e.g., ``("Halpha", "Hbeta", ...)``)
        matching ``eline_fluxes`` column order.

    eline_wavelengths : ndarray or None
        Rest-frame vacuum wavelengths [Angstrom] of emission lines, matching
        ``eline_fluxes`` column order.

    Notes
    -----
    **Derived quantities**: The ``derived`` property computes stellar mass, SFR,
    sSFR, etc. by re-running the forward model on all samples. For MAP results,
    returns scalars; for MCMC/VI results, returns arrays (one per sample).

    **Emission line diagnostics**: Methods ``line_fluxes()``, ``bpt_nii()``,
    and ``balmer_decrement()`` provide astrophysical diagnostics on emission
    lines. Require ``eline_mode != "none"`` in Spectroscopy config.

    **Convergence diagnostics**: Use ``check_convergence()``, ``autocorrelation()``,
    and ``effective_sample_size()`` to assess MCMC chain quality.

    **Resampling and refinement**: Use ``resample()`` to draw new samples from
    the posterior, and ``refine()`` to improve results by running additional
    inference iterations (requires ``_fitter``).

    See Also
    --------
    Fitter.run : Returns Posterior with all attributes populated.
    Fitter : Primary interface for inference.

    Examples
    --------
    **Basic usage:**

    >>> result = fitter.run("mcmc_nuts")  # Returns Posterior
    >>> print(result.summary_table())
    >>> params_phys = result.params
    >>> samples = result.samples

    **Derived quantities:**

    >>> derived = result.derived
    >>> stellar_masses = derived["stellar_mass"]  # Shape (n_samples,)
    >>> med, lo, hi = np.percentile(stellar_masses, [50, 16, 84])

    **Emission line diagnostics:**

    >>> fluxes = result.line_fluxes()
    >>> ha_med, ha_lo, ha_hi = fluxes["Halpha"]
    >>> x, y = result.bpt_nii()  # BPT diagram coordinates
    >>> plt.scatter(x, y, alpha=0.3)

    **Convergence checks:**

    >>> converged = result.check_convergence()
    >>> ess = result.effective_sample_size()
    >>> print(f"Effective sample size: {ess['stellar_mass']:.0f}")

    **Refinement via resampling:**

    >>> refined_samples = result.resample(key, n=50)  # Resample with replacement
    >>> refined = fitter.run("vi", init_from=result)  # Refine posterior
    """

    samples: dict | None
    params: dict
    method: str
    wall_time_s: float
    diagnostics: dict
    loss_history: jnp.ndarray | None = None
    log_evidence: float | None = None
    _model: object = field(default=None, repr=False)
    _fitter: object = field(default=None, repr=False)
    eline_fluxes: jnp.ndarray | None = field(default=None, repr=False)
    eline_flux_cov: jnp.ndarray | None = field(default=None, repr=False)
    eline_names: tuple | None = field(default=None, repr=False)
    eline_wavelengths: jnp.ndarray | None = field(default=None, repr=False)

    # ── Derived quantities ────────────────────────────────────────

    @functools.cached_property
    def derived(self) -> dict:
        """Derived physical quantities (stellar mass, SFR, sSFR).

        For MAP: computed on the single best-fit → dict of scalars.
        For NUTS/geoVI: computed on all samples → dict of arrays.

        Returns
        -------
        dict
            Keys: derived quantity names (``"stellar_mass"``, ``"sfr_100myr"``, etc.).
            For MAP: values are scalars.
            For sampling: values are arrays of shape (n_samples,).
            Units match the forward model convention (stellar mass in [Msun],
            SFR in [Msun/yr]).

        Notes
        -----
        This is a cached property — computed on first access and cached thereafter.
        Requires ``_model`` to be set (populated automatically by ``Fitter.run()``).
        For stochastic SFH, unresolved bursts are included in ``sfr_10myr`` and
        ``sfr_100myr`` outputs.

        Examples
        --------
        >>> derived = result.derived
        >>> stellar_masses = derived["stellar_mass"]  # Shape (n_samples,)
        >>> med, lo, hi = np.percentile(stellar_masses, [50, 16, 84])
        """
        if self._model is None:
            raise RuntimeError("No model reference — cannot compute derived quantities")

        if self.samples is None:
            # MAP: single point
            return self._model.predict_derived(self.params)

        # Sampling: compute for each sample
        n_samples = next(iter(self.samples.values())).shape[0]
        derived_lists = {}

        for i in range(n_samples):
            sample_i = {k: v[i] for k, v in self.samples.items()}
            d_i = self._model.predict_derived(sample_i)
            for k, v in d_i.items():
                if k not in derived_lists:
                    derived_lists[k] = []
                derived_lists[k].append(v)

        return {k: _stack_or_nan(v, n_samples) for k, v in derived_lists.items()}

    def line_fluxes(self) -> dict[str, tuple[float, float, float]]:
        """Emission line flux posterior summaries.

        Returns median and 68% credible interval for each emission line.

        Returns
        -------
        dict
            ``{line_name: (median, lo_68, hi_68)}`` for each emission line.
            Flux units match the input data [erg/s/cm²]. For MAP results, all three
            values are the same (single-point estimate).

        Raises
        ------
        ValueError
            If no emission line fluxes are available. Set
            ``eline_mode="marginalized"`` or ``"fitted"`` in ``Spectroscopy`` to enable.

        Notes
        -----
        Each line's credible interval is computed as the 16th, 50th, and 84th
        percentiles of the posterior samples. For single samples (MAP), all three
        values coincide.

        Examples
        --------
        ::

            fluxes = result.line_fluxes()
            ha_median, ha_lo, ha_hi = fluxes["Halpha"]
        """
        if self.eline_fluxes is None:
            raise ValueError(
                "No emission line fluxes available. "
                "Set eline_mode='marginalized' or 'fitted' in Spectroscopy."
            )
        result = {}
        for i, name in enumerate(self.eline_names):
            if self.eline_fluxes.ndim == 1:
                # MAP: single estimate
                val = float(self.eline_fluxes[i])
                result[name] = (val, val, val)
            else:
                flux_i = self.eline_fluxes[:, i]
                lo, med, hi = jnp.percentile(flux_i, jnp.array([16.0, 50.0, 84.0]))
                result[name] = (float(med), float(lo), float(hi))
        return result

    def bpt_nii(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """BPT-NII ([NII]/Hα vs [OIII]/Hβ) diagram coordinates.

        Returns log10 line ratios for each posterior sample.

        Returns
        -------
        log_nii_ha : ndarray, shape (n_samples,) or scalar
            log10([NII]6584 / Hα). For MAP, returns scalar.
        log_oiii_hb : ndarray, shape (n_samples,) or scalar
            log10([OIII]5007 / Hβ). For MAP, returns scalar.

        Raises
        ------
        ValueError
            If emission line fluxes are not available or BPT lines are missing.

        Notes
        -----
        The BPT diagram is a standard AGN/SF diagnostic that uses the ratios
        [NII]/Hα (x-axis) and [OIII]/Hβ (y-axis). Non-detections (negative
        or zero fluxes) are returned as NaN and will not be plotted.
        Diagnostic lines follow Kewley et al. (2001, 2006) conventions.

        Examples
        --------
        ::

            x, y = result.bpt_nii()
            plt.scatter(x, y, alpha=0.3)
            # Overlay diagnostic lines from starburst/Seyfert boundaries
        """
        if self.eline_fluxes is None:
            raise ValueError("No emission line fluxes available.")
        names = list(self.eline_names)
        required = ["NII_6584", "Halpha", "OIII_5007", "Hbeta"]
        missing = [n for n in required if n not in names]
        if missing:
            raise ValueError(f"BPT lines not in catalog: {missing}")

        def _get(name):
            """Return emission line flux (scalar or array) for the given line name."""
            idx = names.index(name)
            if self.eline_fluxes.ndim == 1:
                return self.eline_fluxes[idx]
            return self.eline_fluxes[:, idx]

        nii = _get("NII_6584")
        ha = _get("Halpha")
        oiii = _get("OIII_5007")
        hb = _get("Hbeta")
        # NaN for non-detections (negative amplitudes); clamping to 1e-30 would
        # give log10(1e-30/F) ~ -30 and corrupt BPT diagrams.
        log_nii_ha = jnp.where(
            (nii > 0) & (ha > 0),
            jnp.log10(jnp.maximum(nii, 1e-30) / jnp.maximum(ha, 1e-30)),
            jnp.nan,
        )
        log_oiii_hb = jnp.where(
            (oiii > 0) & (hb > 0),
            jnp.log10(jnp.maximum(oiii, 1e-30) / jnp.maximum(hb, 1e-30)),
            jnp.nan,
        )
        return log_nii_ha, log_oiii_hb

    def bpt_class(self):
        r"""Classify each posterior draw as SF / composite / AGN on the BPT-NII diagram.

        Uses the standard demarcation lines:

        - **Kauffmann et al. 2003** [1]_ — separates pure SF from
          composite (SF + AGN admixture).
        - **Kewley et al. 2001** [2]_ — separates composite from
          pure AGN/Seyfert.

        Below Kauffmann ⇒ ``"SF"``; between Kauffmann and Kewley ⇒
        ``"composite"``; above Kewley ⇒ ``"AGN"``. Non-detections
        (NaN ratios) return ``"unknown"``.

        Returns
        -------
        str or ndarray of dtype <U9
            For MAP results: a single label string.
            For sampling results: a length-``n_samples`` array of labels.

        Raises
        ------
        ValueError
            If emission line fluxes are unavailable or BPT lines absent.

        Notes
        -----
        Kauffmann+2003 demarcation:

        .. math::

            \log_{10}\!\frac{[\mathrm{O\,III}]}{H\beta} =
            \frac{0.61}{\log_{10}([\mathrm{N\,II}]/H\alpha) - 0.05} + 1.30
            \quad (\text{for }\log [\mathrm{N\,II}]/H\alpha < 0.05)

        Kewley+2001 demarcation:

        .. math::

            \log_{10}\!\frac{[\mathrm{O\,III}]}{H\beta} =
            \frac{0.61}{\log_{10}([\mathrm{N\,II}]/H\alpha) - 0.47} + 1.19
            \quad (\text{for }\log [\mathrm{N\,II}]/H\alpha < 0.47)

        Points right of the asymptote (:math:`\log [\mathrm{N\,II}]/H\alpha
        \ge 0.47`) are classified as AGN regardless of [O III]/Hβ.

        References
        ----------
        .. [1] Kauffmann, G. et al., 2003, MNRAS, 346, 1055.
        .. [2] Kewley, L. J. et al., 2001, ApJ, 556, 121.

        Examples
        --------
        >>> labels = result.bpt_class()
        >>> import numpy as np
        >>> agn_frac = float(np.mean(np.asarray(labels) == "AGN"))
        """
        x, y = self.bpt_nii()  # log_nii_ha, log_oiii_hb
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        scalar_input = x_arr.ndim == 0
        x_arr = np.atleast_1d(x_arr)
        y_arr = np.atleast_1d(y_arr)

        # Demarcation curves; defined only for x < asymptote.
        with np.errstate(divide="ignore", invalid="ignore"):
            kauffmann = 0.61 / (x_arr - 0.05) + 1.30
            kewley = 0.61 / (x_arr - 0.47) + 1.19

        labels = np.full(x_arr.shape, "unknown", dtype="<U9")
        finite = np.isfinite(x_arr) & np.isfinite(y_arr)

        # AGN region: right of Kewley asymptote (x >= 0.47) OR above Kewley curve.
        agn = finite & ((x_arr >= 0.47) | (y_arr > kewley))
        # Composite: above Kauffmann but below Kewley (and left of Kewley asymptote).
        composite = (
            finite
            & ~agn
            & ((x_arr >= 0.05) | (y_arr > kauffmann))
        )
        # SF: everything finite that's not AGN or composite.
        sf = finite & ~agn & ~composite

        labels[sf] = "SF"
        labels[composite] = "composite"
        labels[agn] = "AGN"

        if scalar_input:
            return str(labels[0])
        return labels

    def balmer_decrement(self) -> tuple[float, float, float]:
        """Observed Hα/Hβ ratio from posterior line fluxes.

        Returns the posterior distribution of the Balmer decrement
        (Hα/Hβ), which is a direct dust attenuation diagnostic.
        The intrinsic Case B ratio is 2.86; higher values indicate dust.

        Returns
        -------
        tuple
            ``(median, lo_68, hi_68)`` of Hα/Hβ (dimensionless ratio).
            For MAP results, all three values are equal.

        Raises
        ------
        ValueError
            If Hα or Hβ fluxes are not available.

        Notes
        -----
        The Balmer decrement (Hα/Hβ flux ratio) is insensitive to stellar
        population age and metallicity; deviations from the intrinsic Case B
        value of 2.86 (Osterbrock 1989) directly indicate dust attenuation.
        A Balmer decrement > 3.0 typically indicates significant dust.

        Examples
        --------
        ::

            med, lo, hi = result.balmer_decrement()
            print(f"Ha/Hb = {med:.2f} [{lo:.2f}, {hi:.2f}]")
            # Intrinsic Case B = 2.86; excess indicates dust attenuation
        """
        if self.eline_fluxes is None:
            raise ValueError("No emission line fluxes available.")
        names = list(self.eline_names)
        for n in ["Halpha", "Hbeta"]:
            if n not in names:
                raise ValueError(f"Required line '{n}' not in catalog.")

        def _get(name):
            """Return emission line flux (scalar or array) for the given line name."""
            idx = names.index(name)
            if self.eline_fluxes.ndim == 1:
                return self.eline_fluxes[idx]
            return self.eline_fluxes[:, idx]

        eps = 1e-30
        ratio = jnp.maximum(_get("Halpha"), eps) / jnp.maximum(_get("Hbeta"), eps)
        if ratio.ndim == 0:
            v = float(ratio)
            return (v, v, v)
        lo, med, hi = jnp.percentile(ratio, jnp.array([16.0, 50.0, 84.0]))
        return (float(med), float(lo), float(hi))

    def balmer_av(self) -> tuple[float, float, float]:
        r"""Visual extinction A(V) from the Balmer decrement (Calzetti+2000).

        Converts the observed Hα/Hβ ratio to a V-band attenuation
        assuming Case B recombination (intrinsic ratio 2.86) and the
        Calzetti et al. 2000 [1]_ starburst attenuation law
        (:math:`R_V = 4.05`, :math:`k(H\alpha) = 2.53`, :math:`k(H\beta) = 3.61`).

        Returns
        -------
        tuple
            ``(median, lo_68, hi_68)`` of A(V) in [mag]. For MAP results
            all three values are equal. Negative values are returned
            as-is (unphysical, but informative for noisy data — clip
            externally if desired).

        Raises
        ------
        ValueError
            If Hα or Hβ fluxes are not available.

        Notes
        -----
        .. math::

            E(B-V) &= \frac{\log_{10}\!\left(R_{\rm obs}/2.86\right)}
            {0.4\,\left(k(H\beta) - k(H\alpha)\right)} \\
            A(V) &= R_V \cdot E(B-V)

        For Calzetti+2000:
        :math:`0.4 \cdot (k(H\beta) - k(H\alpha)) = 0.432`,
        :math:`R_V = 4.05`, so
        :math:`A(V) \approx 9.375 \, \log_{10}(R_{\rm obs}/2.86)`.

        References
        ----------
        .. [1] Calzetti, D. et al., 2000, ApJ, 533, 682.

        Examples
        --------
        >>> av_med, av_lo, av_hi = result.balmer_av()
        """
        med, lo, hi = self.balmer_decrement()
        # Calzetti+2000 conversion constants
        _K_HALPHA = 2.53
        _K_HBETA = 3.61
        _R_V = 4.05
        _DENOM = 0.4 * (_K_HBETA - _K_HALPHA)  # 0.432
        _CASE_B = 2.86

        def _to_av(r: float) -> float:
            return _R_V * float(np.log10(r / _CASE_B)) / _DENOM

        return _to_av(med), _to_av(lo), _to_av(hi)

    def equivalent_widths(
        self,
        window_aa: float = 20.0,
        continuum_width_aa: float = 50.0,
    ) -> dict[str, tuple[float, float, float]]:
        """Rest-frame emission line equivalent widths from posterior samples.

        For each emission line, predicts the rest-frame SED for each posterior
        sample (or the MAP point estimate) using the attached forward model
        and integrates the line flux relative to the local continuum estimated
        from sidebands flanking the line. Sign convention follows
        :func:`tengri.analysis.diagnostics.spectral.equivalent_width`:
        positive EW for emission, negative for absorption.

        Parameters
        ----------
        window_aa : float, optional
            Half-width of the line integration window [Angstrom]. Default 20.
        continuum_width_aa : float, optional
            Width of each sideband used to estimate the continuum [Angstrom].
            Sidebands sit at ``[lambda_0 +/- window +/- continuum_width]``.
            Default 50.

        Returns
        -------
        dict
            ``{line_name: (median_EW, lo_68, hi_68)}`` in [Angstrom]. For MAP
            results, all three values coincide.

        Raises
        ------
        ValueError
            If ``eline_fluxes`` / ``eline_wavelengths`` are unavailable, or if
            no ``_model`` is attached to compute the continuum.

        Notes
        -----
        The continuum is estimated locally per line from the model-predicted
        rest-frame SED, *not* from a precomputed continuum-only grid. The
        sideband choice therefore must avoid contamination from neighbouring
        emission lines; defaults are tuned for optical BPT lines.

        Examples
        --------
        ::

            ew = result.equivalent_widths()
            ha_med, ha_lo, ha_hi = ew["Halpha"]
        """
        if self.eline_fluxes is None or self.eline_wavelengths is None:
            raise ValueError(
                "No emission line fluxes available. "
                "Set eline_mode='marginalized' or 'fitted' in Spectroscopy."
            )
        if self._model is None:
            raise ValueError(
                "No forward model attached; cannot compute continuum for "
                "equivalent_widths(). Refit with the model accessible."
            )

        from tengri.analysis.diagnostics.spectral import equivalent_width

        names = list(self.eline_names)
        wavelengths = [float(w) for w in np.asarray(self.eline_wavelengths)]

        def _ew_for_params(p: dict) -> jnp.ndarray:
            sed = self._model.predict_rest_sed(p)
            return jnp.stack(
                [
                    equivalent_width(
                        sed.wavelength, sed.sed, lc, window_aa, continuum_width_aa
                    )
                    for lc in wavelengths
                ]
            )

        if self.samples is None:
            # MAP: one prediction.
            ew_arr = np.asarray(_ew_for_params(self.params))
            return {name: (float(ew_arr[i]),) * 3 for i, name in enumerate(names)}

        # Sampling: predict per draw, then take percentiles over draws.
        keys = [k for k, v in self.samples.items() if v.ndim >= 1]
        if not keys:
            ew_arr = np.asarray(_ew_for_params(self.params))
            return {name: (float(ew_arr[i]),) * 3 for i, name in enumerate(names)}

        n = int(self.samples[keys[0]].shape[0])
        ew_samples = np.empty((n, len(names)))
        for i in range(n):
            params_i = {k: (v[i] if v.ndim >= 1 else v) for k, v in self.samples.items()}
            ew_samples[i] = np.asarray(_ew_for_params(params_i))

        result: dict[str, tuple[float, float, float]] = {}
        for j, name in enumerate(names):
            lo, med, hi = np.percentile(ew_samples[:, j], [16.0, 50.0, 84.0])
            result[name] = (float(med), float(lo), float(hi))
        return result

    # ── Summary statistics ────────────────────────────────────────

    def stats(self) -> dict:
        """Median and 68% credible intervals for all parameters.

        Returns
        -------
        dict
            Keys: parameter names (excluding ``"psd_xi"`` latent field).
            Values: dict with ``"median"``, ``"lo_68"``, ``"hi_68"``
            for sampling methods, or ``"value"`` for MAP.

        Notes
        -----
        For MCMC and VI results, credible intervals are 16th and 84th percentiles.
        For MAP results, returns point estimates without intervals.
        Does not include high-dimensional latent fields (``psd_xi``).

        Examples
        --------
        >>> stats = result.stats()
        >>> print(stats["stellar_mass"])
        {"median": 10.5, "lo_68": 10.3, "hi_68": 10.7}  # sampling
        # or
        {"value": 10.5}  # MAP
        """
        result = {}

        if self.samples is None:
            # MAP: point estimates
            for name, val in self.params.items():
                if name == "psd_xi":
                    continue
                result[name] = {"value": float(jnp.mean(val))}
        else:
            # Sampling: percentiles
            for name, arr in self.samples.items():
                if name == "psd_xi":
                    continue
                if arr.ndim == 1:
                    vals = np.array(arr)
                    result[name] = {
                        "median": float(np.median(vals)),
                        "lo_68": float(np.percentile(vals, 16)),
                        "hi_68": float(np.percentile(vals, 84)),
                    }

        return result

    def summary_table(self) -> str:
        """Return a formatted string table of parameter summaries.

        For MAP: shows parameter values.
        For sampling: shows median with 68% credible intervals and ESS.

        Returns
        -------
        str
            Formatted table string with method, sample count, wall time,
            parameter statistics, and diagnostics.

        Notes
        -----
        The table includes:
        - Method name and number of samples (or ``"MAP"``)
        - Wall-clock time in seconds
        - Parameter names with median and credible intervals
        - Effective sample size (ESS) for sampling methods
        - Method-specific diagnostics (accept rate, divergences, loss, etc.)
        - Log evidence (if available from nested sampling)

        Examples
        --------
        >>> print(result.summary_table())
        Posterior  method: mcmc_nuts  samples: 1000  wall_time: 5.2s
        ─────────────────────────────────────────────────────────────
          Parameter                   Median        16%        84%     ESS
          ─────────────────────────────────────────────────────────────
          ...
        """
        sep = "─" * 66
        lines: list[str] = []
        n_samples = "MAP" if self.samples is None else next(iter(self.samples.values())).shape[0]
        lines.append(
            f"Posterior  method: {self.method}  "
            f"samples: {n_samples}  "
            f"wall_time: {self.wall_time_s:.1f}s"
        )
        lines.append(sep)

        stats = self.stats()
        if not stats:
            lines.append("  (no parameters)")
            lines.append(sep)
            return "\n".join(lines)

        if self.samples is None:
            # MAP table
            hdr = f"  {'Parameter':<32s} {'Value':>12s}"
            lines.append(hdr)
            lines.append("  " + "─" * 44)
            for name in sorted(stats):
                val = stats[name]["value"]
                lines.append(f"  {name:<32s} {val:>12.4f}")
        else:
            # Sampling table with credible intervals
            # Try to get ESS if available
            ess = self.diagnostics.get("ess_bulk", {})
            hdr = f"  {'Parameter':<28s} {'Median':>9s} {'16%':>9s} {'84%':>9s} {'ESS':>7s}"
            lines.append(hdr)
            lines.append("  " + "─" * 64)
            for name in sorted(stats):
                s = stats[name]
                med = f"{s['median']:.4f}"
                lo = f"{s['lo_68']:.4f}"
                hi = f"{s['hi_68']:.4f}"
                ess_val = ess.get(name)
                ess_str = f"{ess_val:.0f}" if ess_val is not None else "—"
                lines.append(f"  {name:<28s} {med:>9s} {lo:>9s} {hi:>9s} {ess_str:>7s}")

        # Diagnostics summary
        diag = self.diagnostics
        diag_parts: list[str] = []
        if "accept_rate" in diag:  # key from raytrace (fitter.py:3509)
            diag_parts.append(f"accept={diag['accept_rate']:.1%}")
        if "n_divergent" in diag:  # key from NUTS (nuts.py:193)
            diag_parts.append(f"divergences={diag['n_divergent']}")
        if "final_loss" in diag:
            diag_parts.append(f"loss={diag['final_loss']:.2f}")
        if diag_parts:
            lines.append("")
            lines.append(f"  Diagnostics: {', '.join(diag_parts)}")

        if self.log_evidence is not None:
            err = self.diagnostics.get("log_evidence_err")
            if err is not None:
                lines.append(f"  log Z (evidence) = {self.log_evidence:.2f} ± {err:.2f}")
            else:
                lines.append(f"  log Z (evidence) = {self.log_evidence:.2f}")

        lines.append(sep)
        return "\n".join(lines)

    # ── Autocorrelation and effective sample size ─────────────────

    @staticmethod
    def _autocorrelation_1d(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
        """Compute normalized autocorrelation function for a 1D chain.

        Uses FFT for efficiency (O(N log N) instead of O(N * max_lag)).
        """
        n = len(x)
        if max_lag is None:
            max_lag = n // 2
        max_lag = min(max_lag, n - 1)

        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-30:
            return np.zeros(max_lag + 1)

        # FFT-based autocorrelation
        fft_size = 2 ** int(np.ceil(np.log2(2 * n)))
        fft_x = np.fft.rfft(x, n=fft_size)
        acf_full = np.fft.irfft(fft_x * np.conj(fft_x))
        acf = acf_full[: max_lag + 1] / (var * n)
        return acf

    def autocorrelation(self, max_lag: int | None = None) -> dict:
        """Compute autocorrelation function for each scalar parameter.

        Parameters
        ----------
        max_lag : int, optional
            Maximum lag. Default: n_samples // 2.

        Returns
        -------
        dict
            Keys: parameter names.
            Values: ndarray of autocorrelation from lag 0 to max_lag.
            ACF[0] = 1.0 by definition.

        Notes
        -----
        Uses FFT for O(N log N) efficiency instead of naive O(N * max_lag).
        Normalized so that ACF[0] = 1 and ACF[k] ∈ [-1, 1].
        Parameters with zero variance (fixed parameters) return zero ACF.

        Examples
        --------
        >>> acf = result.autocorrelation()
        >>> lag_cutoff = np.argmax(acf["stellar_mass"] < 0.05)
        >>> print(f"ACF drops below 0.05 at lag {lag_cutoff}")
        """
        if self.samples is None:
            raise ValueError("Autocorrelation requires samples (not MAP)")

        result = {}
        for name, arr in self.samples.items():
            if name == "psd_xi":
                continue
            if arr.ndim == 1:
                result[name] = self._autocorrelation_1d(np.array(arr), max_lag)
        return result

    def effective_sample_size(self) -> dict:
        """Estimate effective sample size (ESS) for each parameter.

        Uses Sokal's self-consistent window method (Behroozi 2025):
        τ = 1 + 2 Σ ρ(k), truncated at k > 5τ. Takes the max of
        standard and absolute-deviation autocorrelation times for
        a conservative estimate.

        Returns
        -------
        dict
            Keys: parameter names. Values: ESS (float).
            ESS = N / τ, where N is the total number of samples
            and τ is the integrated autocorrelation time.

        Notes
        -----
        ESS measures the number of independent samples. Low ESS
        (< 100 for typical analyses) indicates poor mixing.
        The threshold N > 5τ (equivalently ESS > N/5) indicates
        adequate sampling for most purposes.

        Examples
        --------
        >>> ess = result.effective_sample_size()
        >>> print(f"Stellar mass ESS: {ess['stellar_mass']:.0f}")
        >>> if ess["stellar_mass"] < 100:
        ...     print("Warning: low ESS, may need more samples")
        """
        if self.samples is None:
            raise ValueError("ESS requires samples (not MAP)")

        from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

        ess_info = effective_sample_size({k: np.asarray(v) for k, v in self.samples.items()})
        return {name: info["ess"] for name, info in ess_info.items()}

    def autocorrelation_time(self) -> dict:
        """Estimate integrated autocorrelation time for each parameter.

        Uses Sokal's self-consistent window method with both standard
        and absolute-deviation modes (Behroozi 2025).

        Returns
        -------
        dict
            Keys: parameter names.
            Values: dict with ``'tau_standard'``, ``'tau_absolute'``, ``'tau_max'``
            (integrated autocorrelation time), ``'ess'`` (effective sample size),
            ``'chain_converged'`` (bool, True if N > 5τ_max).

        Notes
        -----
        Two autocorrelation time estimates are computed:
        - tau_standard: based on standard ACF
        - tau_absolute: based on absolute-deviation ACF (robust to mean/variance changes)
        The maximum is returned (conservative). ESS = N / tau_max.
        Convergence flag uses the criterion N > 5τ_max from Behroozi (2025).

        Examples
        --------
        >>> tau_dict = result.autocorrelation_time()
        >>> for param, info in tau_dict.items():
        ...     converged = info["chain_converged"]
        ...     print(
        ...         f"{param}: tau_max={info['tau_max']:.1f}, "
        ...         f"ESS={info['ess']:.0f}, converged={converged}"
        ...     )
        """
        if self.samples is None:
            raise ValueError("Autocorrelation time requires samples (not MAP)")

        from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

        return effective_sample_size({k: np.asarray(v) for k, v in self.samples.items()})

    def check_convergence(self, verbose: bool = True) -> dict:
        """Check chain convergence using autocorrelation diagnostics.

        Follows Behroozi (2025): chain is converged when N > 5τ for
        all parameters.

        Parameters
        ----------
        verbose : bool
            Print diagnostics table.

        Returns
        -------
        dict
            Keys: ``'all_converged'`` (bool), ``'params'`` (dict per-parameter
            convergence info), ``'warnings'`` (list of unconverged parameters).

        Notes
        -----
        Convergence criterion: N > 5τ_max for all parameters.
        If N < 5τ for any parameter, the chain is flagged as unconverged.
        See Behroozi (2025) for justification of the 5τ threshold.

        Examples
        --------
        >>> conv = result.check_convergence()
        >>> if conv["all_converged"]:
        ...     print("Chain converged!")
        ... else:
        ...     print(f"Unconverged: {conv['warnings']}")
        ...     print("Run additional samples and use refine()")
        """
        if self.samples is None:
            raise ValueError("Convergence check requires samples (not MAP)")

        from tengri.analysis.diagnostics.autocorrelation import check_chain_length

        return check_chain_length(
            {k: np.asarray(v) for k, v in self.samples.items()},
            verbose=verbose,
        )

    def posterior_predictive(
        self,
        data: jnp.ndarray,
        noise: jnp.ndarray,
        n_samples: int | None = None,
        key=None,
    ) -> dict[str, jnp.ndarray]:
        r"""Posterior predictive predictions, residuals, and chi^2 distribution.

        Pushes posterior draws (or the MAP point estimate) through the
        attached forward model's ``predict_photometry`` and reports
        per-draw predictions, standardised residuals, and a chi^2
        distribution against the supplied data + noise.

        Parameters
        ----------
        data : array_like, shape (n_obs,)
            Observed data the posterior was conditioned on. Same units
            as ``predict_photometry``'s output.
        noise : array_like, shape (n_obs,)
            Per-observation 1-sigma uncertainty (Gaussian).
        n_samples : int, optional
            How many posterior draws to evaluate. ``None`` (default)
            uses every available draw; for MAP results this is
            implicitly 1. For sampling results, draws are selected
            via :meth:`resample` (with replacement) using ``key``.
        key : PRNGKey, optional
            JAX PRNG key for resampling. If ``None``, defaults to
            ``jax.random.PRNGKey(0)``.

        Returns
        -------
        dict
            ``predictions``: shape ``(N, n_obs)``,
            ``residuals``: ``(data - prediction) / noise`` of shape
            ``(N, n_obs)``,
            ``chi2``: per-draw :math:`\chi^2 = \sum_i r_i^2`, shape
            ``(N,)``,
            ``chi2_median``, ``chi2_lo``, ``chi2_hi``: 16/50/84
            percentiles of the chi^2 distribution (scalars).
            ``N`` is 1 for MAP, otherwise ``n_samples`` (or the full
            chain length if ``n_samples is None``).

        Raises
        ------
        ValueError
            If no ``_model`` is attached.

        Notes
        -----
        This is a deterministic posterior predictive (no extra noise
        realisation per draw). For replicated PPCs that draw observation
        noise per sample, layer ``noise * jax.random.normal`` on top of
        ``predictions``.
        """
        if self._model is None:
            raise ValueError(
                "No forward model attached; cannot compute "
                "posterior_predictive(). Refit with the model accessible."
            )
        data_arr = jnp.asarray(data)
        noise_arr = jnp.asarray(noise)

        def _predict_one(p: dict) -> jnp.ndarray:
            return jnp.asarray(self._model.predict_photometry(p))

        if self.samples is None:
            preds = _predict_one(self.params)[None, :]
        else:
            sample_keys = [k for k, v in self.samples.items() if v.ndim >= 1]
            n_total = (
                int(self.samples[sample_keys[0]].shape[0]) if sample_keys else 0
            )
            if not sample_keys or n_total == 0:
                preds = _predict_one(self.params)[None, :]
            else:
                if n_samples is None or n_samples >= n_total:
                    indices = np.arange(n_total)
                else:
                    if key is None:
                        key = jax.random.PRNGKey(0)
                    indices = np.asarray(
                        jax.random.choice(
                            key, n_total, shape=(int(n_samples),), replace=True
                        )
                    )
                preds_list = []
                for idx in indices:
                    params_i = {
                        k: (v[idx] if v.ndim >= 1 else v)
                        for k, v in self.samples.items()
                    }
                    preds_list.append(_predict_one(params_i))
                preds = jnp.stack(preds_list, axis=0)

        residuals = (data_arr[None, :] - preds) / noise_arr[None, :]
        chi2 = jnp.sum(residuals**2, axis=1)
        chi2_np = np.asarray(chi2)
        if chi2_np.shape[0] >= 1:
            lo, med, hi = np.percentile(chi2_np, [16.0, 50.0, 84.0])
        else:
            lo = med = hi = float("nan")
        return {
            "predictions": preds,
            "residuals": residuals,
            "chi2": chi2,
            "chi2_median": float(med),
            "chi2_lo": float(lo),
            "chi2_hi": float(hi),
        }

    def rhat(self, exclude_prefixes: tuple[str, ...] = ("psd_xi",)) -> dict[str, float]:
        r"""Per-parameter split-:math:`\hat R` (Gelman-Rubin).

        Splits each parameter chain in half and computes the classical
        :math:`\hat R = \sqrt{\hat V / W}` against the two halves.
        :math:`\hat R \approx 1.0` indicates convergence;
        :math:`\hat R > 1.01` (Vehtari+2021) suggests failure to mix.

        Parameters
        ----------
        exclude_prefixes : tuple of str, optional
            Parameter name prefixes to skip. Default skips ``psd_xi``
            (GP latent vector — high-D, not informative per-component).

        Returns
        -------
        dict
            Parameter name → :math:`\hat R`. Static (zero-variance) and
            excluded parameters are dropped.

        Raises
        ------
        ValueError
            If this is a MAP result (no samples to split).

        See Also
        --------
        tengri.analysis.diagnostics.autocorrelation.split_rhat
            Underlying implementation.

        Examples
        --------
        >>> rh = result.rhat()
        >>> bad = {k: v for k, v in rh.items() if v > 1.05}
        >>> if bad:
        ...     print(f"Unconverged: {bad}")
        """
        if self.samples is None:
            raise ValueError("R-hat requires samples (not a MAP result).")
        from tengri.analysis.diagnostics.autocorrelation import rhat as _rhat

        return _rhat(
            {k: np.asarray(v) for k, v in self.samples.items()},
            exclude_prefixes=exclude_prefixes,
        )

    def diagnostics_summary(self) -> str:
        """Print a diagnostics summary with ESS and credible intervals.

        Returns
        -------
        str
            Formatted table of per-parameter ESS and 68% credible intervals.

        Notes
        -----
        For MAP results, returns a simple string indicating no samples are available.
        For sampling methods, tabulates ESS for each parameter alongside median
        and 68% credible intervals. Use :meth:`rhat` for the split-chain
        Gelman-Rubin diagnostic.

        Examples
        --------
        >>> print(result.diagnostics_summary())
        Method: mcmc_nuts
        Samples: 1000
        Wall time: 5.2s

        Parameter                  Median           68% CI          ESS
        ────────────────────────────────────────────────────────────────
        stellar_mass               10.500  [10.300, 10.700]        800
        ...
        """
        if self.samples is None:
            return f"MAP result (no samples): method={self.method}"

        ess = self.effective_sample_size()
        summary = self.stats()

        lines = [
            f"Method: {self.method}",
            f"Samples: {next(iter(self.samples.values())).shape[0]}",
            f"Wall time: {self.wall_time_s:.1f}s",
            "",
            f"{'Parameter':<22s} {'Median':>8s} {'68% CI':>18s} {'ESS':>6s}",
            "-" * 58,
        ]
        for name in sorted(summary.keys()):
            s = summary[name]
            if "median" in s:
                e = ess.get(name, float("nan"))
                ci = f"[{s['lo_68']:.3f}, {s['hi_68']:.3f}]"
                lines.append(f"{name:<22s} {s['median']:>8.3f} {ci:>18s} {e:>6.0f}")

        return "\n".join(lines)

    # ── Resampling ────────────────────────────────────────────────

    def resample(self, key, n=1) -> dict:
        """Resample from posterior with replacement.

        Parameters
        ----------
        key : PRNGKey
            JAX random key.
        n : int
            Number of resamples.

        Returns
        -------
        dict
            If n=1: parameter name → scalar value.
            If n>1: parameter name → array of shape (n, ...).

        Notes
        -----
        For MAP results, returns the point estimate (repeated n times if n > 1).
        For sampling results, draws n indices uniformly from [0, n_samples) with
        replacement. Use for Monte Carlo propagation of posterior uncertainty
        through forward models.

        Examples
        --------
        >>> key = jax.random.PRNGKey(0)
        >>> sample = result.resample(key, n=1)  # Single draw
        >>> samples = result.resample(key, n=100)  # 100 resamples
        >>> sfhs = [model.predict_sfh(samples) for ...]  # Propagate
        """
        if self.samples is None:
            # MAP: just return the point estimate
            if n == 1:
                return dict(self.params)
            return {k: jnp.broadcast_to(v, (n, *v.shape)) for k, v in self.params.items()}

        n_available = next(iter(self.samples.values())).shape[0]
        indices = jax.random.choice(key, n_available, shape=(n,), replace=True)

        if n == 1:
            idx = indices[0]
            return {k: v[idx] for k, v in self.samples.items()}

        return {k: v[indices] for k, v in self.samples.items()}

    # ── Conversion ────────────────────────────────────────────────

    def to_param_spec(self):
        """Convert posterior to an empirical Parameters.

        For MAP: all parameters become Fixed at their best-fit values.
        For sampling: fit clipped Gaussian to each marginal.

        Returns
        -------
        Parameters
            New Parameters object with priors fit to the posterior.

        Notes
        -----
        For MAP results, all parameters become ``Fixed`` at the MAP value.
        For sampling methods, each parameter gets a ``Gaussian`` prior with:
        - mean: median of samples
        - sigma: standard deviation of samples
        - bounds: [min, max] from samples (clipping)
        Inherits ``stochastic`` and ``n_grid`` settings from the original model.

        Examples
        --------
        >>> posterior_params = result.to_param_spec()
        >>> # Use as starting point for next fit
        >>> refined = fitter.run("mcmc_nuts", init_from=posterior_params)
        """
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed, Gaussian

        kwargs = {}

        if self.samples is None:
            # MAP: all Fixed
            for name, val in self.params.items():
                if name == "psd_xi":
                    continue
                kwargs[name] = Fixed(float(jnp.mean(val)))
        else:
            # Sampling: fit Gaussian to each marginal
            for name, arr in self.samples.items():
                if name == "psd_xi":
                    continue
                if arr.ndim == 1:
                    vals = np.array(arr)
                    kwargs[name] = Gaussian(
                        mu=float(np.median(vals)),
                        sigma=float(np.std(vals)),
                        lo=float(np.min(vals)),
                        hi=float(np.max(vals)),
                    )

        # Copy settings from original spec
        if self._model is not None:
            kwargs["stochastic"] = self._model.spec.stochastic
            kwargs["n_grid"] = self._model.spec.n_grid

        return Parameters(**kwargs)

    def to_arviz(self):
        """Convert to ArviZ InferenceData for diagnostics.

        Returns
        -------
        az.InferenceData
            ArviZ InferenceData object with posterior group containing
            all scalar parameters.

        Notes
        -----
        Requires arviz to be installed: ``pip install arviz``.
        High-dimensional latent fields (``psd_xi``) are excluded.
        Samples are reshaped to (1, n_samples) format (1 chain).
        Use ArviZ tools for advanced visualization and diagnostics
        (forest plots, rank plots, etc.).

        Examples
        --------
        >>> idata = result.to_arviz()
        >>> az.plot_forest(idata)
        >>> az.summary(idata)
        """
        try:
            import arviz as az
        except ImportError:
            raise ImportError("arviz required: pip install arviz") from None

        if self.samples is None:
            raise ValueError("Cannot convert MAP result to ArviZ (no samples)")

        posterior = {}
        for name, arr in self.samples.items():
            if name == "psd_xi":
                continue  # skip high-dimensional latent
            arr_np = np.array(arr)
            if arr_np.ndim == 1:
                # Add chain dimension: (1, n_samples)
                posterior[name] = arr_np[np.newaxis, :]

        return az.from_dict(posterior=posterior)

    # ── HDF5 serialization ────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save posterior to HDF5 file.

        Serializes samples, params, diagnostics, loss history, and
        emission line data. Model and fitter references are NOT saved
        (they are non-serializable runtime objects).

        Parameters
        ----------
        path : str
            Output HDF5 file path.

        Returns
        -------
        None

        Notes
        -----
        Saves to HDF5 format with groups:
        - ``samples``: posterior samples (if available)
        - ``params``: best-fit or MAP parameters
        - ``loss_history``: optimization loss over iterations (if available)
        - ``diagnostics``: method-specific convergence metrics
        - ``eline``: emission line fluxes, covariances, names, wavelengths
        Use ``load()`` to restore the Posterior from disk.

        Examples
        --------
        >>> result.save("posterior_result.h5")
        >>> later_result = Posterior.load("posterior_result.h5")
        """
        import h5py

        with h5py.File(path, "w") as f:
            f.attrs["method"] = self.method
            f.attrs["wall_time_s"] = self.wall_time_s
            if self.log_evidence is not None:
                f.attrs["log_evidence"] = self.log_evidence

            if self.samples is not None:
                grp = f.create_group("samples")
                for name, arr in self.samples.items():
                    grp.create_dataset(name, data=np.asarray(arr))

            grp = f.create_group("params")
            for name, val in self.params.items():
                grp.create_dataset(name, data=np.asarray(val))

            if self.loss_history is not None:
                f.create_dataset("loss_history", data=np.asarray(self.loss_history))

            self._save_diagnostics(f, self.diagnostics)

            if self.eline_fluxes is not None:
                eline = f.create_group("eline")
                eline.create_dataset("fluxes", data=np.asarray(self.eline_fluxes))
                if self.eline_flux_cov is not None:
                    eline.create_dataset("flux_cov", data=np.asarray(self.eline_flux_cov))
                if self.eline_names is not None:
                    eline.attrs["names"] = list(self.eline_names)
                if self.eline_wavelengths is not None:
                    eline.create_dataset("wavelengths", data=np.asarray(self.eline_wavelengths))

    @staticmethod
    def _save_diagnostics(f, diagnostics: dict) -> None:
        """Save diagnostics dict to HDF5, handling nested dicts and mixed types."""
        grp = f.create_group("diagnostics")
        for key, val in diagnostics.items():
            if isinstance(val, dict):
                sub = grp.create_group(key)
                for k2, v2 in val.items():
                    if isinstance(v2, (int, float, np.integer, np.floating)):
                        sub.attrs[k2] = float(v2)
                    elif isinstance(v2, str):
                        sub.attrs[k2] = v2
            elif isinstance(val, (int, float, np.integer, np.floating)):
                grp.attrs[key] = float(val)
            elif isinstance(val, str):
                grp.attrs[key] = val
            elif isinstance(val, (np.ndarray, jnp.ndarray)):
                grp.create_dataset(key, data=np.asarray(val))
            elif isinstance(val, (list, tuple)):
                try:
                    grp.create_dataset(key, data=np.asarray(val))
                except (TypeError, ValueError):
                    grp.attrs[key] = str(val)

    @classmethod
    def load(cls, path: str, model=None) -> Posterior:
        """Load a Posterior from an HDF5 file.

        Parameters
        ----------
        path : str
            Path to HDF5 file saved by :meth:`save`.
        model : SEDModel, optional
            Model reference for derived quantity computation.
            If provided, enables ``derived``, ``plot_sed()``, ``plot_sfh()``.

        Returns
        -------
        Posterior
            Loaded posterior with all attributes restored.

        Notes
        -----
        Reads HDF5 file saved by ``save()``. The ``_fitter`` back-reference
        is not restored (it is runtime-only). To use ``refine()`` or ``validate()``,
        set the ``_fitter`` attribute manually or reload using ``model.fit(...)``.
        Provide ``model`` to enable derived quantity computation.

        Examples
        --------
        >>> result = Posterior.load("posterior_result.h5", model=model)
        >>> print(result.summary_table())
        >>> derived = result.derived  # If model is provided
        """
        import h5py

        with h5py.File(path, "r") as f:
            method = str(f.attrs["method"])
            wall_time_s = float(f.attrs["wall_time_s"])
            log_evidence = float(f.attrs["log_evidence"]) if "log_evidence" in f.attrs else None

            def _read_ds(ds):
                """Load an HDF5 dataset into a JAX array, handling both scalar and array shapes."""
                return jnp.asarray(ds[()]) if ds.shape == () else jnp.asarray(ds[:])

            samples = None
            if "samples" in f:
                samples = {name: _read_ds(ds) for name, ds in f["samples"].items()}

            params = {name: _read_ds(ds) for name, ds in f["params"].items()}

            loss_history = None
            if "loss_history" in f:
                loss_history = jnp.asarray(f["loss_history"][:])

            diagnostics = cls._load_diagnostics(f)

            eline_fluxes = None
            eline_flux_cov = None
            eline_names = None
            eline_wavelengths = None
            if "eline" in f:
                eg = f["eline"]
                eline_fluxes = jnp.asarray(eg["fluxes"][:])
                if "flux_cov" in eg:
                    eline_flux_cov = jnp.asarray(eg["flux_cov"][:])
                if "names" in eg.attrs:
                    eline_names = tuple(eg.attrs["names"])
                if "wavelengths" in eg:
                    eline_wavelengths = jnp.asarray(eg["wavelengths"][:])

        return cls(
            samples=samples,
            params=params,
            method=method,
            wall_time_s=wall_time_s,
            diagnostics=diagnostics,
            loss_history=loss_history,
            log_evidence=log_evidence,
            _model=model,
            eline_fluxes=eline_fluxes,
            eline_flux_cov=eline_flux_cov,
            eline_names=eline_names,
            eline_wavelengths=eline_wavelengths,
        )

    @staticmethod
    def _load_diagnostics(f) -> dict:
        """Load diagnostics dict from HDF5."""
        diagnostics: dict = {}
        if "diagnostics" not in f:
            return diagnostics
        grp = f["diagnostics"]
        for key in grp.attrs:
            diagnostics[key] = grp.attrs[key]
            if isinstance(diagnostics[key], bytes):
                diagnostics[key] = diagnostics[key].decode()
        for key in grp:
            item = grp[key]
            if hasattr(item, "shape"):
                diagnostics[key] = np.asarray(item[:])
            else:
                sub = {}
                for k2 in item.attrs:
                    sub[k2] = item.attrs[k2]
                    if isinstance(sub[k2], bytes):
                        sub[k2] = sub[k2].decode()
                diagnostics[key] = sub
        return diagnostics

    # ── Plotting ──────────────────────────────────────────────────

    def plot_corner(
        self, params=None, truths=None, figsize=None, color="C0", fig=None, axes=None, label=None
    ):
        """Plot corner (triangle) plot of posterior distributions.

        Parameters
        ----------
        params : list of str, optional
            Parameter names to include. Defaults to all scalar physical params.
            Automatically excludes ``psd_xi`` (latent field) and constant parameters.
        truths : dict, optional
            True values to mark with dashed lines. Keys should match parameter names.
        figsize : tuple, optional
            Figure size (width, height). Default: auto-scaled.
        color : str
            Color for this posterior's contours and histograms.
        fig, axes : matplotlib Figure, ndarray of Axes, optional
            If provided, overlay on existing corner plot (for comparing posteriors).
        label : str, optional
            Legend label for this posterior (appears in legend on diagonal).

        Returns
        -------
        fig : matplotlib Figure
            The corner plot figure.

        Notes
        -----
        Creates an N×N triangle plot (lower triangle only). Diagonal shows
        1D marginal distributions with KDE overlay and quantile lines.
        Off-diagonal shows 2D histograms with credible region contours at
        68% and 95%. Includes derived quantities (stellar_mass, sfr_100myr,
        sfr_10myr) if ``_model`` is available. Derived quantities are
        plotted in log10 space.

        Examples
        --------
        >>> fig = result.plot_corner(color="C0", label="VI")
        >>> fig = result_mcmc.plot_corner(fig=fig, axes=fig.axes, color="C1", label="MCMC")
        >>> plt.show()
        """
        import matplotlib.pyplot as plt

        if self.samples is None:
            raise ValueError("Corner plot requires posterior samples (not MAP)")

        if params is None:
            # Exclude psd_xi (high-dimensional) and fixed params (constant values)
            params = []
            for k in sorted(self.samples.keys()):
                if k == "psd_xi":
                    continue
                arr = self.samples[k]
                if arr.ndim != 1:
                    continue
                # Skip if all values are identical (fixed parameter)
                if float(jnp.std(arr)) < 1e-10:
                    continue
                params.append(k)

        # Add derived quantities if model is available
        derived = {}
        derived_truths = {}
        if self._model is not None:
            try:
                d = self.derived
                for k in ["stellar_mass", "sfr_100myr", "sfr_10myr"]:
                    if k in d:
                        derived[k] = np.array(d[k])
            except (AttributeError, TypeError, ValueError) as exc:
                logger.debug("derived quantity computation failed: %s", exc)
            # Compute truth derived quantities for truth lines
            if truths is not None:
                try:
                    d_true = self._model.predict_derived(truths)
                    for k in derived:
                        if k in d_true:
                            derived_truths[k] = float(d_true[k])
                except (AttributeError, TypeError, ValueError) as exc:
                    logger.debug("derived truth computation failed: %s", exc)

        n = len(params) + len(derived)
        if fig is None or axes is None:
            if figsize is None:
                figsize = (min(2.0 * n, 14), min(2.0 * n, 14))
            fig, axes = plt.subplots(n, n, figsize=figsize)
        if n == 1:
            axes = np.array([[axes]])
        # If axes is a flat list (e.g., from fig.axes), reshape to 2D
        if isinstance(axes, list):
            axes = np.array(axes).reshape(n, n)

        all_names = list(params) + list(derived.keys())
        all_data = {}
        for name in params:
            all_data[name] = np.array(self.samples[name])
        all_data.update(derived)

        # Labels
        label_map = {
            "sfh_alpha": r"$\alpha$",
            "sfh_beta": r"$\beta$",
            "sfh_tau_peak_gyr": r"$\tau_{\rm peak}$ (Gyr)",
            "sfh_peak_sfr": r"SFR$_{\rm peak}$",
            "psd_sigma": r"$\sigma_{\rm burst}$",
            "psd_tau_myr": r"$\tau_{\rm burst}$ (Myr)",
            "met_logzsol": r"log Z",
            "dust_tau_bc": r"$\tau_{\rm bc}$",
            "dust_tau_diff": r"$\tau_{\rm diff}$",
            "stellar_mass": r"$\log\,M_*$",
            "sfr_100myr": r"$\log\,$SFR$_{100}$",
            "sfr_10myr": r"$\log\,$SFR$_{10}$",
        }

        # Merge derived truths into truths dict for truth-line plotting
        all_truths = dict(truths) if truths else {}
        all_truths.update(derived_truths)

        for i, name_i in enumerate(all_names):
            xi = all_data[name_i]
            _log_derived = ("stellar_mass", "sfr_100myr", "sfr_10myr")
            if name_i in _log_derived:
                xi = np.log10(np.maximum(xi, 1e-30))

            for j, name_j in enumerate(all_names):
                ax = axes[i, j]
                xj = all_data[name_j]
                if name_j in _log_derived:
                    xj = np.log10(np.maximum(xj, 1e-30))

                if j > i:
                    ax.set_visible(False)
                    continue

                if i == j:
                    # Diagonal: 1D histogram + KDE + quantile title
                    # (following corner.py conventions)
                    n_bins = min(20, max(5, len(xi) // 3))
                    ax.hist(
                        xi,
                        bins=n_bins,
                        color=color,
                        alpha=0.2,
                        density=True,
                        edgecolor="white",
                        lw=0.5,
                    )
                    try:
                        from scipy.stats import gaussian_kde

                        kde = gaussian_kde(xi)
                        x_grid = np.linspace(np.min(xi), np.max(xi), 200)
                        lbl = label if (i == 0 and label) else None
                        ax.plot(x_grid, kde(x_grid), color=color, lw=1.5, label=lbl)
                    except (ImportError, np.linalg.LinAlgError):
                        pass  # fall back to histogram only
                    # Quantile lines + title (corner.py style)
                    q16, q50, q84 = np.percentile(xi, [16, 50, 84])
                    ax.axvline(q50, color=color, ls="-", lw=1.0, alpha=0.6)
                    ax.axvline(q16, color=color, ls=":", lw=0.7, alpha=0.4)
                    ax.axvline(q84, color=color, ls=":", lw=0.7, alpha=0.4)
                    if all_truths and name_i in all_truths:
                        tv = all_truths[name_i]
                        if name_i in ("stellar_mass", "sfr_100myr", "sfr_10myr"):
                            tv = np.log10(max(float(tv), 1e-30))
                        ax.axvline(tv, color="#4682b4", ls="--", lw=1.5)
                else:
                    # Off-diagonal: 2D histogram contours (corner.py style)
                    # hist2d is more robust than KDE at low ESS
                    try:
                        from scipy.ndimage import gaussian_filter

                        n_bins_2d = min(30, max(10, int(np.sqrt(len(xj)))))
                        H, xe, ye = np.histogram2d(
                            xj,
                            xi,
                            bins=n_bins_2d,
                            range=[[np.min(xj), np.max(xj)], [np.min(xi), np.max(xi)]],
                        )
                        H = gaussian_filter(H, sigma=1.0)
                        # Contour levels at 68% and 95% credible regions
                        H_sorted = np.sort(H.ravel())[::-1]
                        H_cumsum = np.cumsum(H_sorted) / np.sum(H_sorted)
                        level_68 = H_sorted[np.searchsorted(H_cumsum, 0.68)]
                        level_95 = H_sorted[np.searchsorted(H_cumsum, 0.95)]
                        xc = 0.5 * (xe[:-1] + xe[1:])
                        yc = 0.5 * (ye[:-1] + ye[1:])
                        X, Y = np.meshgrid(xc, yc)
                        # Guard: contourf needs strictly increasing levels
                        levels = sorted(set([level_95, level_68, H.max()]))
                        if len(levels) < 2 or H.max() <= 0:
                            raise ValueError("degenerate histogram")
                        cs = ax.contourf(
                            X,
                            Y,
                            H.T,
                            levels=levels,
                            colors=[color],
                            alpha=np.linspace(0.1, 0.3, len(levels) - 1),
                        )
                        # Remove white edges between contour bands.
                        for c in getattr(cs, "collections", []):
                            c.set_edgecolor("face")
                            c.set_rasterized(True)
                        ax.contour(
                            X,
                            Y,
                            H.T,
                            levels=levels[:-1],
                            colors=[color],
                            linewidths=0.8,
                            alpha=0.7,
                        )
                    except (ImportError, np.linalg.LinAlgError, ValueError):
                        # Fallback to scatter if hist2d fails
                        ax.scatter(xj, xi, s=8, alpha=0.4, color=color, edgecolors="none")
                    if all_truths and name_j in all_truths and name_i in all_truths:
                        tj = float(all_truths[name_j])
                        ti = float(all_truths[name_i])
                        if name_j in _log_derived:
                            tj = np.log10(max(tj, 1e-30))
                        if name_i in _log_derived:
                            ti = np.log10(max(ti, 1e-30))
                        ax.axvline(tj, color="#4682b4", ls="--", lw=0.8, alpha=0.5)
                        ax.axhline(ti, color="#4682b4", ls="--", lw=0.8, alpha=0.5)

                # Labels on edges only
                if i == n - 1:
                    ax.set_xlabel(label_map.get(name_j, name_j), fontsize=11)
                    ax.tick_params(axis="x", labelsize=9, rotation=45)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(label_map.get(name_i, name_i), fontsize=11)
                    ax.tick_params(axis="y", labelsize=9)
                else:
                    ax.set_yticklabels([])

                ax.tick_params(labelsize=9)

        plt.tight_layout()
        return fig

    def plot_sed(self, n_draws=200, wave_range=(1000, 30000), ax=None):
        """Plot posterior predictive SED with credible interval.

        Draws ``n_draws`` parameter samples from the posterior, computes
        the rest-frame SED for each, and shades the 16th–84th percentile
        band around the median.

        Parameters
        ----------
        n_draws : int
            Number of posterior draws to use for the band. Ignored for
            MAP results (plots single SED).
        wave_range : (float, float)
            Wavelength range in [Angstrom] to display.
        ax : matplotlib Axes, optional
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        fig : matplotlib Figure
            The SED plot figure.

        Notes
        -----
        Plots λ F_λ (rest-frame spectral energy density) normalized at 5500 Å.
        For MAP results, shows the single best-fit SED.
        For sampling methods, draws n_draws random samples and computes
        percentiles of the SED over those draws.
        Requires ``_model`` to be available (set by ``model.fit()``).
        Uses log-log axes for visibility across wavelength range.

        Examples
        --------
        >>> fig = result.plot_sed(n_draws=500, wave_range=(1000, 10000))
        >>> plt.show()
        """
        import matplotlib.pyplot as plt

        if self._model is None:
            raise ValueError("plot_sed requires a model reference (produced by model.fit())")

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()

        model = self._model

        if self.samples is not None and n_draws > 1:
            key = jax.random.PRNGKey(0)
            n_samples = next(iter(self.samples.values())).shape[0]
            idx = jax.random.choice(
                key, n_samples, shape=(min(n_draws, n_samples),), replace=False
            )
            seds = []
            for i in np.array(idx):
                p = {
                    k: float(v[i]) if v.ndim == 1 else np.array(v[i])
                    for k, v in self.samples.items()
                }
                try:
                    sed = np.array(model.predict_rest_sed(p).sed)
                    seds.append(sed)
                except (AttributeError, TypeError, ValueError):
                    # AttributeError: predict_rest_sed doesn't exist or .sed attribute missing
                    # TypeError: wrong arguments or np.array() conversion failed
                    # ValueError: invalid parameter dict
                    pass
            if seds:
                seds = np.stack(seds, axis=0)
                wave = np.array(model.wavelengths)
                mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
                wave_m = wave[mask]
                lo = np.percentile(seds[:, mask], 16, axis=0)
                med = np.percentile(seds[:, mask], 50, axis=0)
                hi = np.percentile(seds[:, mask], 84, axis=0)
                norm = float(med[np.argmin(np.abs(wave_m - 5500))]) or 1.0
                ax.fill_between(
                    wave_m,
                    lo * wave_m / norm,
                    hi * wave_m / norm,
                    alpha=0.3,
                    color="C0",
                    label="16–84%",
                )
                ax.plot(wave_m, med * wave_m / norm, color="C0", lw=1.8, label="median")
        else:
            wave = np.array(model.wavelengths)
            mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
            sed = np.array(model.predict_rest_sed(self.params).sed)
            norm = float(sed[np.argmin(np.abs(wave - 5500))]) or 1.0
            ax.plot(
                wave[mask], sed[mask] * wave[mask] / norm, color="C0", lw=1.8, label="best-fit"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
        ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)", fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        ax.set_title("Posterior Predictive SED", fontsize=11)
        fig.tight_layout()
        return fig

    def plot_sfh(self, n_draws=200, ax=None):
        """Plot posterior SFH with credible interval.

        Parameters
        ----------
        n_draws : int
            Number of posterior draws. Ignored for MAP (plots single SFH).
        ax : matplotlib Axes, optional
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        fig : matplotlib Figure
            The SFH plot figure.

        Notes
        -----
        Plots both the stochastic SFH (burst component) and smooth component.
        For MAP: shows single best-fit SFH (both stochastic and smooth).
        For sampling: draws n_draws random samples and computes 16th–84th
        percentile bands.
        X-axis: lookback time [Gyr].
        Y-axis: SFR [Msun/yr].
        Requires ``_model`` to be available (set by ``model.fit()``).

        Examples
        --------
        >>> fig = result.plot_sfh(n_draws=500)
        >>> plt.show()
        """
        import matplotlib.pyplot as plt

        if self._model is None:
            raise ValueError("plot_sfh requires a model reference (produced by model.fit())")

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.get_figure()

        model = self._model

        if self.samples is not None and n_draws > 1:
            key = jax.random.PRNGKey(1)
            n_samples = next(iter(self.samples.values())).shape[0]
            idx = jax.random.choice(
                key, n_samples, shape=(min(n_draws, n_samples),), replace=False
            )
            sfhs_mean, sfhs_full, t_gyr = [], [], None
            for i in np.array(idx):
                p = {
                    k: float(v[i]) if v.ndim == 1 else np.array(v[i])
                    for k, v in self.samples.items()
                }
                try:
                    sfh_dict = model.predict_sfh(p)
                    if t_gyr is None:
                        t_gyr = np.array(sfh_dict["t_gyr"])
                    sfhs_mean.append(np.array(sfh_dict["sfr_mean"]))
                    sfhs_full.append(np.array(sfh_dict["sfr_full"]))
                except (AttributeError, TypeError, ValueError, KeyError):
                    # AttributeError: predict_sfh doesn't exist
                    # TypeError: wrong arguments or np.array() conversion failed
                    # ValueError: invalid parameter dict
                    # KeyError: missing t_gyr, sfr_mean, or sfr_full in result
                    pass
            if sfhs_mean and t_gyr is not None:
                sfhs_mean = np.stack(sfhs_mean, axis=0)
                sfhs_full = np.stack(sfhs_full, axis=0)
                lo = np.percentile(sfhs_full, 16, axis=0)
                med = np.percentile(sfhs_full, 50, axis=0)
                hi = np.percentile(sfhs_full, 84, axis=0)
                ax.fill_between(t_gyr, lo, hi, alpha=0.3, color="C0", label="16–84%")
                ax.plot(t_gyr, med, color="C0", lw=1.8, label="median (stochastic)")
                med_mean = np.percentile(sfhs_mean, 50, axis=0)
                ax.plot(t_gyr, med_mean, color="0.5", lw=1.2, ls="--", label="median (smooth)")
        else:
            sfh_dict = model.predict_sfh(self.params)
            t_gyr = np.array(sfh_dict["t_gyr"])
            ax.plot(t_gyr, np.array(sfh_dict["sfr_full"]), color="C0", lw=1.8, label="best-fit")
            ax.plot(
                t_gyr,
                np.array(sfh_dict["sfr_mean"]),
                color="0.5",
                lw=1.2,
                ls="--",
                label="smooth component",
            )

        ax.set_xlabel("Lookback time [Gyr]", fontsize=11)
        ax.set_ylabel(r"SFR [$M_\odot$ yr$^{-1}$]", fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        ax.set_title("Star Formation History", fontsize=11)
        fig.tight_layout()
        return fig

    # ── Display ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        n = "None" if self.samples is None else next(iter(self.samples.values())).shape[0]
        return (
            f"Posterior(method='{self.method}', n_samples={n}, wall_time={self.wall_time_s:.1f}s)"
        )

    # ── Method chaining ───────────────────────────────────────────

    def refine(self, method: str, **kwargs):
        """Re-run inference from this result using a different method.

        Requires that this Posterior was produced by ``model.fit()`` or
        ``fitter.run()`` — both set the ``._fitter`` back-reference.

        Parameters
        ----------
        method : str
            Any canonical method name accepted by ``Fitter.run()``.
            E.g. ``"mcmc_raytrace"``, ``"mcmc_nuts"``, ``"vi"``.
        **kwargs
            Passed to ``Fitter.run()`` (e.g. ``n_steps``, ``n_warmup``).

        Returns
        -------
        Posterior
            New result warm-started from this posterior.

        Raises
        ------
        RuntimeError
            If ``._fitter`` is not set (Posterior created outside model.fit/fitter.run).

        Notes
        -----
        Warm-starts the new inference from this posterior's parameters or samples.
        Common use cases: VI → MCMC refinement (exact inference on top of variational
        fit), MCMC → different sampler (e.g. raytrace → nuts), or quick method
        → expensive method for publication.

        Examples
        --------
        >>> result_vi = model.fit(flux, noise)
        >>> result_exact = result_vi.refine("mcmc_raytrace", n_steps=1000)
        """
        if self._fitter is None:
            raise RuntimeError(
                "Posterior.refine() requires a back-reference to its Fitter. "
                "Use model.fit() or fitter.run() to produce this Posterior. "
                "Posteriors loaded from disk or created manually lack this reference."
            )
        return self._fitter.run(method, init_from=self, **kwargs)

    def validate(self, n_steps: int = 200, **kwargs):
        """Run a short MCMC check and return a validation summary.

        Runs ``n_steps`` of Ray Tracing (or NUTS for D≤20) from this
        posterior's MAP estimate, then computes the marginal overlap
        between this posterior and the MCMC check posterior for each
        parameter.

        Parameters
        ----------
        n_steps : int
            Number of MCMC steps. Default 200 (quick sanity check).
        **kwargs
            Forwarded to the MCMC run.

        Returns
        -------
        dict
            Keys: ``"mcmc_result"`` (Posterior from MCMC check),
            ``"overlap"`` (dict of float per parameter, 1.0 = perfect overlap,
            0.0 = no overlap), ``"passed"`` (bool, True when all overlaps > 0.5).

        Raises
        ------
        RuntimeError
            If ``._fitter`` is not set.

        Notes
        -----
        Validation checks whether a quick MCMC run agrees with the current
        posterior (typically from VI or MAP). High overlap (> 0.5) indicates
        the method is reliable; low overlap suggests the posterior may be
        biased or misspecified.
        Overlap is computed as the histogram intersection at each parameter.
        For sampling methods (VI, geoVI), validates the approximate posterior.
        For MCMC methods, serves as a sanity check for chain convergence.

        Examples
        --------
        >>> result_vi = model.fit(flux, noise, method="vi")
        >>> val = result_vi.validate(n_steps=500)
        >>> print(f"Validation passed: {val['passed']}")
        >>> for param, ov in val["overlap"].items():
        ...     print(f"{param}: overlap={ov:.3f}")
        """
        if self._fitter is None:
            raise RuntimeError(
                "Posterior.validate() requires a back-reference to its Fitter. "
                "Use model.fit() or fitter.run() to produce this Posterior."
            )
        d = self._fitter.spec.n_free
        mcmc_method = "mcmc_nuts" if d <= 20 else "mcmc_raytrace"
        mcmc_result = self._fitter.run(mcmc_method, init_from=self, n_steps=n_steps, **kwargs)

        # Compute per-parameter marginal overlap (histogram intersection)
        overlap: dict[str, float] = {}
        if self.samples is not None and mcmc_result.samples is not None:
            import numpy as np

            for name in self.samples:
                if name == "psd_xi":
                    continue
                vi_arr = np.array(self.samples[name])
                mc_arr = np.array(mcmc_result.samples[name])
                if vi_arr.ndim != 1:
                    continue
                lo = min(vi_arr.min(), mc_arr.min())
                hi = max(vi_arr.max(), mc_arr.max())
                if hi <= lo:
                    overlap[name] = 1.0
                    continue
                bins = np.linspace(lo, hi, 30)
                h_vi, _ = np.histogram(vi_arr, bins=bins, density=True)
                h_mc, _ = np.histogram(mc_arr, bins=bins, density=True)
                bin_w = bins[1] - bins[0]
                overlap[name] = float(np.sum(np.minimum(h_vi, h_mc)) * bin_w)

        passed = all(v > 0.5 for v in overlap.values()) if overlap else True
        return {"mcmc_result": mcmc_result, "overlap": overlap, "passed": passed}
