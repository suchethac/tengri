# SPDX-License-Identifier: BSD-3-Clause
"""Mock galaxy population generator for hierarchical PSD recovery studies.

Produces populations with injected truths and realistic measurement noise,
checking that injected values are discriminable from prior returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

__all__ = ["MockPopulation", "assert_truth_is_discriminating", "make_population"]


def assert_truth_is_discriminating(value, bounds, *, name, rel_tol=0.08):
    r"""Reject an injected truth that a prior-returning estimator could fake.

    Two distinct points in a bounded prior are indistinguishable from "the
    estimator returned its prior", depending on the standardization in force:

    - **Arithmetic midpoint** ``0.5(lo + hi)`` — where a uniform prior or
      sigmoid-standardized logit-normal returns nothing (prior expectation).
    - **Geometric mean** ``sqrt(lo * hi)`` — where a log-uniform prior returns
      nothing (same value as the lognormal median by mathematical identity).

    Which applies depends on tengri's standardization, which has changed
    historically, so both points are excluded rather than guessing which is
    current.

    Note: The geometric mean and lognormal median ``exp(0.5*(log(lo)+log(hi)))``
    are mathematically identical.

    Parameters
    ----------
    value : float
        The truth to inject [units vary by parameter].
    bounds : tuple of float
        ``(lo, hi)`` prior support, in the same units as ``value``.
    name : str
        Parameter name, for the error message.
    rel_tol : float, optional
        Fractional distance from a characteristic point that counts as too
        close [dimensionless]. Default 0.08.
        Note: For very wide bounds (e.g. (1e6, 3e8)), the span is dominated
        by the upper end; ``rel_tol * span`` may not scale intuitively.
        This guard is designed for bounded priors like (10, 500) Myr.

    Raises
    ------
    ValueError
        If ``value`` lies within ``rel_tol`` of any characteristic point.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    characteristic = {
        "arithmetic midpoint": 0.5 * (lo + hi),
        "geometric mean": float(np.sqrt(lo * hi)),
    }
    span = hi - lo
    for label, point in characteristic.items():
        if abs(value - point) < rel_tol * span:
            raise ValueError(
                f"Injected truth {name}={value:g} is within {rel_tol:.0%} of the "
                f"prior's {label} ({point:g}) on bounds ({lo:g}, {hi:g}). A "
                f"recovered value there is indistinguishable from the prior: an "
                f"estimator that learned nothing returns the same number. Choose "
                f"a truth away from {sorted(set(round(p, 4) for p in characteristic.values()))}."
            )


@dataclass
class MockPopulation:
    """Mock galaxy population with truths and realistic noise.

    Attributes
    ----------
    table : ndarray
        Per-galaxy record array with keys for photometry fluxes, errors,
        line fluxes, and line errors [various units].
    truth_params : list[dict[str, ndarray]]
        List of per-galaxy parameter dicts sampled from the prior
        with injected ``sigma`` and ``tau_myr`` values.
    n_halpha_absorption : int
        Count of galaxies whose Halpha is predicted in absorption
        (non-positive flux). Never dropped; reported to avoid selection bias.
    """

    table: np.ndarray
    truth_params: list[dict[str, Any]]
    n_halpha_absorption: int


def make_population(
    model,
    *,
    n_galaxies: int,
    sigma_true: float,
    tau_true_myr: float,
    key: jax.Array,
    snr_phot: float,
    snr_line: float,
) -> MockPopulation:
    """Generate a mock galaxy population with injected PSD parameters.

    Draws each galaxy by sampling its parameters from the prior, overriding
    the PSD burstiness ``sigma`` and timescale ``tau_myr`` with injected
    truths, then predicting photometry and emission lines with realistic
    noise. Halpha absorption events are counted but never dropped, as removal
    would bias survivors toward line-bright cases.

    Parameters
    ----------
    model : SEDModel
        The parametrized SED model with an established prior.
    n_galaxies : int
        Number of mock galaxies to generate [count].
    sigma_true : float
        Injected PSD burstiness parameter [dex].
    tau_true_myr : float
        Injected PSD timescale [Myr].
    key : jax.Array
        PRNGKey for galaxy sampling and noise realization.
    snr_phot : float
        Signal-to-noise ratio for photometry [dimensionless].
    snr_line : float
        Signal-to-noise ratio for emission lines [dimensionless].

    Returns
    -------
    MockPopulation
        A population record with observed photometry and lines, their errors,
        and per-galaxy truth parameters.

    Notes
    -----
    Uses ``model.measure_line_fluxes(params, line_defs, fast=False)`` to
    measure emission lines, the same operator the likelihood uses, ensuring
    mock self-consistency. Halpha in absorption is a legitimate outcome of
    burstiness during a lull and is always reported via ``n_halpha_absorption``.
    """
    from tengri.observation.line_measurement import default_line_defs

    # Validate the injected truths against their priors
    sigma_bounds = model.spec._distributions["sfh_field_psd_sigma"].bounds
    tau_bounds = model.spec._distributions["sfh_field_psd_tau_myr"].bounds

    assert_truth_is_discriminating(sigma_true, sigma_bounds, name="sfh_field_psd_sigma")
    assert_truth_is_discriminating(tau_true_myr, tau_bounds, name="sfh_field_psd_tau_myr")

    # Generate keys for each galaxy
    keys = jax.random.split(key, n_galaxies)

    # Sample parameters and override PSD terms
    truth_params = []
    phot_true_list = []
    phot_noise_list = []
    phot_obs_list = []
    line_flux_obs_list = []
    line_flux_noise_list = []
    halpha_absorption_count = 0

    # Define emission lines for measurement
    # Strong star-forming set: drops Hgamma and [NII]_6548 (near-zero fluxes let
    # SNR-scaled errors dominate chi-squared). Halpha/Hbeta enable Balmer decrement
    # dust constraint; dust-SFR degeneracy is key to the study.
    # Wavelengths sourced from canonical catalog, not hardcoded, for consistency.
    from tengri.observation.line_list import LineList

    line_names = [
        "OII_3726",
        "OII_3729",
        "OIII_4959",
        "OIII_5007",
        "Halpha",
        "Hbeta",
        "SII_6717",
        "NII_6584",
    ]
    # Select lines from canonical catalog (LineList.default_optical) by name
    # (select() returns lines in wavelength order, so reorder to match line_names)
    canonical_lines = LineList.default_optical().select(names=line_names)
    canonical_dict = {
        name: wave
        for name, wave in zip(canonical_lines.names, canonical_lines.wavelengths)
    }
    wavelengths = np.array([canonical_dict[name] for name in line_names])
    line_defs = default_line_defs(
        wavelengths=wavelengths,
        names=line_names,
    )

    for i, k_i in enumerate(keys):
        # Sample parameters from prior
        params = model.spec.sample(k_i)

        # Override PSD parameters with truths
        params["sfh_field_psd_sigma"] = sigma_true
        params["sfh_field_psd_tau_myr"] = tau_true_myr

        truth_params.append(params)

        # Predict photometry
        phot_true = np.asarray(model.predict_photometry(params))
        phot_noise = phot_true / snr_phot

        # Add photometric noise
        key_phot = jax.random.fold_in(k_i, i * 2)
        phot_obs = phot_true + phot_noise * np.asarray(
            jax.random.normal(key_phot, shape=phot_true.shape)
        )

        phot_true_list.append(phot_true)
        phot_noise_list.append(phot_noise)
        phot_obs_list.append(phot_obs)

        # Measure emission lines (returns array of shape (n_line,))
        line_flux_true = np.asarray(
            model.measure_line_fluxes(params, line_defs=line_defs, fast=False)
        )
        line_flux_noise = np.abs(line_flux_true) / snr_line

        # Add line-flux noise (absolute value to handle zero/negative fluxes)
        key_line = jax.random.fold_in(k_i, i * 2 + 1)
        line_flux_obs = line_flux_true + line_flux_noise * np.asarray(
            jax.random.normal(key_line, shape=line_flux_true.shape)
        )

        line_flux_obs_list.append(line_flux_obs)
        line_flux_noise_list.append(line_flux_noise)

        # Count Halpha absorption (non-positive predicted flux)
        halpha_idx = line_names.index("Halpha")
        if line_flux_true[halpha_idx] <= 0:
            halpha_absorption_count += 1

    # Stack into record array
    n_bands = len(phot_true_list[0])
    n_lines = len(line_flux_obs_list[0])

    # Create structured array with photometry and line data
    dtype = [
        ("phot_flux_obs", f"({n_bands},)f8"),
        ("phot_flux_err", f"({n_bands},)f8"),
        ("line_flux_obs", f"({n_lines},)f8"),
        ("line_flux_err", f"({n_lines},)f8"),
    ]
    table = np.zeros(n_galaxies, dtype=dtype)

    for i in range(n_galaxies):
        table[i]["phot_flux_obs"] = phot_obs_list[i]
        table[i]["phot_flux_err"] = phot_noise_list[i]
        table[i]["line_flux_obs"] = line_flux_obs_list[i]
        table[i]["line_flux_err"] = line_flux_noise_list[i]

    return MockPopulation(
        table=table,
        truth_params=truth_params,
        n_halpha_absorption=halpha_absorption_count,
    )
