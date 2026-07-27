"""Shared setup for the tengri notebooks.

Holds only what is genuinely incidental to the science: the framework notices
worth quieting, the filter effective-wavelength helper every photometry plot
needs, and the one HMC recipe the fitting notebooks share. They live here once
so a reader meets them once, and so the notebooks differ only where the
*science* differs.

Loading the SSP grid deliberately stays **in** each notebook: which stellar
library you fit with is a scientific choice, and ``tengri.download_ssp`` is
part of the public API a reader should see.

Usage::

    from _setup import HMC_VALIDATED, effective_wavelengths_um, quiet

    quiet()

Companion module: :mod:`_plot_style` (matplotlib configuration).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import numpy as np

# The convergence-validated fixed-length HMC recipe. Fixed-length HMC (rather
# than NUTS) keeps the leapfrog count off the critical path, so the cost of a
# fit is predictable; these settings mix the six-to-eight parameter posteriors
# in the fitting notebooks cleanly (max split-R-hat < 1.01, few divergences).
# Pass a `key=` at the call site — a shared seed would make the notebooks'
# chains identical by accident rather than by design.
HMC_VALIDATED = dict(
    method="mcmc_hmc",
    n_warmup=1000,
    n_samples=600,
    n_leapfrog_steps=20,
    dense_mass_matrix=True,
    target_accept_rate=0.9,
)


def quiet():
    """Silence framework notices that do not change the science shown.

    Baked-in nebular emission, the precompute approximations, and the
    recipe/parameter-provenance notices are all deliberate choices in these
    notebooks, and their warnings would otherwise repeat on every cell.
    Genuine deprecations in user-facing calls are fixed in the code, never
    hidden here.
    """
    for message in (
        ".*BakedInBackend.*",
        ".*WavePrecomp.*",
        ".*was marked FIXED.*",
        ".*Composable AGN.*",
        ".*before the Big Bang.*",
    ):
        warnings.filterwarnings("ignore", message=message)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def effective_wavelengths_um(photometry):
    """Transmission-weighted effective wavelength of each filter.

    Parameters
    ----------
    photometry : Photometry
        The photometric schema, e.g. ``observation.photometry``.

    Returns
    -------
    ndarray, shape (n_filters,)
        Effective wavelengths [micron], for plotting photometry against a
        wavelength axis.
    """
    return (
        np.array(
            [
                np.trapezoid(w * t, w) / np.trapezoid(t, w)
                for w, t in zip(photometry.filter_waves, photometry.filter_trans)
            ]
        )
        / 1e4
    )
