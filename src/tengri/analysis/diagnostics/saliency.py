# SPDX-License-Identifier: BSD-3-Clause
"""Gradient SEDs / saliency maps: ∂flux(λ)/∂θ across wavelength.

Computes the sensitivity of each wavelength to each physical parameter.
This tells you:

- Which wavelength ranges are most informative for each parameter
- How sensitivity changes with galaxy type or redshift
- Which spectral features drive the constraints

Only possible because the full pipeline is differentiable.
"""

import jax
import jax.numpy as jnp


def compute_gradient_sed(forward_model, params, param_name):
    """Compute ∂SED(λ)/∂θ: the gradient of the SED w.r.t. one parameter.

    Parameters
    ----------
    forward_model: ForwardModel
        Configured forward model.
    params: dict
        Parameter values.
    param_name: str
        Which parameter to differentiate with respect to.

    Returns
    -------
    gradient_sed: array, shape (n_wave,)
        ∂SED/∂θ at each wavelength.
    wavelength: array, shape (n_wave,)
        Rest-frame wavelengths (Angstrom).
    """

    def sed_as_fn_of_param(val):
        """Compute SED with modified parameter value.

        Parameters
        ----------
        val: float
            Parameter value to substitute.

        Returns
        -------
        array, shape (n_wave,)
            SED at the given parameter value.
        """
        p = dict(params)
        p[param_name] = val
        return forward_model._predict_rest_sed(p).sed

    # Per-wavelength gradient: jacobian of SED w.r.t. the scalar parameter.
    gradient_sed = jax.jacobian(sed_as_fn_of_param)(params[param_name])

    return gradient_sed, forward_model.ssp_data.ssp_wave


def compute_all_gradient_seds(forward_model, params, param_names=None):
    """Compute gradient SEDs for all physical parameters.

    Parameters
    ----------
    forward_model: ForwardModel
        Configured forward model.
    params: dict
        Parameter values.
    param_names: list of str, optional
        Parameters to compute gradients for.

    Returns
    -------
    gradients: dict of {param_name: array (n_wave,)}
        Gradient SED per parameter.
    wavelength: array, shape (n_wave,)
        Rest-frame wavelengths.
    """
    if param_names is None:
        param_names = [
            "psd_sigma",
            "psd_tau_yr",
            "alpha",
            "beta",
            "tau_sfh",
            "sfr_norm",
            "log_z_abs",
            "tau_bc",
            "tau_diff",
            "dust_slope",
        ]

    gradients = {}
    for name in param_names:
        grad_sed, wave = compute_gradient_sed(forward_model, params, name)
        gradients[name] = grad_sed

    return gradients, wave


def compute_photometry_sensitivity(forward_model, params, param_names=None):
    """Compute ∂flux_band/∂θ for each filter and parameter.

    Returns a matrix showing how sensitive each photometric band
    is to each parameter: useful for understanding degeneracies
    and planning filter sets.

    Parameters
    ----------
    forward_model: ForwardModel
        Configured forward model (must have filters set).
    params: dict
        Parameter values.
    param_names: list of str, optional
        Parameters to include.

    Returns
    -------
    sensitivity: array, shape (n_filters, n_params)
        ∂flux_band/∂θ_param matrix.
    param_names: list of str
        Parameter names (columns).
    """
    if param_names is None:
        param_names = [
            "psd_sigma",
            "psd_tau_yr",
            "alpha",
            "beta",
            "tau_sfh",
            "sfr_norm",
            "log_z_abs",
            "tau_bc",
            "tau_diff",
            "dust_slope",
        ]

    def predict_from_flat(flat):
        """Map flattened parameters to photometric predictions.

        Parameters
        ----------
        flat: array, shape (n_params,)
            Flattened free parameter vector.

        Returns
        -------
        array, shape (n_filters,)
            Predicted photometry.
        """
        p = dict(params)
        for i, name in enumerate(param_names):
            p[name] = flat[i]
        return forward_model.predict_photometry(p)

    flat = jnp.array([float(params[n]) for n in param_names])
    jac = jax.jacobian(predict_from_flat)(flat)  # (n_filters, n_params)

    return jac, param_names
