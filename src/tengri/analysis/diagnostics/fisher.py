"""Fisher Information Matrix (FIM) computation.

The FIM quantifies how much information the data carries about each
parameter. Because the entire forward model is differentiable, we
can compute the exact FIM via autodiff — no finite differences needed.

Applications:
1. Fisher forecasting: which parameters are constrained by which data?
2. Optimal filter design: differentiate FIM w.r.t. filter choices
3. Survey design: what S/N is needed to break the age-dust degeneracy?
4. Laplace approximation: cheap posteriors from Hessian at MAP

The FIM at parameters θ with Gaussian noise is:

    F_ij = sum_k (1/σ_k^2) * (∂m_k/∂θ_i) * (∂m_k/∂θ_j)

where m_k is the model prediction at data point k and σ_k is the noise.
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0,))
def compute_jacobian(predict_fn, params, param_keys):
    """Compute the Jacobian ∂m/∂θ of model predictions w.r.t. parameters.

    Parameters
    ----------
    predict_fn : callable
        Function mapping a flat parameter array to model predictions.
    params : array, shape (n_params,)
        Parameter values.
    param_keys : tuple of str
        Parameter names (for static hashing).

    Returns
    -------
    array, shape (n_data, n_params)
        Jacobian matrix.
    """
    return jax.jacobian(predict_fn)(params)


def compute_fisher_matrix(forward_model, params, noise, data_type="photometry", param_names=None):
    """Compute the Fisher Information Matrix.

    F_ij = sum_k (1/sigma_k^2) * (dm_k/dtheta_i) * (dm_k/dtheta_j)

    Parameters
    ----------
    forward_model : ForwardModel
        Configured forward model.
    params : dict
        Parameter values at which to evaluate the FIM.
    noise : array
        1-sigma uncertainties on the data.
    data_type : str
        "photometry" or "spectroscopy".
    param_names : list of str, optional
        Which parameters to include. Defaults to all physical params
        (excludes xi — the GP latent is high-dimensional).

    Returns
    -------
    fim : array, shape (n_params, n_params)
        Fisher Information Matrix.
    names : list of str
        Parameter names corresponding to FIM rows/columns.
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

    # Build a function that maps a flat array of the selected params to predictions
    def predict_from_flat(flat_params):
        """Map flattened parameter array to model predictions.

        Parameters
        ----------
        flat_params : array, shape (n_params,)
            Flattened free parameter vector.

        Returns
        -------
        array
            Model predictions (photometry or spectrum).
        """
        p = dict(params)  # copy
        for i, name in enumerate(param_names):
            p[name] = flat_params[i]

        if data_type == "photometry":
            return forward_model.predict_photometry(p)
        elif data_type == "spectroscopy":
            return forward_model.predict_spectrum(p, forward_model._wave_obs)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    # Current parameter values as flat array
    flat = jnp.array([float(params[n]) for n in param_names])

    # FIM = J^T N^{-1} J with N^{-1} = diag(1/sigma^2).
    jac = jax.jacobian(predict_from_flat)(flat)
    noise_inv = 1.0 / noise**2
    weighted_jac = jac * noise_inv[:, None]
    fim = jac.T @ weighted_jac

    return fim, param_names


def fisher_parameter_errors(fim):
    """Compute 1-sigma parameter uncertainties from the FIM.

    sigma_i = sqrt((F^{-1})_ii)

    Parameters
    ----------
    fim : array, shape (n_params, n_params)
        Fisher Information Matrix.

    Returns
    -------
    array, shape (n_params,)
        1-sigma marginal uncertainties.
    """
    cov = jnp.linalg.inv(fim)
    return jnp.sqrt(jnp.diag(cov))


def fisher_correlation_matrix(fim):
    """Compute parameter correlation matrix from the FIM.

    Parameters
    ----------
    fim : array, shape (n_params, n_params)
        Fisher Information Matrix.

    Returns
    -------
    array, shape (n_params, n_params)
        Correlation matrix (values in [-1, 1]).
    """
    cov = jnp.linalg.inv(fim)
    diag = jnp.sqrt(jnp.diag(cov))
    return cov / jnp.outer(diag, diag)
