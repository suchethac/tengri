"""tengri: Differentiable SED fitting with IFT star formation history priors.

A modular, fully differentiable JAX pipeline:
PSD-governed GP → SFH → DSPS SED → photometry/spectroscopy.
"""

# Enable float64 — required for cosmological distance calculations
# (dL^2 at z>0.01 overflows float32)
import jax

jax.config.update("jax_enable_x64", True)

# Enable persistent XLA compilation cache — avoids re-compilation
# across sessions/restarts. ~10x first-call speedup on subsequent runs.
# ~/.cache persists across reboots (unlike /tmp on macOS).
import os as _os

jax.config.update(
    "jax_compilation_cache_dir",
    _os.path.join(_os.path.expanduser("~"), ".cache", "tengri_jax_cache"),
)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

__version__ = "0.1.0"

# --- New high-level API ---
from tengri.core.mock import MockData, generate_mock
from tengri.core.model import Model, PriorPredictive
from tengri.core.noise import (
    compute_effective_noise,
    compute_std_inv,
    has_noise_model,
    uses_student_t,
    variable_noise_hamiltonian,
)
from tengri.core.param_spec import Parameters, ParamSpec
from tengri.core.prediction import (
    DerivedQuantities,
    EmissionLines,
    Prediction,
    SEDQuantities,
    SFHQuantities,
)
from tengri.distributions import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from tengri.inference.fitter import Fitter
from tengri.inference.hierarchical import (
    HierarchicalFitter,
    HierarchicalResult,
    PopulationFitter,
    PopulationPosterior,
)
from tengri.inference.posterior import Posterior
from tengri.inference.raytrace import sample_raytrace
from tengri.inference.vi_config import VIConfig
from tengri.models.agn.agn_config import AGNConfig
from tengri.models.dust.attenuation import two_component_dust
from tengri.models.observation.filters import load_filter_set
from tengri.models.observation.line_catalog import LineCatalog, LineList
from tengri.models.observation.noise_config import NoiseConfig, NoiseModel
from tengri.models.observation.observation import Observation
from tengri.models.observation.photometry_config import Photometry
from tengri.models.observation.spectroscopy_config import Spectroscopy, SpectroscopyConfig
from tengri.models.sfh.gp_sfh import (
    compute_sqrt_power_drw,
    generate_gp_batch,
    generate_gp_fourier,
    gp_from_xi,
    make_log_age_grid,
)
from tengri.models.sfh.mean_sfh import (
    AGEMAX_YR,
    constant_sfh,
    delayed_exponential_sfh,
    delayed_tau,
    double_powerlaw,
    dpl,
    exponential_sfh,
    lnorm,
    norm,
    snorm,
    triweight_burst,
    tsnorm,
)
from tengri.models.sfh.psd_models import drw_acf, drw_variance, psd_drw
from tengri.models.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)
from tengri.models.sps.dsps_wrapper import (
    SSPData,
    effective_metallicity,
    has_alpha_grid,
    interpolate_met_alpha,
    load_ssp_data,
    salaris_feh_from_mh,
    salaris_mh_from_feh,
)


def posteriors_to_dataframe(results: list, params: list[str] | None = None):
    """Summarise a list of Posteriors into a pandas DataFrame.

    Requires ``pandas`` (``pip install pandas``).

    Parameters
    ----------
    results : list of Posterior
        Output of ``model.fit_catalog()`` or any list of Posterior objects.
    params : list of str or None
        Parameter names to include. Default: all scalar free parameters,
        excluding ``psd_xi``.

    Returns
    -------
    pandas.DataFrame
        One row per galaxy, columns: ``{param}_median``, ``{param}_lo68``,
        ``{param}_hi68`` for each requested parameter.

    Examples
    --------
    >>> df = tengri.posteriors_to_dataframe(results, params=["met_logzsol", "dust_tau_bc"])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "posteriors_to_dataframe() requires pandas: pip install pandas"
        ) from None

    import numpy as np

    rows = []
    for result in results:
        row: dict = {}

        if result.samples is None:
            # MAP: use point estimates
            for name, val in result.params.items():
                if name == "psd_xi":
                    continue
                if params is not None and name not in params:
                    continue
                row[f"{name}_value"] = float(np.mean(np.array(val)))
        else:
            # Sampling: use median + 68% CI
            for name, arr in result.samples.items():
                if name == "psd_xi":
                    continue
                if params is not None and name not in params:
                    continue
                arr_np = np.array(arr)
                if arr_np.ndim != 1:
                    continue
                row[f"{name}_median"] = float(np.median(arr_np))
                row[f"{name}_lo68"] = float(np.percentile(arr_np, 16))
                row[f"{name}_hi68"] = float(np.percentile(arr_np, 84))

        rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    # High-level API
    "AGEMAX_YR",
    # Registry
    "FIELD_MODEL_REGISTRY",
    "SFH_REGISTRY",
    "AGNConfig",
    "DerivedQuantities",
    "EmissionLines",
    "Fitter",
    "Fixed",
    "Gaussian",
    "HierarchicalFitter",
    "HierarchicalResult",
    "LineCatalog",
    "LogNormal",
    "LogUniform",
    "MockData",
    "Model",
    "NoiseConfig",
    "Observation",
    "ParamSpec",
    "Photometry",
    "Posterior",
    "Prediction",
    "PriorPredictive",
    "SEDQuantities",
    "SFHQuantities",
    "SSPData",
    "SpectroscopyConfig",
    "StudentT",
    "Uniform",
    "VIConfig",
    "compute_effective_noise",
    "compute_field_gp",
    "compute_sqrt_power_drw",
    "compute_std_inv",
    # SFH models
    "constant_sfh",
    "delayed_exponential_sfh",
    "delayed_tau",
    "double_powerlaw",
    "dpl",
    "drw_acf",
    "drw_variance",
    "exponential_sfh",
    "generate_gp_batch",
    "generate_gp_fourier",
    "generate_mock",
    "gp_from_xi",
    "has_noise_model",
    "lnorm",
    "load_filter_set",
    "load_ssp_data",
    "make_log_age_grid",
    "norm",
    "posteriors_to_dataframe",
    "psd_drw",
    "resolve_sfh",
    "sample_raytrace",
    "snorm",
    "triweight_burst",
    "tsnorm",
    "two_component_dust",
]

# Plotting utilities
from tengri.plotting import (
    COLORS,
    SDSS_WAVE_EFF,
    SPECTRAL_FEATURES,
    diagnostics_table,
    plot_corner_comparison,
    plot_sed_fit,
    plot_sfh,
    plot_sfh_comparison,
    plot_spectrum_fit,
    safe_corner,
    setup_style,
)
