# SPDX-License-Identifier: BSD-3-Clause
"""Cue neural network emulator for nebular emission (pure JAX).

A JAX re-implementation of the Cue emulator (Li et al. 2025) that predicts
nebular emission lines and continuum from 12 nebular parameters using
pre-trained Speculator neural networks with PCA output basis.

The weights are loaded from a single npz file (produced by
``scripts/convert_cue_weights.py``) so there is NO dependency on
TensorFlow, dill, or sklearn at runtime.

Architecture per sub-network
----------------------------
1. Normalize inputs: ``x = (params - shift) / scale``
2. Hidden layers with learned Swish: ``x * (beta + (1 - beta) * sigmoid(alpha * x))``
3. Linear output layer -> PCA coefficients
4. Rescale PCA: ``coeffs * pca_scale + pca_shift``
5. PCA inverse: ``coeffs @ components + mean`` (sklearn convention)
6. Rescale spectrum: ``result * log_spec_scale + log_spec_shift``
7. Output: log10 luminosity (lines in Lsun/Q_H, continuum in Lsun/Hz/Q_H)

The 12 input parameters (user-facing)
--------------------------------------
ionspec_index1..4, ionspec_logLratio1..3 (7 ionizing spectrum shape params),
gas_logu, gas_logn, gas_logz, gas_logno, gas_logco (5 gas properties).

IMPORTANT: The network internally takes gas_logq (not gas_logu), and
gas_logn is converted to linear density (10**logn). The conversion is
handled transparently.

Ionizing spectrum parameterization
-----------------------------------
Cue parameterizes the ionizing SED as a broken power law in wavelength space
across four segments (approximately 50–100 Å, 100–228 Å, 228–300 Å, 300–912 Å).
Each segment has a slope parameter (``ionspec_index1..4``, where
``F_ν ∝ λ^index`` in that segment) and the three log-luminosity ratios between
adjacent segment boundaries (``ionspec_logLratio1..3``) fix the relative
normalization.

This 7-parameter description generalizes the single-slope power law used in
Feltre+2016 (``alpha``) and the two-segment model common in older CLOUDY runs.

Comparison with BEAGLE nebular models
---------------------------------------
BEAGLE (Chevallard & Charlot 2016) uses fixed-ionizing-SED CLOUDY grids (BC03
stellar populations for HII regions; Feltre+2016 broken power law for AGN NLR).
Cue differs in the following ways:

- **Arbitrary ionizing SED**: the 7-parameter ionizing spectrum can represent any
  stellar population shape (hot stars, stripped-star envelopes, WR-enriched
  bursts, AGN power-laws) without being pre-committed to a specific SSP library.
  The conversion to Cue parameters for AGN inputs is in ``agn_nebular.py``.

- **~271 emission lines** vs 18 lines in the Gutkin+2016 HII grid used by BEAGLE.
  This enables cross-matching with JWST NIRSpec line maps and rest-UV diagnostics
  (e.g. CIII]1909, CIV1548, HeII1640) that are absent in the BEAGLE grids.

- **C/O and N/O as free parameters**: BEAGLE/Gutkin+2016 has C/O as a discrete
  grid axis (9 values) but fixes N/O to scaled-solar.  Cue accepts continuous
  ``gas_logco`` and ``gas_logno`` offset parameters, enabling smooth gradient-
  based inference over abundance ratios.

- **Differentiable by design**: as a neural network, Cue is smooth and
  differentiable through JAX, enabling VI (ELBO gradients) and HMC
  (Hamiltonian gradients) over nebular parameters jointly with SFH and dust.
  CLOUDY grid interpolation in BEAGLE is non-differentiable.

- **No age axis**: Cue was trained for time-averaged ionizing spectra (similar to
  BEAGLE's 10^8-yr constant-SFR assumption for HII regions).  For explicit age-
  dependent nebular evolution, use ``CB19Backend`` (``cloudy_cb19.py``).

Relation to Synthesizer grids
------------------------------
The Synthesizer (Lovell et al. 2025; Roper et al. 2026) AGN grids (CLOUDY c23.01, 215 lines,
6 axes: BH mass, Eddington ratio, cos(inclination), metallicity, log U, n_H)
are structurally the closest published counterpart to the grids on which Cue
was trained.  Key similarities: c17+ CLOUDY physics, broad line coverage, and
physical BH-mass + Eddington-ratio parameterization.  Key difference: Cue
replaces the physical-BH axes with the 7 ionizing-spectrum shape parameters
(``ionspec_index1..4``, ``ionspec_logLratio1..3``), which makes it agnostic
to the specific accretion-disc model.

The Cue emulator does NOT require grid files at inference time — the neural
network weights are loaded from ``data/cue_weights.npz``.  The Synthesizer
test grids at ``data/synthesizer_grids/test_grid_agn-nlr.hdf5`` and
``data/synthesizer_grids/test_grid_agn-blr.hdf5`` are available for
cross-validation only (2-point per axis; 19 MB each).  Production grids
require Synthesizer Box credentials (``synthesizer-download --agn-grids``).

References
----------

- Li et al. 2024, ApJ, 969, 28 (Cue v1)
- Li et al. 2025, ApJ, 986, 9 (Cue v2, AGN extension)
- Gutkin, Charlot & Bruzual 2016, MNRAS, 462, 1757 (BEAGLE HII grids)
- Chevallard & Charlot 2016, MNRAS, 462, 1415 (BEAGLE)
- Alsing et al. 2020, ApJS (Speculator architecture)

Notes
-----
All functions are JIT-compatible and differentiable through JAX.

"""

import os
import warnings
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from tengri.components.nebular._constants import _LOG10_ZSUN
from tengri.components.nebular._recombination_coeffs import lyc_dust_escape_factor
from tengri.components.nebular._shared import render_nebular_lines

# ── Physical constants ────────────────────────────────────────────
from tengri.utils.physics_constants import (
    C_CGS as _C_CGS,
    L_SUN_CUE as _LSUN_ERG,  # 3.839e33 — Cue training convention, NOT IAU 2015
)
from tengri.utils.scale import LN10, pow10

_LOG_LSUN = jnp.log10(_LSUN_ERG)
_LOG_4PI = jnp.log10(4.0 * jnp.pi)
_LOG_C = jnp.log10(_C_CGS)

# Maximum SSP age contributing to nebular emission. Re-exported from
# ionizing_spectrum.MAX_NEB_LOG_AGE so the precompute (which lives there)
# and the downstream forward filter (here) share one source of truth.
from tengri.components.nebular.ionizing_spectrum import (
    MAX_NEB_LOG_AGE as _MAX_NEB_LOG_AGE,
)

# Flag to track whether the ionspec defaults warning has been issued (once per process)
_IONSPEC_DEFAULT_WARNED: bool = False


# ── Data containers (immutable NamedTuples for JAX tracing) ───────


class SubNetWeights(NamedTuple):
    """Weights for a single Speculator sub-network.

    Immutable container holding all parameters and learned weights for one
    sub-network in the Cue ensemble.

    Parameters
    ----------
    W : tuple of ndarray
        Weight matrices per layer, shape (in, out) for each layer.
    b : tuple of ndarray
        Bias vectors per layer, shape (out,) for each layer.
    alphas : tuple of ndarray
        Learned Swish activation scale parameters (hidden layers only).
    betas : tuple of ndarray
        Learned Swish activation offset parameters (hidden layers only).
    param_shift : ndarray, shape (n_params,)
        Input normalization offset [dimensionless].
    param_scale : ndarray, shape (n_params,)
        Input normalization scale [dimensionless].
    pca_shift : ndarray, shape (n_pcas,)
        PCA centering offset [dimensionless].
    pca_scale : ndarray, shape (n_pcas,)
        PCA scaling factor [dimensionless].
    log_spec_shift : ndarray, shape (n_wavelengths,)
        Log-spectrum shift [dimensionless].
    log_spec_scale : ndarray, shape (n_wavelengths,)
        Log-spectrum scale [dimensionless].
    pca_components : ndarray, shape (n_pcas, n_wavelengths)
        PCA basis vectors (sklearn convention).
    pca_mean : ndarray, shape (n_wavelengths,)
        PCA mean spectrum [log10 Lsun/Hz/Q_H].
    n_layers : int
        Total number of layers (including output layer).

    Notes
    -----
    **JIT-compatible**: yes — all fields are JAX arrays or tuples thereof.

    """

    W: tuple  # tuple of (in, out) weight matrices per layer
    b: tuple  # tuple of (out,) bias vectors per layer
    alphas: tuple  # tuple of (out,) activation params (hidden layers only)
    betas: tuple  # tuple of (out,) activation params (hidden layers only)
    param_shift: jnp.ndarray  # (n_params,)
    param_scale: jnp.ndarray  # (n_params,)
    pca_shift: jnp.ndarray  # (n_pcas,)
    pca_scale: jnp.ndarray  # (n_pcas,)
    log_spec_shift: jnp.ndarray  # (n_wavelengths,)
    log_spec_scale: jnp.ndarray  # (n_wavelengths,)
    pca_components: jnp.ndarray  # (n_pcas, n_wavelengths) — sklearn PCA basis
    pca_mean: jnp.ndarray  # (n_wavelengths,) — sklearn PCA centering
    n_layers: int


class CueWeights(NamedTuple):
    """All Cue weights for lines + continuum.

    Immutable container holding all neural network weights, precomputed grids,
    and metadata for fast batch-mode inference. Includes precomputed batched
    weight arrays for fast line prediction. The ``batched_*`` fields are
    computed once at load time from individual ``line_nets`` and stored as
    dense JAX arrays.

    Parameters
    ----------
    line_nets : tuple of SubNetWeights
        16 sub-networks, one per emission line group.
    cont_net : SubNetWeights
        Single continuum sub-network.
    line_names : tuple of str
        Line group names corresponding to ``line_nets``.
    line_wav_selections : tuple of ndarray
        Wavelength selection masks per sub-network.
    sorted_line_wav : ndarray, shape (n_lines_total,)
        Sorted emission line wavelengths [Angstrom].
    nn_line_wav : ndarray, shape (n_nn_lines,)
        Concatenated NN output wavelengths [Angstrom].
    line_old_idx : ndarray
        Indices of CLOUDY/FSPS-matched (old) lines.
    cont_wav : ndarray, shape (n_wave_cont,)
        Continuum wavelength grid [Angstrom].
    batched_param_shifts : ndarray, shape (16, 12)
        Stacked input normalization offsets.
    batched_param_scales : ndarray, shape (16, 12)
        Stacked input normalization scales.
    batched_W_hidden : tuple of ndarray
        Stacked weight matrices for hidden layers.
    batched_b_hidden : tuple of ndarray
        Stacked bias vectors for hidden layers.
    batched_alpha_hidden : tuple of ndarray
        Stacked Swish activation scales (hidden layers).
    batched_beta_hidden : tuple of ndarray
        Stacked Swish activation offsets (hidden layers).
    batched_W_out : ndarray, shape (16, 256, max_pcas)
        Stacked output weights (zero-padded).
    batched_b_out : ndarray, shape (16, max_pcas)
        Stacked output biases.
    batched_pca_scale : ndarray, shape (16, max_pcas)
        Stacked PCA scaling factors.
    batched_pca_shift : ndarray, shape (16, max_pcas)
        Stacked PCA centering offsets.
    batched_pca_comp : ndarray, shape (16, max_pcas, max_lines)
        Stacked PCA basis vectors (zero-padded).
    batched_pca_mean : ndarray, shape (16, max_lines)
        Stacked PCA means (zero-padded).
    batched_spec_scale : ndarray, shape (16, max_lines)
        Stacked log-spectrum scaling factors (zero-padded).
    batched_spec_shift : ndarray, shape (16, max_lines)
        Stacked log-spectrum shifts (zero-padded).
    batched_n_lines : tuple of int
        Actual number of output lines per sub-network.
    batched_sort_idx : ndarray, shape (n_total_lines,)
        Wavelength sort indices for line ordering.

    Notes
    -----
    **JIT-compatible**: yes — all fields are JAX arrays or tuples thereof.

    """

    line_nets: tuple  # tuple of SubNetWeights, one per line sub-network
    cont_net: SubNetWeights
    line_names: tuple  # tuple of str
    line_wav_selections: tuple  # tuple of int arrays per sub-network
    sorted_line_wav: jnp.ndarray  # (n_lines_total,) sorted line wavelengths
    nn_line_wav: jnp.ndarray  # (n_nn_lines,) concatenated NN output wavelengths
    line_old_idx: jnp.ndarray  # indices of "old" (cloudyfsps) lines
    cont_wav: jnp.ndarray  # (n_wave_cont,) continuum wavelength grid
    # Precomputed batched arrays for fast line prediction
    batched_param_shifts: jnp.ndarray = None  # (16, 12)
    batched_param_scales: jnp.ndarray = None  # (16, 12)
    batched_W_hidden: tuple = None  # tuple of (16, in, out) per hidden layer
    batched_b_hidden: tuple = None  # tuple of (16, out) per hidden layer
    batched_alpha_hidden: tuple = None  # tuple of (16, out) per hidden layer
    batched_beta_hidden: tuple = None  # tuple of (16, out) per hidden layer
    batched_W_out: jnp.ndarray = None  # (16, 256, max_pcas) zero-padded
    batched_b_out: jnp.ndarray = None  # (16, max_pcas)
    batched_pca_scale: jnp.ndarray = None  # (16, max_pcas)
    batched_pca_shift: jnp.ndarray = None  # (16, max_pcas)
    batched_pca_comp: jnp.ndarray = None  # (16, max_pcas, max_lines)
    batched_pca_mean: jnp.ndarray = None  # (16, max_lines)
    batched_spec_scale: jnp.ndarray = None  # (16, max_lines)
    batched_spec_shift: jnp.ndarray = None  # (16, max_lines)
    batched_n_lines: tuple = None  # tuple of int (actual n_lines per net)
    batched_sort_idx: jnp.ndarray = None  # (n_total_lines,) wavelength sort


# ── Weight loading ────────────────────────────────────────────────

_LINE_NAMES = (
    "H1",
    "He1",
    "He2",
    "C1",
    "C2C3",
    "C4",
    "N",
    "O1",
    "O2",
    "O3",
    "ionE_1",
    "ionE_2",
    "S4",
    "Ar4",
    "Ne3",
    "Ne4",
)


def _load_subnet(npz: dict, prefix: str) -> SubNetWeights:
    """Extract a SubNetWeights from the flat npz dict.

    NumPy, not ``jnp`` (#1631). These are trained constants, and the arrays
    end up cached on a module-level backend; a NumPy array cannot be
    trace-scoped under any trace, so the cache is trace-independent by
    construction. See :func:`load_cue_weights`.
    """
    n_layers = int(npz[f"{prefix}_n_layers"])

    W = tuple(np.asarray(npz[f"{prefix}_W_{i}"]) for i in range(n_layers))
    b = tuple(np.asarray(npz[f"{prefix}_b_{i}"]) for i in range(n_layers))
    alphas = tuple(np.asarray(npz[f"{prefix}_alpha_{i}"]) for i in range(n_layers - 1))
    betas = tuple(np.asarray(npz[f"{prefix}_beta_{i}"]) for i in range(n_layers - 1))

    return SubNetWeights(
        W=W,
        b=b,
        alphas=alphas,
        betas=betas,
        param_shift=np.asarray(npz[f"{prefix}_parameters_shift"]),
        param_scale=np.asarray(npz[f"{prefix}_parameters_scale"]),
        pca_shift=np.asarray(npz[f"{prefix}_pca_shift"]),
        pca_scale=np.asarray(npz[f"{prefix}_pca_scale"]),
        log_spec_shift=np.asarray(npz[f"{prefix}_log_spectrum_shift"]),
        log_spec_scale=np.asarray(npz[f"{prefix}_log_spectrum_scale"]),
        pca_components=np.asarray(npz[f"{prefix}_pca_components"]),
        pca_mean=np.asarray(npz[f"{prefix}_pca_mean"]),
        n_layers=n_layers,
    )


def load_cue_weights(npz_path: str) -> CueWeights:
    """Load all Cue weights from the npz file.

    Parses the pre-trained Speculator neural network weights from a NumPy
    archive and constructs batched weight arrays for fast vectorized inference.

    Parameters
    ----------
    npz_path : str
        Path to ``cue_weights.npz`` produced by ``convert_cue_weights.py``.

    Returns
    -------
    CueWeights
        Immutable container with all weights on **NumPy** arrays, including
        precomputed batched arrays for 16 line sub-networks.

    Notes
    -----
    **JIT-compatible**: no — performs file I/O and array padding at load time.
    Call once per model initialization; results are re-used for all inference.

    **Why NumPy and not** ``jnp`` (#1631): these arrays are cached — the AGN
    NLR path memoizes a backend in ``nlr_cloudy._CUE_AGN_BACKEND``. Under
    omnistaging every ``jnp`` call inside a jit trace stages out to the jaxpr
    and returns a ``DynamicJaxprTracer``, *even on constant inputs*. So a
    first caller that reached this function from inside a trace cached
    trace-scoped arrays, and the next trace raised ``UnexpectedTracerError``
    on ``float32[16,12]`` — 16 line sub-networks x 12 Cue parameters, i.e.
    ``batched_param_shifts``. A NumPy array cannot be trace-scoped under any
    trace, so the cache is trace-independent by construction rather than by
    discipline. Consumers wrap with ``jnp.asarray`` at use. Identical fix and
    reasoning to ``load_grahsp_templates`` (#1462).

    **Batching strategy**: Hidden layers are stacked as (16, in, out); output
    layers and PCA arrays are zero-padded to max dimensions (max_pcas, max_lines)
    to enable uniform batch matrix multiplication during inference.

    References
    ----------
    .. [1] Li et al. 2025, "Cue: A fast neural network emulator for nebular
        emission line and continuum predictions", ApJ, 986, 9 (2025).
        arXiv:2405.04598. https://doi.org/10.3847/1538-4357/ad7fe3
    .. [2] Charlot & Fall 2000, "A simple model for the absorption of starlight
        by dust grains and its application to metal-rich galaxies", ApJ 539, 718

    """
    npz = dict(np.load(npz_path, allow_pickle=True))

    # Line sub-networks
    line_nets = []
    line_wav_sels = []
    for name in _LINE_NAMES:
        prefix = f"line_{name}"
        if f"{prefix}_n_layers" not in npz:
            raise FileNotFoundError(
                f"Missing line sub-network '{name}' in {npz_path}. Re-run convert_cue_weights.py."
            )
        line_nets.append(_load_subnet(npz, prefix))
        line_wav_sels.append(np.asarray(npz[f"{prefix}_wav_selection"]))

    # Continuum sub-network
    cont_net = _load_subnet(npz, "cont")

    # Precompute batched weight arrays for fast line prediction
    nets = line_nets
    n_hidden = nets[0].n_layers - 1

    # Hidden layers: stack (16, in, out) — all same architecture
    b_W_h = tuple(np.stack([n.W[i] for n in nets]) for i in range(n_hidden))
    b_b_h = tuple(np.stack([n.b[i] for n in nets]) for i in range(n_hidden))
    b_a_h = tuple(np.stack([n.alphas[i] for n in nets]) for i in range(n_hidden))
    b_beta_h = tuple(np.stack([n.betas[i] for n in nets]) for i in range(n_hidden))

    # Output layer + PCA: pad to max dims
    max_pcas = max(n.W[n_hidden].shape[1] for n in nets)
    max_lines = max(n.pca_components.shape[1] for n in nets)

    def _pad2d(arr, target_r, target_c, fill=0.0):
        """Pad a 2D array to target shape along both dimensions."""
        return np.pad(
            arr, ((0, target_r - arr.shape[0]), (0, target_c - arr.shape[1])), constant_values=fill
        )

    def _pad1d(arr, target, fill=0.0):
        """Pad a 1D array to target shape."""
        return np.pad(arr, (0, target - arr.shape[0]), constant_values=fill)

    nn_line_wav = np.asarray(npz["nn_line_wavelength"])

    return CueWeights(
        line_nets=tuple(line_nets),
        cont_net=cont_net,
        line_names=_LINE_NAMES,
        line_wav_selections=tuple(line_wav_sels),
        sorted_line_wav=np.asarray(npz["sorted_line_wavelength"]),
        nn_line_wav=nn_line_wav,
        line_old_idx=np.asarray(npz["line_old_idx"]),
        cont_wav=np.asarray(npz["cont_wavelength"]),
        # Precomputed batched arrays
        batched_param_shifts=np.stack([n.param_shift for n in nets]),
        batched_param_scales=np.stack([n.param_scale for n in nets]),
        batched_W_hidden=b_W_h,
        batched_b_hidden=b_b_h,
        batched_alpha_hidden=b_a_h,
        batched_beta_hidden=b_beta_h,
        batched_W_out=np.stack([_pad2d(n.W[n_hidden], 256, max_pcas) for n in nets]),
        batched_b_out=np.stack([_pad1d(n.b[n_hidden], max_pcas) for n in nets]),
        batched_pca_scale=np.stack([_pad1d(n.pca_scale, max_pcas, fill=1.0) for n in nets]),
        batched_pca_shift=np.stack([_pad1d(n.pca_shift, max_pcas) for n in nets]),
        batched_pca_comp=np.stack([_pad2d(n.pca_components, max_pcas, max_lines) for n in nets]),
        batched_pca_mean=np.stack([_pad1d(n.pca_mean, max_lines) for n in nets]),
        batched_spec_scale=np.stack([_pad1d(n.log_spec_scale, max_lines, fill=1.0) for n in nets]),
        batched_spec_shift=np.stack([_pad1d(n.log_spec_shift, max_lines) for n in nets]),
        batched_n_lines=tuple(n.pca_components.shape[1] for n in nets),
        batched_sort_idx=np.argsort(nn_line_wav),
    )


# ── Neural network forward pass (pure JAX, JIT-compatible) ────────


def _speculator_activation(x: jnp.ndarray, alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """Learned Swish activation for Speculator neural network.

    Implements x * (beta + (1 - beta) * sigmoid(alpha * x)), a learnable variant
    of the Swish activation.

    Parameters
    ----------
    x : array
        Input activations.
    alpha : array
        Scaling parameter (learned during training).
    beta : array
        Offset parameter (learned during training).

    Returns
    -------
    array
        Activated output.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    """
    return x * (beta + (1.0 - beta) * jax.nn.sigmoid(alpha * x))


def _speculator_forward_pca(
    params: jnp.ndarray,
    net: SubNetWeights,
) -> jnp.ndarray:
    """Forward pass through Speculator NN, returning rescaled PCA coefficients.

    Parameters
    ----------
    params : array, shape (12,) or (batch, 12)
        Network input parameters (already converted: logq, 10**logn, etc.).
    net : SubNetWeights
        Pre-loaded network weights.

    Returns
    -------
    array, shape (..., n_pcas)
        Rescaled PCA coefficients (before PCA inverse transform).

    """
    # Normalize inputs
    x = (params - net.param_shift) / net.param_scale

    # Hidden layers with learned activation
    for i in range(net.n_layers - 1):
        x = x @ net.W[i] + net.b[i]
        x = _speculator_activation(x, net.alphas[i], net.betas[i])

    # Linear output layer
    x = x @ net.W[net.n_layers - 1] + net.b[net.n_layers - 1]

    # Rescale PCA coefficients
    return x * net.pca_scale + net.pca_shift


def _speculator_log_spectrum(
    params: jnp.ndarray,
    net: SubNetWeights,
) -> jnp.ndarray:
    """Full forward pass: parameters -> log10 spectrum.

    Parameters
    ----------
    params : array, shape (12,) or (batch, 12)
        Network input parameters.
    net : SubNetWeights
        Pre-loaded weights.

    Returns
    -------
    array, shape (..., n_wavelengths)
        Log10 spectrum (luminosity per Q_H).

    """
    pca_coeffs = _speculator_forward_pca(params, net)

    # PCA inverse transform: coeffs @ components + mean
    # (sklearn IncrementalPCA convention)
    log_spec_normalized = pca_coeffs @ net.pca_components + net.pca_mean

    # Denormalize
    return log_spec_normalized * net.log_spec_scale + net.log_spec_shift


# ── Parameter conversion: user-facing -> network input ────────────


def _logq_from_logu(
    gas_logu: jnp.ndarray, gas_logn: jnp.ndarray, log_R: float = 19.0
) -> jnp.ndarray:
    """Convert ionization parameter logU to logQ.

    Computes logQ = logU + log(4*pi*c) + 2*log(R) + logn using the Stromgren
    radius as a reference scale.

    Parameters
    ----------
    gas_logu : array
        Ionization parameter [log10(U)].
    gas_logn : array
        Gas density [log10(n_H / cm^-3)].
    log_R : float, optional
        Reference radius [log10(R / cm)]. Default: 19.0 (Stromgren radius).

    Returns
    -------
    array
        log10(Q) ionizing photon rate.

    Notes
    -----
    **JIT-compatible**: yes — simple arithmetic.

    This formula matches Cue's internal convention with R = 1e19 cm
    (approximately the Stromgren radius for typical HII regions).

    """
    return gas_logu + _LOG_4PI + 2.0 * log_R + gas_logn + _LOG_C


def _prepare_nn_params(
    ionspec_index1: jnp.ndarray,
    ionspec_index2: jnp.ndarray,
    ionspec_index3: jnp.ndarray,
    ionspec_index4: jnp.ndarray,
    ionspec_logLratio1: jnp.ndarray,
    ionspec_logLratio2: jnp.ndarray,
    ionspec_logLratio3: jnp.ndarray,
    gas_logu: jnp.ndarray,
    gas_logn: jnp.ndarray,
    gas_logz: jnp.ndarray,
    gas_logno: jnp.ndarray,
    gas_logco: jnp.ndarray,
) -> jnp.ndarray:
    """Convert user-facing parameters to Cue neural network input vector.

    Stacks 12 Cue parameters into a vector suitable for forward pass through
    Speculator networks. Converts logU -> logQ and logn -> linear density.

    Parameters
    ----------
    ionspec_index1, ionspec_index2, ionspec_index3, ionspec_index4 : array
        Ionizing spectrum slope per segment.
    ionspec_logLratio1, ionspec_logLratio2, ionspec_logLratio3 : array
        Log luminosity ratios between adjacent segment boundaries.
    gas_logu : array
        Ionization parameter [log10(U)].
    gas_logn : array
        Gas density [log10(n_H / cm^-3)].
    gas_logz : array
        Gas metallicity relative to solar [log10(Z/Zsun)].
    gas_logno, gas_logco : array
        N/O and C/O abundance offsets [log10(X/X_sun)].

    Returns
    -------
    array, shape (..., 12)
        Stacked NN input vector: [index1..4, logLratio1..3, logq, n_linear,
        logz, logno, logco].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The network expects logq (not logu) and linear density (not logn).
    This function handles the conversion via _logq_from_logu().

    """
    gas_logq = _logq_from_logu(gas_logu, gas_logn)
    gas_n_linear = 10.0**gas_logn

    return jnp.stack(
        [
            ionspec_index1,
            ionspec_index2,
            ionspec_index3,
            ionspec_index4,
            ionspec_logLratio1,
            ionspec_logLratio2,
            ionspec_logLratio3,
            gas_logq,
            gas_n_linear,
            gas_logz,
            gas_logno,
            gas_logco,
        ],
        axis=-1,
    )


# ── Line prediction ───────────────────────────────────────────────


def predict_all_lines(
    nn_params: jnp.ndarray,
    weights: CueWeights,
    gas_logq: jnp.ndarray,
    gas_logqion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict all emission line luminosities.

    Uses batched matrix multiplications across all 16 line sub-networks
    for the shared hidden layers (same architecture: 12→256→256→256),
    then individual output layers + PCA inverse transforms.

    Parameters
    ----------
    nn_params : array, shape (12,)
        NN-ready parameters [dimensionless].
    weights : CueWeights
        Pre-loaded weights.
    gas_logq : float
        log10(Q) ionization charge parameter [dimensionless].
    gas_logqion : float
        log10(Q_H) total ionizing photon rate [log10(photons/s)].

    Returns
    -------
    wavelengths : array, shape (n_lines,)
        Line wavelengths in Angstrom (sorted, vacuum) [Angstrom].
    luminosities : array, shape (n_lines,)
        Line luminosities in Lsun [Lsun].

    Notes
    -----
    **JIT-compatible**: yes — all operations use stacked ``jnp`` primitives.

    **Batching**: Hidden layers process all 16 sub-networks simultaneously via
    stacked weight arrays. Individual output layers and PCA transforms follow,
    with zero-padding handling truncated output dimensions.

    References
    ----------
    .. [1] Li et al. 2025, "Cue: A fast neural network emulator for nebular
        emission line and continuum predictions", ApJ, 986, 9 (2025).
        arXiv:2405.04598. https://doi.org/10.3847/1538-4357/ad7fe3

    """
    # Fully batched forward pass over the 16 sub-emulators using the
    # stacked/padded weight arrays prepared in load_cue_weights().
    x = (nn_params[None, :] - weights.batched_param_shifts) / weights.batched_param_scales

    for W, b, alpha, beta in zip(
        weights.batched_W_hidden,
        weights.batched_b_hidden,
        weights.batched_alpha_hidden,
        weights.batched_beta_hidden,
    ):
        x = jnp.einsum("ni,nio->no", x, W) + b
        x = x * (beta + (1.0 - beta) * jax.nn.sigmoid(alpha * x))

    pca_coeffs = jnp.einsum("ni,nio->no", x, weights.batched_W_out) + weights.batched_b_out
    pca_coeffs = pca_coeffs * weights.batched_pca_scale + weights.batched_pca_shift

    log_spec = (
        jnp.einsum("np,npl->nl", pca_coeffs, weights.batched_pca_comp) + weights.batched_pca_mean
    )
    log_spec = log_spec * weights.batched_spec_scale + weights.batched_spec_shift

    all_log_lum = []
    for i, n_lines_i in enumerate(weights.batched_n_lines):
        all_log_lum.append(log_spec[i, :n_lines_i])
    log_lum_concat = jnp.concatenate(all_log_lum, axis=-1)

    log_lum_sorted = log_lum_concat[weights.batched_sort_idx]
    wav_sorted = weights.nn_line_wav[weights.batched_sort_idx]

    # Stay in Lsun internally to avoid 10^x overflow; predict_nebular_sed
    # converts to erg/s at the boundary. The clip is the only defense against
    # NaN/inf poisoning a JAX gradient if gas_logq / gas_logqion go pathological
    # (e.g. inf from a float32 SSP overflow in fit_ionizing_spectrum, cf. #469).
    # ±50 dex is intentionally tight: physical line luminosities sit within
    # ~±20 dex of Lsun for typical galaxies, so saturation here is a load-loud
    # signal of an upstream bug. ±100 (the original) was wide enough that the
    # +51-dex `gas_logq = logU` bug fixed in #477 produced near-physical
    # silently-wrong output rather than blatantly-saturated output.
    exponent = log_lum_sorted - gas_logq + gas_logqion - _LOG_LSUN
    exponent_safe = jnp.clip(exponent, -50.0, 50.0)
    luminosities = 10.0**exponent_safe

    return wav_sorted, luminosities


# ── Continuum prediction ──────────────────────────────────────────


def predict_continuum(
    nn_params: jnp.ndarray,
    weights: CueWeights,
    gas_logq: jnp.ndarray,
    gas_logqion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict nebular continuum SED.

    Evaluates the continuum sub-network at the given parameters and rescales
    from log10(Lsun/Hz/Q_H) to erg/s/Hz via ionization parameter normalization.

    Parameters
    ----------
    nn_params : array, shape (12,)
        NN-ready parameters [dimensionless].
    weights : CueWeights
        Pre-loaded weights.
    gas_logq : float
        log10(Q) ionization charge parameter [dimensionless].
    gas_logqion : float
        log10(Q_H) for normalization [log10(photons/s)].

    Returns
    -------
    wavelength : array, shape (n_wave,)
        Continuum wavelength grid in Angstrom (sorted) [Angstrom].
    luminosity : array, shape (n_wave,)
        Nebular continuum in erg/s/Hz [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Conversion**: Internal output is log10(Lsun/Hz/Q_H). Rescaled to
    Lsun/Hz via Q_H, then to erg/s/Hz via L_sun constant (3.839e33 erg/s).

    References
    ----------
    .. [1] Li et al. 2025, "Cue: A fast neural network emulator for nebular
        emission line and continuum predictions", ApJ, 986, 9 (2025).
        arXiv:2405.04598. https://doi.org/10.3847/1538-4357/ad7fe3

    """
    log_spec = _speculator_log_spectrum(nn_params, weights.cont_net)

    # Sort by wavelength (Cue does wavind_sorted = argsort(cont_wavelength))
    #
    # ``jnp.asarray`` at use, because ``cont_wav`` is NumPy when it comes from
    # ``load_cue_weights`` and a tracer when it arrives threaded as
    # ``template_data`` (#1631). ``jnp.argsort`` stages out under jit, so
    # ``sort_idx`` is traced either way -- and indexing a *NumPy* array with a
    # tracer calls ``__array__()`` on it and raises
    # ``TracerArrayConversionError``.
    cont_wav = jnp.asarray(weights.cont_wav)
    sort_idx = jnp.argsort(cont_wav)
    log_spec_sorted = log_spec[sort_idx]
    wav_sorted = cont_wav[sort_idx]

    # Convert from log10(Lsun/Hz/Q_H) to Lsun/Hz:
    # Internal computation stays in Lsun/Hz to avoid exponent overflow;
    # converted to erg/s/Hz at predict_nebular_sed return. Clip tightened
    # from ±100 to ±50 dex in this revision — see predict_all_lines for the
    # full rationale (#477 follow-up).
    exponent = log_spec_sorted - gas_logq + gas_logqion - _LOG_LSUN
    luminosity = 10.0 ** jnp.clip(exponent, -50.0, 50.0)

    # Zero out wavelengths below Lyman limit (Cue convention)
    luminosity = jnp.where(wav_sorted > 911.6, luminosity, 0.0)

    return wav_sorted, luminosity


# ── Ionizing-spectrum warnings ────────────────────────────────────


class CueWNESSPWarning(UserWarning):
    """Deprecated alias retained for backwards compatibility.

    Previously emitted as a warning when wNE SSPs were detected at
    ``CueBackend`` construction; now superseded by ``CueWNESSPError``,
    which raises immediately. Kept so any user code that filters this
    warning class continues to import cleanly.
    """


class CueWNESSPError(ValueError):
    """Raised when CueBackend is constructed with a wNE (with-Nebular-Emission) SSP.

    SSP grids labeled 'wNE' have nebular emission baked in: the ionizing
    photons have already been absorbed by an internal nebular layer, so the
    SSP spectrum reports ``log10(Q_H) ~ 0`` instead of the physical 47–50.
    Feeding such an SSP to Cue produces line luminosities that are
    under-predicted by 4–7 dex — silently — because Cue infers Q_H from the
    SSP rather than receiving it explicitly.

    Detection
    ---------
    Two independent checks run during ``CueBackend.__init__`` when
    ``ssp_data`` is set. They differ in kind, and so does their bypass:

    1. **Declaration** — ``ssp_data.nebular == "included"``, resolved by
       :func:`load_ssp_data` from the ``nebular_included`` HDF5 attribute
       or the ``wNE`` filename convention. Checked first, because the
       retained-LyC wNE class keeps a physical-looking ``Q_H`` that no
       heuristic can catch. **Not bypassable** (#1579): a declaration has
       no false-positive mode, and Cue on top of baked-in nebular emission
       is double-counting in every case.
    2. **Q_H heuristic** — the maximum ``log10(Q_H)`` across SSP bins
       younger than 10 Myr falling outside ``[44, 52]``, on either side of
       the physical 47-50 range for bare stellar populations. This one is a
       suspicion with routine false positives (synthetic test grids read
       ~62), so it is downgradable to ``CueWNESSPWarning`` by setting
       ``TENGRI_ALLOW_WNE_CUE=1``.

    Resolution
    ----------
    Pick one of:

    1. **Use a bare-stellar SSP** (no baked-in nebular). Examples in
       ``data/``: ``fsps_prsc_miles_chabrier.h5``, ``fsps_mist_c3k_a_chabrier.h5``.
       The hosted catalog at https://halos.as.arizona.edu/suchethacooray/
       ssp-spectra/ ships only bare-stellar SSPs.
    2. **Keep the SSP and drop** ``neb={'type': 'cue'}`` — a wNE grid's
       baked-in nebular backend already models the lines.
    3. **Pass ssp_data=None** to ``CueBackend`` and provide Q_H externally.
       Suitable when you have your own ionizing-spectrum source.
    """


# log10(Q_H) band outside which an SSP is suspected to be wNE.
# Normal bare O/B stars at < 10 Myr give log10(Q_H) ~ 47–50 per Msun.
# wNE SSPs fail in BOTH directions: grids with the LyC pre-absorbed report
# Q_H ≈ 0 (log10 stored as –99, caught by the lower bound), while grids
# that keep the baked-in nebular continuum corrupt the ionizing power-law
# fit UPWARD (observed: log10(Q_H) ≈ 62 for ssp_prsc_*_wNE_* files — the
# nebular continuum in the fit window masquerades as ionizing flux).
# The band gives > 2 dex headroom on each side of the physical range.
_WNE_LOGQH_THRESHOLD: float = 44.0
_WNE_LOGQH_UPPER: float = 52.0

# Age cutoff (log10 yr) for young SSP bins used in the wNE check.
_YOUNG_LOG_AGE_MAX: float = 7.0  # 10 Myr


# ── Cue backend: NN emulator of CLOUDY's nebular emission ─────────


class CueBackend:
    """Cue neural net emulator backend for nebular emission.

    Loads pre-trained Speculator network weights and predicts nebular
    emission lines and continuum as a function of 12 parameters (7
    ionizing spectrum shape + 5 gas properties).

    Unlike the CLOUDY grid backend, Cue does NOT depend on SSP age bins
    or stellar metallicity for the nebular prediction itself. Instead,
    the ionizing spectrum shape and Q_H are provided directly (either
    from power-law fitting of the SSP or as free parameters).

    Parameters
    ----------
    weights_path : str
        Path to ``cue_weights.npz`` file containing pre-trained network weights.
    ssp_data : object, optional
        SSP data container with fields ``ssp_wave``, ``ssp_flux``,
        ``ssp_lgmet``, ``ssp_lg_age_gyr``. If provided, ionizing spectrum
        parameters and Q_H are precomputed for all (metallicity, age) bins
        and cached for fast interpolation. Default: None (no precomputation).
    default_gas_logqion : float, optional
        Default log10(Q_H) normalization [log10(photons/s)] when not specified
        per call. Default: 49.1 (typical for young stellar populations).

    Notes
    -----
    **JIT-compatible**: Methods return JAX arrays suitable for JIT compilation.
    Neural network forward passes are pure functions with no side effects.

    **Precomputation**: If ``ssp_data`` is provided, ionizing spectrum
    parameters are precomputed once at init time via
    ``precompute_ionizing_params_table``. At inference time, values are
    interpolated bilinearly on the (metallicity, age) grid for fast
    gradient evaluation. Warnings are emitted if the SSP appears to have
    baked-in nebular emission (wNE), which violates Cue's assumptions.

    **Calling conventions**: Two modes are supported—

    1. **High-level** (CloudyGridBackend compatible): Pass ``ssp_weights``,
       ``ssp_log_ages_yr``, ``log_z`` to ``predict_nebular_*`` methods.
       Q_H and ionizing spectrum are derived from the SSP internally.
    2. **Low-level** (direct Cue parameters): Pass ``gas_logu``, ``gas_logz``,
       ``gas_logqion``, and ``ionspec_*`` explicitly. Useful for fitting
       these parameters as free parameters or using external ionizing spectra.

    """

    def __init__(
        self,
        weights_path: str,
        ssp_data=None,
        default_gas_logqion: float = 49.1,
    ) -> None:
        self.name = "cue"
        self.has_free_params = True
        self.has_continuum = True
        self.weights = load_cue_weights(weights_path)
        self.default_gas_logqion = default_gas_logqion

        # Cache sorted wavelength arrays. NumPy, for the reason in
        # ``load_cue_weights``: this backend is memoized in a module-level
        # global (``nlr_cloudy._CUE_AGN_BACKEND``), so anything built here
        # inside a jit trace would be handed to the *next* trace (#1631).
        self._line_sort_idx = np.argsort(self.weights.nn_line_wav)
        self._cont_sort_idx = np.argsort(self.weights.cont_wav)

        # Precompute ionizing spectrum parameters from SSP if provided.
        # These serve as defaults when ionspec params are not explicitly
        # specified. Users can override by setting ionspec_index1..4 and
        # ionspec_logLratio1..3 as free params in Parameters.
        self._ionspec_table = None
        self._logqion_table = None
        self._seglum_table = None
        self._ssp_lgmet = None
        self._ssp_log_age_yr = None
        if ssp_data is not None:
            self._precompute_ionizing_params(ssp_data)

    def _precompute_ionizing_params(self, ssp_data) -> None:
        """Precompute ionizing spectrum parameters for all SSP (met, age)."""
        import numpy as np

        from tengri.components.nebular.ionizing_spectrum import precompute_ionizing_params_table

        # Metadata check (#1014): a grid flagged nebular-included is refused
        # outright, BEFORE the Q_H heuristic below — the retained-LyC wNE
        # class keeps its ionizing continuum (measured young-bin log Q_H =
        # 46.91, identical to the bare parent grid), so no physics heuristic
        # can catch it. The flag comes from the ``nebular_included`` HDF5
        # attribute or the wNE filename convention via ``load_ssp_data``.
        if getattr(ssp_data, "nebular", "unknown") == "included":
            # Deliberately NOT bypassable by TENGRI_ALLOW_WNE_CUE (#1579).
            # That switch exists for the Q_H *heuristic* below, whose false
            # positives are routine — the synthetic fixtures trip it, which
            # is why tests/conftest.py sets it suite-wide. This branch is a
            # *declaration*, not a suspicion: it reads the nebular_included
            # attribute (or the wNE filename), and there is no science case
            # for Cue on top of baked-in nebular emission. Wiring both to one
            # switch let the fixture accommodation silently license real wNE
            # grids, and the N=8 PSD pilot fit a double-counted model for it.
            raise CueWNESSPError(
                "CueBackend received an SSP flagged nebular-included (wNE): "
                "nebular continuum and lines are already baked into the "
                "templates, so adding Cue on top double-counts nebular "
                "emission.\n"
                "\n"
                "Fix (one of):\n"
                "  1. Use a bare-stellar SSP (e.g. fsps_prsc_miles_chabrier.h5;\n"
                "     see tengri.data.download_ssp / list_remote_ssps).\n"
                "  2. Keep this SSP and drop neb={'type': 'cue'} — the\n"
                "     baked-in nebular backend already models the lines."
            )

        result = precompute_ionizing_params_table(
            np.array(ssp_data.ssp_wave),
            np.array(ssp_data.ssp_flux),
            np.array(ssp_data.ssp_lgmet),
            ssp_log_age_yr=np.array(ssp_data.ssp_lg_age_gyr) + 9.0,
        )
        self._ionspec_table = jnp.array(result["ionspec_table"])
        self._logqion_table = jnp.array(result["logqion_table"])
        self._seglum_table = jnp.array(result["seglum_table"])
        self._ssp_lgmet = jnp.array(ssp_data.ssp_lgmet)
        self._ssp_log_age_yr = jnp.array(ssp_data.ssp_lg_age_gyr) + 9.0

        # wNE SSP detection: young stellar populations (< 10 Myr) in a bare SSP
        # produce log10(Q_H) ~ 47–50.  If the maximum across all metallicities for
        # young bins is below the threshold, the SSP likely has baked-in nebular
        # emission (wNE), making Cue's ionizing-spectrum fit unreliable.
        log_age_yr_np = np.array(self._ssp_log_age_yr)
        young_mask = log_age_yr_np <= _YOUNG_LOG_AGE_MAX
        if young_mask.any():
            logqion_np = np.array(self._logqion_table)  # (n_met, n_age)
            max_logqion_young = float(logqion_np[:, young_mask].max())
            if not (_WNE_LOGQH_THRESHOLD <= max_logqion_young <= _WNE_LOGQH_UPPER):
                direction = (
                    "well below the ~47-50 floor for bare stellar populations "
                    "— the ionizing photons were pre-absorbed by a baked-in "
                    "nebular layer. Cue's ionizing-spectrum fit will "
                    "under-predict line luminosities by 4-7 dex"
                    if max_logqion_young < _WNE_LOGQH_THRESHOLD
                    else "far above the physical ~47-50 range for bare stellar "
                    "populations — baked-in nebular continuum in the fit "
                    "window masquerades as ionizing flux, corrupting Cue's "
                    "power-law fit"
                )
                msg = (
                    "CueBackend received a wNE (with-Nebular-Emission) SSP. "
                    f"Max log10(Q_H) for bins younger than 10 Myr is "
                    f"{max_logqion_young:.1f}, {direction}.\n"
                    "\n"
                    "Fix (one of):\n"
                    "  1. Use a bare-stellar SSP. The four recipes that\n"
                    "     use Cue (star_forming_photometry, quiescent_z0,\n"
                    "     stochastic_sfh_jwst, agn_panchromatic) need a\n"
                    "     file like fsps_prsc_miles_chabrier.h5. Download with:\n"
                    "         from tengri.data import download_ssp\n"
                    "         path = download_ssp('fsps_prsc_miles_chabrier.h5')\n"
                    "     then  load_ssp_data(str(path)). See\n"
                    "     tengri.data.list_remote_ssps() for the full catalog.\n"
                    "  2. Pass ssp_data=None to CueBackend and provide Q_H\n"
                    "     externally.\n"
                    "  3. Bypass for testing: set TENGRI_ALLOW_WNE_CUE=1\n"
                    "     (downgrades to a warning)."
                )
                if os.environ.get("TENGRI_ALLOW_WNE_CUE"):
                    warnings.warn(msg, CueWNESSPWarning, stacklevel=3)
                else:
                    raise CueWNESSPError(msg)

    def get_ionizing_params_at(
        self,
        log_z: float,
        log_age_yr: float,
    ) -> tuple[jnp.ndarray, float]:
        """Get precomputed ionizing params at (Z, age) via interpolation.

        Retrieves cached ionizing spectrum parameters and Q_H via bilinear
        interpolation on the SSP (metallicity, age) grid. Returns (None, None)
        if precomputation was not requested at initialization.

        Parameters
        ----------
        log_z : float
            Target stellar metallicity log10(Z) [log10(Z)].
        log_age_yr : float
            Target stellar age log10(age/yr) [log10(yr)].

        Returns
        -------
        ionspec_7 : ndarray, shape (7,) or None
            Interpolated ionizing spectrum parameters:
            [ionspec_index1, ionspec_index2, ionspec_index3, ionspec_index4,
            ionspec_logLratio1, ionspec_logLratio2, ionspec_logLratio3]
            [dimensionless] or None if not precomputed.
        logqion : float or None
            Interpolated log10(Q_H) [log10(photons/s)] or None if not precomputed.

        Notes
        -----
        **JIT-compatible**: yes — bilinear interpolation uses ``jnp`` primitives.

        **Caching**: Results are precomputed once at CueBackend.__init__ if
        ``ssp_data`` is provided. Subsequent calls perform fast interpolation
        on the cached tables without recomputing power-law fits.

        """
        if self._ionspec_table is None:
            return None, None

        from tengri.components.nebular.ionizing_spectrum import interpolate_ionizing_params

        return interpolate_ionizing_params(
            self._ionspec_table,
            self._logqion_table,
            self._ssp_lgmet,
            self._ssp_log_age_yr,
            log_z,
            log_age_yr,
        )

    # ── High-level entry points — same signatures as CloudyGridBackend ────

    # Default ionizing spectrum shape (young starburst)
    _IONSPEC_DEFAULTS: ClassVar[dict] = dict(
        ionspec_index1=19.7,
        ionspec_index2=5.3,
        ionspec_index3=1.6,
        ionspec_index4=0.6,
        ionspec_logLratio1=3.9,
        ionspec_logLratio2=0.01,
        ionspec_logLratio3=0.2,
    )

    def _resolve_cue_params(
        self,
        ssp_weights=None,
        ssp_log_ages_yr=None,
        log_z=None,
        neb_logU=-3.0,
        neb_logZ_gas=None,
        gas_logu=None,
        gas_logn=2.0,
        gas_logz=None,
        gas_logno=0.0,
        gas_logco=0.0,
        gas_logqion=None,
        ionspec_index1=None,
        ionspec_index2=None,
        ionspec_index3=None,
        ionspec_index4=None,
        ionspec_logLratio1=None,
        ionspec_logLratio2=None,
        ionspec_logLratio3=None,
        **_kwargs,
    ) -> dict:
        """Resolve Cue params from high-level or low-level inputs.

        High-level (ssp_weights provided): derives Q_H, ionspec, gas_logz
        from SSP.  Explicit overrides take precedence over derived values.
        Low-level (ssp_weights=None): fills from defaults.

        Returns a flat dict with all 12 Cue params + gas_logqion.
        """
        if ssp_weights is not None:
            derived = self._compute_weighted_cue_params(
                ssp_weights,
                ssp_log_ages_yr,
                log_z,
                neb_logU=neb_logU,
                neb_logZ_gas=neb_logZ_gas,
                gas_logn=gas_logn,
                gas_logno=gas_logno,
                gas_logco=gas_logco,
            )
        else:
            derived = {}

        # Check if we're about to silently use the built-in young-starburst defaults
        # This happens when: (1) no SSP data at construction time, (2) no SSP weights passed,
        # and (3) no explicit ionspec overrides provided.
        import sys

        if (
            self._ionspec_table is None
            and ssp_weights is None
            and ionspec_index1 is None
            and ionspec_index2 is None
            and ionspec_index3 is None
            and ionspec_index4 is None
            and ionspec_logLratio1 is None
            and ionspec_logLratio2 is None
            and ionspec_logLratio3 is None
        ):
            # Access the module-level flag via sys.modules to avoid closure issues
            this_module = sys.modules[__name__]
            if not getattr(this_module, "_IONSPEC_DEFAULT_WARNED", False):
                msg = (
                    "CueBackend: using built-in young-starburst ionizing spectrum defaults "
                    "(index1=19.7, index2=5.3, index3=1.6, index4=0.6, "
                    "logLratio1=3.9, logLratio2=0.01, logLratio3=0.2) because ssp_data was not "
                    "provided and no ionspec_* override was passed. For an older / quiescent "
                    "population this will overestimate emission lines. Either (a) pass ssp_data "
                    "at CueBackend construction, or (b) override ionspec_index{1..4} and "
                    "ionspec_logLratio{1..3} explicitly. To suppress: "
                    "warnings.filterwarnings('ignore', category=UserWarning, "
                    "module='tengri.components.nebular.cue')"
                )
                warnings.warn(msg, UserWarning, stacklevel=3)
                this_module._IONSPEC_DEFAULT_WARNED = True

        def _pick(name, explicit, default):
            """Return explicit value, derived value, or default in priority order."""
            if explicit is not None:
                return explicit
            if name in derived:
                return derived[name]
            return default

        return dict(
            gas_logu=_pick("gas_logu", gas_logu, neb_logU),
            gas_logn=gas_logn,
            gas_logz=_pick("gas_logz", gas_logz, 0.0),
            gas_logno=gas_logno,
            gas_logco=gas_logco,
            gas_logqion=_pick("gas_logqion", gas_logqion, self.default_gas_logqion),
            ionspec_index1=_pick("ionspec_index1", ionspec_index1, 19.7),
            ionspec_index2=_pick("ionspec_index2", ionspec_index2, 5.3),
            ionspec_index3=_pick("ionspec_index3", ionspec_index3, 1.6),
            ionspec_index4=_pick("ionspec_index4", ionspec_index4, 0.6),
            ionspec_logLratio1=_pick("ionspec_logLratio1", ionspec_logLratio1, 3.9),
            ionspec_logLratio2=_pick("ionspec_logLratio2", ionspec_logLratio2, 0.01),
            ionspec_logLratio3=_pick("ionspec_logLratio3", ionspec_logLratio3, 0.2),
        )

    def _forward_lines(
        self,
        p: dict,
        cloudyfsps_only=True,
        neb_fesc=0.0,
        neb_fesc_lya=0.0,
        neb_fdust=0.0,
        template_data=None,
    ):
        """Low-level line prediction from resolved param dict."""
        weights = template_data if template_data is not None else self.weights
        nn_params = _prepare_nn_params(
            jnp.asarray(p["ionspec_index1"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_index2"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_index3"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_index4"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_logLratio1"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_logLratio2"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_logLratio3"], dtype=jnp.float32),
            jnp.asarray(p["gas_logu"], dtype=jnp.float32),
            jnp.asarray(p["gas_logn"], dtype=jnp.float32),
            jnp.asarray(p["gas_logz"], dtype=jnp.float32),
            jnp.asarray(p["gas_logno"], dtype=jnp.float32),
            jnp.asarray(p["gas_logco"], dtype=jnp.float32),
        )
        gas_logq = _logq_from_logu(
            jnp.asarray(p["gas_logu"], dtype=jnp.float32),
            jnp.asarray(p["gas_logn"], dtype=jnp.float32),
        )
        wav, lum = predict_all_lines(
            nn_params,
            weights,
            gas_logq,
            jnp.asarray(p["gas_logqion"], dtype=jnp.float32),
        )
        # Apply nebular emission scaling from ionizing photon loss.
        # Following CIGALE nebular.py, use the k-factor (Inoue 2011) which
        # accounts for both escape fraction and dust absorption via the
        # recombination coefficient ratio alpha_1 / alpha_B.
        k = lyc_dust_escape_factor(neb_fesc, neb_fdust)
        lum = lum * k
        # Ly-alpha special handling. All lines (incl. Ly-alpha) are already
        # scaled by the general ionization-budget factor ``k`` above. Ly-alpha
        # is *additionally* suppressed by its own resonant escape/destruction
        # fraction ``neb_fesc_lya``, so the surviving Ly-alpha is
        # ``L_orig · k · (1 - neb_fesc_lya)``.
        #
        # The previous code multiplied by ``(1 - neb_fesc_lya) / (1 - neb_fesc)``,
        # which divided out the general suppression: as ``neb_fesc → 1`` that
        # ratio diverges (``1 / 1e-10``) and *amplified* Ly-alpha by ~60×
        # instead of suppressing it (P-11 BUG: lines not suppressed at fesc=1).
        # It was also unphysical — with all ionizing photons escaped (k → 0),
        # Ly-alpha would have survived at ``L_orig · (1 - neb_fesc_lya)``.
        lya_idx = jnp.argmin(jnp.abs(wav - 1215.67))
        lya_scale = 1.0 - neb_fesc_lya
        lum = lum.at[lya_idx].multiply(lya_scale)
        if cloudyfsps_only:
            old_idx = weights.line_old_idx
            return wav[old_idx], lum[old_idx]
        return wav, lum

    def _forward_continuum(self, p: dict, template_data=None):
        """Low-level continuum prediction from resolved param dict."""
        weights = template_data if template_data is not None else self.weights
        nn_params = _prepare_nn_params(
            jnp.asarray(p["ionspec_index1"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_index2"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_index3"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_index4"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_logLratio1"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_logLratio2"], dtype=jnp.float32),
            jnp.asarray(p["ionspec_logLratio3"], dtype=jnp.float32),
            jnp.asarray(p["gas_logu"], dtype=jnp.float32),
            jnp.asarray(p["gas_logn"], dtype=jnp.float32),
            jnp.asarray(p["gas_logz"], dtype=jnp.float32),
            jnp.asarray(p["gas_logno"], dtype=jnp.float32),
            jnp.asarray(p["gas_logco"], dtype=jnp.float32),
        )
        gas_logq = _logq_from_logu(
            jnp.asarray(p["gas_logu"], dtype=jnp.float32),
            jnp.asarray(p["gas_logn"], dtype=jnp.float32),
        )
        return predict_continuum(
            nn_params,
            weights,
            gas_logq,
            jnp.asarray(p["gas_logqion"], dtype=jnp.float32),
        )

    def _compute_weighted_cue_params(
        self,
        ssp_weights: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        gas_logn: float = 2.0,
        gas_logno: float = 0.0,
        gas_logco: float = 0.0,
    ) -> dict:
        """Derive Cue parameters from SSP weights.

        Computes total Q_H by summing mass-weighted per-Msun Q_H over
        young age bins, and picks the ionizing spectrum shape from the
        dominant (highest Q_H × weight) age bin.

        Parameters
        ----------
        ssp_weights : array (n_age,)
            CSP mass weights (Msun per age bin).
        ssp_log_ages_yr : array (n_age,)
            log10(age/yr) of SSP age bins.
        log_z : float
            Stellar metallicity log10(Z) (absolute).
        neb_logU : float
            Ionization parameter log10(U).
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) (absolute). None = tie to stellar.
        gas_logn, gas_logno, gas_logco : float
            Cue gas properties (defaults match Cue paper).

        Returns
        -------
        dict
            Ready-to-use kwargs for the low-level Cue predict methods.

        """
        if self._ionspec_table is None:
            raise RuntimeError(
                "CueBackend requires ssp_data at init for the high-level "
                "interface.  Pass ssp_data= to the constructor."
            )

        # Vectorized, JIT-safe computation over all age bins.
        # Use jnp.where masks instead of boolean fancy indexing.
        young_mask = ssp_log_ages_yr <= _MAX_NEB_LOG_AGE  # (n_age,) bool

        # Vectorize interpolate_ionizing_params over all ages at once.
        # ionspec_table is (n_met, n_age, 7), logqion_table is (n_met, n_age).
        # We need logqion at each age bin for the given metallicity.
        from tengri.components.nebular.ionizing_spectrum import (
            interpolate_ionizing_params,
            interpolate_ionizing_seglum,
        )

        # Vectorize over age axis: get (ionspec_7, logqion) for each age
        # ionspec_all: (n_age, 7), logqion_all: (n_age,)
        ionspec_all, logqion_all = jax.vmap(
            lambda log_age_yr: interpolate_ionizing_params(
                self._ionspec_table,
                self._logqion_table,
                self._ssp_lgmet,
                self._ssp_log_age_yr,
                log_z,
                log_age_yr,
            )
        )(ssp_log_ages_yr)

        # Q_H via log-domain logsumexp (#1206): log10( sum_a w_a * 10**logqion_a )
        # Prevents float32 overflow (10**52 -> inf). Old linear arithmetic overflows
        # on young populations (logqion ~ 44-52 in f32 range), silently zeroing
        # emission via the #1001 isfinite guard. The log-domain formulation is
        # exact in f64 and finite-and-correct in f32.
        # #1001 defense: non-finite table rows are sanitized (replaced with 0.0 log)
        # so they don't poison the exponent (0*exp(nan) is still nan).
        finite_q = jnp.isfinite(logqion_all)  # (n_age,)
        valid = young_mask & (ssp_weights > 0) & finite_q  # (n_age,) bool
        logq_safe = jnp.where(finite_q, logqion_all, 0.0)
        w_valid = jnp.where(valid, ssp_weights, 0.0)
        lse = logsumexp(LN10 * logq_safe, b=w_valid)  # ln( sum w*10**logqion )
        any_valid = jnp.any(valid)
        total_logqion = jnp.where(any_valid, lse / LN10, -99.0)

        # ── Effective ionizing-spectrum shape via per-segment max-offset (#1018) ──
        # Cue is trained on the *time-averaged* ionizing spectrum of the whole
        # population (see the module header), so the shape must combine every
        # ionizing age bin — not the single argmax-dominant one. The old
        # ``i7 = ionspec_all[jnp.argmax(weighted_qh)]`` made the forward
        # DISCONTINUOUS in metallicity and SFH (the dominant bin flips → [OIII]
        # steps ~33 %) and, because ``argmax`` has no gradient and ``ionspec_all``
        # does not depend on ``ssp_weights``, it forced d(shape)/d(SFH) ≡ 0 —
        # silently starving gradient-based inference.
        #
        # Combine the way the physics does, now in log-domain for float32 safety.
        # The 7 params describe a 4-segment broken power law: 4 slopes + 3 log-
        # ratios of the INTEGRATED segment luminosities. Across populations:
        #   * segment luminosities ADD linearly     → L_k = Σ_a w_a L_k,a
        #   * the slope of a sum of power laws is the per-segment luminosity-
        #     weighted mean          → α_k = Σ_a (w_a L_k,a / L_k) · α_k,a
        #   * the log-ratios follow as diff(log10 L_k)
        # Averaging the log-ratios directly (a geometric mean where an arithmetic
        # one is required) biases [OIII] by ~12 % — worse than the argmax it would
        # replace. This rule lands within ~1 % of re-fitting the true composite
        # spectrum, and is smooth + differentiable.
        log_seglum_all = jax.vmap(
            lambda log_age_yr: interpolate_ionizing_seglum(
                self._seglum_table,
                self._ssp_lgmet,
                self._ssp_log_age_yr,
                log_z,
                log_age_yr,
            )
        )(ssp_log_ages_yr)  # (n_age, 4)

        # Per-segment max-offset scaling for log-domain arithmetic (#1206).
        # For segment k: seg_w[a,k] = w_a * 10**log_seglum[a,k]. Factor out
        # the per-segment peak m_k so 10**(log_seg_w - m_k) stays O(<=1);
        # alpha (a ratio) is m_k-invariant, and log10(seg_tot_k) = m_k +
        # log10(sum 10**(log_seg_w - m_k)).
        w_pos = ssp_weights > 0  # (n_age,)
        log_w = jnp.where(w_pos, jnp.log10(jnp.where(w_pos, ssp_weights, 1.0)), -jnp.inf)
        finite_seg = jnp.isfinite(log_seglum_all)  # (n_age,4)
        seg_valid = young_mask[:, None] & w_pos[:, None] & finite_seg
        log_seg_w = jnp.where(
            seg_valid, log_w[:, None] + jnp.where(finite_seg, log_seglum_all, 0.0), -jnp.inf
        )  # (n_age,4)
        m = jnp.max(log_seg_w, axis=0)  # (4,); -inf if a segment is fully invalid
        m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
        seg_w_scaled = pow10(log_seg_w - m_safe[None, :])  # (n_age,4); pow10(-inf)=0
        seg_tot_scaled = jnp.sum(seg_w_scaled, axis=0)  # (4,)
        seg_pos = seg_tot_scaled > 0
        alpha_eff = jnp.sum(seg_w_scaled * ionspec_all[:, :4], axis=0) / jnp.where(
            seg_pos, seg_tot_scaled, 1.0
        )  # (4,) — m_k cancels in the ratio
        log_seg_tot = jnp.where(
            seg_pos, m_safe + jnp.log10(jnp.where(seg_pos, seg_tot_scaled, 1.0)), -jnp.inf
        )  # (4,)
        logLratio_eff = jnp.diff(log_seg_tot)  # (3,)
        i7_weighted = jnp.concatenate([alpha_eff, logLratio_eff])

        # Degenerate (no valid ionizing bin): dominant-bin fallback, picked in LOG
        # space so the index selection never materializes 10**logqion.
        # total_logqion == -99 already zeros emission.
        log_terms = jnp.where(valid, logq_safe + log_w, -jnp.inf)  # (n_age,)
        i7 = jnp.where(any_valid, i7_weighted, ionspec_all[jnp.argmax(log_terms)])

        # Gas metallicity: convert absolute → Z/Zsun for Cue
        gas_logz = neb_logZ_gas if neb_logZ_gas is not None else log_z
        gas_logz_rel = gas_logz - _LOG10_ZSUN

        return dict(
            gas_logu=neb_logU,
            gas_logn=gas_logn,
            gas_logz=gas_logz_rel,
            gas_logno=gas_logno,
            gas_logco=gas_logco,
            gas_logqion=total_logqion,
            ionspec_index1=i7[0],
            ionspec_index2=i7[1],
            ionspec_index3=i7[2],
            ionspec_index4=i7[3],
            ionspec_logLratio1=i7[4],
            ionspec_logLratio2=i7[5],
            ionspec_logLratio3=i7[6],
        )

    def predict_nebular_line_luminosities(
        self,
        ssp_weights: jnp.ndarray | None = None,
        ssp_log_ages_yr: jnp.ndarray | None = None,
        log_z: float | None = None,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 0.0,
        neb_fdust: float = 0.0,
        cloudyfsps_only: bool = True,
        # Cue-specific overrides (bypass SSP-derived params)
        gas_logu: float | None = None,
        gas_logn: float = 2.0,
        gas_logz: float | None = None,
        gas_logno: float = 0.0,
        gas_logco: float = 0.0,
        gas_logqion: float | None = None,
        ionspec_index1: float | None = None,
        ionspec_index2: float | None = None,
        ionspec_index3: float | None = None,
        ionspec_index4: float | None = None,
        ionspec_logLratio1: float | None = None,
        ionspec_logLratio2: float | None = None,
        ionspec_logLratio3: float | None = None,
        template_data: Any | None = None,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict emission line luminosities.

        Supports two calling conventions:

        1. **High-level** (CloudyGridBackend-compatible): pass
           ``ssp_weights``, ``ssp_log_ages_yr``, ``log_z``.
           Q_H and ionizing spectrum are derived from the SSP.
        2. **Low-level** (direct Cue params): pass ``gas_logu``,
           ``gas_logz``, ``gas_logqion``, ``ionspec_*`` explicitly.

        Parameters
        ----------
        ssp_weights : array or None
            CSP mass weights.  If provided, activates high-level mode.
        ssp_log_ages_yr : array or None
            log10(age/yr) of SSP age bins.
        log_z : float or None
            Stellar metallicity log10(Z) (absolute).
        neb_logU : float
            Ionization parameter log10(U). Default -3.0.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) (absolute). None = tie to stellar.
        neb_fesc : float
            Escape fraction [0, 1].
        neb_fesc_lya : float
            Ly-alpha escape fraction [0, 1].
        neb_fdust : float
            Dust-absorption fraction of ionizing photons in HII regions [0, 1].
        cloudyfsps_only : bool
            If True, return 128 CLOUDY/FSPS-matched lines.
        gas_logu, gas_logn, gas_logz, gas_logno, gas_logco : float
            Cue gas params (low-level). Override high-level derivation.
        gas_logqion : float or None
            log10(Q_H) total. Override high-level derivation.
        ionspec_* : float or None
            Ionizing spectrum shape. Override high-level derivation.

        Returns
        -------
        wavelengths : ndarray, shape (n_lines,)
            Emission line wavelengths (vacuum) [Angstrom].
        luminosities : ndarray, shape (n_lines,)
            Emission line luminosities [erg/s].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        **High-level mode**: When ``ssp_weights``, ``ssp_log_ages_yr``,
        and ``log_z`` are provided, Q_H and ionizing spectrum parameters
        are interpolated from precomputed SSP tables (or computed on-the-fly
        if precomputation was not requested). This mode is compatible with
        the CloudyGridBackend interface.

        **Low-level mode**: When ``gas_logu``, ``gas_logqion``, and
        ``ionspec_*`` are specified explicitly, they take precedence
        over SSP-derived values. Useful for direct parameter fitting.

        **Escape fraction**: When ``neb_fesc > 0`` or ``neb_fesc_lya > 0``,
        line luminosities are suppressed proportionally. This approximation
        assumes optically thin escape (energy-conserving).

        """
        p = self._resolve_cue_params(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            gas_logu=gas_logu,
            gas_logn=gas_logn,
            gas_logz=gas_logz,
            gas_logno=gas_logno,
            gas_logco=gas_logco,
            gas_logqion=gas_logqion,
            ionspec_index1=ionspec_index1,
            ionspec_index2=ionspec_index2,
            ionspec_index3=ionspec_index3,
            ionspec_index4=ionspec_index4,
            ionspec_logLratio1=ionspec_logLratio1,
            ionspec_logLratio2=ionspec_logLratio2,
            ionspec_logLratio3=ionspec_logLratio3,
        )
        wav, lum = self._forward_lines(
            p,
            cloudyfsps_only=cloudyfsps_only,
            neb_fesc=neb_fesc,
            neb_fesc_lya=neb_fesc_lya,
            neb_fdust=neb_fdust,
            template_data=template_data,
        )
        return wav, lum * _LSUN_ERG

    def predict_nebular_continuum(
        self,
        ssp_weights: jnp.ndarray | None = None,
        ssp_log_ages_yr: jnp.ndarray | None = None,
        log_z: float | None = None,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fdust: float = 0.0,
        gas_logu: float | None = None,
        gas_logn: float = 2.0,
        gas_logz: float | None = None,
        gas_logno: float = 0.0,
        gas_logco: float = 0.0,
        gas_logqion: float | None = None,
        ionspec_index1: float | None = None,
        ionspec_index2: float | None = None,
        ionspec_index3: float | None = None,
        ionspec_index4: float | None = None,
        ionspec_logLratio1: float | None = None,
        ionspec_logLratio2: float | None = None,
        ionspec_logLratio3: float | None = None,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict nebular continuum SED.

        Supports the same high-level and low-level calling conventions
        as predict_nebular_line_luminosities. Returns the nebular continuum
        spectrum on the Cue wavelength grid.

        Parameters
        ----------
        ssp_weights : array or None
            CSP mass weights. If provided, activates high-level mode.
        ssp_log_ages_yr : array or None
            log10(age/yr) of SSP age bins.
        log_z : float or None
            Stellar metallicity log10(Z) (absolute).
        neb_logU : float
            Ionization parameter log10(U). Default -3.0.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) (absolute). None = tie to stellar.
        neb_fesc : float
            Escape fraction [0, 1]. Suppresses continuum luminosity.
        neb_fdust : float
            Dust-absorption fraction of ionizing photons in HII regions [0, 1].
        gas_logu, gas_logn, gas_logz, gas_logno, gas_logco : float
            Cue gas params (low-level). Override high-level derivation.
        gas_logqion : float or None
            log10(Q_H) total. Override high-level derivation.
        ionspec_* : float or None
            Ionizing spectrum shape. Override high-level derivation.

        Returns
        -------
        wavelength : ndarray, shape (n_wave,)
            Wavelength grid (sorted) [Angstrom].
        luminosity : ndarray, shape (n_wave,)
            Nebular continuum [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        **Wavelength grid**: Returns the Cue native grid (fixed for all calls).
        For interpolation to an arbitrary wavelength grid, use
        ``predict_nebular_sed`` with ``ssp_wave`` argument.

        **Escape fraction**: When ``neb_fesc > 0``, ionizing photons escape
        without photoionizing nebular gas. The continuum is suppressed by the
        CIGALE ionizing-budget k-factor
        :func:`~tengri.components.nebular._recombination_coeffs.lyc_dust_escape_factor`
        ``(neb_fesc, neb_fdust)``, which → 0 as ``neb_fesc + neb_fdust → 1``
        (no surviving nebular emission when all ionizing photons are lost).

        """
        p = self._resolve_cue_params(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            gas_logu=gas_logu,
            gas_logn=gas_logn,
            gas_logz=gas_logz,
            gas_logno=gas_logno,
            gas_logco=gas_logco,
            gas_logqion=gas_logqion,
            ionspec_index1=ionspec_index1,
            ionspec_index2=ionspec_index2,
            ionspec_index3=ionspec_index3,
            ionspec_index4=ionspec_index4,
            ionspec_logLratio1=ionspec_logLratio1,
            ionspec_logLratio2=ionspec_logLratio2,
            ionspec_logLratio3=ionspec_logLratio3,
        )
        cont_wav, cont_lum = self._forward_continuum(p)
        # Apply CIGALE nebular emission scaling factor (k) accounting for both
        # ionizing photon escape (neb_fesc) and dust absorption (neb_fdust).
        # The k-factor applies the recombination coefficient ratio (alpha_1/alpha_B).
        k = lyc_dust_escape_factor(neb_fesc, neb_fdust)
        return cont_wav, cont_lum * k

    def predict_nebular_sed(
        self,
        ssp_wave: jnp.ndarray | None = None,
        ssp_weights: jnp.ndarray | None = None,
        ssp_log_ages_yr: jnp.ndarray | None = None,
        log_z: float | None = None,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 0.0,
        neb_fdust: float = 0.0,
        line_sigma_aa: float = 0.0,
        line_sigma_kms: float = 0.0,
        template_data: Any | None = None,
        **neb_params,
    ) -> jnp.ndarray:
        """Predict total nebular emission on an arbitrary wavelength grid.

        Supports the same high-level interface as CloudyGridBackend:
        pass ``ssp_weights``, ``ssp_log_ages_yr``, ``log_z`` and the
        mass-weighted Q_H and ionizing spectrum are derived internally.

        Parameters
        ----------
        ssp_wave : array, shape (n_wave,)
            Output wavelength grid in Angstrom.
        ssp_weights : array or None
            CSP mass weights.  Activates high-level mode.
        ssp_log_ages_yr : array or None
            log10(age/yr) of SSP age bins.
        log_z : float or None
            Stellar metallicity log10(Z) (absolute).
        neb_logU : float
            Ionization parameter.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) (absolute). None = tie to stellar.
        neb_fesc, neb_fesc_lya : float
            Escape fractions [dimensionless, in [0, 1]].
        neb_fdust : float
            Dust-absorption fraction of ionizing photons in HII regions
            [dimensionless, in [0, 1]]. Reduces nebular emission via the
            CIGALE k-factor (Inoue 2011).
        line_sigma_aa : float
            Gaussian width for emission lines (Angstrom). 0 = delta function.
        template_data : Any | None, optional
            Cue weights object. When provided, overrides ``self.weights``
            for JIT purposes (threading as a runtime parameter instead of
            closure-capturing). Default ``None`` uses ``self.weights``.
        **neb_params
            Additional Cue-specific overrides (gas_logn, gas_logno, etc.).

        Returns
        -------
        ndarray, shape (n_wave,)
            Total nebular SED in erg/s/Hz on the user-provided wavelength grid
            [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.

        **Line+Continuum**: This method computes both lines (via
        ``predict_nebular_line_luminosities``) and continuum (via
        ``predict_nebular_continuum``), then combines them on the user
        wavelength grid. For efficiency, parameters are resolved once and
        reused for both predictions (avoiding double computation).

        **Line profiles**: Emission lines can be placed as Gaussians
        (``line_sigma_aa > 0``) or delta functions (``line_sigma_aa = 0``).
        Delta functions use nearest-pixel scatter-add with bin-width
        averaging to produce proper flux units.

        **Ionizing photon loss**: Controlled via ``neb_fesc`` (escape fraction)
        and ``neb_fdust`` (dust-absorption fraction). Both reduce nebular
        emission (lines + continuum) via the CIGALE k-factor (Inoue 2011),
        accounting for the recombination coefficient ratio (alpha_1 / alpha_B).
        ``neb_fesc_lya`` applies additional Ly-alpha-specific suppression.

        """
        # Resolve params once (avoids double computation for lines + continuum)
        p = self._resolve_cue_params(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            **neb_params,
        )

        # Lines (all 138 for SED construction)
        line_wav, line_lum = self._forward_lines(
            p,
            cloudyfsps_only=False,
            neb_fesc=neb_fesc,
            neb_fesc_lya=neb_fesc_lya,
            neb_fdust=neb_fdust,
            template_data=template_data,
        )

        # Continuum (same resolved params — no double computation)
        cont_wav, cont_lum = self._forward_continuum(p, template_data=template_data)
        # Apply CIGALE nebular emission scaling factor (k) accounting for both
        # ionizing photon escape (neb_fesc) and dust absorption (neb_fdust).
        # The k-factor applies to both continuum and lines per CIGALE nebular.py.
        k = lyc_dust_escape_factor(neb_fesc, neb_fdust)
        cont_lum = cont_lum * k

        # Interpolate continuum onto SSP grid
        neb_sed = jnp.interp(ssp_wave, cont_wav, cont_lum, left=0.0, right=0.0)

        # Add emission lines via the shared renderer: velocity triweight when
        # ``line_sigma_kms > 0`` (Prospector-style intrinsic width), else the
        # fixed-Å Gaussian / nearest-pixel delta fallbacks.
        neb_sed = neb_sed + render_nebular_lines(
            line_wav, line_lum, ssp_wave, line_sigma_aa, line_sigma_kms
        )

        # Convert from internal Lsun/Hz to erg/s/Hz
        return neb_sed * _LSUN_ERG


# ── JIT-compiled pure-functional API (for use in inference loops) ─


@jax.jit
def predict_lines_jit(
    nn_params_12: jnp.ndarray,
    weights: CueWeights,
    gas_logq: jnp.ndarray,
    gas_logqion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled emission line prediction (functional API).

    Wrapper around predict_all_lines with @jax.jit decorator for use in
    inference loops and differentiable workflows.

    Parameters
    ----------
    nn_params_12 : array, shape (12,)
        NN-ready parameters (already converted via _prepare_nn_params).
    weights : CueWeights
        Pre-loaded weights (treated static by JIT via pytree).
    gas_logq : scalar
        log10(Q) ionizing photon rate normalization.
    gas_logqion : scalar
        log10(Q_H) total ionizing photon rate.

    Returns
    -------
    wavelengths : array, shape (n_lines,)
        Line wavelengths [Angstrom].
    luminosities : array, shape (n_lines,)
        Line luminosities [Lsun].

    Notes
    -----
    **JIT-compatible**: yes — decorated with @jax.jit.

    """
    return predict_all_lines(nn_params_12, weights, gas_logq, gas_logqion)


@jax.jit
def predict_continuum_jit(
    nn_params_12: jnp.ndarray,
    weights: CueWeights,
    gas_logq: jnp.ndarray,
    gas_logqion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled nebular continuum prediction (functional API).

    Wrapper around predict_continuum with @jax.jit decorator for use in
    inference loops and differentiable workflows.

    Parameters
    ----------
    nn_params_12 : array, shape (12,)
        NN-ready parameters (already converted via _prepare_nn_params).
    weights : CueWeights
        Pre-loaded weights (treated static by JIT via pytree).
    gas_logq : scalar
        log10(Q) ionizing photon rate normalization.
    gas_logqion : scalar
        log10(Q_H) total ionizing photon rate.

    Returns
    -------
    wavelength : array, shape (n_wave,)
        Wavelength grid [Angstrom].
    luminosity : array, shape (n_wave,)
        Nebular continuum [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — decorated with @jax.jit.

    """
    return predict_continuum(nn_params_12, weights, gas_logq, gas_logqion)


# ── JAX pytree registration for CueWeights (enables JIT with string fields)


def _cue_weights_flatten(cw):
    """Flatten CueWeights for JAX pytree: arrays as children, strings/ints as aux."""
    children = (
        cw.line_nets,
        cw.cont_net,
        cw.sorted_line_wav,
        cw.nn_line_wav,
        cw.line_old_idx,
        cw.cont_wav,
        # Batched arrays (JAX traceable)
        cw.batched_param_shifts,
        cw.batched_param_scales,
        cw.batched_W_hidden,
        cw.batched_b_hidden,
        cw.batched_alpha_hidden,
        cw.batched_beta_hidden,
        cw.batched_W_out,
        cw.batched_b_out,
        cw.batched_pca_scale,
        cw.batched_pca_shift,
        cw.batched_pca_comp,
        cw.batched_pca_mean,
        cw.batched_spec_scale,
        cw.batched_spec_shift,
        cw.batched_sort_idx,
        # ``line_wav_selections`` is a tuple of int arrays. It must live in
        # children, not aux_data: arrays in aux trigger a ``ValueError``
        # ("arrays cannot be passed as metadata fields") on the second JIT
        # cache lookup, because aux equality compares with ``==`` which
        # returns an array on ndarray inputs. See issue #464.
        cw.line_wav_selections,
    )
    # Non-array aux: strings and int tuples only.
    aux_data = (cw.line_names, cw.batched_n_lines)
    return children, aux_data


def _cue_weights_unflatten(aux_data, children):
    """Unflatten CueWeights from JAX pytree."""
    (
        line_nets,
        cont_net,
        sorted_line_wav,
        nn_line_wav,
        line_old_idx,
        cont_wav,
        b_ps,
        b_psc,
        b_Wh,
        b_bh,
        b_ah,
        b_beh,
        b_Wo,
        b_bo,
        b_pcas,
        b_pcash,
        b_pcac,
        b_pcam,
        b_ss,
        b_ssh,
        b_si,
        line_wav_selections,
    ) = children
    line_names, batched_n_lines = aux_data
    return CueWeights(
        line_nets=line_nets,
        cont_net=cont_net,
        line_names=line_names,
        line_wav_selections=line_wav_selections,
        sorted_line_wav=sorted_line_wav,
        nn_line_wav=nn_line_wav,
        line_old_idx=line_old_idx,
        cont_wav=cont_wav,
        batched_param_shifts=b_ps,
        batched_param_scales=b_psc,
        batched_W_hidden=b_Wh,
        batched_b_hidden=b_bh,
        batched_alpha_hidden=b_ah,
        batched_beta_hidden=b_beh,
        batched_W_out=b_Wo,
        batched_b_out=b_bo,
        batched_pca_scale=b_pcas,
        batched_pca_shift=b_pcash,
        batched_pca_comp=b_pcac,
        batched_pca_mean=b_pcam,
        batched_spec_scale=b_ss,
        batched_spec_shift=b_ssh,
        batched_n_lines=batched_n_lines,
        batched_sort_idx=b_si,
    )


# Register after class definition
jax.tree_util.register_pytree_node(CueWeights, _cue_weights_flatten, _cue_weights_unflatten)


def _subnet_flatten(sw):
    """Flatten SubNetWeights: arrays as children, n_layers as aux."""
    children = (
        sw.W,
        sw.b,
        sw.alphas,
        sw.betas,
        sw.param_shift,
        sw.param_scale,
        sw.pca_shift,
        sw.pca_scale,
        sw.log_spec_shift,
        sw.log_spec_scale,
        sw.pca_components,
        sw.pca_mean,
    )
    return children, sw.n_layers


def _subnet_unflatten(n_layers, children):
    """Unflatten SubNetWeights."""
    (
        W,
        b,
        alphas,
        betas,
        param_shift,
        param_scale,
        pca_shift,
        pca_scale,
        log_spec_shift,
        log_spec_scale,
        pca_components,
        pca_mean,
    ) = children
    return SubNetWeights(
        W=W,
        b=b,
        alphas=alphas,
        betas=betas,
        param_shift=param_shift,
        param_scale=param_scale,
        pca_shift=pca_shift,
        pca_scale=pca_scale,
        log_spec_shift=log_spec_shift,
        log_spec_scale=log_spec_scale,
        pca_components=pca_components,
        pca_mean=pca_mean,
        n_layers=n_layers,
    )


jax.tree_util.register_pytree_node(SubNetWeights, _subnet_flatten, _subnet_unflatten)
