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

References
----------
- Li et al. 2025, ApJ (Cue)
- Alsing et al. 2020, ApJS (Speculator architecture)

Notes
-----
All functions are JIT-compatible and differentiable through JAX.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_C_CGS = 2.9979e10  # cm/s
_LSUN_ERG = 3.839e33  # erg/s  (Cue convention, NOT IAU 2015)
_LOG_LSUN = jnp.log10(_LSUN_ERG)
_LOG_4PI = jnp.log10(4.0 * jnp.pi)
_LOG_C = jnp.log10(_C_CGS)

# Solar metallicity (Asplund+2009).
# SSP ssp_lgmet stores absolute log10(Z); Cue uses log10(Z/Zsun).
_LOG10_ZSUN = -1.8477116556169435

# Maximum SSP age contributing to nebular emission
_MAX_NEB_LOG_AGE = 8.0  # log10(100 Myr in yr)


# ---------------------------------------------------------------------------
# Data containers (immutable NamedTuples for JAX tracing)
# ---------------------------------------------------------------------------


class SubNetWeights(NamedTuple):
    """Weights for a single Speculator sub-network."""

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

    Includes precomputed batched weight arrays for fast inference.
    The ``batched_*`` fields are computed once at load time from
    the individual ``line_nets`` and stored as dense JAX arrays.
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


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

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
    """Extract a SubNetWeights from the flat npz dict."""
    n_layers = int(npz[f"{prefix}_n_layers"])

    W = tuple(jnp.array(npz[f"{prefix}_W_{i}"]) for i in range(n_layers))
    b = tuple(jnp.array(npz[f"{prefix}_b_{i}"]) for i in range(n_layers))
    alphas = tuple(jnp.array(npz[f"{prefix}_alpha_{i}"]) for i in range(n_layers - 1))
    betas = tuple(jnp.array(npz[f"{prefix}_beta_{i}"]) for i in range(n_layers - 1))

    return SubNetWeights(
        W=W,
        b=b,
        alphas=alphas,
        betas=betas,
        param_shift=jnp.array(npz[f"{prefix}_parameters_shift"]),
        param_scale=jnp.array(npz[f"{prefix}_parameters_scale"]),
        pca_shift=jnp.array(npz[f"{prefix}_pca_shift"]),
        pca_scale=jnp.array(npz[f"{prefix}_pca_scale"]),
        log_spec_shift=jnp.array(npz[f"{prefix}_log_spectrum_shift"]),
        log_spec_scale=jnp.array(npz[f"{prefix}_log_spectrum_scale"]),
        pca_components=jnp.array(npz[f"{prefix}_pca_components"]),
        pca_mean=jnp.array(npz[f"{prefix}_pca_mean"]),
        n_layers=n_layers,
    )


def load_cue_weights(npz_path: str) -> CueWeights:
    """Load all Cue weights from the npz file.

    Parameters
    ----------
    npz_path : str
        Path to ``cue_weights.npz`` produced by ``convert_cue_weights.py``.

    Returns
    -------
    CueWeights
        Immutable container with all weights on JAX arrays.
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
        line_wav_sels.append(jnp.array(npz[f"{prefix}_wav_selection"]))

    # Continuum sub-network
    cont_net = _load_subnet(npz, "cont")

    # Precompute batched weight arrays for fast line prediction
    nets = line_nets
    n_hidden = nets[0].n_layers - 1

    # Hidden layers: stack (16, in, out) — all same architecture
    b_W_h = tuple(jnp.stack([n.W[i] for n in nets]) for i in range(n_hidden))
    b_b_h = tuple(jnp.stack([n.b[i] for n in nets]) for i in range(n_hidden))
    b_a_h = tuple(jnp.stack([n.alphas[i] for n in nets]) for i in range(n_hidden))
    b_beta_h = tuple(jnp.stack([n.betas[i] for n in nets]) for i in range(n_hidden))

    # Output layer + PCA: pad to max dims
    max_pcas = max(n.W[n_hidden].shape[1] for n in nets)
    max_lines = max(n.pca_components.shape[1] for n in nets)

    def _pad2d(arr, target_r, target_c, fill=0.0):
        return jnp.pad(
            arr, ((0, target_r - arr.shape[0]), (0, target_c - arr.shape[1])), constant_values=fill
        )

    def _pad1d(arr, target, fill=0.0):
        return jnp.pad(arr, (0, target - arr.shape[0]), constant_values=fill)

    nn_line_wav = jnp.array(npz["nn_line_wavelength"])

    return CueWeights(
        line_nets=tuple(line_nets),
        cont_net=cont_net,
        line_names=_LINE_NAMES,
        line_wav_selections=tuple(line_wav_sels),
        sorted_line_wav=jnp.array(npz["sorted_line_wavelength"]),
        nn_line_wav=nn_line_wav,
        line_old_idx=jnp.array(npz["line_old_idx"]),
        cont_wav=jnp.array(npz["cont_wavelength"]),
        # Precomputed batched arrays
        batched_param_shifts=jnp.stack([n.param_shift for n in nets]),
        batched_param_scales=jnp.stack([n.param_scale for n in nets]),
        batched_W_hidden=b_W_h,
        batched_b_hidden=b_b_h,
        batched_alpha_hidden=b_a_h,
        batched_beta_hidden=b_beta_h,
        batched_W_out=jnp.stack([_pad2d(n.W[n_hidden], 256, max_pcas) for n in nets]),
        batched_b_out=jnp.stack([_pad1d(n.b[n_hidden], max_pcas) for n in nets]),
        batched_pca_scale=jnp.stack([_pad1d(n.pca_scale, max_pcas, fill=1.0) for n in nets]),
        batched_pca_shift=jnp.stack([_pad1d(n.pca_shift, max_pcas) for n in nets]),
        batched_pca_comp=jnp.stack([_pad2d(n.pca_components, max_pcas, max_lines) for n in nets]),
        batched_pca_mean=jnp.stack([_pad1d(n.pca_mean, max_lines) for n in nets]),
        batched_spec_scale=jnp.stack(
            [_pad1d(n.log_spec_scale, max_lines, fill=1.0) for n in nets]
        ),
        batched_spec_shift=jnp.stack([_pad1d(n.log_spec_shift, max_lines) for n in nets]),
        batched_n_lines=tuple(n.pca_components.shape[1] for n in nets),
        batched_sort_idx=jnp.argsort(nn_line_wav),
    )


# ---------------------------------------------------------------------------
# Neural network forward pass (pure JAX, JIT-compatible)
# ---------------------------------------------------------------------------


def _speculator_activation(x: jnp.ndarray, alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """Learned Swish activation: x * (beta + (1 - beta) * sigmoid(alpha * x))."""
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


# ---------------------------------------------------------------------------
# Parameter conversion: user-facing -> network input
# ---------------------------------------------------------------------------


def _logq_from_logu(
    gas_logu: jnp.ndarray, gas_logn: jnp.ndarray, log_R: float = 19.0
) -> jnp.ndarray:
    """Convert ionization parameter logU to logQ.

    logQ = logU + log(4*pi) + 2*log(R) + logn + log(c)

    This matches cue.utils.logQ with R=1e19 (default Stromgren radius).
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
    """Convert user-facing parameters to the 12-element NN input vector.

    The network expects:
    [index1, index2, index3, index4, logLratio1, logLratio2, logLratio3,
     gas_logq, 10**gas_logn, gas_logz, gas_logno, gas_logco]
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


def prepare_nn_params_from_dict(params: dict) -> jnp.ndarray:
    """Convert a parameter dictionary to the 12-element NN input vector.

    Convenience wrapper for _prepare_nn_params.
    """
    return _prepare_nn_params(
        ionspec_index1=params["ionspec_index1"],
        ionspec_index2=params["ionspec_index2"],
        ionspec_index3=params["ionspec_index3"],
        ionspec_index4=params["ionspec_index4"],
        ionspec_logLratio1=params["ionspec_logLratio1"],
        ionspec_logLratio2=params["ionspec_logLratio2"],
        ionspec_logLratio3=params["ionspec_logLratio3"],
        gas_logu=params["gas_logu"],
        gas_logn=params["gas_logn"],
        gas_logz=params["gas_logz"],
        gas_logno=params["gas_logno"],
        gas_logco=params["gas_logco"],
    )


# ---------------------------------------------------------------------------
# Line prediction
# ---------------------------------------------------------------------------


def _predict_lines_single_net(
    nn_params: jnp.ndarray,
    net: SubNetWeights,
) -> jnp.ndarray:
    """Predict log10 line luminosities from a single line sub-network.

    Parameters
    ----------
    nn_params : array, shape (12,)
        NN-ready parameters (logq, linear n, etc.).
    net : SubNetWeights
        One line sub-network's weights.

    Returns
    -------
    array, shape (n_lines_for_this_net,)
        Log10(luminosity) in Lsun/Q_H for each line this net predicts.
    """
    return _speculator_log_spectrum(nn_params, net)


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
        NN-ready parameters.
    weights : CueWeights
        Pre-loaded weights.
    gas_logq : float
        log10(Q) = logU + log(4*pi*c) + 2*logR + logn.
    gas_logqion : float
        log10(Q_H) — total ionizing photon rate for normalization.

    Returns
    -------
    wavelengths : array, shape (n_lines,)
        Line wavelengths in Angstrom (sorted).
    luminosities : array, shape (n_lines,)
        Line luminosities in Lsun.
    """
    # --- Fully batched forward pass using precomputed weight arrays ---
    # All stacking/padding was done once at load time in load_cue_weights()

    # Step 1: Normalize inputs (16 different shifts/scales)
    x = (nn_params[None, :] - weights.batched_param_shifts) / weights.batched_param_scales

    # Step 2: Batched hidden layers (precomputed stacked weights)
    for W, b, alpha, beta in zip(
        weights.batched_W_hidden,
        weights.batched_b_hidden,
        weights.batched_alpha_hidden,
        weights.batched_beta_hidden,
    ):
        x = jnp.einsum("ni,nio->no", x, W) + b
        x = x * (beta + (1.0 - beta) * jax.nn.sigmoid(alpha * x))

    # Step 3: Batched output layer (precomputed padded weights)
    pca_coeffs = jnp.einsum("ni,nio->no", x, weights.batched_W_out) + weights.batched_b_out
    pca_coeffs = pca_coeffs * weights.batched_pca_scale + weights.batched_pca_shift

    # Step 4: Batched PCA inverse (precomputed padded components)
    log_spec = (
        jnp.einsum("np,npl->nl", pca_coeffs, weights.batched_pca_comp) + weights.batched_pca_mean
    )
    log_spec = log_spec * weights.batched_spec_scale + weights.batched_spec_shift

    # Extract actual (unpadded) lines and concatenate
    all_log_lum = []
    for i, n_lines_i in enumerate(weights.batched_n_lines):
        all_log_lum.append(log_spec[i, :n_lines_i])
    log_lum_concat = jnp.concatenate(all_log_lum, axis=-1)

    # Sort by wavelength (precomputed index)
    log_lum_sorted = log_lum_concat[weights.batched_sort_idx]
    wav_sorted = weights.nn_line_wav[weights.batched_sort_idx]

    # Convert from log10(Lsun/Q_H) to Lsun (gradient-safe)
    exponent = log_lum_sorted - gas_logq + gas_logqion - _LOG_LSUN
    exponent_safe = jnp.clip(exponent, -100.0, 100.0)
    luminosities = 10.0**exponent_safe

    return wav_sorted, luminosities


# ---------------------------------------------------------------------------
# Continuum prediction
# ---------------------------------------------------------------------------


def predict_continuum(
    nn_params: jnp.ndarray,
    weights: CueWeights,
    gas_logq: jnp.ndarray,
    gas_logqion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict nebular continuum SED.

    Parameters
    ----------
    nn_params : array, shape (12,)
        NN-ready parameters.
    weights : CueWeights
        Pre-loaded weights.
    gas_logq : float
        log10(Q).
    gas_logqion : float
        log10(Q_H) for normalization.

    Returns
    -------
    wavelength : array, shape (n_wave,)
        Continuum wavelength grid in Angstrom.
    luminosity : array, shape (n_wave,)
        Nebular continuum in Lsun/Hz.
    """
    log_spec = _speculator_log_spectrum(nn_params, weights.cont_net)

    # Sort by wavelength (Cue does wavind_sorted = argsort(cont_wavelength))
    sort_idx = jnp.argsort(weights.cont_wav)
    log_spec_sorted = log_spec[sort_idx]
    wav_sorted = weights.cont_wav[sort_idx]

    # Convert from log10(Lsun/Hz/Q_H) to Lsun/Hz:
    # Following Cue emulator.py predict_cont():
    #   cont = 10**(log_spec - gas_logq + gas_logqion - log10(3.839E33))
    # Clamp exponent to avoid inf/underflow (gradient-safe)
    exponent = log_spec_sorted - gas_logq + gas_logqion - _LOG_LSUN
    luminosity = 10.0 ** jnp.clip(exponent, -100.0, 100.0)

    # Zero out wavelengths below Lyman limit (Cue convention)
    luminosity = jnp.where(wav_sorted > 911.6, luminosity, 0.0)

    return wav_sorted, luminosity


# ---------------------------------------------------------------------------
# Backend class (matches CloudyGridBackend interface)
# ---------------------------------------------------------------------------


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
        Path to ``cue_weights.npz``.
    default_gas_logqion : float
        Default log10(Q_H) normalization when not specified per call.
    """

    def __init__(
        self,
        weights_path: str,
        ssp_data=None,
        default_gas_logqion: float = 49.1,
    ) -> None:
        self.name = "cue"
        self.has_free_params = True
        self.weights = load_cue_weights(weights_path)
        self.default_gas_logqion = default_gas_logqion

        # Cache sorted wavelength arrays
        self._line_sort_idx = jnp.argsort(self.weights.nn_line_wav)
        self._cont_sort_idx = jnp.argsort(self.weights.cont_wav)

        # Precompute ionizing spectrum parameters from SSP if provided.
        # These serve as defaults when ionspec params are not explicitly
        # specified. Users can override by setting ionspec_index1..4 and
        # ionspec_logLratio1..3 as free params in ParamSpec.
        self._ionspec_table = None
        self._logqion_table = None
        self._ssp_lgmet = None
        self._ssp_log_age_yr = None
        if ssp_data is not None:
            self._precompute_ionizing_params(ssp_data)

    def _precompute_ionizing_params(self, ssp_data) -> None:
        """Precompute ionizing spectrum parameters for all SSP (met, age)."""
        import numpy as np

        from diffsed.models.nebular.ionizing_spectrum import precompute_ionizing_params_table

        result = precompute_ionizing_params_table(
            np.array(ssp_data.ssp_wave),
            np.array(ssp_data.ssp_flux),
            np.array(ssp_data.ssp_lgmet),
        )
        self._ionspec_table = jnp.array(result["ionspec_table"])
        self._logqion_table = jnp.array(result["logqion_table"])
        self._ssp_lgmet = jnp.array(ssp_data.ssp_lgmet)
        self._ssp_log_age_yr = jnp.array(ssp_data.ssp_lg_age_gyr) + 9.0

    def get_ionizing_params_at(
        self,
        log_z: float,
        log_age_yr: float,
    ) -> tuple[jnp.ndarray, float]:
        """Get precomputed ionizing params at (Z, age) via interpolation.

        Returns (ionspec_7, logqion) or (None, None) if not precomputed.
        """
        if self._ionspec_table is None:
            return None, None

        from diffsed.models.nebular.ionizing_spectrum import interpolate_ionizing_params

        return interpolate_ionizing_params(
            self._ionspec_table,
            self._logqion_table,
            self._ssp_lgmet,
            self._ssp_log_age_yr,
            log_z,
            log_age_yr,
        )

    # ------------------------------------------------------------------
    # High-level interface (matches CloudyGridBackend)
    # ------------------------------------------------------------------

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

        # Young age bins only (< 100 Myr)
        young_mask = ssp_log_ages_yr <= _MAX_NEB_LOG_AGE
        young_ages = ssp_log_ages_yr[young_mask]
        young_weights = ssp_weights[young_mask]

        # Sum mass-weighted Q_H (per Msun) over young age bins
        total_qh = 0.0
        best_qh_weight = -1.0
        best_age_idx = 0
        for i in range(len(young_ages)):
            w_i = float(young_weights[i])
            if w_i <= 0:
                continue
            _, logqion_i = self.get_ionizing_params_at(log_z, float(young_ages[i]))
            if logqion_i is None:
                continue
            qh_i = 10.0 ** float(logqion_i)
            total_qh += w_i * qh_i
            if w_i * qh_i > best_qh_weight:
                best_qh_weight = w_i * qh_i
                best_age_idx = i

        if total_qh <= 0:
            total_logqion = -99.0
        else:
            total_logqion = np.log10(total_qh)

        # Ionizing spectrum shape from dominant age bin
        ionspec_7, _ = self.get_ionizing_params_at(log_z, float(young_ages[best_age_idx]))
        i7 = np.array(ionspec_7) if ionspec_7 is not None else np.zeros(7)

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
            ionspec_index1=float(i7[0]),
            ionspec_index2=float(i7[1]),
            ionspec_index3=float(i7[2]),
            ionspec_index4=float(i7[3]),
            ionspec_logLratio1=float(i7[4]),
            ionspec_logLratio2=float(i7[5]),
            ionspec_logLratio3=float(i7[6]),
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
        wavelengths : array
        luminosities : array (Lsun)
        """
        # High-level mode: derive Cue params from SSP weights
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
            # Explicit overrides take precedence
            if gas_logu is None:
                gas_logu = derived["gas_logu"]
            if gas_logz is None:
                gas_logz = derived["gas_logz"]
            if gas_logqion is None:
                gas_logqion = derived["gas_logqion"]
            if ionspec_index1 is None:
                ionspec_index1 = derived["ionspec_index1"]
            if ionspec_index2 is None:
                ionspec_index2 = derived["ionspec_index2"]
            if ionspec_index3 is None:
                ionspec_index3 = derived["ionspec_index3"]
            if ionspec_index4 is None:
                ionspec_index4 = derived["ionspec_index4"]
            if ionspec_logLratio1 is None:
                ionspec_logLratio1 = derived["ionspec_logLratio1"]
            if ionspec_logLratio2 is None:
                ionspec_logLratio2 = derived["ionspec_logLratio2"]
            if ionspec_logLratio3 is None:
                ionspec_logLratio3 = derived["ionspec_logLratio3"]

        # Fill remaining defaults for pure low-level calls
        if gas_logu is None:
            gas_logu = neb_logU
        if gas_logz is None:
            gas_logz = 0.0
        if gas_logqion is None:
            gas_logqion = self.default_gas_logqion
        if ionspec_index1 is None:
            ionspec_index1 = 19.7
        if ionspec_index2 is None:
            ionspec_index2 = 5.3
        if ionspec_index3 is None:
            ionspec_index3 = 1.6
        if ionspec_index4 is None:
            ionspec_index4 = 0.6
        if ionspec_logLratio1 is None:
            ionspec_logLratio1 = 3.9
        if ionspec_logLratio2 is None:
            ionspec_logLratio2 = 0.01
        if ionspec_logLratio3 is None:
            ionspec_logLratio3 = 0.2

        nn_params = _prepare_nn_params(
            jnp.asarray(ionspec_index1, dtype=jnp.float32),
            jnp.asarray(ionspec_index2, dtype=jnp.float32),
            jnp.asarray(ionspec_index3, dtype=jnp.float32),
            jnp.asarray(ionspec_index4, dtype=jnp.float32),
            jnp.asarray(ionspec_logLratio1, dtype=jnp.float32),
            jnp.asarray(ionspec_logLratio2, dtype=jnp.float32),
            jnp.asarray(ionspec_logLratio3, dtype=jnp.float32),
            jnp.asarray(gas_logu, dtype=jnp.float32),
            jnp.asarray(gas_logn, dtype=jnp.float32),
            jnp.asarray(gas_logz, dtype=jnp.float32),
            jnp.asarray(gas_logno, dtype=jnp.float32),
            jnp.asarray(gas_logco, dtype=jnp.float32),
        )

        gas_logq = _logq_from_logu(
            jnp.asarray(gas_logu, dtype=jnp.float32),
            jnp.asarray(gas_logn, dtype=jnp.float32),
        )

        wav, lum = predict_all_lines(
            nn_params,
            self.weights,
            gas_logq,
            jnp.asarray(gas_logqion, dtype=jnp.float32),
        )

        # Apply general escape fraction
        lum = lum * (1.0 - neb_fesc)

        # Apply differential Ly-alpha escape fraction
        # Ly-alpha at 1215.67 A: replace generic fesc with Ly-alpha-specific one
        lya_idx = jnp.argmin(jnp.abs(wav - 1215.67))
        lya_scale = (1.0 - neb_fesc_lya) / jnp.maximum(1.0 - neb_fesc, 1e-10)
        lum = lum.at[lya_idx].multiply(lya_scale)

        if cloudyfsps_only:
            old_idx = self.weights.line_old_idx
            return wav[old_idx], lum[old_idx]
        return wav, lum

    def predict_nebular_continuum(
        self,
        gas_logu: float = -2.5,
        gas_logn: float = 2.0,
        gas_logz: float = 0.0,
        gas_logno: float = 0.0,
        gas_logco: float = 0.0,
        gas_logqion: float | None = None,
        ionspec_index1: float = 19.7,
        ionspec_index2: float = 5.3,
        ionspec_index3: float = 1.6,
        ionspec_index4: float = 0.6,
        ionspec_logLratio1: float = 3.9,
        ionspec_logLratio2: float = 0.01,
        ionspec_logLratio3: float = 0.2,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict nebular continuum SED.

        Parameters
        ----------
        See predict_nebular_line_luminosities for parameter descriptions.

        Returns
        -------
        wavelength : array, shape (n_wave,)
            Wavelength grid in Angstrom.
        luminosity : array, shape (n_wave,)
            Nebular continuum in Lsun/Hz.
        """
        if gas_logqion is None:
            gas_logqion = self.default_gas_logqion

        nn_params = _prepare_nn_params(
            jnp.asarray(ionspec_index1, dtype=jnp.float32),
            jnp.asarray(ionspec_index2, dtype=jnp.float32),
            jnp.asarray(ionspec_index3, dtype=jnp.float32),
            jnp.asarray(ionspec_index4, dtype=jnp.float32),
            jnp.asarray(ionspec_logLratio1, dtype=jnp.float32),
            jnp.asarray(ionspec_logLratio2, dtype=jnp.float32),
            jnp.asarray(ionspec_logLratio3, dtype=jnp.float32),
            jnp.asarray(gas_logu, dtype=jnp.float32),
            jnp.asarray(gas_logn, dtype=jnp.float32),
            jnp.asarray(gas_logz, dtype=jnp.float32),
            jnp.asarray(gas_logno, dtype=jnp.float32),
            jnp.asarray(gas_logco, dtype=jnp.float32),
        )

        gas_logq = _logq_from_logu(
            jnp.asarray(gas_logu, dtype=jnp.float32),
            jnp.asarray(gas_logn, dtype=jnp.float32),
        )

        return predict_continuum(
            nn_params,
            self.weights,
            gas_logq,
            jnp.asarray(gas_logqion, dtype=jnp.float32),
        )

    def predict_nebular_sed(
        self,
        ssp_wave: jnp.ndarray = None,
        ssp_weights: jnp.ndarray = None,
        ssp_log_ages_yr: jnp.ndarray = None,
        log_z: float = None,
        neb_logU: float = -3.0,
        neb_logZ_gas: float = None,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 0.0,
        line_sigma_aa: float = 0.0,
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
            Escape fractions.
        line_sigma_aa : float
            Gaussian width for emission lines (Angstrom). 0 = delta function.
        **neb_params
            Additional Cue-specific overrides (gas_logn, gas_logno, etc.).

        Returns
        -------
        array, shape (n_wave,)
            Total nebular SED in Lsun/Hz on the SSP wavelength grid.
        """
        # Build the shared kwargs for lines and continuum
        shared = dict(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            neb_fesc=neb_fesc,
            neb_fesc_lya=neb_fesc_lya,
            **neb_params,
        )

        # Lines
        line_wav, line_lum = self.predict_nebular_line_luminosities(
            cloudyfsps_only=False,
            **shared,
        )

        # Continuum (reuse the same high-level → low-level dispatch)
        # For now, pass through to predict_nebular_continuum with the
        # same derived params.  We re-derive to keep it simple.
        if ssp_weights is not None:
            derived = self._compute_weighted_cue_params(
                ssp_weights,
                ssp_log_ages_yr,
                log_z,
                neb_logU=neb_logU,
                neb_logZ_gas=neb_logZ_gas,
                **{
                    k: v
                    for k, v in neb_params.items()
                    if k in ("gas_logn", "gas_logno", "gas_logco")
                },
            )
            cont_wav, cont_lum = self.predict_nebular_continuum(**derived)
        else:
            cont_wav, cont_lum = self.predict_nebular_continuum(**neb_params)

        # Interpolate continuum onto SSP grid
        neb_sed = jnp.interp(ssp_wave, cont_wav, cont_lum, left=0.0, right=0.0)

        # Add emission lines
        _C_CGS * 1e8  # Angstrom/s
        if line_sigma_aa > 0:
            # Gaussian profiles
            # For each line: convert L_line (Lsun) to Lsun/Hz via Gaussian
            # sigma_nu = sigma_lambda * c / lambda^2 (in Hz)
            # profile in Hz^-1, integrated over nu gives 1
            for j in range(len(line_wav)):
                lw = line_wav[j]
                ll = line_lum[j]
                sigma_nu = line_sigma_aa * _C_CGS / (lw * 1e-8) ** 2
                profile = jnp.exp(-0.5 * ((ssp_wave - lw) / line_sigma_aa) ** 2)
                profile = profile / (jnp.sqrt(2.0 * jnp.pi) * sigma_nu)
                neb_sed = neb_sed + ll * _LSUN_ERG * profile
        else:
            # Delta function: add to nearest pixel
            for j in range(len(line_wav)):
                idx = jnp.argmin(jnp.abs(ssp_wave - line_wav[j]))
                dwave = jnp.abs(ssp_wave[idx + 1] - ssp_wave[idx - 1]) / 2.0
                dnu = _C_CGS / (ssp_wave[idx] * 1e-8) ** 2 * dwave * 1e-8
                line_flux_density = line_lum[j] / dnu  # Lsun/Hz
                neb_sed = neb_sed.at[idx].add(line_flux_density)

        return neb_sed


# ---------------------------------------------------------------------------
# JIT-compiled pure-functional API (for use in inference loops)
# ---------------------------------------------------------------------------


@jax.jit
def predict_lines_jit(
    nn_params_12: jnp.ndarray,
    weights: CueWeights,
    gas_logq: jnp.ndarray,
    gas_logqion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled line prediction (functional API).

    Parameters
    ----------
    nn_params_12 : array, shape (12,)
        NN-ready parameters (already converted via _prepare_nn_params).
    weights : CueWeights
        Pre-loaded weights (treated as static by JIT via pytree).
    gas_logq : scalar
        log10(Q).
    gas_logqion : scalar
        log10(Q_H).

    Returns
    -------
    wavelengths, luminosities : arrays
    """
    return predict_all_lines(nn_params_12, weights, gas_logq, gas_logqion)


@jax.jit
def predict_continuum_jit(
    nn_params_12: jnp.ndarray,
    weights: CueWeights,
    gas_logq: jnp.ndarray,
    gas_logqion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled continuum prediction (functional API).

    Parameters
    ----------
    nn_params_12 : array, shape (12,)
        NN-ready parameters.
    weights : CueWeights
        Pre-loaded weights.
    gas_logq : scalar
        log10(Q).
    gas_logqion : scalar
        log10(Q_H).

    Returns
    -------
    wavelength, luminosity : arrays
    """
    return predict_continuum(nn_params_12, weights, gas_logq, gas_logqion)


# ---------------------------------------------------------------------------
# JAX pytree registration for CueWeights (enables JIT with string fields)
# ---------------------------------------------------------------------------


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
    )
    # Non-array aux: strings, int tuples
    aux_data = (cw.line_names, cw.line_wav_selections, cw.batched_n_lines)
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
    ) = children
    line_names, line_wav_selections, batched_n_lines = aux_data
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
