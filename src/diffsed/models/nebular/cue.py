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

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_C_CGS = 2.9979e10       # cm/s
_LSUN_ERG = 3.839e33     # erg/s  (Cue convention, NOT IAU 2015)
_LOG_LSUN = jnp.log10(_LSUN_ERG)
_LOG_4PI = jnp.log10(4.0 * jnp.pi)
_LOG_C = jnp.log10(_C_CGS)


# ---------------------------------------------------------------------------
# Data containers (immutable NamedTuples for JAX tracing)
# ---------------------------------------------------------------------------

class SubNetWeights(NamedTuple):
    """Weights for a single Speculator sub-network."""
    W: tuple          # tuple of (in, out) weight matrices per layer
    b: tuple          # tuple of (out,) bias vectors per layer
    alphas: tuple     # tuple of (out,) activation params (hidden layers only)
    betas: tuple      # tuple of (out,) activation params (hidden layers only)
    param_shift: jnp.ndarray    # (n_params,)
    param_scale: jnp.ndarray    # (n_params,)
    pca_shift: jnp.ndarray      # (n_pcas,)
    pca_scale: jnp.ndarray      # (n_pcas,)
    log_spec_shift: jnp.ndarray  # (n_wavelengths,)
    log_spec_scale: jnp.ndarray  # (n_wavelengths,)
    pca_components: jnp.ndarray  # (n_pcas, n_wavelengths) — sklearn PCA basis
    pca_mean: jnp.ndarray        # (n_wavelengths,) — sklearn PCA centering
    n_layers: int


class CueWeights(NamedTuple):
    """All Cue weights for lines + continuum."""
    line_nets: tuple       # tuple of SubNetWeights, one per line sub-network
    cont_net: SubNetWeights
    line_names: tuple      # tuple of str
    line_wav_selections: tuple  # tuple of int arrays per sub-network
    sorted_line_wav: jnp.ndarray   # (n_lines_total,) sorted line wavelengths
    nn_line_wav: jnp.ndarray       # (n_nn_lines,) concatenated NN output wavelengths
    line_old_idx: jnp.ndarray      # indices of "old" (cloudyfsps) lines
    cont_wav: jnp.ndarray          # (n_wave_cont,) continuum wavelength grid


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

_LINE_NAMES = (
    "H1", "He1", "He2", "C1", "C2C3", "C4", "N", "O1",
    "O2", "O3", "ionE_1", "ionE_2", "S4", "Ar4", "Ne3", "Ne4",
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
                f"Missing line sub-network '{name}' in {npz_path}. "
                "Re-run convert_cue_weights.py."
            )
        line_nets.append(_load_subnet(npz, prefix))
        line_wav_sels.append(jnp.array(npz[f"{prefix}_wav_selection"]))

    # Continuum sub-network
    cont_net = _load_subnet(npz, "cont")

    return CueWeights(
        line_nets=tuple(line_nets),
        cont_net=cont_net,
        line_names=_LINE_NAMES,
        line_wav_selections=tuple(line_wav_sels),
        sorted_line_wav=jnp.array(npz["sorted_line_wavelength"]),
        nn_line_wav=jnp.array(npz["nn_line_wavelength"]),
        line_old_idx=jnp.array(npz["line_old_idx"]),
        cont_wav=jnp.array(npz["cont_wavelength"]),
    )


# ---------------------------------------------------------------------------
# Neural network forward pass (pure JAX, JIT-compatible)
# ---------------------------------------------------------------------------

def _speculator_activation(x: jnp.ndarray, alpha: jnp.ndarray,
                           beta: jnp.ndarray) -> jnp.ndarray:
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

def _logq_from_logu(gas_logu: jnp.ndarray, gas_logn: jnp.ndarray,
                    log_R: float = 19.0) -> jnp.ndarray:
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
    gas_n_linear = 10.0 ** gas_logn

    return jnp.stack([
        ionspec_index1, ionspec_index2, ionspec_index3, ionspec_index4,
        ionspec_logLratio1, ionspec_logLratio2, ionspec_logLratio3,
        gas_logq, gas_n_linear, gas_logz, gas_logno, gas_logco,
    ], axis=-1)


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
    # Predict log10 luminosity from each sub-network, concatenate
    all_log_lum = []
    for net in weights.line_nets:
        log_lum = _predict_lines_single_net(nn_params, net)
        all_log_lum.append(log_lum)

    # Concatenated in sub-network order (matches nn_line_wav)
    log_lum_concat = jnp.concatenate(all_log_lum, axis=-1)

    # Sort by wavelength (nn_line_wav is the concatenated unsorted wavelengths)
    sort_idx = jnp.argsort(weights.nn_line_wav)
    log_lum_sorted = log_lum_concat[sort_idx]
    wav_sorted = weights.nn_line_wav[sort_idx]

    # Convert from log10(Lsun/Q_H) to Lsun:
    # L = 10^(log_lum - gas_logq + gas_logqion - log10(Lsun_cgs))
    # Following Cue emulator.py predict_lines():
    #   line_nn_spectra = 10**(line_nn_spectra - gas_logq + gas_logqion - log10(3.839E33))
    luminosities = 10.0 ** (log_lum_sorted - gas_logq + gas_logqion - _LOG_LSUN)

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
    luminosity = 10.0 ** (log_spec_sorted - gas_logq + gas_logqion - _LOG_LSUN)

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
        default_gas_logqion: float = 49.1,
    ) -> None:
        self.name = "cue"
        self.has_free_params = True
        self.weights = load_cue_weights(weights_path)
        self.default_gas_logqion = default_gas_logqion

        # Cache sorted wavelength arrays
        self._line_sort_idx = jnp.argsort(self.weights.nn_line_wav)
        self._cont_sort_idx = jnp.argsort(self.weights.cont_wav)

    def predict_nebular_line_luminosities(
        self,
        gas_logu: float = -2.5,
        gas_logn: float = 2.0,
        gas_logz: float = 0.0,
        gas_logno: float = 0.0,
        gas_logco: float = 0.0,
        gas_logqion: float = None,
        ionspec_index1: float = 19.7,
        ionspec_index2: float = 5.3,
        ionspec_index3: float = 1.6,
        ionspec_index4: float = 0.6,
        ionspec_logLratio1: float = 3.9,
        ionspec_logLratio2: float = 0.01,
        ionspec_logLratio3: float = 0.2,
        cloudyfsps_only: bool = True,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict emission line luminosities.

        Parameters
        ----------
        gas_logu : float
            Ionization parameter log10(U). Range: [-4, -1].
        gas_logn : float
            Gas density log10(n_H/cm^-3). Range: [1, 4].
        gas_logz : float
            Gas metallicity log10(Z/Zsun). Range: [-2.2, 0.5].
        gas_logno : float
            [N/O] abundance ratio. Range: [-1, log10(5.4)].
        gas_logco : float
            [C/O] abundance ratio. Range: [-1, log10(5.4)].
        gas_logqion : float or None
            log10(Q_H) normalization. None uses default.
        ionspec_index1..4 : float
            Power-law slope segments of ionizing spectrum.
        ionspec_logLratio1..3 : float
            Flux ratios between adjacent segments.
        cloudyfsps_only : bool
            If True, return only the 128 lines matching the CLOUDY/FSPS grid.
            If False, return all 138 lines.

        Returns
        -------
        wavelengths : array
            Rest-frame wavelengths in Angstrom.
        luminosities : array
            Line luminosities in Lsun.
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

        wav, lum = predict_all_lines(
            nn_params, self.weights,
            gas_logq, jnp.asarray(gas_logqion, dtype=jnp.float32),
        )

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
        gas_logqion: float = None,
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
            nn_params, self.weights,
            gas_logq, jnp.asarray(gas_logqion, dtype=jnp.float32),
        )

    def predict_nebular_sed(
        self,
        ssp_wave: jnp.ndarray,
        line_sigma_aa: float = 0.0,
        **neb_params,
    ) -> jnp.ndarray:
        """Predict total nebular emission on an arbitrary wavelength grid.

        Combines emission lines + nebular continuum, interpolated onto
        the provided wavelength grid.

        Parameters
        ----------
        ssp_wave : array, shape (n_wave,)
            Output wavelength grid in Angstrom.
        line_sigma_aa : float
            Gaussian width for emission lines (Angstrom). 0 = delta function.
        **neb_params
            All Cue nebular parameters (see predict_nebular_line_luminosities).

        Returns
        -------
        array, shape (n_wave,)
            Total nebular SED in Lsun/Hz on the SSP wavelength grid.
        """
        # Lines
        line_wav, line_lum = self.predict_nebular_line_luminosities(
            cloudyfsps_only=False, **neb_params,
        )

        # Continuum
        cont_wav, cont_lum = self.predict_nebular_continuum(**neb_params)

        # Interpolate continuum onto SSP grid
        neb_sed = jnp.interp(ssp_wave, cont_wav, cont_lum, left=0.0, right=0.0)

        # Add emission lines
        c_angstrom = _C_CGS * 1e8  # Angstrom/s
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
