#!/usr/bin/env python3
"""Convert Cue neural network weights from TensorFlow pickle files to a single npz.

This script requires dill and numpy (and indirectly sklearn, since the PCA
pickles contain IncrementalPCA objects). The output npz file does NOT require
TensorFlow or dill to load.

Usage
-----
    python scripts/convert_cue_weights.py [--cue-dir /path/to/cue/data] [--output data/cue_weights.npz]

The script reads:
    - speculator_line_new_*.pkl   (16 line sub-network weights)
    - speculator_cont_new.pkl     (1 continuum sub-network weights)
    - pca_line_new_*.pkl          (16 line PCA objects)
    - pca_cont_new.pkl            (1 continuum PCA object)
    - lineList_wav.npy            (line wavelengths)
    - lineList_replaceblnd_name.npy  (line names)
    - FSPSlam.dat                 (continuum wavelength grid)

And writes everything to a single flat-dict npz file.

References
----------
- Li et al. 2025, Cue: https://github.com/yi-jia-li/cue
"""

import argparse
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Comprehensive TensorFlow mock so dill/pickle can resolve references to
# TF classes without importing TensorFlow itself (which segfaults on macOS
# with certain builds).
#
# The pickle files reference:
#   - tensorflow.python.trackable.data_structures.ListWrapper (just a list)
#   - Cue's own classes (nn.Speculator, cont_pca.SpectrumPCA) which do
#     `import tensorflow as tf` at module level and access tf.float32,
#     tf.function, tf.keras.Model, tf.keras.optimizers.Adam
# ---------------------------------------------------------------------------


class _MockTFModule(types.ModuleType):
    """Module that returns child mocks for any attribute access."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        child = types.ModuleType(f"{self.__name__}.{name}")
        child.__class__ = _MockTFModule
        sys.modules[child.__name__] = child
        return child

    def __call__(self, *args, **kwargs):
        return self


def _install_tf_mock():
    """Install a comprehensive TF mock into sys.modules."""
    tf = _MockTFModule("tensorflow")
    tf.float32 = "float32"
    tf.function = lambda f: f  # decorator passthrough

    class _FakeModel:
        pass

    keras = _MockTFModule("tensorflow.keras")
    keras.Model = _FakeModel
    optimizers = _MockTFModule("tensorflow.keras.optimizers")
    optimizers.Adam = lambda **kw: None
    keras.optimizers = optimizers
    tf.keras = keras

    ds = _MockTFModule("tensorflow.python.trackable.data_structures")
    ds.ListWrapper = list

    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.keras"] = keras
    sys.modules["tensorflow.keras.optimizers"] = optimizers
    sys.modules["tensorflow.python"] = _MockTFModule("tensorflow.python")
    sys.modules["tensorflow.python.trackable"] = _MockTFModule(
        "tensorflow.python.trackable"
    )
    sys.modules["tensorflow.python.trackable.data_structures"] = ds


_install_tf_mock()

import dill as pickle  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Sub-network names (must match nn_name in cue/utils.py)
# ---------------------------------------------------------------------------
LINE_NAMES = [
    "H1", "He1", "He2", "C1", "C2C3", "C4", "N", "O1",
    "O2", "O3", "ionE_1", "ionE_2", "S4", "Ar4", "Ne3", "Ne4",
]


def _load_speculator_pkl(filepath: str) -> dict:
    """Load a speculator pickle and return numpy arrays.

    The pickle contains 18 attributes in order:
    [W_, b_, alphas_, betas_, parameters_shift_, parameters_scale_,
     pca_shift_, pca_scale_, log_spectrum_shift_, log_spectrum_scale_,
     pca_transform_matrix_, n_parameters, n_wavelengths, wavelengths,
     n_pcas, n_hidden, n_layers, architecture]
    """
    with open(filepath, "rb") as f:
        attrs = pickle.load(f)

    (W, b, alphas, betas, param_shift, param_scale,
     pca_shift, pca_scale, log_spec_shift, log_spec_scale,
     pca_transform_matrix, n_parameters, n_wavelengths, wavelengths,
     n_pcas, n_hidden, n_layers, architecture) = attrs

    return {
        "W": W,
        "b": b,
        "alphas": alphas,
        "betas": betas,
        "parameters_shift": np.asarray(param_shift, dtype=np.float32),
        "parameters_scale": np.asarray(param_scale, dtype=np.float32),
        "pca_shift": np.asarray(pca_shift, dtype=np.float32),
        "pca_scale": np.asarray(pca_scale, dtype=np.float32),
        "log_spectrum_shift": np.asarray(log_spec_shift, dtype=np.float32),
        "log_spectrum_scale": np.asarray(log_spec_scale, dtype=np.float32),
        "pca_transform_matrix": np.asarray(pca_transform_matrix, dtype=np.float32),
        "n_parameters": int(n_parameters),
        "n_wavelengths": int(n_wavelengths),
        "wavelengths": np.asarray(wavelengths) if wavelengths is not None else None,
        "n_pcas": int(n_pcas),
        "n_hidden": list(n_hidden),
        "n_layers": int(n_layers),
        "architecture": list(architecture),
    }


def _load_pca_pkl(filepath: str) -> dict:
    """Load a PCA pickle and extract the IncrementalPCA components and mean.

    The PCA objects use sklearn IncrementalPCA. For our JAX reimplementation
    we only need:
        - components_  (n_pcas, n_wavelengths)  — the PCA basis vectors
        - mean_        (n_wavelengths,)         — the PCA centering vector

    inverse_transform(X) = X @ components_ + mean_
    """
    with open(filepath, "rb") as f:
        pca_obj = pickle.load(f)

    return {
        "components": np.asarray(pca_obj.PCA.components_, dtype=np.float32),
        "mean": np.asarray(pca_obj.PCA.mean_, dtype=np.float32),
    }


def _flatten_nn_weights(prefix: str, nn: dict, npz_dict: dict) -> None:
    """Flatten a single sub-network's arrays into the npz dict.

    Keys are like:
        line_H1_W_0, line_H1_b_0, line_H1_alpha_0, line_H1_beta_0,
        line_H1_parameters_shift, line_H1_pca_shift, ...
    """
    n_layers = nn["n_layers"]
    for i, w in enumerate(nn["W"]):
        npz_dict[f"{prefix}_W_{i}"] = np.asarray(w, dtype=np.float32)
    for i, b in enumerate(nn["b"]):
        npz_dict[f"{prefix}_b_{i}"] = np.asarray(b, dtype=np.float32)
    for i, a in enumerate(nn["alphas"]):
        npz_dict[f"{prefix}_alpha_{i}"] = np.asarray(a, dtype=np.float32)
    for i, b in enumerate(nn["betas"]):
        npz_dict[f"{prefix}_beta_{i}"] = np.asarray(b, dtype=np.float32)

    for key in ("parameters_shift", "parameters_scale",
                "pca_shift", "pca_scale",
                "log_spectrum_shift", "log_spectrum_scale",
                "pca_transform_matrix"):
        npz_dict[f"{prefix}_{key}"] = nn[key]

    npz_dict[f"{prefix}_n_layers"] = np.array(n_layers, dtype=np.int32)
    npz_dict[f"{prefix}_architecture"] = np.array(nn["architecture"], dtype=np.int32)


def _flatten_pca_weights(prefix: str, pca: dict, npz_dict: dict) -> None:
    """Flatten PCA components and mean into the npz dict."""
    npz_dict[f"{prefix}_pca_components"] = pca["components"]
    npz_dict[f"{prefix}_pca_mean"] = pca["mean"]


def convert(cue_dir: str, output_path: str) -> None:
    """Main conversion: read all pickles, write one npz."""
    cue_dir = Path(cue_dir)
    npz = {}

    # ------------------------------------------------------------------
    # Line sub-networks (16)
    # ------------------------------------------------------------------
    print(f"Loading {len(LINE_NAMES)} line sub-networks...")
    for name in LINE_NAMES:
        nn_path = cue_dir / f"speculator_line_new_{name}.pkl"
        pca_path = cue_dir / f"pca_line_new_{name}.pkl"

        if not nn_path.exists():
            print(f"  WARNING: {nn_path} not found, skipping {name}")
            continue
        if not pca_path.exists():
            print(f"  WARNING: {pca_path} not found, skipping {name}")
            continue

        nn = _load_speculator_pkl(str(nn_path))
        pca = _load_pca_pkl(str(pca_path))

        prefix = f"line_{name}"
        _flatten_nn_weights(prefix, nn, npz)
        _flatten_pca_weights(prefix, pca, npz)
        print(f"  {name}: {nn['n_layers']} layers, architecture={nn['architecture']}")

    # ------------------------------------------------------------------
    # Continuum sub-network (1)
    # ------------------------------------------------------------------
    print("Loading continuum sub-network...")
    cont_nn_path = cue_dir / "speculator_cont_new.pkl"
    cont_pca_path = cue_dir / "pca_cont_new.pkl"

    if not cont_nn_path.exists():
        print(f"ERROR: {cont_nn_path} not found")
        sys.exit(1)
    if not cont_pca_path.exists():
        print(f"ERROR: {cont_pca_path} not found")
        sys.exit(1)

    cont_nn = _load_speculator_pkl(str(cont_nn_path))
    cont_pca = _load_pca_pkl(str(cont_pca_path))

    _flatten_nn_weights("cont", cont_nn, npz)
    _flatten_pca_weights("cont", cont_pca, npz)
    print(f"  continuum: {cont_nn['n_layers']} layers, "
          f"architecture={cont_nn['architecture']}")

    # ------------------------------------------------------------------
    # Wavelength grids and line metadata
    # ------------------------------------------------------------------
    print("Loading wavelength grids and line metadata...")

    # Line wavelengths and names
    line_wav_path = cue_dir / "lineList_wav.npy"
    line_name_path = cue_dir / "lineList_replaceblnd_name.npy"

    if line_wav_path.exists():
        line_wav = np.load(str(line_wav_path))
        npz["lineList_wav"] = np.asarray(line_wav, dtype=np.float64)
        print(f"  lineList_wav: {line_wav.shape}")
    else:
        print(f"  WARNING: {line_wav_path} not found")

    if line_name_path.exists():
        line_names_arr = np.load(str(line_name_path))
        # Store as bytes for npz compatibility
        npz["lineList_name"] = line_names_arr
        print(f"  lineList_name: {line_names_arr.shape}")
    else:
        print(f"  WARNING: {line_name_path} not found")

    # Continuum wavelength grid (FSPSlam.dat)
    fsps_lam_path = cue_dir / "FSPSlam.dat"
    if fsps_lam_path.exists():
        cont_lam = np.genfromtxt(str(fsps_lam_path))
        npz["cont_wavelength_full"] = np.asarray(cont_lam, dtype=np.float64)
        # Cue uses cont_lam[122:] — wavelengths redward of Lyman limit
        npz["cont_wavelength"] = np.asarray(cont_lam[122:], dtype=np.float64)
        print(f"  cont_wavelength: {cont_lam[122:].shape} "
              f"(from {cont_lam[122]:.1f} to {cont_lam[-1]:.1f} A)")
    else:
        print(f"  WARNING: {fsps_lam_path} not found")

    # ------------------------------------------------------------------
    # Metadata: sub-network names, ion groupings, index arrays
    # ------------------------------------------------------------------
    npz["line_network_names"] = np.array(LINE_NAMES, dtype="U10")
    npz["n_line_networks"] = np.array(len(LINE_NAMES), dtype=np.int32)

    # Reconstruct nn_ion groupings and wav_selection indices
    # (replicate the logic from cue/utils.py)
    if line_wav_path.exists() and line_name_path.exists():
        unsorted_lam = np.load(str(line_wav_path))
        unsorted_name = np.load(str(line_name_path))
        sorted_idx = np.argsort(unsorted_lam)
        sorted_lam = unsorted_lam[sorted_idx]
        sorted_name = unsorted_name[sorted_idx]
        ele_arr = np.array([n[:4].rstrip() for n in sorted_name])

        nn_ion = [
            ["H  1"], ["He 1"], ["He 2"], ["C  1"], ["C  2", "C  3"], ["C  4"],
            ["N  1", "N  2", "N  3"], ["O  1"], ["O  2"], ["O  3"],
            ["Mg 2", "Fe 2", "Si 2", "Al 2", "P  2", "S  2", "Cl 2", "Ar 2"],
            ["Al 3", "Si 3", "S  3", "Cl 3", "Ar 3", "Ne 2"],
            ["S  4"], ["Ar 4"], ["Ne 3"], ["Ne 4"],
        ]

        # For each sub-network, store which indices in the sorted line
        # array it predicts. Use variable-length by zero-padding.
        all_wav_sel = []
        max_n = 0
        for ion_set in nn_ion:
            flat_ions = [ion for sublist in ([ion_set] if isinstance(ion_set, str) else [ion_set]) for ion in sublist]
            sel = np.where(np.isin(ele_arr, flat_ions))[0]
            all_wav_sel.append(sel)
            max_n = max(max_n, len(sel))

        # Store each sub-network's wav selection
        for i, (name, sel) in enumerate(zip(LINE_NAMES, all_wav_sel)):
            npz[f"line_{name}_wav_selection"] = np.asarray(sel, dtype=np.int32)

        # The concatenated wavelength array that the NN predicts
        nn_wavelength = sorted_lam[np.concatenate(all_wav_sel)]
        npz["nn_line_wavelength"] = np.asarray(nn_wavelength, dtype=np.float64)

        # "line_old" index: lines that match the cloudyfsps subset
        line_new_added = np.where(
            (sorted_lam == 4685.68) | (sorted_lam == 1550.77) |
            (sorted_lam == 1548.19) | (sorted_lam == 1750.00) |
            (sorted_lam == 2424.28) | (sorted_lam == 1882.71) |
            (sorted_lam == 1892.03) | (sorted_lam == 1406.02) |
            (sorted_lam == 4711.26) | (sorted_lam == 4740.12)
        )[0]
        n_total = len(sorted_lam)
        line_old_idx = np.arange(n_total)[~np.isin(np.arange(n_total), line_new_added)]
        npz["line_old_idx"] = np.asarray(line_old_idx, dtype=np.int32)

        # Sorted line wavelengths
        npz["sorted_line_wavelength"] = np.asarray(sorted_lam, dtype=np.float64)
        npz["sorted_line_name"] = sorted_name
        npz["line_sort_idx"] = np.asarray(sorted_idx, dtype=np.int32)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **npz)

    # Report
    total_params = 0
    for key, val in npz.items():
        if isinstance(val, np.ndarray):
            total_params += val.size
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(npz)} arrays ({total_params:,} total elements) "
          f"to {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cue TF pickle weights to a single npz file."
    )
    parser.add_argument(
        "--cue-dir",
        default="/tmp/cue_astro/src/cue/data",
        help="Path to Cue data directory containing pkl/npy files.",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/cue_weights.npz",
        help="Output npz file path (relative to diffsed root).",
    )
    args = parser.parse_args()
    convert(args.cue_dir, args.output)


if __name__ == "__main__":
    main()
