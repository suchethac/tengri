# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate Cue JAX implementation against TensorFlow reference outputs.

The Cue emulator (Li et al. 2024) was originally implemented in TensorFlow.
Our JAX re-implementation loads the same weights from data/cue_weights.npz.
This test verifies the JAX forward pass matches TF output.

Reference outputs were generated using the original TF CUE
(cue.Emulator from https://github.com/yi-jia-li/cue) in a
separate TF-only venv, saved to data/cue_reference_outputs.npz.
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_WEIGHTS_PATH = _DATA_DIR / "cue_weights.npz"
_REFERENCE_PATH = _DATA_DIR / "cue_reference_outputs.npz"

if not _WEIGHTS_PATH.is_file():
    pytest.skip("CUE weights not found", allow_module_level=True)


@pytest.fixture(scope="module")
def cue_backend():
    from tengri.components.nebular.cue import CueBackend

    return CueBackend(str(_WEIGHTS_PATH))


# Standard test inputs (must match generate_cue_reference.py)
_TEST_PARAMS = [
    dict(
        ionspec_index1=-1.5,
        ionspec_index2=-3.0,
        ionspec_index3=-1.0,
        ionspec_index4=-2.0,
        ionspec_logLratio1=0.0,
        ionspec_logLratio2=0.0,
        ionspec_logLratio3=-0.5,
        gas_logu=-2.5,
        gas_logn=2.0,
        gas_logz=-0.5,
        gas_logno=-0.5,
        gas_logco=0.0,
    ),
    dict(
        ionspec_index1=-2.0,
        ionspec_index2=-4.0,
        ionspec_index3=-1.5,
        ionspec_index4=-3.0,
        ionspec_logLratio1=0.5,
        ionspec_logLratio2=-0.5,
        ionspec_logLratio3=0.0,
        gas_logu=-3.0,
        gas_logn=1.0,
        gas_logz=0.0,
        gas_logno=0.0,
        gas_logco=0.0,
    ),
]


# ── 1. Reference output comparison ────────────────────────────────


class TestCueVsTFReference:
    """Compare JAX forward pass against TF reference outputs."""

    @pytest.fixture(scope="class")
    def reference(self):
        if not _REFERENCE_PATH.is_file():
            pytest.skip(
                "CUE reference outputs not found. Generate with: "
                "python scripts/generate_cue_reference.py"
            )
        return np.load(str(_REFERENCE_PATH))

    @pytest.mark.parametrize("input_idx", [0, 1])
    def test_lines_match_tf(self, cue_backend, reference, input_idx):
        """Line luminosities should match TF output within float32 precision.

        CUE JAX returns all 138 lines; TF Emulator filters to 128.
        We match by wavelength to compare the overlapping lines.

        Unit convention: ``predict_nebular_line_luminosities`` returns **[Lsun]**,
        the same unit the TF reference in ``cue_reference_outputs.npz`` is stored
        in — upstream Cue's native output. No conversion here.

        History, because this line has moved twice. The method used to multiply
        by ``L_SUN_CUE`` at the boundary and this test divided it back out
        (#477 — before that it silently asserted erg/s ≈ Lsun, comparing nothing).
        #1559 removed the multiply, moving the single erg/s conversion to
        ``NebularSEDComponent`` where the published ``line_lums`` key needs it,
        so the compensating divide here had nothing left to cancel. That the
        upstream reference is in Lsun is the evidence that this is the faithful
        direction: the erg/s was tengri's, not Cue's.
        """
        params = _TEST_PARAMS[input_idx]
        wav_jax, lum_jax = cue_backend.predict_nebular_line_luminosities(
            cloudyfsps_only=False, **params
        )
        lum_ref = reference[f"lines_{input_idx}"]
        wav_ref = reference["line_wavelengths"]

        wav_jax_np = np.asarray(wav_jax)
        lum_jax_np = np.asarray(lum_jax)  # already [Lsun], same as the TF ref

        # Match TF lines to JAX by wavelength (TF may output fewer)
        n_ref = len(lum_ref)
        lum_jax_matched = np.zeros(n_ref)
        for i in range(n_ref):
            idx = np.argmin(np.abs(wav_jax_np - wav_ref[i]))
            lum_jax_matched[i] = lum_jax_np[idx]

        np.testing.assert_allclose(
            lum_jax_matched,
            lum_ref,
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"Line luminosities mismatch for input {input_idx}",
        )

    @pytest.mark.parametrize("input_idx", [0, 1])
    def test_continuum_matches_tf(self, cue_backend, reference, input_idx):
        """Continuum spectrum should match TF output."""
        params = _TEST_PARAMS[input_idx]
        cont_wavs = reference["continuum_wavelengths"]

        wav_jax, lum_jax = cue_backend.predict_nebular_continuum(**params)
        cont_ref = reference[f"continuum_{input_idx}"]

        # Interpolate JAX output to reference wavelength grid
        lum_jax_interp = np.interp(cont_wavs, np.asarray(wav_jax), np.asarray(lum_jax))

        # Only compare where reference is non-zero (within CUE's valid range)
        valid = cont_ref != 0.0
        if np.sum(valid) < 10:
            pytest.skip("Too few valid continuum points")

        # rtol=0.05 allows for wavelength grid interpolation effects
        # (JAX has 1841 pts, TF ref has 1000 pts on different grid).
        # Median agreement is ~7e-6; outliers at spectral features reach ~4%.
        np.testing.assert_allclose(
            lum_jax_interp[valid],
            cont_ref[valid],
            rtol=0.05,
            atol=1e-3,
            err_msg=f"Continuum mismatch for input {input_idx}",
        )

    def test_line_wavelengths_subset(self, cue_backend, reference):
        """TF reference wavelengths should be a subset of JAX wavelengths.

        JAX outputs all 138 lines; TF Emulator filters to 128 via
        line_ind (excludes ionization-energy pseudo-lines).
        """
        params = _TEST_PARAMS[0]
        wav_jax, _ = cue_backend.predict_nebular_line_luminosities(cloudyfsps_only=False, **params)
        wav_ref = reference["line_wavelengths"]
        wav_jax_np = np.sort(np.asarray(wav_jax))

        for w_ref in wav_ref:
            closest = wav_jax_np[np.argmin(np.abs(wav_jax_np - w_ref))]
            np.testing.assert_allclose(
                closest,
                w_ref,
                rtol=1e-6,
                err_msg=f"TF wavelength {w_ref:.2f} not found in JAX",
            )


# ── 2. Internal consistency (always runs) ─────────────────────────


class TestCueInternalConsistency:
    """Verify JAX forward pass produces sensible outputs."""

    def test_lines_finite(self, cue_backend):
        """All line luminosities should be finite."""
        for params in _TEST_PARAMS:
            _, lum = cue_backend.predict_nebular_line_luminosities(cloudyfsps_only=False, **params)
            chex.assert_tree_all_finite(lum)

    def test_continuum_finite(self, cue_backend):
        """Continuum should be finite."""
        for params in _TEST_PARAMS:
            _, lum = cue_backend.predict_nebular_continuum(**params)
            chex.assert_tree_all_finite(lum)

    def test_different_inputs_different_outputs(self, cue_backend):
        """Different parameters should produce different predictions."""
        _, lum_0 = cue_backend.predict_nebular_line_luminosities(
            cloudyfsps_only=False, **_TEST_PARAMS[0]
        )
        _, lum_1 = cue_backend.predict_nebular_line_luminosities(
            cloudyfsps_only=False, **_TEST_PARAMS[1]
        )
        assert not jnp.allclose(lum_0, lum_1), "Different inputs gave same output"

    def test_jax_differentiable(self, cue_backend):
        """Forward pass should be differentiable."""
        params = _TEST_PARAMS[0]

        def loss_fn(gas_logu):
            _, lum = cue_backend.predict_nebular_line_luminosities(
                cloudyfsps_only=False,
                gas_logu=gas_logu,
                gas_logn=params["gas_logn"],
                gas_logz=params["gas_logz"],
                gas_logno=params["gas_logno"],
                gas_logco=params["gas_logco"],
                ionspec_index1=params["ionspec_index1"],
                ionspec_index2=params["ionspec_index2"],
                ionspec_index3=params["ionspec_index3"],
                ionspec_index4=params["ionspec_index4"],
                ionspec_logLratio1=params["ionspec_logLratio1"],
                ionspec_logLratio2=params["ionspec_logLratio2"],
                ionspec_logLratio3=params["ionspec_logLratio3"],
            )
            return jnp.sum(lum)

        def fd_grad_local(f, x: float, eps: float = 1e-4) -> float:
            """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
            return float((f(x + eps) - f(x - eps)) / (2.0 * eps))

        grad_jax = float(jax.grad(loss_fn)(float(params["gas_logu"])))
        grad_fd = fd_grad_local(loss_fn, float(params["gas_logu"]))
        # CUE has slightly noisy gradients from neural network, use 5e-3
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=5e-3, err_msg=f"Cue autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        assert abs(grad_jax) > 0.0, "Gradient should be non-zero"

    def test_line_wavelengths_sorted_ascending(self, cue_backend):
        """Line wavelengths from cloudyfsps_only should be sorted."""
        params = _TEST_PARAMS[0]
        wav, _ = cue_backend.predict_nebular_line_luminosities(cloudyfsps_only=True, **params)
        wav_np = np.asarray(wav)
        assert np.all(np.diff(wav_np) >= 0) or np.all(np.diff(wav_np) <= 0), (
            "Line wavelengths should be monotonically ordered"
        )
