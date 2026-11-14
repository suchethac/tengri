"""Closed-form Gaussian point readout in ln lambda (#3512, #3436).

The point readout of the sigma_v correction is the exact segment integral of B(lambda F)/lambda
against a unit Gaussian in ln lambda, not a lattice sample. Two independent references:

- the analytic reference of ``tests/_lsf_reference.py``, which convolves F (not lambda F). Its
  identity ``B(lambda F)(l_c)/lambda_c = e^{s^2/2} R(l_c + s^2)`` maps it onto this readout;
- brute-force quadrature of B(lambda F)/lambda on a fine ln-lambda grid, including the Lyman
  step (:func:`edge_interp` semantics).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.lyc import LYMAN_LIMIT_AA, edge_interp
from tengri.observation.spectrum import point_readout_gaussian_ln
from tests._lsf_reference import convolve_piecewise_linear

jax.config.update("jax_enable_x64", True)

_C_KMS = 299792.458


def _random_case(seed: int, n_pix: int = 30):
    rng = np.random.default_rng(seed)
    wn = np.unique(np.sort(rng.uniform(1000.0, 10000.0, rng.integers(3, 9))))
    fn = rng.uniform(0.1, 5.0, wn.size)
    z = float(rng.uniform(0.0, 0.5))
    wobs = np.sort(rng.uniform(wn[0] * (1 + z) * 1.02, wn[-1] * (1 + z) * 0.98, n_pix))
    sig_ln = rng.uniform(10.0, 2000.0, n_pix) / _C_KMS
    return wn, fn, z, wobs, sig_ln


def test_closed_form_matches_reference_identity_to_1e8():
    """Exact on both sides: the closed form equals the reference at the shifted centre."""
    for seed in range(8):
        wn, fn, z, wobs, s = _random_case(seed)
        got = np.asarray(
            point_readout_gaussian_ln(
                jnp.asarray(fn), jnp.asarray(wn), jnp.asarray(wobs), z, jnp.asarray(s)
            )
        )
        ref = convolve_piecewise_linear(wn, fn, wobs, z, s, quantity="nufnu")
        rel = np.max(np.abs(got - ref) / np.abs(ref))
        assert rel <= 1e-8, f"seed {seed}: closed form vs reference {rel:.3e}"


def test_closed_form_matches_quadrature_across_lyman_step():
    """Step model at the Lyman edge: the bracket cell is two constant segments."""
    wn = np.array([700.0, 850.0, 900.0, 950.0, 1100.0, 2000.0, 5000.0])
    fn = np.array([3.0, 1.5, 2.0, 0.5, 1.0, 0.8, 0.3])
    z = 0.0
    wobs = np.linspace(760.0, 4500.0, 25)
    s = np.full(wobs.shape, 300.0 / _C_KMS)
    got = np.asarray(
        point_readout_gaussian_ln(
            jnp.asarray(fn),
            jnp.asarray(wn),
            jnp.asarray(wobs),
            z,
            jnp.asarray(s),
            has_lyc_edge=True,
        )
    )
    x = np.linspace(np.log(wn[0]), np.log(wn[-1]), 4_000_001)
    f_fine = np.asarray(edge_interp(jnp.asarray(np.exp(x)), jnp.asarray(wn), jnp.asarray(fn)))
    lam_f = f_fine * np.exp(x)
    dx = x[1] - x[0]
    quad = np.empty(wobs.shape)
    for i, (w, sig) in enumerate(zip(wobs, s, strict=True)):
        xq = np.log(w / (1 + z))
        kern = np.exp(-0.5 * ((x - xq) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        quad[i] = np.sum(lam_f * kern) * dx / np.exp(xq)
    assert wn[1] < LYMAN_LIMIT_AA and wn[3] > LYMAN_LIMIT_AA
    rel = np.max(np.abs(got - quad) / np.abs(quad))
    assert rel <= 1e-6, f"Lyman-step closed form vs quadrature {rel:.3e}"
