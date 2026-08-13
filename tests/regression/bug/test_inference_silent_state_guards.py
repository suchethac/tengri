# SPDX-License-Identifier: BSD-3-Clause
"""Eigenvalue clipping assigns a variance; it must not do so silently (#1515).

``run_laplace`` floors the Hessian spectrum at ``min_eigenvalue`` to force
positive-definiteness, then takes ``cov = H^-1``. Because the covariance is the
*inverse*, the floor does not damp a clipped direction — it **assigns** it
variance ``1 / min_eigenvalue``: ``1e6``, i.e. std 1000, at the default. The
directions the data constrain least come back with the widest draws, and that
width is an artifact of the floor rather than a measurement.

The count was already computed, but surfaced only under ``if verbose``, which
is off by default, so a fit could return floor-derived error bars in silence.

Both directions are tested. A ceiling warning that also fired on healthy fits
would be filtered wholesale, which is how a guard stops protecting anything.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import LaplaceVarianceCeilingWarning
from tengri.inference.backends.laplace import _regularize_hessian

pytestmark = pytest.mark.regression_bug


def test_variance_ceiling_warns_and_reports_the_number():
    """A clipped eigenvalue must report the variance the floor assigns (#1515).

    The 2x2 with one exactly-zero eigenvalue is the shape an unidentifiable
    parameter pair produces — see #1095, where ``met_alpha_fe`` and
    ``met_logzsol`` are bit-identically degenerate on a standard (3D) SSP grid.
    """
    hessian = jnp.array([[4.0, 0.0], [0.0, 0.0]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _eigenvalues, clipped, _vecs, _hessian_reg, n_clipped = _regularize_hessian(
            hessian, min_eigenvalue=1e-6
        )

    assert n_clipped == 1, f"expected exactly one clipped direction, got {n_clipped}"
    assert float(jnp.min(clipped)) == pytest.approx(1e-6)

    ceiling = [w for w in caught if issubclass(w.category, LaplaceVarianceCeilingWarning)]
    assert ceiling, (
        f"a clipped eigenvalue produced no warning; got {[w.category.__name__ for w in caught]}"
    )

    text = str(ceiling[0].message)
    # The whole point of the warning is the variance the floor assigns.
    assert "1e+06" in text or "1000000" in text, (
        f"warning did not report the assigned variance: {text}"
    )
    assert "1/2" in text, f"warning did not report how many directions clipped: {text}"


def test_no_warning_when_nothing_clips():
    """A well-conditioned Hessian must stay quiet (#1515)."""
    hessian = jnp.array([[4.0, 0.0], [0.0, 2.0]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, _, _, n_clipped = _regularize_hessian(hessian, min_eigenvalue=1e-6)

    assert n_clipped == 0
    assert not [w for w in caught if issubclass(w.category, LaplaceVarianceCeilingWarning)]


def test_unregularized_path_is_untouched_and_silent():
    """``regularize=False`` returns the spectrum as-is and issues no warning.

    The unregularized path is how callers ask for an indefinite Hessian to be
    reported rather than repaired (#1537 relies on it), so the new warning must
    not intrude on it.
    """
    hessian = jnp.array([[4.0, 0.0], [0.0, 0.0]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eigenvalues, clipped, _vecs, hessian_reg, n_clipped = _regularize_hessian(
            hessian, min_eigenvalue=1e-6, regularize=False
        )

    assert n_clipped == 0
    assert np.array_equal(np.asarray(eigenvalues), np.asarray(clipped)), (
        "regularize=False must return the spectrum untouched"
    )
    assert np.array_equal(np.asarray(hessian_reg), np.asarray(hessian)), (
        "regularize=False must return the Hessian untouched"
    )
    assert not [w for w in caught if issubclass(w.category, LaplaceVarianceCeilingWarning)]


def test_reported_variance_tracks_the_floor():
    """The reported ceiling must follow ``min_eigenvalue``, not be hard-coded."""
    hessian = jnp.array([[4.0, 0.0], [0.0, 0.0]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _regularize_hessian(hessian, min_eigenvalue=1e-4)

    ceiling = [w for w in caught if issubclass(w.category, LaplaceVarianceCeilingWarning)]
    assert ceiling
    text = str(ceiling[0].message)
    assert "10000" in text, f"variance should be 1/1e-4 = 1e4, got: {text}"
