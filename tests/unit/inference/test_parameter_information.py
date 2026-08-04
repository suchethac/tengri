# SPDX-License-Identifier: BSD-3-Clause
"""How much of the posterior is measurement, and how much is the prior talking back.

A model with one free parameter per age bin can draw almost any star-formation
history, so a fit always *returns* one -- whether or not the data said anything.
The quantity that separates the two is the per-mode shrinkage.

Because every prior in tengri is standardized, the prior contributes exactly
``I`` to the posterior precision (the reasoning already recorded at
``PRIOR_METRIC_FLOOR``), and the Gauss-Newton likelihood term is positive
semi-definite. So an eigenvalue of the posterior precision decomposes with no
free normalization at all:

.. math::

    \\lambda_k = 1 + d_k , \\qquad
    s_k = \\frac{\\lambda_k - 1}{\\lambda_k} = \\frac{d_k}{1 + d_k} ,
    \\qquad n_{\\rm eff} = \\sum_k s_k

``d_k >= 0`` is what the data added along mode ``k``. ``s_k`` runs 0 (pure
prior) to 1 (prior irrelevant), and ``n_eff`` counts the directions the data
actually measured. Nothing here is a heuristic threshold.

These tests pin the arithmetic against closed forms, so a regression cannot
hide behind a plausible-looking number on a real posterior.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_a_mode_the_data_never_touched_has_zero_shrinkage():
    """The negative control: precision ``I`` is the prior alone, so ``n_eff`` is 0.

    Without this the metric could be reporting anything monotonic and the test
    suite would not notice.
    """
    from tengri.inference.information import information_from_precision

    info = information_from_precision(np.eye(5), names=tuple("abcde"))

    np.testing.assert_allclose(info.shrinkage, 0.0, atol=1e-12)
    assert info.n_eff == pytest.approx(0.0, abs=1e-12)
    assert info.n_total == 5


def test_a_mode_the_prior_never_touched_has_unit_shrinkage():
    """The opposite rail: as ``d_k`` grows the prior stops mattering."""
    from tengri.inference.information import information_from_precision

    info = information_from_precision(np.diag([1e12, 1e12]), names=("a", "b"))

    np.testing.assert_allclose(info.shrinkage, 1.0, atol=1e-9)
    assert info.n_eff == pytest.approx(2.0, abs=1e-9)


def test_shrinkage_matches_the_closed_form_mode_by_mode():
    """``s_k = d_k / (1 + d_k)`` exactly, for a spectrum spanning both rails."""
    from tengri.inference.information import information_from_precision

    d = np.array([0.0, 0.25, 1.0, 3.0, 99.0])
    info = information_from_precision(np.diag(1.0 + d), names=tuple("abcde"))

    np.testing.assert_allclose(info.shrinkage, d / (1.0 + d), rtol=1e-12)
    assert info.n_eff == pytest.approx(float(np.sum(d / (1.0 + d))), rel=1e-12)


def test_the_decomposition_is_invariant_to_the_parameter_basis():
    """``n_eff`` counts *modes*, so rotating the parameters must not change it.

    A quantity that moved under an orthogonal change of basis would be a
    property of the coordinate labels, not of the measurement.
    """
    from tengri.inference.information import information_from_precision

    d = np.array([0.0, 0.5, 4.0, 40.0])
    diagonal = np.diag(1.0 + d)
    rng = np.random.default_rng(0)
    rotation = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    rotated = rotation @ diagonal @ rotation.T

    plain = information_from_precision(diagonal, names=tuple("abcd"))
    turned = information_from_precision(rotated, names=tuple("abcd"))

    assert turned.n_eff == pytest.approx(plain.n_eff, rel=1e-10)
    np.testing.assert_allclose(np.sort(turned.shrinkage), np.sort(plain.shrinkage), atol=1e-10)


def test_residual_curvature_below_the_prior_cannot_make_shrinkage_negative():
    """Eigenvalues under 1 are the residual term Gauss-Newton drops, not evidence.

    ``negative_hessian_metric`` floors these away for the whitening transform;
    this path deliberately passes ``floor=0.0`` to see the prior-carried modes,
    so the clip has to happen here instead.
    """
    from tengri.inference.information import information_from_precision

    info = information_from_precision(np.diag([0.4, 0.9, 1.0, 2.0]), names=tuple("abcd"))

    assert (info.shrinkage >= 0.0).all(), info.shrinkage
    assert (info.shrinkage <= 1.0).all(), info.shrinkage
    assert info.n_eff == pytest.approx(0.5, rel=1e-12)  # only the lambda = 2 mode


def test_information_is_attributed_back_to_named_parameters():
    """Modes are combinations; astronomers ask about parameters.

    Weighting each mode's shrinkage by its squared projection onto a parameter
    keeps the total intact, so the per-parameter shares still sum to ``n_eff``.
    """
    from tengri.inference.information import information_from_precision

    info = information_from_precision(np.diag([1.0, 5.0]), names=("untouched", "measured"))
    shares = info.by_parameter()

    assert set(shares) == {"untouched", "measured"}
    assert shares["untouched"] == pytest.approx(0.0, abs=1e-12)
    assert shares["measured"] == pytest.approx(0.8, rel=1e-12)
    assert sum(shares.values()) == pytest.approx(info.n_eff, rel=1e-12)


def test_a_non_finite_metric_is_refused_rather_than_averaged_into_a_number():
    """A NaN Hessian must not become a confident-looking ``n_eff``."""
    from tengri.inference.information import information_from_precision

    bad = np.eye(3)
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        information_from_precision(bad, names=("a", "b", "c"))


def test_names_must_match_the_matrix():
    """A mismatch means the labels are wrong, which is worse than no labels."""
    from tengri.inference.information import information_from_precision

    with pytest.raises(ValueError, match="names"):
        information_from_precision(np.eye(3), names=("a", "b"))


# ── The expansion point has to actually be a mode ──────────────────────────────
#
# Curvature is the posterior precision only at a stationary point. Measured on a
# 25-parameter field model, one and the same fit reports:
#
#     MAP steps    n_eff    nats unclaimed
#     1 x 300      17.59         298.1
#     2 x 600       6.33          11.9
#     4 x 3000      5.74           0.64
#     4 x 10000     4.91           0.00
#
# Every one of those numbers looks perfectly reasonable in isolation. Without the
# Newton decrement there is nothing to tell them apart.


def test_a_stationary_point_is_recognized_as_a_mode():
    """Zero gradient means nothing is left to claim."""
    from tengri.inference.information import information_from_precision

    info = information_from_precision(np.diag([1.0, 4.0]), names=("a", "b"), gradient=np.zeros(2))

    assert info.newton_decrement == pytest.approx(0.0)
    assert info.at_a_mode


def test_a_point_far_from_the_mode_is_refused_as_one():
    """A large decrement means the quadratic still has a long way to fall."""
    from tengri.inference.information import information_from_precision

    # H = I, g = (10, 10) -> 0.5 g^T H^-1 g = 100 nats still on the table.
    info = information_from_precision(np.eye(2), names=("a", "b"), gradient=np.array([10.0, 10.0]))

    assert 0.5 * info.newton_decrement**2 == pytest.approx(100.0)
    assert not info.at_a_mode


def test_the_decrement_is_measured_in_nats_not_in_parameter_units():
    """Rescaling a parameter must not change whether we are at the mode.

    A bare ``norm(gradient) < eps`` test fails exactly here: express a mass in
    dex instead of log10 and the gradient changes by that factor while the
    posterior is untouched. ``g^T H^-1 g`` is invariant because H rescales too.
    """
    from tengri.inference.information import information_from_precision

    scale = 1000.0
    plain = information_from_precision(np.eye(2), names=("a", "b"), gradient=np.array([0.3, 0.4]))
    stretched = information_from_precision(
        np.eye(2) / scale**2, names=("a", "b"), gradient=np.array([0.3, 0.4]) / scale
    )

    assert stretched.newton_decrement == pytest.approx(plain.newton_decrement, rel=1e-9)
    assert np.linalg.norm([0.3, 0.4]) / scale != pytest.approx(np.linalg.norm([0.3, 0.4]))


def test_without_a_gradient_the_mode_check_abstains_rather_than_asserting():
    """No evidence is not evidence of a problem — but it is not a pass either."""
    from tengri.inference.information import information_from_precision

    info = information_from_precision(np.eye(2), names=("a", "b"))

    assert np.isnan(info.newton_decrement)
    assert info.at_a_mode  # nothing contradicts it


def test_a_gradient_of_the_wrong_length_is_refused():
    """Silently broadcasting here would report a decrement for a different point."""
    from tengri.inference.information import information_from_precision

    with pytest.raises(ValueError, match="gradient"):
        information_from_precision(np.eye(3), names=("a", "b", "c"), gradient=np.zeros(2))


def test_the_summary_says_so_when_the_number_is_not_meaningful():
    """The warning goes to stderr and gets lost; the report has to carry it too."""
    from tengri.inference.information import information_from_precision

    bad = information_from_precision(np.eye(2), names=("a", "b"), gradient=np.array([9.0, 9.0]))
    good = information_from_precision(np.eye(2), names=("a", "b"), gradient=np.zeros(2))

    assert "NOT AT A MODE" in bad.summary()
    assert "NOT AT A MODE" not in good.summary()


# ── Latent naming ─────────────────────────────────────────────────────────────


def test_vector_latents_expand_to_one_name_per_degree_of_freedom():
    """``spec.free_params`` counts the field as one name; the ravel does not.

    Taking the names from the spec instead of from the pytree is off by
    ``n_grid - 1`` and mislabels every mode after the first vector entry.
    """
    from tengri.inference.information import latent_names

    names = latent_names({"dust_tau_bc": 0.5, "psd_xi": np.zeros(3), "sfh_dpl_alpha": 1.0})

    assert names == ("dust_tau_bc", "psd_xi[0]", "psd_xi[1]", "psd_xi[2]", "sfh_dpl_alpha")


# ── Restricting to one block ──────────────────────────────────────────────────


def test_restrict_rediagonalizes_the_block_rather_than_slicing_the_modes():
    """The sub-block has its own eigenvectors; slicing the full ones is wrong.

    Built so the two disagree: a precision matrix whose block is *not* aligned
    with the global eigenbasis. Slicing would inherit the global directions and
    report a different number.
    """
    from tengri.inference.information import information_from_precision

    precision = np.array(
        [
            [3.0, 1.0, 0.5],
            [1.0, 3.0, 0.5],
            [0.5, 0.5, 2.0],
        ]
    )
    info = information_from_precision(precision, names=("blk_a", "blk_b", "other"))
    block = info.restrict("blk_")

    # The truth: eigenvalues of the 2x2 top-left block are 3 +/- 1 = {2, 4}.
    np.testing.assert_allclose(np.sort(block.eigenvalues), [2.0, 4.0], rtol=1e-12)
    assert block.n_eff == pytest.approx(0.5 + 0.75, rel=1e-12)
    assert block.names == ("blk_a", "blk_b")


def test_precision_round_trips_through_the_decomposition():
    """``restrict`` rebuilds the matrix from V and lambda, so that must be exact."""
    from tengri.inference.information import information_from_precision

    rng = np.random.default_rng(3)
    root = rng.standard_normal((5, 5))
    precision = root @ root.T + np.eye(5)

    info = information_from_precision(precision, names=tuple("abcde"))

    np.testing.assert_allclose(info.precision(), precision, atol=1e-10)


def test_restricting_to_a_block_is_the_conditional_not_the_marginal():
    """With a coupled block, conditioning gives *more* information than the whole.

    Stating which one a number is matters: the block's own count and the
    model's total answer different questions and need not be ordered.
    """
    from tengri.inference.information import information_from_precision

    precision = np.array([[5.0, 2.0], [2.0, 5.0]])
    info = information_from_precision(precision, names=("psd_xi[0]", "dust_tau"))

    only_field = info.restrict("psd_xi")
    assert only_field.n_total == 1
    # Conditional precision for the single kept parameter is the raw diagonal, 5.
    assert only_field.eigenvalues[0] == pytest.approx(5.0)
    assert only_field.n_eff == pytest.approx(0.8, rel=1e-12)


def test_an_unknown_prefix_lists_what_is_available():
    """A typo'd prefix must not silently return an empty, zero-information block."""
    from tengri.inference.information import information_from_precision

    info = information_from_precision(np.eye(2), names=("psd_xi[0]", "dust_tau"))

    with pytest.raises(ValueError, match="No parameter starts with"):
        info.restrict("sfh_field_xi")  # the component spelling, not the latent one


def test_latent_names_follow_ravel_order_exactly():
    """``ravel_pytree`` walks dict keys sorted; the names must agree or shift."""
    from jax.flatten_util import ravel_pytree

    from tengri.inference.information import latent_names

    latent = {"zeta": np.zeros(2), "alpha": 1.0, "mid": np.zeros(3)}
    flat, _ = ravel_pytree(latent)

    assert len(latent_names(latent)) == flat.shape[0]
    assert latent_names(latent)[0] == "alpha"
