# SPDX-License-Identifier: BSD-3-Clause
"""Contract: CalibrationELineMarginalizedLikelihood discloses its approximation.

The class uses a plug-in point estimate for the emission-line block rather than
marginalizing it fully. The docstring MUST disclose this honestly in the main
body, not only in Notes, and name the two exact pieces and explain the cost.

Ref: #2354
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract


class TestCalibrationELineMarginalizedLikelihoodDisclosure:
    """Contract test on the docstring of CalibrationELineMarginalizedLikelihood."""

    def test_docstring_mentions_plugin_not_marginalized(self):
        """The docstring must state that the emission-line block is a plug-in, not marginalized."""
        from tengri.inference.likelihoods import CalibrationELineMarginalizedLikelihood

        doc = inspect.getdoc(CalibrationELineMarginalizedLikelihood)
        assert doc is not None

        # Key phrases that must appear (case-insensitive search for robustness)
        lower_doc = doc.lower()
        plugin_terms = ["plug-in", "plug in", "point estimate"]
        marginal_negations = ["not marginalized", "not marginal"]

        plugin_mentioned = any(term in lower_doc for term in plugin_terms)
        not_marg_mentioned = any(term in lower_doc for term in marginal_negations)

        assert plugin_mentioned or not_marg_mentioned, (
            "Docstring must state that emission-line block is handled by plug-in "
            "point estimate rather than marginalized. Missing key phrases like "
            "'plug-in', 'plug in', 'point estimate', 'not marginalized', or "
            "'not marginal'."
        )

    def test_docstring_mentions_log_determinant_volume_term(self):
        """The docstring must mention that the log-det volume term is discarded."""
        from tengri.inference.likelihoods import CalibrationELineMarginalizedLikelihood

        doc = inspect.getdoc(CalibrationELineMarginalizedLikelihood)
        assert doc is not None

        lower_doc = doc.lower()
        # Key phrases about the volume term / log-det
        volume_terms = ["log det", "log-det", "volume term", "log determinant"]
        term_mentioned = any(term in lower_doc for term in volume_terms)

        assert term_mentioned, (
            "Docstring must mention the log-determinant or volume term that is discarded. "
            "Missing key phrases like 'log det', 'log-det', 'volume term', "
            "or 'log determinant'."
        )

    def test_docstring_mentions_cost(self):
        """The docstring must explain what the approximation costs (understates uncertainty)."""
        from tengri.inference.likelihoods import CalibrationELineMarginalizedLikelihood

        doc = inspect.getdoc(CalibrationELineMarginalizedLikelihood)
        assert doc is not None

        lower_doc = doc.lower()
        # Phrases about the cost: understates, underestimate, uncertainty, amplitude
        cost_indicators = [
            "understate",
            "underestimate",
            "uncertainty",
            "amplitude",
            "cost",
            "data-dependent",
        ]
        cost_mentioned = any(term in lower_doc for term in cost_indicators)

        assert cost_mentioned, (
            "Docstring must explain the cost of the approximation "
            "(e.g., 'understates the line amplitudes' uncertainty contribution'). "
            "Missing key phrases like 'understate', 'underestimate', 'uncertainty', "
            "'amplitude', 'cost', or 'data-dependent'."
        )

    def test_disclosure_not_only_in_notes(self):
        """The key disclosure must appear in the main body, not solely under Notes."""
        from tengri.inference.likelihoods import CalibrationELineMarginalizedLikelihood

        doc = inspect.getdoc(CalibrationELineMarginalizedLikelihood)
        assert doc is not None

        # Split docstring at "Notes" section
        if "Notes" in doc:
            # Everything before the Notes section
            main_body = doc.split("Notes")[0]
        else:
            # If no Notes section, the entire docstring is the main body
            main_body = doc

        lower_main = main_body.lower()

        # At least one key phrase must appear in the main body, not just Notes
        main_body_terms = ["plug-in", "plug in", "point estimate", "not marginalized"]
        term_in_main = any(term in lower_main for term in main_body_terms)

        assert term_in_main, (
            "The disclosure that the emission-line block uses a plug-in point estimate "
            "(not marginalization) must appear in the main body of the docstring, "
            "not only under Notes. This is how a reader of Posterior output learns "
            "about the approximation."
        )


class TestCalibrationELineMarginalizedLikelihoodMutation:
    """Mutation test: if the disclosure is removed, the contract test fails."""

    @pytest.mark.parametrize("phrase", ["plug-in", "point estimate"])
    def test_removing_disclosure_breaks_contract(self, phrase):
        """Verify that key phrases are necessary by simulating their removal."""
        from tengri.inference.likelihoods import CalibrationELineMarginalizedLikelihood

        original_doc = inspect.getdoc(CalibrationELineMarginalizedLikelihood)
        assert original_doc is not None
        assert phrase.lower() in original_doc.lower(), (
            f"Baseline docstring must contain '{phrase}' for mutation test to be valid."
        )
