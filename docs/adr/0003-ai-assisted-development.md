# ADR 0003: Treat AI-assisted development as the primary mode

**Status:** Accepted

**Date:** 2026-04-23

## Context

The initial tengri codebase was drafted largely with AI tools (Claude Code). Denying this is dishonest and tactically weak. Industry retrospectives (2024–2025) report AI code is correct approximately 2/3 of the time. This creates a verification burden, not a taboo.

## Decision

Acknowledge AI assistance openly. Compensate with stricter verification discipline than typical hand-written code.

**Four rules:**

1. **Primary sources.** Every physics module must cite a primary paper and include a regression test validating against it (see docs/dev/verification-protocol.md).

2. **Reference-anchored over invented.** When an upstream reference implementation exists (Prospector, DSPS, bagpipes, Cue, NIFTy), tengri's version independently implements the same published algorithm, names the reference symbol in a verifiable comment block, and is validated against the reference — not a reinvention and not a copy. Header example:
   ```python
   # Implements the same model as bagpipes (Carnall et al. 2019 [1]_); validated against it. JAX-differentiable.
   ```

3. **PR transparency.** Every PR includes an AI-use disclosure checkbox in the PR template (see .github/pull_request_template.md).

4. **Human governance.** Releases are cut by humans. AI is never listed as an author.

## Consequences

**Positive:**
- Visible honesty. This turns the riskiest part of the story into its most credible.
- Enables stricter verification standards than typical projects.
- Attracts contributors who respect transparency.

**Negative:**
- Slower velocity than a vibe-coded project.
- Higher documentation burden (citations, regression tests).

**Mitigation:** Automate test scaffolding and citation linting. Reserve AI for exploration phases, humans for architecture and verification.
