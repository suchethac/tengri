# Governance

## Current Model

Tengri is led by Suchetha Cooray as a single maintainer (BDFL-lite model). All major decisions rest with the maintainer. The project is pre-1.0, so scope and API may change.

## Response Expectations

Issue triage typically occurs within two weeks. Response time for pull requests depends on complexity and maintainer availability. For urgent security issues, please contact cooray@stanford.edu directly.

## Decision Process

**Small, low-risk changes** (bug fixes, documentation, minor enhancements, utility functions) may be merged directly by the maintainer after code review.

**Non-trivial public-API or file-format changes** must be proposed in a GitHub issue labelled `rfc` (request for comments) before work begins. This ensures community input and avoids wasted effort on rejected designs.

**Architectural decisions** are documented as ADRs (Architecture Decision Records) in `docs/adr/`. See the ADR template in that directory for the format.

## Succession Plan

If the maintainer is unavailable for more than six months and cannot delegate decision authority, the project may be:

1. Transferred to an Astropy affiliated package for continued maintenance.
2. Frozen with an archive notice and a tagged release indicating end-of-life.
3. Handed to a named collaborator (announced in advance when possible).

The maintainer will attempt to notify active contributors and the community before any such action.
