# ADR 0001: Adopt Architecture Decision Records

**Status**: Accepted

## Context

Tengri is developed with significant assistance from AI (Claude Code). Development happens in discrete sessions, sometimes with different models or revised instructions. Without a persistent record of *why* decisions were made, the rationale can be lost between sessions. The codebase can drift—old decisions get reconsidered or forgotten, and new contributors (human or AI) lack context for understanding the architecture.

A lightweight, searchable archive of major decisions helps everyone—current and future contributors, and future AI tools—make informed choices without re-litigating settled questions.

## Decision

Adopt Architecture Decision Records (ADRs) for tengri. Store them in `docs/adr/`, numbered sequentially (0001, 0002, ...) with kebab-case titles. Each ADR documents a significant decision with its context, the choice made, and the consequences.

## Consequences

**Benefits:**
- Permanent, searchable record of why things are the way they are.
- New contributors and future development sessions start informed.
- Decisions can be updated or superseded with full traceability.

**Trade-offs:**
- Small overhead: authors must document decisions after they are made.
- ADRs work best if the team uses them consistently (no enforced automation; relies on discipline).
- Outdated ADRs can mislead if not kept current; expect periodic reviews.

**Mitigation:**
- Aim to write an ADR within a week of each major decision.
- Review ADRs quarterly for accuracy; mark obsolete ones `Superseded`.
