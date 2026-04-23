# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the tengri project. ADRs document significant design and technical decisions, the context behind them, and their implications.

## File Naming

Each ADR is a separate file named with a four-digit number and a kebab-case title:
- `0001-adopt-adrs.md`
- `0002-use-jax-for-computation.md`
- etc.

## Format

Each ADR includes:

- **Title**: A brief description of the decision.
- **Status**: One of `Proposed`, `Accepted`, `Deprecated`, or `Superseded by ADR-XXXX`.
- **Context**: Why this decision was needed. What problem or opportunity prompted it?
- **Decision**: What was chosen and why.
- **Consequences**: What benefits and trade-offs does this decision introduce?

## Why ADRs Matter

Tengri is built with significant AI assistance. Decisions can drift between development sessions if not recorded. ADRs provide a permanent, searchable archive that helps:

- Future contributors understand *why* a choice was made, not just *what* was chosen.
- Future AI systems (in later sessions or sessions with new models) avoid re-litigating settled decisions.
- Maintainers make informed changes by seeing past reasoning.
- Onboarding new team members quickly.

## How to Write an ADR

1. Copy the template structure from an existing ADR (e.g., 0001-adopt-adrs.md).
2. Assign the next available number.
3. Write in plain English; avoid jargon where possible.
4. Keep the decision section concise (2-3 sentences).
5. Consequences should be honest about trade-offs, not just benefits.
6. If the decision is later changed, mark the old ADR `Superseded by ADR-XXXX`.
