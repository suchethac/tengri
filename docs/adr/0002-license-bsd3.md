# ADR 0002: Use BSD-3-Clause for tengri

**Status:** Accepted

**Date:** 2026-04-23

## Context

The tengri project will be released as open-source software for the scientific Python community. The choice of license directly impacts adoption by downstream projects, particularly in survey pipelines and institutional data reduction workflows.

Astropy-ecosystem convention favors permissive licenses (BSD-3-Clause or MIT). Statistical analysis of downstream adoption shows permissive licenses correlate strongly with uptake in survey infrastructure. GPL licenses were evaluated but analysis shows they measurably reduce adoption by data pipelines with proprietary components.

Apache-2.0 was considered but rejected: while technically strong, BSD-3-Clause aligns better with astro convention and is simpler to explain to contributors.

## Decision

We adopt BSD-3-Clause license for tengri.

**Implementation requirement:** All new Python source files must include SPDX license headers. See `docs/dev/spdx-headers.md` for the standard form.

## Consequences

**Positive:**
- Maximizes adoption latitude in both open and proprietary pipelines
- Aligns with scientific Python ecosystem norms
- Simple contributor understanding

**Negative:**
- Requires NOTICE file for upstream attribution (already in place)
- Precludes direct porting of GPL code from upstream references (Prospector, DSPS may have GPL components)

**Mitigation:** Ported code must be rewritten or explicitly licensed separately. See ported-over-invented policy in ADR 0003.
