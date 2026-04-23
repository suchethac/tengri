# SPDX Licensing Headers

## What is SPDX?

SPDX (Software Package Data Exchange) is an open standard for communicating software licenses and copyright information in a machine-readable format. The SPDX License List provides standardized license identifiers (e.g., BSD-3-Clause, MIT, Apache-2.0) that are recognized by tools, registries, and automated license compliance checkers. Using SPDX headers makes license attribution explicit and auditable.

## Header Format

Every new Python file under `src/tengri/` and `tests/` must include this header at the top:

```python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Suchetha Cooray and the tengri developers.
```

Place this header before any docstrings, imports, or other code. This single block is sufficient for files where all contributions are original work or ports with compatible licenses (see Ported Code Policy below).

## Ported Code Policy

When code is ported from another repository, add an additional attribution line immediately after the copyright line:

```python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Suchetha Cooray and the tengri developers.
# Ported from <upstream/repo>@<sha>, <LICENSE>, <YYYY-MM-DD>.
```

**Ported code requirements:**
- Upstream code must use a compatible license: MIT, BSD-2-Clause, BSD-3-Clause, or Apache-2.0.
- GPL and other copyleft licenses cannot be merged into tengri (it would require the entire codebase to be GPL).
- Include the upstream repository name (e.g., `prospector`), commit SHA (first 8 characters), license identifier, and the date of porting.
- Preserve or adapt upstream docstrings and citations where applicable.

Example:

```python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Suchetha Cooray and the tengri developers.
# Ported from prospector@a1b2c3d4, BSD-3-Clause, 2025-10-15.
```

## Grandfathering

Existing files in the repository are not required to have headers retroactively added. This rule applies only to new files created going forward. When making substantial refactors or significant new work in an existing file, consider adding the header at that time.

## Future: CI Check

A future CI job will validate that all new Python files include a valid SPDX header. This will be a linting pass similar to ruff format checks, ensuring compliance at commit time.
