# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in tengri, please report it responsibly.

**Do not open a public GitHub issue for security bugs.** Instead, either:

1. Email cooray@stanford.edu with the details.
2. Use GitHub's private security advisory feature (Settings > Security > Report a vulnerability).

Please include:
- A clear description of the vulnerability.
- Steps to reproduce (if possible).
- The affected version(s).
- Any potential impact.

## Disclosure Timeline

We aim for a 90-day coordinated disclosure window. After we receive your report:

1. We will acknowledge receipt within one week.
2. We will investigate and develop a fix.
3. We will release a patched version.
4. We will publicly disclose the issue and credit you (unless you prefer anonymity).

If a fix takes longer than 90 days, we will communicate a revised timeline.

## Scope

Security reports are welcome for:

- Input parsers (FITS readers, HDF5 deserialization).
- Unsafe file-path handling that could lead to traversal attacks.
- Deserialization of untrusted data that could cause code execution.

## Out of Scope

The following are typically not in scope for security reports:

- Denial-of-service attacks on scientific workloads (e.g., pathological input sizes).
- Vulnerabilities in upstream dependencies (report those to the dependency maintainers).
- Issues in external tools that tengri wraps (e.g., DSPS, JAX).

## Context

Tengri is scientific software, not production infrastructure. The threat model emphasizes data corruption and malformed input rather than adversarial compromise. That said, we take input safety seriously.
