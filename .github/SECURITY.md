# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue for a security bug.** Report it privately, either way:

1. Email: cooray@stanford.edu
2. GitHub's private security advisory feature (Settings > Security > Report a vulnerability)

Include:
- A clear description of the vulnerability.
- Steps to reproduce (if possible).
- The affected version(s).
- Any potential impact.

## Disclosure Timeline

Our process follows a 90-day coordinated disclosure window:

1. Acknowledge receipt within one week.
2. Investigate and develop a fix.
3. Release a patched version.
4. Publicly disclose and credit you (unless you prefer anonymity).

If a fix takes longer than 90 days, we will communicate a revised timeline.

## Scope

In scope:

- Input parsers (FITS readers, HDF5 deserialization).
- Unsafe file-path handling.
- Deserialization of untrusted data.

Out of scope:

- Denial-of-service attacks on scientific workloads.
- Upstream dependency vulnerabilities.
- Issues in external tools that tengri wraps.

## Context

Tengri is scientific software. The threat model emphasizes data corruption and malformed input. We take input safety seriously.
