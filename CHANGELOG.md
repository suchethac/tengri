# Changelog

All notable changes to tengri are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Galaxy facade class with `from_arrays` and `from_observation` constructors for ergonomic observation handling.
- `tengri.doctor` environment health check utility; run `python -m tengri doctor` to verify dependencies and configuration.
- Citations subsystem: `Citation` dataclass, registry with 16 seed entries, `cite()` and `cite_all()` helper functions for academic attribution.
- Presets module with factory functions: `starforming()`, `quiescent()`, `high_z()` for common model configurations.
- `FitResult` and `Provenance` wrapper classes with optional HDF5 save/load for reproducible inference workflows.
- Preprocessing module with zero-point registry, systematic-error-floor helper, and upper-limit utilities for photometry.
- I/O module with readers for SDSS, DESI, and generic FITS spectra; adapter for `specutils.Spectrum1D` integration.
- `tengri` CLI with `doctor` and `cite` subcommands.
- LICENSE file (BSD-3-Clause).
- CONTRIBUTING.md with contributor guidelines.
- Docstring standard reference in `docs/dev/spdx-headers.md`.

### Changed

- Declared license updated from MIT to BSD-3-Clause in `pyproject.toml` and `CITATION.cff`.

### Fixed

- (None in this release.)

---

## Notes for Pre-1.0 Users

Tengri is pre-1.0 software. The public API, configuration format, and file layout may change without semantic versioning guarantees until a stable 1.0 release is declared. We appreciate early feedback and encourage users to report breaking changes or feature requests via GitHub Issues.
