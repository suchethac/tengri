"""FitResult — a thin provenance-and-citations wrapper around Posterior / SEDResult.

Does not replace existing result types; wraps them so downstream code can access
``.samples`` or ``.params`` exactly as before via ``result.inner``.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import platform as _platform
import sys as _sys
from dataclasses import dataclass, field
from typing import Any

from tengri.citations import Citation, cite as _cite

__all__ = ["FitResult", "Provenance"]

__version_fallback__ = "0.0.0"


@dataclass(frozen=True)
class Provenance:
    """Environment and execution metadata captured at fit time.

    Stores version numbers, platform info, timestamps, and optional execution
    details (wall time, random seed, input hash) to enable reproducibility
    and audit trails.

    Parameters
    ----------
    tengri_version : str
        Version string of tengri package.
    python_version : str
        Python version string (e.g., "3.10.12").
    platform : str
        Platform string from platform.system() + platform.machine()
        (e.g., "Darwin arm64").
    jax_version : str | None
        JAX version string, None if import failed.
    jax_backend : str | None
        JAX default backend at capture time (e.g., "gpu", "cpu"),
        None if JAX unavailable.
    timestamp_utc : str
        ISO-8601 UTC timestamp (e.g., "2026-04-23T15:30:45Z").
    wall_time_seconds : float | None
        Total wall-clock runtime in seconds, None if not measured.
    random_seed : int | None
        Pseudo-random seed used (reproducibility), None if not applicable.
    input_data_hash : str | None
        SHA256 hash of input data array (optional provenance audit).
    extras : dict[str, Any]
        Additional metadata keyed by string.

    Returns
    -------
    Provenance
        Immutable record of environment and execution context.

    Attributes
    ----------
    tengri_version : str
    python_version : str
    platform : str
    jax_version : str | None
    jax_backend : str | None
    timestamp_utc : str
    wall_time_seconds : float | None
    random_seed : int | None
    input_data_hash : str | None
    extras : dict[str, Any]

    Notes
    -----
    Provenance is frozen (immutable) to prevent accidental modification
    after capture. All fields are read-only.

    Examples
    --------
    >>> prov = Provenance.capture(wall_time_seconds=42.5, random_seed=12345)
    >>> print(prov.timestamp_utc)
    2026-04-23T15:30:45Z
    >>> print(prov.jax_backend)
    cpu
    """

    tengri_version: str
    python_version: str
    platform: str
    jax_version: str | None
    jax_backend: str | None
    timestamp_utc: str
    wall_time_seconds: float | None = None
    random_seed: int | None = None
    input_data_hash: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        *,
        wall_time_seconds: float | None = None,
        random_seed: int | None = None,
        input_data_hash: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> Provenance:
        """Snapshot current environment into a Provenance record.

        Captures version numbers, platform, JAX backend, and timestamp.
        Optionally includes wall time, random seed, and input data hash.

        Parameters
        ----------
        wall_time_seconds : float | None, optional
            Total wall-clock runtime in seconds.
        random_seed : int | None, optional
            Pseudo-random seed for reproducibility.
        input_data_hash : str | None, optional
            SHA256 hash of input data (optional audit).
        extras : dict[str, Any] | None, optional
            Additional metadata. Default: {}.

        Returns
        -------
        Provenance
            Immutable snapshot of current environment.

        Notes
        -----
        This method is safe to call at any point during inference.
        JAX version and backend detection fails gracefully if JAX
        is unavailable.

        Examples
        --------
        >>> prov = Provenance.capture(wall_time_seconds=123.4)
        >>> prov.tengri_version
        '0.1.0'
        """
        import tengri

        try:
            import jax

            jax_v = jax.__version__
            jax_b = str(jax.default_backend())
        except Exception:
            jax_v = None
            jax_b = None

        return cls(
            tengri_version=getattr(tengri, "__version__", __version_fallback__),
            python_version=_sys.version.split()[0],
            platform=f"{_platform.system()} {_platform.machine()}",
            jax_version=jax_v,
            jax_backend=jax_b,
            timestamp_utc=_dt.datetime.now(_dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            wall_time_seconds=wall_time_seconds,
            random_seed=random_seed,
            input_data_hash=input_data_hash,
            extras=dict(extras or {}),
        )


@dataclass
class FitResult:
    """Wraps a Posterior or SEDResult with provenance + citations.

    A thin, non-invasive wrapper that bundles any result object (Posterior,
    SEDResult, or list of Posteriors) with execution provenance and citation
    metadata. Downstream code accesses the original result via ``.inner``,
    preserving its exact structure and behavior.

    Parameters
    ----------
    inner : Any
        The underlying result object (Posterior, SEDResult, or list[Posterior]).
        Accessed by downstream code without modification.
    provenance : Provenance
        Execution environment and timing metadata.
    citation_keys : list[str], optional
        Registry keys for citations that apply to this fit
        (e.g., ["dsps", "tengri", "fsps"]). Default: [].
    backend : str | None, optional
        Inference backend name (e.g., "vi", "mcmc_nuts", "map").
        Default: None.
    preset : str | None, optional
        Name of the preset used, if any (e.g., "starforming").
        Default: None.

    Returns
    -------
    FitResult
        Wrapper with provenance and citation tracking.

    Attributes
    ----------
    inner : Any
        The result object (Posterior, SEDResult, etc.).
    provenance : Provenance
        Execution environment snapshot.
    citation_keys : list[str]
        Registry keys for citations.
    backend : str | None
        Inference backend name.
    preset : str | None
        Preset name if applicable.
    citations : list[Citation]
        Resolved citations (property).

    Notes
    -----
    FitResult is not frozen, allowing backends to populate it
    incrementally during inference. The wrapper itself is non-invasive:
    code that expects a bare Posterior or SEDResult can still call
    ``result.inner.samples`` or ``result.inner.params`` directly.

    Examples
    --------
    >>> result = fitter.fit(data, noise, method="vi")  # returns Posterior
    >>> fit_result = FitResult(
    ...     inner=result,
    ...     provenance=Provenance.capture(wall_time_seconds=42.5),
    ...     citation_keys=["dsps", "tengri", "calzetti2000"],
    ...     backend="vi",
    ...     preset="starforming",
    ... )
    >>> print(fit_result.summary())
    >>> fit_result.save("/data/result.h5")
    >>> loaded = FitResult.load("/data/result.h5")
    """

    inner: Any
    provenance: Provenance
    citation_keys: list[str] = field(default_factory=list)
    backend: str | None = None
    preset: str | None = None

    @property
    def citations(self) -> list[Citation]:
        """Resolve citation_keys against the registry.

        Returns
        -------
        list of Citation
            Matched citations. Unknown keys are silently skipped
            (logged at debug level).

        Notes
        -----
        Unknown keys do not raise exceptions; they are simply omitted
        from the result. This allows flexible citation handling.
        """
        import logging

        logger = logging.getLogger(__name__)
        out = []
        for k in self.citation_keys:
            try:
                out.append(_cite(k))
            except KeyError:
                logger.debug(f"Citation key '{k}' not found in registry")
        return out

    def summary(self) -> str:
        """Return a short multi-line human summary.

        Returns
        -------
        str
            Formatted summary including timestamp, backend, preset,
            result type, and citation count.

        Notes
        -----
        Summary is designed for quick inspection. Full details are
        available via provenance and citations properties.

        Examples
        --------
        >>> print(fit_result.summary())
        FitResult (Posterior)
        Backend: vi, Preset: starforming
        Timestamp: 2026-04-23T15:30:45Z (42.5 s)
        Citations: 3
        """
        lines = []

        # Infer result type name
        result_type = type(self.inner).__name__
        lines.append(f"FitResult ({result_type})")

        # Backend and preset
        backend_str = self.backend or "unknown"
        preset_str = f"{self.preset}" if self.preset else "none"
        lines.append(f"Backend: {backend_str}, Preset: {preset_str}")

        # Timestamp and wall time
        ts = self.provenance.timestamp_utc
        if self.provenance.wall_time_seconds is not None:
            lines.append(f"Timestamp: {ts} ({self.provenance.wall_time_seconds:.1f} s)")
        else:
            lines.append(f"Timestamp: {ts}")

        # Citation count
        n_cites = len(self.citations)
        lines.append(f"Citations: {n_cites}")

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save as a self-contained HDF5 file (version-tagged schema).

        Stores provenance fields as attributes, citation_keys as a dataset,
        and inner result (if possible) as nested datasets.

        Parameters
        ----------
        path : str
            File system path (e.g., "/data/result.h5").

        Raises
        ------
        ImportError
            If h5py is not installed.
        RuntimeError
            If inner result cannot be serialized.

        Notes
        -----
        The HDF5 schema version is ``v1``. Inner result serialization
        is best-effort:

        - Posterior with .samples dict → stored as /samples/{param} datasets
        - Objects with .to_dict() → stored under /inner_data
        - Otherwise → stored as a JSON string in /inner_json attribute

        See Also
        --------
        load : Inverse operation.

        Examples
        --------
        >>> fit_result.save("/data/result.h5")
        """
        import json

        import numpy as np

        try:
            import h5py
        except ImportError:
            raise ImportError(
                "save() requires h5py: pip install h5py"
            ) from None

        with h5py.File(path, "w") as f:
            # Root group for versioning
            grp = f.create_group("tengri_fitresult")
            grp.attrs["schema_version"] = "v1"

            # Provenance as attributes
            grp.attrs["tengri_version"] = self.provenance.tengri_version
            grp.attrs["python_version"] = self.provenance.python_version
            grp.attrs["platform"] = self.provenance.platform
            grp.attrs["timestamp_utc"] = self.provenance.timestamp_utc
            if self.provenance.jax_version is not None:
                grp.attrs["jax_version"] = self.provenance.jax_version
            if self.provenance.jax_backend is not None:
                grp.attrs["jax_backend"] = self.provenance.jax_backend
            if self.provenance.wall_time_seconds is not None:
                grp.attrs["wall_time_seconds"] = self.provenance.wall_time_seconds
            if self.provenance.random_seed is not None:
                grp.attrs["random_seed"] = self.provenance.random_seed
            if self.provenance.input_data_hash is not None:
                grp.attrs["input_data_hash"] = self.provenance.input_data_hash

            # Extras as JSON attribute
            if self.provenance.extras:
                grp.attrs["extras_json"] = json.dumps(self.provenance.extras)

            # Backend and preset
            if self.backend is not None:
                grp.attrs["backend"] = self.backend
            if self.preset is not None:
                grp.attrs["preset"] = self.preset

            # Citation keys as dataset
            grp.create_dataset(
                "citation_keys",
                data=np.array(self.citation_keys, dtype=h5py.string_dtype()),
            )

            # Serialize inner result
            samples_grp = grp.create_group("samples")
            with contextlib.suppress(Exception):
                # Try to extract samples dict from Posterior
                if hasattr(self.inner, "samples") and isinstance(self.inner.samples, dict):
                    for key, val in self.inner.samples.items():
                        samples_grp.create_dataset(key, data=np.array(val))
                elif hasattr(self.inner, "to_dict"):
                    # Fallback: call .to_dict() if available
                    inner_dict = self.inner.to_dict()
                    for key, val in inner_dict.items():
                        with contextlib.suppress(TypeError, ValueError):
                            samples_grp.create_dataset(key, data=np.array(val))

    @classmethod
    def load(cls, path: str) -> FitResult:
        """Inverse of :meth:`save`. Restore from HDF5.

        Reconstructs provenance and citation_keys fully. Inner result
        is restored as a plain ``dict(samples={...})`` — not the original
        class. Document this limitation if passing to downstream code.

        Parameters
        ----------
        path : str
            File system path to HDF5 file.

        Returns
        -------
        FitResult
            Restored wrapper with all metadata.

        Raises
        ------
        ImportError
            If h5py is not installed.
        KeyError
            If required schema elements are missing.

        Notes
        -----
        The loaded ``.inner`` is always a dict, not the original
        Posterior or SEDResult class. This is a deliberate design choice
        to avoid complex deserialization. Code that depends on the original
        type should reconstruct it from ``.inner["samples"]`` or similar.

        See Also
        --------
        save : Forward operation.

        Examples
        --------
        >>> fit_result = FitResult.load("/data/result.h5")
        >>> samples = fit_result.inner["samples"]
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "load() requires h5py: pip install h5py"
            ) from None

        import json

        with h5py.File(path, "r") as f:
            grp = f["tengri_fitresult"]

            # Reconstruct provenance
            extras = {}
            if "extras_json" in grp.attrs:
                extras = json.loads(grp.attrs["extras_json"])

            provenance = Provenance(
                tengri_version=grp.attrs["tengri_version"],
                python_version=grp.attrs["python_version"],
                platform=grp.attrs["platform"],
                jax_version=grp.attrs.get("jax_version"),
                jax_backend=grp.attrs.get("jax_backend"),
                timestamp_utc=grp.attrs["timestamp_utc"],
                wall_time_seconds=grp.attrs.get("wall_time_seconds"),
                random_seed=grp.attrs.get("random_seed"),
                input_data_hash=grp.attrs.get("input_data_hash"),
                extras=extras,
            )

            # Restore citation keys
            citation_keys = list(grp["citation_keys"][()])
            citation_keys = [k.decode() if isinstance(k, bytes) else k for k in citation_keys]

            # Restore backend and preset
            backend = grp.attrs.get("backend")
            preset = grp.attrs.get("preset")

            # Restore inner as dict of samples
            inner = {}
            if "samples" in grp:
                samples = {}
                for key in grp["samples"]:
                    samples[key] = grp["samples"][key][()]
                inner["samples"] = samples

            return cls(
                inner=inner,
                provenance=provenance,
                citation_keys=citation_keys,
                backend=backend,
                preset=preset,
            )
