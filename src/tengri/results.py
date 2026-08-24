# SPDX-License-Identifier: BSD-3-Clause
"""FitResult: a thin record-and-citations wrapper around Posterior / SEDResult.

Does not replace existing result types; wraps them so downstream code can access
``.samples`` or ``.params`` exactly as before via ``result.inner``.
"""

from __future__ import annotations

import datetime as _dt
import platform as _platform
import sys as _sys
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any

from tengri.analysis.mock import MockData, generate_mock
from tengri.citations import Citation, cite as _cite
from tengri.inference.catalog_fitter import CatalogPosterior
from tengri.inference.hierarchical import PopulationPosterior
from tengri.inference.posterior import Posterior

__all__ = [
    "CatalogPosterior",
    "FitRecord",
    "FitResult",
    "MockData",
    "PopulationPosterior",
    "Posterior",
    "ResultSerializationError",
    "generate_mock",
    "posteriors_to_dataframe",
]

__version_fallback__ = "0.0.0"


class ResultSerializationError(RuntimeError):
    """A saved fit is missing data that was asked to be written.

    Raised by :meth:`FitResult.save` when one or more sample entries could not
    be written. The file is still created and holds everything that *could* be
    written, with the omitted names recorded in its ``samples.skipped_keys``
    attribute; so the failure is recoverable, not fatal.
    :meth:`FitResult.load` reads that attribute back and warns, so a later
    reader of an incomplete file learns of it too.

    It raises rather than warning because the failure is invisible in the
    artifact: a saved fit that silently lacks its samples looks exactly like a
    complete one until someone loads it, potentially long after the run that
    produced it is gone.
    """


@dataclass(frozen=True)
class FitRecord:
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
        SHA256 hash of input data array (optional audit field).
    extras : dict[str, Any]
        Additional metadata keyed by string.

    Returns
    -------
    FitRecord
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
    FitRecord is frozen (immutable) to prevent accidental modification
    after capture. All fields are read-only.

    Examples
    --------
    >>> rec = FitRecord.capture(wall_time_seconds=42.5, random_seed=12345)
    >>> print(rec.timestamp_utc)
    2026-04-23T15:30:45Z
    >>> print(rec.jax_backend)
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
    ) -> FitRecord:
        """Snapshot current environment into a FitRecord.

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
        FitRecord
            Immutable snapshot of current environment.

        Notes
        -----
        This method is safe to call at any point during inference.
        JAX version and backend detection fails gracefully if JAX
        is unavailable.

        Examples
        --------
        >>> rec = FitRecord.capture(wall_time_seconds=123.4)
        >>> rec.tengri_version
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
            timestamp_utc=_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            wall_time_seconds=wall_time_seconds,
            random_seed=random_seed,
            input_data_hash=input_data_hash,
            extras=dict(extras or {}),
        )


@dataclass
class FitResult:
    """Wraps a Posterior or SEDResult with a fit record + citations.

    A thin, non-invasive wrapper that bundles any result object (Posterior,
    SEDResult, or list of Posteriors) with execution metadata and citation
    information. Downstream code can access the result's attributes (e.g.,
    ``.samples``, ``.params``) directly on the FitResult object via attribute
    forwarding. For direct access to the result object, use ``.inner``.

    Parameters
    ----------
    inner : Any
        The underlying result object (Posterior, SEDResult, or list[Posterior]).
        Attributes are forwarded via __getattr__ for transparent access.
    record : FitRecord
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
        Wrapper with fit-record and citation tracking.

    Attributes
    ----------
    inner : Any
        The result object (Posterior, SEDResult, etc.).
    record : FitRecord
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
    incrementally during inference. The wrapper itself is transparent:
    accessing ``result.samples`` or ``result.params`` automatically forwards
    to ``result.inner.samples`` or ``result.inner.params`` via ``__getattr__``.

    Examples
    --------
    >>> result = fitter.fit(data, noise, method="vi")  # returns Posterior
    >>> fit_result = FitResult(
    ...     inner=result,
    ...     record=FitRecord.capture(wall_time_seconds=42.5),
    ...     citation_keys=["dsps", "tengri", "calzetti2000"],
    ...     backend="vi",
    ...     preset="starforming",
    ... )
    >>> print(fit_result.summary())
    >>> # Access result attributes directly; forwarding is transparent
    >>> print(fit_result.samples)  # forwards to fit_result.inner.samples
    >>> fit_result.save("/data/result.h5")
    >>> loaded = FitResult.load("/data/result.h5")
    """

    inner: Any
    record: FitRecord
    citation_keys: list[str] = field(default_factory=list)
    backend: str | None = None
    preset: str | None = None

    @property
    def provenance(self) -> FitRecord:
        """Deprecated alias for :attr:`record`. Will be removed in tengri v1.0."""
        _warnings.warn(
            "FitResult.provenance is deprecated; use FitResult.record instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.record

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to inner result object.

        Allows transparent access to result attributes (e.g., ``.samples``,
        ``.params``) without explicit ``.inner`` reference.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            Attribute value from inner result object.

        Raises
        ------
        AttributeError
            If the attribute does not exist on inner.

        Notes
        -----
        This is called only for attributes not found on FitResult itself.
        Attributes like ``inner``, ``record``, ``citations``, etc.
        are resolved normally without calling this method.
        """
        # Avoid infinite recursion on dataclass initialization
        if name in ("inner", "record", "citation_keys", "backend", "preset"):
            raise AttributeError(f"FitResult has no attribute {name!r}")
        try:
            inner = object.__getattribute__(self, "inner")
        except AttributeError:
            raise AttributeError(f"FitResult has no attribute {name!r}") from None
        return getattr(inner, name)

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
        ts = self.record.timestamp_utc
        if self.record.wall_time_seconds is not None:
            lines.append(f"Timestamp: {ts} ({self.record.wall_time_seconds:.1f} s)")
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
        ResultSerializationError
            If any sample entry could not be written, or if the inner
            result's ``to_dict()`` raised. A subclass of ``RuntimeError``.
            The file is still created and holds every entry that *could*
            be written, so the failure is recoverable.

        Notes
        -----
        The HDF5 schema version is ``v1``. The inner result contributes a
        ``/tengri_fitresult/samples`` group, populated from whichever of these
        is available:

        - ``.samples`` dict -> one dataset per key
        - else ``.to_dict()`` -> one dataset per key
        - else nothing; the group is created but left empty

        Serialization is per-key, not all-or-nothing: an entry h5py has no
        dtype for costs exactly that entry, and its name is recorded in the
        group's ``skipped_keys`` attribute *and* raised. Until 2026-08 the
        whole block sat under ``contextlib.suppress(Exception)``, so an
        unwritable entry silently took every entry after it with it and
        ``save`` still returned normally: this docstring already promised the
        ``RuntimeError`` that the code did not raise.

        See Also
        --------
        load : Inverse operation. Warns if the file records skipped keys.

        Examples
        --------
        >>> fit_result.save("/data/result.h5")
        """
        import json

        import numpy as np

        try:
            import h5py
        except ImportError:
            raise ImportError("save() requires h5py: pip install h5py") from None

        with h5py.File(path, "w") as f:
            # Root group for versioning
            grp = f.create_group("tengri_fitresult")
            grp.attrs["schema_version"] = "v1"

            # Provenance as attributes
            grp.attrs["tengri_version"] = self.record.tengri_version
            grp.attrs["python_version"] = self.record.python_version
            grp.attrs["platform"] = self.record.platform
            grp.attrs["timestamp_utc"] = self.record.timestamp_utc
            if self.record.jax_version is not None:
                grp.attrs["jax_version"] = self.record.jax_version
            if self.record.jax_backend is not None:
                grp.attrs["jax_backend"] = self.record.jax_backend
            if self.record.wall_time_seconds is not None:
                grp.attrs["wall_time_seconds"] = self.record.wall_time_seconds
            if self.record.random_seed is not None:
                grp.attrs["random_seed"] = self.record.random_seed
            if self.record.input_data_hash is not None:
                grp.attrs["input_data_hash"] = self.record.input_data_hash

            # Extras as JSON attribute
            if self.record.extras:
                grp.attrs["extras_json"] = json.dumps(self.record.extras)

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

            # Serialize inner result.
            #
            # This whole block used to sit under `contextlib.suppress(Exception)`,
            # so the first unwritable value abandoned every entry after it while
            # `save()` returned normally. Measured with
            # `{"good": ndarray, "bad": object()}`: the file kept `good`, dropped
            # `bad`, and said nothing. A partial silent write, whose extent
            # depends on dict order: an unwritable first key costs everything
            # behind it. Of the ten blanket suppressors this is the only one that
            # loses data rather than merely hiding a diagnostic.
            #
            # Two changes. Per-key failures no longer abandon the remaining
            # keys, so one bad entry costs exactly one entry regardless of where
            # it sits in the dict. And whatever could not be written is recorded
            # on the group and raised, so the caller learns at save time rather
            # than at load time.
            samples_grp = grp.create_group("samples")
            skipped: list[str] = []

            source: dict | None = None
            if isinstance(getattr(self.inner, "samples", None), dict):
                source = self.inner.samples
            elif hasattr(self.inner, "to_dict"):
                try:
                    source = self.inner.to_dict()
                except Exception as exc:
                    raise ResultSerializationError(
                        f"{type(self.inner).__name__}.to_dict() raised while saving to "
                        f"{path!r}: {exc!r}. Nothing was written for `samples`."
                    ) from exc
            if not isinstance(source, dict):
                source = {}

            for key, val in source.items():
                try:
                    samples_grp.create_dataset(key, data=np.array(val))
                except (TypeError, ValueError) as exc:
                    # h5py has no dtype for this value. Keep going: one
                    # unwritable entry must not cost the others.
                    skipped.append(f"{key} ({type(val).__name__}: {exc})")

            if skipped:
                samples_grp.attrs["skipped_keys"] = json.dumps(skipped)
                raise ResultSerializationError(
                    f"{len(skipped)} of {len(source)} sample entries could not be "
                    f"written to {path!r} and are missing from the file:\n  "
                    + "\n  ".join(skipped)
                    + "\n\nEverything else was written, and the names above are "
                    "recorded in the file's `samples.skipped_keys` attribute. "
                    "This raises rather than passing quietly because a saved fit "
                    "that silently lacks its samples is indistinguishable from a "
                    "complete one until you try to use it."
                )

    @classmethod
    def load(cls, path: str) -> FitResult:
        """Inverse of :meth:`save`. Restore from HDF5.

        Reconstructs provenance and citation_keys fully. Inner result
        is restored as a plain ``dict(samples={...})``: not the original
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
            raise ImportError("load() requires h5py: pip install h5py") from None

        import json

        with h5py.File(path, "r") as f:
            grp = f["tengri_fitresult"]

            # Reconstruct provenance
            extras = {}
            if "extras_json" in grp.attrs:
                extras = json.loads(grp.attrs["extras_json"])

            record = FitRecord(
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

                # `save` raises on entries it could not write, but that raise is
                # heard once, by whoever ran the fit. The person who loads the
                # file later: possibly on another machine, after the run is
                # gone: would otherwise see a dict that looks complete. The
                # record has to be read where the incomplete data is used, or
                # writing it just moves the silence downstream.
                if "skipped_keys" in grp["samples"].attrs:
                    dropped = json.loads(grp["samples"].attrs["skipped_keys"])
                    _warnings.warn(
                        f"{path!r} is an incomplete save: {len(dropped)} sample "
                        f"entr{'y' if len(dropped) == 1 else 'ies'} could not be "
                        "written when it was created and are absent from the "
                        "loaded result:\n  " + "\n  ".join(dropped),
                        UserWarning,
                        stacklevel=2,
                    )

            return cls(
                inner=inner,
                record=record,
                citation_keys=citation_keys,
                backend=backend,
                preset=preset,
            )


def posteriors_to_dataframe(results: list, params: list[str] | None = None):
    """Summarize a list of Posteriors into a pandas DataFrame.

    Requires ``pandas`` (``pip install pandas``).

    Parameters
    ----------
    results : list of Posterior
        Output of ``model.fit_batch()`` or any list of Posterior objects.
    params : list of str or None
        Parameter names to include. Default: all scalar free parameters,
        excluding ``psd_xi``.

    Returns
    -------
    pandas.DataFrame
        One row per galaxy, columns: ``{param}_median``, ``{param}_lo68``,
        ``{param}_hi68`` for each requested parameter.

    Notes
    -----
    **JIT-compatible**: no, pure Python, requires pandas library.

    Examples
    --------
    >>> df = tengri.results.posteriors_to_dataframe(results, params=["met_logzsol", "dust_tau_bc"])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "posteriors_to_dataframe() requires pandas: pip install pandas"
        ) from None

    import numpy as np

    rows = []
    for result in results:
        row: dict = {}

        if result.samples is None:
            for name, val in result.params.items():
                if name == "psd_xi":
                    continue
                if params is not None and name not in params:
                    continue
                row[f"{name}_value"] = float(np.mean(np.array(val)))
        else:
            for name, arr in result.samples.items():
                if name == "psd_xi":
                    continue
                if params is not None and name not in params:
                    continue
                arr_np = np.array(arr)
                if arr_np.ndim != 1:
                    continue
                row[f"{name}_median"] = float(np.median(arr_np))
                row[f"{name}_lo68"] = float(np.percentile(arr_np, 16))
                row[f"{name}_hi68"] = float(np.percentile(arr_np, 84))

        rows.append(row)

    return pd.DataFrame(rows)
