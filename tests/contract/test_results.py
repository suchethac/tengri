"""Tests for tengri.results — FitResult wrapper and FitRecord."""

from __future__ import annotations

import dataclasses
import tempfile
import warnings
from pathlib import Path

import pytest

from tengri.results import FitRecord, FitResult

pytestmark = pytest.mark.bounds


class TestFitRecord:
    """Unit tests for FitRecord dataclass."""

    def test_capture_shape(self):
        """FitRecord.capture() returns non-empty fields."""
        rec = FitRecord.capture()
        assert rec.tengri_version
        assert rec.python_version
        assert rec.platform
        assert rec.timestamp_utc
        # JAX may or may not be available
        assert isinstance(rec.jax_version, (str, type(None)))
        assert isinstance(rec.jax_backend, (str, type(None)))

    def test_capture_with_extras(self):
        """FitRecord.capture() stores extras."""
        extras = {"model_name": "test_model", "galaxy_id": 12345}
        rec = FitRecord.capture(extras=extras)
        assert rec.extras == extras

    def test_is_frozen(self):
        """FitRecord frozen=True prevents modification."""
        rec = FitRecord.capture()
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.tengri_version = "1.0.0"

    def test_with_timing(self):
        """FitRecord stores wall time and random seed."""
        rec = FitRecord.capture(wall_time_seconds=42.5, random_seed=12345)
        assert rec.wall_time_seconds == 42.5
        assert rec.random_seed == 12345

    def test_with_input_hash(self):
        """FitRecord stores input data hash."""
        hash_val = "abc123def456"
        rec = FitRecord.capture(input_data_hash=hash_val)
        assert rec.input_data_hash == hash_val


class TestFitResult:
    """Unit tests for FitResult wrapper."""

    def test_fitresult_construction(self):
        """FitResult constructed with inner, record, citation_keys."""
        inner = {"x": 1}
        rec = FitRecord.capture()
        result = FitResult(
            inner=inner,
            record=rec,
            citation_keys=["jax", "dsps"],
        )
        assert result.inner == inner
        assert result.record == rec
        assert result.citation_keys == ["jax", "dsps"]

    def test_fitresult_citations_property(self):
        """FitResult.citations resolves keys against registry."""
        rec = FitRecord.capture()
        result = FitResult(
            inner={},
            record=rec,
            citation_keys=["jax", "dsps"],
        )
        cites = result.citations
        assert len(cites) >= 2
        assert all(hasattr(c, "key") for c in cites)

    def test_fitresult_unknown_key_silently_skipped(self):
        """Unknown citation keys are silently skipped."""
        rec = FitRecord.capture()
        result = FitResult(
            inner={},
            record=rec,
            citation_keys=["jax", "not_a_real_key_xyz_123"],
        )
        cites = result.citations
        assert len(cites) >= 1
        assert all(c.key != "not_a_real_key_xyz_123" for c in cites)

    def test_fitresult_summary_nonempty(self):
        """FitResult.summary() returns non-empty string."""
        rec = FitRecord.capture()
        result = FitResult(
            inner={"samples": {"param": [1.0, 2.0]}},
            record=rec,
            citation_keys=["jax"],
            backend="vi",
            preset="starforming",
        )
        summary = result.summary()
        assert summary
        assert "FitResult" in summary
        assert "vi" in summary
        assert "starforming" in summary

    def test_fitresult_summary_no_backend(self):
        """FitResult.summary() handles None backend."""
        rec = FitRecord.capture()
        result = FitResult(inner={}, record=rec)
        summary = result.summary()
        assert "unknown" in summary or "Backend" in summary

    def test_fitresult_summary_with_wall_time(self):
        """FitResult.summary() includes wall time if available."""
        rec = FitRecord.capture(wall_time_seconds=123.4)
        result = FitResult(inner={}, record=rec)
        summary = result.summary()
        assert "123.4" in summary

    def test_provenance_attribute_is_deprecated(self):
        """Old `.provenance` attribute still works but warns."""
        rec = FitRecord.capture()
        result = FitResult(inner={}, record=rec)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert result.provenance is rec
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    @pytest.mark.skipif(
        __import__("importlib.util").util.find_spec("h5py") is None,
        reason="h5py not installed",
    )
    def test_fitresult_save_load_roundtrip(self):
        """FitResult save/load roundtrip preserves metadata."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.h5"

            rec = FitRecord.capture(
                wall_time_seconds=42.5,
                random_seed=12345,
                input_data_hash="abc123",
            )
            inner = {"samples": {"param": np.array([1.0, 2.0, 3.0])}}
            result = FitResult(
                inner=inner,
                record=rec,
                citation_keys=["jax", "dsps"],
                backend="vi",
                preset="starforming",
            )

            result.save(str(path))
            assert path.exists()

            loaded = FitResult.load(str(path))

            assert loaded.record.tengri_version == rec.tengri_version
            assert loaded.record.python_version == rec.python_version
            assert loaded.record.timestamp_utc == rec.timestamp_utc
            assert loaded.record.wall_time_seconds == 42.5
            assert loaded.record.random_seed == 12345
            assert loaded.record.input_data_hash == "abc123"

            assert loaded.backend == "vi"
            assert loaded.preset == "starforming"
            assert set(loaded.citation_keys) == {"jax", "dsps"}

            assert "samples" in loaded.inner
            if loaded.inner["samples"]:
                assert "param" in loaded.inner["samples"]

    def test_fitresult_save_handles_missing_h5py(self):
        """FitResult.save() raises ImportError gracefully when h5py unavailable."""
        rec = FitRecord.capture()
        result = FitResult(inner={}, record=rec)
        assert result is not None

    @pytest.mark.skipif(
        __import__("importlib.util").util.find_spec("h5py") is None,
        reason="h5py not installed",
    )
    def test_fitresult_save_empty_inner(self):
        """FitResult.save() handles empty inner dict gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.h5"
            rec = FitRecord.capture()
            result = FitResult(inner={}, record=rec)
            result.save(str(path))
            assert path.exists()

            loaded = FitResult.load(str(path))
            assert loaded.record.timestamp_utc == rec.timestamp_utc
