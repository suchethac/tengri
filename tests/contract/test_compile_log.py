# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for compile_log diagnostic tracer."""

from __future__ import annotations

import json
import threading
import time

import pytest

from tengri.utils.compile_log import (
    CompileEvent,
)

pytestmark = pytest.mark.contract


class TestCompileEventDataclass:
    """Tests for CompileEvent immutable dataclass."""

    def test_compile_event_creation(self) -> None:
        """Create a CompileEvent and verify immutability."""
        event = CompileEvent(
            timestamp="2026-05-06T14:30:45.123Z",
            name="run_hmc",
            method="mcmc_hmc",
            signature="(('shape', (12,)),)",
            duration_s=25.5,
            inferred_cache_hit=False,
        )
        assert event.timestamp == "2026-05-06T14:30:45.123Z"
        assert event.name == "run_hmc"
        assert event.method == "mcmc_hmc"
        assert event.duration_s == 25.5
        assert event.inferred_cache_hit is False

    def test_compile_event_immutable(self) -> None:
        """Verify CompileEvent is frozen (immutable)."""
        event = CompileEvent(
            timestamp="2026-05-06T14:30:45.123Z",
            name="run_hmc",
            method="mcmc_hmc",
            signature="(('shape', (12,)),)",
            duration_s=25.5,
            inferred_cache_hit=False,
        )
        with pytest.raises(AttributeError):
            event.duration_s = 30.0  # type: ignore


class TestIsEnabled:
    """Tests for is_enabled() function."""

    def test_disabled_by_default(self, monkeypatch) -> None:
        """When TENGRI_LOG_COMPILES is unset, is_enabled() returns False."""
        monkeypatch.delenv("TENGRI_LOG_COMPILES", raising=False)
        # Must reimport to pick up the new env var
        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)
        assert cl.is_enabled() is False

    def test_enabled_with_env_var(self, monkeypatch) -> None:
        """When TENGRI_LOG_COMPILES=1, is_enabled() returns True."""
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "1")
        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)
        assert cl.is_enabled() is True

    def test_enabled_with_yes(self, monkeypatch) -> None:
        """When TENGRI_LOG_COMPILES=yes, is_enabled() returns True."""
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "yes")
        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)
        assert cl.is_enabled() is True

    def test_disabled_with_0(self, monkeypatch) -> None:
        """When TENGRI_LOG_COMPILES=0, is_enabled() returns False."""
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "0")
        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)
        assert cl.is_enabled() is False


class TestLogPath:
    """Tests for log_path() resolution."""

    def test_default_log_path(self, tmp_path, monkeypatch) -> None:
        """Default log path is ~/.cache/tengri_jax_cache/compile.log."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("TENGRI_COMPILE_LOG_PATH", raising=False)
        monkeypatch.delenv("TENGRI_JAX_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)
        path = cl.log_path()
        assert "compile.log" in str(path)
        assert str(path).endswith("compile.log")

    def test_log_path_with_override(self, tmp_path, monkeypatch) -> None:
        """When TENGRI_COMPILE_LOG_PATH is set, use that path."""
        override_log = tmp_path / "custom.log"
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(override_log))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)
        path = cl.log_path()
        assert path == override_log

    def test_log_path_creates_directory(self, tmp_path, monkeypatch) -> None:
        """log_path() creates the parent directory if missing."""
        override_log = tmp_path / "nested" / "dir" / "compile.log"
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(override_log))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)
        # Call log_path() multiple times to ensure parent is created
        path = cl.log_path()
        # Verify the parent was created
        assert path.parent.exists()
        # Verify the path itself is what we expect
        assert path == override_log


class TestRecordCompileEvent:
    """Tests for record_compile_event() function."""

    def test_no_write_when_disabled(self, tmp_path, monkeypatch) -> None:
        """When disabled, record_compile_event() does not write."""
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "0")
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(tmp_path / "compile.log"))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)

        event = CompileEvent(
            timestamp="2026-05-06T14:30:45.123Z",
            name="test",
            method="test",
            signature="test_sig",
            duration_s=1.0,
            inferred_cache_hit=False,
        )
        cl.record_compile_event(event)

        # File should not be created
        log_file = tmp_path / "compile.log"
        assert not log_file.exists()

    def test_write_compile_event(self, tmp_path, monkeypatch) -> None:
        """record_compile_event() writes a JSON line to the log."""
        log_file = tmp_path / "compile.log"
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "1")
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(log_file))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)

        event = CompileEvent(
            timestamp="2026-05-06T14:30:45.123Z",
            name="signal_response",
            method="vi",
            signature="(('model_sig',),)",
            duration_s=5.2,
            inferred_cache_hit=False,
        )
        cl.record_compile_event(event)

        # Verify file was written with correct JSON
        assert log_file.exists()
        with open(log_file) as f:
            line = f.readline()
        data = json.loads(line)
        assert data["name"] == "signal_response"
        assert data["method"] == "vi"
        assert data["duration_s"] == 5.2
        assert data["inferred_cache_hit"] is False

    def test_record_dict_event(self, tmp_path, monkeypatch) -> None:
        """record_compile_event() accepts a dict as well as CompileEvent."""
        log_file = tmp_path / "compile.log"
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "1")
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(log_file))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)

        event_dict = {
            "timestamp": "2026-05-06T14:30:45.123Z",
            "name": "run_map",
            "method": "map",
            "signature": "sig",
            "duration_s": 2.1,
            "inferred_cache_hit": True,
        }
        cl.record_compile_event(event_dict)

        assert log_file.exists()
        with open(log_file) as f:
            data = json.loads(f.readline())
        assert data["name"] == "run_map"

    def test_record_event_thread_safe(self, tmp_path, monkeypatch) -> None:
        """record_compile_event() is thread-safe with concurrent writes."""
        log_file = tmp_path / "compile.log"
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "1")
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(log_file))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)

        def write_events(thread_id: int, count: int) -> None:
            for i in range(count):
                event = CompileEvent(
                    timestamp="2026-05-06T14:30:45.123Z",
                    name=f"event_{thread_id}_{i}",
                    method=f"method_{thread_id}",
                    signature=f"sig_{thread_id}",
                    duration_s=1.0,
                    inferred_cache_hit=False,
                )
                cl.record_compile_event(event)

        threads = [threading.Thread(target=write_events, args=(tid, 25)) for tid in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all 100 events (4 threads × 25 events) were written
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 100
        assert all(json.loads(line) for line in lines)


class TestCompileTimer:
    """Tests for compile_timer() context manager."""

    def test_timer_no_op_when_disabled(self, monkeypatch) -> None:
        """When disabled, compile_timer() is a no-op."""
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "0")

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)

        with cl.compile_timer("test", ("sig",), method="test"):
            time.sleep(0.01)

        # No assertion — just verify no exception is raised

    def test_timer_records_event(self, tmp_path, monkeypatch) -> None:
        """compile_timer() records timing and signature."""
        log_file = tmp_path / "compile.log"
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "1")
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(log_file))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)

        sig = (("shape", (12,)), ("method", "test"))
        with cl.compile_timer("test_phase", sig, method="test_method"):
            time.sleep(0.05)

        with open(log_file) as f:
            data = json.loads(f.readline())

        assert data["name"] == "test_phase"
        assert data["method"] == "test_method"
        assert data["duration_s"] >= 0.05
        assert "shape" in data["signature"]

    def test_timer_inferred_cache_hit(self, tmp_path, monkeypatch) -> None:
        """compile_timer() sets inferred_cache_hit based on duration."""
        log_file = tmp_path / "compile.log"
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "1")
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(log_file))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)

        # Fast (< 1 s) should be inferred as cache hit
        with cl.compile_timer("fast", ("sig",)):
            pass

        with open(log_file) as f:
            data = json.loads(f.readline())

        assert data["inferred_cache_hit"] is True
        assert data["duration_s"] < 1.0

    def test_timer_exception_still_records(self, tmp_path, monkeypatch) -> None:
        """compile_timer() records the event even if an exception is raised."""
        log_file = tmp_path / "compile.log"
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "1")
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(log_file))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)

        try:
            with cl.compile_timer("failing", ("sig",)):
                raise ValueError("test error")
        except ValueError:
            pass

        assert log_file.exists()
        with open(log_file) as f:
            data = json.loads(f.readline())
        assert data["name"] == "failing"


class TestCompileLogIntegration:
    """Integration tests using actual compile scenarios."""

    def test_multiple_phases(self, tmp_path, monkeypatch) -> None:
        """Simulate a multi-phase workflow (forward + inference)."""
        log_file = tmp_path / "compile.log"
        monkeypatch.setenv("TENGRI_LOG_COMPILES", "1")
        monkeypatch.setenv("TENGRI_COMPILE_LOG_PATH", str(log_file))

        from importlib import reload

        import tengri.utils.compile_log as cl

        reload(cl)

        # Simulate forward compile (fast, cache hit)
        with cl.compile_timer("forward", ("model_sig",), method=None):
            time.sleep(0.001)

        # Simulate MAP compile (slow, cache miss) — sleep > 1s to ensure cache miss
        with cl.compile_timer("run_map", ("model_sig", "map_sig"), method="map"):
            time.sleep(1.1)

        # Simulate HMC compile (slow, cache miss)
        with cl.compile_timer("run_hmc", ("model_sig", "hmc_sig"), method="mcmc_hmc"):
            time.sleep(1.1)

        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 3
        events = [json.loads(line) for line in lines]
        assert events[0]["name"] == "forward"
        assert events[1]["name"] == "run_map"
        assert events[2]["name"] == "run_hmc"
        assert events[0]["inferred_cache_hit"] is True
        assert events[1]["inferred_cache_hit"] is False
        assert events[2]["inferred_cache_hit"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
