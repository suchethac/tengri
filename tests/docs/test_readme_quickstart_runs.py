"""The README quick-start Python blocks run, with a synthetic SSP (#3452).

A new user copies these blocks in order, sharing one namespace. The only
substitutions are the SSP download (replaced by the session
``synthetic_ssp_wide`` fixture, so CI needs no grid on disk) and the inference
step (``fitter = ...`` onward), which takes minutes and is covered by the
fitting tests.
"""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"
_FENCE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
_SSP_LINE = "ssp = load_ssp_data(tengri.download_ssp())"


def _quick_start_blocks() -> list[str]:
    text = README.read_text(encoding="utf-8")
    start = text.index("\n## Quick start\n")
    end = text.index("\n## Tutorials", start)
    return _FENCE.findall(text[start:end])


def _without_download_or_inference(src: str) -> str:
    src = src.replace(_SSP_LINE, "pass  # ssp supplied by the test fixture")
    return src.split("\nfitter = ", 1)[0] + "\n"


def test_quick_start_blocks_run_in_order(synthetic_ssp_wide):
    blocks = _quick_start_blocks()
    assert len(blocks) >= 2
    assert any(_SSP_LINE in block for block in blocks), "SSP loading line moved"
    namespace: dict[str, object] = {"ssp": synthetic_ssp_wide}
    for index, block in enumerate(blocks):
        code = _without_download_or_inference(block)
        exec(compile(code, f"README.md quick start block {index}", "exec"), namespace)
    assert "sed" in namespace and "forward" in namespace
