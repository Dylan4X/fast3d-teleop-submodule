"""CLI entry point for fast3d-build-trt."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "build_trt.py"


def main(argv: list[str] | None = None) -> None:
    if _SCRIPT.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_trt", _SCRIPT)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.main(argv)
    else:
        print(f"ERROR: script not found: {_SCRIPT}", file=sys.stderr)
        sys.exit(1)
