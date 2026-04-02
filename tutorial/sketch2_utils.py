"""Helper utilities shared by Sketch2 Python scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple


def get_lib_paths() -> Tuple[Path, Path]:
    raw = os.environ.get("SKETCH2_LIB")
    if not raw:
        raise SystemExit("SKETCH2_LIB must point to the directory containing libsketch2.so")
    lib_dir = Path(raw)
    if not lib_dir.exists():
        raise SystemExit(f"SKETCH2_LIB points to missing directory: {lib_dir}")
    if not lib_dir.is_dir():
        raise SystemExit(f"SKETCH2_LIB must be a directory, got: {lib_dir}")
    lib_path = lib_dir / "libsketch2.so"
    if not lib_path.exists():
        raise SystemExit(f"libsketch2.so not found inside SKETCH2_LIB directory: {lib_dir}")
    wrapper_path = lib_dir / "sketch2_wrapper.py"
    if not wrapper_path.exists():
        raise SystemExit(f"sketch2_wrapper.py not found inside SKETCH2_LIB directory: {lib_dir}")
    return lib_dir, lib_path


def get_db_path() -> Path:
    raw = os.environ.get("SKETCH2_CONFIG")
    if not raw:
        raise SystemExit("SKETCH2_CONFIG must point to the config file used by Sketch2")
    config_path = Path(raw)
    if not config_path.exists():
        raise SystemExit(f"SKETCH2_CONFIG points to missing file: {config_path}")
    if not config_path.is_file():
        raise SystemExit(f"SKETCH2_CONFIG must be a file, got: {config_path}")
    return config_path.resolve().parent


def load_sketch2_types(lib_dir: Path) -> Tuple["Sketch2", "Sketch2Error"]:
    str_path = str(lib_dir.resolve())
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
    try:
        from sketch2_wrapper import Sketch2, Sketch2Error
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Failed to import sketch2_wrapper from {lib_dir}: {exc}") from exc
    return Sketch2, Sketch2Error
