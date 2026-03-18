"""Shared helpers for sketch2 integration tests."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sketch2_wrapper import Sketch2

PYTEST_DIR = str(Path(__file__).resolve().parent)


def lib_path() -> str:
    """Return the path to the sketch2 shared library."""
    return str(Sketch2._default_lib_path())


def subprocess_env() -> dict[str, str]:
    """Return an os.environ copy with PYTEST_DIR prepended to PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTEST_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_subprocess(script: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a Python script in a child process with PYTHONPATH set."""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=subprocess_env(), timeout=timeout,
    )


def format_process_error(
    returncode: int,
    stdout: str,
    stderr: str,
    context: str = "",
) -> str:
    """Format a diagnostic message for a failed subprocess."""
    parts = []
    if context:
        parts.append(context)
    parts.append(f"exit code: {returncode}")
    if stderr and stderr.strip():
        parts.append(f"stderr:\n{stderr.strip()}")
    if stdout and stdout.strip():
        parts.append(f"stdout:\n{stdout.strip()}")
    return "\n".join(parts)


class IntegTestBase(unittest.TestCase):
    """Base class for integration tests with temp directory management."""

    _tmpdir_prefix: str = "sketch2_integ_"

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix=self._tmpdir_prefix))
        self.dataset_name = "ds"
        self.dataset_dir = self.root / self.dataset_name
        print(f"\n--- {type(self).__name__}.{self._testMethodName} ---", flush=True)

    def progress(self, msg: str) -> None:
        """Print a progress message (visible with pytest -s)."""
        print(f"  {msg}", flush=True)

    def tearDown(self) -> None:
        try:
            with Sketch2(self.root) as ps:
                ps.drop(self.dataset_name)
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)

    def count_files(self, ext: str) -> int:
        """Count files with the given extension in the dataset directory."""
        return len(list(self.dataset_dir.glob(f"*{ext}")))

    def ini_path(self) -> str:
        """Return path to the dataset ini file."""
        return str(self.root / f"{self.dataset_name}.ini")

    def assert_subprocess_ok(
        self,
        result: subprocess.CompletedProcess,
        context: str = "",
    ) -> None:
        """Assert subprocess exited with code 0; include full output on failure."""
        if result.returncode == 0:
            return
        self.fail(format_process_error(
            result.returncode, result.stdout, result.stderr, context,
        ))
