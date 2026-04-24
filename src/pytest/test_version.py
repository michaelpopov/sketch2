from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from integ_helpers import lib_path
from sketch2_wrapper import Sketch2


def expected_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    shared_consts = repo_root / "src" / "core" / "utils" / "shared_consts.h"
    text = shared_consts.read_text(encoding="utf-8")
    match = re.search(r'kSketch2Version\[\]\s*=\s*"([^"]*)"', text)
    if not match:
        raise AssertionError("Failed to parse kSketch2Version from shared_consts.h")
    return match.group(1)


class Sketch2VersionTest(unittest.TestCase):
    def test_version_matches_native_constant(self) -> None:
        expected = expected_version()
        with tempfile.TemporaryDirectory(prefix="sketch2_version_test_") as tmpdir:
            with Sketch2(tmpdir, lib_path=lib_path()) as ps:
                self.assertEqual(expected, ps.version())


if __name__ == "__main__":
    unittest.main()
