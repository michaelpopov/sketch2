"""ctypes wrapper for libsketch2.so."""

from __future__ import annotations

from collections.abc import Iterable

import ctypes
import os
from ctypes import POINTER, c_bool, c_char, c_char_p, c_double, c_int, c_size_t, c_uint, c_uint64, c_void_p
from pathlib import Path


class Sketch2Error(RuntimeError):
    """Raised when a libsketch2 call reports an error code and message."""
    def __init__(self, operation: str, message: str, code: int = -1):
        super().__init__(f"{operation} failed (code={code}): {message}")
        self.operation = operation
        self.code = code
        self.message = message


class Sketch2:
    """Python-facing wrapper around the sketch2 C API.

    The class exists to hide the raw ctypes configuration and expose the
    dataset lifecycle, mutation, query, and diagnostic operations as Python methods.
    """
    def __init__(self, db_path: str | Path, lib_path: str | Path | None = None):
        self.lib_path = Path(lib_path) if lib_path else self._default_lib_path()
        if not self.lib_path.exists():
            raise FileNotFoundError(f"libsketch2.so not found at: {self.lib_path}")

        self.db_path = Path(db_path)
        self.lib = ctypes.CDLL(str(self.lib_path))
        self._configure()
        self.handle = self.lib.sk_new_handle(str(self.db_path).encode("utf-8"))
        if not self.handle:
            raise RuntimeError("sk_new_handle() returned null handle")

    # Temporary setting for the shared library search path.
    # TODO: Think about a better way to set it.
    @staticmethod
    def _default_lib_path() -> Path:
        configured_dir = os.environ.get("SKETCH2_LIB")
        if configured_dir:
            configured_path = Path(configured_dir).resolve() / "libsketch2.so"
            if configured_path.exists():
                return configured_path

        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "bin-dbg-hwy" / "libsketch2.so",
            repo_root / "bin-hwy" / "libsketch2.so",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _configure(self) -> None:
        self.lib.sk_new_handle.argtypes = [c_char_p]
        self.lib.sk_new_handle.restype = c_void_p

        self.lib.sk_release_handle.argtypes = [c_void_p]
        self.lib.sk_release_handle.restype = None

        self.lib.sk_create.argtypes = [
            c_void_p,
            c_char_p,
            c_char_p,
            c_uint,
            c_char_p,
            c_uint,
            c_char_p,
        ]
        self.lib.sk_create.restype = c_int

        self.lib.sk_drop.argtypes = [c_void_p, c_char_p]
        self.lib.sk_drop.restype = c_int

        self.lib.sk_open.argtypes = [c_void_p, c_char_p]
        self.lib.sk_open.restype = c_int

        self.lib.sk_close.argtypes = [c_void_p]
        self.lib.sk_close.restype = c_int

        self.lib.sk_knn.argtypes = [
            c_void_p,
            c_char_p,
            c_uint,
            POINTER(POINTER(c_uint64)),
            POINTER(c_size_t),
        ]
        self.lib.sk_knn.restype = c_int

        self.lib.sk_merge_delta.argtypes = [c_void_p]
        self.lib.sk_merge_delta.restype = c_int

        self.lib.sk_get.argtypes = [c_void_p, c_uint64, POINTER(c_char_p)]
        self.lib.sk_get.restype = c_int

        self.lib.sk_free.argtypes = [c_void_p]
        self.lib.sk_free.restype = None

        self.lib.sk_print.argtypes = [c_void_p]
        self.lib.sk_print.restype = c_int

        self.lib.sk_start_writing.argtypes = [c_void_p]
        self.lib.sk_start_writing.restype = c_int

        self.lib.sk_write_vector.argtypes = [c_void_p, c_uint64, c_char_p]
        self.lib.sk_write_vector.restype = c_int

        self.lib.sk_write_deleted.argtypes = [c_void_p, c_uint64]
        self.lib.sk_write_deleted.restype = c_int

        self.lib.sk_abort_writing.argtypes = [c_void_p]
        self.lib.sk_abort_writing.restype = c_int

        self.lib.sk_complete_writing.argtypes = [c_void_p]
        self.lib.sk_complete_writing.restype = c_int

        self.lib.sk_generate_test_data.argtypes = [c_void_p, c_char_p, c_uint64, c_uint64, c_char_p, c_bool]
        self.lib.sk_generate_test_data.restype = c_int

        self.lib.sk_generate_test_metadata.argtypes = [c_void_p, c_char_p, c_uint64, c_uint64]
        self.lib.sk_generate_test_metadata.restype = c_int

        self.lib.sk_load_file.argtypes = [c_void_p, c_char_p]
        self.lib.sk_load_file.restype = c_int

        self.lib.sk_stats.argtypes = [c_void_p, c_char_p]
        self.lib.sk_stats.restype = c_int

        self.lib.sk_set_log_level.argtypes = [c_char_p]
        self.lib.sk_set_log_level.restype = None

        self.lib.sk_version.argtypes = [POINTER(c_char), c_size_t]
        self.lib.sk_version.restype = None

        self.lib.sk_error.argtypes = [c_void_p]
        self.lib.sk_error.restype = c_int

        self.lib.sk_error_message.argtypes = [c_void_p]
        self.lib.sk_error_message.restype = c_char_p

    @staticmethod
    def _format_dirs_arg(dirs: str | Iterable[str] | None) -> bytes | None:
        if dirs is None:
            return None
        if isinstance(dirs, (bytes, bytearray)):
            return bytes(dirs)
        if isinstance(dirs, str):
            return dirs.encode("utf-8")
        if isinstance(dirs, Iterable):
            return ", ".join(str(entry) for entry in dirs).encode("utf-8")
        raise TypeError("dirs must be None, a string, bytes, or an iterable of strings")

    def close_handle(self) -> None:
        if self.handle:
            self.lib.sk_release_handle(self.handle)
            self.handle = None

    def __enter__(self) -> "Sketch2":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_handle()

    def _require_connected(self) -> None:
        if not self.handle:
            raise RuntimeError("Sketch2 handle is not connected.")

    def _check(self, operation: str, rc: int) -> None:
        self._require_connected()
        if rc == 0:
            return
        raise Sketch2Error(operation, self.error_message(), self.error())

    def error(self) -> int:
        self._require_connected()
        return int(self.lib.sk_error(self.handle))

    def error_message(self) -> str:
        self._require_connected()
        msg = self.lib.sk_error_message(self.handle)
        if not msg:
            return ""
        return msg.decode("utf-8", errors="replace")

    def create(self, name: str, type_name: str = "f32", dim: int = 4,
               range_size: int = 1000, dist_func: str = "dot",
               dirs: str | Iterable[str] | None = None) -> None:
        dirs_arg = self._format_dirs_arg(dirs)
        self._check(
            "sk_create",
            self.lib.sk_create(
                self.handle,
                name.encode("utf-8"),
                dirs_arg,
                c_uint(dim),
                type_name.encode("utf-8"),
                c_uint(range_size),
                dist_func.encode("utf-8"),
            ),
        )
    def drop(self, name: str) -> None:
        self._check("sk_drop", self.lib.sk_drop(self.handle, name.encode("utf-8")))

    def open(self, name: str) -> None:
        self._check("sk_open", self.lib.sk_open(self.handle, name.encode("utf-8")))

    def close(self) -> None:
        self._check("sk_close", self.lib.sk_close(self.handle))

    def merge_delta(self) -> None:
        self._check("sk_merge_delta", self.lib.sk_merge_delta(self.handle))

    def knn(self, vec: str, count: int) -> list[int]:
        if count < 1:
            raise ValueError("count must be >= 1")

        ids = POINTER(c_uint64)()
        size = c_size_t()
        self._check(
            "sk_knn",
            self.lib.sk_knn(
                self.handle,
                vec.encode("utf-8"),
                c_uint(count),
                ctypes.byref(ids),
                ctypes.byref(size),
            ),
        )
        try:
            return [int(ids[index]) for index in range(size.value)]
        finally:
            if ids:
                self.lib.sk_free(ctypes.cast(ids, c_void_p))

    def get(self, item_id: int) -> str:
        out = c_char_p()
        self._check("sk_get", self.lib.sk_get(self.handle, c_uint64(item_id), ctypes.byref(out)))
        try:
            if not out:
                return ""
            return ctypes.string_at(out).decode("utf-8", errors="replace")
        finally:
            if out:
                self.lib.sk_free(ctypes.cast(out, c_void_p))

    def print(self) -> None:
        self._check("sk_print", self.lib.sk_print(self.handle))

    def start_writing(self) -> None:
        self._check("sk_start_writing", self.lib.sk_start_writing(self.handle))

    def write_vector(self, item_id: int, data: str) -> None:
        self._check(
            "sk_write_vector",
            self.lib.sk_write_vector(self.handle, c_uint64(item_id), data.encode("utf-8")),
        )

    def write_deleted(self, item_id: int) -> None:
        self._check("sk_write_deleted", self.lib.sk_write_deleted(self.handle, c_uint64(item_id)))

    def abort_writing(self) -> None:
        self._check("sk_abort_writing", self.lib.sk_abort_writing(self.handle))

    def complete_writing(self) -> None:
        self._check("sk_complete_writing", self.lib.sk_complete_writing(self.handle))

    def generate_test_data(
        self,
        file_path: str | Path,
        count: int,
        start_id: int | None = None,
        pattern: str | None = None,
        binary: bool = False,
    ) -> None:
        if start_id is None:
            start_id = 0
        pattern_arg = None if pattern is None else pattern.encode("utf-8")
        self._check(
            "generate_test_data",
            self.lib.sk_generate_test_data(
                self.handle,
                str(file_path).encode("utf-8"),
                c_uint64(count),
                c_uint64(start_id),
                pattern_arg,
                c_bool(bool(binary)),
            ),
        )

    def generate_test_metadata(
        self,
        file_path: str | Path,
        count: int,
        start_id: int | None = None,
    ) -> None:
        if start_id is None:
            start_id = 0
        self._check(
            "generate_test_metadata",
            self.lib.sk_generate_test_metadata(
                self.handle,
                str(file_path).encode("utf-8"),
                c_uint64(count),
                c_uint64(start_id),
            ),
        )

    def load_file(self, path: str | Path) -> None:
        self._check("sk_load_file", self.lib.sk_load_file(self.handle, str(path).encode("utf-8")))

    def stats(self, path: str | Path | None = None) -> None:
        encoded = b"" if path is None else str(path).encode("utf-8")
        self._check("sk_stats", self.lib.sk_stats(self.handle, encoded))

    def set_log_level(self, level: str) -> None:
        self.lib.sk_set_log_level(level.encode("utf-8"))

    def version(self, buf_size: int = 128) -> str:
        if buf_size < 1:
            raise ValueError("buf_size must be >= 1")
        buf = ctypes.create_string_buffer(buf_size)
        self.lib.sk_version(buf, c_size_t(buf_size))
        return buf.value.decode("utf-8", errors="replace")
