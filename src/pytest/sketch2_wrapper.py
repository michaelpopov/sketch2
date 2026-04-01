"""ctypes wrapper for libsketch2.so."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_char_p, c_double, c_int, c_size_t, c_uint, c_uint64, c_void_p
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

        self.handle = self.lib.sk_connect(str(self.db_path).encode("utf-8"))
        if not self.handle:
            raise RuntimeError("sk_connect() returned null handle")

        self._open_datasets: list[str] = []

    # Temporary setting for the shared library search path.
    # TODO: Think about a better way to set it.
    @staticmethod
    def _default_lib_path() -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "bin" / "libsketch2.so",
            repo_root / "bin-dbg" / "libsketch2.so",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _configure(self) -> None:
        self.lib.sk_connect.argtypes = [c_char_p]
        self.lib.sk_connect.restype = c_void_p

        self.lib.sk_disconnect.argtypes = [c_void_p]
        self.lib.sk_disconnect.restype = None

        self.lib.sk_create.argtypes = [c_void_p, c_char_p, c_uint, c_char_p, c_uint, c_char_p]
        self.lib.sk_create.restype = c_int

        self.lib.sk_drop.argtypes = [c_void_p, c_char_p]
        self.lib.sk_drop.restype = c_int

        self.lib.sk_open.argtypes = [c_void_p, c_char_p]
        self.lib.sk_open.restype = c_int

        self.lib.sk_close.argtypes = [c_void_p, c_char_p]
        self.lib.sk_close.restype = c_int

        self.lib.sk_knn.argtypes = [
            c_void_p,
            c_char_p,
            c_uint,
            POINTER(POINTER(c_uint64)),
            POINTER(c_size_t),
        ]
        self.lib.sk_knn.restype = c_int

        self.lib.sk_mdelta.argtypes = [c_void_p]
        self.lib.sk_mdelta.restype = c_int

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

        self.lib.sk_complete_writing.argtypes = [c_void_p]
        self.lib.sk_complete_writing.restype = c_int

        self.lib.sk_generate.argtypes = [c_void_p, c_uint64, c_uint64, c_int]
        self.lib.sk_generate.restype = c_int

        self.lib.sk_generate_bin.argtypes = [c_void_p, c_uint64, c_uint64, c_int]
        self.lib.sk_generate_bin.restype = c_int

        self.lib.sk_load_file.argtypes = [c_void_p, c_char_p]
        self.lib.sk_load_file.restype = c_int

        self.lib.sk_stats.argtypes = [c_void_p]
        self.lib.sk_stats.restype = c_int

        self.lib.sk_error.argtypes = [c_void_p]
        self.lib.sk_error.restype = c_int

        self.lib.sk_error_message.argtypes = [c_void_p]
        self.lib.sk_error_message.restype = c_char_p

    def close_handle(self) -> None:
        if self.handle:
            for name in self._open_datasets:
                try:
                    self.lib.sk_close(self.handle, name.encode("utf-8"))
                except Exception:
                    pass
            self._open_datasets.clear()
            self.lib.sk_disconnect(self.handle)
            self.handle = None

    def __enter__(self) -> "Sketch2":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_handle()

    def _check(self, operation: str, rc: int) -> None:
        if rc == 0:
            return
        raise Sketch2Error(operation, self.error_message(), self.error())

    def error(self) -> int:
        return int(self.lib.sk_error(self.handle))

    def error_message(self) -> str:
        msg = self.lib.sk_error_message(self.handle)
        if not msg:
            return ""
        return msg.decode("utf-8", errors="replace")

    def create(self, name: str, type_name: str = "f32", dim: int = 4,
               range_size: int = 1000, dist_func: str = "l1") -> None:
        self._check(
            "sk_create",
            self.lib.sk_create(
                self.handle,
                name.encode("utf-8"),
                c_uint(dim),
                type_name.encode("utf-8"),
                c_uint(range_size),
                dist_func.encode("utf-8"),
            ),
        )
        self._open_datasets.append(name)

    def drop(self, name: str) -> None:
        self._check("sk_drop", self.lib.sk_drop(self.handle, name.encode("utf-8")))

    def open(self, name: str) -> None:
        self._check("sk_open", self.lib.sk_open(self.handle, name.encode("utf-8")))
        self._open_datasets.append(name)

    def close(self, name: str) -> None:
        self._check("sk_close", self.lib.sk_close(self.handle, name.encode("utf-8")))
        try:
            self._open_datasets.remove(name)
        except ValueError:
            pass

    def merge_delta(self) -> None:
        self._check("sk_mdelta", self.lib.sk_mdelta(self.handle))

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

    def complete_writing(self) -> None:
        self._check("sk_complete_writing", self.lib.sk_complete_writing(self.handle))

    def generate(self, count: int, start_id: int, pattern: int) -> None:
        self._check(
            "sk_generate",
            self.lib.sk_generate(self.handle, c_uint64(count), c_uint64(start_id), c_int(pattern)),
        )

    def generate_bin(self, count: int, start_id: int, pattern: int) -> None:
        self._check(
            "sk_generate_bin",
            self.lib.sk_generate_bin(self.handle, c_uint64(count), c_uint64(start_id), c_int(pattern)),
        )

    def load_file(self, path: str | Path) -> None:
        self._check("sk_load_file", self.lib.sk_load_file(self.handle, str(path).encode("utf-8")))

    def stats(self) -> None:
        self._check("sk_stats", self.lib.sk_stats(self.handle))
