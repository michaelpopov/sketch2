"""ctypes wrapper for libparasol.so."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_char_p, c_double, c_int, c_int64, c_uint, c_uint64, c_void_p, byref
from pathlib import Path


class ParasolError(RuntimeError):
    """Raised when a libparasol call reports an error code and message."""
    def __init__(self, operation: str, message: str, code: int = -1):
        super().__init__(f"{operation} failed (code={code}): {message}")
        self.operation = operation
        self.code = code
        self.message = message


class Parasol:
    """Python-facing wrapper around the parasol C API.

    The class exists to hide the raw ctypes configuration and expose the
    dataset lifecycle, mutation, query, and diagnostic operations as Python methods.
    """
    def __init__(self, db_path: str | Path, lib_path: str | Path | None = None):
        self.lib_path = Path(lib_path) if lib_path else self._default_lib_path()
        if not self.lib_path.exists():
            raise FileNotFoundError(f"libparasol.so not found at: {self.lib_path}")

        self.db_path = Path(db_path)
        self.lib = ctypes.CDLL(str(self.lib_path))
        self._configure()

        self.handle = self.lib.sk_connect(str(self.db_path).encode("utf-8"))
        if not self.handle:
            raise RuntimeError("sk_connect() returned null handle")

    @staticmethod
    def _default_lib_path() -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "build" / "lib" / "libparasol.so",
            repo_root / "bin" / "libparasol.so",
            repo_root / "build-dbg" / "lib" / "libparasol.so",
            repo_root / "bin-dbg" / "libparasol.so",
            repo_root / "build-san" / "lib" / "libparasol.so",
            repo_root / "bin-san" / "libparasol.so",
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

        self.lib.sk_upsert.argtypes = [c_void_p, c_uint64, c_char_p]
        self.lib.sk_upsert.restype = c_int

        self.lib.sk_ups2.argtypes = [c_void_p, c_uint64, c_double]
        self.lib.sk_ups2.restype = c_int

        self.lib.sk_del.argtypes = [c_void_p, c_uint64]
        self.lib.sk_del.restype = c_int

        self.lib.sk_knn.argtypes = [c_void_p, c_char_p, c_uint]
        self.lib.sk_knn.restype = c_int

        self.lib.sk_kres.argtypes = [c_void_p, c_int64]
        self.lib.sk_kres.restype = c_uint64

        self.lib.sk_macc.argtypes = [c_void_p]
        self.lib.sk_macc.restype = c_int

        self.lib.sk_mdelta.argtypes = [c_void_p]
        self.lib.sk_mdelta.restype = c_int

        self.lib.sk_get.argtypes = [c_void_p, c_uint64]
        self.lib.sk_get.restype = c_int

        self.lib.sk_gres.argtypes = [c_void_p]
        self.lib.sk_gres.restype = c_char_p

        self.lib.sk_gid.argtypes = [c_void_p, c_char_p]
        self.lib.sk_gid.restype = c_int

        self.lib.sk_ires.argtypes = [c_void_p, POINTER(c_uint64)]
        self.lib.sk_ires.restype = c_int

        self.lib.sk_print.argtypes = [c_void_p]
        self.lib.sk_print.restype = c_int

        self.lib.sk_generate.argtypes = [c_void_p, c_uint64, c_uint64, c_int]
        self.lib.sk_generate.restype = c_int

        self.lib.sk_stats.argtypes = [c_void_p]
        self.lib.sk_stats.restype = c_int

        self.lib.sk_error.argtypes = [c_void_p]
        self.lib.sk_error.restype = c_int

        self.lib.sk_error_message.argtypes = [c_void_p]
        self.lib.sk_error_message.restype = c_char_p

    def close_handle(self) -> None:
        if self.handle:
            self.lib.sk_disconnect(self.handle)
            self.handle = None

    def __enter__(self) -> "Parasol":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_handle()

    def _check(self, operation: str, rc: int) -> None:
        if rc == 0:
            return
        raise ParasolError(operation, self.error_message(), self.error())

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

    def drop(self, name: str) -> None:
        self._check("sk_drop", self.lib.sk_drop(self.handle, name.encode("utf-8")))

    def open(self, name: str) -> None:
        self._check("sk_open", self.lib.sk_open(self.handle, name.encode("utf-8")))

    def close(self, name: str) -> None:
        self._check("sk_close", self.lib.sk_close(self.handle, name.encode("utf-8")))

    def upsert(self, item_id: int, value: str) -> None:
        self._check("sk_upsert", self.lib.sk_upsert(self.handle, c_uint64(item_id), value.encode("utf-8")))

    def ups2(self, item_id: int, value: float) -> None:
        self._check("sk_ups2", self.lib.sk_ups2(self.handle, c_uint64(item_id), c_double(value)))

    def delete(self, item_id: int) -> None:
        self._check("sk_del", self.lib.sk_del(self.handle, c_uint64(item_id)))

    def merge_accumulator(self) -> None:
        self._check("sk_macc", self.lib.sk_macc(self.handle))

    def merge_delta(self) -> None:
        self._check("sk_mdelta", self.lib.sk_mdelta(self.handle))

    def knn(self, vec: str, count: int) -> list[int]:
        if count < 1:
            raise ValueError("count must be >= 1")

        self._check("sk_knn", self.lib.sk_knn(self.handle, vec.encode("utf-8"), c_uint(count)))
        size = int(self.lib.sk_kres(self.handle, c_int64(-1)))
        return [self.kres(index) for index in range(size)]

    def kres(self, index: int) -> int:
        return int(self.lib.sk_kres(self.handle, c_int64(index)))

    def get(self, item_id: int) -> str:
        self._check("sk_get", self.lib.sk_get(self.handle, c_uint64(item_id)))
        out = self.lib.sk_gres(self.handle)
        if not out:
            return ""
        return out.decode("utf-8", errors="replace")

    def gid(self, vec: str) -> int:
        self._check("sk_gid", self.lib.sk_gid(self.handle, vec.encode("utf-8")))
        value = c_uint64()
        self._check("sk_ires", self.lib.sk_ires(self.handle, byref(value)))
        return int(value.value)

    def print(self) -> None:
        self._check("sk_print", self.lib.sk_print(self.handle))

    def generate(self, count: int, start_id: int, pattern: int) -> None:
        self._check(
            "sk_generate",
            self.lib.sk_generate(self.handle, c_uint64(count), c_uint64(start_id), c_int(pattern)),
        )

    def stats(self) -> None:
        self._check("sk_stats", self.lib.sk_stats(self.handle))
