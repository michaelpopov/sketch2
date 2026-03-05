"""ctypes wrapper for libparasol.so."""

from __future__ import annotations

import ctypes
from ctypes import c_char, c_char_p, c_int, c_uint, c_uint64, c_void_p, POINTER, byref
from pathlib import Path


class SkDatasetMetadata(ctypes.Structure):
    _fields_ = [
        ("dir", c_char * 256),
        ("type", c_char * 16),
        ("dim", c_uint),
        ("range_size", c_uint),
        ("data_merge_ratio", c_uint),
    ]


class ParasolError(RuntimeError):
    def __init__(self, operation: str, message: str, code: int = -1):
        super().__init__(f"{operation} failed (code={code}): {message}")
        self.operation = operation
        self.code = code
        self.message = message


class Parasol:
    def __init__(self, lib_path: str | Path | None = None):
        self.lib_path = Path(lib_path) if lib_path else self._default_lib_path()
        if not self.lib_path.exists():
            raise FileNotFoundError(f"libparasol.so not found at: {self.lib_path}")

        self.lib = ctypes.CDLL(str(self.lib_path))
        self._configure()

        self.handle = self.lib.connect()
        if not self.handle:
            raise RuntimeError("connect() returned null handle")

    @staticmethod
    def _default_lib_path() -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "build-dbg" / "lib" / "libparasol.so"

    def _configure(self) -> None:
        self.lib.connect.argtypes = []
        self.lib.connect.restype = c_void_p

        self.lib.disconnect.argtypes = [c_void_p]
        self.lib.disconnect.restype = None

        self.lib.sk_create.argtypes = [c_void_p, SkDatasetMetadata]
        self.lib.sk_create.restype = c_int

        self.lib.sk_drop.argtypes = [c_void_p]
        self.lib.sk_drop.restype = c_int

        self.lib.sk_open.argtypes = [c_void_p, c_char_p]
        self.lib.sk_open.restype = c_int

        self.lib.sk_close.argtypes = [c_void_p]
        self.lib.sk_close.restype = c_int

        self.lib.sk_add.argtypes = [c_void_p, c_uint64, c_char_p]
        self.lib.sk_add.restype = c_int

        self.lib.sk_delete.argtypes = [c_void_p, c_uint64]
        self.lib.sk_delete.restype = c_int

        self.lib.sk_load.argtypes = [c_void_p]
        self.lib.sk_load.restype = c_int

        self.lib.sk_knn.argtypes = [c_void_p, c_char_p, POINTER(c_uint64), POINTER(c_uint64)]
        self.lib.sk_knn.restype = c_int

        self.lib.sk_error.argtypes = [c_void_p]
        self.lib.sk_error.restype = c_int

        self.lib.sk_error_message.argtypes = [c_void_p]
        self.lib.sk_error_message.restype = c_char_p

    def close_handle(self) -> None:
        if self.handle:
            self.lib.disconnect(self.handle)
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

    @staticmethod
    def _metadata(dataset_dir: str | Path, type_name: str, dim: int, range_size: int, data_merge_ratio: int) -> SkDatasetMetadata:
        md = SkDatasetMetadata()
        md.dir = str(Path(dataset_dir)).encode("utf-8")
        md.type = type_name.encode("utf-8")
        md.dim = dim
        md.range_size = range_size
        md.data_merge_ratio = data_merge_ratio
        return md

    def create(
        self,
        dataset_dir: str | Path,
        type_name: str = "f32",
        dim: int = 4,
        range_size: int = 1000,
        data_merge_ratio: int = 2,
    ) -> None:
        md = self._metadata(dataset_dir, type_name, dim, range_size, data_merge_ratio)
        self._check("sk_create", self.lib.sk_create(self.handle, md))

    def drop(self) -> None:
        self._check("sk_drop", self.lib.sk_drop(self.handle))

    def open(self, dataset_dir: str | Path) -> None:
        self._check("sk_open", self.lib.sk_open(self.handle, str(Path(dataset_dir)).encode("utf-8")))

    def close(self) -> None:
        self._check("sk_close", self.lib.sk_close(self.handle))

    def add(self, item_id: int, value: str) -> None:
        self._check("sk_add", self.lib.sk_add(self.handle, c_uint64(item_id), value.encode("utf-8")))

    def delete(self, item_id: int) -> None:
        self._check("sk_delete", self.lib.sk_delete(self.handle, c_uint64(item_id)))

    def load(self) -> None:
        self._check("sk_load", self.lib.sk_load(self.handle))

    def knn(self, vec: str, count: int) -> list[int]:
        if count < 1:
            raise ValueError("count must be >= 1")

        ids = (c_uint64 * count)()
        ids_count = c_uint64(count)
        rc = self.lib.sk_knn(self.handle, vec.encode("utf-8"), ids, byref(ids_count))
        self._check("sk_knn", rc)
        actual = int(ids_count.value)
        return [int(ids[i]) for i in range(actual)]
