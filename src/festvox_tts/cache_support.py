"""Small cache primitives shared by the FestVox runtime.

These caches own only reproducible in-memory data.  Clearing them never
touches source voicebanks, generated voices, projects, exports, or other
files.  File-backed model values are invalidated by path, timestamps, size,
filesystem identity, and a content digest.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import fields, is_dataclass
import hashlib
import os
from pathlib import Path
import sys
import threading
from typing import Callable, Generic, Iterator, TypeVar


T = TypeVar("T")


class FrozenDict(dict):
    """Read-only ``dict`` used for cache values shared with callers."""

    @staticmethod
    def _immutable(*_args, **_kwargs):
        raise TypeError("cached metadata is read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self


class ReadOnlyMapping(Mapping):
    """O(1) read-only view over a mapping, freezing nested values lazily."""

    def __init__(self, value):
        self._value = value

    def __getitem__(self, key):
        return read_only_view(self._value[key])

    def __iter__(self):
        return iter(self._value)

    def __len__(self):
        return len(self._value)

    def __eq__(self, other):
        return self._value == other

    def __repr__(self):
        return repr(self._value)


class ReadOnlySequence(Sequence):
    """O(1) sequence view used with :class:`ReadOnlyMapping`."""

    def __init__(self, value):
        self._value = value

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(read_only_view(item) for item in self._value[index])
        return read_only_view(self._value[index])

    def __len__(self):
        return len(self._value)

    def __eq__(self, other):
        return self._value == other

    def __repr__(self):
        return repr(self._value)


def read_only_view(value):
    """Publish a nested container without recursively copying or walking it."""
    if isinstance(value, Mapping):
        return ReadOnlyMapping(value)
    if isinstance(value, (tuple, list)):
        return ReadOnlySequence(value)
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    return value


def deep_freeze(value):
    """Recursively freeze JSON-like data before publishing it from a cache."""
    if isinstance(value, Mapping):
        return FrozenDict({key: deep_freeze(item)
                           for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(item) for item in value)
    return value


def _file_digest(path: Path) -> bytes:
    """Return a deterministic digest for small configuration/model files."""
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as source:
        while True:
            block = source.read(64 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.digest()


_WINDOWS_CHANGE_API = None
_WINDOWS_CHANGE_API_FAILED = False


def _windows_file_change_time(path: Path) -> int:
    """Return the NT file-change timestamp, distinct from restored mtime.

    Python exposes creation time as ``st_ctime`` on Windows.  NTFS also
    maintains a ChangeTime field which advances for in-place data and
    metadata writes, including writes whose size and mtime are restored.
    """
    global _WINDOWS_CHANGE_API, _WINDOWS_CHANGE_API_FAILED
    if _WINDOWS_CHANGE_API_FAILED:
        raise OSError("Windows file-change timestamps are unavailable")
    if _WINDOWS_CHANGE_API is None:
        try:
            import ctypes
            from ctypes import wintypes

            class FileBasicInfo(ctypes.Structure):
                _fields_ = [
                    ("CreationTime", ctypes.c_longlong),
                    ("LastAccessTime", ctypes.c_longlong),
                    ("LastWriteTime", ctypes.c_longlong),
                    ("ChangeTime", ctypes.c_longlong),
                    ("FileAttributes", wintypes.DWORD),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateFileW.argtypes = [
                wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                wintypes.HANDLE,
            ]
            kernel32.CreateFileW.restype = wintypes.HANDLE
            kernel32.GetFileInformationByHandleEx.argtypes = [
                wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                wintypes.DWORD,
            ]
            kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            _WINDOWS_CHANGE_API = (ctypes, kernel32, FileBasicInfo)
        except (AttributeError, ImportError, OSError):
            _WINDOWS_CHANGE_API_FAILED = True
            raise OSError("Windows file-change timestamps are unavailable")

    ctypes, kernel32, file_basic_info = _WINDOWS_CHANGE_API
    # FILE_READ_ATTRIBUTES, all sharing modes, OPEN_EXISTING.
    handle = kernel32.CreateFileW(str(path), 0x80, 0x7, None, 0x3, 0x80,
                                  None)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = file_basic_info()
        if not kernel32.GetFileInformationByHandleEx(
                handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not info.ChangeTime:
            raise OSError("filesystem returned no file-change timestamp")
        return int(info.ChangeTime)
    finally:
        kernel32.CloseHandle(handle)


def file_change_token(path: Path | str, stat=None) -> tuple[str, object]:
    """Return a cheap token that changes for in-place file rewrites.

    The hot WAV path cannot hash entire recordings on every cache lookup.
    POSIX ctime and Windows' native ChangeTime cover same-inode, same-size,
    restored-mtime writes.  A digest is the correctness fallback on Windows
    filesystems that do not expose ChangeTime.
    """
    source = Path(path)
    current = stat if stat is not None else source.stat()
    if os.name == "nt":
        try:
            return "windows-change-time", _windows_file_change_time(source)
        except OSError:
            return "content-digest", _file_digest(source)
    return "posix-ctime", int(getattr(current, "st_ctime_ns", 0))


def estimate_size_bytes(value: object, _seen: set[int] | None = None) -> int:
    """Return a deterministic-enough in-process size estimate.

    ``sys.getsizeof`` already includes storage for bytes, strings, arrays,
    and most scalar objects.  Containers and dataclasses are traversed while
    object identities prevent double counting shared values.  NumPy arrays
    are recognized without importing NumPy.
    """
    if _seen is None:
        _seen = set()
    identity = id(value)
    if identity in _seen:
        return 0
    _seen.add(identity)
    try:
        size = int(sys.getsizeof(value))
    except TypeError:
        size = 0

    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and hasattr(value, "shape"):
        return max(size, nbytes)
    if isinstance(value, Mapping):
        return size + sum(
            estimate_size_bytes(key, _seen) +
            estimate_size_bytes(item, _seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset, OrderedDict)):
        return size + sum(estimate_size_bytes(item, _seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return size + sum(
            estimate_size_bytes(getattr(value, item.name), _seen)
            for item in fields(value)
        )
    return size


class FileIdentityCache(Generic[T]):
    """Thread-safe bounded LRU for immutable values parsed from files."""

    def __init__(self, owner: str, *, max_entries: int = 4,
                 max_bytes: int = 8 * 1024 * 1024):
        self.owner = str(owner)
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1, int(max_bytes))
        self._items: OrderedDict[tuple, tuple[T, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.RLock()

    @staticmethod
    def _identity(path: Path | str) -> tuple[Path, tuple]:
        source = Path(path).expanduser().resolve()
        stat = source.stat()
        # These files are only a few KiB.  The digest closes the same-size,
        # restored-timestamp replacement hole without burdening WAV caches.
        return source, (
            str(source), int(stat.st_mtime_ns), int(stat.st_size),
            int(getattr(stat, "st_dev", 0)), int(getattr(stat, "st_ino", 0)),
            _file_digest(source),
        )

    def get(self, path: Path | str, loader: Callable[[Path], T]) -> T:
        source, key = self._identity(path)
        with self._lock:
            cached = self._items.pop(key, None)
            if cached is not None:
                self._items[key] = cached
                self._hits += 1
                return cached[0]
            self._misses += 1

            # A changed file replaces its stale identity immediately rather
            # than occupying another slot until ordinary LRU eviction.
            for old_key in [item for item in self._items
                            if item[0] == key[0]]:
                _old_value, old_bytes = self._items.pop(old_key)
                self._bytes -= old_bytes
                self._evictions += 1

            value = loader(source)
            byte_count = estimate_size_bytes(value)
            if byte_count <= self.max_bytes:
                self._items[key] = (value, byte_count)
                self._bytes += byte_count
                while (len(self._items) > self.max_entries or
                       self._bytes > self.max_bytes):
                    _old_key, (_old_value, old_bytes) = \
                        self._items.popitem(last=False)
                    self._bytes -= old_bytes
                    self._evictions += 1
            return value

    def clear(self) -> dict[str, int | str]:
        with self._lock:
            removed_entries = len(self._items)
            removed_bytes = self._bytes
            self._items.clear()
            self._bytes = 0
            return {
                "owner": self.owner,
                "entries": removed_entries,
                "bytes": removed_bytes,
            }

    def info(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "owner": self.owner,
                "entries": len(self._items),
                "bytes": self._bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }


class BoundedMemoryCache(MutableMapping, Generic[T]):
    """Thread-safe entry/byte-bounded LRU with ordinary mapping semantics."""

    def __init__(self, owner: str, *, max_entries: int, max_bytes: int,
                 size_func: Callable[[T], int] | None = None):
        self.owner = str(owner)
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1, int(max_bytes))
        self._size_func = size_func or estimate_size_bytes
        self._items = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            try:
                value, byte_count = self._items.pop(key)
            except KeyError:
                self._misses += 1
                raise
            self._items[key] = (value, byte_count)
            self._hits += 1
            return value

    def __setitem__(self, key, value: T):
        byte_count = max(0, int(self._size_func(value)))
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            if byte_count > self.max_bytes:
                return
            self._items[key] = (value, byte_count)
            self._bytes += byte_count
            while (len(self._items) > self.max_entries or
                   self._bytes > self.max_bytes):
                _old_key, (_old_value, old_bytes) = \
                    self._items.popitem(last=False)
                self._bytes -= old_bytes
                self._evictions += 1

    def __delitem__(self, key):
        with self._lock:
            _value, byte_count = self._items.pop(key)
            self._bytes -= byte_count

    def __iter__(self) -> Iterator:
        with self._lock:
            return iter(tuple(self._items))

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def peek(self, key, default=None):
        """Read without changing LRU order; useful for diagnostic accounting."""
        with self._lock:
            row = self._items.get(key)
            return default if row is None else row[0]

    def info(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "owner": self.owner,
                "entries": len(self._items),
                "bytes": self._bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }
