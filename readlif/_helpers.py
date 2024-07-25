from __future__ import annotations

import io
import os
import struct
from typing import TypeAlias

FileDescriptorOrPath: TypeAlias = (
    int | str | bytes | os.PathLike[str] | os.PathLike[bytes]
)


def read_u64(handle: io.BufferedReader) -> int:
    """Reads eight bytes, returns the uint64 (Private)."""
    (long_data,) = struct.unpack("Q", handle.read(8))

    if not isinstance(long_data, int):
        handle.close()
        msg = "Unable to read a long int from the file"
        raise ValueError(msg)

    return long_data


def read_u32(handle: io.BufferedReader) -> int:
    """Reads four bytes, returns the uint32 (Private)."""
    (int_data,) = struct.unpack("I", handle.read(4))

    if not isinstance(int_data, int):
        handle.close()
        msg = "Unable to read an int from the file"
        raise ValueError(msg)

    return int_data


def check_truncated(handle: io.BufferedReader) -> bool:
    """Checks if the LIF file is truncated by reading in 100 bytes."""
    handle.seek(-4, 1)
    if handle.read(100) == (b"\x00" * 100):
        handle.seek(-100, 1)
        return True
    handle.seek(-100, 1)
    return False


def check_magic(handle: io.BufferedReader, bool_return: bool = False) -> bool:
    """Checks for lif file magic bytes (Private)."""
    if handle.read(4) == b"\x70\x00\x00\x00":
        return True

    if not bool_return:
        curr_pos = handle.tell()
        handle.close()
        msg = f"This is probably not a LIF file. Expected LIF magic byte at {curr_pos}"
        raise ValueError(msg)

    return False


def check_mem(handle: io.BufferedReader, bool_return: bool = False) -> bool:
    """Checks for 'memory block' bytes (Private)."""
    if handle.read(1) == b"\x2a":
        return True

    if not bool_return:
        curr_pos = handle.tell()
        handle.close()
        msg = f"Expected LIF memory byte at {curr_pos}"
        raise ValueError(msg)

    return False


def get_len(handle: io.BufferedReader) -> int:
    """Returns total file length (Private)."""
    position = handle.tell()
    handle.seek(0, 2)
    file_len = handle.tell()
    handle.seek(position)
    return file_len
