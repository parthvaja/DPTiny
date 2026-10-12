import gzip
import os
import struct

import numpy as np
import pytest

from dptiny.data.fashion_mnist import CLASSES, _read_idx_gz, parse_idx


def _idx_bytes(arr, type_code=0x08):
    header = struct.pack(">BBBB", 0, 0, type_code, arr.ndim)
    header += struct.pack(">" + "I" * arr.ndim, *arr.shape)
    return header + arr.astype(np.uint8).tobytes()


def test_parse_roundtrip():
    arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    out = parse_idx(_idx_bytes(arr))
    assert out.dtype == np.uint8
    assert np.array_equal(out, arr)


def test_parse_labels_1d():
    labels = np.array([9, 0, 3], dtype=np.uint8)
    assert np.array_equal(parse_idx(_idx_bytes(labels)), labels)


def test_rejects_non_uint8_type():
    with pytest.raises(ValueError):
        parse_idx(_idx_bytes(np.zeros(4, dtype=np.uint8), type_code=0x0D))


def test_rejects_bad_magic():
    raw = bytearray(_idx_bytes(np.zeros(4, dtype=np.uint8)))
    raw[0] = 1
    with pytest.raises(ValueError):
        parse_idx(bytes(raw))


def test_rejects_size_mismatch():
    raw = _idx_bytes(np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        parse_idx(raw[:-1])  # truncated
    with pytest.raises(ValueError):
        parse_idx(raw + b"\x00")  # extra byte


def test_rejects_short_header():
    with pytest.raises(ValueError):
        parse_idx(b"\x00\x00")


def test_read_gz(tmp_path):
    arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
    path = os.path.join(tmp_path, "x.gz")
    with gzip.open(path, "wb") as f:
        f.write(_idx_bytes(arr))
    assert np.array_equal(_read_idx_gz(path), arr)


def test_classes():
    assert isinstance(CLASSES, tuple)
    assert len(CLASSES) == 10
    assert CLASSES[0] == "T-shirt/top" and CLASSES[9] == "Ankle boot"
