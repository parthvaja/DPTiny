"""Fashion-MNIST dataset loader.

Fashion-MNIST (Zalando Research) is a drop-in replacement for MNIST: 70,000
grayscale 28x28 images of clothing in 10 classes, split 60,000 / 10,000.

The four gzipped IDX files are downloaded from the project's GitHub on first
use, cached under ``~/.cache/dptiny/fashion_mnist`` and parsed with NumPy only.
``get_fashion_mnist`` returns exactly what ``get_mnist`` returns, so training
code can swap one call for the other.
"""

import gzip
import os
import tempfile
import urllib.request
from typing import Optional, Tuple

import numpy as np

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny", "fashion_mnist")

_BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
)

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Class names in label order (label 0 is "T-shirt/top", label 9 is "Ankle boot").
CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

_IDX_UINT8 = 0x08


def parse_idx(raw: bytes) -> np.ndarray:
    """Parse the bytes of an (uncompressed) uint8 IDX file into an array.

    IDX header layout (all integers big-endian):
        bytes 0-1   zero
        byte  2     data-type code (0x08 = unsigned byte)
        byte  3     number of dimensions, d
        next 4*d    size of each dimension as uint32
    followed by prod(sizes) raw data bytes.

    Raises:
        ValueError: if the file is not a uint8 IDX file, or its size does not
            match the header.
    """
    if len(raw) < 4:
        raise ValueError("Not an IDX file: shorter than the 4-byte magic number.")
    if raw[0] != 0 or raw[1] != 0:
        raise ValueError("Not an IDX file: the first two bytes must be zero.")
    if raw[2] != _IDX_UINT8:
        raise ValueError(
            f"Not a uint8 IDX file: data-type code is 0x{raw[2]:02x}, expected 0x08."
        )

    ndim = raw[3]
    if ndim == 0:
        raise ValueError("Invalid IDX file: number of dimensions is 0.")

    header_size = 4 + 4 * ndim
    if len(raw) < header_size:
        raise ValueError("Invalid IDX file: header is truncated.")

    shape = tuple(int(d) for d in np.frombuffer(raw, dtype=">u4", count=ndim, offset=4))
    expected = header_size + int(np.prod(shape))
    if len(raw) != expected:
        raise ValueError(
            f"IDX size mismatch: header {shape} needs {expected} bytes, "
            f"file has {len(raw)}."
        )

    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape)


def _read_idx_gz(path: str) -> np.ndarray:
    """Decompress a ``.gz`` IDX file and parse it."""
    with gzip.open(path, "rb") as f:
        return parse_idx(f.read())


def _download(url: str, dest: str) -> None:
    """Download ``url`` to ``dest`` unless it is already cached.

    The data is written to a temporary file in the same directory, checked,
    and only then renamed to ``dest`` with ``os.replace`` (an atomic rename).
    If the download fails or is interrupted (even with Ctrl-C), the temporary
    file is deleted, so the cache never holds a half-written file.
    """
    if os.path.exists(dest):
        return

    directory = os.path.dirname(dest)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as resp:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        _read_idx_gz(tmp_path)  # refuse to cache a corrupt download
        os.replace(tmp_path, dest)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the Fashion-MNIST dataset.

    Same interface and outputs as ``dptiny.data.mnist.get_mnist``.

    Args:
        normalize: If ``True``, scale pixel values to ``[0, 1]``.
        flatten: If ``True``, return images as ``(N, 784)`` vectors,
            otherwise as ``(N, 1, 28, 28)``.
        data_home: Directory to cache the downloaded files in.

    Returns:
        ``(X_train, X_test, y_train, y_test)``: float32 images and int32
        labels; 60,000 training and 10,000 test samples.
    """
    cache_dir = data_home if data_home is not None else _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    arrays = {}
    for key, filename in _FILES.items():
        path = os.path.join(cache_dir, filename)
        _download(_BASE_URL + filename, path)
        arrays[key] = _read_idx_gz(path)

    X_train, X_test = arrays["train_images"], arrays["test_images"]
    y_train, y_test = arrays["train_labels"], arrays["test_labels"]

    for name, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        if X.ndim != 3 or X.shape[1:] != (28, 28) or y.ndim != 1:
            raise ValueError(f"Unexpected {name} shapes: {X.shape}, {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{name}: {X.shape[0]} images but {y.shape[0]} labels")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test
