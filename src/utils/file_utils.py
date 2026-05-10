import os
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Generator


@contextmanager
def temp_file_from_bytes(file_bytes: bytes, suffix: str = ".pdf") -> Generator[str, None, None]:
    """Write bytes to a temporary file, yield the path, and clean up afterwards.

    Args:
        file_bytes: The byte payload to write.
        suffix: File extension for the temporary file.

    Yields:
        The string path to the generated temporary file.
    """
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        yield tmp_path
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
