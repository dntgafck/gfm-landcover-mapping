"""Utility functions for inference server."""

import hashlib
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager for timing code blocks.

    Usage:
        with timer() as t:
            do_work()
        print(f"Elapsed: {t['elapsed']:.3f}s")
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


class TimingAccumulator:
    """Accumulate multiple timing measurements."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def time(self, name: str) -> Generator[None, None, None]:
        """Time a named section."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = time.perf_counter() - start

    def get_timings(self) -> dict[str, float]:
        """Get all recorded timings."""
        return dict(self.timings)


def generate_inference_id() -> str:
    """Generate a unique inference ID based on timestamp and random bytes.

    Returns:
        Unique ID like 'inf-20260107-190000-a1b2c3d4'
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    random_suffix = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:8]
    return f"inf-{timestamp}-{random_suffix}"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        The same path for chaining
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm to use

    Returns:
        Hex digest of file hash
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
