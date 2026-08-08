"""Writes that either land completely or not at all.

Every cache in this project follows the same shape: if the file is there, trust
it. That is the right design -- re-downloading a 500 MB point cloud to prove it
is still fine would be absurd -- but it puts the whole weight of correctness on
a file never existing in a partial state.

It failed once, expensively. ``points.laz`` was written straight to its final
name, a download died partway through, and because a truncated LAZ still
advertises the full point count in its header, every later retry picked it up
and believed it. A tile built from a fraction of its ground returns is a wrong
model rather than a failed build, and nothing anywhere would have said so.

So writes go to a temporary in the destination directory and are renamed into
place only on success. Same directory because ``os.replace`` is only atomic
within one filesystem, and a temp file in ``/tmp`` would silently degrade to a
copy across devices -- reintroducing exactly the partial-file window this
exists to close.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def staged(final: Path, suffix: str | None = None):
    """A temporary path beside [final], moved onto it if the block succeeds.

    The temporary is hidden (leading dot) and says what it is, so a cloud left
    by a killed process is recognisable and stays out of glob patterns that go
    looking for finished data.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        dir=final.parent,
        prefix=f".{final.stem}-partial-",
        suffix=final.suffix if suffix is None else suffix)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        os.replace(tmp, final)
    finally:
        # A no-op once the rename has happened. On any failure this is what
        # stops the next attempt inheriting a partial file under any name.
        tmp.unlink(missing_ok=True)


def write_text(final: Path, text: str, encoding: str = "utf-8") -> None:
    """Write [text] to [final], leaving it untouched if anything goes wrong."""
    with staged(final) as tmp:
        tmp.write_text(text, encoding=encoding)
