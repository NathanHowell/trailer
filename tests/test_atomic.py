"""The write discipline every cache in this project depends on.

Each cache trusts a file's existence as proof of its completeness. That is only
sound if a partial file can never exist under the real name.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from trailer import atomic


def _names(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


def test_nothing_appears_until_the_block_succeeds(tmp_path):
    final = tmp_path / "cloud.laz"
    with atomic.staged(final) as tmp:
        tmp.write_bytes(b"half")
        assert not final.exists(), "visible before the block finished"
    assert final.read_bytes() == b"half"
    assert _names(tmp_path) == ["cloud.laz"]


def test_a_failure_leaves_no_trace(tmp_path):
    final = tmp_path / "cloud.laz"
    with pytest.raises(RuntimeError):
        with atomic.staged(final) as tmp:
            tmp.write_bytes(b"partial download")
            raise RuntimeError("connection reset")
    assert not final.exists()
    assert _names(tmp_path) == [], f"left behind: {_names(tmp_path)}"


def test_an_existing_file_survives_a_failed_rewrite(tmp_path):
    # Retrying over good data must not destroy it on the way to failing. A
    # truncate-then-write would be worse than never retrying.
    final = tmp_path / "cloud.laz"
    final.write_bytes(b"good")
    with pytest.raises(RuntimeError):
        with atomic.staged(final):
            raise RuntimeError("nope")
    assert final.read_bytes() == b"good"


def test_an_existing_file_is_replaced_on_success(tmp_path):
    final = tmp_path / "cloud.laz"
    final.write_bytes(b"old")
    with atomic.staged(final) as tmp:
        tmp.write_bytes(b"new")
    assert final.read_bytes() == b"new"


def test_the_temporary_lives_beside_the_target(tmp_path):
    # The precondition for the whole thing. os.replace is atomic only within a
    # filesystem; a temporary in /tmp would silently degrade to a copy across
    # devices and reopen the partial-file window this exists to close.
    final = tmp_path / "nested" / "cloud.laz"
    seen: list[Path] = []
    with atomic.staged(final) as tmp:
        seen.append(tmp)
        tmp.write_bytes(b"x")
    assert seen[0].parent == final.parent
    assert seen[0].name.startswith(".cloud-partial-")
    assert seen[0].suffix == ".laz"


def test_it_creates_the_destination_directory(tmp_path):
    final = tmp_path / "a" / "b" / "cloud.laz"
    with atomic.staged(final) as tmp:
        tmp.write_bytes(b"x")
    assert final.exists()


def test_write_text_round_trips(tmp_path):
    final = tmp_path / "index.geojson"
    atomic.write_text(final, '{"ok": true}')
    assert final.read_text() == '{"ok": true}'
    assert _names(tmp_path) == ["index.geojson"]


def test_write_text_leaves_the_old_content_on_failure(tmp_path):
    # The JSON caches fail loudly on a half-written file rather than silently,
    # but nothing cleans them up, so a tile would stay broken until someone
    # deleted the file by hand.
    final = tmp_path / "osm.json"
    final.write_text('{"elements": []}')
    with pytest.raises(TypeError):
        atomic.write_text(final, None)  # type: ignore[arg-type]
    assert final.read_text() == '{"elements": []}'
    assert _names(tmp_path) == ["osm.json"]
