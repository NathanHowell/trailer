"""Guards against silently trusting a partial point cloud.

``build_aoi`` reuses ``points.laz`` whenever it exists, and a truncated LAZ is
not obviously broken -- the header still advertises the full point count. That
combination turns one interrupted download into every subsequent build of that
tile quietly using a fraction of its ground returns, which is a wrong model
rather than a failed build.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from trailer import build, rasters


def _leftovers(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


def test_a_failed_extract_leaves_nothing_behind(tmp_path, monkeypatch):
    # The whole point: after a failure there must be no file a retry could
    # mistake for a finished cloud -- not under the real name, and not under
    # the temporary one either.
    out = tmp_path / "tile" / "points.laz"

    def boom(stages, workdir, tag):
        # PDAL writes some of the file before dying, exactly as it did on
        # h_362193_s1186299.
        Path(stages[-1]["filename"]).write_bytes(b"LASF" + b"\0" * 500)
        raise RuntimeError("pdal [points] failed: connection reset")

    monkeypatch.setattr(rasters, "run_pdal", boom)

    with pytest.raises(RuntimeError, match="connection reset"):
        rasters.extract_points("ept://x", 37.8, -119.5, 1000, out)

    assert not out.exists(), "a partial cloud must not appear under the real name"
    assert _leftovers(out.parent) == [], (
        f"partial file left behind: {_leftovers(out.parent)}")


def test_a_successful_extract_moves_into_place(tmp_path, monkeypatch):
    out = tmp_path / "tile" / "points.laz"
    payload = b"LASF-complete"

    def ok(stages, workdir, tag):
        written = Path(stages[-1]["filename"])
        assert written != out, "PDAL should write to a temporary, not the target"
        written.write_bytes(payload)

    monkeypatch.setattr(rasters, "run_pdal", ok)

    rasters.extract_points("ept://x", 37.8, -119.5, 1000, out)

    assert out.read_bytes() == payload
    assert _leftovers(out.parent) == ["points.laz"], (
        f"temporary not cleaned up: {_leftovers(out.parent)}")


def test_the_target_is_replaced_only_on_success(tmp_path, monkeypatch):
    # A retry over an existing good file must not destroy it on the way to
    # failing. Truncating the target first and then failing would be worse than
    # not retrying at all.
    out = tmp_path / "tile" / "points.laz"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"previous-good-cloud")

    monkeypatch.setattr(rasters, "run_pdal",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))

    with pytest.raises(RuntimeError):
        rasters.extract_points("ept://x", 37.8, -119.5, 1000, out)

    assert out.read_bytes() == b"previous-good-cloud"


def test_check_complete_accepts_a_whole_cloud():
    build.check_complete(Path("points.laz"), 81_423_483, 81_423_483)


def test_check_complete_rejects_a_short_read():
    # The case the atomic write does not cover: PDAL exited 0 and the file is
    # still short. Both counts go in the message, because "truncated" without
    # them tells you nothing about how much was lost.
    with pytest.raises(ValueError) as e:
        build.check_complete(Path("points.laz"), 81_423_483, 12_000_000)
    assert "81,423,483" in str(e.value)
    assert "12,000,000" in str(e.value)
    assert "truncated" in str(e.value)


def test_check_complete_rejects_an_empty_cloud():
    # Zero decoded is the shape a completely failed read takes, and it must not
    # slip through as "well, nothing disagreed".
    with pytest.raises(ValueError):
        build.check_complete(Path("points.laz"), 1_000, 0)
