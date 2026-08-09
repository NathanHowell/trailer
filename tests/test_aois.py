"""The AOI registry's claims about itself.

`aois.py` is hand-annotated prose sitting next to numbers a reader will act on,
which is a combination that rots quietly. abandoned_south carried the note
"Paired abandoned:highway=path and active path" through a full training run and
into a README status paragraph; the tile holds one way and no active path at
all, and a held-out f1 of 0.000 got read as a model failure on the strength of
it. These tests pin the properties that would have caught that.
"""
from __future__ import annotations

import re

from trailer import aois


def test_advisory_names_the_reason_rather_than_flagging():
    # A boolean would let the number be dropped silently. The string is what
    # gets printed beside the score, so it has to say something.
    reason = aois.advisory("abandoned_south")
    assert reason, "abandoned_south's score is not evidence and must say so"
    assert len(reason) > 40, f"advisory is too terse to be useful: {reason!r}"

    # A tile whose score *is* evidence must not carry one, or the warning stops
    # meaning anything.
    assert aois.advisory("junction_pass") == ""
    assert aois.advisory("north_guard") == ""


def test_advisory_on_an_unknown_key_is_empty_not_an_error():
    # Callers hold a built directory's name, which may have left the registry.
    assert aois.advisory("h_381872_s1190153") == ""
    assert aois.advisory("") == ""


def test_only_scored_tiles_can_carry_an_advisory():
    # An advisory on a training tile is dead text: training tiles are never
    # scored, so nothing would ever print it.
    for a in aois.AOIS:
        if a.advisory:
            assert a.role in ("eval", "control"), (
                f"{a.key} is role {a.role}, so its advisory would never print")


def test_every_held_out_tile_is_described_in_measured_terms():
    # The eval and control tiles are the only honest estimate this project has.
    # Each one's note must at least say where it is and how rough, so the next
    # reader can judge what a number from it is worth -- abandoned_south's
    # one-line note was the tell that it had never been vetted like the others.
    for a in aois.AOIS:
        if a.role not in ("eval", "control"):
            continue
        assert re.search(r"\b\d{3,4} m\b", a.notes), (
            f"{a.key}'s note records no elevation: {a.notes!r}")
        assert len(a.notes) > 80, (
            f"{a.key}'s note is too thin for a held-out tile: {a.notes!r}")
