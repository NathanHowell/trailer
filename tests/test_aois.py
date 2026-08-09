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


#: Metres of clearance a held-out tile needs from any training tile. It has to
#: exceed the model's context window, not merely the tile edge, or a training
#: crop sees pixels a held-out crop also sees.
BUFFER_M = 600.0


def _gap_m(a, b):
    """Clearance between two axis-aligned square AOIs, in metres.

    Separated if *either* axis clears, so this is the max of the two gaps.
    """
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
    ax, ay = tf.transform(a.lon, a.lat)
    bx, by = tf.transform(b.lon, b.lat)
    half = a.size_m / 2 + b.size_m / 2
    return max(abs(ax - bx) - half, abs(ay - by) - half)


def test_no_training_tile_touches_a_held_out_tile():
    """The harvest grid steps a kilometre, so neighbours are one cell apart.

    Promoting a harvested tile to eval without checking this leaks context
    straight across the split. The first attempt at the held-out spread picked
    three tiles that were *adjacent* to training cells -- gap about 0 m, tiles
    literally touching -- and nothing in the pipeline would have said so.
    """
    registry = aois.all_aois()
    held = [a for a in registry if a.role in ("eval", "control")]
    training = [a for a in registry if a.role in ("train", "harvest")]
    assert held and training, "registry looks empty"

    leaks = [(h.key, t.key, _gap_m(h, t))
             for h in held for t in training if _gap_m(h, t) < BUFFER_M]
    assert not leaks, "held-out tiles with training neighbours inside the " \
        f"{BUFFER_M:.0f} m buffer: " + \
        ", ".join(f"{h} <- {t} ({g:.0f} m)" for h, t, g in sorted(leaks))


def test_every_visibility_class_is_held_out_on_several_aois():
    # One tile per class is not an estimate: per-tile lifecycle F1 ranges from
    # 0.00 to 0.94 across this corpus, so a single draw says nothing.
    from trailer import osm

    counts = {c: 0 for c in osm.VISIBILITY_CLASSES}
    for a in aois.AOIS:
        if a.role != "eval" or a.advisory:
            continue
        for c in counts:
            # The measured form -- "faint 2.06 km" -- not a prose mention of
            # the word, which "the alpine end of the faint range" would satisfy
            # while recording nothing. A class only counts where the note
            # carries at least a kilometre of it.
            m = re.search(rf"\b{c} (\d+\.\d+) km", a.notes)
            if m and float(m.group(1)) >= 1.0:
                counts[c] += 1
    for c, n in counts.items():
        assert n >= 3, f"only {n} eval AOIs carry {c}; the spread needs 3+"


def test_a_hand_annotated_tile_beats_a_stale_harvest_record(tmp_path):
    """`data/harvest.json` is gitignored, so promotion lives only in this file.

    A workspace whose registry still lists a promoted tile must not load it
    twice with two roles -- the harvest copy would win by insertion order and
    the tile would go quietly back to training, dissolving the eval split on
    whichever machine happened to have a stale registry.
    """
    import json

    promoted = next(a for a in aois.AOIS if a.role == "eval")
    stale = tmp_path / "harvest.json"
    stale.write_text(json.dumps([
        {"key": promoted.key, "name": "stale", "lat": promoted.lat,
         "lon": promoted.lon, "role": "harvest", "size_m": 1000,
         "notes": "", "flags": []},
        {"key": "h_fresh_000000", "name": "fresh", "lat": 37.0, "lon": -119.0,
         "role": "harvest", "size_m": 1000, "notes": "", "flags": []},
    ]))
    loaded = aois.load_harvest(stale)
    assert [a.key for a in loaded] == ["h_fresh_000000"]
    assert promoted.key not in {a.key for a in loaded}
