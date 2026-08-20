"""Craig multitrack support: filename parsing, discovery, merge, mapping."""
import zipfile

import pytest

from transcripts_ai.craig import (
    CraigError,
    Segment,
    apply_speaker_map,
    discover_tracks,
    extract_craig_zip,
    merge_segments,
    parse_track_filename,
    relabel_transcript_text,
    render_transcript,
    speaker_map_from_mapping,
)
from transcripts_ai.transcript import parse_transcript

MAPPING = {
    "campaign": "Heckuva Side Quest",
    "dm_labels": ["DM", "NorthTitan"],
    "players": [
        {"player_id": "1", "player_label": "crunchynutmilk", "character_name": "Jinx"},
        {"player_id": "2", "player_label": "BearDiego", "character_name": "Diego"},
    ],
    "overrides": [
        {"player_id": "2", "player_label": "BearDiego", "character_name": "Diego the Bear"},
    ],
}


class TestFilenameParsing:
    @pytest.mark.parametrize("name,expected", [
        ("1-crunchynutmilk.aac", (1, "crunchynutmilk")),
        ("10-NorthTitan.flac", (10, "NorthTitan")),
        ("2-old_user_1234.aac", (2, "old_user")),      # old discriminator
        ("3-name_12345.opus", (3, "name_12345")),      # 5 digits: real name
        ("4-guy.m4a", (4, "guy")),
    ])
    def test_tracks(self, name, expected):
        assert parse_track_filename(name) == expected

    @pytest.mark.parametrize("name", [
        "info.txt", "raw.dat", "ffmpeg-LICENSE.txt",
        "notes-1.txt", "craig.aac",          # no NN- prefix
    ])
    def test_non_tracks(self, name):
        assert parse_track_filename(name) is None


class TestDiscovery:
    def _make_zip(self, tmp_path):
        source = tmp_path / "craig-abc.aac.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("craig-abc/1-NorthTitan.aac", b"a")
            archive.writestr("craig-abc/2-crunchynutmilk.aac", b"b")
            archive.writestr("craig-abc/info.txt", b"metadata")
            archive.writestr("craig-abc/raw.dat", b"x")
        return source

    def test_zip_extraction_keeps_only_tracks(self, tmp_path):
        dest = extract_craig_zip(self._make_zip(tmp_path), tmp_path / "tracks")
        names = sorted(p.name for p in dest.iterdir())
        assert names == ["1-NorthTitan.aac", "2-crunchynutmilk.aac"]

    def test_discover_orders_by_track_number(self, tmp_path):
        dest = extract_craig_zip(self._make_zip(tmp_path), tmp_path / "tracks")
        tracks = discover_tracks(dest)
        assert [(t.number, t.speaker) for t in tracks] == \
            [(1, "NorthTitan"), (2, "crunchynutmilk")]

    def test_empty_folder_is_an_error(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(CraigError, match="no per-speaker tracks"):
            discover_tracks(empty)


class TestMergeAndMapping:
    PER_TRACK = {
        "NorthTitan": [(0.0, 4.0, "You enter the spire."),
                       (10.0, 12.0, "Roll initiative.")],
        "crunchynutmilk": [(5.0, 7.0, "Jinx draws her blade."),
                           (10.5, 11.0, "Yes!"), (20.0, 21.0, "  ")],
    }

    def test_merge_interleaves_by_time_and_drops_blank(self):
        merged = merge_segments(self.PER_TRACK)
        assert [s.speaker for s in merged] == \
            ["NorthTitan", "crunchynutmilk", "NorthTitan", "crunchynutmilk"]
        assert merged[-1].text == "Yes!"          # blank 20.0s segment dropped

    def test_mapping_renames_dm_and_characters(self):
        speaker_map = speaker_map_from_mapping(MAPPING)
        merged, unmapped = apply_speaker_map(merge_segments(self.PER_TRACK),
                                             speaker_map)
        assert {s.speaker for s in merged} == {"DM", "Jinx"}
        assert unmapped == []

    def test_override_wins_and_unmapped_reported(self):
        speaker_map = speaker_map_from_mapping(MAPPING)
        assert speaker_map["beardiego"] == "Diego the Bear"
        merged, unmapped = apply_speaker_map(
            [Segment(0.0, 1.0, "somebody_new", "hi")], speaker_map)
        assert merged[0].speaker == "somebody_new"
        assert unmapped == ["somebody_new"]

    def test_rendered_transcript_is_engine_parseable(self):
        speaker_map = speaker_map_from_mapping(MAPPING)
        merged, _ = apply_speaker_map(merge_segments(self.PER_TRACK),
                                      speaker_map)
        text = render_transcript(merged)
        parsed = parse_transcript(text, source_path="craig.md")
        assert len(parsed.entries) == 4
        assert parsed.entries[0].speaker == "DM"
        assert parsed.entries[0].is_dm
        assert parsed.entries[1].speaker == "Jinx"
        assert parsed.entries[0].start_ms == 0
        assert parsed.entries[1].start_ms == 5000
        assert not parsed.unparsed_lines

    def test_render_empty_is_empty(self):
        assert render_transcript([]) == ""


class TestRelabelText:
    SPEAKER_MAP = speaker_map_from_mapping(MAPPING)

    def test_bare_lines_relabelled_text_untouched(self):
        text = ("crunchynutmilk: I got the prayer beads.\n\n"
                "NorthTitan: Roll me 6d20.\n")
        out, unmapped = relabel_transcript_text(text, self.SPEAKER_MAP)
        assert out == ("Jinx: I got the prayer beads.\n\n"
                       "DM: Roll me 6d20.\n")
        assert unmapped == []

    def test_timestamped_lines_keep_timestamps(self):
        text = "[00:01:02.000 - 00:01:04.500] - BearDiego: hi there\n"
        out, _ = relabel_transcript_text(text, self.SPEAKER_MAP)
        assert out == "[00:01:02.000 - 00:01:04.500] - Diego the Bear: hi there\n"

    def test_unknown_speaker_kept_and_reported(self):
        text = "somebody_new: hello\ncrunchynutmilk: hi\n"
        out, unmapped = relabel_transcript_text(text, self.SPEAKER_MAP)
        assert out.startswith("somebody_new: hello\n")
        assert unmapped == ["somebody_new"]

    def test_line_numbers_survive(self):
        text = ("NorthTitan: one\n\nnot a speaker line at all...!?\n\n"
                "crunchynutmilk: two\n")
        out, _ = relabel_transcript_text(text, self.SPEAKER_MAP)
        assert len(out.splitlines()) == len(text.splitlines())
        before = parse_transcript(text, source_path="a")
        after = parse_transcript(out, source_path="b")
        assert [e.line_number for e in after.entries] == \
            [e.line_number for e in before.entries]

    def test_relabelled_output_parses_with_new_speakers(self):
        text = "vulkare is not a line\nNorthTitan: the child rolls a d100\n"
        out, _ = relabel_transcript_text(text, self.SPEAKER_MAP)
        parsed = parse_transcript(out, source_path="x")
        assert parsed.entries[0].speaker == "DM"
        assert parsed.entries[0].is_dm
