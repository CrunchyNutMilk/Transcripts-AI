import pytest

from transcripts_ai.schemas import SchemaError
from transcripts_ai.transcript import (
    chunk_transcript,
    parse_transcript,
    validate_chunks,
)

SAMPLE = """\
[00:00:01.000 - 00:00:04.500] - DM: You enter the ruined keep of Silverspire.
[00:00:05.000 - 00:00:07.250] - Diego: I check the door for traps.
[00:00:08.000 - 00:00:09.000] - DM: Roll investigation.
[00:00:10.000 - 00:00:12.000] - Diego: That's a 17.

[00:00:13.000 - 00:00:20.000] - DM: You spot a tripwire connected to a bell.
Narrator note without speaker prefix
[00:06:30.000 - 00:06:32.000] - Diego: After the break, I cut the wire.
"""


class TestParser:
    def test_parses_timestamped_lines(self):
        parsed = parse_transcript(SAMPLE, source_path="t.md")
        assert len(parsed.entries) == 6
        first = parsed.entries[0]
        assert first.speaker == "DM"
        assert first.start_ms == 1000
        assert first.end_ms == 4500
        assert first.line_number == 1
        assert first.is_dm

    def test_unparsed_lines_preserved(self):
        parsed = parse_transcript(SAMPLE)
        assert parsed.unparsed_lines == [(7, "Narrator note without speaker prefix")]

    def test_bare_speaker_lines(self):
        parsed = parse_transcript("Diego: hello there\nDM: hi\n")
        assert [e.speaker for e in parsed.entries] == ["Diego", "DM"]
        assert parsed.entries[0].start_ms is None

    def test_continuation_lines_append(self):
        text = "[00:00:01.000 - 00:00:02.000] - DM: The inscription reads\n    'beware the depths'\n"
        parsed = parse_transcript(text)
        assert len(parsed.entries) == 1
        assert "beware the depths" in parsed.entries[0].text

    def test_speakers_in_first_seen_order(self):
        parsed = parse_transcript(SAMPLE)
        assert parsed.speakers == ["DM", "Diego"]

    def test_source_never_modified(self):
        parsed = parse_transcript(SAMPLE)
        from transcripts_ai.schemas import text_sha256
        assert parsed.source_hash == text_sha256(SAMPLE)


class TestChunker:
    def test_exact_reconstruction(self):
        parsed = parse_transcript(SAMPLE)
        chunks = chunk_transcript(parsed, target_chars=100, max_chars=200, min_chars=10)
        validate_chunks(parsed, chunks)
        assert [c.chunk_number for c in chunks] == list(range(1, len(chunks) + 1))

    def test_large_gap_forces_boundary(self):
        parsed = parse_transcript(SAMPLE)
        chunks = chunk_transcript(parsed, target_chars=10_000, max_chars=20_000, min_chars=1)
        # The 6-minute silence before the final line must split chunks even
        # though the char budget would fit everything in one.
        assert len(chunks) == 2
        assert chunks[1].entries[0].text.startswith("After the break")

    def test_hard_max_chars(self):
        parsed = parse_transcript(SAMPLE)
        chunks = chunk_transcript(parsed, target_chars=50, max_chars=90, min_chars=1)
        assert all(c.char_count <= 90 or len(c.entries) == 1 for c in chunks)
        validate_chunks(parsed, chunks)

    def test_max_entries(self):
        lines = "\n".join(f"Diego: line {i}" for i in range(100))
        parsed = parse_transcript(lines)
        chunks = chunk_transcript(
            parsed, target_chars=10**6, max_chars=10**6, min_chars=1, max_entries=10
        )
        assert all(len(c.entries) <= 10 for c in chunks)
        validate_chunks(parsed, chunks)

    def test_runt_merge(self):
        parsed = parse_transcript("DM: aaaaaa\nDg: b\n")
        chunks = chunk_transcript(parsed, target_chars=8, max_chars=100, min_chars=6)
        assert len(chunks) == 1

    def test_invalid_config_rejected(self):
        parsed = parse_transcript(SAMPLE)
        with pytest.raises(SchemaError):
            chunk_transcript(parsed, target_chars=10, max_chars=5, min_chars=1)

    def test_chunk_metadata(self):
        parsed = parse_transcript(SAMPLE)
        chunks = chunk_transcript(parsed)
        chunk = chunks[0]
        assert chunk.line_start == 1
        assert "DM" in chunk.speakers and "Diego" in chunk.speakers
        assert chunk.start_ms == 1000
