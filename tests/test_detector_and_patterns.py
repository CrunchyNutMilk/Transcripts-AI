from transcripts_ai.detector import detect_names
from transcripts_ai.dnd_patterns import (
    GameEvent,
    assess_entry,
    detect_events,
    initiative_order,
)
from transcripts_ai.schemas import EpistemicStatus
from transcripts_ai.transcript import parse_transcript

SESSION = """\
[00:00:01.000 - 00:00:04.000] - DM: Welcome to Silverspire, the last free city.
[00:00:05.000 - 00:00:08.000] - Diego: Ah, okay. I've heard about this place.
[00:00:09.000 - 00:00:14.000] - DM: A hooded figure approaches. My name is Daragon, he says.
[00:00:15.000 - 00:00:18.000] - Diego: I think Daragon is probably working for the cult, lol just kidding.
[00:00:19.000 - 00:00:24.000] - DM: Roll initiative! Two cultists leap from the shadows.
[00:00:25.000 - 00:00:27.000] - Diego: Diego rolled a 17.
[00:00:28.000 - 00:00:30.000] - DM: The cultist got a 12.
[00:00:31.000 - 00:00:36.000] - Diego: I cast Fire Bolt at the first cultist.
[00:00:37.000 - 00:00:41.000] - DM: It takes 8 fire damage and drops dead. Combat is over.
[00:00:42.000 - 00:00:47.000] - DM: You find 50 gp and a strange sickle. You notice something seems off about it.
[00:00:48.000 - 00:00:52.000] - Diego: We should take a long rest before heading to Silverspire keep.
"""


def entries():
    return parse_transcript(SESSION).entries


class TestNameDetection:
    def test_finds_introduced_npc_and_location(self):
        names = {d.folded: d for d in detect_names(entries())}
        assert "daragon" in names
        assert any(r in ("self_introduction", "introduction") for r in names["daragon"].reasons)
        assert "silverspire" in names

    def test_fillers_never_detected(self):
        names = {d.folded for d in detect_names(entries())}
        for filler in ("ah", "okay", "i've", "i", "the"):
            assert filler not in names

    def test_mechanics_words_never_detected(self):
        names = {d.folded for d in detect_names(entries())}
        assert "fire" not in names
        assert "roll" not in names
        assert "initiative" not in names

    def test_known_names_tracked_lowercase(self):
        text = "Diego: we went back to silverspire yesterday.\n"
        found = detect_names(
            parse_transcript(text).entries, known_names=frozenset({"silverspire"})
        )
        assert any(d.folded == "silverspire" for d in found)

    def test_repeated_mentions_counted(self):
        names = {d.folded: d for d in detect_names(entries())}
        assert names["daragon"].mentions >= 2


class TestEventDetection:
    def test_core_events_found(self):
        found = {e.event for e in detect_events(entries())}
        assert GameEvent.INITIATIVE in found
        assert GameEvent.SPELL_CAST in found
        assert GameEvent.DAMAGE in found
        assert GameEvent.DEATH in found
        assert GameEvent.COMBAT_END in found
        assert GameEvent.LOOT in found
        assert GameEvent.LONG_REST in found
        assert GameEvent.NPC_INTRODUCTION in found

    def test_initiative_order_extraction(self):
        order = initiative_order(entries())
        assert ("Diego", 17) in order
        by_name = dict(order)
        assert by_name.get("cultist") == 12 or by_name.get("The cultist") == 12

    def test_no_initiative_no_order(self):
        text = "DM: You walk through a quiet forest.\n"
        assert initiative_order(parse_transcript(text).entries) == []


class TestEpistemicAssessment:
    def test_dm_statement_strong(self):
        e = entries()[0]
        result = assess_entry(e)
        assert result.status is EpistemicStatus.STRONGLY_SUPPORTED

    def test_joke_is_table_talk(self):
        joke = entries()[3]  # "probably working for the cult, lol just kidding"
        result = assess_entry(joke)
        assert result.status is EpistemicStatus.TABLE_TALK
        assert "joke" in result.cues

    def test_dm_hedge_is_hint(self):
        hint = entries()[9]  # "You notice something seems off"
        assert hint.is_dm
        result = assess_entry(hint)
        assert result.status is EpistemicStatus.DM_HINT

    def test_player_plan_downgraded(self):
        plan = entries()[10]  # "We should take a long rest"
        result = assess_entry(plan)
        assert result.status is EpistemicStatus.UNCONFIRMED_THEORY
        assert "planning" in result.cues

    def test_player_speculation(self):
        e = parse_transcript("Diego: I think the mayor might be a doppelganger.\n").entries[0]
        result = assess_entry(e)
        assert result.status is EpistemicStatus.PLAYER_ASSUMPTION

    def test_ooc_chatter(self):
        e = parse_transcript("Diego: can we order pizza before next week's session?\n").entries[0]
        assert assess_entry(e).status is EpistemicStatus.TABLE_TALK

    def test_rules_talk(self):
        e = parse_transcript("Diego: rules-wise, does hex stack with hunter's mark?\n").entries[0]
        result = assess_entry(e)
        assert result.status is EpistemicStatus.TABLE_TALK
        assert "rules_discussion" in result.cues
