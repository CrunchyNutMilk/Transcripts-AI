"""Tests for the staged-evidence extensions: scenes, session context,
time status, speaker modes and confidence bands."""
import pytest

from transcripts_ai.dnd_patterns import (
    assess_speaker_mode,
    assess_time_status,
)
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.resolver import BandAction, decide_band_action
from transcripts_ai.scenes import SceneType, scene_for_line, segment_scenes
from transcripts_ai.schemas import (
    EntityKind,
    EpistemicStatus,
    SchemaError,
    SpeakerMode,
    TimeStatus,
)
from transcripts_ai.session_context import PlayerMapping, SessionContext
from transcripts_ai.transcript import parse_transcript


class TestTimeStatus:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("We travel there and enter the building.", TimeStatus.HAPPENED),
            ("I cast Fireball at the cultist.", TimeStatus.HAPPENED),
            ("We should go to the Iron Mines tomorrow.", TimeStatus.PLANNED),
            ("I might attack him.", TimeStatus.PLANNED),
            ("I was going to attack, but I changed my mind.", TimeStatus.NEGATED),
            ("We didn't enter the cave.", TimeStatus.NEGATED),
            ("The spell would have worked if he failed.", TimeStatus.NEGATED),
            ("What if we just burn it down?", TimeStatus.HYPOTHETICAL),
            ("The sky is purple here.", TimeStatus.UNKNOWN),
        ],
    )
    def test_classification(self, text, expected):
        assert assess_time_status(text) is expected


class TestSpeakerMode:
    def entry(self, line):
        return parse_transcript(line).entries[0]

    def test_dm_narration(self):
        e = self.entry("DM: The door creaks open into darkness.\n")
        assert assess_speaker_mode(e) is SpeakerMode.DM_NARRATION

    def test_npc_dialogue_through_dm(self):
        e = self.entry('DM: The ferryman says, "I work for the king."\n')
        assert assess_speaker_mode(e) is SpeakerMode.NPC_DIALOGUE

    def test_mechanical_result(self):
        e = self.entry("DM: The ogre takes 12 slashing damage.\n")
        assert assess_speaker_mode(e) is SpeakerMode.MECHANICAL_RESULT

    def test_player_statement(self):
        e = self.entry("Diego: I sneak along the wall.\n")
        assert assess_speaker_mode(e) is SpeakerMode.PLAYER_STATEMENT

    def test_table_talk(self):
        e = self.entry("Diego: lol imagine if the mayor was the lich, jk.\n")
        assert assess_speaker_mode(e) is SpeakerMode.TABLE_TALK


class TestScenes:
    TEXT = """\
[00:00:01.000 - 00:00:03.000] - DM: You set out on the road and travel north for two days.
[00:00:04.000 - 00:00:06.000] - Diego: We keep heading north to the pass.
[00:00:07.000 - 00:00:09.000] - DM: You arrive at the gates. Welcome to Moon Crest.
[00:00:10.000 - 00:00:12.000] - DM: Roll initiative! Bandits attack.
[00:00:13.000 - 00:00:15.000] - Diego: I attack the first bandit.
[00:00:16.000 - 00:00:18.000] - DM: He drops. Combat is over.
[00:00:19.000 - 00:00:21.000] - Diego: can we pause? my mic is acting up
[00:00:22.000 - 00:00:24.000] - Marta: yeah discord is lagging for me too
[00:00:25.000 - 00:00:27.000] - Kim: same, my stream died, give me a minute
"""

    def scenes(self):
        return segment_scenes(parse_transcript(self.TEXT).entries)

    def test_combat_bounded_by_initiative_and_end(self):
        scenes = self.scenes()
        combat = [s for s in scenes if s.scene_type is SceneType.COMBAT]
        assert len(combat) == 1
        assert combat[0].line_start == 4
        assert combat[0].line_end == 6

    def test_ooc_run_detected(self):
        scenes = self.scenes()
        assert scenes[-1].scene_type is SceneType.OUT_OF_CHARACTER
        assert scenes[-1].line_start == 7

    def test_travel_before_combat(self):
        scenes = self.scenes()
        assert scenes[0].scene_type in (SceneType.TRAVEL, SceneType.LOCATION_ARRIVAL)

    def test_scene_lookup(self):
        scenes = self.scenes()
        assert scene_for_line(scenes, 5).scene_type is SceneType.COMBAT
        assert scene_for_line(scenes, 999) is None

    def test_short_ooc_blip_not_a_scene(self):
        text = (
            "DM: You travel along the road to the pass.\n"
            "Diego: sorry, mic issue.\n"
            "DM: The road leads to a bridge, you keep travelling.\n"
        )
        scenes = segment_scenes(parse_transcript(text).entries)
        assert all(s.scene_type is not SceneType.OUT_OF_CHARACTER for s in scenes)


class TestSessionContext:
    def context(self):
        return SessionContext(
            campaign_id="camp-a",
            session_id="2026-08-01",
            game_name="Test",
            session_date="2026-08-01",
            dm_labels=frozenset({"Alex"}),
            mappings=[
                PlayerMapping("111", "diego_discord", "Daragon"),
                PlayerMapping("222", "marta_discord", "Black Storm Cloud"),
            ],
            overrides=[PlayerMapping("111", "diego_discord", "Guest Wizard")],
        )

    def test_override_beats_default(self):
        assert self.context().character_for_speaker("diego_discord") == "Guest Wizard"

    def test_dm_resolution(self):
        ctx = self.context()
        assert ctx.character_for_speaker("Alex") == "DM"
        assert ctx.is_dm("alex")
        assert ctx.character_for_speaker("DM") == "DM"

    def test_unknown_speaker_none(self):
        assert self.context().character_for_speaker("randomer") is None

    def test_pc_names_mapping_authoritative(self):
        names = self.context().pc_names
        assert "Guest Wizard" in names and "Black Storm Cloud" in names

    def test_requires_dm(self):
        with pytest.raises(SchemaError):
            SessionContext(
                campaign_id="c", session_id="s", game_name="g", session_date="d",
                dm_labels=frozenset(), mappings=[],
            )

    def test_register_pcs_canon_and_scoped(self, tmp_path):
        memory = CampaignMemory(tmp_path / "m.sqlite")
        try:
            self.context().register_pcs(memory, actor="human:owner")
            pc = memory.find_entity("camp-a", "Daragon")
            assert pc is not None
            assert pc.kind is EntityKind.PC
            assert pc.status is EpistemicStatus.CONFIRMED_CANON
            assert pc.attributes["player_id"] == "111"
            assert memory.find_entity("camp-b", "Daragon") is None
        finally:
            memory.close()


class TestConfidenceBands:
    def test_bands(self):
        assert decide_band_action(0.97, strong_evidence=True) is BandAction.AUTO_LINK
        # High score without strong evidence never auto-links.
        assert decide_band_action(0.97, strong_evidence=False) is BandAction.SUGGEST
        assert decide_band_action(0.85) is BandAction.SUGGEST
        assert decide_band_action(0.70) is BandAction.SAVE_FOR_REVIEW
        assert decide_band_action(0.30) is BandAction.DROP

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ENGINE_BAND_SUGGEST", "0.90")
        assert decide_band_action(0.85) is BandAction.SAVE_FOR_REVIEW
