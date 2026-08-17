"""Scene segmentation and classification.

Entity classification is more accurate when the engine knows what kind of
scene a line belongs to ("Fireball" in combat is a spell cast; in a rules
discussion it is table talk). Scenes are contiguous entry ranges labelled by
majority signal, with combat boundaries and extended out-of-character runs
detected explicitly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .dnd_patterns import GameEvent, _EVENT_PATTERNS, _OOC_CUES, _RULES_TALK_CUES
from .transcript import TranscriptEntry


class SceneType(str, Enum):
    ROLEPLAY = "roleplay"
    TRAVEL = "travel"
    LOCATION_ARRIVAL = "location_arrival"
    NPC_CONVERSATION = "npc_conversation"
    COMBAT = "combat"
    LOOT = "loot"
    PLANNING = "planning"
    REST = "rest"
    RULES_DISCUSSION = "rules_discussion"
    OUT_OF_CHARACTER = "out_of_character"


_SCENE_HINTS: tuple[tuple[SceneType, re.Pattern[str]], ...] = (
    (SceneType.TRAVEL, re.compile(r"\b(?:travel|journey|set(?:s|ting)? (?:off|out)|on the road|we ride|we walk|head(?:s|ing)? (?:to|north|south|east|west))\b", re.I)),
    (SceneType.LOCATION_ARRIVAL, re.compile(r"\b(?:you (?:arrive|enter|reach)|welcome to|the (?:city|town|village|building|room) of)\b", re.I)),
    (SceneType.NPC_CONVERSATION, re.compile(r"\b(?:says|asks|replies|tells you|my name is|greets you)\b", re.I)),
    (SceneType.LOOT, re.compile(r"\b(?:loot|treasure|you find|gold pieces|\d+ ?gp|divide|split the|inventory)\b", re.I)),
    (SceneType.PLANNING, re.compile(r"\b(?:the plan|we should|let'?s|we could|before we go|our next move|strategy)\b", re.I)),
    (SceneType.REST, re.compile(r"\b(?:short rest|long rest|make camp|set(?:s|ting)? up camp|watch order|go to sleep)\b", re.I)),
)

_COMBAT_START = _EVENT_PATTERNS[0][1]  # roll initiative
_COMBAT_END = next(p for e, p in _EVENT_PATTERNS if e is GameEvent.COMBAT_END)


@dataclass
class Scene:
    scene_type: SceneType
    line_start: int
    line_end: int
    entries: list[TranscriptEntry] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for entry in self.entries:
            seen.setdefault(entry.speaker, None)
        return list(seen)


def _entry_signals(entry: TranscriptEntry) -> set[SceneType]:
    signals: set[SceneType] = set()
    text = entry.text
    if _OOC_CUES.search(text):
        signals.add(SceneType.OUT_OF_CHARACTER)
    if _RULES_TALK_CUES.search(text):
        signals.add(SceneType.RULES_DISCUSSION)
    for scene_type, pattern in _SCENE_HINTS:
        if pattern.search(text):
            signals.add(scene_type)
    return signals


def segment_scenes(
    entries: list[TranscriptEntry], *, window: int = 6, ooc_run_length: int = 3
) -> list[Scene]:
    """Label contiguous scene ranges.

    Combat is stateful: "roll initiative" opens a combat scene, an explicit
    combat-end phrase closes it. Outside combat, each entry gets the majority
    signal over a sliding window (default roleplay), and ``ooc_run_length``
    consecutive out-of-character entries become an OUT_OF_CHARACTER scene.
    """
    if not entries:
        return []

    labels: list[SceneType] = []
    in_combat = False
    signal_cache = [_entry_signals(e) for e in entries]
    for index, entry in enumerate(entries):
        if not in_combat and _COMBAT_START.search(entry.text):
            in_combat = True
        if in_combat:
            labels.append(SceneType.COMBAT)
            if _COMBAT_END.search(entry.text):
                in_combat = False
            continue
        # Majority vote over the local window, preferring specific signals.
        lo = max(0, index - window // 2)
        hi = min(len(entries), index + window // 2 + 1)
        counts: dict[SceneType, int] = {}
        for signals in signal_cache[lo:hi]:
            for signal in signals:
                counts[signal] = counts.get(signal, 0) + 1
        own = signal_cache[index]
        if own:
            # Own strong signal wins ties over neighbourhood noise.
            for signal in own:
                counts[signal] = counts.get(signal, 0) + 1
        label = max(counts, key=lambda s: counts[s]) if counts else SceneType.ROLEPLAY
        labels.append(label)

    # Enforce the OOC run rule: only sustained runs count as OOC scenes.
    run_start = 0
    for index in range(1, len(labels) + 1):
        if index < len(labels) and labels[index] == labels[index - 1]:
            continue
        if labels[run_start] is SceneType.OUT_OF_CHARACTER and index - run_start < ooc_run_length:
            replacement = (
                labels[run_start - 1] if run_start > 0 else
                (labels[index] if index < len(labels) else SceneType.ROLEPLAY)
            )
            for j in range(run_start, index):
                labels[j] = replacement
        run_start = index

    # Merge consecutive same-label entries into Scene ranges.
    scenes: list[Scene] = []
    for entry, label in zip(entries, labels):
        if scenes and scenes[-1].scene_type is label:
            scenes[-1].entries.append(entry)
            scenes[-1].line_end = entry.line_number
        else:
            scenes.append(
                Scene(scene_type=label, line_start=entry.line_number,
                      line_end=entry.line_number, entries=[entry])
            )
    return scenes


def scene_for_line(scenes: list[Scene], line_number: int) -> Scene | None:
    for scene in scenes:
        if scene.line_start <= line_number <= scene.line_end:
            return scene
    return None
