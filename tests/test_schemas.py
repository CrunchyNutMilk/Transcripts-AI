import pytest

from transcripts_ai.schemas import (
    AI_ASSIGNABLE_STATUSES,
    AliasRecord,
    ChangeType,
    ContextPackage,
    ContextSource,
    EntityKind,
    EntityRecord,
    EpistemicStatus,
    Fact,
    FactCategory,
    Provenance,
    ReviewItem,
    SchemaError,
    text_sha256,
)


def make_provenance(**overrides):
    base = dict(
        campaign_id="camp-a",
        session_id="2026-08-01",
        source_path="Transcript Mapped/2026-08-01/session.md",
        source_hash=text_sha256("hello"),
        line_start=3,
        line_end=5,
        speaker="DM",
        quote="the dragon roared",
    )
    base.update(overrides)
    return Provenance(**base)


class TestEpistemicModel:
    def test_confirmed_canon_is_strongest(self):
        assert EpistemicStatus.CONFIRMED_CANON.stronger_than(EpistemicStatus.TABLE_TALK)
        assert not EpistemicStatus.TABLE_TALK.stronger_than(EpistemicStatus.DM_HINT)

    def test_ai_cannot_assign_canon(self):
        assert EpistemicStatus.CONFIRMED_CANON not in AI_ASSIGNABLE_STATUSES
        assert EpistemicStatus.STRONGLY_SUPPORTED in AI_ASSIGNABLE_STATUSES


class TestProvenance:
    def test_valid(self):
        make_provenance().validate()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"campaign_id": ""},
            {"session_id": ""},
            {"source_path": ""},
            {"source_hash": "nothex"},
            {"line_start": 0},
            {"line_start": 9, "line_end": 3},
        ],
    )
    def test_invalid(self, overrides):
        with pytest.raises(SchemaError):
            make_provenance(**overrides).validate()


class TestFact:
    def test_round_trip(self):
        fact = Fact(
            statement="The party found the Moon Sickle",
            category=FactCategory.LOOT,
            change_type=ChangeType.DISCOVERED,
            entities=["Moon Sickle"],
            status=EpistemicStatus.STRONGLY_SUPPORTED,
            confidence=0.9,
            provenance=make_provenance(),
        )
        again = Fact.from_json(fact.to_json())
        assert again.fact_id == fact.fact_id
        assert again.category is FactCategory.LOOT
        assert again.provenance == fact.provenance

    def test_stable_id(self):
        a = Fact(
            statement="X",
            category=FactCategory.OTHER,
            change_type=ChangeType.MENTIONED,
            entities=[],
            status=EpistemicStatus.TABLE_TALK,
            confidence=0.5,
            provenance=make_provenance(),
        )
        b = Fact(
            statement="X",
            category=FactCategory.QUEST,  # id ignores category by design
            change_type=ChangeType.MENTIONED,
            entities=[],
            status=EpistemicStatus.TABLE_TALK,
            confidence=0.5,
            provenance=make_provenance(),
        )
        assert a.fact_id == b.fact_id

    def test_confidence_bounds(self):
        with pytest.raises(SchemaError):
            Fact(
                statement="X",
                category=FactCategory.OTHER,
                change_type=ChangeType.MENTIONED,
                entities=[],
                status=EpistemicStatus.TABLE_TALK,
                confidence=1.5,
                provenance=make_provenance(),
            )


class TestEntitiesAndAliases:
    def test_entity_id_stability(self):
        a = EntityRecord(name="Daragon", kind=EntityKind.NPC, campaign_id="camp-a")
        b = EntityRecord(name="daragon", kind=EntityKind.NPC, campaign_id="camp-a")
        assert a.entity_id == b.entity_id
        c = EntityRecord(name="Daragon", kind=EntityKind.NPC, campaign_id="camp-b")
        assert c.entity_id != a.entity_id

    def test_alias_cannot_self_reference(self):
        with pytest.raises(SchemaError):
            AliasRecord(
                campaign_id="camp-a",
                observed="Cloud",
                canonical="cloud",
                entity_id=None,
                approved_by="human:1",
            )

    def test_alias_human_flag(self):
        alias = AliasRecord(
            campaign_id="camp-a",
            observed="Cloud",
            canonical="Black Storm Cloud",
            entity_id="e1",
            approved_by="human:426983004020277248",
        )
        assert alias.human_approved


class TestContextPackage:
    def make_package(self):
        text = "The DM said: the dragon roared over Silverspire."
        return ContextPackage(
            campaign_id="camp-a",
            task="extractor",
            sources=[
                ContextSource(
                    source_id="S1",
                    kind="transcript",
                    path="t.md",
                    content_hash=text_sha256(text),
                    chars=len(text),
                )
            ],
            sections={"S1": text},
            budget_chars=1000,
        )

    def test_quote_containment(self):
        package = self.make_package()
        assert package.contains_quote("the dragon roared")
        assert package.contains_quote("THE  DRAGON\nROARED")  # ws/case tolerant
        assert not package.contains_quote("the dragon whispered")
        assert not package.contains_quote("")

    def test_budget_enforced(self):
        text = "x" * 100
        with pytest.raises(SchemaError):
            ContextPackage(
                campaign_id="c",
                task="extractor",
                sources=[
                    ContextSource("S1", "transcript", "t.md", text_sha256(text), 100)
                ],
                sections={"S1": text},
                budget_chars=10,
            )

    def test_manifest_hash_stable(self):
        assert self.make_package().manifest_hash == self.make_package().manifest_hash


class TestReviewItem:
    def test_ids_stable(self):
        a = ReviewItem(
            campaign_id="c", session_id="s", item_type="spelling",
            subject="Aragon", reason="close match to Daragon", evidence=["..."],
        )
        b = ReviewItem(
            campaign_id="c", session_id="s", item_type="spelling",
            subject="Aragon", reason="different reason", evidence=[],
        )
        assert a.item_id == b.item_id
