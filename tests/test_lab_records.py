"""Training-lab Layer 1: versioned records, ledger export, session splits."""
import json

import pytest

from transcripts_ai.lab.records import (
    RECORD_SCHEMA_VERSION,
    RecordError,
    TrainingRecord,
    export_from_feedback,
    read_records,
    split_records,
    split_sessions,
    tally,
    write_records,
)
from transcripts_ai.memory import CampaignMemory

CAMPAIGN = "heckuva"


@pytest.fixture
def memory(tmp_path):
    mem = CampaignMemory(tmp_path / "memory.sqlite")
    yield mem
    mem.close()


def make_record(**overrides):
    base = dict(
        campaign_id=CAMPAIGN,
        session_id="2026-08-09",
        item_type="entity",
        subject="Gomra",
        proposal="Gomra->Ghomra",
        decision="alias",
        choice="Ghomra",
        accepted=True,
        actor="human:harry",
        created_at=1723200000.0,
    )
    base.update(overrides)
    return TrainingRecord(**base)


class TestRecordSchema:
    def test_round_trip(self):
        record = make_record(extras={"score": 0.91})
        again = TrainingRecord.from_json(record.to_json())
        assert again == record

    def test_unknown_version_fails_closed(self):
        data = json.loads(make_record().to_json())
        data["record_version"] = RECORD_SCHEMA_VERSION + 1
        with pytest.raises(RecordError, match="unsupported"):
            TrainingRecord.from_json(json.dumps(data))

    def test_missing_version_fails_closed(self):
        data = json.loads(make_record().to_json())
        del data["record_version"]
        with pytest.raises(RecordError, match="unsupported"):
            TrainingRecord.from_json(json.dumps(data))

    def test_not_json_fails_closed(self):
        with pytest.raises(RecordError, match="not valid JSON"):
            TrainingRecord.from_json("{broken")

    @pytest.mark.parametrize("field", ["campaign_id", "session_id", "subject", "decision"])
    def test_required_fields(self, field):
        with pytest.raises(RecordError):
            make_record(**{field: ""}).validate()


class TestRecordFiles:
    def test_write_read_round_trip(self, tmp_path):
        path = tmp_path / "records.jsonl"
        records = [make_record(), make_record(subject="jink", decision="reject",
                                              accepted=False, choice=None)]
        assert write_records(path, records) == 2
        assert read_records(path) == records

    def test_read_reports_bad_line_number(self, tmp_path):
        path = tmp_path / "records.jsonl"
        path.write_text(make_record().to_json() + "\n\nnot json\n", encoding="utf-8")
        with pytest.raises(RecordError, match="line 3"):
            read_records(path)


class TestExportFromFeedback:
    def test_export_includes_rejections_and_parses_arrows(self, memory):
        memory.record_feedback(CAMPAIGN, "s1", kind="correction_accepted",
                               subject="Gomra->Ghomra", accepted=True, actor="human:harry")
        memory.record_feedback(CAMPAIGN, "s1", kind="correction_rejected",
                               subject="jink->link", accepted=False, actor="human:harry")
        memory.record_feedback(CAMPAIGN, "s2", kind="entity_misclassified",
                               subject="gonf", accepted=True, actor="human:harry")
        memory.record_feedback("other-campaign", "s1", kind="alias_added",
                               subject="X->Y", accepted=True, actor="human:harry")

        records = export_from_feedback(memory, CAMPAIGN)
        assert len(records) == 3  # campaign isolation holds
        accepted = records[0]
        assert (accepted.item_type, accepted.decision) == ("spelling", "correct")
        assert (accepted.subject, accepted.choice) == ("Gomra", "Ghomra")
        assert accepted.proposal == "Gomra->Ghomra"
        rejected = records[1]
        assert rejected.accepted is False and rejected.decision == "reject"
        assert records[2].item_type == "entity" and records[2].choice is None
        for record in records:
            record.validate()

    def test_unknown_kind_kept_as_other(self, memory):
        # A future ledger kind this reader doesn't know must survive export.
        memory._conn.execute(
            "INSERT INTO feedback (campaign_id, session_id, kind, subject, subject_folded,"
            " accepted, detail_json, actor, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (CAMPAIGN, "s1", "kind_from_the_future", "Thing", "thing", 1, "{}",
             "human:harry", 1723200000.0),
        )
        records = export_from_feedback(memory, CAMPAIGN)
        assert len(records) == 1
        assert records[0].item_type == "other"
        assert records[0].decision == "kind_from_the_future"

    def test_tally(self):
        records = [make_record(), make_record(session_id="s2", accepted=False)]
        counts = tally(records)
        assert counts["by_item_type"]["entity"] == 2
        assert counts["by_session"]["s2"] == 1
        assert counts["by_accepted"] == {"accepted": 1, "rejected": 1}


class TestSessionSplit:
    SESSIONS = [f"2026-08-{d:02d}" for d in range(1, 11)]

    def test_deterministic(self):
        first = split_sessions(self.SESSIONS, eval_fraction=0.2)
        second = split_sessions(list(reversed(self.SESSIONS)), eval_fraction=0.2)
        assert first == second

    def test_partition_is_complete_and_disjoint(self):
        split = split_sessions(self.SESSIONS, eval_fraction=0.3)
        assert not set(split.train_sessions) & set(split.eval_sessions)
        assert sorted(split.train_sessions + split.eval_sessions) == self.SESSIONS

    def test_frozen_always_eval(self):
        frozen = [self.SESSIONS[0], self.SESSIONS[5]]
        split = split_sessions(self.SESSIONS, frozen=frozen, eval_fraction=0.2)
        assert set(frozen) <= set(split.eval_sessions)
        assert not set(frozen) & set(split.train_sessions)
        # frozen already covers the 20% quota: nothing else joins eval
        assert set(split.eval_sessions) == set(frozen)

    def test_unknown_frozen_rejected(self):
        with pytest.raises(RecordError, match="frozen sessions not in data"):
            split_sessions(self.SESSIONS, frozen=["never-played"])

    def test_bad_fraction_rejected(self):
        with pytest.raises(RecordError):
            split_sessions(self.SESSIONS, eval_fraction=1.0)

    def test_split_records_never_leaks_items(self):
        records = [make_record(session_id=s, subject=f"subj-{i}")
                   for i, s in enumerate(self.SESSIONS * 3)]
        split = split_sessions(self.SESSIONS, eval_fraction=0.3)
        train, evaluation = split_records(records, split)
        assert len(train) + len(evaluation) == len(records)
        assert {r.session_id for r in train} <= set(split.train_sessions)
        assert {r.session_id for r in evaluation} <= set(split.eval_sessions)
        assert not {r.session_id for r in train} & {r.session_id for r in evaluation}
