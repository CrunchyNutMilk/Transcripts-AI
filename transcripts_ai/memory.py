"""Campaign-scoped persistent memory.

One SQLite database holds every campaign, but every table is keyed by
campaign_id and every query requires one — there is no cross-campaign read
path. All learned information carries provenance and an approval source, is
append-only where it matters (audit log, contradictions, feedback), and every
learned row can be inspected and removed by id (`forget`).

Design rules enforced here rather than by convention:
- AI-assigned epistemic status can never be CONFIRMED_CANON (schema rule);
  ``promote_fact`` is the only path to canon and demands a human actor.
- Contradictions never overwrite: both facts survive, linked.
- The file index makes vault scans incremental (mtime+size fast path, hash
  confirmation) so unchanged files are never re-parsed.
- Every mutation writes an audit row in the same transaction.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schemas import (
    AliasRecord,
    EntityKind,
    EntityRecord,
    EpistemicStatus,
    Fact,
    ReviewAction,
    ReviewItem,
    SchemaError,
    canonical_json,
    text_sha256,
)

_SCHEMA_VERSION = 1


class MemoryError_(RuntimeError):
    """Raised on contract violations in the memory layer."""


@dataclass(frozen=True)
class FileIndexEntry:
    path: str
    mtime_ns: int
    size: int
    content_hash: str


@dataclass(frozen=True)
class ContradictionRecord:
    contradiction_id: int
    campaign_id: str
    fact_id: str
    conflicting_fact_id: str
    note: str
    resolved: bool
    resolution: str | None


class CampaignMemory:
    """Storage facade. One instance may serve many campaigns; every public
    method takes an explicit campaign_id."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._fts_available = self._detect_fts()
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    # -- schema -------------------------------------------------------------

    def _detect_fts(self) -> bool:
        try:
            self._conn.execute("CREATE VIRTUAL TABLE temp.__fts_probe USING fts5(x)")
            self._conn.execute("DROP TABLE temp.__fts_probe")
            return True
        except sqlite3.OperationalError:
            return False

    def _migrate(self) -> None:
        c = self._conn
        with c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row and int(row["value"]) != _SCHEMA_VERSION:
                raise MemoryError_(
                    f"memory schema {row['value']} != supported {_SCHEMA_VERSION}"
                )
            c.execute(
                "INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    name_folded TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    vault_path TEXT,
                    status TEXT NOT NULL,
                    attributes_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    UNIQUE(campaign_id, name_folded, kind)
                )"""
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_campaign ON entities(campaign_id, name_folded)"
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS aliases (
                    alias_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    observed TEXT NOT NULL,
                    observed_folded TEXT NOT NULL,
                    canonical TEXT NOT NULL,
                    entity_id TEXT,
                    approved_by TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    UNIQUE(campaign_id, observed_folded, canonical)
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    statement TEXT NOT NULL,
                    fact_json TEXT NOT NULL,
                    superseded_by TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (campaign_id, fact_id)
                )"""
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(campaign_id, session_id)"
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS fact_entities (
                    campaign_id TEXT NOT NULL,
                    fact_id TEXT NOT NULL,
                    entity_folded TEXT NOT NULL,
                    PRIMARY KEY (campaign_id, fact_id, entity_folded)
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS contradictions (
                    contradiction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    fact_id TEXT NOT NULL,
                    conflicting_fact_id TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    resolved INTEGER NOT NULL DEFAULT 0,
                    resolution TEXT,
                    created_at REAL NOT NULL
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS review_items (
                    item_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    item_json TEXT NOT NULL,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (campaign_id, item_id)
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    subject_folded TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    actor TEXT NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_subject ON feedback(campaign_id, kind, subject_folded)"
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS file_index (
                    campaign_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    PRIMARY KEY (campaign_id, path)
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS summaries (
                    campaign_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary_md TEXT NOT NULL,
                    manifest_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (campaign_id, session_id)
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    actor TEXT NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )
            if self._fts_available:
                c.execute(
                    """CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                        statement, campaign_id UNINDEXED, fact_id UNINDEXED
                    )"""
                )

    # -- audit --------------------------------------------------------------

    def _audit(
        self, campaign_id: str, action: str, subject: str, actor: str, detail: dict | None = None
    ) -> None:
        self._conn.execute(
            "INSERT INTO audit_log (campaign_id, action, subject, detail_json, actor, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (campaign_id, action, subject, canonical_json(detail or {}), actor, time.time()),
        )

    def audit_entries(self, campaign_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE campaign_id=? ORDER BY audit_id DESC LIMIT ?",
            (campaign_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- entities -----------------------------------------------------------

    def upsert_entity(self, entity: EntityRecord, *, actor: str) -> EntityRecord:
        with self._conn:
            existing = self._conn.execute(
                "SELECT entity_id, status FROM entities WHERE campaign_id=? AND name_folded=? AND kind=?",
                (entity.campaign_id, entity.name.casefold(), entity.kind.value),
            ).fetchone()
            if existing:
                # Never let a weaker status downgrade a stronger record.
                old = EpistemicStatus(existing["status"])
                status = entity.status if entity.status.stronger_than(old) else old
                self._conn.execute(
                    "UPDATE entities SET description=?, vault_path=COALESCE(?, vault_path),"
                    " status=?, attributes_json=? WHERE entity_id=?",
                    (
                        entity.description,
                        entity.vault_path,
                        status.value,
                        canonical_json(entity.attributes),
                        existing["entity_id"],
                    ),
                )
                entity.entity_id = existing["entity_id"]
                entity.status = status
            else:
                self._conn.execute(
                    "INSERT INTO entities (entity_id, campaign_id, name, name_folded, kind,"
                    " description, vault_path, status, attributes_json, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        entity.entity_id,
                        entity.campaign_id,
                        entity.name,
                        entity.name.casefold(),
                        entity.kind.value,
                        entity.description,
                        entity.vault_path,
                        entity.status.value,
                        canonical_json(entity.attributes),
                        time.time(),
                    ),
                )
            self._audit(entity.campaign_id, "upsert_entity", entity.name, actor,
                        {"entity_id": entity.entity_id, "kind": entity.kind.value})
        return entity

    def entities(self, campaign_id: str) -> list[EntityRecord]:
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE campaign_id=? ORDER BY name_folded", (campaign_id,)
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def find_entity(self, campaign_id: str, name: str) -> EntityRecord | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE campaign_id=? AND name_folded=?",
            (campaign_id, name.casefold()),
        ).fetchone()
        return self._row_to_entity(row) if row else None

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> EntityRecord:
        return EntityRecord(
            name=row["name"],
            kind=EntityKind(row["kind"]),
            campaign_id=row["campaign_id"],
            entity_id=row["entity_id"],
            description=row["description"],
            vault_path=row["vault_path"],
            status=EpistemicStatus(row["status"]),
            attributes=json.loads(row["attributes_json"]),
        )

    # -- aliases ------------------------------------------------------------

    def add_alias(self, alias: AliasRecord, *, actor: str) -> AliasRecord:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO aliases (alias_id, campaign_id, observed, observed_folded,"
                " canonical, entity_id, approved_by, reason, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    alias.alias_id,
                    alias.campaign_id,
                    alias.observed,
                    alias.observed.casefold(),
                    alias.canonical,
                    alias.entity_id,
                    alias.approved_by,
                    alias.reason,
                    time.time(),
                ),
            )
            self._audit(alias.campaign_id, "add_alias",
                        f"{alias.observed} -> {alias.canonical}", actor,
                        {"alias_id": alias.alias_id, "approved_by": alias.approved_by})
        return alias

    def aliases(self, campaign_id: str) -> list[AliasRecord]:
        rows = self._conn.execute(
            "SELECT * FROM aliases WHERE campaign_id=? ORDER BY observed_folded", (campaign_id,)
        ).fetchall()
        return [
            AliasRecord(
                campaign_id=r["campaign_id"],
                observed=r["observed"],
                canonical=r["canonical"],
                entity_id=r["entity_id"],
                approved_by=r["approved_by"],
                reason=r["reason"],
                alias_id=r["alias_id"],
            )
            for r in rows
        ]

    def resolve_alias(self, campaign_id: str, observed: str) -> AliasRecord | None:
        row = self._conn.execute(
            "SELECT * FROM aliases WHERE campaign_id=? AND observed_folded=?",
            (campaign_id, observed.casefold()),
        ).fetchone()
        if not row:
            return None
        return AliasRecord(
            campaign_id=row["campaign_id"],
            observed=row["observed"],
            canonical=row["canonical"],
            entity_id=row["entity_id"],
            approved_by=row["approved_by"],
            reason=row["reason"],
            alias_id=row["alias_id"],
        )

    # -- facts --------------------------------------------------------------

    def remember_fact(self, fact: Fact, *, actor: str) -> Fact:
        if fact.status is EpistemicStatus.CONFIRMED_CANON and not actor.startswith("human:"):
            raise MemoryError_(
                "only a human actor may store confirmed_canon; use promote_fact"
            )
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO facts (fact_id, campaign_id, session_id, category,"
                " status, confidence, statement, fact_json, superseded_by, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,NULL,?)",
                (
                    fact.fact_id,
                    fact.provenance.campaign_id,
                    fact.provenance.session_id,
                    fact.category.value,
                    fact.status.value,
                    fact.confidence,
                    fact.statement,
                    fact.to_json(),
                    time.time(),
                ),
            )
            self._conn.execute(
                "DELETE FROM fact_entities WHERE campaign_id=? AND fact_id=?",
                (fact.provenance.campaign_id, fact.fact_id),
            )
            for entity in fact.entities:
                self._conn.execute(
                    "INSERT OR IGNORE INTO fact_entities VALUES (?,?,?)",
                    (fact.provenance.campaign_id, fact.fact_id, entity.casefold()),
                )
            if self._fts_available:
                self._conn.execute(
                    "DELETE FROM facts_fts WHERE fact_id=? AND campaign_id=?",
                    (fact.fact_id, fact.provenance.campaign_id),
                )
                self._conn.execute(
                    "INSERT INTO facts_fts (statement, campaign_id, fact_id) VALUES (?,?,?)",
                    (fact.statement, fact.provenance.campaign_id, fact.fact_id),
                )
            self._audit(fact.provenance.campaign_id, "remember_fact", fact.fact_id, actor,
                        {"status": fact.status.value, "session": fact.provenance.session_id})
        return fact

    def get_fact(self, campaign_id: str, fact_id: str) -> Fact | None:
        row = self._conn.execute(
            "SELECT fact_json FROM facts WHERE campaign_id=? AND fact_id=?",
            (campaign_id, fact_id),
        ).fetchone()
        return Fact.from_json(row["fact_json"]) if row else None

    def promote_fact(self, campaign_id: str, fact_id: str, *, actor: str, note: str = "") -> Fact:
        """The only path to CONFIRMED_CANON. Requires a human actor."""
        if not actor.startswith("human:"):
            raise MemoryError_("promotion to canon requires a human actor")
        fact = self.get_fact(campaign_id, fact_id)
        if fact is None:
            raise MemoryError_(f"unknown fact {fact_id}")
        fact.status = EpistemicStatus.CONFIRMED_CANON
        with self._conn:
            self._conn.execute(
                "UPDATE facts SET status=?, fact_json=? WHERE campaign_id=? AND fact_id=?",
                (fact.status.value, fact.to_json(), campaign_id, fact_id),
            )
            self._audit(campaign_id, "promote_fact", fact_id, actor, {"note": note})
        return fact

    def facts_for_session(self, campaign_id: str, session_id: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT fact_json FROM facts WHERE campaign_id=? AND session_id=?"
            " AND superseded_by IS NULL ORDER BY created_at",
            (campaign_id, session_id),
        ).fetchall()
        return [Fact.from_json(r["fact_json"]) for r in rows]

    def facts_for_entity(self, campaign_id: str, entity_name: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT f.fact_json FROM facts f JOIN fact_entities fe"
            " ON f.campaign_id=fe.campaign_id AND f.fact_id=fe.fact_id"
            " WHERE f.campaign_id=? AND fe.entity_folded=? AND f.superseded_by IS NULL"
            " ORDER BY f.created_at",
            (campaign_id, entity_name.casefold()),
        ).fetchall()
        return [Fact.from_json(r["fact_json"]) for r in rows]

    def search_facts(self, campaign_id: str, query: str, *, limit: int = 20) -> list[Fact]:
        if self._fts_available:
            try:
                rows = self._conn.execute(
                    "SELECT fact_id FROM facts_fts WHERE facts_fts MATCH ? AND campaign_id=?"
                    " ORDER BY bm25(facts_fts) LIMIT ?",
                    (query, campaign_id, limit),
                ).fetchall()
                facts = [self.get_fact(campaign_id, r["fact_id"]) for r in rows]
                return [f for f in facts if f is not None]
            except sqlite3.OperationalError:
                pass  # malformed FTS query -> lexical fallback
        needle = f"%{query.casefold()}%"
        rows = self._conn.execute(
            "SELECT fact_json FROM facts WHERE campaign_id=?"
            " AND lower(statement) LIKE ? AND superseded_by IS NULL LIMIT ?",
            (campaign_id, needle, limit),
        ).fetchall()
        return [Fact.from_json(r["fact_json"]) for r in rows]

    def forget_fact(self, campaign_id: str, fact_id: str, *, actor: str, reason: str) -> bool:
        """Remove a learned fact (reversible-learning contract). Audited."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM facts WHERE campaign_id=? AND fact_id=?", (campaign_id, fact_id)
            )
            self._conn.execute(
                "DELETE FROM fact_entities WHERE campaign_id=? AND fact_id=?",
                (campaign_id, fact_id),
            )
            if self._fts_available:
                self._conn.execute(
                    "DELETE FROM facts_fts WHERE campaign_id=? AND fact_id=?",
                    (campaign_id, fact_id),
                )
            self._audit(campaign_id, "forget_fact", fact_id, actor, {"reason": reason})
        return cur.rowcount > 0

    # -- contradictions -----------------------------------------------------

    def record_contradiction(
        self, campaign_id: str, fact_id: str, conflicting_fact_id: str, *, note: str, actor: str
    ) -> int:
        """Link two conflicting facts. Neither is deleted; the older drops to
        CONFLICTING so retrieval flags it until a human resolves."""
        older = self.get_fact(campaign_id, fact_id)
        newer = self.get_fact(campaign_id, conflicting_fact_id)
        if older is None or newer is None:
            raise MemoryError_("both facts must exist to record a contradiction")
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO contradictions (campaign_id, fact_id, conflicting_fact_id,"
                " note, created_at) VALUES (?,?,?,?,?)",
                (campaign_id, fact_id, conflicting_fact_id, note, time.time()),
            )
            if older.status is not EpistemicStatus.CONFIRMED_CANON:
                older.status = EpistemicStatus.CONFLICTING
                self._conn.execute(
                    "UPDATE facts SET status=?, fact_json=? WHERE campaign_id=? AND fact_id=?",
                    (older.status.value, older.to_json(), campaign_id, fact_id),
                )
            self._audit(campaign_id, "record_contradiction",
                        f"{fact_id} vs {conflicting_fact_id}", actor, {"note": note})
        return int(cur.lastrowid)

    def open_contradictions(self, campaign_id: str) -> list[ContradictionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM contradictions WHERE campaign_id=? AND resolved=0", (campaign_id,)
        ).fetchall()
        return [
            ContradictionRecord(
                contradiction_id=r["contradiction_id"],
                campaign_id=r["campaign_id"],
                fact_id=r["fact_id"],
                conflicting_fact_id=r["conflicting_fact_id"],
                note=r["note"],
                resolved=bool(r["resolved"]),
                resolution=r["resolution"],
            )
            for r in rows
        ]

    def resolve_contradiction(
        self, campaign_id: str, contradiction_id: int, *, actor: str,
        resolution: str, winning_fact_id: str | None = None, retcon: bool = False,
    ) -> None:
        if not actor.startswith("human:"):
            raise MemoryError_("contradiction resolution requires a human actor")
        row = self._conn.execute(
            "SELECT * FROM contradictions WHERE campaign_id=? AND contradiction_id=?",
            (campaign_id, contradiction_id),
        ).fetchone()
        if row is None:
            raise MemoryError_(f"unknown contradiction {contradiction_id}")
        with self._conn:
            self._conn.execute(
                "UPDATE contradictions SET resolved=1, resolution=? WHERE contradiction_id=?",
                (resolution, contradiction_id),
            )
            for fid in (row["fact_id"], row["conflicting_fact_id"]):
                fact = self.get_fact(campaign_id, fid)
                if fact is None:
                    continue
                if winning_fact_id is not None and fid == winning_fact_id:
                    fact.status = EpistemicStatus.CONFIRMED_CANON
                elif retcon or winning_fact_id is not None:
                    fact.status = EpistemicStatus.RETCONNED
                self._conn.execute(
                    "UPDATE facts SET status=?, fact_json=? WHERE campaign_id=? AND fact_id=?",
                    (fact.status.value, fact.to_json(), campaign_id, fid),
                )
            self._audit(campaign_id, "resolve_contradiction", str(contradiction_id), actor,
                        {"resolution": resolution, "winner": winning_fact_id, "retcon": retcon})

    # -- review queue -------------------------------------------------------

    def enqueue_review(self, item: ReviewItem, *, actor: str) -> ReviewItem:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO review_items (item_id, campaign_id, session_id,"
                " item_type, subject, item_json, resolved, created_at)"
                " VALUES (?,?,?,?,?,?,0,?)",
                (
                    item.item_id,
                    item.campaign_id,
                    item.session_id,
                    item.item_type,
                    item.subject,
                    canonical_json(_review_item_payload(item)),
                    time.time(),
                ),
            )
            self._audit(item.campaign_id, "enqueue_review", item.subject, actor,
                        {"item_id": item.item_id, "type": item.item_type})
        return item

    def pending_reviews(self, campaign_id: str, *, session_id: str | None = None) -> list[ReviewItem]:
        sql = "SELECT item_json FROM review_items WHERE campaign_id=? AND resolved=0"
        args: list[Any] = [campaign_id]
        if session_id:
            sql += " AND session_id=?"
            args.append(session_id)
        rows = self._conn.execute(sql + " ORDER BY created_at", args).fetchall()
        return [_review_item_from_payload(json.loads(r["item_json"])) for r in rows]

    def resolve_review(
        self,
        campaign_id: str,
        item_id: str,
        *,
        action: ReviewAction,
        actor: str,
        detail: dict | None = None,
    ) -> ReviewItem:
        row = self._conn.execute(
            "SELECT item_json FROM review_items WHERE campaign_id=? AND item_id=?",
            (campaign_id, item_id),
        ).fetchone()
        if row is None:
            raise MemoryError_(f"unknown review item {item_id}")
        item = _review_item_from_payload(json.loads(row["item_json"]))
        item.resolved = action is not ReviewAction.SAVE_FOR_REVIEW
        item.resolution_action = action
        item.resolution_detail = detail or {}
        with self._conn:
            self._conn.execute(
                "UPDATE review_items SET resolved=?, item_json=? WHERE campaign_id=? AND item_id=?",
                (int(item.resolved), canonical_json(_review_item_payload(item)), campaign_id, item_id),
            )
            self._audit(campaign_id, "resolve_review", item.subject, actor,
                        {"item_id": item_id, "action": action.value, **(detail or {})})
        return item

    # -- feedback (continuous improvement) ----------------------------------

    FEEDBACK_KINDS = frozenset(
        {
            "correction_accepted", "correction_rejected", "alias_added",
            "entity_confirmed", "entity_misclassified", "fact_missed",
            "fact_false_positive", "summary_corrected", "vault_link_chosen",
            "deferred",
        }
    )

    def record_feedback(
        self,
        campaign_id: str,
        session_id: str,
        *,
        kind: str,
        subject: str,
        accepted: bool,
        actor: str,
        detail: dict | None = None,
    ) -> int:
        if kind not in self.FEEDBACK_KINDS:
            raise MemoryError_(f"unknown feedback kind {kind!r}")
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO feedback (campaign_id, session_id, kind, subject, subject_folded,"
                " accepted, detail_json, actor, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    campaign_id, session_id, kind, subject, subject.casefold(),
                    int(accepted), canonical_json(detail or {}), actor, time.time(),
                ),
            )
            self._audit(campaign_id, "record_feedback", subject, actor,
                        {"kind": kind, "accepted": accepted})
        return int(cur.lastrowid)

    def feedback_for(self, campaign_id: str, kind: str, subject: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM feedback WHERE campaign_id=? AND kind=? AND subject_folded=?"
            " ORDER BY created_at",
            (campaign_id, kind, subject.casefold()),
        ).fetchall()
        return [dict(r) for r in rows]

    def was_rejected_before(self, campaign_id: str, kind: str, subject: str) -> bool:
        """True when the most recent human feedback for this subject rejected it.

        Used so the engine does not re-propose a correction the user already
        said no to (unless later feedback accepted it again).
        """
        rows = self.feedback_for(campaign_id, kind, subject)
        return bool(rows) and not rows[-1]["accepted"]

    def forget_feedback(self, campaign_id: str, feedback_id: int, *, actor: str, reason: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM feedback WHERE campaign_id=? AND feedback_id=?",
                (campaign_id, feedback_id),
            )
            self._audit(campaign_id, "forget_feedback", str(feedback_id), actor, {"reason": reason})
        return cur.rowcount > 0

    # -- summaries ----------------------------------------------------------

    def remember_summary(
        self, campaign_id: str, session_id: str, summary_md: str, manifest_hash: str, *, actor: str
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?)",
                (campaign_id, session_id, summary_md, manifest_hash, time.time()),
            )
            self._audit(campaign_id, "remember_summary", session_id, actor,
                        {"manifest_hash": manifest_hash})

    def previous_summaries(
        self, campaign_id: str, *, before_session: str, limit: int = 2
    ) -> list[tuple[str, str]]:
        """Most-recent-first (session_id, summary_md) strictly before the
        given session id (ids sort chronologically: date-prefixed)."""
        rows = self._conn.execute(
            "SELECT session_id, summary_md FROM summaries WHERE campaign_id=?"
            " AND session_id < ? ORDER BY session_id DESC LIMIT ?",
            (campaign_id, before_session, limit),
        ).fetchall()
        return [(r["session_id"], r["summary_md"]) for r in rows]

    # -- incremental file index ---------------------------------------------

    def changed_files(
        self, campaign_id: str, root: str | Path, *, suffixes: tuple[str, ...] = (".md",)
    ) -> tuple[list[FileIndexEntry], list[str]]:
        """Return (new_or_changed, deleted_paths) under root for this campaign.

        Unchanged files (same mtime_ns + size) are skipped without reading;
        a touched file with identical content hash is re-stamped but not
        reported as changed.
        """
        root = Path(root)
        known = {
            r["path"]: r
            for r in self._conn.execute(
                "SELECT * FROM file_index WHERE campaign_id=?", (campaign_id,)
            ).fetchall()
        }
        changed: list[FileIndexEntry] = []
        seen: set[str] = set()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes or path.is_symlink():
                continue
            rel = str(path.relative_to(root))
            seen.add(rel)
            stat = path.stat()
            row = known.get(rel)
            if row and row["mtime_ns"] == stat.st_mtime_ns and row["size"] == stat.st_size:
                continue
            content_hash = text_sha256(path.read_text(encoding="utf-8", errors="replace"))
            entry = FileIndexEntry(rel, stat.st_mtime_ns, stat.st_size, content_hash)
            if row and row["content_hash"] == content_hash:
                self._stamp_file(campaign_id, entry)  # touched, not changed
                continue
            changed.append(entry)
        deleted = sorted(set(known) - seen)
        return changed, deleted

    def _stamp_file(self, campaign_id: str, entry: FileIndexEntry) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO file_index VALUES (?,?,?,?,?)",
                (campaign_id, entry.path, entry.mtime_ns, entry.size, entry.content_hash),
            )

    def mark_file_indexed(self, campaign_id: str, entry: FileIndexEntry) -> None:
        self._stamp_file(campaign_id, entry)

    def mark_file_deleted(self, campaign_id: str, path: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM file_index WHERE campaign_id=? AND path=?", (campaign_id, path)
            )

    # -- fine-tuning export --------------------------------------------------

    def export_verified_dataset(self, campaign_id: str) -> list[dict[str, Any]]:
        """Human-decided rows only — the optional future fine-tuning corpus."""
        examples: list[dict[str, Any]] = []
        rows = self._conn.execute(
            "SELECT * FROM feedback WHERE campaign_id=? AND actor LIKE 'human:%' ORDER BY created_at",
            (campaign_id,),
        ).fetchall()
        for r in rows:
            examples.append(
                {
                    "kind": r["kind"],
                    "subject": r["subject"],
                    "accepted": bool(r["accepted"]),
                    "detail": json.loads(r["detail_json"]),
                    "session_id": r["session_id"],
                }
            )
        return examples


def _review_item_payload(item: ReviewItem) -> dict[str, Any]:
    return {
        "campaign_id": item.campaign_id,
        "session_id": item.session_id,
        "item_type": item.item_type,
        "subject": item.subject,
        "reason": item.reason,
        "evidence": item.evidence,
        "suggestions": item.suggestions,
        "confidence": item.confidence,
        "item_id": item.item_id,
        "resolved": item.resolved,
        "resolution_action": item.resolution_action.value if item.resolution_action else None,
        "resolution_detail": item.resolution_detail,
    }


def _review_item_from_payload(data: dict[str, Any]) -> ReviewItem:
    action = data.pop("resolution_action", None)
    item = ReviewItem(
        campaign_id=data["campaign_id"],
        session_id=data["session_id"],
        item_type=data["item_type"],
        subject=data["subject"],
        reason=data["reason"],
        evidence=list(data.get("evidence", [])),
        suggestions=list(data.get("suggestions", [])),
        confidence=float(data.get("confidence", 0.0)),
        item_id=data["item_id"],
        resolved=bool(data.get("resolved", False)),
        resolution_detail=dict(data.get("resolution_detail", {})),
    )
    item.resolution_action = ReviewAction(action) if action else None
    return item
