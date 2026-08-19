"""Trainable suggestion scorer — the engine's own learning, no external AI.

A small logistic-regression model (pure Python, no dependencies) over
interpretable features of a (observed name → candidate) pair. It starts from
hand-set weights equivalent to the resolver's heuristics, and every training
pass re-fits the weights from the campaign's human feedback ledger: accepted
corrections/aliases are positive examples, rejected ones negative. Weights
are stored per campaign in the memory database, so learning is inspectable
(`weights()`), reversible (`reset()`), and campaign-isolated like everything
else.

This is deliberately not a neural network: with hundreds (not millions) of
labelled decisions per campaign, a transparent linear model both performs
adequately and can *explain* its score feature by feature.
"""
from __future__ import annotations

import json
import math
import time

from .memory import CampaignMemory
from .phonetics import damerau_levenshtein, phonetic_key, similarity_ratio

LEARNER_VERSION = "suggestion-learner-v1"

FEATURES = (
    "bias",
    "similarity",          # Damerau ratio 0..1
    "edit_distance_1",     # exactly one edit apart
    "phonetic_match",      # same phonetic key
    "token_containment",   # one name's tokens inside the other
    "length_penalty",      # short observed names are riskier
    "previously_accepted", # this exact pair was accepted before
    "previously_rejected", # this exact pair was rejected before
)

# Heuristic-equivalent starting point (used until enough feedback exists).
DEFAULT_WEIGHTS = {
    "bias": -1.0,
    "similarity": 2.2,
    "edit_distance_1": 1.2,
    "phonetic_match": 1.0,
    "token_containment": 1.4,
    "length_penalty": -1.5,
    "previously_accepted": 3.0,
    "previously_rejected": -4.0,
}

MIN_TRAINING_EXAMPLES = 8


def pair_features(
    memory: CampaignMemory, campaign_id: str, observed: str, candidate: str
) -> dict[str, float]:
    observed_f = " ".join(observed.casefold().split())
    candidate_f = " ".join(candidate.casefold().split())
    obs_tokens = {t for t in observed_f.split() if len(t) > 3}
    cand_tokens = {t for t in candidate_f.split() if len(t) > 3}
    containment = bool(obs_tokens) and (
        obs_tokens.issubset(set(candidate_f.split()))
        or cand_tokens.issubset(set(observed_f.split()))
    )
    subject = f"{observed}->{candidate}"
    accepted = rejected = 0.0
    rows = memory.feedback_for(campaign_id, "correction_accepted", subject) + \
        memory.feedback_for(campaign_id, "alias_added", subject)
    if any(r["accepted"] for r in rows):
        accepted = 1.0
    rej = memory.feedback_for(campaign_id, "correction_rejected", subject)
    if rej and not rej[-1]["accepted"]:
        rejected = 1.0
    return {
        "bias": 1.0,
        "similarity": similarity_ratio(observed_f, candidate_f),
        "edit_distance_1": 1.0 if damerau_levenshtein(observed_f, candidate_f, cap=2) == 1 else 0.0,
        "phonetic_match": 1.0 if phonetic_key(observed) and phonetic_key(observed) == phonetic_key(candidate) else 0.0,
        "token_containment": 1.0 if containment else 0.0,
        "length_penalty": 1.0 if len(observed_f) <= 4 else 0.0,
        "previously_accepted": accepted,
        "previously_rejected": rejected,
    }


class SuggestionLearner:
    def __init__(self, memory: CampaignMemory):
        self.memory = memory

    # -- persistence --------------------------------------------------------

    def _meta_key(self, campaign_id: str) -> str:
        return f"learner:{campaign_id}:{LEARNER_VERSION}"

    def weights(self, campaign_id: str) -> dict[str, float]:
        row = self.memory._conn.execute(
            "SELECT value FROM meta WHERE key=?", (self._meta_key(campaign_id),)
        ).fetchone()
        if row is None:
            return dict(DEFAULT_WEIGHTS)
        stored = json.loads(row["value"])
        return {f: float(stored.get(f, DEFAULT_WEIGHTS[f])) for f in FEATURES}

    def _save(self, campaign_id: str, weights: dict[str, float], examples: int) -> None:
        payload = dict(weights)
        payload["_trained_at"] = time.time()
        payload["_examples"] = examples
        with self.memory._conn:
            self.memory._conn.execute(
                "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                (self._meta_key(campaign_id), json.dumps(payload)),
            )
            self.memory._audit(campaign_id, "train_learner", LEARNER_VERSION,
                               "engine-learner", {"examples": examples})

    def reset(self, campaign_id: str, *, actor: str) -> None:
        with self.memory._conn:
            self.memory._conn.execute(
                "DELETE FROM meta WHERE key=?", (self._meta_key(campaign_id),)
            )
            self.memory._audit(campaign_id, "reset_learner", LEARNER_VERSION, actor, {})

    # -- scoring ------------------------------------------------------------

    def score(self, campaign_id: str, observed: str, candidate: str) -> tuple[float, dict[str, float]]:
        """Return (probability, per-feature contributions) — explainable."""
        weights = self.weights(campaign_id)
        features = pair_features(self.memory, campaign_id, observed, candidate)
        contributions = {f: weights[f] * features[f] for f in FEATURES}
        z = sum(contributions.values())
        probability = 1.0 / (1.0 + math.exp(-z))
        return probability, contributions

    # -- training -----------------------------------------------------------

    def training_examples(self, campaign_id: str) -> list[tuple[dict[str, float], int]]:
        """(features, label) pairs from human feedback: accepted=1, rejected=0."""
        rows = self.memory._conn.execute(
            "SELECT kind, subject, accepted FROM feedback WHERE campaign_id=?"
            " AND actor LIKE 'human:%' AND kind IN"
            " ('correction_accepted','correction_rejected','alias_added')"
            " ORDER BY created_at",
            (campaign_id,),
        ).fetchall()
        examples = []
        for row in rows:
            subject = row["subject"]
            if "->" not in subject:
                continue
            observed, _, candidate = subject.partition("->")
            label = 1 if row["accepted"] else 0
            features = pair_features(self.memory, campaign_id, observed, candidate)
            # The history features would leak the label during training.
            features["previously_accepted"] = 0.0
            features["previously_rejected"] = 0.0
            examples.append((features, label))
        return examples

    def train(
        self, campaign_id: str, *, epochs: int = 200, learning_rate: float = 0.5,
        l2: float = 0.01,
    ) -> dict[str, float] | None:
        """Fit weights by gradient descent; None if too little feedback yet."""
        examples = self.training_examples(campaign_id)
        if len(examples) < MIN_TRAINING_EXAMPLES:
            return None
        labels = {e[1] for e in examples}
        if len(labels) < 2:
            return None  # need both accepted and rejected examples
        weights = dict(DEFAULT_WEIGHTS)
        n = len(examples)
        for _ in range(epochs):
            gradient = {f: 0.0 for f in FEATURES}
            for features, label in examples:
                z = sum(weights[f] * features[f] for f in FEATURES)
                p = 1.0 / (1.0 + math.exp(-z))
                error = p - label
                for f in FEATURES:
                    gradient[f] += error * features[f]
            for f in FEATURES:
                regular = 0.0 if f == "bias" else l2 * weights[f]
                weights[f] -= learning_rate * (gradient[f] / n + regular)
        # Feedback-history features keep their strong priors post-training.
        weights["previously_accepted"] = DEFAULT_WEIGHTS["previously_accepted"]
        weights["previously_rejected"] = DEFAULT_WEIGHTS["previously_rejected"]
        self._save(campaign_id, weights, len(examples))
        return weights
