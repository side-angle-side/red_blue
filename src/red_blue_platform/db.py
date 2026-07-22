from __future__ import annotations

import sqlite3
from pathlib import Path
import hashlib
import json
from typing import Iterable

from .blue_memory import structured_experience_similarity, text_embedding
from .blue_team import SYSTEM_PROMPT
from .schema import AttackRecord, Seed


SCHEMA = """
CREATE TABLE IF NOT EXISTS seeds (
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  seed_prompt TEXT NOT NULL,
  strategy TEXT NOT NULL DEFAULT '',
  category TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (source, source_id)
);

CREATE TABLE IF NOT EXISTS attacks (
  data_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  seed_prompt TEXT NOT NULL,
  wrapped_prompt TEXT NOT NULL,
  strategy TEXT NOT NULL,
  response TEXT,
  blue_decision TEXT,
  blue_memory_top_score REAL,
  blue_memory_matches TEXT,
  blue_detector_version_id TEXT,
  eval_result TEXT,
  eval_reason TEXT,
  eval_confidence REAL,
  parent_data_id TEXT,
  generation INTEGER NOT NULL DEFAULT 0,
  evolution_reason TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (source, source_id) REFERENCES seeds(source, source_id),
  FOREIGN KEY (parent_data_id) REFERENCES attacks(data_id)
);

CREATE INDEX IF NOT EXISTS idx_attacks_eval_result ON attacks(eval_result);
CREATE INDEX IF NOT EXISTS idx_attacks_response_missing ON attacks(response);
CREATE INDEX IF NOT EXISTS idx_attacks_parent_data_id ON attacks(parent_data_id);
CREATE INDEX IF NOT EXISTS idx_attacks_generation ON attacks(generation);

CREATE TABLE IF NOT EXISTS blue_experiences (
  memory_id TEXT PRIMARY KEY,
  data_id TEXT NOT NULL UNIQUE,
  lesson_text TEXT NOT NULL,
  intent TEXT NOT NULL DEFAULT '',
  attack_pattern TEXT NOT NULL DEFAULT '',
  obfuscation_style TEXT NOT NULL DEFAULT '',
  trigger_features TEXT NOT NULL DEFAULT '',
  repair_hint TEXT NOT NULL DEFAULT '',
  safe_decision TEXT NOT NULL DEFAULT '',
  source_data_id TEXT NOT NULL DEFAULT '',
  embedding TEXT NOT NULL DEFAULT '{}',
  detector_training_used INTEGER NOT NULL DEFAULT 0,
  detector_holdout_used INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (data_id) REFERENCES attacks(data_id)
);

CREATE INDEX IF NOT EXISTS idx_blue_experiences_training_used
ON blue_experiences(detector_training_used);

CREATE TABLE IF NOT EXISTS blue_detector_versions (
  version_id TEXT PRIMARY KEY,
  generation INTEGER NOT NULL,
  parent_version_id TEXT,
  system_prompt TEXT NOT NULL,
  evolution_reason TEXT NOT NULL,
  training_memory_ids TEXT NOT NULL DEFAULT '[]',
  holdout_memory_ids TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'active',
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate_attacks(conn)
            self._migrate_blue_memory(conn)
        self.ensure_blue_detector_baseline()

    def _migrate_attacks(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(attacks)")}
        if "parent_data_id" not in columns:
            conn.execute("ALTER TABLE attacks ADD COLUMN parent_data_id TEXT")
        if "generation" not in columns:
            conn.execute("ALTER TABLE attacks ADD COLUMN generation INTEGER NOT NULL DEFAULT 0")
        if "evolution_reason" not in columns:
            conn.execute("ALTER TABLE attacks ADD COLUMN evolution_reason TEXT")
        if "blue_decision" not in columns:
            conn.execute("ALTER TABLE attacks ADD COLUMN blue_decision TEXT")
        if "blue_memory_top_score" not in columns:
            conn.execute("ALTER TABLE attacks ADD COLUMN blue_memory_top_score REAL")
        if "blue_memory_matches" not in columns:
            conn.execute("ALTER TABLE attacks ADD COLUMN blue_memory_matches TEXT")
        if "blue_detector_version_id" not in columns:
            conn.execute("ALTER TABLE attacks ADD COLUMN blue_detector_version_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attacks_parent_data_id ON attacks(parent_data_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attacks_generation ON attacks(generation)")

    def _migrate_blue_memory(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blue_experiences (
              memory_id TEXT PRIMARY KEY,
              data_id TEXT NOT NULL UNIQUE,
              lesson_text TEXT NOT NULL,
              intent TEXT NOT NULL DEFAULT '',
              attack_pattern TEXT NOT NULL DEFAULT '',
              obfuscation_style TEXT NOT NULL DEFAULT '',
              trigger_features TEXT NOT NULL DEFAULT '',
              repair_hint TEXT NOT NULL DEFAULT '',
              safe_decision TEXT NOT NULL DEFAULT '',
              source_data_id TEXT NOT NULL DEFAULT '',
              embedding TEXT NOT NULL DEFAULT '{}',
              detector_training_used INTEGER NOT NULL DEFAULT 0,
              detector_holdout_used INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY (data_id) REFERENCES attacks(data_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blue_detector_versions (
              version_id TEXT PRIMARY KEY,
              generation INTEGER NOT NULL,
              parent_version_id TEXT,
              system_prompt TEXT NOT NULL,
              evolution_reason TEXT NOT NULL,
              training_memory_ids TEXT NOT NULL DEFAULT '[]',
              holdout_memory_ids TEXT NOT NULL DEFAULT '[]',
              status TEXT NOT NULL DEFAULT 'active',
              is_active INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        exp_columns = {row["name"] for row in conn.execute("PRAGMA table_info(blue_experiences)")}
        if "detector_training_used" not in exp_columns:
            conn.execute("ALTER TABLE blue_experiences ADD COLUMN detector_training_used INTEGER NOT NULL DEFAULT 0")
        if "detector_holdout_used" not in exp_columns:
            conn.execute("ALTER TABLE blue_experiences ADD COLUMN detector_holdout_used INTEGER NOT NULL DEFAULT 0")
        if "safe_decision" not in exp_columns:
            conn.execute("ALTER TABLE blue_experiences ADD COLUMN safe_decision TEXT NOT NULL DEFAULT ''")
        if "source_data_id" not in exp_columns:
            conn.execute("ALTER TABLE blue_experiences ADD COLUMN source_data_id TEXT NOT NULL DEFAULT ''")
        if "embedding" not in exp_columns:
            conn.execute("ALTER TABLE blue_experiences ADD COLUMN embedding TEXT NOT NULL DEFAULT '{}'")
        version_columns = {row["name"] for row in conn.execute("PRAGMA table_info(blue_detector_versions)")}
        if "holdout_memory_ids" not in version_columns:
            conn.execute("ALTER TABLE blue_detector_versions ADD COLUMN holdout_memory_ids TEXT NOT NULL DEFAULT '[]'")
        if "status" not in version_columns:
            conn.execute("ALTER TABLE blue_detector_versions ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blue_experiences_training_used "
            "ON blue_experiences(detector_training_used)"
        )

    def upsert_seeds(self, seeds: Iterable[Seed]) -> int:
        rows = list(seeds)
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO seeds (source, source_id, seed_prompt, strategy, category)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source, source_id) DO UPDATE SET
                  seed_prompt = excluded.seed_prompt,
                  strategy = excluded.strategy,
                  category = excluded.category
                """,
                [(s.source, s.source_id, s.seed_prompt, s.strategy, s.category) for s in rows],
            )
        return len(rows)

    def fetch_seeds_for_generation(self, limit: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT s.*
                    FROM seeds s
                    WHERE NOT EXISTS (
                      SELECT 1 FROM attacks a
                      WHERE a.source = s.source AND a.source_id = s.source_id
                    )
                    ORDER BY s.created_at, s.source, s.source_id
                    LIMIT ?
                    """,
                    (limit,),
                )
            )

    def insert_attack(self, source: str, source_id: str, record: AttackRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO attacks (
                  data_id, source, source_id, seed_prompt, wrapped_prompt, strategy,
                  parent_data_id, generation, evolution_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.data_id,
                    source,
                    source_id,
                    record.seed_prompt,
                    record.wrapped_prompt,
                    record.strategy,
                    record.parent_data_id,
                    record.generation,
                    record.evolution_reason,
                ),
            )

    def insert_generation_failure(self, seed: dict[str, object], reason: str) -> str:
        source = str(seed["source"])
        source_id = str(seed["source_id"])
        digest = hashlib.sha256(f"{source}:{source_id}:generation_failed".encode("utf-8")).hexdigest()[:12]
        data_id = f"{source}:{source_id}:generation-failed:{digest}"
        self.insert_attack(
            source,
            source_id,
            AttackRecord(
                data_id=data_id,
                seed_prompt=str(seed["seed_prompt"]),
                wrapped_prompt="[Red-team generation failed before an attack could be produced.]",
                strategy="generation_failed",
                evolution_reason=reason,
            ),
        )
        self.mark_attack_processing_failed(data_id, reason)
        return data_id

    def insert_evolution_failure(self, attack: dict[str, object], reason: str) -> str:
        parent_data_id = str(attack["data_id"])
        generation = int(attack.get("generation") or 0) + 1
        digest = hashlib.sha256(f"{parent_data_id}:evolution_failed:{generation}".encode("utf-8")).hexdigest()[:12]
        data_id = f"{parent_data_id}:evolution-failed:{digest}"
        self.insert_attack(
            str(attack["source"]),
            str(attack["source_id"]),
            AttackRecord(
                data_id=data_id,
                seed_prompt=str(attack["seed_prompt"]),
                wrapped_prompt="[Red-team evolution failed before a child attack could be produced.]",
                strategy=f"{attack.get('strategy') or 'unknown'}+evolution_failed",
                parent_data_id=parent_data_id,
                generation=generation,
                evolution_reason=reason,
            ),
        )
        self.mark_attack_processing_failed(data_id, reason)
        return data_id

    def mark_attack_processing_failed(self, data_id: str, reason: str) -> None:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE attacks
                SET response = CASE
                      WHEN response IS NULL OR TRIM(response) = ''
                      THEN '[Pipeline processing failed; no blue response was produced.]'
                      ELSE response
                    END,
                    blue_decision = CASE
                      WHEN blue_decision IS NULL OR TRIM(blue_decision) = ''
                      THEN 'processing_failed'
                      ELSE blue_decision
                    END,
                    eval_result = 'processing_failed',
                    eval_reason = ?,
                    eval_confidence = 0.0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE data_id = ?
                """,
                (reason, data_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"No attack row found for Data_ID: {data_id}")

    def export_attacks(
        self,
        only_missing_response: bool = False,
        evolved_only: bool = False,
    ) -> list[AttackRecord]:
        where_parts = []
        if only_missing_response:
            where_parts.append("response IS NULL")
        if evolved_only:
            where_parts.append("parent_data_id IS NOT NULL")
        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM attacks {where} ORDER BY created_at, data_id").fetchall()
        return [row_to_attack(r) for r in rows]

    def fetch_attacks_for_evolution(
        self,
        defense_success_labels: set[str],
        limit: int,
        include_already_evolved: bool = False,
    ) -> list[sqlite3.Row]:
        labels = {_normalize_label(label) for label in defense_success_labels if label.strip()}
        if not labels:
            return []
        label_placeholders = ", ".join("?" for _ in labels)
        where_parts = [
            "eval_result IS NOT NULL",
            f"LOWER(REPLACE(REPLACE(TRIM(eval_result), '-', '_'), ' ', '_')) IN ({label_placeholders})",
            "response IS NOT NULL",
            "TRIM(response) != ''",
        ]
        params: list[object] = sorted(labels)
        if not include_already_evolved:
            where_parts.append(
                """
                NOT EXISTS (
                  SELECT 1 FROM attacks child
                  WHERE child.parent_data_id = attacks.data_id
                )
                """
            )
        where = "WHERE " + " AND ".join(where_parts)
        with self.connect() as conn:
            return list(
                conn.execute(
                    f"""
                    SELECT *
                    FROM attacks
                    {where}
                    ORDER BY updated_at, data_id
                    LIMIT ?
                    """,
                    (*params, limit),
                )
            )

    def update_response(
        self,
        data_id: str,
        response: str,
        blue_decision: str | None = None,
        blue_memory_top_score: float | None = None,
        blue_memory_matches: str | None = None,
        blue_detector_version_id: str | None = None,
    ) -> None:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE attacks
                SET response = ?,
                    blue_decision = ?,
                    blue_memory_top_score = ?,
                    blue_memory_matches = ?,
                    blue_detector_version_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE data_id = ?
                """,
                (
                    response,
                    blue_decision,
                    blue_memory_top_score,
                    blue_memory_matches,
                    blue_detector_version_id,
                    data_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"No attack row found for Data_ID: {data_id}")

    def ensure_blue_detector_baseline(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        with self.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM blue_detector_versions WHERE version_id = 'blue-v0'"
            ).fetchone()
            if exists:
                return
            conn.execute(
                """
                INSERT INTO blue_detector_versions (
                  version_id, generation, parent_version_id, system_prompt,
                  evolution_reason, training_memory_ids, holdout_memory_ids, status, is_active
                )
                VALUES ('blue-v0', 0, NULL, ?, 'Initial blue detector prompt.', '[]', '[]', 'active', 1)
                """,
                (system_prompt,),
            )

    def active_blue_detector(self) -> sqlite3.Row:
        self.ensure_blue_detector_baseline()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM blue_detector_versions
                WHERE is_active = 1
                ORDER BY generation DESC, created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                raise RuntimeError("No active blue detector version is available")
            return row

    def insert_blue_detector_version(
        self,
        *,
        version_id: str,
        parent_version_id: str,
        generation: int,
        system_prompt: str,
        evolution_reason: str,
        training_memory_ids: list[str],
        holdout_memory_ids: list[str] | None = None,
        activate: bool = True,
    ) -> None:
        status = "active" if activate else "rejected"
        with self.connect() as conn:
            if activate:
                conn.execute("UPDATE blue_detector_versions SET is_active = 0")
            conn.execute(
                """
                INSERT OR REPLACE INTO blue_detector_versions (
                  version_id, generation, parent_version_id, system_prompt,
                  evolution_reason, training_memory_ids, holdout_memory_ids, status, is_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    generation,
                    parent_version_id,
                    system_prompt,
                    evolution_reason,
                    json.dumps(training_memory_ids, ensure_ascii=False),
                    json.dumps(holdout_memory_ids or [], ensure_ascii=False),
                    status,
                    int(activate),
                ),
            )
            conn.executemany(
                "UPDATE blue_experiences SET detector_training_used = 1 WHERE memory_id = ?",
                [(memory_id,) for memory_id in training_memory_ids],
            )
            conn.executemany(
                "UPDATE blue_experiences SET detector_holdout_used = 1 WHERE memory_id = ?",
                [(memory_id,) for memory_id in (holdout_memory_ids or [])],
            )

    def insert_replay_attack(self, attack: dict[str, object]) -> bool:
        parent_data_id = str(attack["data_id"])
        generation = int(attack.get("generation") or 0) + 1
        digest = hashlib.sha256(f"{parent_data_id}:replay:{generation}".encode("utf-8")).hexdigest()[:12]
        data_id = f"{parent_data_id}:replay:{digest}"
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO attacks (
                  data_id, source, source_id, seed_prompt, wrapped_prompt, strategy,
                  parent_data_id, generation, evolution_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data_id,
                    attack["source"],
                    attack["source_id"],
                    attack["seed_prompt"],
                    attack["wrapped_prompt"],
                    f"{attack.get('strategy') or 'unknown'}+replay",
                    parent_data_id,
                    generation,
                    "Replay of an attack_success sample for the next blue-team round.",
                ),
            )
            return cursor.rowcount > 0

    def insert_blue_experience(self, data_id: str, experience: dict[str, str]) -> bool:
        memory_id = f"blue-memory:{data_id}"
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO blue_experiences (
                  memory_id, data_id, lesson_text, intent, attack_pattern,
                  obfuscation_style, trigger_features, repair_hint,
                  safe_decision, source_data_id, embedding
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    data_id,
                    experience["lesson_text"],
                    experience.get("intent", ""),
                    experience.get("attack_pattern", ""),
                    experience.get("obfuscation_style", ""),
                    experience.get("trigger_features", ""),
                    experience.get("repair_hint", experience.get("safe_decision", "")),
                    experience.get("safe_decision", experience.get("repair_hint", "")),
                    experience.get("source_data_id", data_id),
                    experience.get("embedding") or text_embedding(experience["lesson_text"]),
                ),
            )
            return cursor.rowcount > 0

    def search_blue_experiences(self, wrapped_prompt: str, top_k: int = 3) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM blue_experiences
                ORDER BY created_at DESC, memory_id
                """
            ).fetchall()
        scored = []
        for row in rows:
            score = structured_experience_similarity(wrapped_prompt, dict(row))
            if score <= 0:
                continue
            scored.append(
                {
                    "memory_id": row["memory_id"],
                    "data_id": row["data_id"],
                    "lesson_text": row["lesson_text"],
                    "intent": row["intent"],
                    "attack_pattern": row["attack_pattern"],
                    "obfuscation_style": row["obfuscation_style"],
                    "trigger_features": row["trigger_features"],
                    "safe_decision": row["safe_decision"],
                    "source_data_id": row["source_data_id"],
                    "similarity_score": round(score, 4),
                }
            )
        scored.sort(key=lambda item: (-float(item["similarity_score"]), str(item["memory_id"])))
        return scored[:top_k]

    def pending_blue_experiences_for_detector(self, limit: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT *
                    FROM blue_experiences
                    WHERE detector_training_used = 0
                      AND detector_holdout_used = 0
                    ORDER BY created_at, memory_id
                    LIMIT ?
                    """,
                    (limit,),
                )
            )

    def pending_blue_experience_split(
        self,
        training_limit: int,
        holdout_limit: int,
    ) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
        total = training_limit + holdout_limit
        with self.connect() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT *
                    FROM blue_experiences
                    WHERE detector_training_used = 0
                      AND detector_holdout_used = 0
                    ORDER BY created_at, memory_id
                    LIMIT ?
                    """,
                    (total,),
                )
            )
        return rows[:training_limit], rows[training_limit:total]

    def update_eval(
        self,
        data_id: str,
        eval_result: str,
        reason: str | None = None,
        confidence: float | None = None,
    ) -> None:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE attacks
                SET eval_result = ?,
                    eval_reason = ?,
                    eval_confidence = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE data_id = ?
                """,
                (eval_result, reason, confidence, data_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"No attack row found for Data_ID: {data_id}")

    def fetch_attacks_for_blue_response(self, limit: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT a.*, s.category
                    FROM attacks a
                    LEFT JOIN seeds s ON s.source = a.source AND s.source_id = a.source_id
                    WHERE a.response IS NULL OR TRIM(a.response) = ''
                    ORDER BY a.created_at, a.data_id
                    LIMIT ?
                    """,
                    (limit,),
                )
            )

    def fetch_attacks_for_evaluation(self, limit: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT a.*, s.category
                    FROM attacks a
                    LEFT JOIN seeds s ON s.source = a.source AND s.source_id = a.source_id
                    WHERE a.response IS NOT NULL
                      AND TRIM(a.response) != ''
                      AND (a.eval_result IS NULL OR TRIM(a.eval_result) = '')
                    ORDER BY a.updated_at, a.data_id
                    LIMIT ?
                    """,
                    (limit,),
                )
            )

    def stats(self, source: str | None = None) -> dict[str, object]:
        seed_where = "WHERE source = ?" if source else ""
        attack_where = "WHERE source = ?" if source else ""
        params = (source,) if source else ()
        with self.connect() as conn:
            seed_count = conn.execute(f"SELECT COUNT(*) FROM seeds {seed_where}", params).fetchone()[0]
            attack_count = conn.execute(f"SELECT COUNT(*) FROM attacks {attack_where}", params).fetchone()[0]
            return {
                "seed_count": seed_count,
                "attack_count": attack_count,
                "seed_categories": _count_rows(conn, "seeds", "category", seed_where, params),
                "seed_strategies": _count_rows(conn, "seeds", "strategy", seed_where, params),
                "attack_strategies": _count_rows(conn, "attacks", "strategy", attack_where, params),
                "eval_results": _count_rows(conn, "attacks", "eval_result", attack_where, params),
            }

    def attack_success_stats(
        self,
        success_labels: set[str],
        source: str | None = None,
    ) -> dict[str, object]:
        where_parts = ["eval_result IS NOT NULL", "TRIM(eval_result) != ''"]
        params: list[str] = []
        if source:
            where_parts.append("source = ?")
            params.append(source)
        where = "WHERE " + " AND ".join(where_parts)

        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT strategy, eval_result, COUNT(*) AS count
                FROM attacks
                {where}
                GROUP BY strategy, eval_result
                ORDER BY strategy, eval_result
                """,
                tuple(params),
            ).fetchall()

        by_strategy: dict[str, dict[str, int]] = {}
        total = 0
        success = 0
        for row in rows:
            strategy = row["strategy"] or "(empty)"
            label = _normalize_label(row["eval_result"])
            count = int(row["count"])
            strategy_stats = by_strategy.setdefault(strategy, {"total": 0, "success": 0})
            strategy_stats["total"] += count
            total += count
            if label in success_labels:
                strategy_stats["success"] += count
                success += count

        return {
            "total": total,
            "success": success,
            "rate": (success / total) if total else None,
            "by_strategy": by_strategy,
        }


def row_to_attack(row: sqlite3.Row) -> AttackRecord:
    return AttackRecord(
        data_id=row["data_id"],
        seed_prompt=row["seed_prompt"],
        wrapped_prompt=row["wrapped_prompt"],
        strategy=row["strategy"],
        response=row["response"],
        eval_result=row["eval_result"],
        eval_reason=row["eval_reason"],
        eval_confidence=row["eval_confidence"],
        parent_data_id=row["parent_data_id"],
        generation=int(row["generation"] or 0),
        evolution_reason=row["evolution_reason"],
        blue_decision=row["blue_decision"],
        blue_memory_top_score=row["blue_memory_top_score"],
        blue_memory_matches=row["blue_memory_matches"],
        blue_detector_version_id=row["blue_detector_version_id"],
    )


def _count_rows(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    where: str,
    params: tuple[str, ...],
) -> list[tuple[str, int]]:
    rows = conn.execute(
        f"""
        SELECT COALESCE({column}, '') AS value, COUNT(*) AS count
        FROM {table}
        {where}
        GROUP BY value
        ORDER BY count DESC, value
        """,
        params,
    ).fetchall()
    return [(row["value"], row["count"]) for row in rows]


def _normalize_label(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")
