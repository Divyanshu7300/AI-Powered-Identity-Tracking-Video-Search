from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


SCHEMA_VERSION = 1


class TrackPersistenceStore:
    """SQLite-backed store for processed sources, track memories, and embeddings."""

    def __init__(
        self,
        db_path: str = "data/mot_reid.sqlite3",
        embeddings_dir: str = "data/embeddings",
    ) -> None:
        self.db_path = Path(db_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def save_run(
        self,
        source_name: str,
        source_path: str,
        output_path: str,
        memories: Iterable[Dict[str, object]],
        track_registry: Dict[int, object],
    ) -> None:
        current_memories = list(memories)
        with self._connect() as conn:
            conn.execute("pragma foreign_keys = on")
            conn.execute("pragma journal_mode = wal")
            conn.execute(
                """
                insert into sources(source_name, source_path, output_path, updated_at)
                values (?, ?, ?, datetime('now'))
                on conflict(source_name) do update set
                    source_path=excluded.source_path,
                    output_path=excluded.output_path,
                    updated_at=excluded.updated_at
                """,
                (source_name, source_path, output_path),
            )
            # A completed rerun is the authoritative view of this source. Remove
            # memories that disappeared so searches never return stale tracks.
            current_ids = [str(memory["memory_id"]) for memory in current_memories]
            if current_ids:
                placeholders = ", ".join("?" for _ in current_ids)
                conn.execute(
                    f"delete from track_memories where source_name = ? and memory_id not in ({placeholders})",
                    [source_name, *current_ids],
                )
            else:
                conn.execute("delete from track_memories where source_name = ?", (source_name,))

            for memory in current_memories:
                memory_id = str(memory["memory_id"])
                track_id = int(memory["track_id"])
                track = track_registry.get(track_id)
                embedding_path = None
                if track is not None and getattr(track, "embedding", None) is not None:
                    embedding_path = str(self._embedding_path(memory_id))
                    if hasattr(track, "embedding_candidates"):
                        embedding = track.embedding_candidates()
                    else:
                        embedding = np.asarray(track.embedding, dtype=np.float32)
                    np.save(embedding_path, np.asarray(embedding, dtype=np.float32))

                payload = {
                    **memory,
                    "source_path": source_path,
                    "output_path": output_path,
                    "embedding_path": embedding_path,
                }
                conn.execute(
                    """
                    insert into track_memories(
                        memory_id, source_name, track_id, source_path, output_path,
                        first_frame, last_frame, duration_frames, best_confidence,
                        embedding_path, payload_json, updated_at
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    on conflict(memory_id) do update set
                        source_name=excluded.source_name,
                        track_id=excluded.track_id,
                        source_path=excluded.source_path,
                        output_path=excluded.output_path,
                        first_frame=excluded.first_frame,
                        last_frame=excluded.last_frame,
                        duration_frames=excluded.duration_frames,
                        best_confidence=excluded.best_confidence,
                        embedding_path=excluded.embedding_path,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        memory_id,
                        str(memory.get("source_name", source_name)),
                        track_id,
                        source_path,
                        output_path,
                        int(memory.get("first_frame") or 0),
                        int(memory.get("last_frame") or 0),
                        int(memory.get("duration_frames") or 0),
                        float(memory.get("best_confidence") or 0.0),
                        embedding_path,
                        json.dumps(payload),
                    ),
                )
            self.cleanup_orphan_embeddings(conn)

    def list_memories(self) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select payload_json from track_memories
                order by source_name, track_id
                """
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def get_memory(self, memory_id: str) -> Dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                "select payload_json from track_memories where memory_id = ?",
                (memory_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def delete_memory(self, memory_id: str) -> None:
        with self._connect() as conn:
            conn.execute("delete from track_memories where memory_id = ?", (memory_id,))
            self.cleanup_orphan_embeddings(conn)

    def clear_all(self) -> None:
        """Purge all stored sources, track memories, and embedding files."""
        with self._connect() as conn:
            conn.execute("delete from track_memories")
            conn.execute("delete from sources")
        if self.embeddings_dir.exists():
            for path in self.embeddings_dir.glob("*.npy"):
                path.unlink(missing_ok=True)

    def update_memory_fields(self, memory_id: str, fields: Dict[str, object]) -> Dict[str, object] | None:
        memory = self.get_memory(memory_id)
        if memory is None:
            return None
        memory.update(fields)
        with self._connect() as conn:
            conn.execute(
                """
                update track_memories
                set payload_json = ?, updated_at = datetime('now')
                where memory_id = ?
                """,
                (json.dumps(memory), memory_id),
            )
        return memory

    def load_embedding(self, memory_id: str) -> np.ndarray | None:
        memory = self.get_memory(memory_id)
        if not memory:
            return None
        embedding_path = memory.get("embedding_path")
        if not embedding_path:
            return None
        path = Path(str(embedding_path))
        if not path.exists():
            return None
        return np.load(path, allow_pickle=False).astype(np.float32)

    def source_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("select count(*) as count from sources").fetchone()
        return int(row["count"] if row else 0)

    def memory_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("select count(*) as count from track_memories").fetchone()
        return int(row["count"] if row else 0)

    def cleanup_orphan_embeddings(self, conn: sqlite3.Connection | None = None) -> int:
        owns_connection = conn is None
        connection = conn or self._connect()
        try:
            rows = connection.execute(
                """
                select embedding_path from track_memories
                where embedding_path is not null and embedding_path != ''
                """
            ).fetchall()
            referenced = {str(row["embedding_path"]) for row in rows}
            removed = 0
            for path in self.embeddings_dir.glob("*.npy"):
                if str(path) not in referenced:
                    path.unlink(missing_ok=True)
                    removed += 1
            return removed
        finally:
            if owns_connection:
                connection.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("pragma foreign_keys = on")
            conn.execute("pragma journal_mode = wal")
            conn.execute(
                """
                create table if not exists schema_meta (
                    key text primary key,
                    value text not null,
                    updated_at text not null
                )
                """
            )
            conn.execute(
                """
                create table if not exists sources (
                    source_name text primary key,
                    source_path text not null,
                    output_path text not null,
                    updated_at text not null
                )
                """
            )
            conn.execute(
                """
                create table if not exists track_memories (
                    memory_id text primary key,
                    source_name text not null,
                    track_id integer not null,
                    source_path text,
                    output_path text,
                    first_frame integer,
                    last_frame integer,
                    duration_frames integer,
                    best_confidence real,
                    embedding_path text,
                    payload_json text not null,
                    updated_at text not null
                )
                """
            )
            conn.execute(
                "create index if not exists idx_track_memories_source_track on track_memories(source_name, track_id)"
            )
            conn.execute(
                "create index if not exists idx_track_memories_frames on track_memories(source_name, first_frame, last_frame)"
            )
            conn.execute(
                """
                insert into schema_meta(key, value, updated_at)
                values ('schema_version', ?, datetime('now'))
                on conflict(key) do update set
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                (str(SCHEMA_VERSION),),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _embedding_path(self, memory_id: str) -> Path:
        safe_name = memory_id.replace(":", "_").replace("/", "_")
        return self.embeddings_dir / f"{safe_name}.npy"
