from __future__ import annotations

from collections.abc import Iterable

import aiosqlite

from hackaton.service.dto import EventDTO, ShiftDTO, UserDTO


class Repository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def upsert_users(self, users: Iterable[UserDTO]) -> int:
        payload = [(u.id, u.location_id, int(u.is_strict_location), int(u.has_mk)) for u in users]
        if not payload:
            return 0
        query = """
        INSERT INTO users(id, location_id, is_strict_location, has_mk)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          location_id=excluded.location_id,
          is_strict_location=excluded.is_strict_location,
          has_mk=excluded.has_mk
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(query, payload)
            await db.commit()
        return len(payload)

    async def upsert_shifts(self, shifts: Iterable[ShiftDTO]) -> int:
        payload = [
            (
                s.id,
                s.start_at.isoformat(),
                s.location_id,
                s.task_type,
                s.employer_id,
                s.workplace_id,
                int(s.need_mk),
                int(s.id_differential),
                s.hours,
                float(s.reward),
                s.capacity,
            )
            for s in shifts
        ]
        if not payload:
            return 0
        query = """
        INSERT INTO shifts(id, start_at, location_id, task_type, employer_id,
                           workplace_id, need_mk, id_differential, hours, reward, capacity)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          start_at=excluded.start_at,
          location_id=excluded.location_id,
          task_type=excluded.task_type,
          employer_id=excluded.employer_id,
          workplace_id=excluded.workplace_id,
          need_mk=excluded.need_mk,
          id_differential=excluded.id_differential,
          hours=excluded.hours,
          reward=excluded.reward,
          capacity=excluded.capacity
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(query, payload)
            await db.commit()
        return len(payload)

    async def insert_events(self, events: Iterable[EventDTO]) -> int:
        payload = [
            (str(e.id), e.shift_id, e.user_id, e.interaction.value, e.ts.isoformat())
            for e in events
        ]
        if not payload:
            return 0
        query = """
        INSERT OR REPLACE INTO events(id, shift_id, user_id, interaction, ts)
        VALUES(?, ?, ?, ?, ?)
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(query, payload)
            await db.commit()
        return len(payload)

    async def count_table(self, table_name: str) -> int:
        if table_name not in {"users", "events", "shifts"}:
            raise ValueError(f"unsupported table: {table_name}")
        query = f"SELECT COUNT(1) FROM {table_name}"  # nosec - table is validated above
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query)
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def find_top_candidates(
        self,
        location_id: str,
        need_mk: bool,
        limit: int,
    ) -> list[str]:
        query = """
        SELECT id
        FROM users
        WHERE location_id = ?
          AND (? = 0 OR has_mk = 1)
        ORDER BY is_strict_location DESC, has_mk DESC, id ASC
        LIMIT ?
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (location_id, int(need_mk), limit))
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def fallback_candidates(self, limit: int) -> list[str]:
        query = "SELECT id FROM users ORDER BY id ASC LIMIT ?"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (limit,))
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def find_candidates_for_ml(
        self,
        location_id: str,
        need_mk: bool,
        limit: int = 3000,
    ) -> list[dict]:
        query = """
        SELECT id, location_id, is_strict_location, has_mk
        FROM users
        WHERE location_id = ?
          AND (? = 0 OR has_mk = 1)
        LIMIT ?
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, (location_id, int(need_mk), limit))
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]
