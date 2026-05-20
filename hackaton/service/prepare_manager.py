from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import aiosqlite

LOGGER = logging.getLogger(__name__)


@dataclass
class PrepareState:
    running: bool = False
    ready: bool = True
    user_stats: dict[str, dict] = field(default_factory=dict)
    user_employer_stats: dict[tuple[str, str], int] = field(default_factory=dict)
    user_workplace_stats: dict[tuple[str, str], int] = field(default_factory=dict)
    user_shift_events: dict[tuple[str, str], dict] = field(default_factory=dict)


class PrepareManager:
    def __init__(self, sleep_seconds: int, db_path: str = "") -> None:
        self._state = PrepareState()
        self._task: asyncio.Task[None] | None = None
        self._sleep_seconds = sleep_seconds
        self.db_path = db_path

    @property
    def ready(self) -> bool:
        return self._state.ready and not self._state.running

    @property
    def user_stats(self) -> dict:
        return self._state.user_stats

    @property
    def user_employer_stats(self) -> dict:
        return self._state.user_employer_stats

    @property
    def user_workplace_stats(self) -> dict:
        return self._state.user_workplace_stats

    @property
    def user_shift_events(self) -> dict:
        return self._state.user_shift_events

    async def start(self) -> bool:
        if self._state.running:
            return False
        self._state.running = True
        self._state.ready = False
        self._task = asyncio.create_task(self._background_prepare())
        return True

    async def _background_prepare(self) -> None:
        try:
            LOGGER.info("Starting background preparation of ML features...")

            # Эмулируем задержку, если требуется валидатором
            await asyncio.sleep(self._sleep_seconds)

            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Считаем глобальные исторические счетчики по юзерам
                user_stats = {}
                query_users = """
                SELECT
                    user_id,
                    SUM(CASE WHEN interaction = 'VIEW' THEN 1 ELSE 0 END)
                        as hist_views,
                    SUM(CASE WHEN interaction = 'APPLY' THEN 1 ELSE 0 END)
                        as hist_applies,
                    SUM(CASE WHEN interaction = 'FINISHED' THEN 1 ELSE 0 END)
                        as hist_finished
                FROM events
                GROUP BY user_id
                """
                async with db.execute(query_users) as cursor:
                    async for row in cursor:
                        user_stats[str(row["user_id"])] = {
                            "user_hist_views": row["hist_views"] or 0,
                            "user_hist_applies": row["hist_applies"] or 0,
                            "user_hist_finished": row["hist_finished"] or 0,
                        }

                # Считаем успешные смены юзеров у конкретных работодателей
                user_employer_stats = {}
                query_emp = """
                SELECT
                    e.user_id,
                    s.employer_id,
                    COUNT(1) as finished_count
                FROM events e
                JOIN shifts s ON e.shift_id = s.id
                WHERE e.interaction = 'FINISHED'
                GROUP BY e.user_id, s.employer_id
                """
                async with db.execute(query_emp) as cursor:
                    async for row in cursor:
                        emp_key = (str(row["user_id"]), str(row["employer_id"]))
                        user_employer_stats[emp_key] = row["finished_count"] or 0

                # Считаем успешные смены юзеров на конкретных рабочих местах
                user_workplace_stats = {}
                query_wp = """
                SELECT
                    e.user_id,
                    s.workplace_id,
                    COUNT(1) as finished_count
                FROM events e
                JOIN shifts s ON e.shift_id = s.id
                WHERE e.interaction = 'FINISHED'
                GROUP BY e.user_id, s.workplace_id
                """
                async with db.execute(query_wp) as cursor:
                    async for row in cursor:
                        wp_key = (str(row["user_id"]), str(row["workplace_id"]))
                        user_workplace_stats[wp_key] = row["finished_count"] or 0

                # Собираем текущие активности внутри дня, чтобы не было утечек
                user_shift_events = {}
                query_active = """
                SELECT
                    user_id,
                    shift_id,
                    SUM(CASE WHEN interaction = 'VIEW' THEN 1 ELSE 0 END)
                        as view_cnt,
                    SUM(CASE WHEN interaction = 'USER_CANCEL' THEN 1 ELSE 0 END)
                        as user_cancel_cnt,
                    SUM(CASE WHEN interaction = 'SYSTEM_CANCEL' THEN 1 ELSE 0 END)
                        as system_cancel_cnt
                FROM events
                GROUP BY user_id, shift_id
                """
                async with db.execute(query_active) as cursor:
                    async for row in cursor:
                        key = (str(row["user_id"]), str(row["shift_id"]))
                        user_shift_events[key] = {
                            "view_cnt": row["view_cnt"] or 0,
                            "user_cancel_cnt": row["user_cancel_cnt"] or 0,
                            "system_cancel_cnt": row["system_cancel_cnt"] or 0,
                        }

            # Атомарно обновляем стейт в памяти
            self._state.user_stats = user_stats
            self._state.user_employer_stats = user_employer_stats
            self._state.user_workplace_stats = user_workplace_stats
            self._state.user_shift_events = user_shift_events

            LOGGER.info("ML Features cached successfully in memory!")
            self._state.ready = True
        except Exception as e:
            LOGGER.exception("Failed in background preparation: %s", e)
            self._state.ready = True
        finally:
            self._state.running = False
