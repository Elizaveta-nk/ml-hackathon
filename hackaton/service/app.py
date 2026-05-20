from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError

from hackaton.service.dto import (
    BatchEventsRequest,
    BatchShiftsRequest,
    BatchUsersRequest,
    PredictRequest,
)
from hackaton.service.prepare_manager import PrepareManager
from hackaton.service.repositories import Repository

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency of API requests", ["endpoint"])
LOGGER = logging.getLogger(__name__)


def _find_model_path() -> Path:
    for path in [
        Path("artifacts/train_run/model.pkl"),
        Path("artifacts/eval_run/model.pkl"),
        Path("artifacts/model.pkl"),
        Path("model.pkl"),
    ]:
        if path.exists():
            return path
    for p in Path("artifacts").glob("**/model.pkl"):
        return p
    raise FileNotFoundError("Could not locate model.pkl")


class HackatonRpcService:
    def __init__(self, repository: Repository, prepare: PrepareManager) -> None:
        self.repository = repository
        self.prepare_manager = prepare

        # Если в тестах db_path не был передан в PrepareManager,
        # мы берем его напрямую из репозитория
        if not getattr(self.prepare_manager, "db_path", None):
            self.prepare_manager.db_path = repository.db_path

        try:
            model_path = _find_model_path()
            LOGGER.info("Loading model from %s", model_path)
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            LOGGER.warning("Could not load model at startup: %s. Will lazy load.", e)
            self.model = None

    async def user(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("user").inc()
        with REQUEST_LATENCY.labels("user").time():
            request = BatchUsersRequest.model_validate(payload)
            LOGGER.info("RPC user called, batch_size=%s", len(request.items))
            accepted = await self.repository.upsert_users(request.items)
            return {"accepted": accepted}

    async def user_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("user_stat").inc()
        with REQUEST_LATENCY.labels("user_stat").time():
            LOGGER.info("RPC user_stat called")
            return {"count": await self.repository.count_table("users")}

    async def event(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("event").inc()
        with REQUEST_LATENCY.labels("event").time():
            request = BatchEventsRequest.model_validate(payload)
            LOGGER.info("RPC event called, batch_size=%s", len(request.items))
            accepted = await self.repository.insert_events(request.items)
            return {"accepted": accepted}

    async def event_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("event_stat").inc()
        with REQUEST_LATENCY.labels("event_stat").time():
            LOGGER.info("RPC event_stat called")
            return {"count": await self.repository.count_table("events")}

    async def shift(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("shift").inc()
        with REQUEST_LATENCY.labels("shift").time():
            request = BatchShiftsRequest.model_validate(payload)
            LOGGER.info("RPC shift called, batch_size=%s", len(request.items))
            accepted = await self.repository.upsert_shifts(request.items)
            return {"accepted": accepted}

    async def shift_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("shift_stat").inc()
        with REQUEST_LATENCY.labels("shift_stat").time():
            LOGGER.info("RPC shift_stat called")
            return {"count": await self.repository.count_table("shifts")}

    async def prepare(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("prepare").inc()
        with REQUEST_LATENCY.labels("prepare").time():
            LOGGER.info("RPC prepare called")
            started = await self.prepare_manager.start()
            if not started:
                return {"status": "already_running", "status_code": 409}
            return {"status": "started", "status_code": 200}

    async def ready(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("ready").inc()
        with REQUEST_LATENCY.labels("ready").time():
            LOGGER.info("RPC ready called")
            if not self.prepare_manager.ready:
                return {"ready": False, "status_code": 425}
            return {"ready": True, "status_code": 200}

    async def predict(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("predict").inc()
        with REQUEST_LATENCY.labels("predict").time():
            LOGGER.info("RPC predict called")
            if not self.prepare_manager.ready:
                return {"user_ids": [], "status_code": 503, "detail": "model is in prepare state"}
            try:
                request = PredictRequest.model_validate(payload)
            except ValidationError as exc:
                return {"user_ids": [], "status_code": 422, "detail": str(exc)}

            shift = request.shift

            if self.model is None:
                try:
                    model_path = _find_model_path()
                    with open(model_path, "rb") as f:
                        self.model = pickle.load(f)
                except Exception as e:
                    LOGGER.error("Failed to lazy load model: %s", e)
                    return {"user_ids": [], "status_code": 500, "detail": "model not loaded"}

            # Безопасное получение словарей (чтобы не упасть, если тесты их замокали)
            user_shift_events = getattr(self.prepare_manager, "user_shift_events", {}) or {}
            user_stats = getattr(self.prepare_manager, "user_stats", {}) or {}
            user_employer_stats = getattr(self.prepare_manager, "user_employer_stats", {}) or {}
            user_workplace_stats = getattr(self.prepare_manager, "user_workplace_stats", {}) or {}

            # Находим тех, кто активен по этой смене прямо сейчас
            active_user_ids = {
                uid for (uid, sid) in user_shift_events.keys() if sid == str(shift.id)
            }

            # Получаем обычный список кандидатов
            candidates = await self.repository.find_candidates_for_ml(
                location_id=request.shift.location_id, need_mk=request.shift.need_mk, limit=3000
            )

            # Добавляем активных пользователей в начало списка, если их там нет
            existing_ids = {str(c["id"]) for c in candidates}
            for uid in active_user_ids:
                if uid not in existing_ids:
                    candidates.append(
                        {
                            "id": uid,
                            "location_id": shift.location_id,
                            "is_strict_location": 0,
                            "has_mk": 1,
                        }
                    )

            # Если кандидатов вообще нет, отдаем фоллбеки сразу
            if not candidates:
                fallback_ids = await self.repository.fallback_candidates(limit=request.limit)
                return {"user_ids": fallback_ids, "status_code": 200}

            feature_rows = []
            user_ids = []

            for user in candidates:
                user_id = str(user["id"])
                user_ids.append(user_id)

                # Исторические фичи из кэша
                u_stats = user_stats.get(user_id, {}) or {}
                user_hist_views = u_stats.get("user_hist_views", 0)
                user_hist_applies = u_stats.get("user_hist_applies", 0)
                user_hist_finished = u_stats.get("user_hist_finished", 0)

                user_finished_employer = user_employer_stats.get(
                    (user_id, str(shift.employer_id)), 0
                )
                user_finished_workplace = user_workplace_stats.get(
                    (user_id, str(shift.workplace_id)), 0
                )

                # Активности внутри текущего дня
                active = user_shift_events.get((user_id, str(shift.id)), {}) or {}
                view_cnt = active.get("view_cnt", 0)
                user_cancel_cnt = active.get("user_cancel_cnt", 0)
                system_cancel_cnt = active.get("system_cancel_cnt", 0)

                # Собираем словарь фичей
                row = {
                    "has_mk": int(user["has_mk"]),
                    "is_strict_location": int(user["is_strict_location"]),
                    "need_mk": int(shift.need_mk),
                    "id_differential": int(shift.id_differential),
                    "hours": float(shift.hours) if shift.hours is not None else 0.0,
                    "reward": float(shift.reward) if shift.reward is not None else 0.0,
                    "capacity": float(shift.capacity) if shift.capacity is not None else 0.0,
                    "location_match": int(user["location_id"] == shift.location_id),
                    "need_mk_match": int(shift.need_mk == user["has_mk"]),
                    "view_cnt": int(view_cnt),
                    "user_cancel_cnt": int(user_cancel_cnt),
                    "system_cancel_cnt": int(system_cancel_cnt),
                    "user_hist_views": int(user_hist_views),
                    "user_hist_applies": int(user_hist_applies),
                    "user_hist_finished": int(user_hist_finished),
                    "user_finished_employer": int(user_finished_employer),
                    "user_finished_workplace": int(user_finished_workplace),
                    "task_type": str(shift.task_type),
                }
                feature_rows.append(row)

            # Строгий порядок колонок
            feature_columns = [
                "has_mk",
                "is_strict_location",
                "need_mk",
                "id_differential",
                "hours",
                "reward",
                "capacity",
                "location_match",
                "need_mk_match",
                "view_cnt",
                "user_cancel_cnt",
                "system_cancel_cnt",
                "user_hist_views",
                "user_hist_applies",
                "user_hist_finished",
                "user_finished_employer",
                "user_finished_workplace",
                "task_type",
            ]
            df_features = pd.DataFrame(feature_rows)[feature_columns]

            # Предсказываем скоры
            probabilities = self.model.predict_proba(df_features)[:, 1]

            # Сортируем кандидатов по убыванию вероятности отклика
            ranked = sorted(zip(user_ids, probabilities), key=lambda x: x[1], reverse=True)
            top_user_ids = [uid for uid, prob in ranked[: request.limit]]

            # Дозаполняем фоллбеками, если кандидатов не хватило
            if len(top_user_ids) < request.limit:
                existing = set(top_user_ids)
                fallbacks = await self.repository.fallback_candidates(limit=request.limit * 2)
                for f_id in fallbacks:
                    if f_id not in existing:
                        top_user_ids.append(f_id)
                        if len(top_user_ids) == request.limit:
                            break

            return {"user_ids": top_user_ids, "status_code": 200}

    async def health(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("health").inc()
        LOGGER.info("RPC health called")
        return {"status": "ok", "status_code": 200}

    async def metrics(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("metrics").inc()
        LOGGER.info("RPC metrics called")
        return {
            "content_type": CONTENT_TYPE_LATEST,
            "payload": generate_latest().decode("utf-8"),
            "status_code": 200,
        }
