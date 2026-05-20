import asyncio
import logging

from zero import ZeroServer

from hackaton.service.app import HackatonRpcService
from hackaton.service.config import settings
from hackaton.service.db import init_db
from hackaton.service.prepare_manager import PrepareManager
from hackaton.service.repositories import Repository

LOGGER = logging.getLogger(__name__)

# Use thread mode to keep shared in-memory state (prepare/ready) consistent.
server = ZeroServer(host=settings.app_host, port=settings.app_port, use_threads=True)
service = HackatonRpcService(
    repository=Repository(db_path=settings.db_path),
    prepare=PrepareManager(settings.prepare_sleep_seconds, db_path=settings.db_path),
)


@server.register_rpc
async def user(payload: dict) -> dict:
    return await service.user(payload)


@server.register_rpc
async def user_stat(payload: dict = None) -> dict:
    return await service.user_stat(payload)


@server.register_rpc
async def event(payload: dict) -> dict:
    return await service.event(payload)


@server.register_rpc
async def event_stat(payload: dict = None) -> dict:
    return await service.event_stat(payload)


@server.register_rpc
async def shift(payload: dict) -> dict:
    return await service.shift(payload)


@server.register_rpc
async def shift_stat(payload: dict = None) -> dict:
    return await service.shift_stat(payload)


@server.register_rpc
async def prepare(payload: dict = None) -> dict:
    return await service.prepare(payload)


@server.register_rpc
async def ready(payload: dict = None) -> dict:
    return await service.ready(payload)


@server.register_rpc
async def predict(payload: dict) -> dict:
    return await service.predict(payload)


@server.register_rpc
async def health(payload: dict = None) -> dict:
    return await service.health(payload)


@server.register_rpc
async def metrics(payload: dict = None) -> dict:
    return await service.metrics(payload)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.info(
        "Starting service: host=%s port=%s db_path=%s prepare_sleep_seconds=%s",
        settings.app_host,
        settings.app_port,
        settings.db_path,
        settings.prepare_sleep_seconds,
    )
    asyncio.run(init_db())
    LOGGER.info("Database initialized successfully, starting RPC server")
    server.run()


if __name__ == "__main__":
    run()
