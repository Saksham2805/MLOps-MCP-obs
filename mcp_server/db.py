import os
import logging
from typing import Any, Dict, List, Optional

import asyncpg

log = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


def _get_dsn() -> str:
    dsn = os.environ.get("TIMESCALEDB_URL")
    if not dsn:
        raise RuntimeError(
            "TIMESCALEDB_URL environment variable is not set. "
            "Example: postgresql://user:pass@host:5432/llm_obs"
        )
    return dsn


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        dsn = _get_dsn()
        _pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        log.info("mcp_server/db: asyncpg pool created (dsn=%s)", dsn.split("@")[-1])
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        log.info("mcp_server/db: asyncpg pool closed")


async def fetch_rows(
    query: str,
    *args: Any,
) -> List[Dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        records = await conn.fetch(query, *args)
    return [dict(r) for r in records]


async def fetch_one(
    query: str,
    *args: Any,
) -> Optional[Dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        record = await conn.fetchrow(query, *args)
    return dict(record) if record is not None else None


async def execute(
    query: str,
    *args: Any,
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(query, *args)
