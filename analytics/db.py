
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.pool
import psycopg2.extras

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level connection pool (lazy-initialised)
# ---------------------------------------------------------------------------
_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def _get_dsn() -> str:
    dsn = os.environ.get("TIMESCALEDB_URL")
    if not dsn:
        raise RuntimeError(
            "TIMESCALEDB_URL environment variable is not set. "
            "Example: postgresql://user:pass@host:5432/llm_obs"
        )
    return dsn


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Return the module-level psycopg2 connection pool, creating it once."""
    global _pool
    if _pool is None:
        dsn = _get_dsn()
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=dsn,
        )
        log.info("analytics/db: psycopg2 pool created (dsn=%s)", dsn.split("@")[-1])
    return _pool


def close_pool() -> None:
    """Close all connections in the pool (call at process shutdown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        log.info("analytics/db: pool closed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_rows(
    query: str,
    params: Optional[Tuple[Any, ...]] = None,
    pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None,
) -> List[Dict[str, Any]]:
    """
    Execute *query* with *params* and return all result rows as a list of
    plain dicts keyed by column name.

    Parameters
    ----------
    query  : SQL string (use %s placeholders).
    params : Optional tuple of bind values.
    pool   : Override the module-level pool (useful for testing).

    Returns
    -------
    List of dicts, one per row.  Empty list when no rows are returned.
    """
    effective_pool = pool if pool is not None else get_pool()
    conn = effective_pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        # fetchall() with RealDictCursor returns RealDictRow objects; convert to
        # plain dicts so callers don't need to worry about the specialised type.
        return [dict(row) for row in rows]
    except Exception:
        conn.rollback()
        raise
    finally:
        effective_pool.putconn(conn)


def execute(
    query: str,
    params: Optional[Tuple[Any, ...]] = None,
    pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None,
) -> None:
    """
    Execute a write statement (*INSERT* / *UPDATE* / *DELETE* / DDL) and commit.

    Parameters
    ----------
    query  : SQL string (use %s placeholders).
    params : Optional tuple of bind values.
    pool   : Override the module-level pool (useful for testing).
    """
    effective_pool = pool if pool is not None else get_pool()
    conn = effective_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        effective_pool.putconn(conn)


def read_one(
    query: str,
    params: Optional[Tuple[Any, ...]] = None,
    pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper: execute *query* and return only the first row as a
    dict, or ``None`` if the result set is empty.
    """
    rows = read_rows(query, params, pool=pool)
    return rows[0] if rows else None
