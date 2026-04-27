import os
import json
import logging
import hashlib
from typing import Any, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# TTL Constants
TTL_AI_REPORT = 3600     # 1 hour
TTL_AI_INSIGHT = 1800    # 30 min

_redis_client: Optional[redis.Redis] = None


async def get_redis() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.ping()
            return _redis_client
        except Exception:
            _redis_client = None

    try:
        _redis_client = redis.from_url(
            REDIS_URL,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        await _redis_client.ping()
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        _redis_client = None
        return None


async def cache_get(key: str) -> Any:
    try:
        client = await get_redis()
        if client is None:
            return None
        data = await client.get(key)
        if data is None:
            return None
        return json.loads(data)
    except Exception as e:
        logger.warning(f"Cache get failed for {key}: {e}")
        return None


async def cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        client = await get_redis()
        if client is None:
            return
        await client.setex(key, ttl, json.dumps(value, default=str))
    except Exception as e:
        logger.warning(f"Cache set failed for {key}: {e}")


async def cache_delete(key: str) -> None:
    try:
        client = await get_redis()
        if client is None:
            return
        await client.delete(key)
    except Exception as e:
        logger.warning(f"Cache delete failed for {key}: {e}")


async def cache_delete_pattern(pattern: str) -> None:
    try:
        client = await get_redis()
        if client is None:
            return
        keys = await client.keys(pattern)
        if keys:
            await client.delete(*keys)
    except Exception as e:
        logger.warning(f"Cache delete pattern failed for {pattern}: {e}")


# Key builders
def key_ai_weekly(user_id: str) -> str:
    return f"meditrack:ai:{user_id}:weekly_report"


def key_ai_med(user_id: str, days: int) -> str:
    return f"meditrack:ai:{user_id}:med_summary:{days}d"


def key_ai_symptom(user_id: str, days: int) -> str:
    return f"meditrack:ai:{user_id}:symptom_analysis:{days}d"


def key_ai_insight(user_id: str, question: str, days: int) -> str:
    q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
    return f"meditrack:ai:{user_id}:insight:{q_hash}:{days}d"
