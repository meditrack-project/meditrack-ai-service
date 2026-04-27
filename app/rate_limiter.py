import logging
from datetime import datetime, timezone
from typing import Tuple

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.cache import get_redis

logger = logging.getLogger(__name__)

# Per user limits per hour
USER_LIMITS = {
    "weekly_report": 5,
    "med_summary": 5,
    "symptom_analysis": 5,
    "insights": 10,
}

# Global limits
GLOBAL_MINUTE_LIMIT = 10
GLOBAL_DAY_LIMIT = 1000


async def check_global_limits(redis_client) -> Tuple[bool, str]:
    """Check global rate limits. Returns (allowed, reason)."""
    if redis_client is None:
        return True, ""

    try:
        now = datetime.now(timezone.utc)

        # Check global minute limit
        minute_key = f"meditrack:ratelimit:global:minute:{now.strftime('%Y-%m-%d-%H-%M')}"
        minute_count = await redis_client.get(minute_key)
        if minute_count and int(minute_count) >= GLOBAL_MINUTE_LIMIT:
            return False, "global_minute"

        # Check global day limit
        day_key = f"meditrack:ratelimit:global:day:{now.strftime('%Y-%m-%d')}"
        day_count = await redis_client.get(day_key)
        if day_count and int(day_count) >= GLOBAL_DAY_LIMIT:
            return False, "global_day"

        return True, ""
    except Exception as e:
        logger.warning(f"Global rate limit check failed: {e}")
        return True, ""


async def check_user_limit(redis_client, user_id: str, endpoint: str) -> Tuple[bool, int, int]:
    """Check per-user rate limit. Returns (allowed, current_count, limit)."""
    if redis_client is None:
        return True, 0, USER_LIMITS.get(endpoint, 5)

    limit = USER_LIMITS.get(endpoint, 5)

    try:
        now = datetime.now(timezone.utc)
        user_key = f"meditrack:ratelimit:user:{user_id}:{endpoint}:{now.strftime('%Y-%m-%d-%H')}"
        current = await redis_client.get(user_key)
        current_count = int(current) if current else 0

        if current_count >= limit:
            return False, current_count, limit

        return True, current_count, limit
    except Exception as e:
        logger.warning(f"User rate limit check failed: {e}")
        return True, 0, limit


async def increment_counters(redis_client, user_id: str, endpoint: str):
    """Increment all rate limit counters."""
    if redis_client is None:
        return

    try:
        now = datetime.now(timezone.utc)

        # User counter
        user_key = f"meditrack:ratelimit:user:{user_id}:{endpoint}:{now.strftime('%Y-%m-%d-%H')}"
        pipe = redis_client.pipeline()
        pipe.incr(user_key)
        pipe.expire(user_key, 3600)

        # Global minute counter
        minute_key = f"meditrack:ratelimit:global:minute:{now.strftime('%Y-%m-%d-%H-%M')}"
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)

        # Global day counter
        day_key = f"meditrack:ratelimit:global:day:{now.strftime('%Y-%m-%d')}"
        pipe.incr(day_key)
        pipe.expire(day_key, 86400)

        await pipe.execute()
    except Exception as e:
        logger.warning(f"Rate limit increment failed: {e}")


def make_rate_limit_error(reason: str, current: int = 0, limit: int = 0):
    """Create appropriate 429 error response."""
    if reason == "global_minute":
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "success": False,
                "message": "System is experiencing high demand. Please wait a moment and try again.",
                "retry_after": "60 seconds",
            },
        )
    elif reason == "global_day":
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "success": False,
                "message": "Daily AI request limit reached. Service resets at midnight UTC.",
                "retry_after": "86400 seconds",
            },
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "success": False,
                "message": f"You have reached the limit of {limit} requests per hour for this feature. Please try again later.",
                "retry_after": "3600 seconds",
                "limit": limit,
                "current": current,
            },
        )


async def check_rate_limits(user_id: str, endpoint: str):
    """Full rate limit check sequence. Call before cache check."""
    redis_client = await get_redis()

    # 1. Check global minute
    allowed, reason = await check_global_limits(redis_client)
    if not allowed:
        make_rate_limit_error(reason)

    # 2. Check per user
    allowed, current, limit = await check_user_limit(redis_client, user_id, endpoint)
    if not allowed:
        make_rate_limit_error("user", current, limit)

    # 3. Increment all counters
    await increment_counters(redis_client, user_id, endpoint)
