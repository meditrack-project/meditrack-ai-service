import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import MedicationSummaryRequest, SymptomAnalysisRequest, InsightsRequest
from app.utils.auth import get_current_user_id
from app.utils.data_fetcher import fetch_all_data, summarize_health_data
from app.utils.ai_client import generate_ai_content, PRESET_QUESTIONS
from app.rate_limiter import check_rate_limits
from app.cache import (
    cache_get, cache_set,
    key_ai_weekly, key_ai_med, key_ai_symptom, key_ai_insight,
    TTL_AI_REPORT, TTL_AI_INSIGHT,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Insights"])


async def _process_ai_request(
    user_id: str,
    endpoint: str,
    cache_key: str,
    ttl: int,
    question: str,
    days: int,
):
    """Common AI request processing flow."""
    await check_rate_limits(user_id, endpoint)

    cached = await cache_get(cache_key)
    if cached is not None:
        return {"success": True, "data": {"response": cached}}

    raw_data = await fetch_all_data(user_id, days)
    summary = summarize_health_data(raw_data, days)

    try:
        ai_response = await generate_ai_content(summary, question, days)
    except Exception as e:
        logger.error(f"All AI providers failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "success": False,
                "message": "AI service temporarily unavailable. Try again later.",
            },
        )

    await cache_set(cache_key, ai_response, ttl)
    return {"success": True, "data": {"response": ai_response}}


@router.post("/weekly-report")
async def weekly_report(
    user_id: str = Depends(get_current_user_id),
):
    days = 7
    cache_key = key_ai_weekly(user_id)
    question = PRESET_QUESTIONS["weekly_report"]
    return await _process_ai_request(
        user_id, "weekly_report", cache_key, TTL_AI_REPORT, question, days
    )


@router.post("/medication-summary")
async def medication_summary(
    body: MedicationSummaryRequest = MedicationSummaryRequest(),
    user_id: str = Depends(get_current_user_id),
):
    days = body.days or 30
    cache_key = key_ai_med(user_id, days)
    question = PRESET_QUESTIONS["med_summary"].format(days=days)
    return await _process_ai_request(
        user_id, "med_summary", cache_key, TTL_AI_REPORT, question, days
    )


@router.post("/symptom-analysis")
async def symptom_analysis(
    body: SymptomAnalysisRequest = SymptomAnalysisRequest(),
    user_id: str = Depends(get_current_user_id),
):
    days = body.days or 14
    cache_key = key_ai_symptom(user_id, days)
    question = PRESET_QUESTIONS["symptom_analysis"].format(days=days)
    return await _process_ai_request(
        user_id, "symptom_analysis", cache_key, TTL_AI_REPORT, question, days
    )


@router.post("/insights")
async def insights(
    body: InsightsRequest,
    user_id: str = Depends(get_current_user_id),
):
    days = body.days or 7
    cache_key = key_ai_insight(user_id, body.question, days)
    return await _process_ai_request(
        user_id, "insights", cache_key, TTL_AI_INSIGHT, body.question, days
    )
