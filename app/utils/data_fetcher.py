import os
import logging
import asyncio
from typing import Any, Dict, List
from collections import Counter
from datetime import date, timedelta

import httpx

logger = logging.getLogger(__name__)

MEDICAL_SERVICE_URL = os.getenv("MEDICAL_SERVICE_URL", "http://medical-service:4002")
HEALTH_SERVICE_URL = os.getenv("HEALTH_SERVICE_URL", "http://health-service:4003")


async def _fetch(client: httpx.AsyncClient, url: str) -> Any:
    """Fetch data from a service endpoint."""
    try:
        response = await client.get(url, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return []


async def fetch_all_data(user_id: str, days: int = 7) -> Dict[str, Any]:
    """Fetch data from medical and health services in parallel."""
    headers = {"X-User-ID": user_id}

    async with httpx.AsyncClient(headers=headers) as client:
        med_urls = [
            f"{MEDICAL_SERVICE_URL}/api/medications",
            f"{MEDICAL_SERVICE_URL}/api/medications/logs/today",
            f"{MEDICAL_SERVICE_URL}/api/medications/adherence?days={days}",
        ]
        health_urls = [
            f"{HEALTH_SERVICE_URL}/api/symptoms?days={days}",
            f"{HEALTH_SERVICE_URL}/api/visits/upcoming",
        ]

        all_urls = med_urls + health_urls
        results = await asyncio.gather(
            *[_fetch(client, url) for url in all_urls],
            return_exceptions=True,
        )

        results = [r if not isinstance(r, Exception) else [] for r in results]

        return {
            "medications": results[0] if isinstance(results[0], list) else [],
            "today_logs": results[1] if isinstance(results[1], dict) else {},
            "adherence": results[2] if isinstance(results[2], dict) else {},
            "symptom_logs": results[3] if isinstance(results[3], list) else [],
            "upcoming_visits": results[4] if isinstance(results[4], list) else [],
        }


def summarize_health_data(raw_data: Dict[str, Any], days: int) -> Dict[str, Any]:
    """Process raw API data into structured summary for AI prompt."""

    medications_raw = raw_data.get("medications", [])
    med_list = []
    for med in medications_raw[:10]:
        med_list.append({
            "name": med.get("name", "Unknown"),
            "dosage": med.get("dosage", ""),
            "frequency": med.get("frequency", ""),
        })

    today_data = raw_data.get("today_logs", {})
    today_total = today_data.get("total", 0)
    today_taken = today_data.get("taken_count", 0)
    today_percent = today_data.get("adherence_today", 0.0)
    today_logs = today_data.get("logs", [])
    status_list = []
    for log in today_logs:
        med_info = log.get("medication", {})
        status_list.append({
            "name": med_info.get("name", "Unknown"),
            "taken": log.get("taken", False),
        })

    adherence_raw = raw_data.get("adherence", {})
    overall_avg = adherence_raw.get("overall_avg", 0.0)
    best = adherence_raw.get("best", {"name": "N/A", "percent": 0.0})
    worst = adherence_raw.get("worst", {"name": "N/A", "percent": 0.0})
    per_med = adherence_raw.get("per_medication", [])[:10]
    per_med_summary = [{"name": m.get("name", ""), "percent": m.get("percent", 0)} for m in per_med]

    symptom_logs = raw_data.get("symptom_logs", [])
    total_entries = len(symptom_logs)

    dates_logged = set()
    moods = []
    energies = []
    severities = []
    all_symptoms_list = []

    for log in symptom_logs:
        dates_logged.add(log.get("date", ""))
        if log.get("mood") is not None:
            moods.append(log["mood"])
        if log.get("energy") is not None:
            energies.append(log["energy"])
        if log.get("severity") is not None:
            severities.append(log["severity"])
        symptoms = log.get("symptoms", [])
        all_symptoms_list.extend(symptoms)

    days_logged = len(dates_logged)
    days_no_log = days - days_logged

    avg_mood = round(sum(moods) / len(moods), 1) if moods else 0.0
    avg_energy = round(sum(energies) / len(energies), 1) if energies else 0.0
    avg_severity = round(sum(severities) / len(severities), 1) if severities else 0.0

    symptom_counter = Counter(all_symptoms_list)
    top_symptoms = [{"symptom": s, "count": c} for s, c in symptom_counter.most_common(5)]

    if len(moods) >= 2:
        half = len(moods) // 2
        first_half_avg = sum(moods[:half]) / half
        second_half_avg = sum(moods[half:]) / (len(moods) - half)
        if second_half_avg > first_half_avg + 0.5:
            mood_trend = "improving"
        elif second_half_avg < first_half_avg - 0.5:
            mood_trend = "declining"
        else:
            mood_trend = "stable"
    else:
        mood_trend = "stable"

    best_day = {"date": "N/A", "mood": 0}
    worst_day = {"date": "N/A", "mood": 10}
    for log in symptom_logs:
        m = log.get("mood")
        if m is not None:
            if m > best_day["mood"]:
                best_day = {"date": log.get("date", "N/A"), "mood": m}
            if m < worst_day["mood"]:
                worst_day = {"date": log.get("date", "N/A"), "mood": m}

    if not moods:
        best_day = {"date": "N/A", "mood": 0}
        worst_day = {"date": "N/A", "mood": 0}

    upcoming_raw = raw_data.get("upcoming_visits", [])
    upcoming = []
    for visit in upcoming_raw[:3]:
        follow_up_str = visit.get("follow_up", "")
        try:
            follow_up_date = date.fromisoformat(follow_up_str)
            days_until = (follow_up_date - date.today()).days
            formatted_date = follow_up_date.strftime("%b %d, %Y")
        except (ValueError, TypeError):
            days_until = 0
            formatted_date = follow_up_str

        upcoming.append({
            "doctor": visit.get("doctor_name", "Unknown"),
            "specialty": visit.get("specialty", "General"),
            "date": formatted_date,
            "days_until": days_until,
        })

    return {
        "period_days": days,
        "medications": {
            "total_count": len(medications_raw),
            "list": med_list,
            "today": {
                "total": today_total,
                "taken": today_taken,
                "taken_percent": today_percent,
                "status_list": status_list,
            },
            "adherence": {
                "overall_avg": overall_avg,
                "best": best,
                "worst": worst,
                "per_medication": per_med_summary,
            },
        },
        "symptoms": {
            "total_entries": total_entries,
            "days_logged": days_logged,
            "days_no_log": days_no_log,
            "avg_mood": avg_mood,
            "avg_energy": avg_energy,
            "avg_severity": avg_severity,
            "mood_trend": mood_trend,
            "top_symptoms": top_symptoms,
            "best_day": best_day,
            "worst_day": worst_day,
        },
        "upcoming_followups": upcoming,
    }
