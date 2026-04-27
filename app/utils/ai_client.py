import os
import logging
from typing import Dict, Any
import httpx
import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """You are MediTrack AI, a personal health assistant.
Analyze the health summary below and respond directly to the user.
Rules:
- DO NOT use introductory pleasantries (e.g., "Hello there", "I'm MediTrack AI", "Based on your summary").
- Start your response IMMEDIATELY with the requested information.
- Warm, supportive, concise (max 200 words)
- Bullet points where helpful
- Never diagnose or recommend medications
- Base answers only on provided data
- End with: "Always consult your doctor for medical decisions."
"""

PRESET_QUESTIONS = {
    "weekly_report": "Generate a complete weekly health summary covering medication adherence, symptoms, mood, energy, and doctor visits. Give 3 specific actionable recommendations for next week.",
    "med_summary": "Analyze my medication adherence for the past {days} days. Tell me how I am doing overall, which medications I miss most, and give practical tips to improve consistency.",
    "symptom_analysis": "Analyze my symptoms, mood, and energy for the past {days} days. Identify trends or patterns and suggest what might be contributing to them.",
}


def build_health_summary(summary: Dict[str, Any]) -> str:
    """Build the health summary section of the prompt from summary dict."""
    lines = [f"HEALTH SUMMARY — Last {summary['period_days']} days\n"]

    # Medications
    meds = summary.get("medications", {})
    lines.append(f"MEDICATIONS ({meds.get('total_count', 0)} total):")
    for m in meds.get("list", []):
        lines.append(f"  • {m['name']} {m['dosage']} — {m['frequency']}")

    today = meds.get("today", {})
    lines.append(f"\nTODAY: {today.get('taken', 0)}/{today.get('total', 0)} taken ({today.get('taken_percent', 0)}%)")
    for s in today.get("status_list", []):
        icon = "✅" if s["taken"] else "⬜"
        lines.append(f"  {icon} {s['name']}")

    adh = meds.get("adherence", {})
    lines.append(f"\nADHERENCE:")
    lines.append(f"  Overall: {adh.get('overall_avg', 0)}%")
    best = adh.get("best", {})
    worst = adh.get("worst", {})
    lines.append(f"  Best: {best.get('name', 'N/A')} — {best.get('percent', 0)}%")
    lines.append(f"  Worst: {worst.get('name', 'N/A')} — {worst.get('percent', 0)}%")
    for pm in adh.get("per_medication", []):
        lines.append(f"  • {pm['name']}: {pm['percent']}%")

    # Symptoms
    sym = summary.get("symptoms", {})
    lines.append(f"\nSYMPTOMS (last {summary['period_days']} days):")
    lines.append(f"  Logged: {sym.get('days_logged', 0)}/{summary['period_days']} days")
    lines.append(f"  Avg mood: {sym.get('avg_mood', 0)}/10 ({sym.get('mood_trend', 'stable')})")
    lines.append(f"  Avg energy: {sym.get('avg_energy', 0)}/10")
    lines.append(f"  Avg severity: {sym.get('avg_severity', 0)}/10")

    top_symptoms = sym.get("top_symptoms", [])
    if top_symptoms:
        symptom_str = ", ".join([f"{s['symptom']}({s['count']})" for s in top_symptoms])
        lines.append(f"  Top symptoms: {symptom_str}")

    best_day = sym.get("best_day", {})
    worst_day = sym.get("worst_day", {})
    lines.append(f"  Best day: {best_day.get('date', 'N/A')} — mood {best_day.get('mood', 0)}/10")
    lines.append(f"  Worst day: {worst_day.get('date', 'N/A')} — mood {worst_day.get('mood', 0)}/10")

    # Upcoming follow-ups
    upcoming = summary.get("upcoming_followups", [])
    lines.append(f"\nUPCOMING FOLLOW-UPS:")
    if not upcoming:
        lines.append("  None scheduled")
    else:
        for v in upcoming:
            lines.append(f"  • {v['doctor']} ({v['specialty']}) — {v['date']} (in {v['days_until']} days)")

    return "\n".join(lines)


def guard_prompt_size(health_summary_text: str, summary: Dict[str, Any]) -> str:
    """Guard prompt size to stay under ~900 tokens."""
    estimated_tokens = len(health_summary_text) / 4

    if estimated_tokens <= 900:
        return health_summary_text

    # Remove best_day and worst_day
    lines = health_summary_text.split("\n")
    lines = [l for l in lines if "Best day:" not in l and "Worst day:" not in l]
    health_summary_text = "\n".join(lines)

    # Reduce top symptoms to 3
    sym = summary.get("symptoms", {})
    top_symptoms = sym.get("top_symptoms", [])[:3]

    estimated_tokens = len(health_summary_text) / 4
    if estimated_tokens <= 900:
        return health_summary_text

    # Remove per_medication list
    new_lines = []
    in_per_med = False
    for line in health_summary_text.split("\n"):
        if "ADHERENCE:" in line:
            in_per_med = False
        if in_per_med and line.strip().startswith("•"):
            continue
        if "Worst:" in line:
            in_per_med = True
        new_lines.append(line)

    return "\n".join(new_lines)


async def _call_gemini(final_prompt: str) -> str:
    """Call Google Gemini API."""
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not configured")
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "max_output_tokens": 400,
            "temperature": 0.4,
            "top_p": 0.9,
        },
    )
    response = model.generate_content(final_prompt)
    return response.text


async def _call_groq(final_prompt: str) -> str:
    """Call Groq API using HTTPX."""
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "user", "content": final_prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 400,
            },
            timeout=15.0
        )
        if response.status_code != 200:
            logger.error(f"Groq Error Body: {response.text}")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def _call_huggingface(final_prompt: str) -> str:
    """Call HuggingFace Inference API using HTTPX."""
    if not HUGGINGFACE_API_KEY:
        raise Exception("HUGGINGFACE_API_KEY not configured")

    # Using Zephyr as it's highly reliable on HF free tier
    model_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    
    # Format prompt for Zephyr
    hf_prompt = f"<|system|>\nYou are MediTrack AI.</s>\n<|user|>\n{final_prompt}</s>\n<|assistant|>\n"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            model_url,
            headers={
                "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": hf_prompt,
                "parameters": {
                    "max_new_tokens": 400,
                    "temperature": 0.4,
                    "return_full_text": False
                }
            },
            timeout=20.0
        )
        if response.status_code != 200:
            logger.error(f"HF Error Body: {response.text}")
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "").strip()
        return str(data)


async def generate_ai_content(summary: Dict[str, Any], user_question: str, days: int) -> str:
    """Build prompt and execute fallback cascade: Gemini -> Groq -> HuggingFace."""
    
    health_summary_text = build_health_summary(summary)
    health_summary_text = guard_prompt_size(health_summary_text, summary)

    final_prompt = f"{SYSTEM_INSTRUCTION}\n\n{health_summary_text}\n\nUSER REQUEST:\n{user_question}"

    # 1. Primary: Gemini
    try:
        logger.info("Attempting Gemini API...")
        return await _call_gemini(final_prompt)
    except Exception as e:
        logger.warning(f"Gemini API failed or quota exceeded: {e}. Falling back to Groq.")

    # 2. Secondary: Groq
    try:
        logger.info("Attempting Groq API...")
        return await _call_groq(final_prompt)
    except Exception as e:
        logger.warning(f"Groq API failed: {e}. Falling back to HuggingFace.")

    # 3. Tertiary: HuggingFace
    try:
        logger.info("Attempting HuggingFace API...")
        return await _call_huggingface(final_prompt)
    except Exception as e:
        logger.error(f"HuggingFace API failed: {e}. All AI providers exhausted.")
    
    raise Exception("All AI providers (Gemini, Groq, HuggingFace) failed.")
