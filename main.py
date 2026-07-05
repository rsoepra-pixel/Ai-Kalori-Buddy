"""
Glucofusion Gemini — Backend API
================================
A FastAPI service acting as a live "bio-optimizer" that connects the
Glucofusion Gemini frontend to the Google Gemini API.

Run locally:
    pip install -r requirements.txt
    cp .env.example .env   # then add your GEMINI_API_KEY
    uvicorn main:app --reload --port 8000
"""

import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import google.generativeai as genai

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("glucofusion")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning(
        "GEMINI_API_KEY is not set. The API will start, but LLM calls will "
        "fail until you provide a key in your .env file."
    )

# Deterministic-leaning generation config: punchy, focused, coach-like.
GENERATION_CONFIG = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 512,
}

# --------------------------------------------------------------------------- #
# Modular Prompt Architecture
# --------------------------------------------------------------------------- #
SYSTEM_PERSONA = (
    "You are the Elite Diet Kenyang AI Coach. Your tone is empathetic but "
    "fiercely disciplined, acting as a high-level bio-optimizer. You speak in "
    "concise, punchy sentences. You do not give generic diet advice. Focus on "
    "the science of glucose spikes, food sequencing, and cellular clearance. "
    "Respond strictly in the user's requested language."
)

# Human-readable descriptions for each diet framework the user can select.
TRACK_FRAMEWORKS = {
    "low-gi": (
        "The user follows the LOW-GLYCEMIC INDEX framework. Prioritize foods "
        "that blunt glucose spikes: high-fiber vegetables, legumes, whole "
        "intact grains, and protein-forward sequencing (fiber and protein "
        "before starch)."
    ),
    "keto-asian": (
        "The user follows the KETO-ASIAN framework. Prioritize very low-carb, "
        "high-healthy-fat Asian-fusion foods: fatty fish, tofu/tempeh, coconut, "
        "leafy greens, eggs, and fermented sides — while keeping insulin low."
    ),
}


def _resolve_language(language: str) -> str:
    """Normalize the requested language into an explicit instruction."""
    lang = (language or "en").strip().lower()
    if lang in ("id", "indonesian", "bahasa", "bahasa indonesia"):
        return "Bahasa Indonesia"
    return "English"


def build_chat_prompt(message: str, language: str, track: str) -> str:
    """Compose the full prompt for the general wellness chat endpoint."""
    language_name = _resolve_language(language)
    framework = TRACK_FRAMEWORKS.get(
        (track or "").strip().lower(),
        "The user has not locked a specific diet framework yet.",
    )
    return (
        f"{SYSTEM_PERSONA}\n\n"
        f"### ACTIVE DIET FRAMEWORK\n{framework}\n\n"
        f"### RESPONSE LANGUAGE\nRespond strictly in {language_name}.\n\n"
        f"### TASK\n"
        f"Contextually suggest foods and tactics from the active diet "
        f"framework above, then directly address the user's message. Stay "
        f"punchy and science-driven.\n\n"
        f"### USER MESSAGE\n{message}"
    )


def build_sos_prompt(craving: str, category: str, language: str) -> str:
    """Compose the aggressive intervention prompt for the S.O.S endpoint."""
    language_name = _resolve_language(language)
    cat = (category or "").strip().lower()
    if "sugar" in cat:
        physiology = (
            "a HIGH-SUGAR craving — explain the rapid glucose spike, the "
            "insulin surge, and the reactive hypoglycemia crash that deepens "
            "insulin resistance."
        )
    else:
        physiology = (
            "a HIGH-FAT craving — explain how excess saturated/processed fat "
            "combined with any carbs drives lipotoxicity and blunts insulin "
            "sensitivity at the cellular level."
        )

    return (
        f"{SYSTEM_PERSONA}\n\n"
        f"### RESPONSE LANGUAGE\nRespond strictly in {language_name}.\n\n"
        f"### CRAVING INTERCEPTION PROTOCOL\n"
        f"The user is in a craving emergency. Their craving: \"{craving}\".\n"
        f"This is {physiology}\n\n"
        f"### REQUIRED STRUCTURE (keep the ENTIRE response under 4 sentences)\n"
        f"1. Acknowledge the specific craving with empathy.\n"
        f"2. Explain the physiological cost (insulin resistance) for this "
        f"category.\n"
        f"3. Offer ONE strategic Asian-fusion alternative.\n"
        f"4. Issue ONE immediate physical micro-task (e.g., 60 seconds of deep "
        f"breathing) to delay gratification.\n"
        f"Be fierce, disciplined, and motivating."
    )


# --------------------------------------------------------------------------- #
# Pydantic Models
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User chat message")
    language: str = Field(default="en", description="'en' or 'id'")
    track: str = Field(default="low-gi", description="'low-gi' or 'keto-asian'")


class SosRequest(BaseModel):
    craving: str = Field(..., min_length=1, description="What the user craves")
    category: str = Field(..., description="'High Sugar' or 'High Fat'")
    language: str = Field(default="en", description="'en' or 'id'")


class AiResponse(BaseModel):
    reply: str


# --------------------------------------------------------------------------- #
# FastAPI App + CORS
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Glucofusion Gemini API",
    description="Live bio-optimizer backend powered by Google Gemini.",
    version="1.0.0",
)

# Explicit local origins. A wildcard ("*") is invalid when credentials are
# enabled — browsers reject that combination — so we list the dev origins the
# frontend is actually served from (file://-opened pages send `Origin: null`).
ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "null",  # pages opened directly from the filesystem (file://)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _generate(prompt: str) -> str:
    """Call Gemini and return clean text, raising HTTP errors on failure."""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server misconfiguration: GEMINI_API_KEY is not set.",
        )
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG,
        )
        response = model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise ValueError("Empty response from model.")
        return text
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — surface a clean 502 to the client.
        logger.exception("Gemini generation failed")
        raise HTTPException(
            status_code=502,
            detail=f"Upstream AI error: {exc}",
        ) from exc


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {"status": "online", "service": "Glucofusion Gemini", "model": MODEL_NAME}


@app.get("/api/health")
def health():
    return {"ok": True, "key_configured": bool(GEMINI_API_KEY)}


@app.post("/api/chat", response_model=AiResponse)
def chat(req: ChatRequest):
    logger.info("CHAT | lang=%s | track=%s", req.language, req.track)
    prompt = build_chat_prompt(req.message, req.language, req.track)
    return AiResponse(reply=_generate(prompt))


@app.post("/api/sos", response_model=AiResponse)
def sos(req: SosRequest):
    logger.info("SOS | lang=%s | category=%s", req.language, req.category)
    prompt = build_sos_prompt(req.craving, req.category, req.language)
    return AiResponse(reply=_generate(prompt))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
