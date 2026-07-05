# Glucofusion Gemini · Live Bio-Optimizer

A bilingual (English / Indonesian) metabolic-health & insulin-control web app that
acts as a live **bio-optimizer**. A Vanilla-JS + Tailwind frontend talks to a
FastAPI backend wired to the Google Gemini API.

## Architecture

| Layer     | Stack                                            |
|-----------|--------------------------------------------------|
| Frontend  | Single `index.html` — Tailwind (CDN), FontAwesome, Vanilla JS |
| Backend   | `main.py` — FastAPI, Uvicorn, google-generativeai, Pydantic |
| Model     | `gemini-1.5-flash` @ temperature `0.4`           |

## Features

- **Biometric Console** — live BMI (with Normal/Overweight/Obese badge), TDEE
  (Mifflin-St Jeor) and a 20% deficit target, recalculated on every keystroke.
- **Diet Path** — Low-GI vs. Keto-Asian, injected into every AI prompt.
- **Hunger S.O.S** — a pulsing emergency button that opens a glassmorphism
  "Craving Interceptor" modal, classifies the craving (High Sugar / High Fat),
  and returns an aggressive, personalized intervention.
- **Gemini AI Wellness Companion** — a persistent chat with a typing indicator
  and graceful backend-error fallbacks.

## Run it

### 1. Backend
```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your GEMINI_API_KEY
uvicorn main:app --reload --port 8000
```

### 2. Frontend
Open `index.html` in a browser (or serve it):
```bash
python -m http.server 5500
# → http://localhost:5500
```
The frontend calls the backend at `http://localhost:8000`. CORS is open for
local development.

## API

| Endpoint        | Body                                                  |
|-----------------|-------------------------------------------------------|
| `POST /api/chat`| `{ "message": str, "language": "en\|id", "track": "low-gi\|keto-asian" }` |
| `POST /api/sos` | `{ "craving": str, "category": "High Sugar\|High Fat", "language": "en\|id" }` |
| `GET  /api/health` | key-configuration status                           |
