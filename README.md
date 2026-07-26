# Style Ayvara — Gemini Live proxy

This server holds `GEMINI_API_KEY` and exposes:

- `POST /v1/live/session` → short-lived `sessionToken`
- `WebSocket /v1/live/ws?token=...` → **real Gemini Live API** session via `@google/genai` `live.connect()`

## Run locally

```bash
cd backend/gemini_live
npm install
set GEMINI_API_KEY=your_key
npm start
```

Set in Flutter `.env`:

```
LIVE_PROXY_BASE_URL=http://10.0.2.2:8787
```

(Use your machine LAN IP for a physical Android device, e.g. `http://192.168.1.10:8787`.)

## Security

The mobile app never receives the Gemini API key for Live mode—only a 10-minute session token.
