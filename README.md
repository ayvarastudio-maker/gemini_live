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
LIVE_PROXY_BASE_URL=wss://gemini-live-3b9i.onrender.com
```

(WebSocket connects to the same host with `wss://`; session HTTP uses the Render HTTPS origin from this base URL in the app.)

## Security

The mobile app never receives the Gemini API key for Live mode—only a 10-minute session token.
