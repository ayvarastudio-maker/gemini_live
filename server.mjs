/**
 * Style Ayvara — Gemini Live API proxy.
 * Keeps GEMINI_API_KEY on the server; clients receive short-lived session tokens only.
 *
 * Env: GEMINI_API_KEY (required), PORT (default 8787), LIVE_MODEL (optional)
 */
import crypto from 'crypto';
import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import { GoogleGenAI, Modality } from '@google/genai';

const PORT = Number(process.env.PORT || 8787);
const API_KEY = process.env.GEMINI_API_KEY?.trim();
const LIVE_MODEL =
  process.env.LIVE_MODEL || 'gemini-2.0-flash-live-001';

if (!API_KEY) {
  console.error('GEMINI_API_KEY is required.');
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: API_KEY });
const sessions = new Map();

/** Temporary: counts outbound calls to Gemini (connect, sendClientContent, sendRealtimeInput). */
let geminiOutboundRequestCount = 0;

const LOG_TAG_REQ = '[GEMINI_REQ_LOG]';
const LOG_TAG_PROXY = '[GEMINI_PROXY_LOG]';

function logTimestamp() {
  return new Date().toISOString();
}

function newOpaqueId(prefix) {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`;
}

function logGeminiOutbound(event, { clientWsId, liveSessionId }) {
  geminiOutboundRequestCount += 1;
  console.log(
    `${LOG_TAG_REQ} #${geminiOutboundRequestCount} ts=${logTimestamp()} event=${event} endpoint=gemini.live model=${LIVE_MODEL} clientWsId=${clientWsId} liveSessionId=${liveSessionId ?? 'n/a'}`,
  );
}

function logProxyLifecycle(event, { clientWsId, liveSessionId, extra = '' }) {
  const suffix = extra ? ` ${extra}` : '';
  console.log(
    `${LOG_TAG_PROXY} ts=${logTimestamp()} event=${event} clientWsId=${clientWsId} liveSessionId=${liveSessionId ?? 'n/a'}${suffix}`,
  );
}

async function geminiSendClientContent(liveSession, payload, meta) {
  logGeminiOutbound('gemini.live.sendClientContent', meta);
  return liveSession.sendClientContent(payload);
}

async function geminiSendRealtimeInput(liveSession, payload, meta, eventSuffix) {
  logGeminiOutbound(`gemini.live.sendRealtimeInput.${eventSuffix}`, meta);
  return liveSession.sendRealtimeInput(payload);
}

const app = express();
app.use(express.json({ limit: '12mb' }));

app.get('/health', (_req, res) => {
  res.json({ ok: true, liveModel: LIVE_MODEL });
});

/** Creates a short-lived token for WebSocket auth (no API key in client). */
app.post('/v1/live/session', (req, res) => {
  const {
    systemInstruction = '',
    initialAnalysisText = '',
    imageBase64 = '',
    imageMimeType = 'image/jpeg',
    priorTranscript = [],
  } = req.body ?? {};

  const sessionToken = crypto.randomUUID();
  const expiresAt = Date.now() + 10 * 60 * 1000;
  sessions.set(sessionToken, {
    expiresAt,
    reconnectUsed: false,
    config: {
      systemInstruction,
      initialAnalysisText,
      imageBase64,
      imageMimeType,
      priorTranscript,
    },
  });

  res.json({
    sessionToken,
    wsPath: '/v1/live/ws',
    expiresInSeconds: 600,
    protocol: 'gemini-live-proxy-v1',
  });
});

const server = http.createServer(app);
const wss = new WebSocketServer({ noServer: true });

server.on('upgrade', (request, socket, head) => {
  const url = new URL(request.url, `http://${request.headers.host}`);
  if (url.pathname !== '/v1/live/ws') {
    socket.destroy();
    return;
  }
  const token =
    url.searchParams.get('token') ||
    request.headers['sec-websocket-protocol']?.split(',')[0]?.trim();
  const session = token ? sessions.get(token) : null;
  if (!session || session.expiresAt < Date.now()) {
    socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
    socket.destroy();
    return;
  }

  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, session, token);
  });
});

function sendClient(ws, payload) {
  if (ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify(payload));
  }
}

wss.on('connection', async (clientWs, session, token) => {
  let liveSession = null;
  let userTranscriptBuffer = '';
  let modelTranscriptBuffer = '';
  let modelSpeaking = false;
  let paused = false;

  const clientWsId = newOpaqueId('cws');
  const liveSessionId = newOpaqueId('gls');
  const geminiMeta = () => ({ clientWsId, liveSessionId });

  logProxyLifecycle('CLIENT_WS_CONNECT', { clientWsId, liveSessionId });

  clientWs.on('close', () => {
    logProxyLifecycle('CLIENT_WS_DISCONNECT', { clientWsId, liveSessionId });
    try {
      liveSession?.close();
    } catch (_) {}
    sessions.delete(token);
  });

  const pushState = (state) => sendClient(clientWs, { type: 'state', state });

  try {
    pushState('Bağlanıyor');

    logGeminiOutbound('gemini.live.connect', geminiMeta());
    liveSession = await ai.live.connect({
      model: LIVE_MODEL,
      config: {
        responseModalities: [Modality.AUDIO],
        inputAudioTranscription: {},
        outputAudioTranscription: {},
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
        },
        systemInstruction: {
          parts: [{ text: session.config.systemInstruction }],
        },
      },
      callbacks: {
        onopen: () => {
          logProxyLifecycle('GEMINI_LIVE_SESSION_START', {
            clientWsId,
            liveSessionId,
          });
          pushState('Düşünüyor');
        },
        onmessage: (message) => {
          if (message.serverContent?.interrupted) {
            modelSpeaking = false;
            pushState('Dinliyorum');
            sendClient(clientWs, { type: 'interrupt' });
          }

          if (message.serverContent?.inputTranscription?.text) {
            userTranscriptBuffer += message.serverContent.inputTranscription.text;
            sendClient(clientWs, {
              type: 'transcript',
              role: 'user',
              text: userTranscriptBuffer,
              final: false,
            });
          }

          if (message.serverContent?.outputTranscription?.text) {
            modelTranscriptBuffer += message.serverContent.outputTranscription.text;
            sendClient(clientWs, {
              type: 'transcript',
              role: 'model',
              text: modelTranscriptBuffer,
              final: false,
            });
          }

          const parts = message.serverContent?.modelTurn?.parts ?? [];
          for (const part of parts) {
            if (part.inlineData?.data) {
              modelSpeaking = true;
              pushState('Konuşuyor');
              sendClient(clientWs, {
                type: 'audio',
                mimeType: part.inlineData.mimeType || 'audio/pcm',
                data: part.inlineData.data,
              });
            }
            if (part.text) {
              modelTranscriptBuffer += part.text;
              sendClient(clientWs, {
                type: 'transcript',
                role: 'model',
                text: modelTranscriptBuffer,
                final: false,
              });
            }
          }

          if (message.serverContent?.turnComplete) {
            if (userTranscriptBuffer.trim()) {
              sendClient(clientWs, {
                type: 'transcript',
                role: 'user',
                text: userTranscriptBuffer.trim(),
                final: true,
              });
              userTranscriptBuffer = '';
            }
            if (modelTranscriptBuffer.trim()) {
              sendClient(clientWs, {
                type: 'transcript',
                role: 'model',
                text: modelTranscriptBuffer.trim(),
                final: true,
              });
              modelTranscriptBuffer = '';
            }
            modelSpeaking = false;
            pushState('Dinliyorum');
          }
        },
        onerror: (e) => {
          sendClient(clientWs, { type: 'error', message: e.message ?? String(e) });
        },
        onclose: () => {
          logProxyLifecycle('GEMINI_LIVE_SESSION_CLOSE', {
            clientWsId,
            liveSessionId,
          });
          sendClient(clientWs, { type: 'closed' });
        },
      },
    });

    // Image once at session start, then written context, then voice kickoff prompt.
    if (session.config.imageBase64) {
      await geminiSendClientContent(
        liveSession,
        {
          turns: [
            {
              role: 'user',
              parts: [
                {
                  inlineData: {
                    mimeType: session.config.imageMimeType,
                    data: session.config.imageBase64,
                  },
                },
              ],
            },
          ],
          turnComplete: true,
        },
        geminiMeta(),
      );
    }

    for (const line of session.config.priorTranscript ?? []) {
      if (!line?.text) continue;
      await geminiSendClientContent(
        liveSession,
        {
          turns: [
            {
              role: line.role === 'user' ? 'user' : 'model',
              parts: [{ text: line.text }],
            },
          ],
          turnComplete: true,
        },
        geminiMeta(),
      );
    }

    if (session.config.initialAnalysisText?.trim()) {
      await geminiSendClientContent(
        liveSession,
        {
          turns: [
            {
              role: 'user',
              parts: [{ text: session.config.initialAnalysisText.trim() }],
            },
          ],
          turnComplete: true,
        },
        geminiMeta(),
      );
    }

    pushState('Dinliyorum');

    clientWs.on('message', async (raw) => {
      try {
        const msg = JSON.parse(raw.toString());
        switch (msg.type) {
          case 'audio':
            if (paused) break;
            await geminiSendRealtimeInput(
              liveSession,
              {
                audio: {
                  data: msg.data,
                  mimeType: msg.mimeType || 'audio/pcm;rate=16000',
                },
              },
              geminiMeta(),
              'audio',
            );
            if (!modelSpeaking) pushState('Dinliyorum');
            break;
          case 'text':
            await geminiSendClientContent(
              liveSession,
              {
                turns: [{ role: 'user', parts: [{ text: msg.text }] }],
                turnComplete: true,
              },
              geminiMeta(),
            );
            pushState('Düşünüyor');
            break;
          case 'interrupt':
            await geminiSendRealtimeInput(
              liveSession,
              { audioStreamEnd: true },
              geminiMeta(),
              'audioStreamEnd',
            );
            modelSpeaking = false;
            pushState('Dinliyorum');
            break;
          case 'pause':
            paused = true;
            pushState('Duraklatıldı');
            break;
          case 'resume':
            paused = false;
            pushState('Dinliyorum');
            break;
          case 'reconnect':
            if (session.reconnectUsed) {
              sendClient(clientWs, {
                type: 'error',
                message: 'Reconnect limit reached',
              });
              return;
            }
            session.reconnectUsed = true;
            sendClient(clientWs, { type: 'reconnected' });
            pushState('Dinliyorum');
            break;
          default:
            break;
        }
      } catch (err) {
        sendClient(clientWs, { type: 'error', message: String(err) });
      }
    });
  } catch (err) {
    sendClient(clientWs, { type: 'error', message: String(err) });
    clientWs.close();
  }
});

server.listen(PORT, () => {
  console.log(`Gemini Live proxy listening on http://0.0.0.0:${PORT}`);
});
