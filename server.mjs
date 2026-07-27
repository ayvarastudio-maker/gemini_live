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
  process.env.LIVE_MODEL || 'gemini-2.5-flash-native-audio-latest';

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
const LOG_TAG_VOICE = '[VOICE_FLOW_LOG]';
const LOG_TAG_CLOSE = '[SESSION_CLOSE_DIAG]';

function safeErrorPayload(err) {
  if (err == null) return 'null';
  if (typeof err === 'string') return err;
  try {
    return JSON.stringify({
      name: err.name,
      message: err.message,
      code: err.code,
      status: err.status,
      statusText: err.statusText,
      reason: err.reason,
      type: err.type,
      stack: err.stack,
    });
  } catch {
    return String(err);
  }
}

function safeEventPayload(event) {
  if (event == null) return 'null';
  if (typeof event === 'string') return event;
  try {
    return JSON.stringify({
      code: event.code,
      reason: event.reason,
      message: event.message,
      type: event.type,
      wasClean: event.wasClean,
    });
  } catch {
    return safeErrorPayload(event);
  }
}

/** Logs immediately before every session / WebSocket teardown (no secrets). */
function logSessionClose(details) {
  const {
    initiator,
    clientWsId,
    liveSessionId,
    clientWsCloseCode,
    clientWsCloseReason,
    geminiErrorPayload,
    geminiClosePayload,
    stackTrace,
    flutterRequestedDisconnect,
    geminiClosedSession,
    proxyClosedClient,
    proxyClosedGemini,
  } = details;

  console.log(
    `${LOG_TAG_CLOSE} ts=${logTimestamp()} initiator=${initiator ?? 'unknown'} ` +
      `clientWsId=${clientWsId ?? 'n/a'} liveSessionId=${liveSessionId ?? 'n/a'} ` +
      `clientWsCloseCode=${clientWsCloseCode ?? 'n/a'} ` +
      `clientWsCloseReason=${JSON.stringify(clientWsCloseReason ?? '')} ` +
      `flutterRequestedDisconnect=${flutterRequestedDisconnect === true} ` +
      `geminiClosedSession=${geminiClosedSession === true} ` +
      `proxyClosedClient=${proxyClosedClient === true} ` +
      `proxyClosedGemini=${proxyClosedGemini === true} ` +
      `geminiErrorPayload=${geminiErrorPayload ?? 'n/a'} ` +
      `geminiClosePayload=${geminiClosePayload ?? 'n/a'} ` +
      `stackTrace=${JSON.stringify(stackTrace ?? '')}`,
  );
}

function captureStackTrace() {
  return new Error('SESSION_CLOSE_DIAG').stack ?? '';
}

function logVoiceFlow(event, { clientWsId, liveSessionId, extra = '' }) {
  const suffix = extra ? ` ${extra}` : '';
  console.log(
    `${LOG_TAG_VOICE} ts=${logTimestamp()} event=${event} clientWsId=${clientWsId} liveSessionId=${liveSessionId ?? 'n/a'}${suffix}`,
  );
}

function audioByteLengthBase64(b64) {
  if (!b64 || typeof b64 !== 'string') return 0;
  try {
    return Buffer.from(b64, 'base64').length;
  } catch {
    return 0;
  }
}

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
  let flutterAudioChunksIn = 0;
  let lastGeminiErrorPayload = 'none';
  let proxyClosingClient = false;
  let proxyClosingGemini = false;
  let geminiSessionClosed = false;

  logProxyLifecycle('CLIENT_WS_CONNECT', { clientWsId, liveSessionId });
  logVoiceFlow('client_connected', { clientWsId, liveSessionId });

  clientWs.on('error', (err) => {
    logSessionClose({
      initiator: 'client_websocket_error',
      clientWsId,
      liveSessionId,
      geminiErrorPayload: safeErrorPayload(err),
      stackTrace: captureStackTrace(),
      flutterRequestedDisconnect: false,
      geminiClosedSession: geminiSessionClosed,
      proxyClosedClient: proxyClosingClient,
      proxyClosedGemini: proxyClosingGemini,
    });
  });

  clientWs.on('close', (code, reasonBuffer) => {
    const reason =
      typeof reasonBuffer === 'string'
        ? reasonBuffer
        : reasonBuffer?.toString?.() ?? '';
    const flutterRequested =
      !proxyClosingClient && !proxyClosingGemini && !geminiSessionClosed;

    logSessionClose({
      initiator: 'client_websocket_close',
      clientWsId,
      liveSessionId,
      clientWsCloseCode: code,
      clientWsCloseReason: reason,
      geminiErrorPayload: lastGeminiErrorPayload,
      stackTrace: captureStackTrace(),
      flutterRequestedDisconnect: flutterRequested,
      geminiClosedSession: geminiSessionClosed,
      proxyClosedClient: proxyClosingClient,
      proxyClosedGemini: proxyClosingGemini,
    });

    logProxyLifecycle('CLIENT_WS_DISCONNECT', { clientWsId, liveSessionId });
    try {
      if (liveSession) {
        proxyClosingGemini = true;
        liveSession.close();
      }
    } catch (closeErr) {
      logSessionClose({
        initiator: 'proxy_liveSession_close_error',
        clientWsId,
        liveSessionId,
        geminiErrorPayload: safeErrorPayload(closeErr),
        stackTrace: captureStackTrace(),
        flutterRequestedDisconnect: false,
        geminiClosedSession: geminiSessionClosed,
        proxyClosedClient: proxyClosingClient,
        proxyClosedGemini: true,
      });
    }
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
          logVoiceFlow('gemini_session_started', {
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
              const outBytes = audioByteLengthBase64(part.inlineData.data);
              logVoiceFlow('gemini_audio_response_received', {
                clientWsId,
                liveSessionId,
                extra: `bytes=${outBytes}`,
              });
              sendClient(clientWs, {
                type: 'audio',
                mimeType: part.inlineData.mimeType || 'audio/pcm',
                data: part.inlineData.data,
              });
              logVoiceFlow('audio_response_sent_to_flutter', {
                clientWsId,
                liveSessionId,
                extra: `bytes=${outBytes}`,
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
          lastGeminiErrorPayload = safeErrorPayload(e);
          logSessionClose({
            initiator: 'gemini_live_onerror',
            clientWsId,
            liveSessionId,
            geminiErrorPayload: lastGeminiErrorPayload,
            stackTrace: captureStackTrace(),
            flutterRequestedDisconnect: false,
            geminiClosedSession: geminiSessionClosed,
            proxyClosedClient: proxyClosingClient,
            proxyClosedGemini: proxyClosingGemini,
          });
          sendClient(clientWs, { type: 'error', message: e.message ?? String(e) });
        },
        onclose: (event) => {
          geminiSessionClosed = true;
          const geminiClosePayload = safeEventPayload(event);
          logSessionClose({
            initiator: 'gemini_live_onclose',
            clientWsId,
            liveSessionId,
            geminiClosePayload,
            geminiErrorPayload: lastGeminiErrorPayload,
            stackTrace: captureStackTrace(),
            flutterRequestedDisconnect: false,
            geminiClosedSession: true,
            proxyClosedClient: proxyClosingClient,
            proxyClosedGemini: proxyClosingGemini,
          });
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
            flutterAudioChunksIn += 1;
            {
              const inBytes = audioByteLengthBase64(msg.data);
              if (
                flutterAudioChunksIn === 1 ||
                flutterAudioChunksIn % 50 === 0
              ) {
                logVoiceFlow('audio_chunk_received_from_flutter', {
                  clientWsId,
                  liveSessionId,
                  extra: `bytes=${inBytes} chunkIndex=${flutterAudioChunksIn}`,
                });
              }
            }
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
            if (
              flutterAudioChunksIn === 1 ||
              flutterAudioChunksIn % 50 === 0
            ) {
              logVoiceFlow('audio_chunk_sent_to_gemini', {
                clientWsId,
                liveSessionId,
                extra: `chunkIndex=${flutterAudioChunksIn}`,
              });
            }
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
          case 'audioStreamEnd':
            await geminiSendRealtimeInput(
              liveSession,
              { audioStreamEnd: true },
              geminiMeta(),
              'audioStreamEnd',
            );
            pushState('Dinliyorum');
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
        lastGeminiErrorPayload = safeErrorPayload(err);
        logSessionClose({
          initiator: 'client_message_handler_error',
          clientWsId,
          liveSessionId,
          geminiErrorPayload: lastGeminiErrorPayload,
          stackTrace: err?.stack ?? captureStackTrace(),
          flutterRequestedDisconnect: false,
          geminiClosedSession: geminiSessionClosed,
          proxyClosedClient: proxyClosingClient,
          proxyClosedGemini: proxyClosingGemini,
        });
        sendClient(clientWs, { type: 'error', message: String(err) });
      }
    });
  } catch (err) {
    logSessionClose({
      initiator: 'proxy_connection_setup_error',
      clientWsId,
      liveSessionId,
      geminiErrorPayload: safeErrorPayload(err),
      stackTrace: err?.stack ?? captureStackTrace(),
      flutterRequestedDisconnect: false,
      geminiClosedSession: geminiSessionClosed,
      proxyClosedClient: true,
      proxyClosedGemini: proxyClosingGemini,
    });
    sendClient(clientWs, { type: 'error', message: String(err) });
    proxyClosingClient = true;
    clientWs.close();
  }
});

server.listen(PORT, () => {
  console.log(`Gemini Live proxy listening on http://0.0.0.0:${PORT}`);
});
