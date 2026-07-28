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
const LOG_TAG_SPEECH_DIAG = '[GEMINI_SPEECH_DIAG]';
const LOG_TAG_SETUP = '[GEMINI_SETUP_LOG]';
const EXPECTED_INPUT_MIME = 'audio/pcm;rate=16000';

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
      wsEventType: event.type,
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

function logGeminiAudioReceived(meta, mimeType, pcmBytes) {
  logVoiceFlow('GEMINI_AUDIO_RECEIVED', {
    ...meta,
    extra: `mimeType=${mimeType} pcmBytes=${pcmBytes}`,
  });
}

function logGeminiAudioForwarded(meta, mimeType, pcmBytes) {
  logVoiceFlow('GEMINI_AUDIO_FORWARDED', {
    ...meta,
    extra: `mimeType=${mimeType} pcmBytes=${pcmBytes}`,
  });
}

function logGeminiTurnComplete(meta, fields = {}) {
  const extras = Object.entries(fields)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(' ');
  logVoiceFlow('GEMINI_TURN_COMPLETE', {
    ...meta,
    extra: extras,
  });
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

function logSpeechDiag(event, meta, fields = {}) {
  const extras = Object.entries(fields)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(' ');
  console.log(
    `${LOG_TAG_SPEECH_DIAG} ts=${logTimestamp()} event=${event} ` +
      `clientWsId=${meta.clientWsId} liveSessionId=${meta.liveSessionId}` +
      (extras ? ` ${extras}` : ''),
  );
}

function summarizeModelTurnParts(parts) {
  const summary = [];
  for (const part of parts ?? []) {
    if (part.inlineData?.data) {
      summary.push({
        kind: 'audio',
        bytes: audioByteLengthBase64(part.inlineData.data),
        mimeType: part.inlineData.mimeType || 'audio/pcm',
      });
    } else if (part.text) {
      summary.push({ kind: 'text', chars: part.text.length });
    } else {
      summary.push({ kind: 'other' });
    }
  }
  return summary;
}

function summarizeGeminiServerMessage(message) {
  const sc = message?.serverContent;
  return {
    setupComplete: message?.setupComplete != null,
    serverContent: sc
      ? {
          turnComplete: sc.turnComplete ?? null,
          generationComplete: sc.generationComplete ?? null,
          interrupted: sc.interrupted ?? null,
          waitingForInput: sc.waitingForInput ?? null,
          turnCompleteReason: sc.turnCompleteReason ?? null,
          inputTranscription: sc.inputTranscription?.text
            ? {
                chars: sc.inputTranscription.text.length,
                finished: sc.inputTranscription.finished ?? null,
              }
            : null,
          modelTurn: sc.modelTurn?.parts
            ? summarizeModelTurnParts(sc.modelTurn.parts)
            : null,
        }
      : null,
    voiceActivity: message?.voiceActivity?.voiceActivityType ?? null,
    vadSignal: message?.voiceActivityDetectionSignal?.vadSignalType ?? null,
  };
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

/** Blocks deprecated Bidi client payloads; logs Gemini wire after audioStreamEnd. */
function guardGeminiLiveTransport(liveSession, meta, turnWatch) {
  const conn = liveSession?.conn;
  if (!conn || typeof conn.send !== 'function' || conn.__ayvaraSendGuarded) {
    return;
  }
  const originalSend = conn.send.bind(conn);
  let setupWireLogged = false;
  conn.send = (message) => {
    if (typeof message === 'string') {
      try {
        const parsed = JSON.parse(message);
        if (parsed?.setup && !setupWireLogged) {
          setupWireLogged = true;
          console.log(
            `${LOG_TAG_SETUP} ts=${logTimestamp()} event=setup_wire_send ` +
              `clientWsId=${meta.clientWsId} liveSessionId=${meta.liveSessionId} ` +
              `payload=${JSON.stringify(redactLiveWireJson(parsed))}`,
          );
        }
        if (turnWatch?.logOutboundAfterStreamEnd) {
          const keys = Object.keys(parsed).filter((k) => parsed[k] != null);
          logSpeechDiag('gemini_wire_outbound_after_audioStreamEnd', meta, {
            keys,
            hasClientContent: parsed.clientContent != null,
            clientTurnComplete: parsed.clientContent?.turnComplete ?? null,
            hasRealtimeInput: parsed.realtimeInput != null,
            realtimeAudioStreamEnd:
              parsed.realtimeInput?.audioStreamEnd ?? null,
            hasActivityStart: parsed.realtimeInput?.activityStart != null,
            hasActivityEnd: parsed.realtimeInput?.activityEnd != null,
          });
        }
        if (
          turnWatch?.streamEndForwarded &&
          parsed.clientContent?.turnComplete === true
        ) {
          logSpeechDiag(
            'BLOCKED_clientContent_turnComplete_after_audioStreamEnd',
            meta,
            {},
          );
          return;
        }
        if (parsed?.type === 'close' || parsed?.close != null) {
          logProxyLifecycle('BLOCKED_LEGACY_GEMINI_CLOSE_PAYLOAD', {
            ...meta,
            extra: 'type=close',
          });
          return;
        }
      } catch {
        // Non-JSON frames are passed through unchanged.
      }
    }
    return originalSend(message);
  };
  conn.__ayvaraSendGuarded = true;
}

/**
 * Official Live API teardown: optional audioStreamEnd, then WebSocket close only.
 * Never sends legacy {"type":"close"} (or similar) JSON to Gemini.
 */
async function shutdownGeminiLiveSession(
  liveSession,
  meta,
  { signalAudioStreamEnd = false } = {},
) {
  if (!liveSession) return;
  if (signalAudioStreamEnd) {
    try {
      await geminiSendRealtimeInput(
        liveSession,
        { audioStreamEnd: true },
        meta,
        'audioStreamEnd',
      );
    } catch (err) {
      logSessionClose({
        initiator: 'shutdown_audio_stream_end_error',
        ...meta,
        geminiErrorPayload: safeErrorPayload(err),
        stackTrace: captureStackTrace(),
        flutterRequestedDisconnect: false,
        geminiClosedSession: false,
        proxyClosedClient: false,
        proxyClosedGemini: true,
      });
    }
  }
  try {
    liveSession.conn?.close?.();
  } catch (err) {
    logSessionClose({
      initiator: 'shutdown_websocket_close_error',
      ...meta,
      geminiErrorPayload: safeErrorPayload(err),
      stackTrace: captureStackTrace(),
      flutterRequestedDisconnect: false,
      geminiClosedSession: false,
      proxyClosedClient: false,
      proxyClosedGemini: true,
    });
  }
}

function modelResourceName() {
  const bare = LIVE_MODEL.replace(/^models\//, '');
  return `models/${bare}`;
}

function buildLiveImageClientContent(mimeType, base64Data) {
  return {
    turns: [
      {
        role: 'user',
        parts: [
          {
            inlineData: {
              mimeType,
              data: base64Data,
            },
          },
        ],
      },
    ],
    turnComplete: true,
  };
}

const LIVE_TURKISH_SYSTEM_INSTRUCTION = {
  parts: [
    {
      text:
        'Sen Türkçe konuşan bir kişisel stil danışmanısın. Kullanıcı hangi dilde konuşursa konuşsun, bütün cevaplarını yalnızca Türkçe ve sesli olarak ver. Kısa, doğal ve anlaşılır konuş.',
    },
  ],
};

/** Minimum Live setup — AUDIO output + Turkish system instruction (no transcription). */
function buildMinimalLiveConnectConfig() {
  return {
    responseModalities: [Modality.AUDIO],
    systemInstruction: LIVE_TURKISH_SYSTEM_INSTRUCTION,
  };
}

function minimalSetupPayloadForLog() {
  return {
    setup: {
      model: modelResourceName(),
      generationConfig: {
        responseModalities: ['AUDIO'],
      },
      systemInstruction: LIVE_TURKISH_SYSTEM_INSTRUCTION,
    },
  };
}

function redactLiveWireJson(value) {
  if (value == null) return value;
  if (typeof value === 'string') {
    if (value.length > 120) {
      return `<string len=${value.length} redacted>`;
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => redactLiveWireJson(item));
  }
  if (typeof value !== 'object') return value;

  const out = {};
  for (const [key, raw] of Object.entries(value)) {
    const lower = key.toLowerCase();
    if (
      lower.includes('key') ||
      lower === 'data' ||
      lower === 'audiobytes' ||
      lower.includes('image') ||
      lower.includes('base64')
    ) {
      if (typeof raw === 'string') {
        out[key] = `<redacted len=${raw.length}>`;
      } else {
        out[key] = '<redacted>';
      }
      continue;
    }
    out[key] = redactLiveWireJson(raw);
  }
  return out;
}

function logMinimalSetupPayload(meta) {
  const payload = minimalSetupPayloadForLog();
  console.log(
    `${LOG_TAG_SETUP} ts=${logTimestamp()} event=setup_payload_outbound ` +
      `clientWsId=${meta.clientWsId} liveSessionId=${meta.liveSessionId} ` +
      `payload=${JSON.stringify(redactLiveWireJson(payload))}`,
  );
}

function logSetupComplete(meta) {
  console.log(
    `${LOG_TAG_SETUP} ts=${logTimestamp()} event=setupComplete_received ` +
      `clientWsId=${meta.clientWsId} liveSessionId=${meta.liveSessionId}`,
  );
}

function logFirstFlutterAudioChunk(meta, byteLength) {
  console.log(
    `${LOG_TAG_SETUP} ts=${logTimestamp()} event=first_flutter_audio_chunk_received ` +
      `clientWsId=${meta.clientWsId} liveSessionId=${meta.liveSessionId} pcmBytes=${byteLength}`,
  );
}

function logFirstAudioForwardedToGemini(meta, byteLength) {
  console.log(
    `${LOG_TAG_SETUP} ts=${logTimestamp()} event=first_audio_forwarded_to_gemini ` +
      `clientWsId=${meta.clientWsId} liveSessionId=${meta.liveSessionId} pcmBytes=${byteLength}`,
  );
}

function logFirstAudioAfterSetupComplete(meta, byteLength) {
  console.log(
    `${LOG_TAG_SETUP} ts=${logTimestamp()} event=first_audio_after_setupComplete ` +
      `clientWsId=${meta.clientWsId} liveSessionId=${meta.liveSessionId} pcmBytes=${byteLength}`,
  );
}

/**
 * Validates Flutter uplink: single base64 layer, raw PCM16 LE mono (no WAV container).
 */
function validateIncomingPcm16Base64(b64, mimeType) {
  if (typeof b64 !== 'string' || b64.length === 0) {
    throw new Error('Audio data must be a non-empty base64 string.');
  }
  if (mimeType && mimeType !== EXPECTED_INPUT_MIME) {
    throw new Error(
      `Unsupported audio mimeType "${mimeType}" (expected ${EXPECTED_INPUT_MIME}).`,
    );
  }
  let buf;
  try {
    buf = Buffer.from(b64, 'base64');
  } catch {
    throw new Error('Audio data is not valid base64.');
  }
  if (buf.length === 0) {
    throw new Error('Decoded audio is empty.');
  }
  if (buf.length % 2 !== 0) {
    throw new Error(
      `PCM16 frame must have even byte length (got ${buf.length}).`,
    );
  }
  if (
    buf.length >= 4 &&
    buf[0] === 0x52 &&
    buf[1] === 0x49 &&
    buf[2] === 0x46 &&
    buf[3] === 0x46
  ) {
    throw new Error('WAV header detected; send raw PCM16 only.');
  }
  return buf.length;
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
  let micReadySent = false;
  let micReadyFallbackTimer = null;
  let geminiSetupComplete = false;
  let firstGeminiAudioForwarded = false;
  let firstFlutterAudioLogged = false;
  let firstAudioForwardedLogged = false;
  let setupCompleteHandled = false;
  let liveImageForwarded = false;
  let suppressModelAudioUntilFirstUserStreamEnd = false;
  let firstUserStreamEndReceived = false;
  const turnWatch = {
    streamEndForwarded: false,
    logOutboundAfterStreamEnd: false,
    automaticActivityDetectionEnabled: true,
  };

  const onGeminiSetupComplete = () => {
    if (setupCompleteHandled) return;
    setupCompleteHandled = true;
    logVoiceFlow('gemini_setup_complete_notify_flutter', {
      clientWsId,
      liveSessionId,
    });
    sendClient(clientWs, { type: 'setupComplete' });
    scheduleMicReadyFallback();
  };

  const sendMicReadyOnce = () => {
    if (micReadySent) return;
    micReadySent = true;
    if (micReadyFallbackTimer) {
      clearTimeout(micReadyFallbackTimer);
      micReadyFallbackTimer = null;
    }
    sendClient(clientWs, { type: 'micReady' });
    pushState('Dinliyorum');
    logVoiceFlow('mic_ready_sent_to_flutter', { clientWsId, liveSessionId });
  };

  const scheduleMicReadyFallback = () => {
    if (micReadyFallbackTimer || micReadySent) return;
    micReadyFallbackTimer = setTimeout(() => {
      micReadyFallbackTimer = null;
      sendMicReadyOnce();
    }, 15000);
  };

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

  clientWs.on('close', async (code, reasonBuffer) => {
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
        await shutdownGeminiLiveSession(liveSession, {
          clientWsId,
          liveSessionId,
        });
        liveSession = null;
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

  clientWs.on('message', async (raw) => {
    try {
      const msg = JSON.parse(raw.toString());
      switch (msg.type) {
        case 'liveImage':
          if (!geminiSetupComplete || !liveSession) break;
          if (liveImageForwarded) break;
          {
            const mimeType = (msg.mimeType || 'image/jpeg').toString();
            const data = msg.data?.toString() ?? '';
            if (!data) {
              sendClient(clientWs, {
                type: 'error',
                message: 'liveImage: empty data',
              });
              sendMicReadyOnce();
              break;
            }
            liveImageForwarded = true;
            try {
              logVoiceFlow('live_image_forward_to_gemini', {
                clientWsId,
                liveSessionId,
                extra: `mimeType=${mimeType} base64Len=${data.length}`,
              });
              logVoiceFlow('live_image_silent_context_forward', {
                clientWsId,
                liveSessionId,
                extra: `mimeType=${mimeType} turnComplete=false`,
              });
              await geminiSendClientContent(
                liveSession,
                buildLiveImageClientContent(mimeType, data),
                geminiMeta(),
              );
              suppressModelAudioUntilFirstUserStreamEnd = true;
              firstUserStreamEndReceived = false;
            } catch (err) {
              liveImageForwarded = false;
              sendClient(clientWs, {
                type: 'error',
                message: `liveImage: ${String(err)}`,
              });
            }
            sendMicReadyOnce();
          }
          break;
        case 'audio':
          if (paused || !liveSession) break;
          if (!geminiSetupComplete) {
            logSpeechDiag('flutter_audio_chunk_dropped', geminiMeta(), {
              reason: 'setupComplete_pending',
            });
            logVoiceFlow('audio_chunk_dropped_before_setupComplete', {
              clientWsId,
              liveSessionId,
            });
            break;
          }
          flutterAudioChunksIn += 1;
          {
            const mimeType = EXPECTED_INPUT_MIME;
            const pcmBytes = validateIncomingPcm16Base64(msg.data, mimeType);
            const shouldLogChunk =
              flutterAudioChunksIn <= 10 || flutterAudioChunksIn % 25 === 0;
            if (shouldLogChunk) {
              logSpeechDiag('flutter_audio_chunk_received', geminiMeta(), {
                chunkIndex: flutterAudioChunksIn,
                pcmBytes,
              });
            }
            if (flutterAudioChunksIn === 1 && !firstFlutterAudioLogged) {
              firstFlutterAudioLogged = true;
              logFirstFlutterAudioChunk(geminiMeta(), pcmBytes);
            }
            if (!firstGeminiAudioForwarded) {
              logFirstAudioAfterSetupComplete(geminiMeta(), pcmBytes);
              firstGeminiAudioForwarded = true;
            }
            if (
              flutterAudioChunksIn === 1 ||
              flutterAudioChunksIn % 50 === 0
            ) {
              logVoiceFlow('audio_chunk_received_from_flutter', {
                clientWsId,
                liveSessionId,
                extra: `bytes=${pcmBytes} chunkIndex=${flutterAudioChunksIn}`,
              });
            }
            await geminiSendRealtimeInput(
              liveSession,
              {
                audio: {
                  data: msg.data,
                  mimeType,
                },
              },
              geminiMeta(),
              'audio',
            );
            if (!firstAudioForwardedLogged) {
              firstAudioForwardedLogged = true;
              logFirstAudioForwardedToGemini(geminiMeta(), pcmBytes);
            }
            if (shouldLogChunk) {
              logSpeechDiag('gemini_audio_chunk_forwarded', geminiMeta(), {
                chunkIndex: flutterAudioChunksIn,
                pcmBytes,
              });
            }
          }
          if (!modelSpeaking) pushState('Dinliyorum');
          break;
        case 'text':
          if (!geminiSetupComplete) break;
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
          if (!geminiSetupComplete) break;
          if (!firstUserStreamEndReceived) {
            firstUserStreamEndReceived = true;
            suppressModelAudioUntilFirstUserStreamEnd = false;
            logSpeechDiag('first_user_stream_end_unlocks_model_audio', geminiMeta(), {});
          }
          logSpeechDiag('flutter_audio_stream_end_received', geminiMeta(), {});
          logSpeechDiag('automatic_activity_detection', geminiMeta(), {
            enabled: turnWatch.automaticActivityDetectionEnabled,
            source:
              'default_live_api_no_realtimeInputConfig_in_setup_audioStreamEnd_used',
          });
          await geminiSendRealtimeInput(
            liveSession,
            { audioStreamEnd: true },
            geminiMeta(),
            'audioStreamEnd',
          );
          turnWatch.streamEndForwarded = true;
          turnWatch.logOutboundAfterStreamEnd = true;
          logSpeechDiag('gemini_audio_stream_end_forwarded', geminiMeta(), {
            note: 'realtimeInput.audioStreamEnd only; no clientContent.turnComplete',
          });
          pushState('Düşünüyor');
          break;
        case 'activityStart':
          if (!geminiSetupComplete) break;
          await geminiSendRealtimeInput(
            liveSession,
            { activityStart: {} },
            geminiMeta(),
            'activityStart',
          );
          break;
        case 'activityEnd':
          if (!geminiSetupComplete) break;
          await geminiSendRealtimeInput(
            liveSession,
            { activityEnd: {} },
            geminiMeta(),
            'activityEnd',
          );
          turnWatch.streamEndForwarded = true;
          turnWatch.logOutboundAfterStreamEnd = true;
          pushState('Düşünüyor');
          break;
        case 'interrupt':
          if (!geminiSetupComplete) break;
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
        case 'close':
        case 'disconnect':
          proxyClosingGemini = true;
          await shutdownGeminiLiveSession(liveSession, geminiMeta(), {
            signalAudioStreamEnd: true,
          });
          liveSession = null;
          sendClient(clientWs, { type: 'closed' });
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

  try {
    pushState('Bağlanıyor');

    logMinimalSetupPayload(geminiMeta());
    logGeminiOutbound('gemini.live.connect', geminiMeta());
    liveSession = await ai.live.connect({
      model: LIVE_MODEL,
      config: buildMinimalLiveConnectConfig(),
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
          if (message.voiceActivity?.voiceActivityType) {
            const activityType = message.voiceActivity.voiceActivityType;
            if (
              activityType === 'ACTIVITY_START' ||
              String(activityType).includes('START')
            ) {
              logSpeechDiag('gemini_speech_start', geminiMeta(), {
                voiceActivityType: activityType,
                activityStart: true,
              });
            } else if (
              activityType === 'ACTIVITY_END' ||
              String(activityType).includes('END')
            ) {
              logSpeechDiag('gemini_speech_end', geminiMeta(), {
                voiceActivityType: activityType,
                activityEnd: true,
              });
            } else {
              logSpeechDiag('gemini_voice_activity', geminiMeta(), {
                voiceActivityType: activityType,
              });
            }
          }
          if (message.voiceActivityDetectionSignal?.vadSignalType) {
            logSpeechDiag('gemini_vad_signal', geminiMeta(), {
              vadSignalType: message.voiceActivityDetectionSignal.vadSignalType,
            });
          }

          if (message.setupComplete && !geminiSetupComplete) {
            geminiSetupComplete = true;
            logSetupComplete(geminiMeta());
            logSpeechDiag('automatic_activity_detection', geminiMeta(), {
              enabled: turnWatch.automaticActivityDetectionEnabled,
              source:
                'setup_has_no_realtimeInputConfig_automaticActivityDetection_defaults_enabled',
            });
            onGeminiSetupComplete();
          }

          if (turnWatch.logOutboundAfterStreamEnd) {
            logSpeechDiag(
              'gemini_server_message_after_audioStreamEnd',
              geminiMeta(),
              summarizeGeminiServerMessage(message),
            );
          }

          if (message.serverContent?.interrupted) {
            modelSpeaking = false;
            pushState('Dinliyorum');
            sendClient(clientWs, { type: 'interrupt' });
          }

          if (message.serverContent?.inputTranscription?.text) {
            const inputText = message.serverContent.inputTranscription.text;
            logSpeechDiag('gemini_input_transcription', geminiMeta(), {
              chars: inputText.length,
              snippet: inputText.slice(0, 120),
              finished:
                message.serverContent.inputTranscription.finished ?? null,
            });
            userTranscriptBuffer += inputText;
          }

          if (message.serverContent?.outputTranscription?.text) {
            modelTranscriptBuffer +=
              message.serverContent.outputTranscription.text;
          }

          const parts = message.serverContent?.modelTurn?.parts ?? [];
          if (parts.length > 0) {
            logSpeechDiag('gemini_model_turn', geminiMeta(), {
              parts: summarizeModelTurnParts(parts),
            });
          }
          for (const part of parts) {
            if (part.inlineData?.data) {
              if (
                suppressModelAudioUntilFirstUserStreamEnd &&
                !firstUserStreamEndReceived
              ) {
                logSpeechDiag(
                  'model_audio_suppressed_awaiting_first_user_question',
                  geminiMeta(),
                  {
                    mimeType: part.inlineData.mimeType || 'audio/pcm',
                  },
                );
                continue;
              }
              modelSpeaking = true;
              pushState('Konuşuyor');
              const mimeType = part.inlineData.mimeType || 'audio/pcm';
              const outBytes = audioByteLengthBase64(part.inlineData.data);
              logGeminiAudioReceived(geminiMeta(), mimeType, outBytes);
              sendClient(clientWs, {
                type: 'audio',
                mimeType,
                data: part.inlineData.data,
              });
              logGeminiAudioForwarded(geminiMeta(), mimeType, outBytes);
            } else if (part.text?.trim()) {
              logSpeechDiag('gemini_model_text_suppressed', geminiMeta(), {
                chars: part.text.length,
                reason: 'audio_response_mode',
              });
            }
          }

          if (message.serverContent?.generationComplete) {
            logSpeechDiag('gemini_generation_complete', geminiMeta(), {
              afterAudioStreamEnd: turnWatch.logOutboundAfterStreamEnd,
            });
          }

          if (message.serverContent?.turnComplete) {
            turnWatch.streamEndForwarded = false;
            turnWatch.logOutboundAfterStreamEnd = false;
            logGeminiTurnComplete(geminiMeta(), {
              generationComplete:
                message.serverContent.generationComplete ?? null,
              waitingForInput: message.serverContent.waitingForInput ?? null,
              turnCompleteReason:
                message.serverContent.turnCompleteReason ?? null,
              interrupted: message.serverContent.interrupted ?? null,
            });
            logSpeechDiag('gemini_turn_complete', geminiMeta(), {
              generationComplete:
                message.serverContent.generationComplete ?? null,
              waitingForInput: message.serverContent.waitingForInput ?? null,
              turnCompleteReason:
                message.serverContent.turnCompleteReason ?? null,
              interrupted: message.serverContent.interrupted ?? null,
            });
            sendClient(clientWs, { type: 'turnComplete' });
            userTranscriptBuffer = '';
            modelTranscriptBuffer = '';
            modelSpeaking = false;
            pushState('Dinliyorum');
            sendMicReadyOnce();
          }
        },
        onerror: (e) => {
          lastGeminiErrorPayload = safeErrorPayload(e);
          logSpeechDiag('gemini_error', geminiMeta(), {
            payload: lastGeminiErrorPayload,
            afterAudioStreamEnd: turnWatch.logOutboundAfterStreamEnd,
          });
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
          logSpeechDiag('gemini_close', geminiMeta(), {
            payload: geminiClosePayload,
            closeCode: event?.code ?? null,
            closeReason: event?.reason ?? null,
            afterAudioStreamEnd: turnWatch.logOutboundAfterStreamEnd,
          });
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
    guardGeminiLiveTransport(liveSession, geminiMeta(), turnWatch);
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
