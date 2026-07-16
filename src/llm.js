// LLM factory — OpenAI-compatible (OpenAI / DeepSeek / Groq) / Anthropic / Gemini
// behind one streaming interface, with capability routing and rate-limit fallback.
// stream({ system, turns:[{role,text}], imageDataUrl, maxTokens, onToken }) -> Promise<fullText>

// Provider registry. `openaiCompat` providers share one code path via baseURL.
const PROVIDERS = {
  openai:    { openaiCompat: true, baseURL: undefined, vision: true },
  deepseek:  { openaiCompat: true, baseURL: 'https://api.deepseek.com', vision: false },
  groq:      { openaiCompat: true, baseURL: 'https://api.groq.com/openai/v1', vision: true },
  anthropic: { openaiCompat: false, vision: true },
  gemini:    { openaiCompat: false, vision: true }
};

// Fallback order when the primary provider fails (rate limit / quota) or
// can't handle the request (no vision). Gemini last = universal safety net.
const FALLBACK_ORDER = ['groq', 'gemini', 'openai', 'anthropic', 'deepseek'];

function stripDataUrl(dataUrl) {
  const m = /^data:(.+?);base64,(.*)$/s.exec(dataUrl || '');
  return m ? { mime: m[1], b64: m[2] } : null;
}

function isLimitError(e) {
  const status = e && (e.status || e.statusCode);
  if (status === 429 || status === 402 || status === 403 || (status >= 500 && status < 600)) return true;
  const msg = ((e && e.message) || '').toLowerCase();
  return /rate.?limit|quota|exceeded|insufficient|overloaded|capacity/.test(msg);
}

async function streamOpenAI({ apiKey, baseURL, model, system, turns, imageDataUrl, maxTokens, onToken }) {
  const OpenAI = require('openai');
  const client = new OpenAI({ apiKey, baseURL });
  const messages = [{ role: 'system', content: system }];
  turns.forEach((t, i) => {
    const last = i === turns.length - 1;
    if (last && imageDataUrl && t.role === 'user') {
      messages.push({ role: 'user', content: [
        { type: 'text', text: t.text },
        { type: 'image_url', image_url: { url: imageDataUrl } }
      ] });
    } else {
      messages.push({ role: t.role, content: t.text });
    }
  });
  const stream = await client.chat.completions.create({ model, messages, stream: true, max_tokens: maxTokens });
  let full = '';
  for await (const part of stream) {
    const d = part.choices && part.choices[0] && part.choices[0].delta && part.choices[0].delta.content;
    if (d) { full += d; onToken(d); }
  }
  return full;
}

async function streamAnthropic({ apiKey, model, system, turns, imageDataUrl, maxTokens, onToken }) {
  const Anthropic = require('@anthropic-ai/sdk');
  const client = new Anthropic({ apiKey });
  const messages = turns.map((t, i) => {
    const last = i === turns.length - 1;
    if (last && imageDataUrl && t.role === 'user') {
      const img = stripDataUrl(imageDataUrl);
      const content = [];
      if (img) content.push({ type: 'image', source: { type: 'base64', media_type: img.mime, data: img.b64 } });
      content.push({ type: 'text', text: t.text });
      return { role: 'user', content };
    }
    return { role: t.role, content: t.text };
  });
  const stream = await client.messages.create({ model, max_tokens: maxTokens, system, messages, stream: true });
  let full = '';
  for await (const ev of stream) {
    if (ev.type === 'content_block_delta' && ev.delta && ev.delta.type === 'text_delta') { full += ev.delta.text; onToken(ev.delta.text); }
  }
  return full;
}

async function streamGemini({ apiKey, model, system, turns, imageDataUrl, maxTokens, onToken }) {
  const { GoogleGenAI } = require('@google/genai');
  const ai = new GoogleGenAI({ apiKey });
  const contents = turns.map((t, i) => {
    const last = i === turns.length - 1;
    const parts = [{ text: t.text }];
    if (last && imageDataUrl && t.role === 'user') {
      const img = stripDataUrl(imageDataUrl);
      if (img) parts.push({ inlineData: { mimeType: img.mime, data: img.b64 } });
    }
    return { role: t.role === 'assistant' ? 'model' : 'user', parts };
  });
  const stream = await ai.models.generateContentStream({
    model, contents, config: { systemInstruction: system, maxOutputTokens: maxTokens }
  });
  let full = '';
  for await (const chunk of stream) {
    const t = chunk && chunk.text;
    if (t) { full += t; onToken(t); }
  }
  return full;
}

function streamWith(provider, args) {
  const def = PROVIDERS[provider];
  if (!def) throw new Error('unknown provider: ' + provider);
  if (def.openaiCompat) return streamOpenAI({ ...args, baseURL: def.baseURL });
  if (provider === 'anthropic') return streamAnthropic(args);
  if (provider === 'gemini') return streamGemini(args);
  throw new Error('unknown provider: ' + provider);
}

function pickModel(settings, provider, tier, needsVision) {
  const models = settings.models || {};
  if (needsVision && provider === 'groq' && settings.visionFallbackModel) return settings.visionFallbackModel;
  return (models[provider] || {})[tier];
}

// Builds the ordered provider chain for one request.
function buildChain(settings, needsVision) {
  const keys = settings.apiKeys || {};
  const order = [settings.provider, ...FALLBACK_ORDER.filter((p) => p !== settings.provider)];
  return order.filter((p) => {
    const def = PROVIDERS[p];
    if (!def || !keys[p]) return false;
    if (needsVision && !def.vision) return false;
    return true;
  });
}

function createLLM(settings, onStatus) {
  const keys = settings.apiKeys || {};
  const provider = settings.provider;
  const tier = settings.smart ? 'smart' : 'fast';
  const maxTokens = settings.smart ? 1400 : 700;
  const notify = typeof onStatus === 'function' ? onStatus : () => {};

  return {
    provider,
    model: pickModel(settings, provider, tier, false),
    apiKey: keys[provider],
    ready: buildChain(settings, false).length > 0,
    async stream(params) {
      const needsVision = !!params.imageDataUrl;
      let chain = buildChain(settings, needsVision);
      if (!chain.length && needsVision) {
        // No vision-capable key at all — degrade to text-only rather than fail.
        chain = buildChain(settings, false);
        if (chain.length) notify(chain[0] + ' cannot see the screen and no vision-capable key is set — answering from text only.');
        params = { ...params, imageDataUrl: null };
      }
      if (!chain.length) throw new Error('no usable provider — add an API key in Settings');

      if (chain[0] !== provider) {
        notify(needsVision
          ? provider + ' has no vision — using ' + chain[0] + ' for this screenshot.'
          : 'using ' + chain[0] + ' (no ' + provider + ' key set).');
      }

      let lastErr = null;
      for (let i = 0; i < chain.length; i++) {
        const p = chain[i];
        const model = pickModel(settings, p, tier, !!params.imageDataUrl);
        if (!model) continue;
        let emitted = false;
        const onToken = (t) => { emitted = true; params.onToken(t); };
        try {
          return await streamWith(p, { apiKey: keys[p], model, maxTokens, ...params, onToken });
        } catch (e) {
          lastErr = e;
          const next = chain[i + 1];
          if (!next || emitted || !isLimitError(e)) throw e;
          notify(p + ' unavailable (' + ((e && e.status) || 'limit') + ') — falling back to ' + next + '.');
        }
      }
      throw lastErr || new Error('no usable provider/model — check Settings');
    }
  };
}

module.exports = { createLLM, PROVIDERS };
