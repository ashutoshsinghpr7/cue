// Speech-to-text factory. Decoupled from the LLM provider because Anthropic and
// DeepSeek have no audio API — we transcribe with whatever audio-capable key is
// available (Groq Whisper first: free, then OpenAI Whisper, then Gemini) and
// fall back across providers. Returns { text, provider } or { text:'', error }.
const { pcmToWav } = require('./wav');

async function transcribeOpenAI(apiKey, wav, model, baseURL) {
  const OpenAI = require('openai');
  const toFile = OpenAI.toFile || require('openai/uploads').toFile;
  const client = new OpenAI({ apiKey, baseURL });
  const file = await toFile(wav, 'audio.wav', { type: 'audio/wav' });
  const res = await client.audio.transcriptions.create({ file, model: model || 'whisper-1' });
  return (res.text || '').trim();
}

function transcribeGroq(apiKey, wav) {
  return transcribeOpenAI(apiKey, wav, 'whisper-large-v3-turbo', 'https://api.groq.com/openai/v1');
}

async function transcribeGemini(apiKey, wav) {
  const { GoogleGenAI } = require('@google/genai');
  const ai = new GoogleGenAI({ apiKey });
  const res = await ai.models.generateContent({
    model: 'gemini-1.5-flash',
    contents: [{ role: 'user', parts: [
      { text: 'Transcribe this audio verbatim. Return only the spoken words with no commentary. If there is no clear speech, return an empty response.' },
      { inlineData: { mimeType: 'audio/wav', data: wav.toString('base64') } }
    ] }]
  });
  return ((res && res.text) || '').trim();
}

function createSTT(settings) {
  const keys = settings.apiKeys || {};
  const chain = [];
  if (keys.groq) chain.push({ p: 'groq', fn: (wav) => transcribeGroq(keys.groq, wav) });
  if (keys.openai) chain.push({ p: 'openai', fn: (wav) => transcribeOpenAI(keys.openai, wav, settings.sttModel) });
  if (keys.gemini) chain.push({ p: 'gemini', fn: (wav) => transcribeGemini(keys.gemini, wav) });

  return {
    available: chain.length > 0,
    providers: chain.map((c) => c.p),
    async transcribe(pcm) {
      if (!chain.length || !pcm || pcm.length < 3200) return { text: '' };
      const wav = pcmToWav(pcm, 16000, 1);
      let lastErr = null;
      for (const c of chain) {
        try {
          const text = await c.fn(wav);
          return { text, provider: c.p };
        } catch (e) {
          lastErr = { status: e && e.status, code: e && e.code, message: (e && e.message) || String(e), provider: c.p };
        }
      }
      return { text: '', error: lastErr };
    }
  };
}

module.exports = { createSTT };
