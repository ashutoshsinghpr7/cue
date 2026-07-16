// Simple JSON-file settings store (avoids native modules so `npm install` stays clean).
const fs = require('fs');
const path = require('path');
const { app } = require('electron');

const FILE = path.join(app.getPath('userData'), 'cue-data.json');

const DEFAULTS = {
  provider: 'openai',
  smart: false,
  apiKeys: { openai: '', anthropic: '', gemini: '', deepseek: '', groq: '', deepgram: '' },
  models: {
    openai: { fast: 'gpt-4o-mini', smart: 'gpt-4o' },
    anthropic: { fast: 'claude-3-5-haiku-latest', smart: 'claude-3-5-sonnet-latest' },
    gemini: { fast: 'gemini-1.5-flash', smart: 'gemini-1.5-pro' },
    deepseek: { fast: 'deepseek-chat', smart: 'deepseek-v4-pro' },
    groq: { fast: 'llama-3.3-70b-versatile', smart: 'meta-llama/llama-4-maverick-17b-128e-instruct' }
  },
  visionFallbackModel: 'meta-llama/llama-4-scout-17b-16e-instruct'
};

let data = null;

function deepMerge(base, over) {
  const out = Array.isArray(base) ? base.slice() : { ...base };
  for (const k of Object.keys(over || {})) {
    if (over[k] && typeof over[k] === 'object' && !Array.isArray(over[k]) && typeof base[k] === 'object') {
      out[k] = deepMerge(base[k], over[k]);
    } else {
      out[k] = over[k];
    }
  }
  return out;
}

function load() {
  if (data) return data;
  try { data = deepMerge(DEFAULTS, JSON.parse(fs.readFileSync(FILE, 'utf8'))); }
  catch { data = deepMerge(DEFAULTS, {}); }
  return data;
}
function save() { try { fs.writeFileSync(FILE, JSON.stringify(data, null, 2)); } catch (e) { /* ignore */ } }

module.exports = {
  getSettings() { return load(); },
  setSettings(patch) { load(); data = deepMerge(data, patch || {}); save(); return data; }
};
