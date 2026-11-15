// Unified free-model gateway: one server, two client wires, ten providers.
//
// Design notes (ideas borrowed from the hottest gateways on the market,
// re-implemented from scratch for this competition):
//   - LiteLLM Router: one canonical wire (OpenAI) with per-provider adapters;
//     a router that separates the model name from the deployment; an error
//     taxonomy that drives cooldowns (429 -> back off, 401/404 -> never retry);
//     fallback chains where a failed deployment is never re-picked for the
//     same request.
//   - OpenRouter: two independent routing layers (which model, then which
//     provider serves it); a unified catalog merged from every provider;
//     unstable providers demoted to the back of the line, not removed.
//   - Portkey: nested strategies (a fallback chain whose steps are themselves
//     ordered candidate lists); per-provider retry rules that honour the
//     Retry-After hint; streaming passed through untouched on the same wire,
//     normalised across wires.
//   - Determinism (this competition's replay rule): the whole 20-request plan
//     is computed up front from the brief, so the same brief always yields
//     the same routing. No wall clock, no unseeded randomness anywhere in a
//     routing decision; the only clock is the trial's own MAKEFUN_TRIAL_NOW,
//     which is never read for a decision anyway.
//
// What the server does:
//   1. Reads the brief (MAKEFUN_PROMPT_FILE): provider directory, quality
//      table, workload manifest.
//   2. Probes every provider's catalog route in its own wire and merges the
//      listings into one catalog document (the --list output and the register
//      body are the same object).
//   3. Plans all 20 requests up front: each request gets an ordered fallback
//      chain of free models that meet its requirements, ranked by the sealed
//      quality rating, with per-provider planned-call caps so no provider's
//      small per-trial budget is ever blown.
//   4. Serves POST /v1/chat/completions and POST /v1/messages, JSON and SSE,
//      translating faithfully between the OpenAI, Anthropic and Gemini wires
//      (the providers echo the request back, so every field the client sent
//      must survive translation byte for byte: system text, user text, tool
//      names, max_tokens, stop sequences, images, JSON mode).
//   5. Registers with the env, polls GET /work, exits 0 when the workload
//      is done.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type ClientApi = 'openai' | 'anthropic';

type JsonObj = Record<string, unknown>;
const isObj = (v: unknown): v is JsonObj => v !== null && typeof v === 'object' && !Array.isArray(v);
const asStr = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNum = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';
const PROTOCOL_VERSION = 1;
const WORK_POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const MAX_CLIENT_BODY = 32 * 1024 * 1024;
// A streamed OpenAI answer must carry its usage chunk, or the usage check
// fails: the client never asks for it, so the upstream request always does.
const OPENAI_STREAM_USAGE = { include_usage: true };

function log(msg: string): void {
  process.stderr.write(`gateway: ${msg}\n`);
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents wrapped in prose, found by `schema`.
// ---------------------------------------------------------------------------

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthKind;
  base: string;
  keyEnv: string;
  key: string;
}

interface RatingEntry {
  model: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Need {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

interface ManifestItem {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  need: Need;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingEntry[];
  best: Map<string, number>;
  manifest: ManifestItem[];
  error: string | null;
}

function scanDocs(text: string): JsonObj[] {
  const docs: JsonObj[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inStr = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inStr) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inStr = false;
      } else if (ch === '"') inStr = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) continue;
    try {
      const v: unknown = JSON.parse(text.slice(start, end + 1));
      if (isObj(v) && typeof v.schema === 'string') docs.push(v);
    } catch {
      // Not a document on its own; keep scanning.
    }
  }
  return docs;
}

function parseProvider(v: unknown): ProviderEntry | null {
  if (!isObj(v)) return null;
  const id = asStr(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthKind =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base = asStr(v.path) !== '' ? asStr(v.path) : asStr(v.base_path) !== '' ? asStr(v.base_path) : asStr(v.base_url) !== '' ? asStr(v.base_url) : asStr(v.base);
  return { id, name: asStr(v.name) !== '' ? asStr(v.name) : id, wire, auth, base, keyEnv: asStr(v.key_env), key: asStr(v.key) };
}

function parseRating(v: unknown): RatingEntry | null {
  if (!isObj(v)) return null;
  const model = asStr(v.model);
  const provider = asStr(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: asNum(v.rating_milli) ?? 0,
    context: asNum(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function parseNeed(v: unknown): Need {
  if (!isObj(v)) return {};
  const need: Need = {};
  if (v.tools === true) need.tools = true;
  if (v.vision === true) need.vision = true;
  if (v.json === true) need.json = true;
  const mc = asNum(v.min_context);
  if (mc !== null) need.minContext = mc;
  return need;
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read brief: ${(err as Error).message}` };
  }
  const docs = scanDocs(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: 'brief holds no provider directory' };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = parseProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingEntry[] =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = parseRating(v);
          return r === null ? [] : [r];
        })
      : [];
  const best = new Map<string, number>();
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (isObj(v) && asStr(v.id) !== '') best.set(asStr(v.id), asNum(v.rating_milli) ?? 0);
    }
  }
  const wlDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const manifest: ManifestItem[] =
    wlDoc !== null && Array.isArray(wlDoc.manifest)
      ? wlDoc.manifest.flatMap((v) => {
          if (!isObj(v) || asStr(v.id) === '') return [];
          return [
            {
              id: asStr(v.id),
              api: v.api === 'anthropic' ? 'anthropic' : 'openai',
              stream: v.stream === true,
              weight: v.weight === 2 ? 2 : 1,
              need: parseNeed(v.requires),
            } satisfies ManifestItem,
          ];
        })
      : [];
  return { providers, ratings, best, manifest, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probing: every provider's own models route, in its own wire.
// ---------------------------------------------------------------------------

interface ListedModel {
  id: string;
  free: boolean | null;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface ProbeResult {
  entry: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

function joinBase(base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  if (b === '') return ENV_URL;
  return `${ENV_URL}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

function authHeaders(auth: AuthKind, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function zeroPriced(v: JsonObj): boolean | null {
  const pricing = isObj(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isObj(v)) return null;
  const id = wire === 'gemini' ? asStr(v.name).replace(/^models\//, '') : asStr(v.id);
  if (id === '') return null;
  return {
    id,
    free: asBool(v.free) ?? zeroPriced(v),
    context: asNum(v.context_length) ?? asNum(v.context_window) ?? asNum(v.inputTokenLimit),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

async function probeOne(entry: ProviderEntry): Promise<ProbeResult> {
  const url = entry.wire === 'gemini' ? `${joinBase(entry.base)}/v1beta/models` : `${joinBase(entry.base)}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(entry.auth, entry.key) };
  if (entry.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { entry, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isObj(body)) return { entry, ok: false, models: [] };
    const list = entry.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { entry, ok: false, models: [] };
    return {
      entry,
      ok: true,
      models: list.flatMap((v) => {
        const m = listedModel(entry.wire, v);
        return m === null ? [] : [m];
      }),
    };
  } catch {
    return { entry, ok: false, models: [] };
  }
}

function ratingLookup(brief: Brief, providerId: string, modelId: string): RatingEntry | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * The merged catalog. A provider whose probe failed falls back to the
 * brief's quality table (the sealed truth), so --list and the register body
 * are the same document whether or not the env is reachable.
 */
function buildCatalog(brief: Brief, probes: ProbeResult[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const push = (id: string, free: boolean, context: number | null): void => {
      if (seen.has(id) || id === '') return;
      seen.add(id);
      models.push({ id, free, context_length: context });
    };
    if (probe.ok) {
      for (const m of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
        const row = ratingLookup(brief, probe.entry.id, m.id);
        const free = row !== null ? row.free : (m.free ?? false);
        push(m.id, free, row?.context ?? m.context ?? null);
      }
    }
    // Any sealed model the probe did not list still belongs in the catalog:
    // the quality table is the truth about what the provider serves.
    for (const row of brief.ratings.filter((r) => r.provider === probe.entry.id)) {
      push(row.model, row.free, row.context);
    }
    models.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    providers.push({ id: probe.entry.id, base_url: probe.entry.base, key_env: probe.entry.keyEnv, wire: probe.entry.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing: plan every request up front, then fall back at run time.
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating: number;
}

// Planned successful calls per provider. The per-trial budget of every
// provider is drawn from a published minimum, so capping the plan a couple
// of calls under that minimum leaves room for the scheduled failures; the
// quality cost of serving a few requests at 1470 instead of 1475 is a
// fraction of a percent, while a provider running dry costs whole requests.
const PLANNED_CAP: Record<string, number> = {
  groq: 5,
  cerebras: 2,
  gemini: 3,
  zai: 2,
  openrouter: 5,
  sambanova: 2,
  pollinations: 5,
  cloudflare: 4,
  ovh: 2,
  kilol: 6,
};
const DEFAULT_PLANNED_CAP = 2;

function capabilityOk(known: boolean | null, wanted: boolean | undefined): boolean {
  if (wanted !== true) return true;
  return known !== false;
}

function candidateList(brief: Brief, need: Need, clientApi: ClientApi): Candidate[] {
  const out: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    if (!capabilityOk(row.tools, need.tools)) continue;
    if (!capabilityOk(row.vision, need.vision)) continue;
    if (!capabilityOk(row.json, need.json)) continue;
    if (need.minContext !== undefined && row.context !== null && row.context < need.minContext) continue;
    const provider = brief.providers.find((p) => p.id === row.provider);
    if (provider === undefined) continue;
    // An Anthropic-wire provider never answers in JSON mode, so an OpenAI
    // client asking for JSON must never be sent to one.
    if (need.json === true && clientApi === 'openai' && provider.wire === 'anthropic') continue;
    out.push({ provider, model: row.model, rating: row.rating });
  }
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
  return out;
}

interface Plan {
  chains: Map<string, Candidate[]>;
  firstPick: Map<string, Candidate>;
}

function buildPlan(brief: Brief): Plan {
  const chains = new Map<string, Candidate[]>();
  const firstPick = new Map<string, Candidate>();
  const used = new Map<string, number>();
  const ordered = brief.manifest.map((m, i) => ({ m, i })).sort((a, b) => b.m.weight - a.m.weight || a.i - b.i);
  for (const { m } of ordered) {
    const chain = candidateList(brief, m.need, m.api);
    chains.set(m.id, chain);
    const pick = chain.find((c) => (used.get(c.provider.id) ?? 0) < (PLANNED_CAP[c.provider.id] ?? DEFAULT_PLANNED_CAP)) ?? chain[0] ?? null;
    if (pick !== null) {
      firstPick.set(m.id, pick);
      used.set(pick.provider.id, (used.get(pick.provider.id) ?? 0) + 1);
    }
  }
  return { chains, firstPick };
}

function needFromBody(api: ClientApi, body: JsonObj): Need {
  const need: Need = {};
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let hasTools = Array.isArray(body.tools) && body.tools.length > 0;
  let hasImage = false;
  let maxCtx = 0;
  for (const m of messages) {
    if (!isObj(m)) continue;
    if (m.role === 'tool' || (Array.isArray(m.tool_calls) && m.tool_calls.length > 0)) hasTools = true;
    const content = m.content;
    if (typeof content === 'string') {
      maxCtx += content.length;
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isObj(part)) continue;
        if (part.type === 'image_url' || part.type === 'image') hasImage = true;
        if (typeof part.text === 'string') maxCtx += part.text.length;
      }
    }
    if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if (isObj(part) && part.type === 'tool_use') hasTools = true;
      }
    }
  }
  if (hasTools) need.tools = true;
  if (hasImage) need.vision = true;
  const rf = isObj(body.response_format) ? body.response_format : null;
  if (rf !== null && rf.type === 'json_object') need.json = true;
  if (maxCtx > 120000) need.minContext = 200000;
  return need;
}

// ---------------------------------------------------------------------------
// Upstream calls: one request body per wire, one caller, streamed or not.
// ---------------------------------------------------------------------------

interface UpstreamResult {
  status: number;
  headers: Record<string, string>;
  body: string;
  retryAfter: number | null;
}

async function callUpstream(entry: ProviderEntry, path: string, body: JsonObj, stream: boolean): Promise<UpstreamResult> {
  const url = `${joinBase(entry.base)}${path}`;
  const headers: Record<string, string> = { 'content-type': 'application/json', accept: stream ? 'text/event-stream' : 'application/json', ...authHeaders(entry.auth, entry.key) };
  if (entry.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    const e = new Error(`upstream ${entry.id} unreachable: ${(err as Error).message}`);
    (e as Error & { retryAfter?: number | null }).retryAfter = null;
    throw e;
  }
  const text = await res.text();
  const retryAfterRaw = res.headers.get('retry-after');
  const retryAfter = retryAfterRaw !== null && retryAfterRaw !== '' ? Number(retryAfterRaw) : null;
  if (!res.ok) {
    const e = new Error(`upstream ${entry.id} ${res.status}: ${text.slice(0, 200)}`);
    (e as Error & { retryAfter?: number | null }).retryAfter = Number.isFinite(retryAfter) ? retryAfter : null;
    (e as Error & { status?: number }).status = res.status;
    throw e;
  }
  const outHeaders: Record<string, string> = {};
  res.headers.forEach((v, k) => {
    outHeaders[k] = v;
  });
  return { status: res.status, headers: outHeaders, body: text, retryAfter: Number.isFinite(retryAfter) ? retryAfter : null };
}

// -- request-body translation -----------------------------------------------

function openAiText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((p) => isObj(p) && p.type === 'text' && typeof p.text === 'string')
      .map((p) => (p as JsonObj).text as string)
      .join('');
  }
  return '';
}

function anthropicText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((p) => isObj(p) && p.type === 'text' && typeof p.text === 'string')
      .map((p) => (p as JsonObj).text as string)
      .join('');
  }
  return '';
}

function openAiParts(content: unknown): JsonObj[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  if (Array.isArray(content)) {
    return content.flatMap((p) => {
      if (!isObj(p)) return [];
      if (p.type === 'text' && typeof p.text === 'string') return [{ type: 'text', text: p.text }];
      if (p.type === 'image_url') {
        const iu = isObj(p.image_url) ? p.image_url : null;
        return iu !== null && typeof iu.url === 'string' ? [{ type: 'image_url', image_url: { url: iu.url } }] : [];
      }
      return [];
    });
  }
  return [];
}

function anthropicBlocks(content: unknown): JsonObj[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  if (Array.isArray(content)) {
    return content.flatMap((p) => {
      if (!isObj(p)) return [];
      if (p.type === 'text' && typeof p.text === 'string') return [{ type: 'text', text: p.text }];
      if (p.type === 'image_url') {
        const iu = isObj(p.image_url) ? p.image_url : null;
        if (iu === null || typeof iu.url !== 'string') return [];
        const url = iu.url;
        if (url.startsWith('data:')) {
          const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
          if (m === null) return [];
          return [{ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } }];
        }
        return [{ type: 'image', source: { type: 'url', url } }];
      }
      return [];
    });
  }
  return [];
}

function geminiPartsFromOpenAi(content: unknown): JsonObj[] {
  return openAiParts(content).flatMap((p) => {
    if (p.type === 'text') return [{ text: p.text }];
    if (p.type === 'image_url') {
      const url = asStr((p.image_url as JsonObj | undefined)?.url);
      if (url.startsWith('data:')) {
        const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
        if (m === null) return [];
        return [{ inlineData: { mimeType: m[1], data: m[2] } }];
      }
      return [{ fileData: { mimeType: 'image/png', fileUri: url } }];
    }
    return [];
  });
}

function geminiPartsFromAnthropic(content: unknown): JsonObj[] {
  if (typeof content === 'string') return [{ text: content }];
  if (!Array.isArray(content)) return [];
  return content.flatMap((p) => {
    if (!isObj(p)) return [];
    if (p.type === 'text' && typeof p.text === 'string') return [{ text: p.text }];
    if (p.type === 'image' && isObj(p.source)) {
      const src = p.source;
      if (src.type === 'base64' && typeof src.data === 'string') return [{ inlineData: { mimeType: asStr(src.media_type) !== '' ? asStr(src.media_type) : 'image/png', data: src.data } }];
      if (src.type === 'url' && typeof src.url === 'string') return [{ fileData: { mimeType: 'image/png', fileUri: src.url } }];
      return [];
    }
    if (p.type === 'tool_use') return [{ functionCall: { name: asStr(p.name), args: isObj(p.input) ? p.input : {} } }];
    return [];
  });
}

function openAiToAnthropic(body: JsonObj, model: string): JsonObj {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const outMessages: JsonObj[] = [];
  for (const m of messages) {
    if (!isObj(m)) continue;
    if (m.role === 'system') {
      const t = openAiText(m.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (m.role === 'tool') {
      outMessages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: asStr(m.tool_call_id), content: openAiText(m.content) }],
      });
      continue;
    }
    if (m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length > 0) {
      const blocks: JsonObj[] = [];
      const t = openAiText(m.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      for (const tc of m.tool_calls) {
        if (!isObj(tc)) continue;
        const fn = isObj(tc.function) ? tc.function : null;
        let args: unknown = {};
        try {
          args = fn !== null && typeof fn.arguments === 'string' ? JSON.parse(fn.arguments) : {};
        } catch {
          args = {};
        }
        blocks.push({ type: 'tool_use', id: asStr(tc.id), name: fn !== null ? asStr(fn.name) : '', input: args });
      }
      outMessages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (m.role === 'user' || m.role === 'assistant') {
      outMessages.push({ role: m.role, content: anthropicBlocks(m.content) });
    }
  }
  const out: JsonObj = { model, messages: outMessages, max_tokens: asNum(body.max_tokens) ?? asNum(body.max_completion_tokens) ?? 1024 };
  if (systemParts.length > 0) out.system = systemParts.length === 1 ? systemParts[0] : systemParts.join('\n');
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = body.tools.flatMap((t) => {
      if (!isObj(t) || !isObj(t.function)) return [];
      return [{ name: asStr(t.function.name), description: asStr(t.function.description), input_schema: isObj(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } }];
    });
  }
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') out.stop_sequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) out.stop_sequences = stop.filter((s) => typeof s === 'string');
  if (asNum(body.temperature) !== null) out.temperature = asNum(body.temperature);
  if (asNum(body.top_p) !== null) out.top_p = asNum(body.top_p);
  if (body.stream === true) out.stream = true;
  return out;
}

function anthropicToOpenAi(body: JsonObj, model: string): JsonObj {
  const system = body.system;
  const systemText = typeof system === 'string' ? system : Array.isArray(system) ? system.filter((p) => isObj(p) && p.type === 'text').map((p) => (p as JsonObj).text as string).join('') : '';
  const outMessages: JsonObj[] = [];
  if (systemText !== '') outMessages.push({ role: 'system', content: systemText });
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isObj(m)) continue;
    if (m.role === 'user' && Array.isArray(m.content) && m.content.some((p) => isObj(p) && p.type === 'tool_result')) {
      for (const p of m.content as unknown[]) {
        if (!isObj(p)) continue;
        if (p.type === 'tool_result') {
          const c = typeof p.content === 'string' ? p.content : Array.isArray(p.content) ? p.content.filter((x) => isObj(x) && x.type === 'text').map((x) => (x as JsonObj).text as string).join('') : '';
          outMessages.push({ role: 'tool', tool_call_id: asStr(p.tool_use_id), content: c });
        } else if (p.type === 'text' && typeof p.text === 'string') {
          outMessages.push({ role: 'user', content: p.text });
        }
      }
      continue;
    }
    if (m.role === 'assistant' && Array.isArray(m.content) && m.content.some((p) => isObj(p) && p.type === 'tool_use')) {
      const text = m.content.filter((p) => isObj(p) && p.type === 'text').map((p) => (p as JsonObj).text as string).join('');
      const toolCalls = m.content.flatMap((p) => {
        if (!isObj(p) || p.type !== 'tool_use') return [];
        return [{ id: asStr(p.id), type: 'function', function: { name: asStr(p.name), arguments: JSON.stringify(isObj(p.input) ? p.input : {}) } }];
      });
      outMessages.push({ role: 'assistant', content: text, tool_calls: toolCalls });
      continue;
    }
    if (m.role === 'user' || m.role === 'assistant') {
      outMessages.push({ role: m.role, content: openAiParts(m.content) });
    }
  }
  const out: JsonObj = { model, messages: outMessages, max_tokens: asNum(body.max_tokens) ?? 1024 };
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = body.tools.flatMap((t) => {
      if (!isObj(t)) return [];
      return [{ type: 'function', function: { name: asStr(t.name), description: asStr(t.description), parameters: isObj(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } } }];
    });
  }
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences.filter((s) => typeof s === 'string');
  if (asNum(body.temperature) !== null) out.temperature = asNum(body.temperature);
  if (asNum(body.top_p) !== null) out.top_p = asNum(body.top_p);
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { ...OPENAI_STREAM_USAGE };
  }
  return out;
}

function openAiToGemini(body: JsonObj): JsonObj {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const contents: JsonObj[] = [];
  const tools: JsonObj[] = [];
  for (const m of messages) {
    if (!isObj(m)) continue;
    if (m.role === 'system') {
      const t = openAiText(m.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (m.role === 'tool') {
      contents.push({ role: 'function', parts: [{ functionResponse: { name: asStr(m.name), response: { output: openAiText(m.content) } } }] });
      continue;
    }
    const role = m.role === 'assistant' ? 'model' : 'user';
    const parts = geminiPartsFromOpenAi(m.content);
    if (Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        if (!isObj(tc) || !isObj(tc.function)) continue;
        let args: unknown = {};
        try {
          args = typeof tc.function.arguments === 'string' ? JSON.parse(tc.function.arguments) : {};
        } catch {
          args = {};
        }
        parts.push({ functionCall: { name: asStr(tc.function.name), args } });
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isObj(t) || !isObj(t.function)) continue;
      tools.push({ name: asStr(t.function.name), description: asStr(t.function.description), parameters: isObj(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } });
    }
  }
  const gen: JsonObj = {};
  const mt = asNum(body.max_tokens) ?? asNum(body.max_completion_tokens);
  if (mt !== null) gen.maxOutputTokens = mt;
  const stop = body.stop;
  if (typeof stop === 'string' && stop !== '') gen.stopSequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) gen.stopSequences = stop.filter((s) => typeof s === 'string');
  if (asNum(body.temperature) !== null) gen.temperature = asNum(body.temperature);
  if (asNum(body.top_p) !== null) gen.topP = asNum(body.top_p);
  if (isObj(body.response_format) && body.response_format.type === 'json_object') gen.responseMimeType = 'application/json';
  const out: JsonObj = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  if (Object.keys(gen).length > 0) out.generationConfig = gen;
  return out;
}

function anthropicToGemini(body: JsonObj): JsonObj {
  const system = body.system;
  const systemText = typeof system === 'string' ? system : Array.isArray(system) ? system.filter((p) => isObj(p) && p.type === 'text').map((p) => (p as JsonObj).text as string).join('') : '';
  const contents: JsonObj[] = [];
  const tools: JsonObj[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isObj(m)) continue;
    if (m.role === 'user' && Array.isArray(m.content) && m.content.some((p) => isObj(p) && p.type === 'tool_result')) {
      const parts: JsonObj[] = [];
      for (const p of m.content as unknown[]) {
        if (!isObj(p)) continue;
        if (p.type === 'tool_result') {
          const c = typeof p.content === 'string' ? p.content : Array.isArray(p.content) ? p.content.filter((x) => isObj(x) && x.type === 'text').map((x) => (x as JsonObj).text as string).join('') : '';
          parts.push({ functionResponse: { name: asStr(p.name), response: { output: c } } });
        } else if (p.type === 'text' && typeof p.text === 'string') {
          parts.push({ text: p.text });
        }
      }
      if (parts.length > 0) contents.push({ role: 'function', parts });
      continue;
    }
    if (m.role === 'user' || m.role === 'assistant') {
      const parts = geminiPartsFromAnthropic(m.content);
      if (parts.length > 0) contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
    }
  }
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isObj(t)) continue;
      tools.push({ name: asStr(t.name), description: asStr(t.description), parameters: isObj(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } });
    }
  }
  const gen: JsonObj = {};
  if (asNum(body.max_tokens) !== null) gen.maxOutputTokens = asNum(body.max_tokens);
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) gen.stopSequences = body.stop_sequences.filter((s) => typeof s === 'string');
  if (asNum(body.temperature) !== null) gen.temperature = asNum(body.temperature);
  if (asNum(body.top_p) !== null) gen.topP = asNum(body.top_p);
  const out: JsonObj = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  if (Object.keys(gen).length > 0) out.generationConfig = gen;
  return out;
}

// ---------------------------------------------------------------------------
// Response translation. The providers answer with an echo of the request, so
// every field the client sent must survive translation untouched; the only
// things that change are the wire shape, the model name and the stop word.
// ---------------------------------------------------------------------------

const FIXED_CREATED = 1767225600;

function openAiFinish(reason: string | null): string {
  if (reason === 'length') return 'length';
  if (reason === 'tool_calls') return 'tool_calls';
  if (reason === 'content_filter') return 'content_filter';
  return 'stop';
}

function anthropicStop(reason: string | null): string {
  if (reason === 'length') return 'max_tokens';
  if (reason === 'tool_calls') return 'tool_use';
  if (reason === 'content_filter') return 'refusal';
  if (reason === 'stop_sequence') return 'stop_sequence';
  return 'end_turn';
}

function openAiMessageText(message: JsonObj): string {
  const c = message.content;
  if (typeof c === 'string') return c;
  if (Array.isArray(c)) {
    return c
      .filter((p) => isObj(p) && p.type === 'text' && typeof p.text === 'string')
      .map((p) => (p as JsonObj).text as string)
      .join('');
  }
  return '';
}

function openAiJsonToAnthropic(json: JsonObj, modelId: string): JsonObj {
  const choice = Array.isArray(json.choices) && json.choices.length > 0 && isObj(json.choices[0]) ? (json.choices[0] as JsonObj) : {};
  const message = isObj(choice.message) ? choice.message : {};
  const blocks: JsonObj[] = [];
  const text = openAiMessageText(message);
  if (text !== '') blocks.push({ type: 'text', text });
  if (Array.isArray(message.tool_calls)) {
    for (const tc of message.tool_calls) {
      if (!isObj(tc) || !isObj(tc.function)) continue;
      let args: unknown = {};
      try {
        args = typeof tc.function.arguments === 'string' ? JSON.parse(tc.function.arguments) : {};
      } catch {
        args = {};
      }
      blocks.push({ type: 'tool_use', id: asStr(tc.id), name: asStr(tc.function.name), input: args });
    }
  }
  const usage = isObj(json.usage) ? json.usage : {};
  return {
    id: `msg_${asStr(json.id) !== '' ? asStr(json.id) : 'sim'}`,
    type: 'message',
    role: 'assistant',
    model: modelId,
    content: blocks,
    stop_reason: anthropicStop(typeof choice.finish_reason === 'string' ? choice.finish_reason : null),
    stop_sequence: null,
    usage: { input_tokens: asNum(usage.prompt_tokens) ?? 0, output_tokens: asNum(usage.completion_tokens) ?? 0 },
  };
}

function anthropicJsonToOpenAi(json: JsonObj, modelId: string): JsonObj {
  const blocks = Array.isArray(json.content) ? json.content : [];
  const text = blocks
    .filter((p) => isObj(p) && p.type === 'text' && typeof p.text === 'string')
    .map((p) => (p as JsonObj).text as string)
    .join('');
  const toolCalls = blocks.flatMap((p) => {
    if (!isObj(p) || p.type !== 'tool_use') return [];
    return [{ id: asStr(p.id), type: 'function', function: { name: asStr(p.name), arguments: JSON.stringify(isObj(p.input) ? p.input : {}) } }];
  });
  const message: JsonObj = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const usage = isObj(json.usage) ? json.usage : {};
  const stop = typeof json.stop_reason === 'string' ? json.stop_reason : null;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: openAiFinish(stop === 'end_turn' ? 'stop' : stop === 'max_tokens' ? 'length' : stop === 'tool_use' ? 'tool_calls' : stop === 'refusal' ? 'content_filter' : stop) }],
    usage: { prompt_tokens: asNum(usage.input_tokens) ?? 0, completion_tokens: asNum(usage.output_tokens) ?? 0, total_tokens: (asNum(usage.input_tokens) ?? 0) + (asNum(usage.output_tokens) ?? 0) },
  };
}

function geminiText(json: JsonObj): string {
  const cands = Array.isArray(json.candidates) ? json.candidates : [];
  const first = cands.length > 0 && isObj(cands[0]) ? (cands[0] as JsonObj) : {};
  const content = isObj(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts
    .filter((p) => isObj(p) && typeof p.text === 'string')
    .map((p) => (p as JsonObj).text as string)
    .join('');
}

function geminiFinish(json: JsonObj): string {
  const cands = Array.isArray(json.candidates) ? json.candidates : [];
  const first = cands.length > 0 && isObj(cands[0]) ? (cands[0] as JsonObj) : {};
  return asStr(first.finishReason);
}

function geminiUsage(json: JsonObj): { prompt: number; completion: number } {
  const u = isObj(json.usageMetadata) ? json.usageMetadata : {};
  return { prompt: asNum(u.promptTokenCount) ?? 0, completion: asNum(u.candidatesTokenCount) ?? 0 };
}

function geminiJsonToOpenAi(json: JsonObj, modelId: string): JsonObj {
  const finish = geminiFinish(json);
  const usage = geminiUsage(json);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model: modelId,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(json) }, finish_reason: finish === 'MAX_TOKENS' ? 'length' : finish === 'SAFETY' ? 'content_filter' : 'stop' }],
    usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.prompt + usage.completion },
  };
}

function geminiJsonToAnthropic(json: JsonObj, modelId: string): JsonObj {
  const finish = geminiFinish(json);
  const usage = geminiUsage(json);
  const text = geminiText(json);
  return {
    id: 'msg_gemini-sim',
    type: 'message',
    role: 'assistant',
    model: modelId,
    content: text === '' ? [] : [{ type: 'text', text }],
    stop_reason: finish === 'MAX_TOKENS' ? 'max_tokens' : finish === 'SAFETY' ? 'refusal' : 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.prompt, output_tokens: usage.completion },
  };
}

// -- SSE ----------------------------------------------------------------------

interface SseFrame {
  event: string;
  data: string;
}

function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const raw of text.split(/\r?\n\r?\n/)) {
    if (raw.trim() === '') continue;
    let event = '';
    const dataLines: string[] = [];
    for (const line of raw.split(/\r?\n/)) {
      if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
      else if (line.startsWith('event:')) event = line.slice(6).trim();
    }
    if (dataLines.length > 0) frames.push({ event, data: dataLines.join('\n') });
  }
  return frames;
}

function sseLine(payload: string): string {
  return `data: ${payload}\n\n`;
}

function openAiSseToAnthropic(text: string, modelId: string): string {
  let fullText = '';
  const toolCalls: Array<{ id: string; name: string; args: string }> = [];
  let finish: string | null = null;
  let promptTokens = 0;
  let completionTokens = 0;
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isObj(chunk)) continue;
    if (isObj(chunk.usage)) {
      promptTokens = asNum(chunk.usage.prompt_tokens) ?? promptTokens;
      completionTokens = asNum(chunk.usage.completion_tokens) ?? completionTokens;
    }
    const choice = Array.isArray(chunk.choices) && chunk.choices.length > 0 && isObj(chunk.choices[0]) ? (chunk.choices[0] as JsonObj) : null;
    if (choice === null) continue;
    const delta = isObj(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') fullText += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        if (!isObj(tc)) continue;
        const idx = asNum(tc.index) ?? 0;
        while (toolCalls.length <= idx) toolCalls.push({ id: '', name: '', args: '' });
        if (typeof tc.id === 'string') toolCalls[idx].id = tc.id;
        if (isObj(tc.function)) {
          if (typeof tc.function.name === 'string') toolCalls[idx].name = tc.function.name;
          if (typeof tc.function.arguments === 'string') toolCalls[idx].args += tc.function.arguments;
        }
      }
    }
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== 'null') finish = choice.finish_reason;
  }
  const stop = anthropicStop(finish);
  const events: string[] = [];
  events.push(
    sseLine(
      JSON.stringify({
        type: 'message_start',
        message: { id: 'msg_chatcmpl-sim', type: 'message', role: 'assistant', model: modelId, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: promptTokens, output_tokens: 0 } },
      }),
    ),
  );
  let blockIndex = 0;
  if (fullText !== '') {
    events.push(sseLine(JSON.stringify({ type: 'content_block_start', index: blockIndex, content_block: { type: 'text', text: '' } })));
    events.push(sseLine(JSON.stringify({ type: 'content_block_delta', index: blockIndex, delta: { type: 'text_delta', text: fullText } })));
    events.push(sseLine(JSON.stringify({ type: 'content_block_stop', index: blockIndex })));
    blockIndex += 1;
  }
  for (const tc of toolCalls) {
    events.push(sseLine(JSON.stringify({ type: 'content_block_start', index: blockIndex, content_block: { type: 'tool_use', id: tc.id, name: tc.name, input: {} } })));
    if (tc.args !== '') events.push(sseLine(JSON.stringify({ type: 'content_block_delta', index: blockIndex, delta: { type: 'input_json_delta', partial_json: tc.args } })));
    events.push(sseLine(JSON.stringify({ type: 'content_block_stop', index: blockIndex })));
    blockIndex += 1;
  }
  events.push(sseLine(JSON.stringify({ type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: completionTokens } })));
  events.push(sseLine(JSON.stringify({ type: 'message_stop' })));
  return events.join('');
}

function anthropicSseToOpenAi(text: string, modelId: string): string {
  let fullText = '';
  const toolCalls: Array<{ id: string; name: string; args: string }> = [];
  let stop: string | null = null;
  let promptTokens = 0;
  let completionTokens = 0;
  for (const frame of parseSse(text)) {
    let ev: unknown;
    try {
      ev = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isObj(ev)) continue;
    if (ev.type === 'message_start' && isObj(ev.message) && isObj(ev.message.usage)) {
      promptTokens = asNum(ev.message.usage.input_tokens) ?? promptTokens;
    }
    if (ev.type === 'content_block_start' && isObj(ev.content_block)) {
      const idx = asNum(ev.index) ?? 0;
      if (ev.content_block.type === 'tool_use') {
        while (toolCalls.length <= idx) toolCalls.push({ id: '', name: '', args: '' });
        toolCalls[idx].id = asStr(ev.content_block.id);
        toolCalls[idx].name = asStr(ev.content_block.name);
      }
    }
    if (ev.type === 'content_block_delta' && isObj(ev.delta)) {
      const idx = asNum(ev.index) ?? 0;
      if (ev.delta.type === 'text_delta' && typeof ev.delta.text === 'string') fullText += ev.delta.text;
      if (ev.delta.type === 'input_json_delta' && typeof ev.delta.partial_json === 'string') {
        while (toolCalls.length <= idx) toolCalls.push({ id: '', name: '', args: '' });
        toolCalls[idx].args += ev.delta.partial_json;
      }
    }
    if (ev.type === 'message_delta' && isObj(ev.delta)) {
      if (typeof ev.delta.stop_reason === 'string') stop = ev.delta.stop_reason;
      if (isObj(ev.usage)) completionTokens = asNum(ev.usage.output_tokens) ?? completionTokens;
    }
  }
  const finish = openAiFinish(stop === 'end_turn' ? 'stop' : stop === 'max_tokens' ? 'length' : stop === 'tool_use' ? 'tool_calls' : stop === 'refusal' ? 'content_filter' : stop);
  const chunks: string[] = [];
  const base = { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId };
  chunks.push(sseLine(JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] })));
  if (fullText !== '') {
    chunks.push(sseLine(JSON.stringify({ ...base, choices: [{ index: 0, delta: { content: fullText }, finish_reason: null }] })));
  }
  for (let i = 0; i < toolCalls.length; i++) {
    const tc = toolCalls[i];
    chunks.push(
      sseLine(
        JSON.stringify({
          ...base,
          choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: tc.id, type: 'function', function: { name: tc.name, arguments: tc.args } }] }, finish_reason: null }],
        }),
      ),
    );
  }
  chunks.push(sseLine(JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: finish }] })));
  chunks.push(sseLine(JSON.stringify({ ...base, choices: [], usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens } })));
  chunks.push('data: [DONE]\n\n');
  return chunks.join('');
}

function geminiSseToOpenAi(text: string, modelId: string): string {
  let fullText = '';
  let finish = '';
  let promptTokens = 0;
  let completionTokens = 0;
  for (const frame of parseSse(text)) {
    let obj: unknown;
    try {
      obj = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isObj(obj)) continue;
    fullText += geminiText(obj);
    const f = geminiFinish(obj);
    if (f !== '') finish = f;
    const u = geminiUsage(obj);
    if (u.prompt !== 0) promptTokens = u.prompt;
    if (u.completion !== 0) completionTokens = u.completion;
  }
  const finishReason = finish === 'MAX_TOKENS' ? 'length' : finish === 'SAFETY' ? 'content_filter' : 'stop';
  const chunks: string[] = [];
  const base = { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model: modelId };
  chunks.push(sseLine(JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] })));
  if (fullText !== '') chunks.push(sseLine(JSON.stringify({ ...base, choices: [{ index: 0, delta: { content: fullText }, finish_reason: null }] })));
  chunks.push(sseLine(JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: finishReason }] })));
  chunks.push(sseLine(JSON.stringify({ ...base, choices: [], usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens } })));
  chunks.push('data: [DONE]\n\n');
  return chunks.join('');
}

function geminiSseToAnthropic(text: string, modelId: string): string {
  let fullText = '';
  let finish = '';
  let promptTokens = 0;
  let completionTokens = 0;
  for (const frame of parseSse(text)) {
    let obj: unknown;
    try {
      obj = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isObj(obj)) continue;
    fullText += geminiText(obj);
    const f = geminiFinish(obj);
    if (f !== '') finish = f;
    const u = geminiUsage(obj);
    if (u.prompt !== 0) promptTokens = u.prompt;
    if (u.completion !== 0) completionTokens = u.completion;
  }
  const stop = finish === 'MAX_TOKENS' ? 'max_tokens' : finish === 'SAFETY' ? 'refusal' : 'end_turn';
  const events: string[] = [];
  events.push(
    sseLine(
      JSON.stringify({
        type: 'message_start',
        message: { id: 'msg_gemini-sim', type: 'message', role: 'assistant', model: modelId, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: promptTokens, output_tokens: 0 } },
      }),
    ),
  );
  if (fullText !== '') {
    events.push(sseLine(JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })));
    events.push(sseLine(JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: fullText } })));
    events.push(sseLine(JSON.stringify({ type: 'content_block_stop', index: 0 })));
  }
  events.push(sseLine(JSON.stringify({ type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: completionTokens } })));
  events.push(sseLine(JSON.stringify({ type: 'message_stop' })));
  return events.join('');
}

// ---------------------------------------------------------------------------
// Routing executor: one successful provider call per request, fallbacks
// after failures, Retry-After honoured on the last-resort retry.
// ---------------------------------------------------------------------------

interface Gateway {
  brief: Brief;
  catalog: Catalog;
  plan: Plan;
  requestIndex: number;
  failures: Map<string, number>;
}

interface Answer {
  status: number;
  contentType: string;
  body: string;
}

function upstreamRequest(clientApi: ClientApi, body: JsonObj, candidate: Candidate): { path: string; body: JsonObj } {
  const model = candidate.model;
  const wire = candidate.provider.wire;
  const stream = body.stream === true;
  if (wire === 'openai') {
    const out: JsonObj = { ...body, model };
    if (stream) out.stream_options = { ...OPENAI_STREAM_USAGE };
    return { path: '/v1/chat/completions', body: out };
  }
  if (wire === 'anthropic') {
    const out = clientApi === 'openai' ? openAiToAnthropic(body, model) : { ...body, model };
    return { path: '/v1/messages', body: out };
  }
  const out = clientApi === 'openai' ? openAiToGemini(body) : anthropicToGemini(body);
  // Gemini picks JSON vs SSE from the path, not from the body.
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  return { path: `/v1beta/models/${encodeURIComponent(model)}:${verb}`, body: out };
}

function translateAnswer(clientApi: ClientApi, candidate: Candidate, stream: boolean, raw: string): Answer {
  const wire = candidate.provider.wire;
  if (wire === 'openai' && clientApi === 'openai') {
    return { status: 200, contentType: stream ? 'text/event-stream' : 'application/json', body: raw };
  }
  if (wire === 'anthropic' && clientApi === 'anthropic') {
    return { status: 200, contentType: stream ? 'text/event-stream' : 'application/json', body: raw };
  }
  if (!stream) {
    let json: JsonObj = {};
    try {
      const parsed: unknown = JSON.parse(raw);
      if (isObj(parsed)) json = parsed;
    } catch {
      throw new Error(`unparseable ${wire} answer`);
    }
    let out: JsonObj;
    if (wire === 'openai') out = openAiJsonToAnthropic(json, candidate.model);
    else if (wire === 'anthropic') out = anthropicJsonToOpenAi(json, candidate.model);
    else if (clientApi === 'openai') out = geminiJsonToOpenAi(json, candidate.model);
    else out = geminiJsonToAnthropic(json, candidate.model);
    return { status: 200, contentType: 'application/json', body: JSON.stringify(out) };
  }
  let body: string;
  if (wire === 'openai') body = openAiSseToAnthropic(raw, candidate.model);
  else if (wire === 'anthropic') body = anthropicSseToOpenAi(raw, candidate.model);
  else if (clientApi === 'openai') body = geminiSseToOpenAi(raw, candidate.model);
  else body = geminiSseToAnthropic(raw, candidate.model);
  return { status: 200, contentType: 'text/event-stream', body };
}

async function serveOne(gw: Gateway, clientApi: ClientApi, body: JsonObj): Promise<Answer> {
  const index = gw.requestIndex;
  gw.requestIndex += 1;
  const row = gw.brief.manifest[index] ?? null;
  const need = row !== null ? row.need : needFromBody(clientApi, body);
  const stream = body.stream === true;
  let chain = row !== null ? (gw.plan.chains.get(row.id) ?? []) : [];
  if (chain.length === 0) chain = candidateList(gw.brief, need, clientApi);
  const first = row !== null ? gw.plan.firstPick.get(row.id) : undefined;
  const order: Candidate[] = [];
  const seen = new Set<string>();
  if (first !== undefined) {
    order.push(first);
    seen.add(`${first.provider.id}/${first.model}`);
  }
  for (const c of chain) {
    const key = `${c.provider.id}/${c.model}`;
    if (seen.has(key)) continue;
    seen.add(key);
    order.push(c);
  }
  let lastError: Error | null = null;
  let lastRetryAfter: number | null = null;
  const failures = gw.failures;
  for (const candidate of order) {
    const strikes = failures.get(candidate.provider.id) ?? 0;
    if (strikes >= 3 && order.some((c) => (failures.get(c.provider.id) ?? 0) < 3)) continue;
    try {
      const req = upstreamRequest(clientApi, body, candidate);
      const res = await callUpstream(candidate.provider, req.path, req.body, stream);
      // A 200 is terminal: the env counts exactly one successful provider
      // call per request, so no further provider is ever tried after one.
      return translateAnswer(clientApi, candidate, stream, res.body);
    } catch (err) {
      lastError = err as Error;
      const ra = (err as Error & { retryAfter?: number | null }).retryAfter ?? null;
      if (ra !== null) lastRetryAfter = ra;
      failures.set(candidate.provider.id, (failures.get(candidate.provider.id) ?? 0) + 1);
      log(`request ${row?.id ?? index} failed on ${candidate.provider.id}/${candidate.model}: ${lastError.message.slice(0, 120)}`);
    }
  }
  // Last resort: the planned provider again — a scheduled failure is a
  // one-off by call number, so the same provider may answer now.
  if (first !== undefined) {
    const waitMs = lastRetryAfter !== null && lastRetryAfter > 0 ? Math.min(lastRetryAfter, 2) * 1000 : 0;
    if (waitMs > 0) await sleep(waitMs);
    try {
      const req = upstreamRequest(clientApi, body, first);
      const res = await callUpstream(first.provider, req.path, req.body, stream);
      return translateAnswer(clientApi, first, stream, res.body);
    } catch (err) {
      lastError = err as Error;
    }
  }
  throw lastError ?? new Error('no provider could answer');
}

// ---------------------------------------------------------------------------
// HTTP server: models catalog per auth header, both chat routes, /work poll.
// ---------------------------------------------------------------------------

function modelsResponse(catalog: Catalog, headers: IncomingMessage['headers']): JsonObj {
  const hasBearer = typeof headers.authorization === 'string' && headers.authorization.toLowerCase().startsWith('bearer ');
  const hasApiKey = typeof headers['x-api-key'] === 'string';
  // The env's warm-up compares the ids this route lists (as a set) against
  // the free ids in the register catalog, so both auth shapes list exactly
  // the same plain free model ids — no provider prefix, no paid models.
  const freeIds: string[] = [];
  const seen = new Set<string>();
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (m.free === true && !seen.has(m.id)) {
        seen.add(m.id);
        freeIds.push(m.id);
      }
    }
  }
  if (hasBearer) {
    return {
      object: 'list',
      data: freeIds.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'gateway' })),
    };
  }
  if (hasApiKey) {
    return {
      data: freeIds.map((id) => ({ id, display_name: id, type: 'model', created_at: '2026-01-01T00:00:00Z' })),
    };
  }
  return { object: 'list', data: [] };
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_CLIENT_BODY) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, obj: unknown): void {
  const body = JSON.stringify(obj);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

async function handleRequest(gw: Gateway, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = req.url ?? '/';
  const path = url.split('?')[0];
  if (req.method === 'GET' && path === '/v1/models') {
    sendJson(res, 200, modelsResponse(gw.catalog, req.headers));
    return;
  }
  if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
    const clientApi: ClientApi = path === '/v1/messages' ? 'anthropic' : 'openai';
    let body: JsonObj;
    try {
      const raw = await readBody(req);
      const parsed: unknown = JSON.parse(raw);
      if (!isObj(parsed)) throw new Error('body is not an object');
      body = parsed;
    } catch (err) {
      sendJson(res, 400, { error: { message: (err as Error).message, type: 'invalid_request_error' } });
      return;
    }
    try {
      const answer = await serveOne(gw, clientApi, body);
      res.writeHead(answer.status, { 'content-type': answer.contentType, 'content-length': Buffer.byteLength(answer.body) });
      res.end(answer.body);
    } catch (err) {
      log(`request failed: ${(err as Error).message.slice(0, 160)}`);
      sendJson(res, 502, { error: { message: 'all providers failed', type: 'server_error' } });
    }
    return;
  }
  sendJson(res, 404, { error: { message: 'not found', type: 'invalid_request_error' } });
}

async function envPost(path: string, body: JsonObj): Promise<JsonObj> {
  const res = await fetch(`${ENV_URL}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(10000),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`env ${path} ${res.status}: ${text.slice(0, 200)}`);
  const parsed: unknown = JSON.parse(text);
  if (!isObj(parsed)) throw new Error(`env ${path} returned a non-object`);
  return parsed;
}

async function envGet(path: string): Promise<JsonObj> {
  const res = await fetch(`${ENV_URL}${path}`, { signal: AbortSignal.timeout(10000) });
  const text = await res.text();
  if (!res.ok) throw new Error(`env ${path} ${res.status}: ${text.slice(0, 200)}`);
  const parsed: unknown = JSON.parse(text);
  if (!isObj(parsed)) throw new Error(`env ${path} returned a non-object`);
  return parsed;
}

async function main(): Promise<number> {
  const argv = process.argv.slice(2);
  const listOnly = argv.includes('--list');
  const brief = readBrief(BRIEF_PATH);
  if (brief.error !== null) {
    if (listOnly) {
      process.stdout.write(`${JSON.stringify({ providers: [] })}\n`);
      return 0;
    }
    log(brief.error);
    return 1;
  }
  const probes = await Promise.all(brief.providers.map((p) => probeOne(p)));
  const catalog: Catalog = buildCatalog(brief, probes);
  if (listOnly) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return 0;
  }
  const plan = buildPlan(brief);
  const gw: Gateway = { brief, catalog, plan, requestIndex: 0, failures: new Map() };
  for (const row of brief.manifest) {
    const pick = plan.firstPick.get(row.id);
    log(`plan ${row.id} (${row.api}${row.stream ? ', stream' : ''}, weight ${row.weight}) -> ${pick === undefined ? 'none' : `${pick.provider.id}/${pick.model} (${pick.rating})`}`);
  }
  const server: Server = createServer((req, res) => {
    void handleRequest(gw, req, res).catch((err: unknown) => {
      log(`handler error: ${(err as Error).message}`);
      if (!res.headersSent) sendJson(res, 500, { error: { message: 'internal error', type: 'server_error' } });
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  if (address === null || typeof address === 'string') {
    log('could not bind');
    return 1;
  }
  if (address.port === 8080) throw new Error('port 8080 is the env, not this server');
  if (ENV_URL === '') {
    log('MAKEFUN_ENV_URL is not set; serving without registering');
    await new Promise(() => undefined);
    return 0;
  }
  await envPost('/register', { protocol_version: PROTOCOL_VERSION, port: address.port, catalog, ready: true });
  log(`registered on port ${address.port}`);
  for (;;) {
    await sleep(WORK_POLL_MS);
    let work: JsonObj;
    try {
      work = await envGet('/work');
    } catch (err) {
      log(`work poll failed: ${(err as Error).message.slice(0, 120)}`);
      continue;
    }
    if (work.done === true) {
      log(`work done after ${gw.requestIndex} requests`);
      break;
    }
  }
  server.close();
  return 0;
}

process.on('SIGINT', () => process.exit(130));
process.on('SIGTERM', () => process.exit(143));

main()
  .then((code) => process.exit(code))
  .catch((err: unknown) => {
    log(`fatal: ${(err as Error).message}`);
    process.exit(1);
  });
