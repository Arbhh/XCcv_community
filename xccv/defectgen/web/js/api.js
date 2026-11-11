/* Defect Augment Studio — 后端 API 封装 */

async function jsonOrThrow(res) {
  if (!res.ok) {
    let msg = res.status + ' ' + res.statusText;
    try { const j = await res.json(); msg = j.detail || j.error || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

export async function health() { return jsonOrThrow(await fetch('/api/health')); }

export async function samStatus() { return jsonOrThrow(await fetch('/api/sam_status')); }

export async function extract({ fileBlob, engine, iters, shape, geometry, shrink, feather }) {
  const fd = new FormData();
  fd.append('file', fileBlob, 'source.png');
  fd.append('engine', engine);
  fd.append('iters', String(iters));
  fd.append('shape', shape);
  fd.append('geometry', JSON.stringify(geometry));
  fd.append('shrink', String(shrink));
  fd.append('feather', String(feather));
  return jsonOrThrow(await fetch('/api/extract', { method: 'POST', body: fd }));
}

export async function poissonPreview({ targetB64, layerB64, origin, blendFlag, featherRadius, edgeBlur, opacity, harmonize, harmonizeStrength, harmonizeRing, signal }) {
  return jsonOrThrow(await fetch('/api/poisson_preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      target_b64: targetB64, layer_b64: layerB64, origin,
      blend_flag: blendFlag || 'normal',
      feather_radius: featherRadius ?? 4, edge_blur: edgeBlur ?? 0,
      opacity: opacity ?? 1,
      harmonize: harmonize || 'off',
      harmonize_strength: harmonizeStrength ?? 0.8,
      harmonize_ring: harmonizeRing ?? 10,
    }),
    signal,
  }));
}

export async function paintDefect({ baseRawB64, rawW, rawH, maskRawB64, kind, strength, grain, seed, tintRgb, warp, signal }) {
  return jsonOrThrow(await fetch('/api/paint_defect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      base_raw_b64: baseRawB64, raw_w: rawW, raw_h: rawH, mask_raw_b64: maskRawB64,
      kind, strength, grain, seed: seed || 7,
      tint_rgb: Array.isArray(tintRgb) ? tintRgb.map((v) => Math.round(v)) : null,
      // v1.4 语义：depth 坑缘最大弯移 px（此前误读 warp.strength 致滑块静默失效）
      warp: warp && Number.isFinite(warp.depth)
        ? { depth: warp.depth, falloff: warp.falloff, radius: warp.radius,
            core: warp.core, shade: warp.shade, swirl: warp.swirl }
        : null,
    }),
    signal,   // 形貌预览超时中止用（本环境 fetch 偶发永久挂起）
  }));
}

export async function instancePreview({ targetB64, rgbaRawB64, w, h, instance, signal }) {
  return jsonOrThrow(await fetch('/api/instance_preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      target_b64: targetB64, rgba_raw_b64: rgbaRawB64, w, h, instance,
    }),
    signal,
  }));
}

export async function scenePreview({ targetB64, items, signal }) {
  return jsonOrThrow(await fetch('/api/scene_preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      target_b64: targetB64,
      items: items.map((it) => ({
        rgba_raw_b64: it.rgbaRawB64, w: it.w, h: it.h, instance: it.instance,
      })),
    }),
    signal,
  }));
}

export async function rawEncode(rawB64, w, h, fmt = '.png') {
  return jsonOrThrow(await fetch('/api/raw_encode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ raw_b64: rawB64, w, h, fmt }),
  }));
}

export async function placementMap(targetB64, topK = 12) {
  return jsonOrThrow(await fetch('/api/placement_map', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target_b64: targetB64, top_k: topK }),
  }));
}

export async function composite(payload, signal) {
  const res = await fetch('/api/composite', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    let msg = res.status + ' ' + res.statusText;
    try { const j = await res.json(); msg = j.detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  const notes = [];
  try { notes.push(...JSON.parse(res.headers.get('X-Composite-Notes') || '[]')); } catch { /* ignore */ }
  const blob = await res.blob();
  const name = (res.headers.get('Content-Disposition') || '').match(/filename="?([^";]+)"?/)?.[1] || 'composite.png';
  return { blob, name, notes };
}

export const batch = {
  async start(payload) { return jsonOrThrow(await fetch('/api/batch/start', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  })); },
  async status(jid) { return jsonOrThrow(await fetch('/api/batch/status/' + jid)); },
  async cancel(jid) { return jsonOrThrow(await fetch('/api/batch/cancel/' + jid, { method: 'POST' })); },
};
