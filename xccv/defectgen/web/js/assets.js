/* Defect Augment Studio — 缺陷资产库（规格 §5.1 / §7.4）
 * RGBA 与掩码存 IndexedDB；面板用 256px 压缩预览；全分辨率位图按需解码。
 * 变体（AssetVariant）仅存外观参数与形态种子，共享原始 RGBA。
 */
import { state, emit, registerHitData, forgetHitData } from './state.js';
import * as history from './history.js';
import { uid, fitSize } from './util.js';

const DB_NAME = 'das-assets';
const STORE = 'assets';
let db = null;

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE, { keyPath: 'id' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function initAssets() {
  try {
    db = await openDB();
    const records = await new Promise((resolve, reject) => {
      const r = db.transaction(STORE, 'readonly').objectStore(STORE).getAll();
      r.onsuccess = () => resolve(r.result || []);
      r.onerror = () => reject(r.error);
    });
    for (const rec of records) {
      try { await hydrateAsset(rec); } catch { /* 单条损坏跳过 */ }
    }
    emit('assets');
  } catch {
    emit('toast', { text: 'IndexedDB 不可用，资产仅本次会话有效', kind: 'warn' });
  }
}

async function hydrateAsset(rec) {
  const [rgbaBitmap, maskBitmap] = await Promise.all([
    createImageBitmap(rec.rgba),
    createImageBitmap(rec.mask),
  ]);
  const asset = {
    id: rec.id, name: rec.name, tags: rec.tags || [],
    w: rec.w, h: rec.h, rev: 1,
    rgbaBitmap, maskBitmap,
    previewUrl: URL.createObjectURL(rec.preview),
    variants: rec.variants || [],
    px_per_mm: rec.px_per_mm ?? null,
    source: rec.source || null,
    created_at: rec.created_at,
  };
  state.assets.set(asset.id, asset);
  registerHit(asset);
  return asset;
}

function registerHit(asset) {
  // 主线程保留 alpha 数据用于精确拾取（规格 §8 交互）
  const cnv = new OffscreenCanvas(asset.w, asset.h);
  const c = cnv.getContext('2d');
  c.drawImage(asset.rgbaBitmap, 0, 0);
  const d = c.getImageData(0, 0, asset.w, asset.h);
  registerHitData(asset.id, asset.w, asset.h, d.data);
}

function persistAsset(asset) {
  if (!db) return;
  // 注意：IndexedDB 事务在微任务队列清空后自动提交，必须先异步编码 blob，再开事务写入
  const cnvToBlob = async (src, w, h) => {
    const c = new OffscreenCanvas(w, h);
    c.getContext('2d').drawImage(src, 0, 0);
    return c.convertToBlob({ type: 'image/png' });
  };
  Promise.all([
    cnvToBlob(asset.rgbaBitmap, asset.w, asset.h),
    cnvToBlob(asset.maskBitmap, asset.w, asset.h),
    fetch(asset.previewUrl).then((r) => r.blob()),
  ]).then(([rgba, mask, preview]) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put({
      id: asset.id, name: asset.name, tags: asset.tags,
      w: asset.w, h: asset.h, rgba, mask, preview,
      variants: asset.variants, px_per_mm: asset.px_per_mm,
      source: asset.source, created_at: asset.created_at,
    });
    tx.oncomplete = () => emit('assets');
    tx.onerror = () => emit('toast', { text: '资产持久化失败：' + (tx.error?.message || ''), kind: 'warn' });
  }).catch((e) => emit('toast', { text: '资产持久化失败：' + e.message, kind: 'warn' }));
}

/* ---------------- 创建资产 ---------------- */

export async function addAsset({ name, rgbaCanvas, maskCanvas, tags = [], source = null }) {
  const w = rgbaCanvas.width, h = rgbaCanvas.height;
  const [rgbaBitmap, maskBitmap] = await Promise.all([
    createImageBitmap(rgbaCanvas),
    createImageBitmap(maskCanvas),
  ]);
  // 压缩预览（最长边 256）
  const { w: pw, h: ph } = fitSize(w, h, 256);
  const prev = new OffscreenCanvas(pw, ph);
  prev.getContext('2d').drawImage(rgbaCanvas, 0, 0, pw, ph);
  const prevBlob = await prev.convertToBlob({ type: 'image/png' });

  const asset = {
    id: uid(), name: name || '缺陷_' + new Date().toISOString().slice(5, 16).replace('T', ' '),
    tags, w, h, rev: 1,
    rgbaBitmap, maskBitmap,
    previewUrl: URL.createObjectURL(prevBlob),
    variants: [], px_per_mm: null, source,
    created_at: Date.now(),
  };
  state.assets.set(asset.id, asset);
  registerHit(asset);
  persistAsset(asset);
  emit('assets');
  emit('toast', { text: `资产已入库：${asset.name}（${w}×${h}）`, kind: 'ok' });
  return asset;
}

export function updateAsset(id, patch) {
  const a = state.assets.get(id);
  if (!a) return;
  Object.assign(a, patch);
  a.rev++;
  persistAsset(a);
  emit('assets');
}

export async function deleteAsset(id, force = false) {
  const usedBy = state.instances.filter((i) => i.asset_id === id).length;
  if (usedBy > 0) {
    if (!force) {
      emit('toast', { text: `资产被画布上 ${usedBy} 个实例使用，请先删除实例（或选择连带删除）`, kind: 'warn' });
      return false;
    }
    // 连带删除画布实例（走历史栈，可撤销）
    const deps = state.instances.filter((i) => i.asset_id === id);
    const handle = history.beginInteractive(`删除实例（连带资产，${deps.length} 个）`);
    for (const inst of deps) {
      handle.markRemoved(inst);
      state.instances.splice(state.instances.indexOf(inst), 1);
    }
    handle.end();
    state.selection.clear();
    state.editCtx = null;
  }
  const a = state.assets.get(id);
  if (!a) return false;
  try { a.rgbaBitmap.close(); a.maskBitmap.close(); } catch { }
  URL.revokeObjectURL(a.previewUrl);
  state.assets.delete(id);
  forgetHitData(id);
  if (db) db.transaction(STORE, 'readwrite').objectStore(STORE).delete(id);
  emit('assets');
  emit('selection');
  emit('render');
  emit('toast', { text: `资产已删除：${a.name}`, kind: 'info' });
  return true;
}

/* 变体管理（规格 §5.1 AssetVariant） */
export function addVariant(assetId, { appearance = null, form_seed = null }) {
  const a = state.assets.get(assetId);
  if (!a) return null;
  const v = { variant_id: uid(), appearance, form_seed };
  a.variants.push(v);
  a.rev++;
  persistAsset(a);
  emit('assets');
  return v;
}
export function removeVariant(assetId, variantId) {
  const a = state.assets.get(assetId);
  if (!a) return;
  a.variants = a.variants.filter((v) => v.variant_id !== variantId);
  a.rev++;
  persistAsset(a);
  emit('assets');
}

/* ---------------- 画布 → 资产（抠图入库） ---------------- */

export async function assetFromExtract(name, tags) {
  const ex = state.extract;
  const src = state.source;
  if (!src || !ex.maskCanvas) return null;
  // RGBA = 源图 × 掩码 alpha
  const rgba = new OffscreenCanvas(src.w, src.h);
  const rc = rgba.getContext('2d');
  rc.drawImage(src.bitmap, 0, 0);
  rc.globalCompositeOperation = 'destination-in';
  rc.drawImage(ex.maskCanvas, 0, 0);

  // 裁剪到有效区域（bbox + 边距）
  const d = rc.getImageData(0, 0, src.w, src.h).data;
  let x0 = src.w, y0 = src.h, x1 = 0, y1 = 0;
  for (let y = 0; y < src.h; y++) {
    for (let x = 0; x < src.w; x++) {
      if (d[(y * src.w + x) * 4 + 3] > 8) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }
  if (x1 <= x0 || y1 <= y0) {
    emit('toast', { text: '掩码为空，无法入库', kind: 'warn' });
    return null;
  }
  const m = 12;
  x0 = Math.max(0, x0 - m); y0 = Math.max(0, y0 - m);
  x1 = Math.min(src.w, x1 + 1 + m); y1 = Math.min(src.h, y1 + 1 + m);
  const cw = x1 - x0, ch = y1 - y0;
  const crop = new OffscreenCanvas(cw, ch);
  crop.getContext('2d').drawImage(rgba, x0, y0, cw, ch, 0, 0, cw, ch);
  const maskCrop = new OffscreenCanvas(cw, ch);
  maskCrop.getContext('2d').drawImage(ex.maskCanvas, x0, y0, cw, ch, 0, 0, cw, ch);

  return addAsset({
    name, tags,
    rgbaCanvas: crop, maskCanvas: maskCrop,
    source: { image: src.name, bbox: [x0, y0, cw, ch] },
  });
}
