/* Defect Augment Studio — 中央状态 + 事件总线
 * 主线程持有编辑器状态的唯一真相；渲染 Worker 不持有业务状态（规格 §11）。
 */
import { uid, normAppearance, elasticMesh, clampMesh, instCorners, pointInQuad } from './util.js';

export const bus = new EventTarget();
export function emit(type, detail) { bus.dispatchEvent(new CustomEvent(type, { detail })); }
export function on(type, fn) { bus.addEventListener(type, (e) => fn(e.detail)); }

export const state = {
  mode: 'edit',                 // edit | extract
  tool: 'select',               // select | pan | poly | rect | ellipse | brush | mesh | persp
  target: null,                 // {id,name,w,h,bitmap, blob|null, proxyScale}
  source: null,                 // 抠图模式源图 {name,w,h,bitmap,proxyScale}
  instances: [],                // z 序：index 0 最底，末位最上
  selection: new Set(),         // instance_id 集合
  view: { scale: 1, tx: 0, ty: 0 },
  clipboard: null,              // [instance 副本]
  editCtx: null,                // {instanceId, kind:'mesh'|'persp'} 双击进入的变形编辑
  spacePan: false,
  cursor: { x: 0, y: 0, inside: false },
  marquee: null,                // {x0,y0,x1,y1} 图像坐标
  snapGuides: [],               // [{axis:'x'|'y', at}] 图像坐标
  recipe: { recording: false, steps: [], srcDims: null, startedAt: null },
  assets: new Map(),            // asset_id -> asset（由 assets.js 填充）
  dpr: window.devicePixelRatio || 1,
  ui: {
    grid: { on: false, size: 16 },
    snap: true,
    boundaryHighlight: false,
    split: { on: false, pos: 0.5 },
    magnifier: { on: false, zoom: 3, r: 90 },
    poissonPreview: true,
    showOriginalOnly: false,
    heatmap: false,        // 放置适宜度热力图叠加
    smartPlace: true,      // 智能吸附放置（平滑区域）
  },
};

export const PROXY_MAX = 2048;  // 编辑代理图最长边（规格 §7.2）

/* ---------------- 实例 ---------------- */

export function createInstance(asset, x, y, opts = {}) {
  const variant = opts.variant || null;
  let mesh = null;
  if (variant?.form_seed) {
    // 形态变体：按种子生成弹性扰动网格（规格 §6.2.4）
    mesh = clampMesh(elasticMesh(variant.form_seed, 4, 4, asset.w, asset.h, 0.22), asset.w, asset.h);
  }
  // 绘制类缺陷（平色块、无内部梯度）走 Poisson 会被梯度域洗掉 → 默认 Feather；
  // 全分辨率烘焙的资产按记录的代理/原图比例缩放（画布上与绘制时等大，导出 1:1）
  const painted = !!asset.source?.painted;
  const baseScale = painted && asset.source?.scale > 0 ? asset.source.scale : 1;
  const inst = {
    id: uid(),
    asset_id: asset.id,
    name: asset.name,
    position: { x, y },
    affine: {
      scale_x: opts.scale_x ?? baseScale, scale_y: opts.scale_y ?? baseScale,
      rotation: opts.rotation ?? 0, shear_x: 0, shear_y: 0,
    },
    perspective_h: null,
    mesh,
    appearance: normAppearance(variant?.appearance || opts.appearance),
    blend: opts.blend || (painted ? 'feather' : 'poisson'),
    blend_flag: 'normal',
    feather_radius: opts.feather_radius ?? (painted ? 2 : 3),
    edge_blur: opts.edge_blur ?? 0,
    opacity: 1,
    noise_seed: (Math.random() * 0x7fffffff) | 0,
    previewPatch: null,
    variant_id: variant?.variant_id || null,
  };
  return inst;
}

/* ---------------- 选择辅助 ---------------- */

export const selectedInstances = () => state.instances.filter((i) => state.selection.has(i.id));

export function setSelection(ids) {
  state.selection = new Set(ids);
  emit('selection');
  emit('render');
}

export function selectOnly(id) { setSelection(id ? [id] : []); }

export function topInstanceAt(pt) {
  // pt: 图像坐标；自顶向下找第一个变形包围体包含该点的实例
  for (let i = state.instances.length - 1; i >= 0; i--) {
    const inst = state.instances[i];
    if (pointInInstance(inst, pt)) return inst;
  }
  return null;
}

let _hitAssetData = new Map(); // asset_id -> {w,h,alpha:Uint8ClampedArray}
export function pointInInstance(inst, pt) {
  const asset = state.assets.get(inst.asset_id);
  if (!asset) return false;
  const { w, h } = asset;
  const m = inst.affine, w2 = w / 2, h2 = h / 2;
  // 快速路径：仿射逆变换 + 局部范围
  const rot = (m.rotation ?? 0) * Math.PI / 180;
  const cos = Math.cos(-rot), sin = Math.sin(-rot);
  const shx = m.shear_x ?? 0, shy = m.shear_y ?? 0;
  const dx = pt.x - inst.position.x, dy = pt.y - inst.position.y;
  const sx0 = (m.scale_x ?? 1), sy0 = (m.scale_y ?? 1);
  // 逆链：X = S⁻¹·SH⁻¹·R⁻¹·p
  const ux = (dx * cos - dy * sin), uy = (dx * sin + dy * cos);
  const det = 1 - shx * shy;
  const vx = (ux - shx * uy) / (det * sx0), vy = (uy - shy * ux) / (det * sy0);
  const lx = vx + w2, ly = vy + h2;
  if (lx < 0 || ly < 0 || lx >= w || ly >= h) return false;
  if (!inst.perspective_h && !inst.mesh) {
    return alphaAt(asset, lx, ly);
  }
  // 带 H/mesh：用变形角点四边形包含测试
  return pointInQuad(pt, instCorners(inst, w, h)) && alphaAt(asset, lx, ly);
}

function alphaAt(asset, x, y) {
  let d = _hitAssetData.get(asset.id);
  if (!d) {
    return true; // 数据未就绪时退化为包围体命中
  }
  const idx = ((y | 0) * d.w + (x | 0)) * 4 + 3;
  return d.data[idx] > 16;
}

/* 缺陷亮度统计（自适应外观用）：alpha>64 像素的 luma 百分位 */
export function assetLumaStats(assetId) {
  const d = _hitAssetData.get(assetId);
  if (!d) return null;
  const arr = d.data;
  const step = Math.max(1, Math.floor((d.w * d.h) / 40000));
  const luma = [];
  for (let i = 0; i < arr.length; i += 4 * step) {
    if (arr[i + 3] > 64) {
      luma.push(0.114 * arr[i] + 0.587 * arr[i + 1] + 0.299 * arr[i + 2]);
    }
  }
  if (luma.length < 20) return null;
  luma.sort((a, b) => a - b);
  const pct = (p) => luma[Math.min(luma.length - 1, Math.floor(p * luma.length))];
  return { p05: pct(0.05), p50: pct(0.5), p95: pct(0.95) };
}

export function registerHitData(assetId, w, h, data) { _hitAssetData.set(assetId, { w, h, data }); }
export function forgetHitData(assetId) { _hitAssetData.delete(assetId); }

/* ---------------- 视图 ---------------- */

export function imageToScreen(p) {
  return { x: p.x * state.view.scale + state.view.tx, y: p.y * state.view.scale + state.view.ty };
}
export function screenToImage(p) {
  return { x: (p.x - state.view.tx) / state.view.scale, y: (p.y - state.view.ty) / state.view.scale };
}

export function fitView(pad = 0.9) {
  if (!state.target) return;
  const cw = state._canvasW || 800, ch = state._canvasH || 600;
  const s = Math.min(cw / state.target.w, ch / state.target.h) * pad;
  state.view.scale = s;
  state.view.tx = (cw - state.target.w * s) / 2;
  state.view.ty = (ch - state.target.h * s) / 2;
  emit('render');
}

export function zoomAt(screenPt, factor) {
  const v = state.view;
  const ns = Math.min(64, Math.max(0.01, v.scale * factor));
  const k = ns / v.scale;
  v.tx = screenPt.x - (screenPt.x - v.tx) * k;
  v.ty = screenPt.y - (screenPt.y - v.ty) * k;
  v.scale = ns;
  emit('render');
}

/* ---------------- 实例增删（历史栈由 history.js 包装） ---------------- */

export function instanceIndex(id) { return state.instances.findIndex((i) => i.id === id); }
export function getInstance(id) { return state.instances.find((i) => i.id === id); }

export function addInstanceRaw(inst, index) {
  if (index == null || index > state.instances.length) index = state.instances.length;
  state.instances.splice(index, 0, inst);
  return inst;
}
export function removeInstanceRaw(id) {
  const idx = instanceIndex(id);
  if (idx >= 0) return state.instances.splice(idx, 1)[0];
  return null;
}

export function raiseSelection() {
  // 将选中实例移到最上层（保持相对顺序）
  const sel = state.instances.filter((i) => state.selection.has(i.id));
  if (!sel.length || sel.length === state.instances.length) return;
  state.instances = state.instances.filter((i) => !state.selection.has(i.id)).concat(sel);
  emit('render');
}
