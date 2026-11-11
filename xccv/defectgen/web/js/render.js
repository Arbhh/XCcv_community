/* Defect Augment Studio — 主线程渲染桥
 * 职责：把状态序列化为帧 JSON 发给渲染 Worker；管理 ImageBitmap 的传输副本；
 * 调度 Poisson 融合预览（异步、防抖、可取消）。
 */
import { state, bus, emit } from './state.js';
import { debounce } from './util.js';
import * as utilRef from './util.js';
import * as api from './api.js';
import { effectiveColor, strokeBox } from './painter.js';

let worker = null;
let scheduled = false;
let sceneRev = 0;
let canvasEl = null;

const assetSentRev = new Map();   // asset_id -> rev 已发送
let imgSent = { target: null, source: null, heatmap: null }; // 已发送的位图对象标识
const layerWaiters = new Map();   // reqId -> resolve
const captureWaiters = new Map(); // reqId -> resolve
let layerReqSeq = 0;
let captureSeq = 0;

export function bumpScene() { sceneRev++; }

export function initRenderer(canvas) {
  canvasEl = canvas;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  const off = canvas.transferControlToOffscreen();
  worker.postMessage({ t: 'init', canvas: off, dpr: state.dpr }, [off]);
  worker.onmessage = onWorkerMessage;

  const wrap = canvas.parentElement;
  const ro = new ResizeObserver(() => {
    const r = wrap.getBoundingClientRect();
    state._canvasW = r.width; state._canvasH = r.height;
    worker.postMessage({ t: 'resize', w: r.width, h: r.height, dpr: state.dpr });
    emit('render');
  });
  ro.observe(wrap);

  bus.addEventListener('render', scheduleFrame);
  scheduleFrame();
}

/* ---------------- 帧发送 ---------------- */

function scheduleFrame() {
  if (scheduled) return;
  scheduled = true;
  // 编辑器为事件驱动重绘（仅在输入变化时），用 setTimeout 合帧即可，
  // 不依赖 rAF——浏览器合成器被占用的环境（如自动化内嵌浏览器）下仍能正常出图。
  setTimeout(() => { scheduled = false; sendFrame(); }, 16);
}

function instSig(inst) {
  return JSON.stringify([
    inst.position, inst.affine, inst.perspective_h, inst.mesh,
    inst.appearance, inst.blend, inst.blend_flag, inst.feather_radius,
    inst.edge_blur, inst.opacity, inst.name,
    inst.harmonize, inst.harmonize_strength, inst.harmonize_ring,
  ]);
}

function updatePreviewBadge(st) {
  const el = document.getElementById('pvBadge');
  if (!el) return;
  if (state.mode !== 'edit' || st === 'on') { el.hidden = true; return; }
  el.hidden = false;
  el.className = 'pv-badge' + (st === 'off' ? ' off' : '');
  el.textContent = st === 'off'
    ? 'Poisson 预览已关闭：画布为近似显示，导出以 Poisson 结果为准'
    : 'Poisson 预览计算中…（当前为近似显示）';
}

function sendFrame() {
  ensureImages();
  if (state.mode === 'edit') for (const inst of state.instances) ensureAsset(inst.asset_id);
  if (state.mode === 'extract') ensureMaskOverlay('extract');
  if (state.mode === 'paint') ensureMaskOverlay('paint');

  const f = state.mode === 'extract' ? extractFrame()
    : state.mode === 'paint' ? paintFrame() : editFrame();
  worker.postMessage({ t: 'frame', frame: f });
  if (state.mode === 'edit') { updatePreviewBadge(f.editPreviewActive); scheduleScenePreview(); }
  else updatePreviewBadge('on');
}

function paintFrame() {
  const p = state.paint;
  const f = baseFrame();
  f.target = state.target ? { w: state.target.w, h: state.target.h } : null;
  f.instances = [];
  const col = effectiveColor();
  f.paint = {
    maskRev: p.maskRev,
    previewRev: p.previewRev,    // 形貌预览层版本（与烘焙同管线，所见即所得）
    brushCursor: p.brush && state.cursor.inside ? state.cursor.img : null,
    // 笔迹大小以原图真实 px 计：光标圈半径按代理比例换算（画布缩放自动跟随）
    brushR: (p.size / 2) * (state.target?.proxyScale > 0 ? state.target.proxyScale : 1),
    eraser: p.brush === 'eraser',
    selBox: p.selected ? strokeBox(p.selected) : null,   // 已选缺陷高亮框
    tint: [col.r, col.g, col.b],   // 掩码预览按当前笔刷颜色着色
  };
  f.overlays = { outlines: [], handles: [], guides: [], marquee: null };
  return f;
}

function baseFrame() {
  return {
    mode: state.mode,
    view: { ...state.view },
    cursor: { ...state.cursor },
    ui: {
      grid: { ...state.ui.grid },
      boundaryHighlight: state.ui.boundaryHighlight,
      split: { ...state.ui.split },
      magnifier: { ...state.ui.magnifier },
      heatmap: state.ui.heatmap,
      showOriginalOnly: state.ui.showOriginalOnly,
    },
    sceneRev,
  };
}

function editFrame() {
  const f = baseFrame();
  f.target = state.target ? { w: state.target.w, h: state.target.h } : null;
  f.instances = state.instances.map((inst) => ({
    id: inst.id, asset_id: inst.asset_id, sig: instSig(inst),
    position: inst.position, affine: inst.affine,
    perspective_h: inst.perspective_h, mesh: inst.mesh,
    appearance: inst.appearance, blend: inst.blend, blend_flag: inst.blend_flag,
    feather_radius: inst.feather_radius, edge_blur: inst.edge_blur ?? 0,
    opacity: inst.opacity,
    noise_seed: inst.noise_seed,
    harmonize: inst.harmonize || 'off',
    harmonize_strength: inst.harmonize_strength ?? 0.8,
    harmonize_ring: inst.harmonize_ring ?? 10,
    previewPatch: inst.previewPatch ? { bitmap: inst.previewPatch.bitmap, x: inst.previewPatch.x, y: inst.previewPatch.y, rev: inst.previewPatch.rev } : null,
  }));
  f.editPreviewRev = scenePreviewRev;   // 整场景预览版本（worker 匹配后整张替换）
  // 预览层是否覆盖当前场景：false 时 worker 只画近似结果，须在 UI 明确提示
  f.editPreviewActive = editPreviewActive();
  f.overlays = buildOverlays();
  return f;
}

function extractFrame() {
  const ex = state.extract;
  const f = baseFrame();
  f.target = state.source ? { w: state.source.w, h: state.source.h } : null;
  f.instances = [];
  f.extract = {
    maskOn: ex.maskOn,
    maskRev: ex.maskRev,
    polygon: ex.shape === 'poly' ? ex.polyPts : null,
    polygonCursor: ex.shape === 'poly' && state.tool === 'poly' && state.cursor.inside && ex.polyPts.length ? state.cursor.img : null,
    shapeRubber: ex.rubber ? { shape: ex.shape, pts: ex.rubber } : null,
    brushCursor: state.tool === 'brush' && state.cursor.inside ? state.cursor.img : null,
    brushR: ex.brush.size / 2,
  };
  f.overlays = { outlines: [], handles: [], guides: [], marquee: null };
  return f;
}

/* ---------------- 叠加层构建（选择轮廓 / 控制柄） ---------------- */

export function buildOverlays() {
  const outlines = [], handles = [];
  const meshPts = [];
  const ec = state.editCtx;

  for (const inst of selectedInstancesSafe()) {
    const asset = state.assets.get(inst.asset_id);
    if (!asset) continue;
    if (ec && ec.instanceId === inst.id && ec.kind === 'mesh') {
      const pts = gridPoints(inst, asset.w, asset.h);
      // 网格线
      const cols = (inst.mesh || { cols: 4 }).cols, rows = (inst.mesh || { rows: 4 }).rows;
      for (let j = 0; j <= rows; j++) {
        outlines.push({ pts: pts.slice(j * (cols + 1), j * (cols + 1) + cols + 1), closed: false, style: 'edit' });
      }
      for (let i = 0; i <= cols; i++) {
        outlines.push({ pts: [pts[i], pts[(rows) * (cols + 1) + i]], closed: false, style: 'edit' });
      }
      pts.forEach((p, idx) => meshPts.push({ x: p.x, y: p.y, on: true, idx }));
    } else {
      outlines.push({ pts: utilRef.instCorners(inst, asset.w, asset.h), closed: true, style: 'sel' });
    }
  }
  if (ec) {
    const inst = state.instances.find((i) => i.id === ec.instanceId);
    const asset = inst && state.assets.get(inst.asset_id);
    if (inst && asset) {
      if (ec.kind === 'persp') {
        const cs = instCornersOf(inst, asset.w, asset.h);
        outlines.push({ pts: cs, closed: true, style: 'edit', dash: true });
        cs.forEach((p, i) => handles.push({ x: p.x, y: p.y, kind: 'persp', idx: i }));
      } else if (ec.kind === 'affine') {
        handles.push(...affineHandles(inst, asset.w, asset.h));
      }
    }
  } else if (state.selection.size === 1) {
    const inst = state.instances.find((i) => state.selection.has(i.id));
    const asset = inst && state.assets.get(inst.asset_id);
    if (inst && asset && !inst.mesh && !inst.perspective_h) {
      handles.push(...affineHandles(inst, asset.w, asset.h));
    }
  }
  return {
    outlines, handles, meshPts,
    guides: state.snapGuides,
    marquee: state.marquee,
  };
}

function selectedInstancesSafe() {
  return state.instances.filter((i) => state.selection.has(i.id));
}

function instCornersOf(inst, w, h) {
  return utilRef.instCorners(inst, w, h);
}

function affineHandles(inst, w, h) {
  const cs = utilRef.instCorners(inst, w, h); // LT RT RB LB
  const m = inst.affine;
  const isGroup = false;
  const hs = [];
  const kinds = ['nw', 'ne', 'se', 'sw'];
  cs.forEach((p, i) => hs.push({ x: p.x, y: p.y, kind: 'scale', idx: i, corner: kinds[i] }));
  for (let i = 0; i < 4; i++) {
    const a = cs[i], b = cs[(i + 1) % 4];
    hs.push({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, kind: 'scaleEdge', idx: i, corner: kinds[i] });
  }
  // 旋转柄：顶边中点上方 26 屏幕像素
  const topMid = { x: (cs[0].x + cs[1].x) / 2, y: (cs[0].y + cs[1].y) / 2 };
  const sp = { x: topMid.x * state.view.scale + state.view.tx, y: topMid.y * state.view.scale + state.view.ty };
  const rp = { x: sp.x, y: sp.y - 26 };
  const rot = {
    x: (rp.x - state.view.tx) / state.view.scale,
    y: (rp.y - state.view.ty) / state.view.scale,
    kind: 'rotate', anchor: topMid,
  };
  hs.push(rot);
  return hs;
}

function gridPoints(inst, w, h) {
  const mesh = inst.mesh || { cols: 4, rows: 4 };
  const pts = [];
  for (let j = 0; j <= mesh.rows; j++)
    for (let i = 0; i <= mesh.cols; i++)
      pts.push(utilRef.instPoint(inst, { x: i * w / mesh.cols, y: j * h / mesh.rows }, w, h));
  return pts;
}

/* ---------------- 位图传输 ---------------- */

async function ensureAsset(assetId) {
  const asset = state.assets.get(assetId);
  if (!asset) return;
  if (assetSentRev.get(assetId) === asset.rev) return;
  assetSentRev.set(assetId, asset.rev);
  try {
    // ImageBitmap 可结构化克隆：postMessage 不带 transfer 列表即复制且源保持可用，
    // 避免部分环境下 createImageBitmap 复制路径挂起的问题。
    worker.postMessage({ t: 'asset', id: asset.id, rev: asset.rev,
      rgba: asset.rgbaBitmap, mask: asset.maskBitmap, w: asset.w, h: asset.h });
    bumpScene();
  } catch { assetSentRev.delete(assetId); }
}

function ensureImages() {
  // 结构化克隆传输位图副本（不 detach 源，见 ensureAsset 注释）
  if (state.target && imgSent.target !== state.target.bitmap) {
    imgSent.target = state.target.bitmap;
    const b = state.target;
    worker.postMessage({ t: 'img', slot: 'target', bitmap: b.bitmap, w: b.w, h: b.h });
    bumpScene();
  }
  const srcBitmap = state.mode === 'extract' ? state.source?.bitmap : null;
  if (srcBitmap && imgSent.source !== srcBitmap) {
    imgSent.source = srcBitmap;
    const b = state.source;
    worker.postMessage({ t: 'img', slot: 'source', bitmap: srcBitmap, w: b.w, h: b.h });
    bumpScene();
  }
  // 放置适宜度热力图
  const heat = state.target?.heatmap;
  if (heat && imgSent.heatmap !== heat.bitmap) {
    imgSent.heatmap = heat.bitmap;
    worker.postMessage({ t: 'img', slot: 'heatmap', bitmap: heat.bitmap, w: heat.w, h: heat.h });
    bumpScene();
  }
}

let maskSentRev = -1;
let maskPosting = false, maskPendingRev = -1;
let paintColorSentRev = -1;
let paintMorphSentRev = -1;
function ensureMaskOverlay(mode) {
  const src = mode === 'extract' ? state.extract : state.paint;
  if (!src || !src.maskCanvas) return;
  const rev = src.maskRev;
  if (rev !== maskSentRev && !maskPosting) {
    maskSentRev = rev;
    maskPosting = true;
    try {
      const img = src.maskCtx.getImageData(0, 0, src.maskCanvas.width, src.maskCanvas.height);
      worker.postMessage({ t: 'mask', image: img, mode });
    } catch { maskSentRev = -1; }
    maskPosting = false;
    if (maskPendingRev > maskSentRev) {
      maskSentRev = -1;
      maskPendingRev = -1;
    }
  }
  if (mode !== 'paint') return;
  // 绘制模式：同步发送每笔各自颜色的预览层
  // （色块/形貌同步必须在 maskRev 门控之外——形貌层由防抖异步生成，算完时
  // maskRev 往往未再变化，若被门控提前 return，消息会滞留到下一次 maskRev
  // 变化才发出，表现为"点击不渲染、撤销/清空后才渲染"）
  if (src.colorCanvas && src.colorRev !== paintColorSentRev) {
    paintColorSentRev = src.colorRev;
    try {
      const img2 = src.colorCtx.getImageData(0, 0, src.colorCanvas.width, src.colorCanvas.height);
      worker.postMessage({ t: 'mask', image: img2, mode: 'paint-color' });
    } catch { paintColorSentRev = -1; }
  }
  // 绘制模式：真实形貌预览层（与烘焙同管线，所见即所得）
  if (src.previewRev !== paintMorphSentRev) {
    paintMorphSentRev = src.previewRev;
    try {
      if (src.previewLayer) {
        const img3 = src.previewLayer.canvas.getContext('2d', { willReadFrequently: true })
          .getImageData(0, 0, src.previewLayer.canvas.width, src.previewLayer.canvas.height);
        worker.postMessage({ t: 'mask', image: img3, mode: 'paint-morph',
          x: src.previewLayer.x, y: src.previewLayer.y, rev: src.previewRev });
      } else {
        worker.postMessage({ t: 'mask', image: null, mode: 'paint-morph', rev: src.previewRev });
      }
    } catch { paintMorphSentRev = -1; }
  }
}

/* ---------------- 融合预览（整场景，与导出同一 render_scene） ----------------
 * 逐实例补丁预览只把该实例合成到原背景上，补丁矩形内会覆盖其它实例——
 * 多缺陷场景下"预览与导出不符"。改为整场景预览：全部实例一起交给后端
 * render_scene（导出用的同一个函数），worker 直接整张替换场景缓冲，
 * 因此预览与导出（view 分辨率）逐像素一致。 */

let scenePreviewJob = null;
let scenePreviewRev = 0;
let scenePreviewSentRev = -1;
let scenePreviewReadySig = '';   // 已成功生成预览对应的场景签名

function editPreviewActive() {
  const need = state.ui.poissonPreview && state.instances.some((i) => i.blend === 'poisson');
  if (!need) return 'off';
  return scenePreviewReadySig === previewSig() ? 'on' : 'pending';
}

function previewSig() {
  return state.instances.map((i) => i.id + ':' + instSig(i) + ':' + i.blend).join('|')
    + '|' + (state.target?.id || '') + '|' + (state.ui.heatmap ? 'h' : '')
    + '|' + (state.ui.showOriginalOnly ? 'o' : '');
}

function clearScenePreview() {
  scenePreviewReadySig = '';
  if (scenePreviewJob) {
    clearTimeout(scenePreviewJob.timer);
    scenePreviewJob.ctrl?.abort();
    scenePreviewJob = null;
  }
  if (scenePreviewSentRev !== -1) {
    scenePreviewSentRev = -1;
    scenePreviewRev++;
    worker.postMessage({ t: 'mask', image: null, mode: 'edit-preview', rev: scenePreviewRev });
    bumpScene();
    emit('render');
  }
}

function scheduleScenePreview() {
  const need = state.ui.poissonPreview && state.instances.some((i) => i.blend === 'poisson');
  if (!need) { clearScenePreview(); return; }
  const sig = previewSig();
  if (scenePreviewJob && scenePreviewJob.sig === sig) return;
  let job = scenePreviewJob;
  if (!job) job = scenePreviewJob = {};
  clearTimeout(job.timer);
  job.ctrl?.abort();
  job.ctrl = new AbortController();
  job.sig = sig;
  job.timer = setTimeout(() => runScenePreview(job, sig), 450);
}

function instPayloadOf(inst) {
  return {
    position: [inst.position.x, inst.position.y],
    affine: { ...inst.affine },
    perspective_h: inst.perspective_h ?? null,
    mesh: inst.mesh ?? null,
    appearance: { ...(inst.appearance || {}) },
    blend: inst.blend,
    blend_flag: inst.blend_flag,
    feather_radius: inst.feather_radius ?? 0,
    edge_blur: inst.edge_blur ?? 0,
    opacity: inst.opacity ?? 1,
    noise_seed: inst.noise_seed ?? 0,
    harmonize: inst.harmonize || 'off',
    harmonize_strength: inst.harmonize_strength ?? 0.8,
    harmonize_ring: inst.harmonize_ring ?? 10,
  };
}

async function runScenePreview(job, sig) {
  if (scenePreviewJob !== job || job.sig !== sig) return;
  if (state._dragging) {   // 拖拽中不重算，抬手后补
    job.timer = setTimeout(() => runScenePreview(job, sig), 400);
    return;
  }
  job.attempt = (job.attempt || 0) + 1;
  try {
    const items = [];
    for (const inst of state.instances) {
      const a = state.assets.get(inst.asset_id);
      if (!a) continue;
      const cnv = new OffscreenCanvas(a.w, a.h);
      const cx = cnv.getContext('2d', { willReadFrequently: true });
      cx.drawImage(a.rgbaBitmap, 0, 0);
      // getImageData 已是 RGBA，直接原样上传——后端按 RGBA→BGRA 解析。
      // 此前又在前端换了一次通道，形成双重交换 → 预览 R/B 颠倒
      //（蓝色缺陷显示成棕黄，而 PNG 导出的颜色正确）
      const d = cx.getImageData(0, 0, a.w, a.h).data;
      const bytes = new Uint8Array(d.length);
      bytes.set(d);
      let s = '';
      for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
      items.push({ rgbaRawB64: btoa(s), w: a.w, h: a.h, instance: instPayloadOf(inst) });
    }
    const targetB64 = await getTargetB64();
    const res = await api.scenePreview({ targetB64, items, signal: job.ctrl.signal });
    if (scenePreviewJob !== job || job.sig !== sig) return;
    const n = res.w * res.h * 4;
    const bin = atob(res.result_raw_b64);
    const px = new Uint8ClampedArray(n);
    for (let i = 0; i < n; i++) px[i] = bin.charCodeAt(i);
    scenePreviewReadySig = sig;
    scenePreviewRev++;
    scenePreviewSentRev = scenePreviewRev;
    worker.postMessage({ t: 'mask', image: new ImageData(px, res.w, res.h),
                         mode: 'edit-preview', rev: scenePreviewRev });
    job.attempt = 0;
    emit('preview-note', { used: 'render_scene', notes: res.notes });
    bumpScene();
    emit('render');
  } catch (e) {
    if (e.name === 'AbortError') return;
    console.warn('[preview] 整场景预览失败（自动重试）:', e);
    // 自动重试：预览层缺失时画布只能显示近似渲染，会与导出的 Poisson 结果不符
    if (scenePreviewJob === job && job.attempt < 3) {
      job.timer = setTimeout(() => runScenePreview(job, sig), 800);
    } else {
      emit('preview-note', { used: 'failed', notes: [String(e.message || e)] });
    }
  }
}

export function requestLayer(inst) {
  return new Promise((resolve, reject) => {
    const reqId = ++layerReqSeq;
    layerWaiters.set(reqId, { resolve, reject });
    worker.postMessage({ t: 'requestLayer', reqId, inst: {
      asset_id: inst.asset_id, position: inst.position, affine: inst.affine,
      perspective_h: inst.perspective_h, mesh: inst.mesh,
      appearance: inst.appearance, noise_seed: inst.noise_seed,
      edge_blur: inst.edge_blur ?? 0,
    } });
    setTimeout(() => {
      if (layerWaiters.has(reqId)) { layerWaiters.delete(reqId); reject(new Error('图层导出超时')); }
    }, 8000);
  });
}

function onWorkerMessage(e) {
  const m = e.data;
  if (m.t === 'workerError') {
    (window.__workerErrors = window.__workerErrors || []).push(m.msg);
    console.error('[render worker]', m.msg);
    return;
  }
  if (m.t === 'layer') {
    const w = layerWaiters.get(m.reqId);
    layerWaiters.delete(m.reqId);
    if (!w) return;
    if (m.error) { w.reject(new Error(m.error)); return; }
    toB64(m.bitmap).then((b64) => {
      w.resolve({ b64, x: m.x, y: m.y, w: m.w, h: m.h, bitmap: m.bitmap });
    });
  } else if (m.t === 'histogram') {
    emit('histogram', { bins: m.bins, total: m.total });
  } else if (m.t === 'capture') {
    const w = captureWaiters.get(m.reqId);
    captureWaiters.delete(m.reqId);
    if (!w) return;
    if (m.error) { w.reject(new Error(m.error)); return; }
    // BMP blob → dataURL（不依赖浏览器图像编码器）
    const reader = new FileReader();
    reader.onload = () => w.resolve(reader.result);
    reader.onerror = () => w.reject(new Error('blob read failed'));
    reader.readAsDataURL(m.blob);
  }
}

/* 应用内截屏：Worker 回传整帧位图（不依赖浏览器合成器） */
export function captureView() {
  return new Promise((resolve, reject) => {
    const reqId = ++captureSeq;
    captureWaiters.set(reqId, { resolve, reject });
    worker.postMessage({ t: 'capture', reqId });
    setTimeout(() => {
      if (captureWaiters.has(reqId)) { captureWaiters.delete(reqId); reject(new Error('capture timeout')); }
    }, 5000);
  });
}

async function toB64(bitmap) {
  // 纯 JS BMP：convertToBlob 偶发永久挂起（后端 cv2.imdecode 可解码 BMP）
  const cnv = new OffscreenCanvas(bitmap.width, bitmap.height);
  const c = cnv.getContext('2d', { willReadFrequently: true });
  c.drawImage(bitmap, 0, 0);
  const d = c.getImageData(0, 0, cnv.width, cnv.height).data;
  const blob = utilRef.encodeBmpBlob(d, cnv.width, cnv.height);
  const buf = new Uint8Array(await blob.arrayBuffer());
  let s = '';
  const CH = 0x8000;
  for (let i = 0; i < buf.length; i += CH) {
    s += String.fromCharCode.apply(null, buf.subarray(i, i + CH));
  }
  return btoa(s);
}

let _targetB64Cache = null, _targetB64Id = null;
async function getTargetB64() {
  if (_targetB64Id === state.target?.id) return _targetB64Cache;
  const cnv = new OffscreenCanvas(state.target.w, state.target.h);
  const c = cnv.getContext('2d', { willReadFrequently: true });
  c.drawImage(state.target.bitmap, 0, 0);
  const d = c.getImageData(0, 0, cnv.width, cnv.height).data;
  const blob = utilRef.encodeBmpBlob(d, cnv.width, cnv.height);
  const buf = new Uint8Array(await blob.arrayBuffer());
  let s = '';
  for (let i = 0; i < buf.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  }
  _targetB64Cache = btoa(s);
  _targetB64Id = state.target.id;
  return _targetB64Cache;
}

export function invalidateTargetB64() { _targetB64Id = null; }
export function invalidateInstancePreview(inst) {
  if (inst.previewPatch) { inst.previewPatch = null; bumpScene(); emit('render'); }
}
