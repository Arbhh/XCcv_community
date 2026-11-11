/* Defect Augment Studio — 启动与顶层编排 */
import { state, emit, on, fitView, addInstanceRaw, selectOnly, screenToImage } from './state.js';
import { initRenderer, bumpScene, invalidateTargetB64, captureView } from './render.js';
import { initInteract } from './interact.js';
import { initPanels } from './panels.js';
import { initWorkflows } from './workflows.js';
import { initRecipe } from './recipe.js';
import { initAssets } from './assets.js';
import { initExtractState, initExtractPanel, enterExtract, markEngineAvailability } from './extract.js';
import { initPaintState, initPaintPanel, enterPaint, exitPaint } from './painter.js';
import * as history from './history.js';
import * as api from './api.js';
import { uid, fitSize, deepClone } from './util.js';

const PROXY_MAX = 2048;

function mimeOf(name) {
  const ext = (name || '').split('.').pop().toLowerCase();
  return { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', bmp: 'image/bmp',
    tif: 'image/tiff', tiff: 'image/tiff', webp: 'image/webp' }[ext] || 'image/png';
}

/* ---------------- 目标图像 ---------------- */

export async function openTarget(fileOrBlob, name) {
  const blob = fileOrBlob instanceof Blob ? fileOrBlob : new Blob([fileOrBlob], { type: mimeOf(name) });
  const full = await createImageBitmap(blob, { imageOrientation: 'from-image' });  // 尊重 EXIF 方向
  const { w, h } = fitSize(full.width, full.height, PROXY_MAX);
  const proxy = new OffscreenCanvas(w, h);
  proxy.getContext('2d').drawImage(full, 0, 0, w, h);
  const fullW = full.width, fullH = full.height;
  full.close();
  const bitmap = await createImageBitmap(proxy);
  state.target = {
    id: uid(), name: name || 'target',
    w, h, bitmap,
    proxyScale: fullW > 0 ? w / fullW : 1,   // 代理/原图比例（全分辨率烘焙与 1:1 导出用）
    originalBlob: blob,      // 全分辨率导出用（规格 §7.2）
    fullW, fullH,
  };
  invalidateTargetB64();
  state.instances = [];
  state.selection.clear();
  state.editCtx = null;
  document.body.classList.add('has-target');
  fitView();
  emit('mode');
  emit('render');
  emit('toast', {
    text: `已载入目标图像：${name || 'target'}（原始 ${(state.target.fullW) || '?'}×${(state.target.fullH) || '?'}，代理 ${w}×${h}）`,
    kind: 'ok',
  });
  loadPlacementMap();   // 异步：放置适宜度热力图 + 平滑热点
}

/* 放置适宜度热力图：梯度能量低（表面平滑）处适合贴缺陷（OPA 思想的确定性实现） */
async function loadPlacementMap() {
  const t = state.target;
  if (!t) return;
  try {
    const cnv = new OffscreenCanvas(t.w, t.h);
    cnv.getContext('2d').drawImage(t.bitmap, 0, 0);
    const blob = await cnv.convertToBlob({ type: 'image/png' });
    const buf = new Uint8Array(await blob.arrayBuffer());
    let bin = '';
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    const res = await api.placementMap(btoa(bin), 12);
    if (state.target !== t) return; // 已切换目标图
    const heatBlob = await (await fetch('data:image/png;base64,' + res.heat_b64)).blob();
    const heatBmp = await createImageBitmap(heatBlob);
    t.heatmap = { bitmap: heatBmp, spots: res.spots, w: res.w, h: res.h };
    bumpScene();
    emit('render');
  } catch { /* 热力图不可用（后端离线）时静默降级 */ }
}

/* 智能吸附放置：优先取落点附近的最平滑热点；无落点取全局最优。Alt 拖拽可绕过 */
function snapToSpot(at, opts = {}) {
  const t = state.target;
  if (!state.ui.smartPlace || opts.noSnap || !t?.heatmap?.spots?.length) return at;
  const spots = t.heatmap.spots;
  let best = null, bestScore = -1;
  if (at) {
    const maxDist = Math.min(t.w, t.h) * 0.3;
    for (const s of spots) {
      const d = Math.hypot(s.x - at.x, s.y - at.y);
      if (d < maxDist && s.score > bestScore) { bestScore = s.score; best = s; }
    }
  } else {
    best = spots[0];
  }
  if (best) {
    emit('toast', { text: `已吸附至平滑区域（适宜度 ${(best.score * 100).toFixed(0)}%）`, kind: 'info' });
    return { x: best.x, y: best.y };
  }
  return at;
}

/* ---------------- 放置实例 ---------------- */

export function placeAsset(assetId, at, opts = {}) {
  const asset = state.assets.get(assetId);
  if (!asset || !state.target) return null;
  const snapped = snapToSpot(at, opts);   // 智能吸附：优先平滑区域（Alt/noSnap 可绕过）
  const pos = snapped || { x: state.target.w / 2, y: state.target.h / 2 };
  const inst = (() => {
    const variant = opts.variant || null;
    const { clampMesh, elasticMesh, normAppearance } = window.__das_util;
    let mesh = null;
    if (variant?.form_seed) {
      mesh = clampMesh(elasticMesh(variant.form_seed, 4, 4, asset.w, asset.h, 0.22), asset.w, asset.h);
    }
    // 绘制类缺陷（平色块、无内部梯度）走 Poisson 会被梯度域洗掉 → 默认 Feather；
    // 全分辨率烘焙的资产按记录的代理/原图比例缩放（画布上与绘制时等大，导出 1:1）
    const painted = !!asset.source?.painted;
    const baseScale = painted && asset.source?.scale > 0 ? asset.source.scale : 1;
    return {
      id: uid(), asset_id: asset.id, name: asset.name,
      position: { ...pos },
      affine: {
        scale_x: opts.scale_x ?? baseScale, scale_y: opts.scale_y ?? baseScale,
        rotation: opts.rotation ?? 0, shear_x: 0, shear_y: 0,
      },
      perspective_h: null, mesh,
      appearance: normAppearance(variant?.appearance || opts.appearance),
      blend: opts.blend || (painted ? 'feather' : 'poisson'), blend_flag: 'normal',
      feather_radius: opts.feather_radius ?? (painted ? 2 : 3), edge_blur: opts.edge_blur ?? 0, opacity: 1,
      noise_seed: (Math.random() * 0x7fffffff) | 0,
      previewPatch: null, variant_id: variant?.variant_id || null,
    };
  })();
  const handle = history.beginInteractive(`放置：${asset.name}`);
  addInstanceRaw(inst);
  handle.markAdded(inst);
  selectOnly(inst.id);   // 选中状态需在条目落盘前设置，重做后才能恢复
  handle.end();
  emit('render');
  return inst;
}

export function placeOnCanvas(instParams) {
  if (!instParams.length) return;
  const handle = history.beginInteractive(`Recipe 回放（${instParams.length} 实例）`);
  const ids = [];
  for (const p of instParams) {
    const inst = { ...deepClone(p), id: uid(), previewPatch: null };
    addInstanceRaw(inst);
    handle.markAdded(inst);
    ids.push(inst.id);
  }
  handle.end();
  selectOnly(ids[ids.length - 1]);
  emit('render');
}

/* ---------------- 画布拖放 ---------------- */

function bindDrop(canvas) {
  canvas.addEventListener('dragover', (e) => e.preventDefault());
  canvas.addEventListener('drop', async (e) => {
    e.preventDefault();
    // 资产拖放
    const assetId = e.dataTransfer.getData('das/asset');
    if (assetId && state.mode === 'edit') {
      const at = screenToImage({ x: e.offsetX, y: e.offsetY });
      placeAsset(assetId, at, { noSnap: e.altKey });
      return;
    }
    // 图像文件拖放
    for (const f of e.dataTransfer.files) {
      if (!f.type.startsWith('image/')) continue;
      if (state.mode === 'extract') break;
      if (!state.target) await openTarget(f, f.name);
      else await enterExtract(f, f.name);
      break;
    }
  });
}

/* ---------------- 启动 ---------------- */

async function boot() {
  initExtractState();
  initPaintState();
  // 按钮点击后失焦：防止后续 Enter/Space 误触发残留焦点的工具栏按钮
  // （如点过"帮助"后按 Enter 闭合多边形，会把帮助对话框重新打开）
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (btn) btn.blur();
  });
  // 对话框关闭（含 Esc）时浏览器会把焦点还给触发按钮——同样释放，
  // 否则 Enter 会再次打开对话框拦截整个界面
  document.querySelectorAll('dialog').forEach((d) => {
    d.addEventListener('close', () => {
      if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    });
  });
  const cv = document.getElementById('cv');
  initRenderer(cv);
  initInteract(cv);
  bindDrop(cv);
  initPanels();
  initWorkflows();
  initRecipe();
  initExtractPanel();
  initPaintPanel();
  on('mode', () => {
    document.body.classList.toggle('mode-extract', state.mode === 'extract');
    document.body.classList.toggle('mode-paint', state.mode === 'paint');
    document.body.classList.toggle('has-target', !!state.target || state.mode === 'extract' || state.mode === 'paint');
  });
  document.getElementById('btnPaintMode').onclick = () => enterPaint();
  // 绘制函数暴露（脚本化操作/测试用）；getter 规避 __studio 未就绪的时序问题
  import('./painter.js').then((P) => { window.__painter = P; });

  await initAssets();

  // 后端状态
  const dot = document.getElementById('apiStatus');
  const checkHealth = async () => {
    try {
      await api.health();
      dot.classList.add('ok');
      dot.title = '后端正常';
    } catch {
      dot.classList.remove('ok');
      dot.title = '后端不可达：抠图/泊松/批量功能不可用';
    }
  };
  await checkHealth();
  checkHealth().then(() => api.samStatus()).then((st) => markEngineAvailability(st)).catch(() => { });
  setInterval(checkHealth, 8000);   // 周期轮询：后端重启后自动恢复指示

  // 全局工具（面板与测试钩子共用）
  window.__das_util = await import('./util.js');
  window.__studio = {
    get painter() { return window.__painter || null; },
    openTarget,
    enterExtract: (fileOrBlob, name) => enterExtract(fileOrBlob, name),
    openTargetFromBytes: async (name, bytes) => {
      const buf = bytes instanceof ArrayBuffer ? bytes : new Uint8Array(bytes).buffer;
      await openTarget(new Blob([buf], { type: mimeOf(name) }), name);
    },
    enterPaint: () => enterPaint(),
    exitPaint: () => exitPaint(),
    exitExtract: () => import('./extract.js').then((m) => m.exitExtract()),
    enterExtract: (fileOrBlob, name) => import('./extract.js').then((m) => m.enterExtract(fileOrBlob, name)),
    openSourceFromBytes: async (name, bytes) => {
      const buf = bytes instanceof ArrayBuffer ? bytes : new Uint8Array(bytes).buffer;
      await enterExtract(new Blob([buf], { type: mimeOf(name) }), name);
    },
    placeAsset: (assetId, at, opts) => placeAsset(assetId, at, opts),
    placeByName: (substr, at, opts) => {
      const a = [...state.assets.values()].find((x) => x.name.includes(substr));
      return a ? placeAsset(a.id, at, opts) : null;
    },
    getState: () => state,
    history,
    emit,
    fitView,
    api,
    captureView,
  };

  // 构建标识：便于确认浏览器是否加载到最新前端（旧缓存会显示旧标识）
  const buildTag = '构建 2026-11-05·r3';
  const tagEl = document.getElementById('buildTag');
  if (tagEl) tagEl.textContent = buildTag;
  window.__studio.buildTag = buildTag;
  emit('render');
  console.log('[DAS] ready (' + buildTag + '). 使用 window.__studio 进行脚本化操作。');
}

// 启动兜底：initRenderer 依赖 Worker/transferControlToOffscreen/ResizeObserver，
// 老浏览器缺任一项会抛错并中断整个 boot（所有按钮无反应、界面看似正常）。
// 捕获后给出可见提示与具体缺失项，而不是静默白屏。
boot().catch((e) => {
  console.error('[DAS] 启动失败:', e);
  const box = document.getElementById('bootErrors');
  if (box) {
    const cap = [];
    if (typeof Worker === 'undefined') cap.push('不支持 Web Worker');
    if (!('transferControlToOffscreen' in HTMLCanvasElement.prototype)) cap.push('不支持 OffscreenCanvas（transferControlToOffscreen）——浏览器过旧');
    if (typeof ResizeObserver === 'undefined') cap.push('不支持 ResizeObserver');
    box.style.display = 'block';
    box.textContent = '[启动失败] ' + (e && e.message || e)
      + (cap.length ? '\n[环境缺项] ' + cap.join('；') + '（请改用新版 Chrome/Edge 打开 http://127.0.0.1:8577）' : '')
      + '\n（若提示 MIME 类型错误，多为系统把 .js 关联成了 text/plain，详见 README 常见问题）\n';
  }
});
