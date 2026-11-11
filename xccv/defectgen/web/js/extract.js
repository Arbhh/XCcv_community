/* Defect Augment Studio — 抠图工作流（规格 §6.1）
 * 多边形/矩形/椭圆选区 → GrabCut（默认）/ FastSAM（吞吐）/ MobileSAM（精度）
 * → 笔刷修正 → 边缘收缩/羽化 → 入库。抠图前代理图最长边 ≤2048。
 */
import { state, emit, on, PROXY_MAX, fitView } from './state.js';
import * as api from './api.js';
import { assetFromExtract } from './assets.js';
import { fitSize } from './util.js';

export function initExtractState() {
  state.extract = {
    shape: 'poly', polyPts: [], rubber: null, vertexEdit: false,
    engine: 'grabcut', iters: 4, shrink: 0, feather: 0,
    brush: { size: 24, mode: 'add' },
    maskCanvas: null, maskCtx: null,
    maskRev: 0, maskOn: true,
    running: false, note: '', engineUsed: '',
  };
}

export async function enterExtract(fileOrBlob, name) {
  const img = await blobToBitmap(fileOrBlob);
  const { w, h } = fitSize(img.width, img.height, PROXY_MAX);
  const proxy = new OffscreenCanvas(w, h);
  proxy.getContext('2d').drawImage(img, 0, 0, w, h);
  img.close();
  const bitmap = await createImageBitmap(proxy);
  state.mode = 'extract';
  state.source = { name: name || 'source', w, h, bitmap, proxyScale: w / bitmap.width };
  initExtractState();
  state.extract.maskCanvas = new OffscreenCanvas(w, h);
  state.extract.maskCtx = state.extract.maskCanvas.getContext('2d');
  state.tool = 'poly';
  fitView();
  emit('mode');
  emit('tool');
  emit('render');
}

export function exitExtract() {
  state.mode = 'edit';
  state.source = null;
  state.tool = 'select';
  fitView();
  emit('mode');
  emit('render');
}

export function setExtractShape(shape) {   // shape=null → 仅清除进行中的形状
  const ex = state.extract;
  if (shape) ex.shape = shape;
  ex.polyPts = [];
  ex.rubber = null;
  ex.vertexEdit = false;
  emit('render');
}

/* 框/椭圆选区 → 可编辑多边形（顶点拖拽变形、Alt+单击删点） */
export function selectionToPolygon() {
  const ex = state.extract;
  if (ex.shape === 'rect' && ex.rubber?.length === 2) {
    const [a, b] = ex.rubber;
    ex.polyPts = [{ x: a.x, y: a.y }, { x: b.x, y: a.y }, { x: b.x, y: b.y }, { x: a.x, y: b.y }];
  } else if (ex.shape === 'ellipse' && ex.rubber?.length === 2) {
    const [a, b] = ex.rubber;
    const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
    const rx = Math.abs(b.x - a.x) / 2, ry = Math.abs(b.y - a.y) / 2;
    ex.polyPts = Array.from({ length: 16 }, (_, i) => {
      const t = (i / 16) * Math.PI * 2;
      return { x: cx + Math.cos(t) * rx, y: cy + Math.sin(t) * ry };
    });
  } else if (ex.shape === 'poly' && ex.polyPts.length >= 3) {
    // 已是多边形 → 进入/退出顶点调整
    ex.vertexEdit = !ex.vertexEdit;
    state.tool = ex.vertexEdit ? 'adjust' : 'poly';
    emit('toast', {
      text: ex.vertexEdit
        ? '顶点调整：拖拽微调，Alt+单击删除，Enter 运行抠图；再次点击「多边形」重画'
        : '已退出顶点调整：单击继续加点',
      kind: 'info',
    });
    if (!ex.vertexEdit) {
      document.querySelectorAll('[data-shape]').forEach((x) => x.classList.toggle('active', x.dataset.shape === 'poly'));
    }
    emit('render');
    return;
  } else {
    emit('toast', { text: '请先用矩形/椭圆/多边形圈出缺陷', kind: 'warn' });
    return;
  }
  ex.shape = 'poly';
  ex.rubber = null;
  ex.vertexEdit = true;
  state.tool = 'adjust';   // 转换结果为闭合多边形 → 直接进入调整模式
  document.querySelectorAll('[data-shape]').forEach((x) => x.classList.remove('active'));
  emit('toast', { text: '已转为可编辑多边形：拖拽顶点微调，Alt+单击删除，点击「运行抠图」提取', kind: 'ok' });
  emit('render');
}

export function resetMask() {
  const ex = state.extract;
  if (!ex.maskCtx) return;
  ex.maskCtx.clearRect(0, 0, ex.maskCanvas.width, ex.maskCanvas.height);
  ex.maskRev++;
  ex.note = '';
  emit('render');
}

export function invertMaskFromSelection() {
  // 以当前选区（无后端）直接生成掩码：选区内全前景
  const ex = state.extract;
  const c = ex.maskCtx;
  if (!c) return;
  c.save();
  c.fillStyle = '#fff';
  const pts = ex.polyPts;
  if (ex.shape === 'poly' && pts.length >= 3) {
    c.beginPath();
    pts.forEach((p, i) => i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y));
    c.closePath(); c.fill();
  } else if (ex.rubber && ex.rubber.length === 2) {
    const [a, b] = ex.rubber;
    if (ex.shape === 'rect') c.fillRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
    else {
      c.beginPath();
      c.ellipse((a.x + b.x) / 2, (a.y + b.y) / 2, Math.abs(b.x - a.x) / 2, Math.abs(b.y - a.y) / 2, 0, 0, Math.PI * 2);
      c.fill();
    }
  }
  c.restore();
  ex.maskRev++;
  emit('render');
}

/* 运行抠图引擎 */
export async function runExtract() {
  const ex = state.extract;
  const src = state.source;
  if (!src || ex.running) return;
  const pts = geometry();
  if (!pts) {
    emit('toast', { text: '请先圈出缺陷的大致范围（多边形至少 3 点 / 拖拽矩形或椭圆）', kind: 'warn' });
    return;
  }
  ex.running = true;
  ex.note = '正在运行 ' + engineLabel(ex.engine) + '…';
  emit('extract-state');
  emit('render');
  try {
    const blob = await canvasToBlob(sourceCanvas());
    const res = await api.extract({
      fileBlob: blob,
      engine: ex.engine,
      iters: ex.iters,
      shape: ex.shape,
      geometry: pts,
      shrink: ex.shrink,
      feather: ex.feather,
    });
    const bmp = await (await fetch('data:image/png;base64,' + res.mask_b64)).blob();
    const maskBmp = await createImageBitmap(bmp);
    const c = ex.maskCtx;
    c.save();
    c.globalCompositeOperation = 'copy';
    c.drawImage(maskBmp, 0, 0);
    c.restore();
    maskBmp.close();
    ex.maskRev++;
    ex.engineUsed = res.engine_used;
    ex.note = [res.note, ...(res.notes || [])].filter(Boolean).join('；') || `${engineLabel(res.engine_used)} 完成`;
    if (res.engine_used !== ex.engine && res.note) {
      emit('toast', { text: res.note, kind: 'warn' }); // SAM 未部署降级提示
    }
  } catch (e) {
    ex.note = '抠图失败：' + e.message;
    emit('toast', { text: ex.note, kind: 'err' });
  } finally {
    ex.running = false;
    emit('extract-state');
    emit('render');
  }
}

function geometry() {
  const ex = state.extract;
  if (ex.shape === 'poly') {
    return ex.polyPts.length >= 3 ? ex.polyPts.map((p) => [Math.round(p.x), Math.round(p.y)]) : null;
  }
  if (ex.rubber?.length === 2) {
    const [a, b] = ex.rubber;
    return [Math.round(Math.min(a.x, b.x)), Math.round(Math.min(a.y, b.y)),
      Math.round(Math.max(a.x, b.x)), Math.round(Math.max(a.y, b.y))];
  }
  return null;
}

export function closePoly() {
  const ex = state.extract;
  // 去除双击第二击/抖动产生的重合点
  ex.polyPts = ex.polyPts.filter((p, i) => i === 0
    || Math.hypot(p.x - ex.polyPts[i - 1].x, p.y - ex.polyPts[i - 1].y) > 1.5);
  if (ex.polyPts.length >= 3) {
    ex.rubber = null;
    ex.vertexEdit = true;   // 仅闭合：进入顶点调整，抠图由「运行抠图」显式触发
    state.tool = 'adjust';
    document.querySelectorAll('[data-shape]').forEach((x) => x.classList.remove('active'));
    emit('toast', { text: '多边形已闭合：拖拽顶点微调，Alt+单击删点，点击「运行抠图」提取', kind: 'ok' });
  } else {
    ex.polyPts = ex.polyPts.slice(0, 2);
    emit('toast', { text: '多边形至少需要 3 个点：单击继续加点', kind: 'warn' });
  }
  emit('render');
}

export function saveToAssets() {
  const name = document.getElementById('exName')?.value?.trim();
  const tags = (document.getElementById('exTags')?.value || '').split(/[,，\s]+/).filter(Boolean);
  return assetFromExtract(name, tags);
}

function engineLabel(e) {
  return { grabcut: 'GrabCut（快速）', fastsam: 'FastSAM（吞吐）', mobilesam: 'MobileSAM（精度）' }[e] || e;
}

function sourceCanvas() {
  const src = state.source;
  const c = new OffscreenCanvas(src.w, src.h);
  c.getContext('2d', { willReadFrequently: true }).drawImage(src.bitmap, 0, 0);
  return c;
}

async function canvasToBlob(cnv) {
  // 纯 JS BMP 编码：convertToBlob 在部分环境偶发永久挂起（抠图请求必经此处）
  const { encodeBmpBlob } = await import('./util.js');
  const d = cnv.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, cnv.width, cnv.height).data;
  return encodeBmpBlob(d, cnv.width, cnv.height);
}

async function blobToBitmap(blob) {
  return createImageBitmap(blob, { imageOrientation: 'from-image' });
}

/* ---------------- 抠图面板绑定 ---------------- */

export function initExtractPanel() {
  const $ = (id) => document.getElementById(id);
  $('exEngine').onchange = (e) => { state.extract.engine = e.target.value; };
  $('exIters').oninput = (e) => { state.extract.iters = +e.target.value; $('exItersN').value = e.target.value; };
  for (const b of document.querySelectorAll('[data-shape]')) {
    b.onclick = () => {
      const shape = b.dataset.shape;
      const btns = document.querySelectorAll('[data-shape]');
      if (state.extract.shape === shape && state.tool === shape) {
        // 再次点击已选中的形状工具 → 取消当前形状，进入顶点调整模式
        setExtractShape(shape);          // 清除进行中的点（保留形状类型）
        state.tool = 'adjust';
        btns.forEach((x) => x.classList.remove('active'));
        emit('toast', { text: '已取消形状绘制：调整模式（拖拽顶点微调，Alt+单击删除）', kind: 'info' });
        emit('render');
        return;
      }
      btns.forEach((x) => x.classList.toggle('active', x === b));
      setExtractShape(shape);
      state.tool = shape;
      emit('render');
    };
  }
  $('btnRunExtract').onclick = runExtract;
  $('btnToPoly').onclick = selectionToPolygon;
  $('btnSelOnly').onclick = () => {
    const ex = state.extract;
    if (ex.shape === 'poly' && ex.polyPts.length >= 3) { invertMaskFromSelection(); return; }
    if (ex.rubber?.length === 2) { invertMaskFromSelection(); return; }
    emit('toast', { text: '请先圈出选区（多边形 3 点以上 / 拖拽矩形或椭圆）', kind: 'warn' });
  };
  $('exBrushSize').oninput = (e) => { state.extract.brush.size = +e.target.value; $('exBrushSizeN').value = e.target.value + 'px'; emit('render'); };
  $('exAdd').onclick = () => { state.extract.brush.mode = 'add'; $('exAdd').classList.add('active'); $('exErase').classList.remove('active'); state.tool = 'brush'; emit('render'); };
  $('exErase').onclick = () => { state.extract.brush.mode = 'erase'; $('exErase').classList.add('active'); $('exAdd').classList.remove('active'); state.tool = 'brush'; emit('render'); };
  $('exShrink').oninput = (e) => { state.extract.shrink = +e.target.value; $('exShrinkN').value = e.target.value; };
  $('exFeather').oninput = (e) => { state.extract.feather = +e.target.value; $('exFeatherN').value = e.target.value + 'px'; };
  $('btnResetMask').onclick = resetMask;
  $('btnSaveAsset').onclick = async () => {
    const a = await saveToAssets();
    if (a) $('btnExitExtract').click();
  };
  $('btnExitExtract').onclick = exitExtract;

  on('extract-close-poly', closePoly);
  on('extract-pop-poly', () => { state.extract.polyPts.pop(); emit('render'); });
  on('extract-run', () => runExtract());
  on('extract-cancel', () => {
    const ex = state.extract;
    if (ex.polyPts.length || ex.rubber || ex.vertexEdit) {
      ex.polyPts = [];
      ex.rubber = null;
      ex.vertexEdit = false;
      state.tool = ex.shape || 'poly';   // 回到形状绘制（Esc 重画）
      document.querySelectorAll('[data-shape]').forEach((x) => x.classList.toggle('active', x.dataset.shape === state.tool));
      emit('render');
    } else {
      exitExtract();
    }
  });
}

export async function markEngineAvailability(status) {
  const sel = document.getElementById('exEngine');
  if (!sel) return;
  for (const opt of sel.options) {
    if (opt.value === 'grabcut') continue;
    const ok = status[opt.value];
    opt.disabled = !ok;
    opt.textContent = opt.textContent.replace(/（未部署）|（可用）/, '') + (ok ? '（可用）' : '（未部署）');
  }
}
