/* Defect Augment Studio — 缺陷绘制工坊（v1.4）
 * 程序化笔刷：白点/黑点/划伤/污渍/凹坑/橡皮 + 自定义颜色/吸管/三色盘。
 * 笔迹以"重放模型"管理：每一笔的 stamp 画在独立画布上（含软化），确认时整层重放，
 * 可撤销。生成资产时按颜色分组把笔迹掩码交给后端 /api/paint_defect 着色
 * （自定义颜色 → tint 混色；否则按笔刷类型自动明暗），合成后入库为可复用资产。
 */
import { state, emit, on } from './state.js';
import * as api from './api.js';
import { pickSaveHandle, saveBlob, uid } from './util.js';
import * as assets from './assets.js';
import { hexToRgb, rgbToHex } from './colorpick.js';

export const KIND_DEFAULT_COLORS = {
  spot_bright: '#f2f2f2',   // 白点
  spot_dark: '#1c1c1c',     // 黑点
  scratch: '#e8e8e8',       // 划伤
  stain: '#7c4a24',         // 污渍
  dent: '#3d3d3d',          // 凹坑
  eraser: '#8a8a8a',
};

export function kindDefaultColor(kind) { return KIND_DEFAULT_COLORS[kind] || '#cccccc'; }

/* 笔迹大小的代理像素尺寸：滑杆以【原图真实 px】计，绘制时按代理/原图比例换算，
 * 保证烘焙（全分辨率）出的缺陷尺寸与设定值严格一致 */
export function brushSizeProxy() {
  const t = state.target;
  const v = state.paint ? state.paint.size : 0;
  return v * (t && t.proxyScale > 0 ? t.proxyScale : 1);
}

/* 当前生效颜色（供渲染 Worker 预览着色与面板色块） */
export function effectiveColor() {
  const { r, g, b } = hexToRgb(state.paint?.color || '#cccccc');
  return { r, g, b, hex: state.paint?.color || '#cccccc' };
}

export function initPaintState() {
  const brush = 'spot_bright';
  state.paint = {
    brush,                  // spot_bright | spot_dark | scratch | stain | dent | eraser
    size: 26,               // 笔迹主尺寸 px（图像坐标）
    irregular: 0.35,        // 0..1 边缘不规则度
    hardness: 0.55,         // 0..1 软硬（1 硬边缘）
    strength: 0.7,          // 0..1 明暗强度（后端着色用）
    grain: 0.3,             // 0..1 颗粒度
    seed: 7,
    customColor: true,      // 自定义颜色（关闭则按笔刷类型自动明暗形貌）
    color: KIND_DEFAULT_COLORS[brush],
    spacing: 1.0,           // 笔迹间距系数（拖拽铺点密度）
    warp: { depth: 22, falloff: 2.0, radius: 1.0, core: 0.35, shade: 0.20, swirl: 0.5 },   // 凹陷酒窝合成：坑缘最大弯移 px / 衰减指数 / 影响带×r_in
    selected: null,         // 当前选中的已画缺陷（null=画笔画笔默认参数）
    adjustMode: false,      // 「调整」工具：点选已画缺陷
    previewLayer: null, previewRev: 0,   // 实时形貌预览层（与烘焙同管线）
    maskCanvas: null, maskCtx: null,
    colorCanvas: null, colorCtx: null,   // 每笔各自颜色的预览层
    maskRev: 0, colorRev: 0,
    strokes: [],            // 重放模型
    cur: null,              // 进行中的笔迹
    note: '',
  };
}

/* 划痕笔迹的种子化 rng：撤销重放完全一致 */
function strokeRng(seed) {
  let s = (seed >>> 0) || 1;
  return () => (s = (s * 1664525 + 1013904223) >>> 0, s / 4294967296);
}

/* 不规则 blob 路径（白点/黑点/污渍单元） */
function blobPath(c, cx, cy, r, irregular, rng) {
  const N = 16;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const rr = r * (1 + irregular * (rng() * 2 - 1) * 0.45);
    pts.push({ x: cx + Math.cos(a) * rr, y: cy + Math.sin(a) * rr });
  }
  c.beginPath();
  c.moveTo((pts[0].x + pts[N - 1].x) / 2, (pts[0].y + pts[N - 1].y) / 2);
  for (let i = 0; i < N; i++) {
    const p = pts[i], q = pts[(i + 1) % N];
    c.quadraticCurveTo(p.x, p.y, (p.x + q.x) / 2, (p.y + q.y) / 2);
  }
  c.closePath();
}

/* 单笔渲染到独立画布（含软化），返回 OffscreenCanvas 或 null */
function renderStroke(stroke, w, h) {
  if (stroke.kind !== 'eraser' && stroke.points.length === 0) return null;
  const cnv = new OffscreenCanvas(w, h);
  const c = cnv.getContext('2d');
  const rng = strokeRng(stroke.seed);
  const isSpot = stroke.kind === 'spot_bright' || stroke.kind === 'spot_dark' || stroke.kind === 'dent';
  c.fillStyle = '#fff';

  if (stroke.kind === 'eraser') {
    // 橡皮：沿路径均匀全强度圆（与拖拽时的即时擦除完全一致——重放必须
    // 复现真实擦除区域，否则抬笔/撤销/形貌重算时被擦区域会弹回）
    for (const pt of stroke.points) {
      c.beginPath();
      c.arc(pt.x, pt.y, stroke.size / 2, 0, Math.PI * 2);
      c.fill();
    }
    return cnv;   // 橡皮不做软化（即时擦除是硬边）
  }

  if (isSpot) {
    // 白点/黑点/凹坑：沿路径铺 blob —— 单击 = 单个缺陷，拖拽 = 一簇连续缺陷
    // （此前只用首尾两点拉长成一个大斑，视觉上"只能点击无法拖动"）
    const pts = stroke.points;
    const r = stroke.size / 2;
    const sp = Math.max(0.15, (stroke.spacing ?? 1) * 0.9);
    let acc = 1e9;
    for (let i = 0; i < pts.length; i++) {
      if (i > 0) acc += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
      if (acc < r * sp) continue;
      acc = 0;
      const jx = (rng() * 2 - 1) * r * 0.18, jy = (rng() * 2 - 1) * r * 0.18;
      const rr = r * (0.72 + rng() * 0.4);
      blobPath(c, pts[i].x + jx, pts[i].y + jy, rr, stroke.irregular, rng);
      c.fill();
    }
  } else {
    // 划伤/污渍：沿路径铺 stamp
    const pts = stroke.points;
    const r = stroke.size / 2;
    let acc = 0;
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1], b = pts[i];
      const d = Math.hypot(b.x - a.x, b.y - a.y);
      acc += d;
      if (stroke.kind === 'scratch') {
        const gap = Math.max(1.2, r / 4);
        if (acc < gap) continue;
        acc = 0;
        const t = i / Math.max(1, pts.length - 1);
        const taper = Math.min(1, Math.min(t, 1 - t) * 6 + 0.25); // 端部收细
        const rr = r * (0.62 + rng() * 0.25) * (0.55 + 0.45 * taper);
        c.beginPath();
        c.arc(b.x, b.y, rr, 0, Math.PI * 2);
        c.fill();
      } else {
        // 污渍：散布 blob 簇
        const gap = r * 0.9;
        if (acc < gap) continue;
        acc = 0;
        const ox = (rng() * 2 - 1) * r * 0.9, oy = (rng() * 2 - 1) * r * 0.9;
        blobPath(c, b.x + ox, b.y + oy, r * (0.45 + rng() * 0.65), stroke.irregular + 0.2, rng);
        c.fill();
      }
    }
  }

  // 软化：硬度 1 → 不模糊
  const softPx = (1 - stroke.hardness) * Math.max(1, stroke.size / 5);
  if (softPx > 0.4) {
    const tmp = new OffscreenCanvas(w, h);
    tmp.getContext('2d').drawImage(cnv, 0, 0);
    c.clearRect(0, 0, w, h);
    c.filter = `blur(${softPx.toFixed(1)}px)`;
    c.drawImage(tmp, 0, 0);
    c.filter = 'none';
  }
  return cnv;
}

/* 重放指定笔迹子集 → 目标掩码画布（橡皮始终生效，其余按成员集合过滤） */
function replaySubset(targetCtx, w, h, all, members) {
  targetCtx.setTransform(1, 0, 0, 1, 0, 0);
  targetCtx.clearRect(0, 0, w, h);
  for (const st of all) {
    const cnv = renderStroke(st, w, h);
    if (!cnv) continue;
    if (st.kind === 'eraser') {
      targetCtx.globalCompositeOperation = 'destination-out';
      targetCtx.drawImage(cnv, 0, 0);
      targetCtx.globalCompositeOperation = 'source-over';
    } else if (!members || members.has(st)) {
      targetCtx.drawImage(cnv, 0, 0);
    }
  }
}

/* 把白色形状 stamp 染成笔迹自身颜色（source-in 保 alpha） */
function colorizedStamp(stroke, w, h) {
  const cnv = renderStroke(stroke, w, h);
  if (!cnv) return null;
  const c2 = cnv.getContext('2d');
  c2.globalCompositeOperation = 'source-in';
  c2.fillStyle = stroke.color || '#cccccc';
  c2.fillRect(0, 0, w, h);
  c2.globalCompositeOperation = 'source-over';
  return cnv;
}

/* 重放每笔各自颜色的预览层（与掩码同形状，颜色取自笔迹记录） */
function replayColorStrokes() {
  const p = state.paint;
  if (!p.colorCtx) return;
  const w = p.colorCanvas.width, h = p.colorCanvas.height;
  p.colorCtx.setTransform(1, 0, 0, 1, 0, 0);
  p.colorCtx.clearRect(0, 0, w, h);
  for (const st of p.strokes) {
    if (st.kind === 'eraser') {
      const cnv = renderStroke(st, w, h);
      if (!cnv) continue;
      p.colorCtx.save();
      p.colorCtx.globalCompositeOperation = 'destination-out';
      p.colorCtx.drawImage(cnv, 0, 0);
      p.colorCtx.restore();
    } else {
      const stamp = colorizedStamp(st, w, h);
      if (stamp) p.colorCtx.drawImage(stamp, 0, 0);
    }
  }
  p.colorRev++;
}

/* 重放全部笔迹 → 掩码画布 + 彩色预览层 */
export function replayStrokes() {
  const p = state.paint;
  if (!p.maskCtx) return;
  replaySubset(p.maskCtx, p.maskCanvas.width, p.maskCanvas.height, p.strokes, null);
  p.maskRev++;
  replayColorStrokes();
  schedulePaintMorph();   // 笔迹变化 → 防抖重算真实形貌预览
  emit('render');
}

/* 进行中笔迹的增量预览：直接临时画（不做软化，性能优先），
 * pointerup 后统一重放得到精确结果。同步在彩色预览层着色。 */
export function paintPreviewSegment(pts) {
  const p = state.paint;
  if (!p.maskCtx || p.cur?.kind === 'eraser') return;
  const c = p.maskCtx;
  c.fillStyle = '#fff';
  const r = brushSizeProxy() / 2;
  const cc = p.colorCtx;
  if (cc) cc.fillStyle = p.cur.color || '#cccccc';
  if (p.cur.kind === 'scratch') {
    for (const pt of pts) {
      c.beginPath();
      c.arc(pt.x, pt.y, r * 0.55, 0, Math.PI * 2);
      c.fill();
      if (cc) { cc.beginPath(); cc.arc(pt.x, pt.y, r * 0.55, 0, Math.PI * 2); cc.fill(); }
    }
  } else if (p.cur.kind === 'spot_bright' || p.cur.kind === 'spot_dark' || p.cur.kind === 'dent') {
    // 黑白点/凹坑拖拽：即时铺斑反馈（抬笔后由 renderStroke 精确重放）
    const rr = r * 0.8;
    for (const pt of pts) {
      c.beginPath(); c.arc(pt.x, pt.y, rr, 0, Math.PI * 2); c.fill();
      if (cc) { cc.beginPath(); cc.arc(pt.x, pt.y, rr, 0, Math.PI * 2); cc.fill(); }
    }
  } else if (p.cur.kind === 'stain') {
    for (const pt of pts) {
      c.beginPath();
      c.arc(pt.x + (Math.random() * 2 - 1) * r, pt.y + (Math.random() * 2 - 1) * r, r * 0.6, 0, Math.PI * 2);
      c.fill();
      if (cc) {
        cc.beginPath();
        cc.arc(pt.x + (Math.random() * 2 - 1) * r, pt.y + (Math.random() * 2 - 1) * r, r * 0.6, 0, Math.PI * 2);
        cc.fill();
      }
    }
  }
  p.maskRev++;
  if (cc) p.colorRev++;
  emit('render');
}

/* ---------------- 已画缺陷的选择与单独调节 ----------------
 * 设计：画笔参数只影响"下一笔"；已画笔迹在落笔瞬间冻结全部形貌参数。
 * 需要修改历史缺陷时，用「调整」工具点选它，滑杆即只作用于该缺陷。 */

export function strokeBox(st) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const pt of st.points) {
    if (pt.x < x0) x0 = pt.x; if (pt.x > x1) x1 = pt.x;
    if (pt.y < y0) y0 = pt.y; if (pt.y > y1) y1 = pt.y;
  }
  const r = Math.max(6, st.size * 0.6);
  return { x: x0 - r, y: y0 - r, w: (x1 - x0) + 2 * r, h: (y1 - y0) + 2 * r };
}

function distToStroke(img, st) {
  const pts = st.points;
  let best = 1e9;
  for (let j = 0; j < pts.length; j++) {
    const a = pts[j], b = pts[Math.min(j + 1, pts.length - 1)];
    const dx = b.x - a.x, dy = b.y - a.y;
    const L2 = dx * dx + dy * dy;
    let t = L2 ? ((img.x - a.x) * dx + (img.y - a.y) * dy) / L2 : 0;
    t = Math.max(0, Math.min(1, t));
    const px = a.x + dx * t, py = a.y + dy * t;
    best = Math.min(best, Math.hypot(img.x - px, img.y - py));
  }
  return best;
}

export function selectStrokeAt(img) {
  const p = state.paint;
  let hit = null;
  for (let i = p.strokes.length - 1; i >= 0; i--) {
    const st = p.strokes[i];
    if (st.kind === 'eraser') continue;
    if (distToStroke(img, st) <= Math.max(5, st.size * 0.58)) { hit = st; break; }
  }
  const had = p.selected;
  p.selected = hit;
  syncPaintPanel();
  if (hit && hit !== had) {
    emit('toast', { text: '已选中缺陷：下方滑杆只调节它（点空白处完成）', kind: 'info' });
  }
  emit('render');
  return !!hit;
}

export function clearStrokeSelection(silent) {
  if (!state.paint.selected) return;
  state.paint.selected = null;
  syncPaintPanel();
  if (!silent) emit('toast', { text: '已结束调节', kind: 'info' });
  emit('render');
}

/* 参数应用：无选中 → 画笔默认（只影响下一笔）；有选中 → 该缺陷并立即重算 */
function applyParam(key, value) {
  const p = state.paint;
  const sel = p.selected;
  if (!sel) {
    if (key.startsWith('warp.')) p.warp[key.slice(5)] = value;
    else p[key] = value;
    return;
  }
  if (key === 'size') {
    const ps = state.target?.proxyScale > 0 ? state.target.proxyScale : 1;
    sel.size = Math.max(2, value * ps);
  } else if (key.startsWith('warp.')) {
    if (sel.kind !== 'dent' && sel.kind !== 'scratch') return;
    sel.warp = { ...(sel.warp || p.warp), [key.slice(5)]: value };
  } else {
    sel[key] = value;
    if (key === 'strength' || key === 'grain' || key === 'spacing') {
      // 同组键包含这些值，改后需重新分组
    }
  }
  replayStrokes();
}

/* ---------------- 笔迹事件（由 interact.js 调用） ---------------- */

export function paintDown(img) {
  const p = state.paint;
  const isWarpKind = p.brush === 'dent' || p.brush === 'scratch';
  // 落笔期间清除形貌预览（回退色块叠加），抬笔后防抖重算真实形貌
  if (p.previewLayer) { p.previewLayer = null; p.previewRev++; }
  clearTimeout(morphTimer);
  p.cur = {
    kind: p.brush, size: brushSizeProxy(), irregular: p.irregular,
    hardness: p.hardness, spacing: p.spacing,
    // 落笔即冻结：此后调画笔参数不得回溯影响这一笔（可选中后单独调节）
    strength: p.strength, grain: p.grain, morphSeed: p.seed,
    seed: (Math.random() * 0xffffffff) >>> 0,
    color: p.color, custom: p.customColor && p.brush !== 'eraser',
    warp: isWarpKind ? { ...p.warp } : null,
    points: [{ ...img }],
  };
  if (p.cur.kind === 'eraser') {
    // 橡皮即时生效（增量 destination-out，掩码与彩色预览层同步）
    p.maskCtx.save();
    p.maskCtx.globalCompositeOperation = 'destination-out';
    p.maskCtx.beginPath();
    const rE = brushSizeProxy() / 2;
    p.maskCtx.arc(img.x, img.y, rE, 0, Math.PI * 2);
    p.maskCtx.fill();
    p.maskCtx.restore();
    if (p.colorCtx) {
      p.colorCtx.save();
      p.colorCtx.globalCompositeOperation = 'destination-out';
      p.colorCtx.beginPath();
      p.colorCtx.arc(img.x, img.y, rE, 0, Math.PI * 2);
      p.colorCtx.fill();
      p.colorCtx.restore();
      p.colorRev++;
    }
    p.maskRev++;
    schedulePaintMorph(500);
    emit('render');
  }
  return p.cur;
}

export function paintMove(img) {
  const p = state.paint;
  if (!p.cur) return;
  const last = p.cur.points[p.cur.points.length - 1];
  const d = Math.hypot(img.x - last.x, img.y - last.y);
  p.cur.points.push({ ...img });
  if (p.cur.kind === 'eraser') {
    p.maskCtx.save();
    p.maskCtx.globalCompositeOperation = 'destination-out';
    p.maskCtx.beginPath();
    const rE = brushSizeProxy() / 2;
    p.maskCtx.arc(img.x, img.y, rE, 0, Math.PI * 2);
    p.maskCtx.fill();
    p.maskCtx.restore();
    if (p.colorCtx) {
      p.colorCtx.save();
      p.colorCtx.globalCompositeOperation = 'destination-out';
      p.colorCtx.beginPath();
      p.colorCtx.arc(img.x, img.y, rE, 0, Math.PI * 2);
      p.colorCtx.fill();
      p.colorCtx.restore();
      p.colorRev++;
    }
    p.maskRev++;
    schedulePaintMorph(500);
    emit('render');
    return;
  }
  if (d > 2) {
    // 所有笔刷都做增量预览（黑白点/凹坑拖拽需要看到铺点过程）
    paintPreviewSegment([{ ...img }]);
  }
}

export function paintUp() {
  const p = state.paint;
  if (!p.cur) return;
  const cur = p.cur;
  p.cur = null;
  // 橡皮笔迹同样入栈：重放/撤销/形貌重算全靠笔迹记录（不入栈则被擦区域
  // 在下一次 replayStrokes 时原样弹回——凹坑形貌正是从笔迹重放生成的）
  const valid = cur.kind === 'scratch' || cur.kind === 'stain'
    ? cur.points.length >= 2
    : cur.points.length >= 1;
  if (!valid) return;
  p.strokes.push(cur);
  replayStrokes();   // 精确重放（含软化与种子化形状）
}

export function paintUndo() {
  const p = state.paint;
  if (p.strokes.length) {
    const removed = p.strokes.pop();
    if (p.selected === removed) { p.selected = null; syncPaintPanel(); }
    replayStrokes();
  }
}

export function paintClear() {
  const p = state.paint;
  p.strokes = [];
  p.selected = null;
  syncPaintPanel();
  replayStrokes();
}

/* ---------------- 吸管取色（画布单击采样目标图像素） ---------------- */

let pickCtx = null;
export function pickColorAt(img) {
  const t = state.target;
  const p = state.paint;
  if (!t || !p) return;
  if (!pickCtx) pickCtx = new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true });
  const x = Math.max(0, Math.min(t.w - 1, Math.floor(img.x)));
  const y = Math.max(0, Math.min(t.h - 1, Math.floor(img.y)));
  pickCtx.clearRect(0, 0, 1, 1);
  pickCtx.drawImage(t.bitmap, x, y, 1, 1, 0, 0, 1, 1);
  const d = pickCtx.getImageData(0, 0, 1, 1).data;
  p.color = rgbToHex(d[0], d[1], d[2]);
  p.customColor = true;
  emit('paint-color');
  emit('toast', { text: `已取色 ${p.color}（已启用自定义颜色）`, kind: 'info' });
  schedulePaintMorph();
  emit('render');
}

/* ---------------- 生成缺陷资产（按颜色分组着色后合成） ----------------
 * buildMorphPatch：烘焙与实时预览共用的着色管线——原始像素直传后端，
 * 返回形貌合成结果（内部抹平/窄带内陷/分组着色/扩展 α）。 */

/* 画布 → 原始像素 base64（BGR / 单通道 α）：绕开浏览器 PNG 编码器
 * （convertToBlob/toBlob 在部分环境偶发永久挂起），后端 np.frombuffer 重建 */
function toRawB64(cnv, gray) {
  const w2 = cnv.width, h2 = cnv.height;
  const d2 = cnv.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w2, h2).data;
  const n = w2 * h2 * (gray ? 1 : 3);
  const bytes = new Uint8Array(n);
  let j = 0;
  for (let i = 0; i < d2.length; i += 4) {
    if (gray) bytes[j++] = d2[i + 3];
    else { bytes[j++] = d2[i + 2]; bytes[j++] = d2[i + 1]; bytes[j++] = d2[i]; }
  }
  let s = '';
  for (let i = 0; i < n; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return { b64: btoa(s), w: w2, h: h2 };
}

/* 原图位图缓存：createImageBitmap 在本环境偶发永久挂起——竞速 6s 超时，
 * 超时/失败回退代理分辨率（烘焙/导出仍有效，仅分辨率降级并提示） */
async function ensureOrigBmp(t) {
  if (t.origBmp) return t.origBmp;
  if (!t.originalBlob) return t.bitmap;
  try {
    const bmp = await Promise.race([
      createImageBitmap(t.originalBlob, { imageOrientation: 'from-image' }),
      new Promise((_, rej) => setTimeout(() => rej(new Error('原图解码超时')), 6000)),
    ]);
    t.origBmp = bmp;
    return bmp;
  } catch { return t.bitmap; }
}

async function buildMorphPatch(useOrig, signal) {
  const p = state.paint;
  const t = state.target;
  if (!t || !p.maskCanvas || !p.strokes.length) return null;
  const w = p.maskCanvas.width, h = p.maskCanvas.height;
  const d = p.maskCtx.getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = 0, y1 = 0, count = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (d[(y * w + x) * 4 + 3] > 24) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
        count++;
      }
    }
  }
  if (count < 30) return null;
  // 凹坑/划伤带材质内陷扭曲光晕（作用半径可调），裁剪 pad 相应扩大
  const warpPad = Math.max(...p.strokes.map((st) =>
    st.kind === 'dent' || st.kind === 'scratch'
      ? st.size * (st.warp ? st.warp.radius * 0.75 + 0.5 : 2.6)
      : 0), 0);
  const pad = Math.max(24, p.size, Math.ceil(warpPad));
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad);
  x1 = Math.min(w, x1 + 1 + pad); y1 = Math.min(h, y1 + 1 + pad);
  const cw = x1 - x0, ch = y1 - y0;

  // 全分辨率烘焙：代理图编辑（≤2048），烘焙从原图裁剪——大图（6K+）缺陷与
  // 扭曲形貌不因代理放大而糊化；扭曲作用范围随笔迹在原图上的真实尺寸计算。
  // 预览（useOrig=false）用代理分辨率，保证交互流畅。
  const scale = (useOrig && t.proxyScale && t.proxyScale > 0 && t.proxyScale < 0.999 && t.originalBlob)
    ? t.proxyScale : 1;
  let baseSrc = t.bitmap, mx0 = x0, my0 = y0, mw = cw, mh = ch;
  if (scale < 1) {
    try {
      const origBmp = await ensureOrigBmp(t);
      mx0 = Math.max(0, Math.round(x0 / scale));
      my0 = Math.max(0, Math.round(y0 / scale));
      mw = Math.min(origBmp.width - mx0, Math.round(cw / scale));
      mh = Math.min(origBmp.height - my0, Math.round(ch / scale));
      if (mw > 2 && mh > 2) baseSrc = origBmp;
      else { mx0 = x0; my0 = y0; mw = cw; mh = ch; baseSrc = t.bitmap; }
    } catch { /* 原图解码失败 → 代理路径 */ }
  }

  // 底图 patch
  const baseCnv = new OffscreenCanvas(mw, mh);
  baseCnv.getContext('2d').drawImage(baseSrc, mx0, my0, mw, mh, 0, 0, mw, mh);
  const baseRaw = toRawB64(baseCnv, false);

  // 合成画布尺寸 = patch 实际尺寸（全分辨率烘焙时为原图分辨率，勿用代理 cw/ch）
  const compCnv = new OffscreenCanvas(mw, mh);
  const compCtx = compCnv.getContext('2d');
  compCtx.drawImage(baseCnv, 0, 0);
  const alphaUnion = new OffscreenCanvas(mw, mh);   // 各笔迹返回层的 α 并集（含扭曲晕圈）
  const actx = alphaUnion.getContext('2d', { willReadFrequently: true });

  // 逐笔独立成像：每笔用自己的掩码/参数/种子单独请求后端。
  // 合并成"组"会让调节某一笔（改强度等）改变同组其它缺陷的种子与索引，
  // 出现"调 A 结果 B 也变了"——隔离性要求每笔独立。
  for (const st of p.strokes) {
    if (st.kind === 'eraser' || !st.points.length) continue;
    // 该笔迹的局部区域（掩码坐标）：点位 + 半径 + 软化余量 + 扭曲晕圈
    const soft = Math.max(1, (1 - st.hardness) * st.size / 5) * 2 + 2;
    let r = st.size / 2 + soft;
    if (st.kind === 'dent' || st.kind === 'scratch') {
      r += st.size * ((st.warp?.radius ?? 1) * 0.75 + 0.5);
    }
    let lx0 = 1e9, ly0 = 1e9, lx1 = -1e9, ly1 = -1e9;
    for (const pt of st.points) {
      lx0 = Math.min(lx0, pt.x); lx1 = Math.max(lx1, pt.x);
      ly0 = Math.min(ly0, pt.y); ly1 = Math.max(ly1, pt.y);
    }
    if (!isFinite(lx0)) continue;
    lx0 = Math.max(0, Math.floor(lx0 - r)); ly0 = Math.max(0, Math.floor(ly0 - r));
    lx1 = Math.min(w, Math.ceil(lx1 + r)); ly1 = Math.min(h, Math.ceil(ly1 + r));
    const lw = lx1 - lx0, lh = ly1 - ly0;
    if (lw < 2 || lh < 2) continue;
    // 掩码：该笔迹 + 其后所有橡皮（replaySubset 中橡皮始终生效）
    const full = new OffscreenCanvas(w, h);
    replaySubset(full.getContext('2d'), w, h, p.strokes, new Set([st]));
    // 转到底图 patch 坐标
    const px0 = Math.max(0, Math.round(lx0 / scale) - mx0);
    const py0 = Math.max(0, Math.round(ly0 / scale) - my0);
    const pw = Math.min(mw - px0, Math.max(1, Math.round(lw / scale)));
    const ph = Math.min(mh - py0, Math.max(1, Math.round(lh / scale)));
    if (pw < 2 || ph < 2) continue;
    const gmask = new OffscreenCanvas(pw, ph);
    const gctx = gmask.getContext('2d', { willReadFrequently: true });
    gctx.imageSmoothingEnabled = true;   // 掩码随 patch 一起升到原图分辨率（α 平滑）
    gctx.drawImage(full, lx0, ly0, lw, lh, 0, 0, pw, ph);
    const gd2 = gctx.getImageData(0, 0, pw, ph).data;
    let alphaSum = 0;
    for (let i = 3; i < gd2.length; i += 4) alphaSum += gd2[i];
    if (alphaSum < 16) continue;         // 该笔迹已被完全擦除
    // 局部底图 patch
    const bCnv = new OffscreenCanvas(pw, ph);
    bCnv.getContext('2d').drawImage(baseSrc, mx0 + px0, my0 + py0, pw, ph, 0, 0, pw, ph);
    const bRaw = toRawB64(bCnv, false);
    const rgb = st.custom ? Object.values(hexToRgb(st.color)) : null;   // RGB
    const gmaskRaw = toRawB64(gmask, true);   // 单通道 α
    // 深度单位 = 当前 patch 坐标系 px（预览为代理 px，烘焙为原图 px）
    const warpEff = st.warp
      ? { ...st.warp, depth: st.warp.depth * (useOrig ? 1 : scale) }
      : null;
    const res = await api.paintDefect({
      baseRawB64: bRaw.b64, rawW: bRaw.w, rawH: bRaw.h,
      maskRawB64: gmaskRaw.b64,
      kind: st.kind,
      strength: st.strength ?? p.strength,
      grain: st.grain ?? p.grain,
      seed: (st.morphSeed ?? p.seed) | 0,   // 每笔稳定种子：不受他笔影响
      tintRgb: rgb,
      warp: warpEff,
      signal,
    });
    // 原始 RGBA 直读（ImageData 重建）：data:URL fetch + createImageBitmap
    // 在部分环境偶发永久挂起——挂起会让形貌预览层永远不渲染
    const nBytes = res.w * res.h * 4;
    const bytes = new Uint8ClampedArray(nBytes);
    const b64 = res.layer_raw_b64;
    for (let i = 0, j = 0; i < b64.length; i += 0x8000) {
      const chunk = atob(b64.slice(i, i + 0x8000));
      for (let k = 0; k < chunk.length; k++) bytes[j++] = chunk.charCodeAt(k);
    }
    const layerCnv = new OffscreenCanvas(res.w, res.h);
    layerCnv.getContext('2d').putImageData(new ImageData(bytes, res.w, res.h), 0, 0);
    compCtx.drawImage(layerCnv, px0, py0);
    // 提取该层 α（白剪影）并入并集——凹坑/划伤的扩展晕圈 α 由后端返回
    const tmp = new OffscreenCanvas(pw, ph);
    const tctx = tmp.getContext('2d');
    tctx.drawImage(layerCnv, 0, 0);
    tctx.globalCompositeOperation = 'source-in';
    tctx.fillStyle = '#fff';
    tctx.fillRect(0, 0, pw, ph);
    actx.drawImage(tmp, px0, py0);
  }

  // 最终形貌 = 合成结果 RGB × α 并集
  const rgbaCanvas = new OffscreenCanvas(mw, mh);
  const rctx = rgbaCanvas.getContext('2d');
  rctx.drawImage(compCnv, 0, 0);
  rctx.globalCompositeOperation = 'destination-in';
  rctx.drawImage(alphaUnion, 0, 0);
  rctx.globalCompositeOperation = 'source-over';

  const maskCrop = new OffscreenCanvas(mw, mh);
  const mctx = maskCrop.getContext('2d');
  mctx.imageSmoothingEnabled = true;
  mctx.drawImage(p.maskCanvas, x0, y0, cw, ch, 0, 0, mw, mh);
  return { rgbaCanvas, maskCrop, x: mx0, y: my0, w: mw, h: mh, scale };
}

export async function bakePaintedAsset(name, tags) {
  const p = state.paint;
  const t = state.target;
  if (!t || !p.maskCanvas) return null;
  const patch = await buildMorphPatch(true);
  if (!patch) {
    emit('toast', { text: '笔迹为空：请先用笔刷绘制缺陷', kind: 'warn' });
    return null;
  }
  const asset = await assets.addAsset({
    name: name || `绘制_${p.brush}_${new Date().toISOString().slice(11, 19).replace(/:/g, '')}`,
    tags: tags,
    rgbaCanvas: patch.rgbaCanvas,
    maskCanvas: patch.maskCrop,
    // scale = 代理/原图比例：放置时按此缩放（画布上与绘制时等大），全分辨率导出时 ×k 归一为 1:1
    source: { image: t.name, bbox: [patch.x, patch.y, patch.w, patch.h], painted: p.brush, scale: patch.scale },
  });
  return asset;
}

/* ---------------- 直接导出图像（不经过资产库） ----------------
 * 全分辨率烘焙当前笔迹 → 合成回原图 → 后端编码 PNG/JPG → 触发下载。
 * 画布→原始像素直传，绕开 convertToBlob（本环境偶发挂起）。 */
export async function exportPaintedImage(fmt = '.png') {
  const p = state.paint;
  const t = state.target;
  if (!t || !p.maskCanvas) { emit('toast', { text: '请先打开目标图像', kind: 'warn' }); return; }
  const patch = await buildMorphPatch(true);
  if (!patch) { emit('toast', { text: '笔迹为空：请先用笔刷绘制缺陷', kind: 'warn' }); return; }
  // 先弹系统保存对话框（用户自选目录/文件名）——必须在点击的用户手势内调用
  const outName = (t.name || 'image').replace(/\.[^.]+$/, '') + '_缺陷' + (fmt === '.jpg' ? '.jpg' : '.png');
  const handle = await pickSaveHandle(outName, fmt === '.jpg' ? 'image/jpeg' : 'image/png');
  if (handle === 'cancel') { emit('toast', { text: '已取消导出', kind: 'info' }); return; }
  emit('toast', { text: '正在合成导出图像…', kind: 'info' });
  const src = await ensureOrigBmp(t);
  const out = new OffscreenCanvas(src.width, src.height);
  const octx = out.getContext('2d');
  octx.drawImage(src, 0, 0);
  octx.drawImage(patch.rgbaCanvas, patch.x, patch.y);
  const raw = toRawB64(out, false);
  const res = await api.rawEncode(raw.b64, raw.w, raw.h, fmt);
  const bin = atob(res.img_b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const how = await saveBlob(new Blob([bytes], { type: res.mime }), outName, handle);
  emit('toast', { text: how === 'saved' ? `已保存 ${outName}` : `已下载 ${outName}`, kind: 'ok' });
}

/* ---------------- 实时形貌预览（与烘焙同一管线，所见即所得） ----------------
 * 笔迹结束/参数变化后防抖执行：后端按当前掩码与参数计算真实形貌
 * （凹坑内部抹平+窄带内陷+分组着色），由渲染 Worker 直接绘制到画布，
 * 替代此前的半透明色块叠加——直接在模板图上绘制时即为真实效果。 */

let morphRun = 0;              // 在飞请求序号：后到的旧结果不得覆盖新结果
let morphRetry = 0;            // 连续失败重试计数（防死循环）
export async function renderPaintMorph() {
  const p = state.paint;
  const t = state.target;
  const run = ++morphRun;
  if (!t || !p.maskCanvas || !p.strokes.length) {
    p.previewLayer = null;
    p.previewRev++;
    emit('render');
    return;
  }
  // 超时中止 + 自动重试：本环境 fetch/解码偶发永久挂起——挂起会让预览层
  // 永不更新（表现为"点击不渲染"，直到下一次交互重排定时器才恢复）
  const ctrl = new AbortController();
  const kill = setTimeout(() => ctrl.abort(), 8000);
  try {
    const patch = await buildMorphPatch(false, ctrl.signal);
    if (run !== morphRun) return;   // 期间又有新笔迹/新请求 → 丢弃过期结果，
                                    // 否则旧层（缺新笔迹）会永久盖住新层
    morphRetry = 0;
    if (patch) {
      p.previewLayer = { canvas: patch.rgbaCanvas, x: patch.x, y: patch.y };
      p.previewRev++;
    }
  } catch (e) {
    console.warn('[paint] 形貌预览失败（保留色块回退）:', e);
    if (run === morphRun && morphRetry < 2) {
      morphRetry++;
      schedulePaintMorph(250);      // 自愈：自动重试，无需用户再交互
    }
  } finally {
    clearTimeout(kill);
  }
  emit('render');
}

let morphTimer = null;
export function schedulePaintMorph(delay = 300) {
  clearTimeout(morphTimer);
  morphTimer = setTimeout(() => { renderPaintMorph(); }, delay);
}

/* ---------------- 模式切换与面板 ---------------- */

export function enterPaint() {
  if (!state.target) {
    emit('toast', { text: '请先打开目标图像', kind: 'warn' });
    return;
  }
  initPaintState();
  const t = state.target;
  state.mode = 'paint';
  state.paint.selected = null;
  state.paint.adjustMode = false;
  const adj0 = document.querySelector('[data-padjust]');
  if (adj0) adj0.classList.remove('active');
  state.paint.maskCanvas = new OffscreenCanvas(t.w, t.h);
  state.paint.maskCtx = state.paint.maskCanvas.getContext('2d', { willReadFrequently: true });
  state.paint.colorCanvas = new OffscreenCanvas(t.w, t.h);
  state.paint.colorCtx = state.paint.colorCanvas.getContext('2d', { willReadFrequently: true });
  state.tool = 'paint';
  fitViewSafe();
  syncPaintPanel();
  emit('mode');
  emit('render');
}

export function exitPaint() {
  state.mode = 'edit';
  state.tool = 'select';
  fitViewSafe();
  emit('mode');
  emit('render');
}

function fitViewSafe() {
  import('./state.js').then((m) => m.fitView());
}

/* 面板滑杆/色块与 state 同步（进入绘制模式时 initPaintState 会重置参数） */
export function syncPaintPanel() {
  const p = state.paint;
  if (!p) return;
  const $ = (id) => document.getElementById(id);
  const sel = p.selected;
  const ps = state.target?.proxyScale > 0 ? state.target.proxyScale : 1;
  const w = (sel && sel.warp) || p.warp;
  const src = sel ? {
    size: sel.size / ps, irregular: sel.irregular, hardness: sel.hardness,
    strength: sel.strength ?? p.strength, grain: sel.grain ?? p.grain,
    spacing: sel.spacing ?? 1,
  } : p;
  // 选中提示条
  const info = $('pbSelInfo');
  if (info) info.hidden = !sel;
  const pair = [
    ['pbSize', 'pbSizeN', Math.round(src.size), (v) => v + 'px'],
    ['pbIrregular', 'pbIrregularN', src.irregular, (v) => (+v).toFixed(2)],
    ['pbHardness', 'pbHardnessN', src.hardness, (v) => (+v).toFixed(2)],
    ['pbStrength', 'pbStrengthN', src.strength, (v) => (+v).toFixed(2)],
    ['pbGrain', 'pbGrainN', src.grain, (v) => (+v).toFixed(2)],
    ['pbSpacing', 'pbSpacingN', src.spacing, (v) => (+v).toFixed(2)],
    ['pbWarpStr', 'pbWarpStrN', w.depth, (v) => v + 'px'],
    ['pbWarpFall', 'pbWarpFallN', w.falloff, (v) => (+v).toFixed(1)],
    ['pbWarpRad', 'pbWarpRadN', w.radius, (v) => (+v).toFixed(2) + '×'],
    ['pbCore', 'pbCoreN', w.core ?? 0.35, (v) => (+v).toFixed(2)],
    ['pbShade', 'pbShadeN', w.shade ?? 0.2, (v) => (+v).toFixed(2)],
    ['pbSwirl', 'pbSwirlN', w.swirl ?? 0.5, (v) => (+v).toFixed(2)],
  ];
  for (const [id, outId, val, fmt] of pair) {
    const el = $(id), out = $(outId);
    if (el) el.value = val;
    if (out) out.textContent = fmt(val);
  }
  if ($('pbCustomColor')) $('pbCustomColor').checked = p.customColor;
  updateColorSwatch();
  // 「调整」工具：点选已画缺陷，滑杆只调节它
  const adjBtn = document.querySelector('[data-padjust]');
  if (adjBtn) {
    adjBtn.onclick = () => {
      const on = !state.paint.adjustMode;
      state.paint.adjustMode = on;
      adjBtn.classList.toggle('active', on);
      if (on) {
        state.paint.brush = '';
        state.tool = 'paint';
        document.querySelectorAll('[data-pbrush]').forEach((x) => x.classList.remove('active'));
        emit('toast', { text: '调整模式：点击画布上的缺陷即可选中并调节', kind: 'info' });
      } else {
        clearStrokeSelection(true);
      }
      emit('render');
    };
  }
  const selClear = $('pbSelClear');
  if (selClear) selClear.onclick = () => clearStrokeSelection();
  const selDel = $('pbSelDel');
  if (selDel) selDel.onclick = () => {
    const st = state.paint.selected;
    if (!st) return;
    state.paint.strokes = state.paint.strokes.filter((x) => x !== st);
    state.paint.selected = null;
    syncPaintPanel();
    replayStrokes();
    emit('toast', { text: '已删除该缺陷', kind: 'info' });
  };

  for (const b of document.querySelectorAll('[data-pbrush]')) {
    b.classList.toggle('active', b.dataset.pbrush === p.brush);
  }
}

function updateColorSwatch() {
  const sw = document.getElementById('pbColorSwatch');
  if (sw) {
    sw.style.background = state.paint?.color || '#ccc';
    sw.title = '当前颜色 ' + (state.paint?.color || '');
  }
}

export function initPaintPanel() {
  const $ = (id) => document.getElementById(id);
  const bind = (id, outId, fmt, fn) => {
    const el = $(id), out = outId && $(outId);
    if (!el) return;
    el.addEventListener('input', () => {
      if (out) out.textContent = fmt(el.value);
      fn(el.value);
      emit('render');
    });
  };
  bind('pbSize', 'pbSizeN', (v) => v + 'px', (v) => applyParam('size', +v));
  bind('pbIrregular', 'pbIrregularN', (v) => (+v).toFixed(2), (v) => applyParam('irregular', +v));
  bind('pbHardness', 'pbHardnessN', (v) => (+v).toFixed(2), (v) => applyParam('hardness', +v));
  // 画笔形貌参数：未选中缺陷时只影响"下一笔"；选中缺陷时调节该缺陷本身。
  // 已画笔迹在落笔时冻结全部参数，画笔画笔参数不再回溯修改历史缺陷。
  bind('pbSpacing', 'pbSpacingN', (v) => (+v).toFixed(2), (v) => applyParam('spacing', +v));
  bind('pbWarpStr', 'pbWarpStrN', (v) => v + 'px', (v) => applyParam('warp.depth', +v));
  bind('pbWarpFall', 'pbWarpFallN', (v) => (+v).toFixed(1), (v) => applyParam('warp.falloff', +v));
  bind('pbWarpRad', 'pbWarpRadN', (v) => (+v).toFixed(2) + '×', (v) => applyParam('warp.radius', +v));
  bind('pbCore', 'pbCoreN', (v) => (+v).toFixed(2), (v) => applyParam('warp.core', +v));
  bind('pbShade', 'pbShadeN', (v) => (+v).toFixed(2), (v) => applyParam('warp.shade', +v));
  bind('pbSwirl', 'pbSwirlN', (v) => (+v).toFixed(2), (v) => applyParam('warp.swirl', +v));
  bind('pbStrength', 'pbStrengthN', (v) => (+v).toFixed(2), (v) => applyParam('strength', +v));
  bind('pbGrain', 'pbGrainN', (v) => (+v).toFixed(2), (v) => applyParam('grain', +v));

  for (const b of document.querySelectorAll('[data-pbrush]')) {
    b.onclick = () => {
      const kind = b.dataset.pbrush;
      if (state.paint.brush === kind) {
        // 再次点击已选中笔刷 → 取消画笔（浏览模式：左键拖拽平移，不会误画）
        state.paint.brush = '';
        state.tool = 'paint';
        document.querySelectorAll('[data-pbrush]').forEach((x) => x.classList.remove('active'));
        emit('toast', { text: '已取消画笔：左键拖拽平移查看，点击任意笔刷恢复绘制', kind: 'info' });
      } else {
        document.querySelectorAll('[data-pbrush]').forEach((x) => x.classList.toggle('active', x === b));
        state.paint.adjustMode = false;
        const ab = document.querySelector('[data-padjust]');
        if (ab) ab.classList.remove('active');
        clearStrokeSelection(true);
        state.paint.brush = kind;
        state.tool = 'paint';
        // 切换笔刷：颜色仍为任一类型默认色（用户未自选）→ 跟随新类型建议色；自选颜色保留
        if (Object.values(KIND_DEFAULT_COLORS).includes(state.paint.color)) {
          state.paint.color = kindDefaultColor(state.paint.brush);
        }
      }
      updateColorSwatch();
      emit('render');
    };
  }

  // 自定义颜色 / 三色盘 / 吸管
  $('pbCustomColor').addEventListener('change', (e) => {
    state.paint.customColor = e.target.checked;
    if (e.target.checked && Object.values(KIND_DEFAULT_COLORS).includes(state.paint.color)) {
      state.paint.color = kindDefaultColor(state.paint.brush);
      updateColorSwatch();
    }
    schedulePaintMorph();
    emit('render');
  });
  $('pbColorSwatch').addEventListener('click', () => {
    import('./colorpick.js').then((m) => m.openPicker($('pbColorSwatch'), state.paint.color, (hex) => {
      state.paint.color = hex;
      state.paint.customColor = true;
      if ($('pbCustomColor')) $('pbCustomColor').checked = true;
      updateColorSwatch();
      schedulePaintMorph();
      emit('render');
    }));
  });
  $('pbPick').addEventListener('click', () => {
    state.tool = 'pick';
    const cv = document.getElementById('cv');
    if (cv) cv.style.cursor = 'crosshair';
    emit('toast', { text: '吸管：在图像上单击取色（绘制中可随时 Alt+单击）', kind: 'info' });
  });
  on('paint-color', updateColorSwatch);

  $('pbUndo').onclick = paintUndo;
  $('pbClear').onclick = paintClear;
  $('pbReroll').onclick = () => { state.paint.seed = (Math.random() * 0xffffffff) >>> 0; schedulePaintMorph(); emit('toast', { text: '已重掷随机种子', kind: 'info' }); };
  $('btnExitPaint').onclick = exitPaint;
  $('btnBakeAsset').onclick = async () => {
    const btn = $('btnBakeAsset');
    btn.disabled = true;
    try {
      const a = await bakePaintedAsset($('pbName')?.value?.trim(), (($('pbTags')?.value) || '').split(/[,，\s]+/).filter(Boolean));
      if (a) {
        emit('toast', { text: `缺陷资产已生成：${a.name}（可从资产库拖拽放置）`, kind: 'ok' });
        exitPaint();
      }
    } catch (e) {
      emit('toast', { text: '生成失败：' + e.message, kind: 'err' });
    } finally {
      btn.disabled = false;
    }
  };
  // 直接导出图像：全分辨率烘焙当前笔迹并下载（不经资产库）。
  // 空值守卫：浏览器缓存可能持有旧版 index.html（无这些新按钮），
  // 直接赋值会让整个 boot 中断（所有界面功能失效）
  const expBtn = $('btnExportPaint');
  if (expBtn) expBtn.onclick = () => exportPaintedImage('.png');
  const expBtnJ = $('btnExportPaintJpg');
  if (expBtnJ) expBtnJ.onclick = () => exportPaintedImage('.jpg');
}
