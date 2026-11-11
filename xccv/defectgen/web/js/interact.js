/* Defect Augment Studio — 交互层
 * 指针状态机（选择/移动/缩放/旋转/网格FFD/透视/框选/平移）、滚轮锚点缩放、
 * 吸附对齐（画布边缘/实例智能参考线/网格）、多选批量变换、Photoshop 兼容快捷键。
 */
import { state, emit, screenToImage, imageToScreen, topInstanceAt, selectOnly,
  setSelection, selectedInstances, fitView, zoomAt, getInstance, removeInstanceRaw,
  addInstanceRaw, raiseSelection } from './state.js';
import * as history from './history.js';
import { buildOverlays } from './render.js';
import { paintDown, paintMove, paintUp, paintUndo, pickColorAt, exitPaint, selectStrokeAt } from './painter.js';
import {
  clamp, deg2rad, rad2deg, deepClone, instCorners, instAABB, unionAABB,
  solveHomography, hApply, meshVertexPos, clampMesh, uid, QUAD_ORDER,
} from './util.js';

let canvas = null;
let drag = null;   // 当前拖拽会话
let lastPolyClick = null;   // 多边形双击闭合检测

export function initInteract(cv) {
  canvas = cv;
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
  canvas.addEventListener('dblclick', onDblClick);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
}

/* ---------------- 指针 ---------------- */

function evPos(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function updateCursor(e) {
  const s = evPos(e);
  state.cursor.x = s.x; state.cursor.y = s.y;
  state.cursor.inside = s.x >= 0 && s.y >= 0 && s.x <= (state._canvasW || 0) && s.y <= (state._canvasH || 0);
  state.cursor.img = screenToImage(s);
  emit('status');
}

function onDown(e) {
  canvas.setPointerCapture?.(e.pointerId);
  updateCursor(e);
  const img = state.cursor.img;
  const screen = { x: state.cursor.x, y: state.cursor.y };

  // 分屏分割线拖拽
  if (state.ui.split.on && state.mode === 'edit' && Math.abs(screen.x - (state._canvasW || 0) * state.ui.split.pos) < 8) {
    drag = { kind: 'split' };
    return;
  }

  // 空格+拖拽 = 平移（所有模式统一）
  if (state.spacePan || e.button === 1 || state.tool === 'pan') {
    drag = { kind: 'pan', sx: screen.x, sy: screen.y, tx: state.view.tx, ty: state.view.ty };
    return;
  }

  if (state.mode === 'extract') { extractDown(e, img); return; }
  if (state.mode === 'paint') {
    if (e.button === 0 && (state.tool === 'pick' || e.altKey)) {
      pickColorAt(img);                       // 取色器 / Alt+单击 快速取色
      if (state.tool === 'pick') { state.tool = 'paint'; emit('tool'); }
      return;
    }
    if (e.button === 0 && state.paint.adjustMode) {
      selectStrokeAt(img);          // 调整模式：点选已画缺陷（不新增笔迹）
      return;
    }
    if (e.button === 0 && state.paint.brush) {
      paintDown(img); drag = { kind: 'paint' };
    } else if (e.button === 0) {
      drag = { kind: 'pan', sx: screen.x, sy: screen.y, tx: state.view.tx, ty: state.view.ty };   // 未选画笔 → 拖拽平移
    } else {
      drag = { kind: 'pan', sx: screen.x, sy: screen.y, tx: state.view.tx, ty: state.view.ty };
    }
    return;
  }

  if (state.spacePan || e.button === 1 || state.tool === 'pan') {
    drag = { kind: 'pan', sx: screen.x, sy: screen.y, tx: state.view.tx, ty: state.view.ty };
    return;
  }
  if (e.button !== 0) return;

  // 1. 控制柄命中
  const h = hitHandle(screen);
  if (h) { startHandleDrag(h, img, e); return; }

  // 2. 实例命中
  const inst = topInstanceAt(img);
  if (inst) {
    if (e.shiftKey) {
      // 多选切换
      if (state.selection.has(inst.id)) state.selection.delete(inst.id);
      else state.selection.add(inst.id);
      emit('selection'); emit('render');
      drag = { kind: 'selected-move-pending', img };
      return;
    }
    if (!state.selection.has(inst.id)) {
      selectOnly(inst.id);
      raiseSelection();
    }
    if (e.ctrlKey || e.metaKey) duplicateSelection(); // Ctrl+拖拽 = 原位复制后拖动
    beginMoveDrag(img, e);
    return;
  }

  // 3. 空白：开始框选（并清空选择）
  if (!e.shiftKey && state.selection.size) { selectOnly(null); }
  drag = { kind: 'marquee', x0: img.x, y0: img.y };
  state.marquee = { x0: img.x, y0: img.y, x1: img.x, y1: img.y };
  emit('render');
}

function beginMoveDrag(img) {
  const sel = selectedInstances();
  if (!sel.length) return;
  const handle = history.beginInteractive(sel.length > 1 ? '移动（多选）' : '移动');
  drag = {
    kind: 'move', handle, img,
    starts: sel.map((i) => ({ inst: i, pos: { ...i.position } })),
  };
}

function duplicateSelection() {
  const sel = selectedInstances();
  if (!sel.length) return;
  const handle = history.beginInteractive('原位复制');
  const clones = sel.map((inst) => {
    const c = deepClone(inst);
    c.id = uid();
    c.previewPatch = null;
    addInstanceRaw(c);
    handle.markAdded(c);
    return c;
  });
  handle.end();
  setSelection(clones.map((c) => c.id));
  raiseSelection();
}

/* ---------------- 控制柄 ---------------- */

function hitHandle(screen) {
  const ov = buildOverlays();
  let best = null, bestD = 9;
  for (const h of ov.handles) {
    const s = imageToScreen(h);
    const d = Math.hypot(s.x - screen.x, s.y - screen.y);
    if (d < bestD) { bestD = d; best = h; }
  }
  for (const p of ov.meshPts || []) {
    const s = imageToScreen(p);
    const d = Math.hypot(s.x - screen.x, s.y - screen.y);
    if (d < bestD) { bestD = d; best = { kind: 'mesh', idx: p.idx, x: p.x, y: p.y }; }
  }
  return best;
}

function startHandleDrag(h, img, e) {
  const ec = state.editCtx;
  if (h.kind === 'mesh') {
    const inst = getInstance(ec.instanceId);
    const asset = state.assets.get(inst.asset_id);
    if (!inst.mesh) inst.mesh = { cols: 4, rows: 4, grid: Array.from({ length: 25 }, () => [0, 0]) };
    const cols = inst.mesh.cols, rows = inst.mesh.rows;
    const s = meshVertexPos(inst.mesh, h.idx % (cols + 1), Math.floor(h.idx / (cols + 1)), asset.w, asset.h);
    drag = {
      kind: 'mesh', handle: history.beginInteractive('网格变形'),
      inst, asset, idx: h.idx, src: s,
    };
    return;
  }
  if (h.kind === 'persp') {
    const inst = getInstance(ec.instanceId);
    const asset = state.assets.get(inst.asset_id);
    drag = {
      kind: 'persp', handle: history.beginInteractive('透视调整'),
      inst, asset, idx: h.idx,
    };
    return;
  }
  const single = state.selection.size === 1 ? selectedInstances()[0] : null;
  if (h.kind === 'rotate') {
    if (!single) return;
    drag = {
      kind: 'rotate', handle: history.beginInteractive('旋转'),
      inst: single, asset: state.assets.get(single.asset_id),
      startAngle: Math.atan2(img.y - single.position.y, img.x - single.position.x),
      startRot: single.affine.rotation || 0,
    };
    return;
  }
  // 缩放柄（单选才显示）
  if (!single) return;
  const asset = state.assets.get(single.asset_id);
  const cs = instCorners(single, asset.w, asset.h);
  if (h.kind === 'scale') {
    const anchor = cs[(h.idx + 2) % 4];
    drag = {
      kind: 'scale', handle: history.beginInteractive('缩放'),
      inst: single, asset, anchor,
      startDist: Math.max(4, Math.hypot(img.x - anchor.x, img.y - anchor.y)),
      startPos: { ...single.position }, startSx: single.affine.scale_x, startSy: single.affine.scale_y,
    };
  } else {
    // 边中点：0顶(scale_y 绕底) 1右(scale_x 绕左) 2底(scale_y 绕顶) 3左(scale_x 绕右)
    const edges = [
      { axis: 'y', anchor: cs[3] }, { axis: 'x', anchor: cs[0] },
      { axis: 'y', anchor: cs[0] }, { axis: 'x', anchor: cs[1] },
    ][h.idx];
    drag = {
      kind: 'scaleEdge', handle: history.beginInteractive('缩放'),
      inst: single, asset, ...edges,
      startDist: Math.max(4, Math.hypot(img.x - edges.anchor.x, img.y - edges.anchor.y)),
      startPos: { ...single.position }, startSx: single.affine.scale_x, startSy: single.affine.scale_y,
    };
  }
}

function firstSelectedId() { return selectedInstances()[0]?.id; }

/* ---------------- 移动 ---------------- */

function onMove(e) {
  updateCursor(e);
  if (!drag) { emit('render'); return; }
  const img = state.cursor.img;
  const screen = { x: state.cursor.x, y: state.cursor.y };

  switch (drag.kind) {
    case 'paint':
      paintMove(img);
      break;
    case 'pan':
      state.view.tx = drag.tx + (screen.x - drag.sx);
      state.view.ty = drag.ty + (screen.y - drag.sy);
      emit('render');
      break;
    case 'split':
      state.ui.split.pos = clamp(screen.x / (state._canvasW || 1), 0.02, 0.98);
      emit('render');
      break;
    case 'marquee':
      state.marquee.x1 = img.x; state.marquee.y1 = img.y;
      emit('render');
      break;
    case 'move': {
      let dx = img.x - drag.img.x, dy = img.y - drag.img.y;
      const snapped = computeSnap(drag.starts, dx, dy);
      dx = snapped.dx; dy = snapped.dy;
      state.snapGuides = snapped.guides;
      for (const s of drag.starts) {
        drag.handle.change(s.inst, { position: { x: s.pos.x + dx, y: s.pos.y + dy } });
      }
      state._dragging = true;
      emit('render');
      break;
    }
    case 'scale': {
      const f = Math.hypot(img.x - drag.anchor.x, img.y - drag.anchor.y) / drag.startDist;
      const a = drag.anchor, s0 = drag.startPos;
      const inst = drag.inst;
      drag.handle.change(inst, {
        position: { x: a.x + (s0.x - a.x) * f, y: a.y + (s0.y - a.y) * f },
        affine: { ...inst.affine, scale_x: drag.startSx * f, scale_y: drag.startSy * f },
      });
      emit('render');
      break;
    }
    case 'scaleEdge': {
      const f = Math.hypot(img.x - drag.anchor.x, img.y - drag.anchor.y) / drag.startDist;
      const a = drag.anchor, s0 = drag.startPos;
      const inst = drag.inst;
      const sx = drag.axis === 'x' ? drag.startSx * f : drag.startSx;
      const sy = drag.axis === 'y' ? drag.startSy * f : drag.startSy;
      drag.handle.change(inst, {
        position: { x: a.x + (s0.x - a.x) * (drag.axis === 'x' ? f : 1), y: a.y + (s0.y - a.y) * (drag.axis === 'y' ? f : 1) },
        affine: { ...inst.affine, scale_x: sx, scale_y: sy },
      });
      emit('render');
      break;
    }
    case 'rotate': {
      const a = Math.atan2(img.y - drag.inst.position.y, img.x - drag.inst.position.x);
      let delta = rad2deg(a - drag.startAngle);
      if (e.shiftKey) delta = Math.round(delta / 15) * 15; // 角度吸附 15°
      drag.handle.change(drag.inst, { affine: { ...drag.inst.affine, rotation: drag.startRot + delta } });
      emit('render');
      break;
    }
    case 'mesh': {
      const inst = drag.inst, asset = drag.asset;
      // 图像坐标 → 仿射前局部坐标（逆链，不含 mesh）
      const local = affineInverse(inst, img, asset.w, asset.h);
      if (!local) break;
      const mesh = deepClone(inst.mesh);
      mesh.grid[drag.idx] = [local.x - drag.src.x, local.y - drag.src.y];
      clampMesh(mesh, asset.w, asset.h);
      drag.handle.change(inst, { mesh });
      emit('render');
      break;
    }
    case 'persp': {
      const inst = drag.inst, asset = drag.asset;
      const cs = instCorners(inst, asset.w, asset.h);
      cs[drag.idx] = { ...img };
      // 局部目标角点 = A⁻¹(图像角点)；源角点 = 资产角点 + 网格角点位移
      const dstLocal = cs.map((p) => affineInverseRaw(inst, p, asset.w, asset.h));
      const srcDisp = QUAD_ORDER.map(([nx, ny], i) => {
        const s = { x: nx * asset.w, y: ny * asset.h };
        if (inst.mesh) {
          const g = inst.mesh.grid[
            Math.round(ny * inst.mesh.rows) * (inst.mesh.cols + 1) + Math.round(nx * inst.mesh.cols)];
          s.x += g[0]; s.y += g[1];
        }
        return s;
      });
      const H = solveHomography(srcDisp, dstLocal);
      if (H) drag.handle.change(inst, { perspective_h: H });
      emit('render');
      break;
    }
    case 'polyVertex': {
      const ex = state.extract;
      if (ex && ex.polyPts[drag.idx]) {
        ex.polyPts[drag.idx] = { ...img };
        ex.maskRev++;      // 顶点变化后旧掩码失效提示
        emit('render');
      }
      break;
    }
    default:
      if (state.mode === 'extract') extractMove(e, img);
  }
}

/* 仿射逆变换（不含 H / mesh）：图像坐标 → 仿射前局部坐标 */
function affineInverseRaw(inst, pt, w, h) {
  const m = inst.affine || {};
  const rot = deg2rad(m.rotation || 0);
  const cos = Math.cos(-rot), sin = Math.sin(-rot);
  const sx = m.scale_x || 1, sy = m.scale_y || 1;
  const shx = m.shear_x || 0, shy = m.shear_y || 0;
  const dx = pt.x - inst.position.x, dy = pt.y - inst.position.y;
  const ux = dx * cos - dy * sin, uy = dx * sin + dy * cos;
  const det = 1 - shx * shy;
  return { x: (ux - shx * uy) / (det * sx) + w / 2, y: (uy - shy * ux) / (det * sy) + h / 2 };
}
function affineInverse(inst, pt, w, h) {
  const p = affineInverseRaw(inst, pt, w, h);
  if (inst.perspective_h) {
    const H = inst.perspective_h;
    const Hinv = solveHomography(
      [hApply(H, 0, 0), hApply(H, 100, 0), hApply(H, 100, 100), hApply(H, 0, 100)],
      [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }, { x: 0, y: 100 }]);
    if (Hinv) return hApply(Hinv, p.x, p.y);
  }
  return p;
}

/* ---------------- 吸附对齐（规格 §8.1） ---------------- */

function computeSnap(starts, dx, dy) {
  const guides = [];
  if (!state.ui.snap || starts.length !== 1) return { dx, dy, guides };
  const inst = starts[0].inst;
  const asset = state.assets.get(inst.asset_id);
  if (!asset || !state.target) return { dx, dy, guides };
  const thr = 8 / state.view.scale;
  const pos = { x: starts[0].pos.x + dx, y: starts[0].pos.y + dy };
  const test = { ...inst, position: pos };
  const box = instAABB(test, asset.w, asset.h);
  const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;

  // 目标线：画布边缘/中心 + 其他实例边/中心
  const tx = [0, state.target.w / 2, state.target.w];
  const ty = [0, state.target.h / 2, state.target.h];
  for (const other of state.instances) {
    if (other.id === inst.id) continue;
    const oa = state.assets.get(other.asset_id);
    if (!oa) continue;
    const ob = instAABB(other, oa.w, oa.h);
    tx.push(ob.x0, (ob.x0 + ob.x1) / 2, ob.x1);
    ty.push(ob.y0, (ob.y0 + ob.y1) / 2, ob.y1);
  }
  let bestX = null, bestY = null;
  for (const src of [box.x0, cx, box.x1]) {
    for (const t of tx) {
      const d = Math.abs((src + dx) - t);
      if (d < thr && (!bestX || d < bestX.d)) bestX = { d, adj: t - src };
    }
  }
  for (const src of [box.y0, cy, box.y1]) {
    for (const t of ty) {
      const d = Math.abs((src + dy) - t);
      if (d < thr && (!bestY || d < bestY.d)) bestY = { d, adj: t - src };
    }
  }
  // 网格吸附
  if (state.ui.grid.on && state.ui.grid.size >= 2) {
    const g = state.ui.grid.size;
    const gx = Math.round((box.x0 + dx) / g) * g, gy = Math.round((box.y0 + dy) / g) * g;
    if (!bestX && Math.abs(gx - (box.x0 + dx)) < thr) bestX = { d: Math.abs(gx - box.x0 - dx), adj: gx - box.x0 };
    if (!bestY && Math.abs(gy - (box.y0 + dy)) < thr) bestY = { d: Math.abs(gy - box.y0 - dy), adj: gy - box.y0 };
  }
  if (bestX) { dx += bestX.adj; guides.push({ axis: 'x', at: box.x0 + bestX.adj + (cx - box.x0) }); }
  if (bestY) { dy += bestY.adj; guides.push({ axis: 'y', at: box.y0 + bestY.adj + (cy - box.y0) }); }
  return { dx, dy, guides };
}

/* ---------------- 抬起 ---------------- */

function onUp(e) {
  if (!drag) return;
  const d = drag; drag = null;
  state._dragging = false;
  switch (d.kind) {
    case 'paint':
      paintUp();
      break;
    case 'polyVertex':
      emit('render');
      break;
    case 'move':
    case 'scale':
    case 'scaleEdge':
    case 'rotate':
    case 'mesh':
    case 'persp':
      d.handle.end();
      state.snapGuides = [];
      emit('render');
      break;
    case 'marquee': {
      const m = state.marquee;
      state.marquee = null;
      if (m && (Math.abs(m.x1 - m.x0) > 3 || Math.abs(m.y1 - m.y0) > 3)) {
        const x0 = Math.min(m.x0, m.x1), x1 = Math.max(m.x0, m.x1);
        const y0 = Math.min(m.y0, m.y1), y1 = Math.max(m.y0, m.y1);
        const hits = [];
        for (const inst of state.instances) {
          const a = state.assets.get(inst.asset_id);
          if (!a) continue;
          const b = instAABB(inst, a.w, a.h);
          if (b.x0 < x1 && b.x1 > x0 && b.y0 < y1 && b.y1 > y0) hits.push(inst.id);
        }
        setSelection(hits);
      }
      emit('render');
      break;
    }
    case 'selected-move-pending':
      break;
  }
  emit('selection');
}

/* ---------------- 双击 ---------------- */

function onDblClick(e) {
  updateCursor(e);
  if (state.mode === 'extract') {
    // 闭合由 pointerdown 手动双击检测处理（优先级高于顶点操作）；此处无额外动作
    return;
  }
  if (state.mode === 'paint') return;
  const img = state.cursor.img;
  const inst = topInstanceAt(img);
  if (inst) {
    state.editCtx = { instanceId: inst.id, kind: inst.mesh ? 'mesh' : 'affine' };
    setSelection([inst.id]);
    emit('editmode');
    emit('render');
  } else {
    fitView(); // 双击空白：适配画布到窗口
  }
}

/* ---------------- 滚轮：裸滚轮缩放画布 / Ctrl+滚轮缩放实例 ---------------- */

function onWheel(e) {
  e.preventDefault();
  updateCursor(e);
  if (e.ctrlKey || e.metaKey) {
    const sel = selectedInstances();
    if (!sel.length) return;
    const f = Math.exp(-e.deltaY * 0.002);
    const boxes = sel.map((i) => {
      const a = state.assets.get(i.asset_id);
      return instAABB(i, a.w, a.h);
    });
    const bb = unionAABB(boxes);
    const c = { x: (bb.x0 + bb.x1) / 2, y: (bb.y0 + bb.y1) / 2 };
    const handle = history.beginInteractive('缩放选中实例');
    for (const inst of sel) {
      handle.change(inst, {
        position: { x: c.x + (inst.position.x - c.x) * f, y: c.y + (inst.position.y - c.y) * f },
        affine: { ...inst.affine, scale_x: (inst.affine.scale_x || 1) * f, scale_y: (inst.affine.scale_y || 1) * f },
      });
    }
    handle.throttle();
    emit('render');
  } else {
    zoomAt({ x: state.cursor.x, y: state.cursor.y }, Math.exp(-e.deltaY * 0.0015));
  }
}

/* ---------------- 键盘（规格 §8.3） ---------------- */

function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
}

function onKeyDown(e) {
  if (isTyping(e)) return;
  // 对话框打开时屏蔽全局快捷键（Esc 由 <dialog> 原生关闭）
  if (document.querySelector('dialog[open]')) return;
  const mod = e.ctrlKey || e.metaKey;
  if (e.code === 'Space' && !e.repeat) { state.spacePan = true; canvas.style.cursor = 'grab'; e.preventDefault(); return; }

  // 多边形绘制中：Ctrl+Z 撤销上一个顶点（优先于历史栈）
  if (mod && e.key.toLowerCase() === 'z' && state.mode === 'extract'
    && state.tool === 'poly' && state.extract?.polyPts.length) {
    e.preventDefault(); emit('extract-pop-poly'); return;
  }
  // 绘制模式：Ctrl+Z 撤销上一笔（笔迹不进历史栈，避免误撤实例操作）
  if (mod && e.key.toLowerCase() === 'z' && !e.shiftKey && state.mode === 'paint') {
    e.preventDefault(); paintUndo(); return;
  }
  if (mod && e.key.toLowerCase() === 'z') {
    e.preventDefault();
    e.shiftKey ? history.redo() : history.undo();
    return;
  }
  if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); history.redo(); return; }
  if (mod && e.key.toLowerCase() === 'a') { e.preventDefault(); setSelection(state.instances.map((i) => i.id)); return; }
  if (mod && e.key.toLowerCase() === 's') { e.preventDefault(); emit('recipe-save-request'); return; }
  if (mod && (e.key.toLowerCase() === 'c')) { copySelection(); e.preventDefault(); return; }
  if (mod && (e.key.toLowerCase() === 'v')) { pasteClipboard(); e.preventDefault(); return; }

  // 多边形绘制中：Backspace 回退一点（优先于实例删除）
  if (e.key === 'Backspace' && state.mode === 'extract'
    && state.tool === 'poly' && state.extract?.polyPts.length) {
    e.preventDefault(); emit('extract-pop-poly'); return;
  }
  if (e.key === 'Delete' || e.key === 'Backspace') { deleteSelection(); e.preventDefault(); return; }
  if (e.key === 'Escape') {
    if (state.mode === 'extract') { emit('extract-cancel'); return; }
    if (state.mode === 'paint') { exitPaint(); return; }
    if (state.editCtx) { state.editCtx = null; emit('editmode'); emit('render'); return; }
    selectOnly(null);
    return;
  }
  if (e.key === 'Enter' && state.mode === 'extract') {
    // 绘制中：闭合多边形（不自动抠图）；调整模式：运行抠图
    if (state.tool === 'poly') { emit('extract-close-poly'); return; }
    if (state.tool === 'adjust' && state.extract?.vertexEdit) { emit('extract-run'); return; }
  }
  if (e.key === 'Tab' && (state.mode === 'edit' || state.mode === 'paint')) {
    // 按住 Tab：只看原图（编辑/绘制模式统一，松开恢复）
    e.preventDefault();
    if (!state.ui.showOriginalOnly) { state.ui.showOriginalOnly = true; emit('render'); }
    return;
  }

  // 方向键微调：1px / Shift+方向键按网格单位（规格 §8.1）
  const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  if (arrows[e.key] && state.selection.size) {
    e.preventDefault();
    const step = e.shiftKey ? (state.ui.grid.on ? state.ui.grid.size : 10) : 1;
    const [ax, ay] = arrows[e.key];
    const handle = history.beginInteractive('微调位置');
    for (const inst of selectedInstances()) {
      handle.change(inst, { position: { x: inst.position.x + ax * step, y: inst.position.y + ay * step } });
    }
    handle.end();
    emit('render');
  }
}

function onKeyUp(e) {
  if (e.code === 'Space') { state.spacePan = false; canvas.style.cursor = ''; }
  if (e.key === 'Tab' && state.ui.showOriginalOnly) { state.ui.showOriginalOnly = false; emit('render'); }
}

function copySelection() {
  const sel = selectedInstances();
  if (sel.length) { state.clipboard = deepClone(sel); emit('toast', { text: `已复制 ${sel.length} 个实例`, kind: 'info' }); }
}

function pasteClipboard() {
  if (!state.clipboard?.length) return;
  const handle = history.beginInteractive('粘贴');
  const clones = state.clipboard.map((c) => {
    const inst = deepClone(c);
    inst.id = uid();
    inst.previewPatch = null;
    inst.position = { x: inst.position.x + 20, y: inst.position.y + 20 };
    addInstanceRaw(inst);
    handle.markAdded(inst);
    return inst;
  });
  setSelection(clones.map((c) => c.id));   // 先选中再落盘
  raiseSelection();
  handle.end();
}

export function deleteSelection() {
  const sel = selectedInstances();
  if (!sel.length) return;
  const handle = history.beginInteractive(sel.length > 1 ? `删除 ${sel.length} 个实例` : `删除：${sel[0].name}`);
  for (const inst of sel) {
    handle.markRemoved(inst);
    removeInstanceRaw(inst.id);
  }
  handle.end();
  state.selection.clear();
  state.editCtx = null;
  emit('selection'); emit('render');
}

/* ---------------- 抠图模式指针（多边形/矩形/椭圆/笔刷） ---------------- */

function extractDown(e, img) {
  const ex = state.extract;
  if (state.tool === 'poly') {
    // 绘制中：双击 = 仅闭合多边形（进入顶点调整，不自动抠图）
    if (e.button === 0) {
      const now = performance.now();
      const thr = 7 / Math.max(0.01, state.view.scale);
      if (lastPolyClick && now - lastPolyClick.t < 250
        && Math.hypot(img.x - lastPolyClick.x, img.y - lastPolyClick.y) < thr) {
        lastPolyClick = null;
        emit('extract-close-poly');
        return;
      }
      lastPolyClick = { t: now, x: img.x, y: img.y };
      ex.polyPts.push({ ...img });
      emit('render');
      return;
    }
    if (e.button === 2 && ex.polyPts.length >= 3) emit('extract-close-poly'); // 右键闭合
    return;
  }
  if (state.tool === 'adjust') {
    // 调整模式：拖拽顶点变形 / Alt+单击删除；单击空白不加点
    if (!ex.polyPts.length) return;
    const thr = 9 / state.view.scale;
    let hit = -1, best = thr;
    ex.polyPts.forEach((p, i) => {
      const d = Math.hypot(p.x - img.x, p.y - img.y);
      if (d < best) { best = d; hit = i; }
    });
    if (hit >= 0) {
      if (e.button === 0 && e.altKey) {
        ex.polyPts.splice(hit, 1);
        ex.maskRev++;
        emit('render');
        return;
      }
      if (e.button === 0) { drag = { kind: 'polyVertex', idx: hit }; return; }
    }
    return;
  }
  if (state.tool === 'rect' || state.tool === 'ellipse') {
    ex.rubber = [{ ...img }, { ...img }];
    drag = { kind: 'rubber' };
    return;
  }
  if (state.tool === 'brush' && e.button === 0) {
    drag = { kind: 'brush', last: img };
    paintBrush(img);
    return;
  }
  if (state.tool === 'select' || state.tool === 'pan') {
    drag = { kind: 'pan', sx: state.cursor.x, sy: state.cursor.y, tx: state.view.tx, ty: state.view.ty };
  }
}

function extractMove(e, img) {
  const ex = state.extract;
  if (drag?.kind === 'rubber' && ex.rubber) {
    ex.rubber[1] = { ...img };
    emit('render');
  } else if (drag?.kind === 'brush') {
    paintBrush(img);
  }
}

function paintBrush(img) {
  const ex = state.extract;
  if (!ex.maskCtx) return;
  const r = ex.brush.size / 2;
  const c = ex.maskCtx;
  c.save();
  if (ex.brush.mode === 'erase') c.globalCompositeOperation = 'destination-out';
  c.fillStyle = '#fff';
  c.beginPath();
  c.arc(img.x, img.y, r, 0, Math.PI * 2);
  c.fill();
  if (drag?.last) {
    // 补间线段避免快速移动断触
    const d = Math.hypot(img.x - drag.last.x, img.y - drag.last.y);
    const steps = Math.ceil(d / (r / 2));
    for (let i = 1; i < steps; i++) {
      c.beginPath();
      c.arc(drag.last.x + (img.x - drag.last.x) * i / steps, drag.last.y + (img.y - drag.last.y) * i / steps, r, 0, Math.PI * 2);
      c.fill();
    }
  }
  c.restore();
  drag.last = { ...img };
  ex.maskRev++;
  emit('render');
}

/* ---------------- 工具切换 ---------------- */

export function setTool(tool) {
  state.tool = tool;
  lastPolyClick = null;
  canvas.style.cursor = tool === 'pan' ? 'grab'
    : tool === 'brush' || tool === 'pick' ? 'crosshair'
      : tool === 'poly' || tool === 'rect' || tool === 'ellipse' ? 'crosshair' : 'default';
  if (state.mode === 'extract' && (tool === 'select')) state.tool = 'pan';
  emit('tool');
  emit('render');
}
