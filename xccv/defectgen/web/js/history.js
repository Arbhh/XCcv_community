/* Defect Augment Studio — 增量快照历史栈（Ledgex 思路，规格 §7.3）
 *
 * 历史条目只记录**发生变更的字段**（position/affine/appearance/blend 等），
 * 未变更字段不复制；图像像素数据不进入历史栈。连续拖动/滑条调整在
 * 300ms 无操作后合并为一条（或由交互在 pointerup 时显式提交）。
 *
 * 条目结构：
 * {
 *   label, ts,
 *   changes: [{id, fields: {字段名: 值}, before: {字段名: 旧值}}],  // 字段级增量
 *   added:   [{index, inst}],       // 新增实例（完整数据，仅在新增时付出内存）
 *   removed: [{index, inst}],       // 删除实例
 * }
 * 选择集与 z 序通过实例数组本身 + selection 快照处理：
 *   order: {before:[ids], after:[ids]}，selection: {before:[ids], after:[ids]}
 */
import { state, emit, addInstanceRaw, removeInstanceRaw, setSelection } from './state.js';
import { deepClone } from './util.js';

const HISTORY_LIMIT = 200;

const stack = [];      // 已提交条目（可撤销）
let index = -1;        // 栈顶位置（-1 表示初始状态）
let pending = null;    // 进行中的条目（拖动/滑条期间）
let pendingTimer = null;

/* 可快照的实例字段（浅层对象字段级快照） */
const SNAP_FIELDS = ['position', 'affine', 'perspective_h', 'mesh', 'appearance',
  'blend', 'blend_flag', 'feather_radius', 'edge_blur', 'opacity', 'name',
  'harmonize', 'harmonize_strength', 'harmonize_ring'];

function snapshotOrder() { return state.instances.map((i) => i.id); }

function begin(label) {
  if (index < stack.length - 1) stack.length = index + 1; // 截断重做分支
  pending = {    label, ts: Date.now(),
    changes: new Map(),   // id -> {inst, fields:{}, before:{}}
    added: [], removed: [],
    orderBefore: snapshotOrder(),
    selBefore: [...state.selection],
  };
  return pending;
}

function trackChange(entry, inst, fields) {
  let rec = entry.changes.get(inst.id);
  if (!rec) {
    rec = { inst, fields: {}, before: {} };
    entry.changes.set(inst.id, rec);
  }
  for (const k of Object.keys(fields)) {
    if (!(k in rec.before)) rec.before[k] = deepClone(inst[k]);   // 首次触碰：捕获旧值
    rec.fields[k] = deepClone(fields[k]);
    inst[k] = deepClone(fields[k]);                              // change 负责赋值
  }
}

/* 语义化提交 API：编辑操作直接调用，生成一条可撤销记录 */
export function commit({ label, changes = [], added = [], removed = [] }) {
  commitPendingIfAny();
  const entry = begin(label);
  for (const c of changes) trackChange(entry, c.inst, c.fields);
  for (const inst of added) {
    entry.added.push({ index: state.instances.indexOf(inst), inst: deepClone(inst) });
  }
  for (const inst of removed) {
    entry.removed.push({ index: inst._removed_index ?? 0, inst: deepClone(inst) });
  }
  finalize();
}

/* 交互式开始：返回句柄，交互过程中调用 .change()，结束时调用 .end()。
 * 同名标签的未完结条目会被复用（滑条连续输入 300ms 节流合并为一条）；不同标签则先落盘再开新条目。 */
export function beginInteractive(label) {
  if (pending && pending.label === label) {
    clearTimeout(pendingTimer);
    return pending._handle;
  }
  commitPendingIfAny();
  if (index < stack.length - 1) stack.length = index + 1; // 截断重做分支
  const entry = begin(label);
  clearTimeout(pendingTimer);
  const handle = {
    change(inst, fields) { trackChange(entry, inst, fields); },
    markAdded(inst) {
      if (!entry.added.find((a) => a.inst.id === inst.id)) {
        entry.added.push({ index: state.instances.indexOf(inst), inst: deepClone(inst) });
      }
    },
    markRemoved(inst) {
      entry.removed.push({ index: Math.max(0, state.instances.indexOf(inst)), inst: deepClone(inst) });
    },
    /* 常规提交（pointerup / 显式确认） */
    end() {
      clearTimeout(pendingTimer);
      finalize();
    },
    /* 节流提交：300ms 无后续变更则自动落盘（滑条场景） */
    throttle() {
      clearTimeout(pendingTimer);
      pendingTimer = setTimeout(() => finalize(), 300);
    },
  };
  entry._handle = handle;
  return handle;
}

/* 拖动期间的轻量接口：对选中实例的某字段做持续变更，自动节流 */
export function dragChange(inst, fields, label = '调整') {
  if (!pending) beginInteractive(label);
  pending.label = label;
  trackChange(pending, inst, fields);
  pending._handle?.throttle();
}

function finalize() {
  if (!pending) return;
  const entry = pending;
  pending = null;
  clearTimeout(pendingTimer);
  if (!entry.changes.size && !entry.added.length && !entry.removed.length
    && entry.orderBefore.join() === snapshotOrder()) return; // 空操作不入栈
  entry.orderAfter = snapshotOrder();
  entry.selAfter = [...state.selection];
  stack.push(entry);
  if (stack.length > HISTORY_LIMIT) stack.shift();
  index = stack.length - 1;
  emit('history');
  if (entry.label.startsWith('录制') || state.recipe.recording) emit('recipe-record', entry);
}

/* ---------------- 撤销 / 重做 ---------------- */

function applyOrder(entry, key) {
  const ids = entry[key];
  if (!ids) return;
  const map = new Map(state.instances.map((i) => [i.id, i]));
  state.instances = ids.map((id) => map.get(id)).filter(Boolean);
}

function applySel(entry, key) {
  const ids = entry[key];
  if (!ids) return;
  state.selection = new Set(ids.filter((id) => state.instances.some((i) => i.id === id)));
}

export function undo() {
  if (index < 0) return false;
  commitPendingIfAny();
  const e = stack[index];
  // 逆序：先撤销 added，再恢复 removed，再还原字段
  for (const a of e.added.slice().reverse()) removeInstanceRaw(a.inst.id);
  for (const r of e.removed.slice().reverse()) addInstanceRaw(deepClone(r.inst), r.index);
  for (const rec of e.changes.values()) {
    const inst = state.instances.find((i) => i.id === rec.inst.id);
    if (inst) for (const k of Object.keys(rec.before)) inst[k] = deepClone(rec.before[k]);
  }
  applyOrder(e, 'orderBefore');
  applySel(e, 'selBefore');
  index--;
  after('撤销：' + e.label);
  return true;
}

export function redo() {
  if (index >= stack.length - 1) return false;
  const e = stack[index + 1];
  for (const r of e.removed) removeInstanceRaw(r.inst.id);
  for (const a of e.added) addInstanceRaw(deepClone(a.inst), a.index);
  for (const rec of e.changes.values()) {
    const inst = state.instances.find((i) => i.id === rec.inst.id);
    if (inst) for (const k of Object.keys(rec.fields)) inst[k] = deepClone(rec.fields[k]);
  }
  applyOrder(e, 'orderAfter');
  applySel(e, 'selAfter');
  index++;
  after('重做：' + e.label);
  return true;
}

function commitPendingIfAny() {
  if (pending) { const p = pending; pending = null; clearTimeout(pendingTimer);
    if (p.changes.size || p.added.length || p.removed.length) {
      p.orderAfter = snapshotOrder(); p.selAfter = [...state.selection];
      stack.push(p); if (stack.length > HISTORY_LIMIT) stack.shift();
      index = stack.length - 1;
    }
  }
}

function after(label) {
  // 清理失效的 previewPatch / 编辑上下文
  for (const inst of state.instances) if (inst.previewPatch && inst.blend !== 'poisson') inst.previewPatch = null;
  if (state.editCtx && !state.instances.some((i) => i.id === state.editCtx.instanceId)) state.editCtx = null;
  emit('history');
  emit('selection');
  emit('render');
  emit('toast', { text: label, kind: 'info' });
}

export function canUndo() { return index >= 0; }
export function canRedo() { return index < stack.length - 1; }
export function historyList() {
  return stack.map((e, i) => ({
    label: e.label, ts: e.ts, current: i === index,
    future: i > index,
  }));
}

/* 点击历史面板跳转到任意状态 */
export function jumpTo(targetIndex) {
  commitPendingIfAny();
  while (index > targetIndex) { if (!undoSilent()) break; }
  while (index < targetIndex) { if (!redoSilent()) break; }
  emit('history'); emit('selection'); emit('render');
}
function undoSilent() {
  if (index < 0) return false;
  const e = stack[index];
  for (const a of e.added.slice().reverse()) removeInstanceRaw(a.inst.id);
  for (const r of e.removed.slice().reverse()) addInstanceRaw(deepClone(r.inst), r.index);
  for (const rec of e.changes.values()) {
    const inst = state.instances.find((i) => i.id === rec.inst.id);
    if (inst) for (const k of Object.keys(rec.before)) inst[k] = deepClone(rec.before[k]);
  }
  applyOrder(e, 'orderBefore'); applySel(e, 'selBefore');
  index--;
  return true;
}
function redoSilent() {
  if (index >= stack.length - 1) return false;
  const e = stack[index + 1];
  for (const r of e.removed) removeInstanceRaw(r.inst.id);
  for (const a of e.added) addInstanceRaw(deepClone(a.inst), a.index);
  for (const rec of e.changes.values()) {
    const inst = state.instances.find((i) => i.id === rec.inst.id);
    if (inst) for (const k of Object.keys(rec.fields)) inst[k] = deepClone(rec.fields[k]);
  }
  applyOrder(e, 'orderAfter'); applySel(e, 'selAfter');
  index++;
  return true;
}
