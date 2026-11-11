/* Defect Augment Studio — Recipe 录制与回放（规格 §6.5）
 * 录制：监听历史条目，转成语义步骤（place/transform/enhance/blend），
 *       位置存为图像宽高比例、缩放锚定短边，保证跨分辨率回放一致。
 * 回放：在任意目标图上重放步骤；批量时由后端按背景图逐张实例化并叠加抖动。
 */
import { state, emit, on } from './state.js';
import { deepClone, uid } from './util.js';

let instIndex = new Map();   // 录制期间 instId -> 步骤索引
let skippedCount = 0;

export function initRecipe() {
  on('recipe-record', (entry) => recordEntry(entry));
  on('recipe-save-request', saveRecipeDialog);
}

export function toggleRecording() {
  const r = state.recipe;
  if (!r.recording) {
    r.recording = true;
    r.steps = [];
    r.srcDims = state.target ? { w: state.target.w, h: state.target.h } : null;
    instIndex = new Map();
    skippedCount = 0;
    emit('toast', { text: '开始录制 Recipe — 此后的放置/变换/外观/融合操作将被记录', kind: 'ok' });
  } else {
    r.recording = false;
    emit('toast', {
      text: `录制完成：${r.steps.length} 个步骤` + (skippedCount ? `（${skippedCount} 个针对录制前实例的操作未纳入）` : ''),
      kind: 'ok',
    });
    if (r.steps.length) saveRecipeDialog();
  }
  emit('render');
}

function recordEntry(entry) {
  const r = state.recipe;
  if (!r.recording || !r.srcDims) return;
  const W = r.srcDims.w, H = r.srcDims.h;
  for (const a of entry.added) {
    const inst = a.inst;
    const idx = r.steps.length;
    instIndex.set(inst.id, idx);
    r.steps.push({
      type: 'place',
      asset_id: inst.asset_id,
      rel_position: [inst.position.x / W, inst.position.y / H],
      affine: deepClone(inst.affine),
      perspective_h: inst.perspective_h ? [...inst.perspective_h] : null,
      mesh: inst.mesh ? deepClone(inst.mesh) : null,
      appearance: deepClone(inst.appearance),
      blend: inst.blend, blend_flag: inst.blend_flag,
      feather_radius: inst.feather_radius, edge_blur: inst.edge_blur ?? 0,
      opacity: inst.opacity,
      noise_seed: inst.noise_seed, variant_id: inst.variant_id || null,
    });
  }
  for (const rec of entry.changes.values()) {
    const idx = instIndex.get(rec.inst.id);
    if (idx == null) { skippedCount++; continue; } // 录制前已存在的实例：跳过
    const f = rec.fields;
    const kind = f.appearance ? 'enhance'
      : (f.blend || f.blend_flag || f.feather_radius != null || f.edge_blur != null || f.opacity != null) ? 'blend'
        : 'transform';
    const step = { type: kind, target: idx };
    for (const k of ['affine', 'perspective_h', 'mesh', 'position']) if (f[k] != null) step[k] = deepClone(f[k]);
    if (f.appearance) step.appearance = deepClone(f.appearance);
    for (const k of ['blend', 'blend_flag', 'feather_radius', 'edge_blur', 'opacity',
      'harmonize', 'harmonize_strength', 'harmonize_ring']) if (f[k] != null) step[k] = f[k];
    r.steps.push(step);
  }
  emit('recipe-state');
}

/* ---------------- 回放 ---------------- */

/* 将 recipe 步骤应用到当前目标图（可传覆盖 dims 用于批量预演）。
 * 返回实例参数列表（place 时创建，transform/enhance/blend 修改其目标）。 */
export function materializeSteps(steps, dims, srcDims) {
  const k = Math.min(dims.w, dims.h) / Math.max(1, Math.min(srcDims.w, srcDims.h)); // 短边比例
  const out = [];
  const map = new Map();
  for (const s of steps) {
    if (s.type === 'place') {
      const inst = {
        asset_id: s.asset_id,
        position: { x: s.rel_position[0] * dims.w, y: s.rel_position[1] * dims.h },
        affine: { ...deepClone(s.affine), scale_x: (s.affine.scale_x ?? 1) * k, scale_y: (s.affine.scale_y ?? 1) * k },
        perspective_h: s.perspective_h ? [...s.perspective_h] : null,
        mesh: s.mesh ? scaleMesh(deepClone(s.mesh), k) : null,
        appearance: deepClone(s.appearance),
        blend: s.blend, blend_flag: s.blend_flag || 'normal',
        feather_radius: (s.feather_radius ?? 3) * k,
        edge_blur: (s.edge_blur ?? 0) * k,
        opacity: s.opacity ?? 1,
        harmonize: s.harmonize || 'off',
        harmonize_strength: s.harmonize_strength ?? 0.8,
        harmonize_ring: Math.round((s.harmonize_ring ?? 10) * k),
        noise_seed: s.noise_seed ?? 0,
        variant_id: s.variant_id || null,
      };
      map.set(out.length, inst);
      out.push(inst);
    } else {
      const inst = map.get(s.target);
      if (!inst) continue;
      if (s.affine) inst.affine = { ...deepClone(s.affine), scale_x: (s.affine.scale_x ?? 1) * k, scale_y: (s.affine.scale_y ?? 1) * k };
      if (s.perspective_h !== undefined) inst.perspective_h = s.perspective_h ? [...s.perspective_h] : null;
      if (s.mesh !== undefined) inst.mesh = s.mesh ? scaleMesh(deepClone(s.mesh), k) : null;
      if (s.appearance) inst.appearance = deepClone(s.appearance);
      if (s.blend) inst.blend = s.blend;
      if (s.blend_flag) inst.blend_flag = s.blend_flag;
      if (s.feather_radius != null) inst.feather_radius = s.feather_radius * k;
      if (s.edge_blur != null) inst.edge_blur = s.edge_blur * k;
      if (s.opacity != null) inst.opacity = s.opacity;
      if (s.harmonize) inst.harmonize = s.harmonize;
      if (s.harmonize_strength != null) inst.harmonize_strength = s.harmonize_strength;
      if (s.harmonize_ring != null) inst.harmonize_ring = Math.round(s.harmonize_ring * k);
    }
  }
  return out;
}

function scaleMesh(mesh, k) {
  return { ...mesh, grid: mesh.grid.map(([dx, dy]) => [dx * k, dy * k]) };
}

export async function replayRecipe(rec, opts = {}) {
  const { placeOnCanvas } = await import('./main.js');
  if (!state.target) { emit('toast', { text: '请先打开目标图像', kind: 'warn' }); return; }
  const insts = materializeSteps(rec.steps, { w: state.target.w, h: state.target.h }, rec.srcDims);
  placeOnCanvas(insts);
  emit('toast', { text: `Recipe「${rec.name}」已回放：${insts.length} 个实例`, kind: 'ok' });
}

/* ---------------- 保存 / 加载 ---------------- */

export function saveRecipeDialog() {
  const r = state.recipe;
  if (!r.steps.length) {
    emit('toast', { text: '当前没有可保存的 Recipe 步骤', kind: 'warn' });
    return;
  }
  const nameInput = document.getElementById('recipeName');
  const name = prompt('Recipe 名称：', nameInput?.value || '缺陷配方_' + new Date().toISOString().slice(0, 10));
  if (!name) return;
  saveRecipe({ recipe_id: uid(), name, srcDims: r.srcDims, steps: deepClone(r.steps), created_at: Date.now() });
}

export function saveRecipe(rec) {
  const blob = new Blob([JSON.stringify(rec, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = rec.name.replace(/[\\/:*?"<>|]/g, '_') + '.recipe.json';
  a.click();
  URL.revokeObjectURL(a.href);
  emit('toast', { text: `Recipe「${rec.name}」已导出 JSON（${rec.steps.length} 步）`, kind: 'ok' });
}

export async function loadRecipeFile(file) {
  try {
    const rec = JSON.parse(await file.text());
    if (!rec.steps || !rec.srcDims) throw new Error('缺少 steps/srcDims 字段');
    if (!state.recipe.recording) {
      state.recipe.steps = rec.steps;
      state.recipe.srcDims = rec.srcDims;
    }
    await replayRecipe(rec);
    return rec;
  } catch (e) {
    emit('toast', { text: 'Recipe 加载失败：' + e.message, kind: 'err' });
    return null;
  }
}

export function currentRecipeSummary() {
  const r = state.recipe;
  return { recording: r.recording, steps: r.steps.length };
}
