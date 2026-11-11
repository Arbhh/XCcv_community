/* Defect Augment Studio — 面板层：资产 / 变换 / 外观增强 / 融合 / 视图 / 历史 / 状态栏 */
import { state, emit, on, selectedInstances, getInstance, setSelection, raiseSelection, assetLumaStats } from './state.js';
import * as history from './history.js';
import * as assets from './assets.js';
import { APPEARANCE_SPEC, normAppearance, DEFAULT_APPEARANCE, deepClone, instAABB, unionAABB, clamp, elasticMesh, clampMesh } from './util.js';
import { setTool, deleteSelection } from './interact.js';
import { bumpScene } from './render.js';

const $ = (id) => document.getElementById(id);
let refreshQueued = false;

export function initPanels() {
  bindToolbar();
  bindTransformPanel();
  bindAppearancePanel();
  bindBlendPanel();
  bindViewPanel();
  bindAlignBar();
  bindHistoryStrip();
  bindAssetPanel();
  bindStatusbar();

  on('render', requestRefresh);
  on('selection', () => { requestRefresh(); updateAlignBar(); });
  on('history', requestRefresh);
  on('assets', renderAssetList);
  on('mode', () => { requestRefresh(); });
  on('tool', updateToolButtons);
  on('extract-state', requestRefresh);
  on('editmode', () => { updateEditModeBar(); requestRefresh(); });
  on('preview-note', (d) => {
    const el = $('blendNote');
    if (el && state.selection.has(d.instId)) {
      el.textContent = d.used === 'poisson' ? 'Poisson 预览已更新' : '已回退 Feather：' + (d.notes?.join('；') || '');
    }
  });
  on('histogram', drawHistogram);
  on('toast', showToast);
  on('status', updateStatus);

  renderAssetList();
  requestRefresh();
}

function requestRefresh() {
  if (refreshQueued) return;
  refreshQueued = true;
  setTimeout(() => { refreshQueued = false; refreshPanels(); }, 16);
}

/* ================= 顶部工具条 ================= */

function bindToolbar() {
  $('btnOpenTarget').onclick = () => pickFile((f) => window.__studio.openTarget(f));
  $('btnExtractMode').onclick = () => pickFile((f) => window.__studio.enterExtract(f, f.name));
  $('btnUndo').onclick = () => history.undo();
  $('btnRedo').onclick = () => history.redo();
  $('btnDelete').onclick = () => deleteSelection();
  $('btnExport').onclick = () => emit('export-request');
  $('btnBatch').onclick = () => emit('batch-request');
  $('btnRecipe').onclick = () => emit('recipe-dialog');
  $('btnRec').onclick = toggleRecord;
  bindToolButtons();
}

function pickFile(cb) {
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = 'image/*';
  inp.onchange = () => { if (inp.files[0]) cb(inp.files[0]); };
  inp.click();
}

function bindToolButtons() {
  for (const b of document.querySelectorAll('[data-tool]')) {
    b.onclick = () => setTool(b.dataset.tool);
  }
  $('chkGrid').onchange = (e) => { state.ui.grid.on = e.target.checked; emit('render'); };
  $('chkSnap').onchange = (e) => { state.ui.snap = e.target.checked; };
  $('chkBoundary').onchange = (e) => { state.ui.boundaryHighlight = e.target.checked; bumpScene(); emit('render'); };
  $('chkSplit').onchange = (e) => { state.ui.split.on = e.target.checked; emit('render'); };
  $('chkMagnifier').onchange = (e) => { state.ui.magnifier.on = e.target.checked; emit('render'); };
  $('chkHeatmap').onchange = (e) => { state.ui.heatmap = e.target.checked; emit('render'); };
  $('chkSmartPlace').onchange = (e) => { state.ui.smartPlace = e.target.checked; };
  $('selGridSize').onchange = (e) => { state.ui.grid.size = +e.target.value; emit('render'); };
  $('btnHelp').onclick = () => $('helpDlg').showModal();
  $('helpClose').onclick = () => $('helpDlg').close();
  window.addEventListener('keydown', (e) => {
    if (e.key === 'F1' && !e.target.closest?.('input,textarea')) { e.preventDefault(); $('helpDlg').showModal(); }
  });
}

function updateToolButtons() {
  for (const b of document.querySelectorAll('[data-tool]')) {
    b.classList.toggle('active', b.dataset.tool === state.tool);
  }
}

async function toggleRecord() {
  const { toggleRecording } = await import('./recipe.js');
  toggleRecording();
}

/* ================= 变换面板 ================= */

function bindTransformPanel() {
  // apply(inst, v) 返回要赋值的新字段对象；由 history.change 统一赋值并捕获增量
  const bind = (id, apply) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener('input', () => {
      const sel = selectedInstances();
      if (!sel.length) return;
      const v = parseFloat(el.value);
      if (Number.isNaN(v)) return;
      const handle = history.beginInteractive('变换调整');
      for (const inst of sel) handle.change(inst, apply(inst, v));
      handle.throttle();
      bumpScene();
      emit('render');
    });
  };

  bind('tfPosX', (i, v) => ({ position: { ...i.position, x: v } }));
  bind('tfPosY', (i, v) => ({ position: { ...i.position, y: v } }));
  bind('tfScaleX', (i, v) => ({ affine: { ...i.affine, scale_x: v } }));
  bind('tfScaleY', (i, v) => ({ affine: { ...i.affine, scale_y: v } }));
  bind('tfRot', (i, v) => ({ affine: { ...i.affine, rotation: v } }));
  bind('tfShearX', (i, v) => ({ affine: { ...i.affine, shear_x: v } }));
  bind('tfShearY', (i, v) => ({ affine: { ...i.affine, shear_y: v } }));

  $('tfLock').onchange = (e) => { state._scaleLock = e.target.checked; };
  // 锁定联动
  for (const id of ['tfScaleX', 'tfScaleY']) {
    $(id).addEventListener('input', () => {
      if (!$('tfLock').checked) return;
      const v = parseFloat($(id).value);
      const otherId = id === 'tfScaleX' ? 'tfScaleY' : 'tfScaleX';
      $(otherId).value = v;
      const key = id === 'tfScaleX' ? 'scale_y' : 'scale_x';
      for (const inst of selectedInstances()) inst.affine[key] = v;
    });
  }
  $('tfFlipH').onclick = () => applyToSelection('水平翻转', (i) => ({ affine: { ...i.affine, scale_x: (i.affine.scale_x || 1) * -1 } }));
  $('tfFlipV').onclick = () => applyToSelection('垂直翻转', (i) => ({ affine: { ...i.affine, scale_y: (i.affine.scale_y || 1) * -1 } }));
  $('tfReset').onclick = () => applyToSelection('重置变换', () => ({
    affine: { scale_x: 1, scale_y: 1, rotation: 0, shear_x: 0, shear_y: 0 },
    perspective_h: null, mesh: null,
  }));
  $('tfRaise').onclick = () => { raiseSelection(); };
  $('btnEditAffine').onclick = () => setEditMode('affine');
  $('btnEditPersp').onclick = () => setEditMode('persp');
  $('btnEditMesh').onclick = () => setEditMode('mesh');
  $('btnElastic').onclick = applyElastic;
  // 实例反存资产：渲染实例图层（含变形/外观）→ 入库
  $('btnInstToAsset').onclick = async () => {
    const sel = selectedInstances();
    if (sel.length !== 1) { showToast({ text: '请先选中单个实例', kind: 'warn' }); return; }
    const inst = sel[0];
    const btn = $('btnInstToAsset');
    btn.disabled = true;
    try {
      const { requestLayer } = await import('./render.js');
      const layer = await requestLayer(inst);
      const bmp = layer.bitmap;
      const rgba = new OffscreenCanvas(bmp.width, bmp.height);
      rgba.getContext('2d').drawImage(bmp, 0, 0);
      // alpha → mask
      const data = rgba.getContext('2d').getImageData(0, 0, bmp.width, bmp.height);
      const mask = new OffscreenCanvas(bmp.width, bmp.height);
      const md = new ImageData(new Uint8ClampedArray(data.data), bmp.width, bmp.height);
      for (let i = 0; i < md.data.length; i += 4) {
        md.data[i] = md.data[i + 1] = md.data[i + 2] = 255;
        // alpha 保持
      }
      mask.getContext('2d').putImageData(md, 0, 0);
      const a = await assets.addAsset({
        name: inst.name + '_变体',
        tags: ['实例导出'],
        rgbaCanvas: rgba,
        maskCanvas: mask,
        source: { image: state.target?.name || '', fromInstance: inst.id },
      });
      showToast({ text: `已存为资产：${a.name}`, kind: 'ok' });
    } catch (e) {
      showToast({ text: '存为资产失败：' + e.message, kind: 'err' });
    } finally {
      btn.disabled = false;
    }
  };
}

function setEditMode(kind) {
  const sel = selectedInstances();
  if (sel.length !== 1) { showToast({ text: '请先选中单个实例', kind: 'warn' }); return; }
  if (kind === 'affine') { state.editCtx = null; emit('editmode'); emit('render'); return; }
  state.editCtx = { instanceId: sel[0].id, kind };
  if (kind === 'mesh' && !sel[0].mesh) {
    sel[0].mesh = { cols: 4, rows: 4, grid: Array.from({ length: 25 }, () => [0, 0]) };
    bumpScene();
  }
  emit('editmode');
  emit('render');
  showToast({ text: { persp: '拖拽四角调整透视', mesh: '拖拽网格交点变形；位移限制在半格内' }[kind], kind: 'info' });
}

function applyElastic() {
  const sel = selectedInstances();
  if (sel.length !== 1) { showToast({ text: '请先选中单个实例', kind: 'warn' }); return; }
  const inst = sel[0];
  const asset = state.assets.get(inst.asset_id);
  const seed = (Math.random() * 0xffffffff) >>> 0;
  const handle = history.beginInteractive('弹性扰动');
  handle.change(inst, { mesh: clampMesh(elasticMesh(seed, 4, 4, asset.w, asset.h, 0.22), asset.w, asset.h) });
  handle.end();
  state.editCtx = { instanceId: inst.id, kind: 'mesh' };
  emit('editmode');
  bumpScene();
  emit('render');
}

/* fn(inst) 返回要赋值的新字段对象（不得直接改实例），由 history.change 统一赋值 */
function applyToSelection(label, fn) {
  const sel = selectedInstances();
  if (!sel.length) return;
  const handle = history.beginInteractive(label);
  for (const inst of sel) {
    const fields = fn(inst);
    if (fields) handle.change(inst, fields);
  }
  handle.end();
  bumpScene();
  emit('render');
}

/* ================= 外观增强面板（规格 §6.3） ================= */

function bindAppearancePanel() {
  for (const spec of APPEARANCE_SPEC) {
    const slider = $('ap_' + spec.key);
    const num = $('apn_' + spec.key);
    if (!slider) continue;
    const onInput = () => {
      const v = parseFloat(slider.value);
      num.value = spec.step < 1 ? v.toFixed(2) : v;
      const sel = selectedInstances();
      if (!sel.length) return;
      const handle = history.beginInteractive('外观调整');
      for (const inst of sel) {
        const app = { ...normAppearance(inst.appearance), [spec.key]: v };
        handle.change(inst, { appearance: app });
      }
      handle.throttle();
      bumpScene();
      emit('render');
    };
    slider.addEventListener('input', onInput);
    num.addEventListener('change', () => {
      slider.value = num.value;
      onInput();
    });
  }
  $('apReset').onclick = () => applyToSelection('重置外观', () => ({ appearance: { ...DEFAULT_APPEARANCE } }));
  $('apApplySel').onclick = () => {
    // 将当前滑条组合应用到所有选中实例
    const sel = selectedInstances();
    if (!sel.length) { showToast({ text: '请先选中实例', kind: 'warn' }); return; }
    const app = {};
    for (const spec of APPEARANCE_SPEC) app[spec.key] = parseFloat($('ap_' + spec.key).value);
    const handle = history.beginInteractive('批量统一外观');
    for (const inst of sel) handle.change(inst, { appearance: { ...app } });
    handle.end();
    bumpScene();
    emit('render');
    showToast({ text: `已将外观参数应用到 ${sel.length} 个实例`, kind: 'ok' });
  };
}

/* ================= 融合面板（规格 §6.4） ================= */

function bindBlendPanel() {
  for (const mode of ['direct', 'feather', 'poisson']) {
    $('bl_' + mode).onchange = () => {
      if (!$('bl_' + mode).checked) return;
      applyToSelection('融合方式→' + mode, () => ({ blend: mode }));
      emit('render');
    };
  }
  $('blMixed').onchange = (e) => applyToSelection('Mixed Clone', () => ({ blend_flag: e.target.checked ? 'mixed' : 'normal' }));
  $('blFeather').addEventListener('input', () => {
    $('blFeatherN').value = $('blFeather').value;
    const sel = selectedInstances();
    if (!sel.length) return;
    const handle = history.beginInteractive('羽化半径');
    for (const inst of sel) handle.change(inst, { feather_radius: parseFloat($('blFeather').value) });
    handle.throttle();
    bumpScene();
    emit('render');
  });
  // 边缘模糊：轮廓与颜色一起向背景渐隐
  $('blEdgeBlur').addEventListener('input', () => {
    $('blEdgeBlurN').value = (+$('blEdgeBlur').value).toFixed(1);
    const sel = selectedInstances();
    if (!sel.length) return;
    const handle = history.beginInteractive('边缘模糊');
    for (const inst of sel) handle.change(inst, { edge_blur: parseFloat($('blEdgeBlur').value) });
    handle.throttle();
    bumpScene();
    emit('render');
  });
  $('blOpacity').addEventListener('input', () => {
    $('blOpacityN').value = $('blOpacity').value;
    const sel = selectedInstances();
    if (!sel.length) return;
    const handle = history.beginInteractive('不透明度');
    for (const inst of sel) handle.change(inst, { opacity: parseFloat($('blOpacity').value) });
    handle.throttle();
    bumpScene();
    emit('render');
  });
  $('blPreview').onchange = (e) => {
    state.ui.poissonPreview = e.target.checked;
    if (!e.target.checked) for (const i of state.instances) i.previewPatch = null;
    bumpScene();
    emit('render');
  };
  // 光照协调（Reinhard 环带统计迁移）
  $('blHarmonize').onchange = (e) => {
    $('blHarmRow').style.display = e.target.checked ? '' : 'none';
    $('blHarmRow2').style.display = e.target.checked ? '' : 'none';
    applyToSelection('光照协调', () => ({ harmonize: e.target.checked ? 'ring' : 'off' }));
  };
  $('blHarmStr').addEventListener('input', () => {
    $('blHarmStrN').value = (+$('blHarmStr').value).toFixed(2);
    const sel = selectedInstances();
    if (!sel.length) return;
    const handle = history.beginInteractive('协调强度');
    for (const inst of sel) handle.change(inst, { harmonize_strength: parseFloat($('blHarmStr').value) });
    handle.throttle();
    bumpScene();
    emit('render');
  });
  $('blHarmRing').addEventListener('input', () => {
    $('blHarmRingN').value = $('blHarmRing').value + 'px';
    const sel = selectedInstances();
    if (!sel.length) return;
    const handle = history.beginInteractive('协调环宽');
    for (const inst of sel) handle.change(inst, { harmonize_ring: parseInt($('blHarmRing').value) });
    handle.throttle();
    bumpScene();
    emit('render');
  });
  // 自适应外观：把缺陷亮度 p5/p95 映射到 40/220，统一数据集灰域
  $('apAutoLevel').onclick = () => {
    const sel = selectedInstances();
    if (!sel.length) { showToast({ text: '请先选中实例', kind: 'warn' }); return; }
    const handle = history.beginInteractive('自适应外观');
    let applied = 0;
    for (const inst of sel) {
      const st = assetLumaStats(inst.asset_id);
      if (!st) continue;
      const cur = normAppearance(inst.appearance);
      const b0 = cur.brightness, c0 = cur.contrast;
      // 以当前参数反推原始亮度范围（v' = (v-128)*c + 128 + b），再求映射到 40..220 的新参数
      const inv = (v) => (v - 128 - b0) / c0 + 128;
      const lo = clamp(inv(st.p05), 0, 255), hi = clamp(inv(st.p95), 0, 255);
      const c1 = clamp(180 / Math.max(8, hi - lo), 0.5, 2);
      const b1 = clamp(40 - (lo - 128) * c1 - 128, -50, 50);
      handle.change(inst, {
        appearance: { ...cur, contrast: +c1.toFixed(2), brightness: Math.round(b1) },
      });
      applied++;
    }
    handle.end();
    bumpScene();
    emit('render');
    showToast({ text: `已自适应 ${applied} 个实例的亮度/对比度（p5→40，p95→220）`, kind: 'ok' });
  };
}

/* ================= 视图面板（分屏/放大镜/网格等开关在工具条） ================= */

function bindViewPanel() {
  $('mgZoom').oninput = (e) => { state.ui.magnifier.zoom = +e.target.value; $('mgZoomN').value = e.target.value + '×'; emit('render'); };
  $('mgRadius').oninput = (e) => { state.ui.magnifier.r = +e.target.value; $('mgRadiusN').value = e.target.value + 'px'; emit('render'); };
}

/* ================= 对齐工具条（规格 §8.2） ================= */

function bindAlignBar() {
  const ops = {
    alLeft: (b, t) => ({ dx: t.x0 - b.x0, dy: 0 }),
    alCenterH: (b, t) => ({ dx: (t.x0 + t.x1) / 2 - (b.x0 + b.x1) / 2, dy: 0 }),
    alRight: (b, t) => ({ dx: t.x1 - b.x1, dy: 0 }),
    alTop: (b, t) => ({ dx: 0, dy: t.y0 - b.y0 }),
    alCenterV: (b, t) => ({ dx: 0, dy: (t.y0 + t.y1) / 2 - (b.y0 + b.y1) / 2 }),
    alBottom: (b, t) => ({ dx: 0, dy: t.y1 - b.y1 }),
  };
  for (const [id, fn] of Object.entries(ops)) {
    $(id).onclick = () => alignInstances(fn);
  }
  $('alDistH').onclick = () => distribute(true);
  $('alDistV').onclick = () => distribute(false);
}

function alignInstances(fn) {
  const sel = selectedInstances();
  if (sel.length < 2) return;
  const infos = sel.map((inst) => {
    const a = state.assets.get(inst.asset_id);
    return { inst, box: instAABB(inst, a.w, a.h) };
  });
  const t = unionAABB(infos.map((i) => i.box));
  const handle = history.beginInteractive('对齐分布');
  for (const { inst, box } of infos) {
    const { dx, dy } = fn(box, t);
    if (dx || dy) handle.change(inst, { position: { x: inst.position.x + dx, y: inst.position.y + dy } });
  }
  handle.end();
  emit('render');
}

function distribute(horizontal) {
  const sel = selectedInstances();
  if (sel.length < 3) { showToast({ text: '等距分布至少需要 3 个实例', kind: 'warn' }); return; }
  const infos = sel.map((inst) => {
    const a = state.assets.get(inst.asset_id);
    return { inst, box: instAABB(inst, a.w, a.h) };
  }).sort((a, b) => horizontal ? a.box.x0 - b.box.x0 : a.box.y0 - b.box.y0);
  const first = infos[0].box, last = infos[infos.length - 1].box;
  const totalW = infos.reduce((s, i) => s + (horizontal ? i.box.x1 - i.box.x0 : i.box.y1 - i.box.y0), 0);
  const span = horizontal ? last.x1 - first.x0 : last.y1 - first.y0;
  const gap = (span - totalW) / (infos.length - 1);
  const handle = history.beginInteractive('等距分布');
  let cursor = horizontal ? first.x0 : first.y0;
  for (const { inst, box } of infos) {
    const size = horizontal ? box.x1 - box.x0 : box.y1 - box.y0;
    const target = cursor;
    const dx = horizontal ? target - box.x0 : 0;
    const dy = horizontal ? 0 : target - box.y0;
    if (dx || dy) handle.change(inst, { position: { x: inst.position.x + dx, y: inst.position.y + dy } });
    cursor += size + gap;
  }
  handle.end();
  emit('render');
}

function updateAlignBar() {
  const bar = $('alignBar');
  bar.style.display = state.selection.size >= 2 ? 'flex' : 'none';
}

function updateEditModeBar() {
  const bar = $('editModeBar');
  const ec = state.editCtx;
  bar.style.display = ec ? 'flex' : 'none';
  if (ec) {
    for (const b of bar.querySelectorAll('[data-edit]')) {
      b.classList.toggle('active', b.dataset.edit === ec.kind);
    }
  }
}

/* ================= 历史条（可点击跳转，规格 §8.5） ================= */

function bindHistoryStrip() {
  on('history', () => {
    const el = $('historyStrip');
    const list = history.historyList();
    el.innerHTML = '';
    const mk = (label, cls, cb) => {
      const b = document.createElement('button');
      b.className = 'hist-item ' + cls;
      b.textContent = label;
      b.onclick = cb;
      el.appendChild(b);
    };
    mk('⏮ 初始', list.length ? (history.canUndo() ? 'future' : 'current') : 'current',
      () => history.jumpTo(-1));
    list.forEach((e, i) => {
      mk((e.future ? '' : '✓ ') + e.label, e.future ? 'future' : 'current', () => history.jumpTo(i));
    });
    el.scrollLeft = el.scrollWidth;
    $('btnUndo').disabled = !history.canUndo();
    $('btnRedo').disabled = !history.canRedo();
  });
}

/* ================= 资产面板（规格 §8.4） ================= */

function bindAssetPanel() {
  $('assetSearch').oninput = renderAssetList;
  $('btnClearAssetFilter')?.addEventListener('click', () => { $('assetSearch').value = ''; renderAssetList(); });

  const panel = $('assetList');
  panel.addEventListener('dragover', (e) => e.preventDefault());
  panel.addEventListener('drop', (e) => {
    e.preventDefault();
    for (const f of e.dataTransfer.files) {
      if (f.type.startsWith('image/')) window.__studio.enterExtract(f, f.name);
    }
  });
}

let activeTagFilter = null;

function renderAssetList() {
  const list = $('assetList');
  if (!list) return;
  const q = ($('assetSearch').value || '').toLowerCase();
  list.innerHTML = '';
  const all = [...state.assets.values()]
    .filter((a) => !q || a.name.toLowerCase().includes(q) || a.tags.some((t) => t.toLowerCase().includes(q)))
    .sort((a, b) => b.created_at - a.created_at);

  // 标签汇总
  const tagBar = $('assetTagBar');
  tagBar.innerHTML = '';
  const tagSet = new Set();
  for (const a of state.assets.values()) a.tags.forEach((t) => tagSet.add(t));
  for (const t of tagSet) {
    const chip = document.createElement('button');
    chip.className = 'tag-chip' + (activeTagFilter === t ? ' active' : '');
    chip.textContent = t;
    chip.onclick = () => { activeTagFilter = activeTagFilter === t ? null : t; renderAssetList(); };
    tagBar.appendChild(chip);
  }

  for (const a of all) {
    if (activeTagFilter && !a.tags.includes(activeTagFilter)) continue;
    const card = document.createElement('div');
    card.className = 'asset-card';
    card.draggable = true;

    const img = document.createElement('img');
    img.src = a.previewUrl;
    img.className = 'asset-thumb';
    img.title = `${a.name}（${a.w}×${a.h}）`;
    card.appendChild(img);

    const nameEl = document.createElement('div');
    nameEl.className = 'asset-name';
    nameEl.textContent = a.name;
    nameEl.title = '双击重命名';
    nameEl.ondblclick = () => {
      const inp = document.createElement('input');
      inp.value = a.name;
      inp.className = 'asset-rename';
      nameEl.replaceWith(inp);
      inp.focus(); inp.select();
      const done = () => { assets.updateAsset(a.id, { name: inp.value.trim() || a.name }); };
      inp.onblur = done;
      inp.onkeydown = (e) => { if (e.key === 'Enter') done(); if (e.key === 'Escape') inp.blur(); };
    };
    card.appendChild(nameEl);

    const meta = document.createElement('div');
    meta.className = 'asset-meta';
    meta.textContent = `${a.w}×${a.h}` + (a.variants.length ? ` · ${a.variants.length} 变体` : '');
    card.appendChild(meta);

    // 变体标记：堆叠展示
    if (a.variants.length) {
      const vs = document.createElement('div');
      vs.className = 'variant-chips';
      a.variants.forEach((v) => {
        const chip = document.createElement('span');
        chip.className = 'variant-chip';
        chip.textContent = v.form_seed ? '形态' + String(v.form_seed).slice(-4) : '外观';
        chip.title = JSON.stringify(v.appearance || {});
        chip.onclick = () => placeAsset(a, { variant: v });
        const x = document.createElement('i');
        x.textContent = '×';
        x.onclick = (e) => { e.stopPropagation(); assets.removeVariant(a.id, v.variant_id); };
        chip.appendChild(x);
        vs.appendChild(chip);
      });
      card.appendChild(vs);
    }

    const btns = document.createElement('div');
    btns.className = 'asset-btns';
    const mkBtn = (label, fn, title) => {
      const b = document.createElement('button');
      b.textContent = label; b.title = title || label;
      b.onclick = fn;
      return b;
    };
    btns.appendChild(mkBtn('放置', () => placeAsset(a), '放置到画布（智能吸附平滑区域；拖拽+Alt 可绕过吸附）'));
    btns.appendChild(mkBtn('存变体', () => saveVariantFromSelection(a), '将选中实例的外观/形态存为该资产的变体'));
    btns.appendChild(mkBtn('删', async () => {
      const usedBy = [...state.instances].filter((i) => i.asset_id === a.id).length;
      if (usedBy > 0) {
        if (!confirm(`资产「${a.name}」被 ${usedBy} 个实例使用，连同实例一起删除？`)) return;
        await assets.deleteAsset(a.id, true);
      } else {
        await assets.deleteAsset(a.id);
      }
    }, '删除资产'));
    card.appendChild(btns);

    card.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('das/asset', a.id);
      e.dataTransfer.effectAllowed = 'copy';
    });
    list.appendChild(card);
  }
  if (!all.length) {
    const empty = document.createElement('div');
    empty.className = 'asset-empty';
    empty.innerHTML = '暂无缺陷资产。<br>点击 <b>抠图模式</b> 或拖入图像开始提取。';
    list.appendChild(empty);
  }
}

function saveVariantFromSelection(asset) {
  const sel = selectedInstances().filter((i) => i.asset_id === asset.id);
  if (!sel.length) { showToast({ text: '请先在画布上选中该资产的实例', kind: 'warn' }); return; }
  const inst = sel[0];
  const v = assets.addVariant(asset.id, {
    appearance: inst.appearance,
    form_seed: inst.mesh ? Math.floor(Math.random() * 0xffffffff) : null,
  });
  showToast({ text: `已为「${asset.name}」添加变体`, kind: 'ok' });
}

export function placeAsset(asset, opts = {}) {
  if (!state.target) { showToast({ text: '请先打开目标图像', kind: 'warn' }); return; }
  window.__studio.placeAsset(asset.id, null, opts);   // null → 智能放置取全局最优平滑区，否则画布中心
}

/* ================= 状态栏 ================= */

function bindStatusbar() {
  $('btnZoom100').onclick = () => {
    if (!state.target) return;
    const v = state.view;
    const cx = ((state._canvasW || 800) / 2 - v.tx) / v.scale;
    const cy = ((state._canvasH || 600) / 2 - v.ty) / v.scale;
    v.scale = 1;
    v.tx = (state._canvasW || 800) / 2 - cx;
    v.ty = (state._canvasH || 600) / 2 - cy;
    emit('render');
  };
  $('btnZoomFit').onclick = () => {
    import('./state.js').then((m) => m.fitView());
  };
}

function updateStatus() {
  const el = $('statusInfo');
  if (!el) return;
  const p = state.cursor.img;
  el.textContent = `${Math.round(state.view.scale * 100)}%  ·  ${p ? `${Math.round(p.x)}, ${Math.round(p.y)}` : '—'}  ·  实例 ${state.instances.length}`;
}

/* ================= 直方图 ================= */

function drawHistogram(d) {
  const cv = $('histCanvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  let max = 1;
  for (const b of d.bins) if (b > max) max = b;
  ctx.fillStyle = '#4ea3ff';
  const bw = cv.width / 256;
  for (let i = 0; i < 256; i++) {
    const h = Math.sqrt(d.bins[i] / max) * (cv.height - 2);
    if (h > 0.5) ctx.fillRect(i * bw, cv.height - h, Math.max(1, bw), h);
  }
}

/* ================= Toast ================= */

function showToast(d) {
  const box = $('toasts');
  if (!box) return;
  const t = document.createElement('div');
  t.className = 'toast ' + (d.kind || 'info');
  t.textContent = d.text;
  box.appendChild(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 300); }, 2600);
}

/* ================= 面板刷新 ================= */

function refreshPanels() {
  const sel = selectedInstances();
  const one = sel.length === 1 ? sel[0] : null;
  // 作用域徽标：右侧所有参数滑杆只对"选中实例"生效
  const badge = $('scopeBadge');
  if (badge) {
    badge.className = 'scope-badge' + (sel.length === 0 ? ' none' : (sel.length > 1 ? ' multi' : ''));
    badge.textContent = sel.length === 0
      ? '未选中实例：下方参数不生效（先点选画布上的缺陷）'
      : (sel.length === 1
        ? `调节对象：1 个实例${one?.name ? '（' + one.name + '）' : ''}`
        : `调节对象：${sel.length} 个选中实例（会同时修改，Ctrl+点击可只选一个）`);
  }
  const setV = (id, v, dis) => {
    const el = $(id);
    if (!el) return;
    el.disabled = !!dis;
    if (document.activeElement !== el && v !== undefined) el.value = v;
  };
  const has = sel.length > 0;
  for (const id of ['tfPosX', 'tfPosY', 'tfScaleX', 'tfScaleY', 'tfRot', 'tfShearX', 'tfShearY']) setV(id, undefined, !has);
  if (one) {
    setV('tfPosX', Math.round(one.position.x));
    setV('tfPosY', Math.round(one.position.y));
    setV('tfScaleX', (one.affine.scale_x ?? 1).toFixed(2));
    setV('tfScaleY', (one.affine.scale_y ?? 1).toFixed(2));
    setV('tfRot', Math.round(one.affine.rotation ?? 0));
    setV('tfShearX', (one.affine.shear_x ?? 0).toFixed(2));
    setV('tfShearY', (one.affine.shear_y ?? 0).toFixed(2));
  }
  // 外观滑条：单选时同步实例值；多选时保持
  const appSrc = one ? normAppearance(one.appearance) : null;
  for (const spec of APPEARANCE_SPEC) {
    const s = $('ap_' + spec.key), n = $('apn_' + spec.key);
    if (!s) continue;
    s.disabled = !has; n.disabled = !has;
    if (appSrc && document.activeElement !== s && document.activeElement !== n) {
      s.value = appSrc[spec.key];
      n.value = spec.step < 1 ? (+appSrc[spec.key]).toFixed(2) : appSrc[spec.key];
    }
  }
  // 融合
  for (const mode of ['direct', 'feather', 'poisson']) {
    const el = $('bl_' + mode);
    if (el) { el.disabled = !has; if (one) el.checked = one.blend === mode; }
  }
  setV('blFeather', one?.feather_radius ?? 3, !has);
  setV('blFeatherN', one?.feather_radius ?? 3, !has);
  setV('blEdgeBlur', one?.edge_blur ?? 0, !has);
  setV('blEdgeBlurN', (one?.edge_blur ?? 0).toFixed(1), !has);
  setV('blOpacity', one?.opacity ?? 1, !has);
  setV('blOpacityN', one?.opacity ?? 1, !has);
  if (one) $('blMixed').checked = one.blend_flag === 'mixed';
  $('blMixed').disabled = !has;
  // 光照协调
  if (one) {
    $('blHarmonize').checked = (one.harmonize || 'off') !== 'off';
    $('blHarmRow').style.display = $('blHarmonize').checked ? '' : 'none';
    $('blHarmRow2').style.display = $('blHarmonize').checked ? '' : 'none';
    if (document.activeElement !== $('blHarmStr')) {
      $('blHarmStr').value = one.harmonize_strength ?? 0.8;
      $('blHarmStrN').value = (one.harmonize_strength ?? 0.8).toFixed(2);
    }
    if (document.activeElement !== $('blHarmRing')) {
      $('blHarmRing').value = one.harmonize_ring ?? 10;
      $('blHarmRingN').value = (one.harmonize_ring ?? 10) + 'px';
    }
  }
  $('blHarmonize').disabled = !has;
  $('blHarmStr').disabled = !has;
  $('blHarmRing').disabled = !has;

  // 编辑模式按钮状态
  updateEditModeBar();
  // 录制按钮
  const rec = $('btnRec');
  if (rec) {
    rec.classList.toggle('recording', state.recipe.recording);
    rec.textContent = state.recipe.recording ? '■ 停止录制' : '● 录制 Recipe';
  }
  updateStatus();
}
