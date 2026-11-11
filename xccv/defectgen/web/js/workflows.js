/* Defect Augment Studio — 批量粘贴到背景池 + 导出（规格 §6.4.3 / §8.5 / §7.2） */
import { state, emit, on } from './state.js';
import * as api from './api.js';
import { materializeSteps } from './recipe.js';
import { deepClone, fmtBytes, pickSaveHandle, saveBlob } from './util.js';

const $ = (id) => document.getElementById(id);

export function initWorkflows() {
  on('export-request', openExportDialog);
  on('batch-request', openBatchDialog);
  on('recipe-dialog', openRecipeDialog);

  $('expDoExport').onclick = doExport;
  $('expClose').onclick = () => $('exportDlg').close();
  $('expFormat').onchange = () => { $('expQualityRow').style.display = $('expFormat').value === 'jpeg' ? '' : 'none'; };

  $('batchSource').onchange = onBatchSourceChange;
  $('batchPickDir').onclick = () => $('batchDirInput').click();
  $('batchPickFiles').onclick = () => $('batchDirInput').click();
  $('batchDirInput').onchange = (e) => {
    const files = [...e.target.files].filter((f) => /\.(png|jpe?g|bmp|tiff?|webp)$/i.test(f.name));
    $('batchDirInfo').textContent = `已选择 ${files.length} 张背景图（上传模式）`;
    $('batchDirInput')._files = files;
  };
  $('batchDoRun').onclick = startBatch;
  $('batchCancelRun').onclick = () => api.batch.cancel(currentJob).catch(() => { });
  $('batchClose').onclick = () => { if (!batchRunning) $('batchDlg').close(); };

  $('rcDoReplay').onclick = async () => {
    const f = $('rcFile').files[0];
    const { loadRecipeFile, replayRecipe } = await import('./recipe.js');
    if (f) await loadRecipeFile(f);
    else {
      const r = state.recipe;
      if (!r.steps.length) { emit('toast', { text: '没有已录制的步骤，也无外部 Recipe 文件', kind: 'warn' }); return; }
      await replayRecipe({ name: '当前录制', steps: r.steps, srcDims: r.srcDims });
    }
    $('recipeDlg').close();
  };
  $('rcClose').onclick = () => $('recipeDlg').close();
  $('rcLoadBtn').onclick = () => $('rcFile').click();
}

/* ================= 导出（全分辨率重算 / 当前视图分辨率） ================= */

let exporting = false;

function openExportDialog() {
  if (!state.target || !state.instances.length) {
    emit('toast', { text: '请先打开目标图像并放置至少一个实例', kind: 'warn' });
    return;
  }
  $('exportDlg').showModal();
}

function buildPayload(k, targetB64, targetW, targetH) {
  const assetsOut = {};
  for (const inst of state.instances) {
    const a = state.assets.get(inst.asset_id);
    if (!a || assetsOut[inst.asset_id]) continue;
    assetsOut[inst.asset_id] = { rgba: a.rgbaBitmap, mask: a.maskBitmap };
  }
  const toB64 = async (bmp) => {
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    c.getContext('2d').drawImage(bmp, 0, 0);
    const blob = await c.convertToBlob({ type: 'image/png' });
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = '';
    for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    return btoa(s);
  };
  return Promise.all(Object.entries(assetsOut).map(async ([id, a]) => [id, {
    rgba_b64: await toB64(a.rgba), mask_b64: await toB64(a.mask),
  }])).then((entries) => ({
    target_b64: targetB64,
    assets: Object.fromEntries(entries),
    instances: state.instances.map((i) => scaleInstance(i, k)),
    format: $('expFormat').value,
    quality: +$('expQuality').value,
    filename: $('expName').value || 'composite',
    _tw: targetW, _th: targetH,
  }));
}

function scaleInstance(inst, k) {
  const s = deepClone(inst);
  delete s.previewPatch; delete s.id; delete s.name;
  s.position = [inst.position.x * k, inst.position.y * k];   // 后端为 [x, y] 数组
  s.affine = { ...inst.affine, scale_x: (inst.affine.scale_x ?? 1) * k, scale_y: (inst.affine.scale_y ?? 1) * k };
  if (s.mesh) s.mesh = { ...s.mesh, grid: s.mesh.grid.map(([dx, dy]) => [dx * k, dy * k]) };
  s.feather_radius = (inst.feather_radius ?? 0) * k;
  s.edge_blur = (inst.edge_blur ?? 0) * k;
  return s;
}

async function canvasToB64(w, h, draw) {
  const c = new OffscreenCanvas(w, h);
  draw(c.getContext('2d'));
  const blob = await c.convertToBlob({ type: 'image/png' });
  const buf = new Uint8Array(await blob.arrayBuffer());
  let s = '';
  for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  return btoa(s);
}

async function doExport() {
  if (exporting) return;
  exporting = true;
  const btn = $('expDoExport');
  btn.disabled = true;
  // 先弹系统保存对话框（自选目录/文件名）；取消则中止
  const fmtSel = $('expFormat').value;
  const extGuess = fmtSel === 'jpeg' ? '.jpg' : (fmtSel === 'tiff' ? '.tif' : '.png');
  const mimeGuess = fmtSel === 'jpeg' ? 'image/jpeg' : (fmtSel === 'tiff' ? 'image/tiff' : 'image/png');
  const suggested = ($('expName').value || 'composite') + extGuess;
  const saveHandle = await pickSaveHandle(suggested, mimeGuess);
  if (saveHandle === 'cancel') {
    $('expStatus').textContent = '已取消导出';
    exporting = false; btn.disabled = false;
    return;
  }
  $('expStatus').textContent = '正在合成…';
  try {
    let targetB64, tw, th;
    if ($('expRes').value === 'full' && state.target.originalBlob) {
      // 原分辨率重算（规格 §7.2）：实例几何按 k 映射回原图坐标系
      const bmp = await createImageBitmap(state.target.originalBlob);
      tw = bmp.width; th = bmp.height;
      targetB64 = await canvasToB64(tw, th, (c) => c.drawImage(bmp, 0, 0));
      bmp.close();
    } else {
      tw = state.target.w; th = state.target.h;
      targetB64 = await canvasToB64(tw, th, (c) => c.drawImage(state.target.bitmap, 0, 0));
    }
    const k = tw / state.target.w;
    const payload = await buildPayload(k, targetB64, tw, th);
    delete payload._tw; delete payload._th;
    const res = await api.composite(payload);
    const how = await saveBlob(res.blob, res.name, saveHandle);
    $('expStatus').textContent = `${how === 'saved' ? '已保存' : '已导出'} ${res.name}（${fmtBytes(res.blob.size)}）`
      + (res.notes.length ? '；降级说明：' + res.notes.join('；') : '');
    if (res.notes.length) emit('toast', { text: '导出完成（含降级项，详见导出面板）', kind: 'warn' });
    else emit('toast', { text: `导出完成：${res.name}`, kind: 'ok' });
  } catch (e) {
    $('expStatus').textContent = '导出失败：' + e.message;
    emit('toast', { text: '导出失败：' + e.message, kind: 'err' });
  } finally {
    exporting = false;
    btn.disabled = false;
  }
}

/* ================= 背景池批量 ================= */

let currentJob = null;
let batchRunning = false;
let pollTimer = null;

function openBatchDialog() {
  if (!state.target || !state.instances.length) {
    emit('toast', { text: '请先在画布上完成一次缺陷放置（批量将复用当前场景或 Recipe）', kind: 'warn' });
    return;
  }
  $('batchDlg').showModal();
  $('batchProgressRow').style.display = 'none';
  $('batchReport').innerHTML = '';
}

function onBatchSourceChange() {
  const v = $('batchSource').value;
  $('batchDirRow').style.display = v === 'dir' ? '' : 'none';
  $('batchUploadRow').style.display = v === 'files' ? '' : 'none';
}

async function startBatch() {
  if (batchRunning) return;
  const src = state.recipe.steps.length && $('batchUseRecipe').checked
    ? 'recipe' : 'canvas';
  // 组装 assets + 基础实例（相对位置 + 短边缩放锚点）
  let baseInstances;
  if (src === 'recipe') {
    baseInstances = materializeSteps(state.recipe.steps, { w: 1, h: 1 }, state.recipe.srcDims)
      .map((i) => ({
        ...i,
        position: [i.position.x, i.position.y],
        rel_position: [i.position.x, i.position.y],
        scale_anchor: Math.min(state.recipe.srcDims.w, state.recipe.srcDims.h),
      }));
  } else {
    const W = state.target.w, H = state.target.h;
    const short = Math.min(W, H);
    baseInstances = state.instances.map((i) => ({
      ...deepClone(i),
      position: [i.position.x, i.position.y],
      rel_position: [i.position.x / W, i.position.y / H],
      scale_anchor: short,
    }));
  }
  for (const b of baseInstances) { delete b.previewPatch; delete b.id; delete b.name; }

  const payload = {
    assets: {},
    instances: baseInstances,
    background: {},
    output_dir: $('batchOutDir').value.trim() || 'output',
    format: $('batchFormat').value,
    quality: 95,
    jitter: $('batchJitter').checked ? {
      pos_pct: +$('batchJitPos').value,
      rot_deg: +$('batchJitRot').value,
      brightness: +$('batchJitBright').value,
    } : null,
    seed: +$('batchSeed').value || 42,
  };
  // 资产编码
  const toB64 = async (bmp) => {
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    c.getContext('2d').drawImage(bmp, 0, 0);
    const blob = await c.convertToBlob({ type: 'image/png' });
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = '';
    for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    return btoa(s);
  };
  try {
    const ids = new Set(baseInstances.map((i) => i.asset_id));
    for (const id of ids) {
      const a = state.assets.get(id);
      if (!a) throw new Error('资产缺失：' + id);
      payload.assets[id] = { rgba_b64: await toB64(a.rgbaBitmap), mask_b64: await toB64(a.maskBitmap) };
    }
    if ($('batchSource').value === 'dir') {
      const files = $('batchDirInput')._files;
      if (files?.length) {
        payload.background = {
          mode: 'files',
          files: await Promise.all(files.map(async (f) => ({
            name: f.name, b64: await new Promise((res) => {
              const r = new FileReader();
              r.onload = () => res(r.result.split(',')[1]);
              r.readAsDataURL(f);
            }),
          }))),
        };
      } else {
        const dir = $('batchDirPath').value.trim();
        if (!dir) { emit('toast', { text: '请填写服务器背景目录路径，或选择上传文件', kind: 'warn' }); return; }
        payload.background = { mode: 'dir', dir };
      }
    } else {
      const files = $('batchDirInput')._files || [];
      if (!files.length) { emit('toast', { text: '请选择背景图像文件', kind: 'warn' }); return; }
      payload.background = {
        mode: 'files',
        files: await Promise.all(files.map(async (f) => ({
          name: f.name, b64: await new Promise((res) => {
            const r = new FileReader();
            r.onload = () => res(r.result.split(',')[1]);
            r.readAsDataURL(f);
          }),
        }))),
      };
    }
    const { job_id } = await api.batch.start(payload);
    currentJob = job_id;
    batchRunning = true;
    $('batchProgressRow').style.display = '';
    $('batchDoRun').disabled = true;
    $('batchReport').innerHTML = '';
    poll();
  } catch (e) {
    emit('toast', { text: '批量任务启动失败：' + e.message, kind: 'err' });
  }
}

async function poll() {
  try {
    const st = await api.batch.status(currentJob);
    $('batchProgressBar').style.width = (st.total ? (st.done_count / st.total) * 100 : 0) + '%';
    $('batchProgressText').textContent = `${st.done_count}/${st.total}`;
    if (st.done) {
      batchRunning = false;
      $('batchDoRun').disabled = false;
      renderReport(st);
      if (st.error) emit('toast', { text: '批量任务出错：' + st.error, kind: 'err' });
      else if (st.cancelled) emit('toast', { text: '批量任务已取消', kind: 'warn' });
      else emit('toast', {
        text: `批量完成：成功 ${st.summary.ok}，失败 ${st.summary.failed}，跳过 ${st.summary.skipped}（${(st.summary.total_ms / 1000).toFixed(1)}s）`,
        kind: st.summary.failed ? 'warn' : 'ok',
      });
      return;
    }
  } catch (e) {
    emit('toast', { text: '进度查询失败：' + e.message, kind: 'err' });
  }
  pollTimer = setTimeout(poll, 800);
}

function renderReport(st) {
  const el = $('batchReport');
  const rows = st.results.map((r) => {
    const cls = r.status === 'ok' ? 'ok' : r.status === 'skipped' ? 'skip' : 'fail';
    return `<tr class="${cls}"><td>${r.name}</td><td>${r.status === 'ok' ? '成功' : r.status === 'skipped' ? '跳过' : '失败'}</td><td>${r.note || ''}</td><td>${r.ms}ms</td></tr>`;
  }).join('');
  el.innerHTML = `
    <div class="report-summary">成功 <b>${st.summary?.ok ?? 0}</b> · 失败 <b>${st.summary?.failed ?? 0}</b> · 跳过 <b>${st.summary?.skipped ?? 0}</b> · 耗时 ${((st.summary?.total_ms ?? 0) / 1000).toFixed(1)}s</div>
    <table class="report-table"><thead><tr><th>背景图</th><th>状态</th><th>说明</th><th>耗时</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/* ================= Recipe 对话框 ================= */

function openRecipeDialog() {
  $('recipeDlg').showModal();
  const r = state.recipe;
  $('rcInfo').textContent = r.recording
    ? `录制中：已记录 ${r.steps.length} 步。停止录制后可保存。`
    : (r.steps.length ? `当前已录制 ${r.steps.length} 步（源图 ${r.srcDims?.w}×${r.srcDims?.h}），可直接回放或加载外部 Recipe JSON。`
      : '尚未录制。点击工具条「● 录制 Recipe」开始，或加载外部 .recipe.json 文件回放。');
}
