/* Defect Augment Studio — 渲染 Worker（规格 §7.1 非协商规则）
 *
 * 画布绘制全部在本 Worker 的 OffscreenCanvas 上进行，主线程只发送命令与帧状态，
 * Worker 不持有业务状态。分层：
 *   sceneCanvas —— 图像(代理)分辨率场景缓冲：背景 + 全部实例（仅在场景变化时重绘）
 *   屏幕绘制    —— 视图变换 blit + 屏幕空间叠加层（选择柄/参考线/分屏/放大镜）
 * 平移缩放只做一次 blit，与实例数量无关；实例编辑才触发场景缓冲重绘。
 */
import {
  instPoint, affine2x3, apply2x3, clamp, encodeBmpBlob,
} from './util.js';

let canvas = null, ctx = null, dpr = 1, vw = 0, vh = 0;
let frame = null;
const images = { target: null, source: null, heatmap: null };   // {bitmap, w, h}
const assets = new Map();                        // id -> {rgba, mask, alpha, w, h, rev, noise:Map, tint}
let extractMask = null;                          // OffscreenCanvas（抠图/绘制模式掩码叠加）
let extractMaskRev = 0;
let extractMaskMode = 'extract';
let paintColorLayer = null;                      // 绘制模式每笔各自颜色的预览层
let paintColorRev = 0;
let paintMorphLayer = null;                      // 真实形貌预览层（与烘焙同管线）
let paintMorphRev = 0;
let editPreviewLayer = null;   // 整场景融合预览（与导出一致）
let editPreviewRev = -1;
let paintMorphX = 0, paintMorphY = 0;
let sceneCanvas = null, sceneCtx = null;
let sceneKey = '';
let dirty = false, scheduled = false;

/* ---------------- 消息入口 ---------------- */

self.onmessage = (e) => {
  const m = e.data;
  switch (m.t) {
    case 'init':
      canvas = m.canvas; ctx = canvas.getContext('2d'); dpr = m.dpr || 1;
      break;
    case 'resize':
      vw = m.w; vh = m.h; dpr = m.dpr || 1;
      if (canvas) { canvas.width = Math.max(1, Math.round(vw * dpr)); canvas.height = Math.max(1, Math.round(vh * dpr)); }
      markDirty();
      break;
    case 'asset': {
      const old = assets.get(m.id);
      if (old) { try { old.rgba.close(); } catch { } try { old.mask.close(); } catch { } }
      assets.set(m.id, { rgba: m.rgba, mask: m.mask, w: m.w, h: m.h, rev: m.rev, alpha: null, noise: new Map() });
      sceneKey = ''; // 资产更新 → 场景缓冲失效
      markDirty();
      break;
    }
    case 'img':
      images[m.slot] = { bitmap: m.bitmap, w: m.w, h: m.h };
      sceneKey = '';
      markDirty();
      break;
    case 'mask': {
      if (m.mode === 'paint-color') {
        if (!paintColorLayer || paintColorLayer.width !== m.image.width || paintColorLayer.height !== m.image.height) {
          paintColorLayer = new OffscreenCanvas(m.image.width, m.image.height);
        }
        paintColorLayer.getContext('2d').putImageData(m.image, 0, 0);
        paintColorRev++;
      } else if (m.mode === 'edit-preview') {
        if (!m.image) {
          editPreviewLayer = null;
          editPreviewRev = m.rev || -1;
        } else {
          if (!editPreviewLayer || editPreviewLayer.width !== m.image.width
              || editPreviewLayer.height !== m.image.height) {
            editPreviewLayer = new OffscreenCanvas(m.image.width, m.image.height);
          }
          editPreviewLayer.getContext('2d').putImageData(m.image, 0, 0);
          editPreviewRev = m.rev || 0;
        }
      } else if (m.mode === 'paint-morph') {
        // 真实形貌预览层（与烘焙同管线）：绘制时所见即所得
        if (!m.image) {
          paintMorphLayer = null;
          paintMorphRev = m.rev || 0;
        } else {
          if (!paintMorphLayer || paintMorphLayer.width !== m.image.width || paintMorphLayer.height !== m.image.height) {
            paintMorphLayer = new OffscreenCanvas(m.image.width, m.image.height);
          }
          paintMorphLayer.getContext('2d').putImageData(m.image, 0, 0);
          paintMorphX = m.x || 0;
          paintMorphY = m.y || 0;
          paintMorphRev = m.rev || 0;
        }
      } else {
        if (!extractMask || extractMask.width !== m.image.width || extractMask.height !== m.image.height) {
          extractMask = new OffscreenCanvas(m.image.width, m.image.height);
        }
        extractMask.getContext('2d').putImageData(m.image, 0, 0);
        extractMaskRev++;
        extractMaskMode = m.mode || 'extract';
      }
      sceneKey = '';
      markDirty();
      break;
    }
    case 'frame':
      frame = m.frame;
      markDirty();
      break;
    case 'disposeInstance': {
      /* 池化画布随 Map 置空由 GC 回收 */
      break;
    }
    case 'requestLayer':
      renderLayerRequest(m);
      break;
    case 'capture': {
      // 调试/测试用：回传当前画布像素（绕过合成器与位图通道）。
      // 纯 JS BMP 编码：convertToBlob 偶发永久挂起
      if (canvas) {
        try {
          const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const blob = encodeBmpBlob(img.data, img.width, img.height);
          self.postMessage({ t: 'capture', reqId: m.reqId, blob, w: vw, h: vh });
        } catch (e) {
          self.postMessage({ t: 'capture', reqId: m.reqId, error: String(e) });
        }
      }
      break;
    }
  }
};

function markDirty() {
  dirty = true;
  if (scheduled) return;
  scheduled = true;
  setTimeout(() => {
    scheduled = false;
    if (dirty) {
      dirty = false;
      try { draw(); } catch (e) {
        self.postMessage({ t: 'workerError', msg: String(e && e.stack || e).slice(0, 500) });
      }
      maybeComputeHistogram();
    }
  }, 8);
}

/* ---------------- 主绘制 ---------------- */

function draw() {
  if (!ctx || !frame) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, vw, vh);
  const bg = ctx.createLinearGradient(0, 0, 0, vh);
  bg.addColorStop(0, '#141518'); bg.addColorStop(1, '#101114');
  ctx.fillStyle = bg; ctx.fillRect(0, 0, vw, vh);

  const isExtract = frame.mode === 'extract';
  const bgImg = isExtract ? images.source : images.target;
  if (!frame.target || !bgImg) return;

  ensureScene(frame.target.w, frame.target.h);
  const key = sceneHash(isExtract);
  if (key !== sceneKey) {
    renderScene(isExtract, bgImg);
    sceneKey = key;
    scheduleHistogram();       // 场景变更后重算直方图（节流）
  }

  // 视图 blit：放大超过 1:1 时用最近邻，呈现像素级锐利视图
  const v = frame.view;
  ctx.save();
  ctx.translate(v.tx, v.ty);
  ctx.scale(v.scale, v.scale);
  ctx.imageSmoothingEnabled = v.scale < 0.99;
  ctx.drawImage(sceneCanvas, 0, 0);
  ctx.restore();

  drawHeatmap();
  drawOverlays(isExtract, bgImg);
}

/* 放置适宜度热力图叠加（jet 色图，35% 透明度） */
function drawHeatmap() {
  if (!frame.ui.heatmap || !images.heatmap) return;
  const v = frame.view;
  ctx.save();
  ctx.globalAlpha = 0.38;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(images.heatmap.bitmap, v.tx, v.ty,
    frame.target.w * v.scale, frame.target.h * v.scale);
  ctx.restore();
}

/* 直方图（场景变更后 600ms 节流，隔行采样亮度） */
let histDirty = false, lastHistAt = 0;
function scheduleHistogram() { histDirty = true; }
function maybeComputeHistogram() {
  if (!histDirty || !sceneCanvas) return;
  const now = Date.now();
  if (now - lastHistAt < 600) return;
  histDirty = false;
  lastHistAt = now;
  const w = Math.min(sceneCanvas.width, frame.target.w);
  const h = Math.min(sceneCanvas.height, frame.target.h);
  const step = Math.max(2, Math.round(Math.max(w, h) / 512));
  try {
    const d = sceneCtx.getImageData(0, 0, w, h).data;
    const bins = new Array(256).fill(0);
    let total = 0;
    for (let y = 0; y < h; y += step) {
      let i = y * w * 4;
      for (let x = 0; x < w; x += step, i += 4 * step) {
        bins[Math.round(0.114 * d[i] + 0.587 * d[i + 1] + 0.299 * d[i + 2]) & 255]++;
        total++;
      }
    }
    self.postMessage({ t: 'histogram', bins, total });
  } catch { /* 画布不可读时忽略 */ }
}

function sceneHash(isExtract) {
  const f = frame;
  const code = isExtract ? 'x' : (f.mode === 'paint' ? 'p' : 'e');
  let h = code + '|' + f.sceneRev + '|' + (f.ui.boundaryHighlight ? 'b' : '')
    + (f.ui.showOriginalOnly ? 'o' : '');
  for (const i of f.instances) {
    h += i.id + ':' + i.sig + ';' + (i.previewPatch ? i.previewPatch.rev : 0) + '|';
  }
  if (isExtract && f.extract) h += f.extract.maskRev + '|' + (f.extract.maskOn !== false);
  if (f.mode === 'edit') h += '|ep' + (f.editPreviewRev ?? -1) + '|' + (editPreviewLayer ? 1 : 0);
  if (f.mode === 'paint' && f.paint) h += f.paint.maskRev + '|' + paintColorRev + '|' + paintMorphRev;
  return h;
}

function ensureScene(w, h) {
  if (!sceneCanvas || sceneCanvas.width !== w || sceneCanvas.height !== h) {
    sceneCanvas = new OffscreenCanvas(w, h);
    sceneCtx = sceneCanvas.getContext('2d');
    sceneKey = '';
  }
}

function renderScene(isExtract, bgImg) {
  const f = frame;
  const isPaint = f.mode === 'paint';
  sceneCtx.setTransform(1, 0, 0, 1, 0, 0);
  sceneCtx.clearRect(0, 0, sceneCanvas.width, sceneCanvas.height);
  sceneCtx.imageSmoothingEnabled = true;
  sceneCtx.imageSmoothingQuality = 'high';
  sceneCtx.drawImage(bgImg.bitmap, 0, 0);

  if (isExtract || isPaint) {
    if (isPaint && !f.ui.showOriginalOnly) {
      // 绘制模式：优先显示真实形貌预览层（与烘焙同管线，所见即所得）；
      // 落笔期间预览被清除 → 回退到每笔各自颜色的叠加层 / 单色 tint
      if (paintMorphLayer && f.paint && f.paint.previewRev === paintMorphRev) {
        sceneCtx.drawImage(paintMorphLayer, paintMorphX, paintMorphY);
      } else if (paintColorLayer) {
        sceneCtx.save();
        sceneCtx.globalAlpha = 0.85;
        sceneCtx.drawImage(paintColorLayer, 0, 0);
        sceneCtx.restore();
      } else if (extractMask) {
        const key = 'paint' + extractMaskRev;
        const tc = f.paint?.tint || [120, 255, 90];
        const tint = getTint(key, extractMask, tc[0], tc[1], tc[2]);
        sceneCtx.drawImage(tint.canvas, 0, 0);
      }
    } else if (isExtract && extractMask && f.extract?.maskOn !== false) {
      const key = 'extract' + extractMaskRev;
      const tint = getTint(key, extractMask, 255, 70, 70);
      sceneCtx.drawImage(tint.canvas, 0, 0);
    }
    // 已选缺陷高亮：虚线框（提示"当前滑杆只作用于它"）
    if (isPaint && f.paint && f.paint.selBox && !f.ui.showOriginalOnly) {
      const b = f.paint.selBox;
      sceneCtx.save();
      sceneCtx.lineWidth = Math.max(1.5, 2 / (f.view.scale || 1));
      sceneCtx.setLineDash([8 / (f.view.scale || 1), 6 / (f.view.scale || 1)]);
      sceneCtx.strokeStyle = 'rgba(90, 230, 160, 0.95)';
      sceneCtx.strokeRect(b.x, b.y, b.w, b.h);
      sceneCtx.restore();
    }
    return;
  }
  if (!f.ui.showOriginalOnly) {
    const pvState = f.editPreviewActive || 'off';
    if (editPreviewLayer && f.editPreviewRev === editPreviewRev && pvState === 'on') {
      // 整场景预览：后端 render_scene 的结果（含全部实例），与导出逐像素一致
      sceneCtx.drawImage(editPreviewLayer, 0, 0);
      if (f.ui.boundaryHighlight) {
        for (const inst of f.instances) {
          const a = assets.get(inst.asset_id);
          if (!a) continue;
          inst._quad = [instPoint(inst, { x: 0, y: 0 }, a.w, a.h), instPoint(inst, { x: a.w, y: 0 }, a.w, a.h),
            instPoint(inst, { x: a.w, y: a.h }, a.w, a.h), instPoint(inst, { x: 0, y: a.h }, a.w, a.h)];
        }
      }
    } else if (pvState === 'pending') {
      // 预览计算中：非 Poisson 实例正常绘制；Poisson 实例以半透明占位（明确提示
      // "尚未最终"，避免用户把近似渲染当作导出的真实结果）
      for (const inst of f.instances) {
        if (inst.blend !== 'poisson') { drawInstance(inst); continue; }
        sceneCtx.save();
        sceneCtx.globalAlpha = 0.4;
        drawInstance(inst);
        sceneCtx.restore();
      }
    } else {
      for (const inst of f.instances) drawInstance(inst);
    }
  }
}

/* ---------------- 实例绘制（几何链：mesh → H → affine 的统一三角网格纹理映射） ---------------- */

const pools = new Map(); // instId -> {temp, work, feath, sil}
function getPool(id) {
  let p = pools.get(id);
  if (!p) { p = {}; pools.set(id, p); }
  return p;
}
function sized(c, key, w, h) {
  if (!c[key] || c[key].width !== w || c[key].height !== h) c[key] = new OffscreenCanvas(w, h);
  return c[key];
}

function instGridDef(inst) {
  if (inst.mesh) return { cols: inst.mesh.cols, rows: inst.mesh.rows };
  if (inst.perspective_h) return { cols: 12, rows: 12 };
  return { cols: 1, rows: 1 };
}

function drawInstance(inst) {
  const asset = assets.get(inst.asset_id);
  if (!asset) return;
  const { w, h } = asset;

  if (inst.previewPatch && inst.blend === 'poisson') {
    sceneCtx.drawImage(inst.previewPatch.bitmap, inst.previewPatch.x, inst.previewPatch.y);
    if (frame.ui.boundaryHighlight) {
      inst._quad = [instPoint(inst, { x: 0, y: 0 }, w, h), instPoint(inst, { x: w, y: 0 }, w, h),
        instPoint(inst, { x: w, y: h }, w, h), instPoint(inst, { x: 0, y: h }, w, h)];
    }
    return;
  }

  // 网格顶点：src（资产像素坐标）→ dst（目标图像坐标）
  const gd = instGridDef(inst);
  const src = [], dst = [];
  for (let j = 0; j <= gd.rows; j++)
    for (let i = 0; i <= gd.cols; i++) {
      const s = { x: i * w / gd.cols, y: j * h / gd.rows };
      src.push(s);
      dst.push(instPoint(inst, s, w, h));
    }

  // 目标包围盒
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const p of dst) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  const margin = 4 + (inst.blend === 'feather' ? inst.feather_radius * 2 : 0)
    + (inst.appearance?.blur ? inst.appearance.blur * 3 : 0)
    + (inst.edge_blur > 0 ? inst.edge_blur * 3 : 0);
  x0 = Math.floor(x0 - margin); y0 = Math.floor(y0 - margin);
  x1 = Math.ceil(x1 + margin); y1 = Math.ceil(y1 + margin);
  const bw = x1 - x0, bh = y1 - y0;
  if (bw <= 0 || bh <= 0) return;

  const pool = getPool(inst.id);
  const temp = sized(pool, 'temp', bw, bh);
  const tctx = temp.getContext('2d');
  tctx.setTransform(1, 0, 0, 1, 0, 0);
  tctx.clearRect(0, 0, bw, bh);
  tctx.imageSmoothingEnabled = true;

  // 三角网格纹理映射（外扩 0.5px 消缝）
  const cellTris = [];
  for (let j = 0; j < gd.rows; j++)
    for (let i = 0; i < gd.cols; i++) {
      const a = j * (gd.cols + 1) + i, b = a + 1, c = a + gd.cols + 1, d = c + 1;
      cellTris.push([a, b, d], [a, d, c]);
    }
  for (const [ia, ib, ic] of cellTris) {
    drawTexturedTri(tctx, asset.rgba, src[ia], src[ib], src[ic],
      { x: dst[ia].x - x0, y: dst[ia].y - y0 },
      { x: dst[ib].x - x0, y: dst[ib].y - y0 },
      { x: dst[ic].x - x0, y: dst[ic].y - y0 });
  }

  const app = inst.appearance || {};

  // 噪声（局部空间纹理，soft-light 近似加性高斯噪声；导出走后端精确实现）
  if (app.noise > 0) {
    const nc = getNoiseCanvas(asset, inst.noise_seed, app.noise);
    if (nc) {
      const work = sized(pool, 'work', bw, bh);
      const wctx = work.getContext('2d');
      wctx.setTransform(1, 0, 0, 1, 0, 0);
      wctx.clearRect(0, 0, bw, bh);
      for (const [ia, ib, ic] of cellTris) {
        drawTexturedTri(wctx, nc, src[ia], src[ib], src[ic],
          { x: dst[ia].x - x0, y: dst[ia].y - y0 },
          { x: dst[ib].x - x0, y: dst[ib].y - y0 },
          { x: dst[ic].x - x0, y: dst[ic].y - y0 });
      }
      wctx.globalCompositeOperation = 'destination-in';
      wctx.drawImage(temp, 0, 0);
      wctx.globalCompositeOperation = 'source-over';
      tctx.globalCompositeOperation = 'soft-light';
      tctx.globalAlpha = Math.min(1, app.noise / 10);
      tctx.drawImage(work, 0, 0);
      tctx.globalAlpha = 1;
      tctx.globalCompositeOperation = 'source-over';
    }
  }

  // 亮度/对比度：LUT 逐像素（与后端语义一致）；色相/饱和度走 GPU filter
  if (Math.abs(app.brightness) > 1e-6 || Math.abs(app.contrast - 1) > 1e-6) {
    applyLut(tctx, temp, app.brightness || 0, app.contrast || 1);
  }

  // 模糊：颜色模糊但保持轮廓 alpha（外观仅作用于 mask 内，规格 §6.3）
  let result = temp;
  if (app.blur > 0) {
    const work = sized(pool, 'work', bw, bh);
    const wctx = work.getContext('2d');
    wctx.setTransform(1, 0, 0, 1, 0, 0);
    wctx.clearRect(0, 0, bw, bh);
    wctx.filter = `blur(${app.blur}px)`;
    wctx.drawImage(temp, 0, 0);
    wctx.filter = 'none';
    wctx.globalCompositeOperation = 'destination-in';
    wctx.drawImage(temp, 0, 0);
    wctx.globalCompositeOperation = 'source-over';
    result = work;
  }

  // Feather 融合：alpha 高斯模糊生成渐变边缘（在模糊结果之上再乘 feather alpha）
  if (inst.blend === 'feather' && inst.feather_radius > 0) {
    if (!asset.alpha) buildAlphaCanvas(asset);
    const feath = sized(pool, 'feath', bw, bh);
    const fctx = feath.getContext('2d');
    fctx.setTransform(1, 0, 0, 1, 0, 0);
    fctx.clearRect(0, 0, bw, bh);
    fctx.filter = `blur(${inst.feather_radius}px)`;
    for (const [ia, ib, ic] of cellTris) {
      drawTexturedTri(fctx, asset.alpha, src[ia], src[ib], src[ic],
        { x: dst[ia].x - x0, y: dst[ia].y - y0 },
        { x: dst[ib].x - x0, y: dst[ib].y - y0 },
        { x: dst[ic].x - x0, y: dst[ic].y - y0 });
    }
    fctx.filter = 'none';
    const out = sized(pool, 'out', bw, bh);
    const octx = out.getContext('2d');
    octx.setTransform(1, 0, 0, 1, 0, 0);
    octx.clearRect(0, 0, bw, bh);
    octx.drawImage(result, 0, 0);
    octx.globalCompositeOperation = 'destination-in';
    octx.drawImage(feath, 0, 0);
    octx.globalCompositeOperation = 'source-over';
    result = out;
  }

  // 边缘模糊：RGBA（含 alpha）整体软化，缺陷边缘的颜色与轮廓同步过渡进背景。
  // 与外观模糊的区别：不夹紧轮廓（无 destination-in），允许 alpha 向外渐隐。
  if (inst.edge_blur > 0) {
    const eout = sized(pool, 'eblur', bw, bh);
    const ectx = eout.getContext('2d');
    ectx.setTransform(1, 0, 0, 1, 0, 0);
    ectx.clearRect(0, 0, bw, bh);
    ectx.filter = `blur(${Math.min(20, inst.edge_blur)}px)`;
    ectx.drawImage(result, 0, 0);
    ectx.filter = 'none';
    result = eout;
  }

  // 光照协调预览：对齐实例周边环带背景的颜色统计（与后端 harmonize_layer 同语义）
  if (inst.harmonize && inst.harmonize !== 'off' && (inst.harmonize_strength ?? 0.8) > 1e-3) {
    harmonizePreview(sceneCtx, result, x0, y0, bw, bh, inst);
  }

  // 合成到场景：色相/饱和度 GPU filter + 不透明度
  const filters = [];
  if (app.hue) filters.push(`hue-rotate(${app.hue}deg)`);
  if (Math.abs((app.saturation ?? 1) - 1) > 1e-6) filters.push(`saturate(${app.saturation})`);
  sceneCtx.save();
  if (filters.length) sceneCtx.filter = filters.join(' ');
  sceneCtx.globalAlpha = clamp(inst.opacity ?? 1, 0, 1);
  sceneCtx.drawImage(result, x0, y0);
  sceneCtx.restore();

  if (frame.ui.boundaryHighlight) {
    inst._bx0 = x0; inst._by0 = y0;
    inst._quad = [dst[0], dst[gd.cols], dst[dst.length - 1], dst[dst.length - gd.cols - 1]];
    drawBoundary(inst, w, h, result);
  }
}

function drawBoundary(inst, w, h, result) {
  // 边界高亮：沿缺陷轮廓的红色描边 + 暗色衬底（红色系缺陷上也可见）。
  // 环 = 轮廓 α − 腐蚀α；腐蚀用 destination-in 交集实现（destination-out 平移法
  // 对凸形会整体删空，不可用）。不遮挡缺陷本体。
  if (!result) return;
  const pool = getPool(inst.id);
  const W = result.width, H = result.height;
  const erode = (key, px) => {
    const e = sized(pool, key, W, H);
    const ec = e.getContext('2d');
    ec.setTransform(1, 0, 0, 1, 0, 0);
    ec.globalCompositeOperation = 'source-over';
    ec.clearRect(0, 0, W, H);
    ec.drawImage(result, 0, 0);
    ec.globalCompositeOperation = 'destination-in';
    for (let d = 1; d <= px; d++) {
      ec.drawImage(result, d, 0); ec.drawImage(result, -d, 0);
      ec.drawImage(result, 0, d); ec.drawImage(result, 0, -d);
    }
    ec.globalCompositeOperation = 'source-over';
    return e;
  };
  const band = (key, eroKey, px, fill) => {
    const ero = erode(eroKey, px);
    const ring = sized(pool, key, W, H);
    const rc = ring.getContext('2d');
    rc.setTransform(1, 0, 0, 1, 0, 0);
    rc.globalCompositeOperation = 'source-over';
    rc.clearRect(0, 0, W, H);
    rc.drawImage(result, 0, 0);
    rc.globalCompositeOperation = 'destination-out';
    rc.drawImage(ero, 0, 0);
    rc.globalCompositeOperation = 'source-in';
    rc.fillStyle = fill;
    rc.fillRect(0, 0, W, H);
    rc.globalCompositeOperation = 'source-over';
    return ring;
  };
  const dark = band('sil2', 'ero2', 4, 'rgba(20,20,20,0.9)');
  const red = band('sil', 'ero1', 2, 'rgba(255,64,64,0.95)');
  sceneCtx.save();
  // 描边画两遍：软边（羽化/软化轮廓）下叠加提升可见度
  sceneCtx.globalAlpha = 0.75;
  sceneCtx.drawImage(dark, inst._bx0 ?? 0, inst._by0 ?? 0);
  sceneCtx.drawImage(dark, inst._bx0 ?? 0, inst._by0 ?? 0);
  sceneCtx.globalAlpha = 0.95;
  sceneCtx.drawImage(red, inst._bx0 ?? 0, inst._by0 ?? 0);
  sceneCtx.drawImage(red, inst._bx0 ?? 0, inst._by0 ?? 0);
  sceneCtx.restore();
}

/* 光照协调预览：环带背景统计（实例绘制前的场景像素）→ 逐通道 LUT 迁移 */
function harmonizePreview(sctx, result, x0, y0, bw, bh, inst) {
  const ring = Math.max(2, Math.round(inst.harmonize_ring || 10));
  const strength = Math.min(1, Math.max(0, inst.harmonize_strength ?? 0.8));
  const rx0 = Math.max(0, x0 - ring), ry0 = Math.max(0, y0 - ring);
  const rx1 = Math.min(sceneCanvas.width, x0 + bw + ring);
  const ry1 = Math.min(sceneCanvas.height, y0 + bh + ring);
  const rw = rx1 - rx0, rh = ry1 - ry0;
  if (rw < 4 || rh < 4) return;
  let ringData;
  try { ringData = sctx.getImageData(rx0, ry0, rw, rh); } catch { return; }
  const rd = ringData.data;
  const cx0 = x0 - rx0, cy0 = y0 - ry0, cx1 = cx0 + bw, cy1 = cy0 + bh;
  const sum = [0, 0, 0], sq = [0, 0, 0];
  let n = 0;
  for (let y = 0; y < rh; y += 2) {
    const inCy = y >= cy0 && y < cy1;
    for (let x = 0; x < rw; x += 2) {
      if (inCy && x >= cx0 && x < cx1) continue;   // 排除缺陷区
      const i = (y * rw + x) * 4;
      for (let c = 0; c < 3; c++) { const v = rd[i + c]; sum[c] += v; sq[c] += v * v; }
      n++;
    }
  }
  if (n < 50) return;
  const rctx = result.getContext('2d');
  let tData;
  try { tData = rctx.getImageData(0, 0, result.width, result.height); } catch { return; }
  const td = tData.data, tw = result.width;
  const sSum = [0, 0, 0], sSq = [0, 0, 0];
  let sn = 0;
  for (let y = 0; y < bh; y += 2) {
    for (let x = 0; x < bw; x += 2) {
      const i = (y * tw + x) * 4;
      if (td[i + 3] > 38) {
        for (let c = 0; c < 3; c++) { const v = td[i + c]; sSum[c] += v; sSq[c] += v * v; }
        sn++;
      }
    }
  }
  if (sn < 20) return;
  const luts = [];
  for (let c = 0; c < 3; c++) {
    const muS = sSum[c] / sn, sdS = Math.sqrt(Math.max(0, sSq[c] / sn - muS * muS)) + 1e-6;
    const muT = sum[c] / n, sdT = Math.sqrt(Math.max(0, sq[c] / n - muT * muT)) + 1e-6;
    const k = 1 + (sdT / sdS - 1) * strength;
    const mu = muS + (muT - muS) * strength;
    const lut = new Uint8ClampedArray(256);
    for (let v = 0; v < 256; v++) lut[v] = Math.max(0, Math.min(255, (v - muS) * k + mu));
    luts.push(lut);
  }
  for (let i = 0; i < td.length; i += 4) {
    if (td[i + 3] === 0) continue;
    td[i] = luts[0][td[i]]; td[i + 1] = luts[1][td[i + 1]]; td[i + 2] = luts[2][td[i + 2]];
  }
  rctx.putImageData(tData, 0, 0);
}

/* 三角形纹理映射：clip + 仿射变换 + 局部 drawImage（源矩形限制绘制区域） */
function drawTexturedTri(c, img, s0, s1, s2, d0, d1, d2) {
  const denom = (s1.x - s0.x) * (s2.y - s0.y) - (s2.x - s0.x) * (s1.y - s0.y);
  if (!denom) return;
  // 目标三角形外扩 0.5px 防相邻三角形接缝
  const cx = (d0.x + d1.x + d2.x) / 3, cy = (d0.y + d1.y + d2.y) / 3;
  const ex = (p) => {
    const dx = p.x - cx, dy = p.y - cy, l = Math.hypot(dx, dy) || 1;
    return { x: p.x + dx / l * 0.5, y: p.y + dy / l * 0.5 };
  };
  const e0 = ex(d0), e1 = ex(d1), e2 = ex(d2);
  const a = ((e1.x - e0.x) * (s2.y - s0.y) - (e2.x - e0.x) * (s1.y - s0.y)) / denom;
  const cc = ((e2.x - e0.x) * (s1.x - s0.x) - (e1.x - e0.x) * (s2.x - s0.x)) / denom;
  const b = ((e1.y - e0.y) * (s2.y - s0.y) - (e2.y - e0.y) * (s1.y - s0.y)) / denom;
  const d = ((e2.y - e0.y) * (s1.x - s0.x) - (e1.y - e0.y) * (s2.x - s0.x)) / denom;
  c.save();
  c.beginPath();
  c.moveTo(e0.x, e0.y); c.lineTo(e1.x, e1.y); c.lineTo(e2.x, e2.y); c.closePath();
  c.clip();
  c.transform(a, b, cc, d, e0.x - (a * s0.x + cc * s0.y), e0.y - (b * s0.x + d * s0.y));
  const bx0 = Math.max(0, Math.floor(Math.min(s0.x, s1.x, s2.x)) - 1);
  const by0 = Math.max(0, Math.floor(Math.min(s0.y, s1.y, s2.y)) - 1);
  const bx1 = Math.min(img.width, Math.ceil(Math.max(s0.x, s1.x, s2.x)) + 1);
  const by1 = Math.min(img.height, Math.ceil(Math.max(s0.y, s1.y, s2.y)) + 1);
  if (bx1 > bx0 && by1 > by0) c.drawImage(img, bx0, by0, bx1 - bx0, by1 - by0, bx0, by0, bx1 - bx0, by1 - by0);
  c.restore();
}

/* ---------------- 资产辅助画布 ---------------- */

function buildAlphaCanvas(asset) {
  const a = new OffscreenCanvas(asset.w, asset.h);
  const c = a.getContext('2d');
  c.drawImage(asset.rgba, 0, 0);
  c.globalCompositeOperation = 'source-in';
  c.fillStyle = '#fff';
  c.fillRect(0, 0, asset.w, asset.h);
  asset.alpha = a;
}

let lutCache = new Map();
function applyLut(tctx, cnv, brightness, contrast) {
  const key = brightness + '|' + contrast;
  let lut = lutCache.get(key);
  if (!lut) {
    lut = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) lut[i] = clamp((i - 128) * contrast + 128 + brightness, 0, 255);
    if (lutCache.size > 64) lutCache.clear();
    lutCache.set(key, lut);
  }
  const img = tctx.getImageData(0, 0, cnv.width, cnv.height);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] === 0) continue;
    d[i] = lut[d[i]]; d[i + 1] = lut[d[i + 1]]; d[i + 2] = lut[d[i + 2]];
  }
  tctx.putImageData(img, 0, 0);
}

function getNoiseCanvas(asset, seed, sigma) {
  const bucket = Math.max(1, Math.min(12, Math.round(sigma / 1.5)));
  const key = seed + ':' + bucket;
  let nc = asset.noise.get(key);
  if (nc) return nc;
  let s = (seed >>> 0) || 1;
  const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0, s / 4294967296);
  const gauss = () => {
    const u = Math.max(1e-9, rnd()), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  nc = new OffscreenCanvas(asset.w, asset.h);
  const c = nc.getContext('2d');
  const img = c.createImageData(asset.w, asset.h);
  const d = img.data;
  const amp = bucket * 1.5 * 10; // 灰度振幅
  for (let i = 0; i < d.length; i += 4) {
    const g = clamp(128 + gauss() * amp * 0.5, 0, 255);
    d[i] = d[i + 1] = d[i + 2] = g; d[i + 3] = 255;
  }
  c.putImageData(img, 0, 0);
  asset.noise.set(key, nc);
  return nc;
}

const tints = new Map(); // 弱缓存：按 bitmap 对象缓存着色剪影
function getTint(key, bitmap, r, g, b) {
  let t = tints.get(key);
  if (!t || t.src !== bitmap) {
    const cnv = new OffscreenCanvas(bitmap.width, bitmap.height);
    const c = cnv.getContext('2d');
    c.drawImage(bitmap, 0, 0);
    c.globalCompositeOperation = 'source-in';
    c.fillStyle = `rgb(${r},${g},${b})`;
    c.fillRect(0, 0, cnv.width, cnv.height);
    t = { src: bitmap, canvas: cnv };
    if (tints.size > 4) tints.clear();
    tints.set(key, t);
  }
  return t;
}

/* ---------------- 叠加层（屏幕空间，线宽不随缩放变化） ---------------- */

function drawOverlays(isExtract, bgImg) {
  const f = frame, v = f.view;
  const toS = (p) => ({ x: p.x * v.scale + v.tx, y: p.y * v.scale + v.ty });

  // 图像边界
  const t0 = toS({ x: 0, y: 0 }), t1 = toS({ x: f.target.w, y: f.target.h });
  ctx.save();
  ctx.strokeStyle = 'rgba(120,140,180,0.5)';
  ctx.lineWidth = 1;
  ctx.strokeRect(t0.x, t0.y, t1.x - t0.x, t1.y - t0.y);

  // 网格
  if (f.ui.grid.on) drawGrid(f, toS);

  // 实例轮廓 / 网格线 / 控制柄
  for (const o of f.overlays.outlines) {
    ctx.beginPath();
    o.pts.forEach((p, i) => { const s = toS(p); i ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
    if (o.closed) ctx.closePath();
    ctx.strokeStyle = o.style === 'edit' ? '#ff9f43' : 'rgba(78,163,255,0.95)';
    ctx.lineWidth = o.style === 'edit' ? 1.5 : 1.2;
    if (o.dash) ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const p of f.overlays.meshPts || []) {
    const s = toS(p);
    ctx.fillStyle = p.on ? '#ff9f43' : '#e8e8e8';
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(s.x - 3.5, s.y - 3.5, 7, 7);
    ctx.fill(); ctx.stroke();
  }
  for (const hd of f.overlays.handles) {
    const s = toS(hd);
    if (hd.kind === 'rotate') {
      const anchor = toS(hd.anchor);
      ctx.strokeStyle = 'rgba(78,163,255,0.9)';
      ctx.beginPath(); ctx.moveTo(anchor.x, anchor.y); ctx.lineTo(s.x, s.y); ctx.stroke();
      ctx.beginPath(); ctx.arc(s.x, s.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#fff'; ctx.fill();
      ctx.strokeStyle = '#4ea3ff'; ctx.stroke();
    } else {
      ctx.fillStyle = hd.kind === 'persp' ? '#ffd166' : '#fff';
      ctx.strokeStyle = '#333';
      ctx.beginPath();
      ctx.rect(s.x - 4, s.y - 4, 8, 8);
      ctx.fill(); ctx.stroke();
    }
  }

  // 边界高亮：实例变形四边形虚线（红色轮廓描边已在场景缓冲绘制；
  // Poisson 预览补丁为不透明矩形无法取剪影，此处虚线即其边界指示）
  if (f.ui.boundaryHighlight && !isExtract) {
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = 'rgba(255,96,64,0.85)';
    ctx.lineWidth = 1;
    for (const inst of f.instances) {
      if (!inst._quad) continue;
      ctx.beginPath();
      inst._quad.forEach((p, i) => { const s = toS(p); i ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
      ctx.closePath();
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // 智能参考线
  if (f.overlays.guides?.length) {
    ctx.strokeStyle = '#ff4fd8';
    ctx.lineWidth = 1;
    for (const g of f.overlays.guides) {
      ctx.beginPath();
      if (g.axis === 'x') { const s = toS({ x: g.at, y: 0 }); ctx.moveTo(s.x, 0); ctx.lineTo(s.x, vh); }
      else { const s = toS({ x: 0, y: g.at }); ctx.moveTo(0, s.y); ctx.lineTo(vw, s.y); }
      ctx.stroke();
    }
  }

  // 抠图模式：多边形橡皮筋 / 选区形状 / 笔刷光标
  if (isExtract && f.extract) {    const ex = f.extract;
    if (ex.polygon?.length) {
      ctx.beginPath();
      ex.polygon.forEach((p, i) => { const s = toS(p); i ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y); });
      if (ex.polygonCursor) { const s = toS(ex.polygonCursor); ctx.lineTo(s.x, s.y); }
      ctx.strokeStyle = '#ffe066'; ctx.lineWidth = 1.5; ctx.stroke();
      // 隐式闭合边（末点→首点）：让"闭合"状态可见
      if (ex.polygon.length >= 2) {
        const a = toS(ex.polygon[ex.polygon.length - 1]), b = toS(ex.polygon[0]);
        ctx.strokeStyle = 'rgba(255,224,102,0.45)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.setLineDash([]);
      }
      for (const p of ex.polygon) {
        const s = toS(p);
        ctx.beginPath(); ctx.arc(s.x, s.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffe066'; ctx.fill();
        ctx.strokeStyle = '#222'; ctx.stroke();
      }
    }
    if (ex.shapeRubber) {
      const a = toS(ex.shapeRubber.pts[0]), b = toS(ex.shapeRubber.pts[1]);
      ctx.strokeStyle = '#ffe066'; ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      if (ex.shapeRubber.shape === 'rect') ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
      else {
        ctx.beginPath();
        ctx.ellipse((a.x + b.x) / 2, (a.y + b.y) / 2, Math.abs(b.x - a.x) / 2, Math.abs(b.y - a.y) / 2, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    if (ex.brushCursor) {
      const s = toS(ex.brushCursor);
      ctx.beginPath(); ctx.arc(s.x, s.y, ex.brushR * v.scale, 0, Math.PI * 2);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.strokeStyle = 'rgba(0,0,0,0.6)';
      ctx.beginPath(); ctx.arc(s.x, s.y, ex.brushR * v.scale + 1, 0, Math.PI * 2); ctx.stroke();
    }
  }

  // 绘制模式：笔刷光标
  if (f.mode === 'paint' && f.paint?.brushCursor) {
    const s = toS(f.paint.brushCursor);
    ctx.beginPath();
    ctx.arc(s.x, s.y, Math.max(2, f.paint.brushR * v.scale), 0, Math.PI * 2);
    ctx.strokeStyle = f.paint.eraser ? '#ff6b6b' : '#7CFC9A';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(s.x, s.y, Math.max(2, f.paint.brushR * v.scale) + 1, 0, Math.PI * 2);
    ctx.stroke();
  }

  // 框选
  if (f.overlays.marquee) {
    const a = toS({ x: f.overlays.marquee.x0, y: f.overlays.marquee.y0 });
    const b = toS({ x: f.overlays.marquee.x1, y: f.overlays.marquee.y1 });
    ctx.strokeStyle = '#4ea3ff'; ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
    ctx.setLineDash([]);
  }
  ctx.restore();

  // Before/After 分屏
  if (f.ui.split?.on && !isExtract) drawSplit(f, bgImg);

  // 局部放大镜
  if (f.ui.magnifier?.on && f.cursor.inside && !isExtract) drawMagnifier(f);
}

function drawGrid(f, toS) {
  const size = f.ui.grid.size;
  if (size < 2) return;
  const nx = Math.floor(f.target.w / size), ny = Math.floor(f.target.h / size);
  if (nx * ny > 4000) return;
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < nx; i++) { const s = toS({ x: i * size, y: 0 }); ctx.moveTo(s.x, s.y); const e = toS({ x: i * size, y: f.target.h }); ctx.lineTo(e.x, e.y); }
  for (let j = 1; j < ny; j++) { const s = toS({ x: 0, y: j * size }); ctx.moveTo(s.x, s.y); const e = toS({ x: f.target.w, y: j * size }); ctx.lineTo(e.x, e.y); }
  ctx.stroke();
}

function drawSplit(f, bgImg) {
  const v = f.view;
  const splitX = Math.round(vw * f.ui.split.pos);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, splitX, vh);
  ctx.clip();
  ctx.translate(v.tx, v.ty);
  ctx.scale(v.scale, v.scale);
  ctx.imageSmoothingEnabled = v.scale < 0.99;
  ctx.drawImage(bgImg.bitmap, 0, 0); // 左侧：原始目标图
  ctx.restore();
  ctx.save();
  ctx.strokeStyle = '#ffd166';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(splitX, 0); ctx.lineTo(splitX, vh); ctx.stroke();
  ctx.font = '12px system-ui, "Microsoft YaHei", sans-serif';
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(8, 8, 76, 22); ctx.fillRect(vw - 92, 8, 84, 22);
  ctx.fillStyle = '#fff';
  ctx.fillText('◀ 原始', 16, 23);
  ctx.fillText('合成 ▶', vw - 84, 23);
  ctx.restore();
}

function drawMagnifier(f) {
  const v = f.view;
  const m = f.ui.magnifier;
  const r = m.r, zoom = m.zoom;
  const ci = { x: (f.cursor.x - v.tx) / v.scale, y: (f.cursor.y - v.ty) / v.scale };
  const span = (2 * r) / (v.scale * zoom);
  ctx.save();
  ctx.beginPath();
  ctx.arc(f.cursor.x, f.cursor.y, r, 0, Math.PI * 2);
  ctx.save();
  ctx.clip();
  ctx.imageSmoothingEnabled = false;
  try {
    ctx.drawImage(sceneCanvas, ci.x - span / 2, ci.y - span / 2, span, span,
      f.cursor.x - r, f.cursor.y - r, 2 * r, 2 * r);
  } catch { /* 源区域越界时忽略 */ }
  ctx.restore();
  ctx.strokeStyle = '#4ea3ff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(f.cursor.x - 10, f.cursor.y); ctx.lineTo(f.cursor.x + 10, f.cursor.y);
  ctx.moveTo(f.cursor.x, f.cursor.y - 10); ctx.lineTo(f.cursor.x, f.cursor.y + 10);
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

/* ---------------- 实例图层导出（Poisson 预览用：几何+外观，不含融合） ---------------- */

function renderLayerRequest(m) {
  const inst = m.inst;
  const asset = assets.get(inst.asset_id);
  if (!asset) { self.postMessage({ t: 'layer', reqId: m.reqId, error: 'asset missing' }); return; }
  const { w, h } = asset;
  const gd = instGridDef(inst);
  const src = [], dst = [];
  for (let j = 0; j <= gd.rows; j++)
    for (let i = 0; i <= gd.cols; i++) {
      const s = { x: i * w / gd.cols, y: j * h / gd.rows };
      src.push(s);
      dst.push(instPoint(inst, s, w, h));
    }
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const p of dst) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  const lpad = 2 + (inst.edge_blur > 0 ? inst.edge_blur * 3 : 0);
  x0 = Math.floor(x0 - lpad); y0 = Math.floor(y0 - lpad);
  x1 = Math.ceil(x1 + lpad); y1 = Math.ceil(y1 + lpad);
  const bw = x1 - x0, bh = y1 - y0;
  if (bw <= 0 || bh <= 0) { self.postMessage({ t: 'layer', reqId: m.reqId, error: 'empty' }); return; }
  const cnv = new OffscreenCanvas(bw, bh);
  const c = cnv.getContext('2d');
  const cellTris = [];
  for (let j = 0; j < gd.rows; j++)
    for (let i = 0; i < gd.cols; i++) {
      const a = j * (gd.cols + 1) + i, b = a + 1, cc2 = a + gd.cols + 1, d = cc2 + 1;
      cellTris.push([a, b, d], [a, d, cc2]);
    }
  for (const [ia, ib, ic] of cellTris) {
    drawTexturedTri(c, asset.rgba, src[ia], src[ib], src[ic],
      { x: dst[ia].x - x0, y: dst[ia].y - y0 },
      { x: dst[ib].x - x0, y: dst[ib].y - y0 },
      { x: dst[ic].x - x0, y: dst[ic].y - y0 });
  }
  const app = inst.appearance || {};
  if (app.noise > 0) {
    const nc = getNoiseCanvas(asset, inst.noise_seed, app.noise);
    if (nc) {
      const wc = new OffscreenCanvas(bw, bh);
      const wctx = wc.getContext('2d');
      for (const [ia, ib, ic] of cellTris) {
        drawTexturedTri(wctx, nc, src[ia], src[ib], src[ic],
          { x: dst[ia].x - x0, y: dst[ia].y - y0 },
          { x: dst[ib].x - x0, y: dst[ib].y - y0 },
          { x: dst[ic].x - x0, y: dst[ic].y - y0 });
      }
      wctx.globalCompositeOperation = 'destination-in';
      wctx.drawImage(cnv, 0, 0);
      c.globalCompositeOperation = 'soft-light';
      c.globalAlpha = Math.min(1, app.noise / 10);
      c.drawImage(wc, 0, 0);
      c.globalAlpha = 1;
      c.globalCompositeOperation = 'source-over';
    }
  }
  if (Math.abs(app.brightness) > 1e-6 || Math.abs(app.contrast - 1) > 1e-6) {
    applyLut(c, cnv, app.brightness || 0, app.contrast || 1);
  }
  if (app.blur > 0) {
    const wc = new OffscreenCanvas(bw, bh);
    const wctx = wc.getContext('2d');
    wctx.filter = `blur(${app.blur}px)`;
    wctx.drawImage(cnv, 0, 0);
    wctx.filter = 'none';
    wctx.globalCompositeOperation = 'destination-in';
    wctx.drawImage(cnv, 0, 0);
    c.clearRect(0, 0, bw, bh);
    c.drawImage(wc, 0, 0);
  }
  // 边缘模糊：不夹紧轮廓，alpha 随颜色一起向外渐隐
  if (inst.edge_blur > 0) {
    const wc = new OffscreenCanvas(bw, bh);
    const wctx = wc.getContext('2d');
    wctx.filter = `blur(${Math.min(20, inst.edge_blur)}px)`;
    wctx.drawImage(cnv, 0, 0);
    wctx.filter = 'none';
    c.clearRect(0, 0, bw, bh);
    c.drawImage(wc, 0, 0);
  }
  createImageBitmap(cnv).then((bmp) => {
    self.postMessage({ t: 'layer', reqId: m.reqId, bitmap: bmp, x: x0, y: y0, w: bw, h: bh }, [bmp]);
  }).catch((err) => self.postMessage({ t: 'layer', reqId: m.reqId, error: String(err) }));
}
