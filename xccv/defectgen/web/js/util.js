/* Defect Augment Studio — 共享几何与工具库
 * 主线程与渲染 Worker 共用同一份几何实现，保证预览与后端导出一致。
 * 几何链（与 backend/composite.py 严格一致）：
 *   源像素 s --网格FFD(+d)--> u --透视H--> p --仿射A--> 目标坐标 t
 */

export const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const deg2rad = (d) => d * Math.PI / 180;
export const rad2deg = (r) => r * 180 / Math.PI;
export const deepClone = (o) => (o == null ? o : JSON.parse(JSON.stringify(o)));

/* ---------------- 仿射 A（绕资产中心：缩放→错切→旋转，平移到 position） ---------------- */

export function affine2x3(inst, w, h) {
  const a = inst.affine || {};
  const sx = a.scale_x ?? 1, sy = a.scale_y ?? 1;
  const rot = deg2rad(a.rotation ?? 0);
  const shx = a.shear_x ?? 0, shy = a.shear_y ?? 0;
  const pos = inst.position || { x: 0, y: 0 };
  const cos = Math.cos(rot), sin = Math.sin(rot);
  // M = R · SH · S
  const m00 = cos * 1 * sx + -sin * shy * sx;   // R·SH 列先算再乘 S 对角
  // 展开：R·SH = [[cos, -sin],[sin, cos]]·[[1, shx],[shy, 1]]
  const rsh00 = cos, rsh01 = cos * shx - sin;
  const rsh10 = sin, rsh11 = sin * shx + cos;
  const M00 = rsh00 * sx, M01 = rsh01 * sy;
  const M10 = rsh10 * sx, M11 = rsh11 * sy;
  const cx = w / 2, cy = h / 2;
  return {
    a: M00, b: M10, c: M01, d: M11,
    e: pos.x - (M00 * cx + M01 * cy),
    f: pos.y - (M10 * cx + M11 * cy),
  };
}

export function apply2x3(m, x, y) {
  return { x: m.a * x + m.c * y + m.e, y: m.b * x + m.d * y + m.f };
}

export function invert2x3(m) {
  const det = m.a * m.d - m.b * m.c;
  if (Math.abs(det) < 1e-12) return null;
  const ia = m.d / det, ib = -m.b / det, ic = -m.c / det, id = m.a / det;
  return { a: ia, b: ib, c: ic, d: id, e: -(ia * m.e + ic * m.f), f: -(ib * m.e + id * m.f) };
}

/* ---------------- 透视 H（3×3，资产局部坐标 → 仿射前局部坐标） ---------------- */

export function hApply(H, x, y) {
  const xp = H[0] * x + H[1] * y + H[2];
  const yp = H[3] * x + H[4] * y + H[5];
  const wp = H[6] * x + H[7] * y + H[8];
  return { x: xp / wp, y: yp / wp };
}

/* 解 3×3 单应：src[4] → dst[4]（点序：左上、右上、右下、左下），高斯消元求 8 参数 */
export function solveHomography(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i], { x: X, y: Y } = dst[i];
    A.push([x, y, 1, 0, 0, 0, -x * X, -y * X]); b.push(X);
    A.push([0, 0, 0, x, y, 1, -x * Y, -y * Y]); b.push(Y);
  }
  // 高斯消元（部分主元）
  const n = 8;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    if (Math.abs(A[piv][col]) < 1e-12) return null;
    [A[col], A[piv]] = [A[piv], A[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = A[r][col] / A[col][col];
      for (let c = col; c < n; c++) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }
  const hv = b.map((v, i) => v / A[i][i]);
  return [...hv, 1];
}

export const QUAD_ORDER = [[0, 0], [1, 0], [1, 1], [0, 1]]; // 归一化角序：LT RT RB LB

/* ---------------- 网格 FFD 位移场 ---------------- */

export function meshVertexPos(mesh, i, j, w, h) {
  // 顶点 (i,j) 的源坐标（位移前）
  return { x: i * w / mesh.cols, y: j * h / mesh.rows };
}

export function meshDisplace(mesh, x, y, w, h) {
  // d(s)：网格顶点位移的双线性插值（与 backend make_disp_fn 一致）
  const cols = mesh.cols, rows = mesh.rows;
  const gx = clamp(x / (w / cols), 0, cols), gy = clamp(y / (h / rows), 0, rows);
  const i0 = Math.min(Math.floor(gx), cols - 1), j0 = Math.min(Math.floor(gy), rows - 1);
  const i1 = i0 + 1, j1 = j0 + 1;
  const fx = gx - i0, fy = gy - j0;
  const g = mesh.grid; // 长度 (rows+1)*(cols+1)，索引 j*(cols+1)+i
  const v00 = g[j0 * (cols + 1) + i0], v10 = g[j0 * (cols + 1) + i1];
  const v01 = g[j1 * (cols + 1) + i0], v11 = g[j1 * (cols + 1) + i1];
  return {
    dx: v00[0] * (1 - fx) * (1 - fy) + v10[0] * fx * (1 - fy) + v01[0] * (1 - fx) * fy + v11[0] * fx * fy,
    dy: v00[1] * (1 - fx) * (1 - fy) + v10[1] * fx * (1 - fy) + v01[1] * (1 - fx) * fy + v11[1] * fx * fy,
  };
}

/* 源坐标 s → 仿射前局部坐标 u（s + d(s)），再经 H → p */
export function localForward(inst, s, w, h) {
  let u = s;
  if (inst.mesh) {
    const d = meshDisplace(inst.mesh, s.x, s.y, w, h);
    u = { x: s.x + d.dx, y: s.y + d.dy };
  }
  if (inst.perspective_h) return hApply(inst.perspective_h, u.x, u.y);
  return u;
}

/* 完整正变换：资产局部坐标 → 目标图像坐标 */
export function instToLocal(inst, s, w, h) { return localForward(inst, s, w, h); }

export function localToImage(inst, p, w, h) {
  const m = affine2x3(inst, w, h);
  return apply2x3(m, p.x, p.y);
}

export function instPoint(inst, s, w, h) {
  return localToImage(inst, localForward(inst, s, w, h), w, h);
}

/* 逆变换：目标图像坐标 → 资产局部源坐标 s（渲染拾取用） */
export function instInverse(inst, pt, w, h) {
  const m = invert2x3(affine2x3(inst, w, h));
  if (!m) return null;
  let p = apply2x3(m, pt.x, pt.y); // → 仿射前局部
  if (inst.perspective_h) {
    const H = inst.perspective_h;
    const Hinv = solveHomography(
      [hApply(H, 0, 0), hApply(H, 100, 0), hApply(H, 100, 100), hApply(H, 0, 100)],
      [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }, { x: 0, y: 100 }]);
    if (!Hinv) return null;
    p = hApply(Hinv, p.x, p.y);
  }
  if (inst.mesh) {
    const d = meshDisplace(inst.mesh, p.x, p.y, w, h);
    p = { x: p.x - d.dx, y: p.y - d.dy };
  }
  return p;
}

/* 实例变形后的角点（图像坐标），用于包围盒/选择框 */
export function instCorners(inst, w, h) {
  return QUAD_ORDER.map(([nx, ny]) => instPoint(inst, { x: nx * w, y: ny * h }, w, h));
}

export function instAABB(inst, w, h) {
  const cs = instCorners(inst, w, h);
  const xs = cs.map((c) => c.x), ys = cs.map((c) => c.y);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
}

/* 多实例联合 AABB */
export function unionAABB(boxes) {
  return {
    x0: Math.min(...boxes.map((b) => b.x0)), y0: Math.min(...boxes.map((b) => b.y0)),
    x1: Math.max(...boxes.map((b) => b.x1)), y1: Math.max(...boxes.map((b) => b.y1)),
  };
}

/* 变形网格顶点（图像坐标）：FFD/Perspective 编辑用 */
export function instGrid(inst, w, h) {
  const mesh = inst.mesh || { cols: 4, rows: 4 };
  const pts = [];
  for (let j = 0; j <= mesh.rows; j++)
    for (let i = 0; i <= mesh.cols; i++)
      pts.push({ ...instPoint(inst, meshVertexPos(mesh, i, j, w, h), w, h), i, j });
  return pts; // 行优先
}

/* 网格位移限幅：位移不超过网格间距的 50%（防自交，规格 §6.2.3） */
export function clampMesh(mesh, w, h) {
  const cx = w / mesh.cols / 2, cy = h / mesh.rows / 2;
  mesh.grid = mesh.grid.map(([dx, dy]) => [clamp(dx, -cx, cx), clamp(dy, -cy, cy)]);
  return mesh;
}

/* 弹性扰动：固定边角，内部顶点按种子随机小幅位移（规格 §6.2.4） */
export function elasticMesh(seed, cols, rows, w, h, ampPct = 0.28) {
  let s = seed >>> 0 || 1;
  const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0, s / 4294967296);
  const grid = [];
  const cx = (w / cols) * ampPct, cy = (h / rows) * ampPct;
  for (let j = 0; j <= rows; j++)
    for (let i = 0; i <= cols; i++) {
      const border = i === 0 || j === 0 || i === cols || j === rows;
      grid.push(border ? [0, 0]
        : [(rnd() * 2 - 1) * cx, (rnd() * 2 - 1) * cy]);
    }
  return { cols, rows, grid };
}

/* ---------------- 点在四边形内（变形包围体拾取） ---------------- */

export function pointInQuad(pt, quad) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = quad[i], b = quad[(i + 1) % 4];
    const cross = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
    if (cross !== 0) {
      const s = Math.sign(cross);
      if (sign === 0) sign = s; else if (s !== sign) return false;
    }
  }
  return true;
}

/* ---------------- 外观/融合默认值 ---------------- */

export const DEFAULT_APPEARANCE = { brightness: 0, contrast: 1, hue: 0, saturation: 1, blur: 0, noise: 0 };
export const APPEARANCE_SPEC = [
  { key: 'brightness', label: '亮度', min: -50, max: 50, step: 1, def: 0 },
  { key: 'contrast', label: '对比度', min: 0.5, max: 2, step: 0.01, def: 1 },
  { key: 'hue', label: '色相偏移°', min: -30, max: 30, step: 1, def: 0 },
  { key: 'saturation', label: '饱和度', min: 0, max: 2, step: 0.01, def: 1 },
  { key: 'blur', label: '模糊 px', min: 0, max: 5, step: 0.1, def: 0 },
  { key: 'noise', label: '噪声 σ', min: 0, max: 15, step: 0.5, def: 0 },
];
export const APPEARANCE_KEYS = APPEARANCE_SPEC.map((s) => s.key);

export function normAppearance(app) {
  const out = { ...DEFAULT_APPEARANCE };
  if (app) for (const k of APPEARANCE_KEYS) if (app[k] != null) out[k] = app[k];
  return out;
}

export function appearanceIsDefault(app) {
  return APPEARANCE_KEYS.every((k) => Math.abs((app?.[k] ?? DEFAULT_APPEARANCE[k]) - DEFAULT_APPEARANCE[k]) < 1e-9);
}

/* ---------------- 杂项 ---------------- */

export function throttleRaf(fn) {
  let queued = false, lastArgs = null;
  return (...args) => {
    lastArgs = args; queued = true;
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; fn(...lastArgs); });
  };
}

export function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

/* 降采样尺寸：最长边不超过 maxSide */
export function fitSize(w, h, maxSide) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return { w: Math.max(1, Math.round(w * s)), h: Math.max(1, Math.round(h * s)), scale: s };
}

export function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1048576).toFixed(2) + ' MB';
}

/* 纯 JS 24 位 BMP 编码：绕开浏览器 PNG 编码器（OffscreenCanvas.convertToBlob /
 * canvas.toBlob 在部分环境偶发永久挂起）。d 为 RGBA ImageData.data。
 * 返回 Blob 可直接用于 FormData 上传或 dataURL（OpenCV/cv2.imdecode 可解码）。 */
export function encodeBmpBlob(d, w, h) {
  const rowSize = (w * 3 + 3) & ~3;
  const dataSize = rowSize * h;
  const buf = new Uint8Array(54 + dataSize);
  const dv = new DataView(buf.buffer);
  buf[0] = 0x42; buf[1] = 0x4d;                   // 'BM'
  dv.setUint32(2, buf.length, true);
  dv.setUint32(10, 54, true);                     // 像素数据偏移
  dv.setUint32(14, 40, true);                     // BITMAPINFOHEADER
  dv.setInt32(18, w, true);
  dv.setInt32(22, h, true);
  dv.setUint16(26, 1, true);                      // planes
  dv.setUint16(28, 24, true);                     // bpp
  dv.setUint32(34, dataSize, true);               // 图像数据大小
  for (let y = 0; y < h; y++) {
    let p = 54 + (h - 1 - y) * rowSize;           // BMP 自底向上
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      buf[p++] = d[i + 2]; buf[p++] = d[i + 1]; buf[p++] = d[i];
    }
  }
  return new Blob([buf], { type: 'image/bmp' });
}

/* ---------------- 保存文件（可选目录） ----------------
 * 支持 File System Access 的浏览器（Chrome/Edge）：弹出系统保存对话框，
 * 用户自选目录与文件名；点"取消"返回 'cancel'，调用方应中止后续工作。
 * 不支持的浏览器回退为常规下载（浏览器设置里的下载目录）。
 * 自动化环境（navigator.webdriver）走常规下载，保证测试可重复。 */

export function canPickSaveDir() {
  return typeof window.showSaveFilePicker === 'function' && !navigator.webdriver;
}

export async function pickSaveHandle(suggestedName, mime = 'image/png') {
  if (!canPickSaveDir()) return null;
  const ext = suggestedName.includes('.') ? suggestedName.slice(suggestedName.lastIndexOf('.')) : '.png';
  try {
    return await window.showSaveFilePicker({
      suggestedName,
      types: [{ description: ext.replace('.', '').toUpperCase() + ' 图像',
                accept: { [mime]: [ext] } }],
    });
  } catch (e) {
    if (e && e.name === 'AbortError') return 'cancel';
    return null;   // 其它错误（权限/策略）→ 回退常规下载
  }
}

export async function writeBlobToHandle(handle, blob) {
  const w = await handle.createWritable();
  await w.write(blob);
  await w.close();
}

export function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/* 保存：有句柄写句柄（用户自选位置），否则常规下载。返回 'saved' | 'downloaded' */
export async function saveBlob(blob, name, handle) {
  if (handle && handle !== 'cancel') {
    await writeBlobToHandle(handle, blob);
    return 'saved';
  }
  downloadBlob(blob, name);
  return 'downloaded';
}
