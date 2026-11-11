/* Defect Augment Studio — 颜色工具（v1.4）
 * 三色盘（HSV：色相条 + 饱和度/明度方盘）、HEX 输入、RGB/HEX/HSV 互转。
 * openPicker(anchorEl, hex, onChange) 弹出取色浮层；供绘制工坊自定义缺陷颜色。
 */

export function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || '').trim());
  if (!m) return { r: 204, g: 204, b: 204 };
  const n = parseInt(m[1], 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

export function rgbToHex(r, g, b) {
  const c = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
  return '#' + c(r) + c(g) + c(b);
}

export function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d > 1e-9) {
    if (mx === r) h = ((g - b) / d) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  return { h, s: mx < 1e-9 ? 0 : d / mx, v: mx };
}

export function hsvToRgb(h, s, v) {
  h = ((h % 360) + 360) % 360;
  const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else { r = c; b = x; }
  return { r: (r + m) * 255, g: (g + m) * 255, b: (b + m) * 255 };
}

let pop = null;   // 单例浮层

export function closePicker() {
  if (pop) { pop.root.remove(); pop = null; }
}

/* anchorEl：定位参照；onChange(hex) 实时回调 */
export function openPicker(anchorEl, hex, onChange) {
  closePicker();
  const st = rgbToHsv(hexToRgb(hex).r, hexToRgb(hex).g, hexToRgb(hex).b);
  const cur = { h: st.h, s: st.s, v: st.v };

  const root = document.createElement('div');
  root.style.cssText = 'position:fixed;z-index:1000;background:#1d1f24;border:1px solid #3a3f4b;'
    + 'border-radius:8px;padding:10px;box-shadow:0 8px 28px rgba(0,0,0,.5);user-select:none;';

  const pad = document.createElement('canvas');
  pad.width = 188; pad.height = 140;
  pad.style.cssText = 'display:block;cursor:crosshair;border-radius:4px;';
  const strip = document.createElement('canvas');
  strip.width = 188; strip.height = 14;
  strip.style.cssText = 'display:block;cursor:crosshair;border-radius:4px;margin-top:8px;';

  const row = document.createElement('div');
  row.style.cssText = 'display:flex;align-items:center;gap:8px;margin-top:8px;';
  const prev = document.createElement('div');
  prev.style.cssText = 'width:34px;height:24px;border-radius:4px;border:1px solid #444;';
  const input = document.createElement('input');
  input.type = 'text'; input.spellcheck = false;
  input.style.cssText = 'flex:1;background:#14151a;color:#e8e8e8;border:1px solid #3a3f4b;'
    + 'border-radius:4px;padding:4px 6px;font:12px/1.4 monospace;';
  row.append(prev, input);

  const tip = document.createElement('div');
  tip.textContent = '拖动取色 · 输入 HEX · Esc 关闭';
  tip.style.cssText = 'margin-top:6px;font-size:11px;color:#8a92a3;';

  root.append(pad, strip, row, tip);
  document.body.appendChild(root);

  // 定位（避开屏幕边缘）
  const ar = anchorEl.getBoundingClientRect();
  const pw = 210, ph = 250;
  let px = Math.min(ar.left, window.innerWidth - pw - 8);
  let py = ar.bottom + 6;
  if (py + ph > window.innerHeight - 8) py = Math.max(8, ar.top - ph - 6);
  root.style.left = Math.max(8, px) + 'px';
  root.style.top = py + 'px';

  const padCtx = pad.getContext('2d');
  const stripCtx = strip.getContext('2d');

  function drawPad() {
    const [r, g, b] = Object.values(hsvToRgb(cur.h, 1, 1));
    padCtx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
    padCtx.fillRect(0, 0, pad.width, pad.height);
    let grd = padCtx.createLinearGradient(0, 0, pad.width, 0);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    padCtx.fillStyle = grd; padCtx.fillRect(0, 0, pad.width, pad.height);
    grd = padCtx.createLinearGradient(0, 0, 0, pad.height);
    grd.addColorStop(0, 'rgba(0,0,0,0)');
    grd.addColorStop(1, 'rgba(0,0,0,1)');
    padCtx.fillStyle = grd; padCtx.fillRect(0, 0, pad.width, pad.height);
    const mx = cur.s * pad.width, my = (1 - cur.v) * pad.height;
    padCtx.beginPath();
    padCtx.arc(mx, my, 5, 0, Math.PI * 2);
    padCtx.strokeStyle = '#fff'; padCtx.lineWidth = 1.5; padCtx.stroke();
    padCtx.strokeStyle = 'rgba(0,0,0,0.7)'; padCtx.lineWidth = 1;
    padCtx.beginPath(); padCtx.arc(mx, my, 6.2, 0, Math.PI * 2); padCtx.stroke();
  }

  function drawStrip() {
    const grd = stripCtx.createLinearGradient(0, 0, strip.width, 0);
    for (let i = 0; i <= 6; i++) grd.addColorStop(i / 6, `hsl(${i * 60},100%,50%)`);
    stripCtx.fillStyle = grd;
    stripCtx.fillRect(0, 0, strip.width, strip.height);
    const mx = (cur.h / 360) * strip.width;
    stripCtx.fillStyle = '#fff';
    stripCtx.fillRect(mx - 1.5, 0, 3, strip.height);
    stripCtx.strokeStyle = 'rgba(0,0,0,0.8)';
    stripCtx.strokeRect(mx - 1.5, 0.5, 3, strip.height - 1);
  }

  function apply(fromInput) {
    const { r, g, b } = hsvToRgb(cur.h, cur.s, cur.v);
    const hx = rgbToHex(r, g, b);
    prev.style.background = hx;
    if (!fromInput) input.value = hx;
    drawPad(); drawStrip();
    onChange?.(hx);
  }

  const dragPad = (ev) => {
    const rc = pad.getBoundingClientRect();
    cur.s = Math.max(0, Math.min(1, (ev.clientX - rc.left) / rc.width));
    cur.v = Math.max(0, Math.min(1, 1 - (ev.clientY - rc.top) / rc.height));
    apply();
  };
  const dragStrip = (ev) => {
    const rc = strip.getBoundingClientRect();
    cur.h = Math.max(0, Math.min(359.9, ((ev.clientX - rc.left) / rc.width) * 360));
    apply();
  };
  const hold = (fn) => (e) => {
    e.preventDefault();
    fn(e);
    const mv = (m) => fn(m);
    const up = () => { window.removeEventListener('pointermove', mv); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
  };
  pad.addEventListener('pointerdown', hold(dragPad));
  strip.addEventListener('pointerdown', hold(dragStrip));
  input.addEventListener('change', () => {
    const rgb = hexToRgb(input.value);
    const hsv = rgbToHsv(rgb.r, rgb.g, rgb.b);
    Object.assign(cur, hsv);
    apply(true);
  });
  const onKey = (e) => { if (e.key === 'Escape') { closePicker(); } };
  window.addEventListener('keydown', onKey);
  setTimeout(() => {
    const onDown = (e) => { if (!root.contains(e.target) && e.target !== anchorEl) closePicker(); };
    window.addEventListener('pointerdown', onDown);
    pop = { root, cleanup: () => { window.removeEventListener('pointerdown', onDown); window.removeEventListener('keydown', onKey); } };
  }, 0);
  pop = { root, cleanup: () => { window.removeEventListener('keydown', onKey); } };
  apply();
  input.focus();
}
