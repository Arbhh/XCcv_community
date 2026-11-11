/* 内嵌浏览器兼容层（QtWebEngine 5.15 = Chromium 83），须先于模块脚本加载。 */
(function () {
  'use strict';

  /* ---- 图像解码 ----
   * imageOrientation:'from-image' 是 Chrome 93 才有的枚举值，旧内核直接拒绝
   * （打开目标图/抠图源图全部失败）；旧内核 Blob 解码也不应用 EXIF 方向，而
   * <img> 解码会应用——统一走 <img> 再转位图。TIFF/16 位等浏览器解不了的
   * 格式交后端转码。 */
  const nativeCIB = window.createImageBitmap.bind(window);
  const fromImageOK = nativeCIB(new ImageData(1, 1), { imageOrientation: 'from-image' })
    .then(() => true, () => false);

  async function viaImg(blob) {
    const url = URL.createObjectURL(blob);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      const c = new OffscreenCanvas(img.naturalWidth, img.naturalHeight);
      c.getContext('2d').drawImage(img, 0, 0);
      return await nativeCIB(c);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function viaBackend(blob) {
    const res = await fetch('/api/decode_image', { method: 'POST', body: blob });
    if (!res.ok) throw new Error('图像解码失败（格式不受支持或文件已损坏）');
    return viaImg(await res.blob());
  }

  window.createImageBitmap = async function (src, ...rest) {
    const cropped = typeof rest[0] === 'number';
    if (!(src instanceof Blob) || cropped) return nativeCIB(src, ...rest);
    if (await fromImageOK) {
      try {
        return await nativeCIB(src, ...rest);
      } catch (_) {
        return viaBackend(src);
      }
    }
    try {
      return await viaImg(src);
    } catch (_) {
      return viaBackend(src);
    }
  };

  /* ---- flex gap（Chrome 84+）----
   * 旧内核解析 gap 但对 flex 容器不生效（按钮/面板挤在一起）：按计算值补外边距。 */
  function flexGapOK() {
    const d = document.createElement('div');
    d.style.cssText = 'display:flex;flex-direction:column;row-gap:1px;position:absolute;visibility:hidden';
    d.appendChild(document.createElement('div'));
    d.appendChild(document.createElement('div'));
    document.documentElement.appendChild(d);
    const ok = d.scrollHeight === 1;
    d.remove();
    return ok;
  }

  function setMargin(el, prop, px) {
    const mark = 'xg' + prop;
    if (el.dataset[mark] === undefined) {
      if (!px || parseFloat(getComputedStyle(el)[prop]) !== 0) return;   // 作者自定义外边距优先
      el.dataset[mark] = '1';
    }
    el.style[prop] = px ? px + 'px' : '';
  }

  function applyGaps() {
    for (const el of document.body.querySelectorAll('*')) {
      const cs = getComputedStyle(el);
      if (!cs.display.includes('flex')) continue;
      const rg = parseFloat(cs.rowGap) || 0;
      const cg = parseFloat(cs.columnGap) || 0;
      if (!rg && !cg) continue;
      const column = cs.flexDirection.startsWith('column');
      const wrap = cs.flexWrap !== 'nowrap';
      const kids = [...el.children].filter((k) => {
        const s = getComputedStyle(k);
        return s.display !== 'none' && s.position !== 'absolute' && s.position !== 'fixed';
      });
      kids.forEach((k, i) => {
        const last = i === kids.length - 1;
        if (column) {
          setMargin(k, 'marginBottom', last ? 0 : rg);
        } else {
          setMargin(k, 'marginRight', last ? 0 : cg);
          if (wrap) setMargin(k, 'marginBottom', rg);
        }
      });
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (flexGapOK()) return;
    let timer = 0;
    const schedule = () => {
      clearTimeout(timer);
      timer = setTimeout(applyGaps, 120);
    };
    applyGaps();
    new MutationObserver(schedule).observe(document.body, {
      childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'hidden'],
    });
  });
})();
