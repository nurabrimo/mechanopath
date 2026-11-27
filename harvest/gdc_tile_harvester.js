/*
 * MechanoPath - GDC tile harvester (browser console script)
 * ---------------------------------------------------------
 * Pulls a fixed, reproducible tile sample from every TCGA-GBM / TCGA-LGG FFPE diagnostic
 * slide through GDC's public tile service, without downloading whole SVS files.
 *
 * How to run (this is how the data in this project were produced, 2026-10-08):
 *   1. Open https://api.gdc.cancer.gov/status in Chrome (same origin as the API, so no CORS issues).
 *   2. Paste this whole file into the DevTools console and press Enter.
 *   3. Parts are saved as mechanopath_gdc_tiles_partNN.zip. Chrome blocks a page's 2nd+
 *      automatic download unless you allow "Automatic downloads" for api.gdc.cancer.gov
 *      (address-bar icon or Settings > Privacy > Site settings). Parts are also kept in
 *      memory: window.MP.partUrls[name] is a blob URL you can re-download from a fresh tab.
 *
 * Per slide (one slide per patient: primary tumour "01", lowest DX number):
 *   - Aperio ImageDescription read with HTTP range requests (AppMag, MPP, scanner, scan date, JPEG Q)
 *   - thumbnail (~1/64 of full resolution, <=1024 px), stitched from Deep Zoom tiles
 *   - tissue mask on the thumbnail (saturation > 0.08, value > 0.2, pen-ink filter)
 *   - haematoxylin density per pixel (Ruifrok-Johnston colour deconvolution)
 *   - 12 tiles of 512x512 at ~0.5 um/px (6 from the top-10% H-dense tissue, 6 uniformly random)
 *   - 4 tiles of 512x512 at ~2 um/px (2 H-dense, 2 random)
 *   - meta.json with all coordinates (level-0 pixels) so every tile can be re-extracted on DCC
 * Sampling is seeded by file_id, so re-running gives the same tiles.
 */
(async () => {
  const CFG = {N20_RAND: 6, N20_H: 6, N5_RAND: 2, N5_H: 2, Q: 0.85, THUMB_MAX: 1024, PER_PART: 90,
               FIRST_PART: 20, SLIDE_CONC: 8, FETCH_CONC: 24, SEED: 20261008};

  // ---------------- manifest: one diagnostic slide per patient ----------------
  const filt = {op: 'and', content: [
    {op: 'in', content: {field: 'cases.project.project_id', value: ['TCGA-GBM', 'TCGA-LGG']}},
    {op: 'in', content: {field: 'data_type', value: ['Slide Image']}},
    {op: 'in', content: {field: 'experimental_strategy', value: ['Diagnostic Slide']}}]};
  const q = new URLSearchParams({filters: JSON.stringify(filt), size: '3000', format: 'json',
    fields: 'file_id,file_name,file_size,cases.submitter_id,cases.project.project_id'});
  const files = (await fetch('/files?' + q).then(r => r.json())).data.hits.map(h => ({
    file_id: h.file_id, file_name: h.file_name, file_size: h.file_size,
    patient: h.cases[0].submitter_id, project: h.cases[0].project.project_id}));
  files.forEach(h => { const m = h.file_name.match(/^TCGA-\w\w-\w{4}-(\d\d)\w-\d\d-DX(\d+)/);
    h.stype = m ? m[1] : 'NA'; h.dx = m ? +m[2] : 99; });
  const byP = {};
  files.forEach(h => { const s = (h.stype === '01' ? 0 : 1000) + h.dx;
    if (!byP[h.patient] || s < byP[h.patient].score) byP[h.patient] = {...h, score: s}; });
  const selected = Object.values(byP);
  const qc = new URLSearchParams({filters: JSON.stringify({op: 'in', content: {field: 'project.project_id', value: ['TCGA-GBM', 'TCGA-LGG']}}),
    fields: 'submitter_id,project.project_id,tissue_source_site.name,tissue_source_site.code', size: '3000', format: 'json'});
  const cases = (await fetch('/cases?' + qc).then(r => r.json())).data.hits.map(h => ({patient: h.submitter_id,
    project: h.project.project_id, tss_code: h.tissue_source_site?.code ?? null, tss_name: h.tissue_source_site?.name ?? null}));

  // ---------------- fetch helpers (global concurrency limit + retries) ----------------
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  let active = 0; const waitq = [];
  const acquire = async () => { if (active < CFG.FETCH_CONC) { active++; return; } await new Promise(r => waitq.push(r)); };
  const release = () => { const n = waitq.shift(); if (n) n(); else active--; };
  async function getBuf(path, opts = {}, expect = 200, tries = 5) {
    let last = '';
    for (let t = 0; t < tries; t++) {
      await acquire(); const ac = new AbortController();
      try { const r = await fetch(path, {...opts, signal: ac.signal});
        if (r.status === expect) return await r.arrayBuffer();     // never read a 200 when a 206 range was asked for
        last = 'HTTP ' + r.status; ac.abort();
      } catch (e) { last = String(e).slice(0, 80); } finally { release(); }
      await sleep(1500 * 2 ** t);
    }
    throw new Error(last);
  }
  const rng = async (fid, a, b) => new DataView(await getBuf('/data/' + fid, {headers: {Range: `bytes=${a}-${b}`}}, 206));
  async function svsDesc(fid) {                                      // TIFF / BigTIFF tag 270 of the first IFD
    const h = await rng(fid, 0, 15); const le = h.getUint16(0) === 0x4949; const big = h.getUint16(2, le) === 43;
    if (big) { const off = Number(h.getBigUint64(8, le)); const n = Number((await rng(fid, off, off + 7)).getBigUint64(0, le));
      const d = await rng(fid, off + 8, off + 8 + n * 20 - 1);
      for (let i = 0; i < n; i++) if (d.getUint16(i * 20, le) === 270) { const c = Number(d.getBigUint64(i * 20 + 4, le)); if (c <= 8) return '';
        const vo = Number(d.getBigUint64(i * 20 + 12, le)); return new TextDecoder('latin1').decode((await rng(fid, vo, vo + c - 1)).buffer); }
    } else { const off = h.getUint32(4, le); const n = (await rng(fid, off, off + 1)).getUint16(0, le);
      const d = await rng(fid, off + 2, off + 2 + n * 12 - 1);
      for (let i = 0; i < n; i++) if (d.getUint16(i * 12, le) === 270) { const c = d.getUint32(i * 12 + 4, le); if (c <= 4) return '';
        const vo = d.getUint32(i * 12 + 8, le); return new TextDecoder('latin1').decode((await rng(fid, vo, vo + c - 1)).buffer); } }
    return '';
  }
  function parseDesc(s) {                                             // operator name ("User") is deliberately not kept
    const o = {}; if (!s) return o; const parts = s.split('|'); const head = parts[0];
    o.aperio = head.split(/\r?\n/)[0].slice(0, 60);
    const qq = head.match(/Q=(\d+)/); if (qq) o.jpeg_q = +qq[1];
    const comp = head.match(/(JPEG|J2K|LZW)[^\s|]*/); if (comp) o.compression = comp[0];
    for (const p of parts.slice(1)) { const m = p.match(/^\s*([^=]+?)\s*=\s*([\s\S]*?)\s*$/); if (!m) continue;
      const k = m[1], v = m[2].replace(/\u0000/g, '');
      if (k === 'AppMag') o.appmag = +v; else if (k === 'MPP') o.mpp = +v; else if (k === 'ScanScope ID') o.scanner = v;
      else if (k === 'Date') o.scan_date = v; else if (k === 'Time') o.scan_time = v; else if (k === 'OriginalWidth') o.orig_w = +v;
      else if (/^Original[hH]eight$/.test(k)) o.orig_h = +v; else if (k === 'Filtered') o.filtered = v;
      else if (k === 'ICC Profile') o.icc = v; else if (k === 'DSR ID') o.dsr = v; else if (k === 'StripeWidth') o.stripe = +v; }
    return o;
  }
  // colour deconvolution (Ruifrok & Johnston H&E-DAB vectors)
  const Mst = [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]].map(r => { const n = Math.hypot(...r); return r.map(x => x / n); });
  const inv3 = m => { const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, D = -(b * i - c * h), E = a * i - c * g, F = -(a * h - b * g),
          G = b * f - c * e, H = -(a * f - c * d), I = a * e - b * d, det = a * A + b * B + c * C;
    return [[A / det, D / det, G / det], [B / det, E / det, H / det], [C / det, F / det, I / det]]; };
  const Minv = inv3(Mst);
  const mulberry32 = a => () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const hashStr = s => { let h = 2166136261; for (const ch of s) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; };
  const sampleK = (arr, k, rnd) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, k); };
  const tileBmp = async (fid, L, c, r) => createImageBitmap(new Blob([await getBuf(`/tile/${fid}?level=${L}&x=${c}&y=${r}`)], {type: 'image/png'}));

  async function processSlide(s) {
    const fid = s.file_id, P = s.patient;
    const rec = {patient: P, project: s.project, file_id: fid, file_name: s.file_name, file_size: s.file_size};
    const md = JSON.parse(new TextDecoder().decode(await getBuf('/tile/metadata/' + fid)));
    const W = +md.Width, H = +md.Height; Object.assign(rec, {W, H, dz_tile: +md.TileSize, dz_overlap: +md.Overlap});
    try { Object.assign(rec, parseDesc(await svsDesc(fid))); } catch (e) { rec.desc_err = String(e).slice(0, 80); }
    const maxL = Math.ceil(Math.log2(Math.max(W, H))); rec.maxL = maxL;
    let k20 = 0;                                                     // level whose pixel size is closest to 0.5 um
    if (rec.mpp > 0) k20 = Math.max(0, Math.round(Math.log2(0.5 / rec.mpp)));
    else if (rec.appmag > 0) k20 = Math.max(0, Math.round(Math.log2(rec.appmag / 20)));
    const L20 = maxL - k20, L5 = L20 - 2; Object.assign(rec, {k20, L20, L5});
    const dim = L => [Math.ceil(W / 2 ** (maxL - L)), Math.ceil(H / 2 ** (maxL - L))];
    const kt = Math.max(0, Math.ceil(Math.log2(Math.max(W, H) / CFG.THUMB_MAX))); const Lt = maxL - kt; const [tw, th] = dim(Lt);
    Object.assign(rec, {Lt, thumb_w: tw, thumb_h: th, thumb_downsample: 2 ** kt});
    const tc = new OffscreenCanvas(tw, th); const tg = tc.getContext('2d', {willReadFrequently: true});
    tg.fillStyle = '#fff'; tg.fillRect(0, 0, tw, th);
    const jobs = [];
    for (let r = 0; r < Math.ceil(th / 512); r++) for (let c = 0; c < Math.ceil(tw / 512); c++)
      jobs.push(tileBmp(fid, Lt, c, r).then(b => { tg.drawImage(b, c * 512 - (c > 0 ? 1 : 0), r * 512 - (r > 0 ? 1 : 0)); b.close(); }));
    await Promise.all(jobs);
    const files = [[`${P}/thumb.jpg`, new Uint8Array(await (await tc.convertToBlob({type: 'image/jpeg', quality: 0.9})).arrayBuffer())]];
    const px = tg.getImageData(0, 0, tw, th).data, n = tw * th, tis = new Uint8Array(n), hd = new Float32Array(n); let ntis = 0;
    for (let i = 0; i < n; i++) { const R = px[4 * i], G = px[4 * i + 1], B = px[4 * i + 2], mx = Math.max(R, G, B), mn = Math.min(R, G, B);
      const sat = mx > 0 ? (mx - mn) / mx : 0, v = mx / 255, pen = (G > R + 5 && G >= B) || (R < 80 && B > 120 && B - R > 70);
      tis[i] = (sat > 0.08 && v > 0.2 && !pen) ? 1 : 0; ntis += tis[i];
      const o0 = -Math.log((R + 1) / 256), o1 = -Math.log((G + 1) / 256), o2 = -Math.log((B + 1) / 256);
      hd[i] = o0 * Minv[0][0] + o1 * Minv[1][0] + o2 * Minv[2][0]; }
    rec.thumb_tissue_frac = +(ntis / n).toFixed(4);
    function candidates(L, minFrac) { const [lw, lh] = dim(L), f = 2 ** (L - Lt), out = [];
      for (let r = 0; r < Math.floor(lh / 512); r++) for (let c = 0; c < Math.floor(lw / 512); c++) {
        const x0 = Math.floor(c * 512 / f), x1 = Math.min(tw, Math.ceil((c + 1) * 512 / f)), y0 = Math.floor(r * 512 / f), y1 = Math.min(th, Math.ceil((r + 1) * 512 / f));
        let t = 0, cnt = 0, hs = 0; for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const i = y * tw + x; cnt++; if (tis[i]) { t++; hs += hd[i]; } }
        if (cnt && t / cnt >= minFrac) out.push({c, r, frac: +(t / cnt).toFixed(3), h: +(t ? hs / t : 0).toFixed(4)}); }
      return out; }
    const rnd = mulberry32(hashStr(fid) ^ CFG.SEED);
    function pick(L, nH, nR, fracs) { let cand = [], used = null;
      for (const fr of fracs) { cand = candidates(L, fr); used = fr; if (cand.length >= nH + nR) break; }
      const sorted = cand.slice().sort((a, b) => b.h - a.h);
      const pool = sorted.slice(0, Math.max(nH, Math.min(sorted.length, Math.max(30, Math.ceil(sorted.length * 0.1)))));
      const selH = sampleK(pool, nH, rnd), ks = new Set(selH.map(t => t.c + '_' + t.r));
      const selR = sampleK(cand.filter(t => !ks.has(t.c + '_' + t.r)), nR, rnd);
      return {sel: [...selH.map(t => ({...t, stratum: 'hdense'})), ...selR.map(t => ({...t, stratum: 'random'}))], ncand: cand.length, minFrac: used}; }
    const p20 = pick(L20, CFG.N20_H, CFG.N20_RAND, [0.75, 0.5, 0.25, 0.05]), p5 = pick(L5, CFG.N5_H, CFG.N5_RAND, [0.5, 0.25, 0.05]);
    Object.assign(rec, {n_cand20: p20.ncand, minfrac20: p20.minFrac, n_cand5: p5.ncand, minfrac5: p5.minFrac});
    const tiles = [...p20.sel.map(t => ({...t, L: L20, mag: 'x20'})), ...p5.sel.map(t => ({...t, L: L5, mag: 'x5'}))];
    const res = await Promise.all(tiles.map(async (t, i) => {
      const b = await tileBmp(fid, t.L, t.c, t.r); const cv = new OffscreenCanvas(512, 512); const g = cv.getContext('2d');
      g.fillStyle = '#fff'; g.fillRect(0, 0, 512, 512); g.drawImage(b, -(t.c > 0 ? 1 : 0), -(t.r > 0 ? 1 : 0)); b.close();   // drop the 1-px Deep Zoom overlap
      const u8 = new Uint8Array(await (await cv.convertToBlob({type: 'image/jpeg', quality: CFG.Q})).arrayBuffer());
      const ds = 2 ** (maxL - t.L), name = `${P}/${t.mag}_${t.stratum}_${String(i).padStart(2, '0')}.jpg`;
      return {name, u8, meta: {name, mag: t.mag, stratum: t.stratum, level: t.L, col: t.c, row: t.r, downsample: ds,
        x0: t.c * 512 * ds, y0: t.r * 512 * ds, size0: 512 * ds, thumb_tissue_frac: t.frac, thumb_hdens: t.h, jpg_bytes: u8.length}}; }));
    for (const r of res) files.push([r.name, r.u8]); rec.tiles = res.map(r => r.meta);
    files.push([`${P}/meta.json`, new TextEncoder().encode(JSON.stringify(rec))]);
    return {rec, files};
  }

  // ---------------- minimal STORE-only zip writer ----------------
  const crcT = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  const crc32 = u8 => { let c = 0xFFFFFFFF; for (let i = 0; i < u8.length; i++) c = crcT[(c ^ u8[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
  class Zip { constructor() { this.ch = []; this.cd = []; this.off = 0; this.n = 0; }
    add(name, u8) { const nb = new TextEncoder().encode(name), crc = crc32(u8);
      const lh = new DataView(new ArrayBuffer(30)); lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(12, 0x21, true);
      lh.setUint32(14, crc, true); lh.setUint32(18, u8.length, true); lh.setUint32(22, u8.length, true); lh.setUint16(26, nb.length, true);
      this.ch.push(lh.buffer, nb, u8);
      const c = new DataView(new ArrayBuffer(46)); c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true);
      c.setUint16(14, 0x21, true); c.setUint32(16, crc, true); c.setUint32(20, u8.length, true); c.setUint32(24, u8.length, true);
      c.setUint16(28, nb.length, true); c.setUint32(42, this.off, true);
      this.cd.push(c.buffer, nb); this.off += 30 + nb.length + u8.length; this.n++; }
    finish() { const cds = this.cd.reduce((a, b) => a + b.byteLength, 0), e = new DataView(new ArrayBuffer(22));
      e.setUint32(0, 0x06054b50, true); e.setUint16(8, this.n, true); e.setUint16(10, this.n, true); e.setUint32(12, cds, true); e.setUint32(16, this.off, true);
      return new Blob([...this.ch, ...this.cd, e.buffer], {type: 'application/zip'}); } }

  // ---------------- run ----------------
  const MP = window.MP = {state: 'running', t0: Date.now(), done: 0, failed: [], parts: [], partUrls: {}, stop: false};
  const S = selected.slice(), rnd0 = mulberry32(CFG.SEED);
  for (let i = S.length - 1; i > 0; i--) { const j = Math.floor(rnd0() * (i + 1)); [S[i], S[j]] = [S[j], S[i]]; }
  const manifestBytes = new TextEncoder().encode(JSON.stringify({selected, cases, all_dx_files: files, config: CFG, order: S.map(s => s.patient)}));
  let zip = new Zip(), inPart = 0, partNo = 1, recs = [];
  function flush() { if (!inPart) return;
    zip.add('part_manifest.json', new TextEncoder().encode(JSON.stringify({part: partNo, slides: recs})));
    zip.add('gdc_manifest.json', manifestBytes);
    const blob = zip.finish(), name = 'mechanopath_gdc_tiles_part' + String(partNo).padStart(2, '0') + '.zip', url = URL.createObjectURL(blob);
    MP.partUrls[name] = url; MP.parts.push({name, slides: inPart, bytes: blob.size});
    const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    partNo++; zip = new Zip(); inPart = 0; recs = []; }
  function addResult(res) { for (const [n, u8] of res.files) zip.add('tiles/' + n, u8); recs.push(res.rec); inPart++; MP.done++;
    if (inPart >= (partNo === 1 ? CFG.FIRST_PART : CFG.PER_PART)) flush(); }
  let idx = 0;
  async function worker() { while (idx < S.length && !MP.stop) { const s = S[idx++];
    try { addResult(await processSlide(s)); } catch (e) { MP.failed.push({patient: s.patient, file_id: s.file_id, err: String(e).slice(0, 100)}); } } }
  await Promise.all(Array.from({length: CFG.SLIDE_CONC}, worker));
  for (const f of MP.failed.splice(0)) { const s = S.find(x => x.file_id === f.file_id);
    try { addResult(await processSlide(s)); } catch (e) { MP.failed.push({...f, err2: String(e).slice(0, 100)}); } }
  flush(); MP.state = 'finished';
  console.log('MechanoPath harvest finished', MP.done, 'slides;', MP.failed.length, 'failed');
})();
