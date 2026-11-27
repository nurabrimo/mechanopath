// Renders report/blocks.json (built by report/build_content.py) into MechanoPath_full_report.docx.
// Run through report/build_report.sh: pass 1 prints placeholder page numbers in the contents list, the PDF of
// pass 1 gives every heading's page (report/toc_pages.py -> toc_pages.json), pass 2 prints the real numbers.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, PageBreak, Footer, Header, PageNumber, LevelFormat, Tab, TabStopType,
  LeaderType, InternalHyperlink, Bookmark,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const blocks = JSON.parse(fs.readFileSync(path.join(__dirname, "blocks.json"), "utf8"));
const OUT = process.argv[2] || path.join(ROOT, "MechanoPath_full_report.docx");
const PAGES = path.join(__dirname, "toc_pages.json");
const tocPages = fs.existsSync(PAGES) ? JSON.parse(fs.readFileSync(PAGES, "utf8")) : null;

const FONT = "Arial";
const MONO = "Courier New";
const CONTENT_W = 9360;            // US Letter, 1" margins, in DXA
const INK = "1A1A1A", INK2 = "52514E", MUTED = "6E6C66", RULE = "C3C2B7", HEAD_FILL = "ECEBE6", CODE_FILL = "F4F3EF";

// ---------- inline markup: **bold**, `code`
function runs(text, opts = {}) {
  const parts = String(text).split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter((s) => s.length);
  return parts.map((s) => {
    if (s.startsWith("**") && s.endsWith("**")) return new TextRun({ text: s.slice(2, -2), bold: true, font: FONT, ...opts });
    if (s.startsWith("`") && s.endsWith("`")) return new TextRun({ text: s.slice(1, -1), font: MONO, ...opts, size: (opts.size || 20) - 2 });
    return new TextRun({ text: s, font: FONT, ...opts });
  });
}

let numberedInstance = 0;
let headingNo = 0;                 // h1/h2 in document order = contents entries in order
const children = [];

// Nature-style legend: bold "Figure N | Title." then the legend text
function caption(text) {
  return new Paragraph({ spacing: { before: 60, after: 240, line: 252 }, keepLines: true,
    children: runs(text, { size: 17, color: INK2 }) });
}

function cell(text, widthDxa, { header = false, fs = 8.5, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: widthDxa, type: WidthType.DXA },
    shading: header ? { fill: HEAD_FILL, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ alignment: align, spacing: { before: 0, after: 0 },
      children: runs(text, { size: Math.round(fs * 2), bold: header, color: header ? INK : INK2 }) })],
  });
}

function heading(level, text, inToc) {
  const run = new TextRun({ text });
  const kids = inToc ? [new Bookmark({ id: `toc_${headingNo++}`, children: [run] })] : [run];
  return new Paragraph({ heading: level, pageBreakBefore: level === HeadingLevel.HEADING_1, children: kids });
}

for (const b of blocks) {
  switch (b.t) {
    case "title": {
      children.push(new Paragraph({ spacing: { before: 2400, after: 200 },
        children: [new TextRun({ text: b.title, bold: true, size: 64, font: FONT, color: INK })] }));
      children.push(new Paragraph({ spacing: { after: 480 }, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "2A78D6", space: 12 } },
        children: [new TextRun({ text: b.subtitle, size: 30, font: FONT, color: INK2 })] }));
      for (const l of b.lines) children.push(new Paragraph({ spacing: { after: 120 },
        children: [new TextRun({ text: l, size: 22, font: FONT, color: INK2 })] }));
      children.push(new Paragraph({ children: [new PageBreak()] }));
      break;
    }
    case "toc": {
      // static contents list: entries link to heading bookmarks; page numbers from toc_pages.json
      children.push(new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "Contents", bold: true, size: 32, font: FONT })] }));
      b.entries.forEach((e, i) => {
        const top = e.level === 1;
        const pg = tocPages && tocPages[i] ? String(tocPages[i]) : "000";
        const sz = top ? 21 : 19;
        children.push(new Paragraph({
          tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W, leader: LeaderType.DOT }],
          indent: { left: top ? 0 : 360 },
          spacing: { before: top ? 120 : 0, after: 30 },
          keepNext: top,
          children: [new InternalHyperlink({ anchor: `toc_${i}`, children: [
            new TextRun({ text: e.text, bold: top, size: sz, font: FONT, color: INK }),
            new TextRun({ children: [new Tab(), pg], bold: top, size: sz, font: FONT, color: INK }),
          ] })],
        }));
      });
      break;
    }
    case "h1":
      children.push(heading(HeadingLevel.HEADING_1, b.text, true));
      break;
    case "h2":
      children.push(heading(HeadingLevel.HEADING_2, b.text, true));
      break;
    case "h3":
      children.push(heading(HeadingLevel.HEADING_3, b.text, false));
      break;
    case "p":
      children.push(new Paragraph({ spacing: { after: 140, line: 276 }, children: runs(b.text, { size: 20, color: INK }) }));
      break;
    case "bullets":
      for (const it of b.items) children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 },
        spacing: { after: 80, line: 264 }, children: runs(it, { size: 20, color: INK }) }));
      children.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
      break;
    case "numbered":
      numberedInstance += 1;
      for (const it of b.items) children.push(new Paragraph({ numbering: { reference: "numbers", level: 0, instance: numberedInstance },
        spacing: { after: 80, line: 264 }, children: runs(it, { size: 20, color: INK }) }));
      children.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
      break;
    case "pagebreak":
      break;                         // every H1 starts on a new page (pageBreakBefore), so no explicit breaks
    case "img": {
      const wpx = Math.round(b.w_in * 96);
      const hpx = Math.round(wpx * b.h / b.w);
      const type = b.path.toLowerCase().endsWith(".png") ? "png" : "jpg";
      const alt = (b.alt || b.caption).replace(/\*\*/g, "");
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 }, keepNext: true,
        children: [new ImageRun({ type, data: fs.readFileSync(b.path), transformation: { width: wpx, height: hpx },
          altText: { title: alt.slice(0, 60), description: alt, name: path.basename(b.path) } })] }));
      children.push(caption(b.caption));
      break;
    }
    case "table": {
      const tot = b.widths.reduce((a, c) => a + c, 0);
      const cols = b.widths.map((w) => Math.floor(CONTENT_W * w / tot));
      cols[cols.length - 1] += CONTENT_W - cols.reduce((a, c) => a + c, 0);
      const num = new Set(b.num || []);
      children.push(new Paragraph({ spacing: { before: 160, after: 80 }, keepNext: true,
        children: [new TextRun({ text: b.caption, bold: true, size: 17, color: INK2, font: FONT })] }));
      const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
      const rows = [new TableRow({ tableHeader: true, cantSplit: true,
        children: b.header.map((h, i) => cell(h, cols[i], { header: true, fs: b.fs })) })];
      for (const r of b.rows) rows.push(new TableRow({ cantSplit: true,
        children: r.map((v, i) => cell(v, cols[i], { fs: b.fs, align: num.has(i) ? AlignmentType.RIGHT : AlignmentType.LEFT })) }));
      children.push(new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: cols, rows,
        borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border } }));
      children.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
      break;
    }
    case "code": {
      const pt = b.size || 7.5;
      const sz = Math.round(pt * 2);
      const line = Math.round(pt * 1.2 * 20);       // exact line height (twips), 1.2 x font size
      if (b.title) children.push(new Paragraph({ children: [new TextRun({ text: b.title, bold: true, size: 18, font: FONT })] }));
      b.lines.forEach((ln, i) => {
        children.push(new Paragraph({
          spacing: { before: i === 0 ? 80 : 0, after: i === b.lines.length - 1 ? 180 : 0, line, lineRule: "exact" },
          shading: { fill: CODE_FILL, type: ShadingType.CLEAR, color: "auto" },
          indent: { left: 120, right: 120 },
          children: [new TextRun({ text: ln.length ? ln.replace(/\t/g, "    ") : " ", font: MONO, size: sz, color: "222222" })],
        }));
      });
      break;
    }
    default:
      throw new Error("unknown block " + b.t);
  }
}

const doc = new Document({
  creator: "Nura Brimo",
  title: "MechanoPath: real-data analysis, methods, results and reproducibility guide",
  description: "MechanoPath technical report",
  styles: {
    default: { document: { run: { font: FONT, size: 20, color: INK } } },
    paragraphStyles: [
      // page numbers take their size and colour from the style (field results ignore run formatting in LibreOffice)
      { id: "Footer", name: "footer", basedOn: "Normal", run: { size: 16, color: MUTED, font: FONT } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: INK },
        paragraph: { spacing: { before: 0, after: 200 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: "2A2A2A" },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: "2A2A2A" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "MechanoPath · real-data analysis report", size: 16, color: MUTED, font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ style: "Footer", alignment: AlignmentType.CENTER,
      children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES].map((c) =>
        new TextRun({ children: [c], size: 16, color: MUTED, font: FONT })) })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, (buf.length / 1e6).toFixed(1), "MB;", tocPages ? "contents with page numbers" : "contents with placeholders");
});
