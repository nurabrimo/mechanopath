#!/usr/bin/env bash
# Builds report/MechanoPath_full_report.docx from the result files, in two passes so that the contents list
# carries page numbers. Needs python3 (pandas, statsmodels, pillow), node with the docx package
# (npm install docx), LibreOffice and poppler (pdftotext).
#   SOFFICE: LibreOffice binary, e.g. /Applications/LibreOffice.app/Contents/MacOS/soffice on macOS
set -euo pipefail
cd "$(dirname "$0")/.."
SOFFICE=${SOFFICE:-soffice}
DOCX=report/MechanoPath_full_report.docx
PDF=report/render/MechanoPath_full_report.pdf
mkdir -p report/render

python3 report/build_content.py                       # report/blocks.json: text, tables, figures, code
rm -f report/toc_pages.json
node report/render_docx.js "$DOCX"                    # pass 1: placeholder page numbers
python3 report/fix_bookmarks.py "$DOCX"               # unique bookmark ids for the contents links
$SOFFICE --headless --convert-to pdf --outdir report/render "$DOCX" > /dev/null
python3 report/toc_pages.py "$PDF"                    # page of every heading
node report/render_docx.js "$DOCX"                    # pass 2: real page numbers
python3 report/fix_bookmarks.py "$DOCX"
$SOFFICE --headless --convert-to pdf --outdir report/render "$DOCX" > /dev/null
python3 report/toc_pages.py "$PDF" --check            # numbers did not move between passes
