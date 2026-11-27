"""docx-js writes w:id="1" on every bookmark, and Word needs unique ids. This renumbers the bookmarkStart /
bookmarkEnd pairs of the contents links in order (they never nest) and rewrites the .docx in place.

    python report/fix_bookmarks.py report/MechanoPath_full_report.docx
"""
import re
import sys
import zipfile
from pathlib import Path

path = Path(sys.argv[1])
with zipfile.ZipFile(path) as z:
    items = [(info, z.read(info.filename)) for info in z.infolist()]

n = [0]


def renumber(m):
    if m.group(1) == "Start":
        n[0] += 1
    return re.sub(r'w:id="\d+"', f'w:id="{n[0]}"', m.group(0))


tmp = path.with_suffix(".tmp")
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
    for info, data in items:
        if info.filename == "word/document.xml":
            data = re.sub(r"<w:bookmark(Start|End)\b[^>]*/>", renumber, data.decode("utf-8")).encode("utf-8")
        z.writestr(info, data)
tmp.replace(path)
print(f"renumbered {n[0]} bookmarks in {path.name}")
