"""Label-masking for TCGA pathology reports (GBM vs lower-grade glioma).

Three levels:
  M0  no masking (shows how much of the task is just reading the diagnosis)
  M1  masks tumour-type names, WHO grade statements, grade adjectives,
      ICD-O morphology codes and molecular findings that define the WHO 2021
      class (IDH, 1p/19q, EGFR, TERT, ATRX, MGMT, CDKN2A, LOH/FISH assays)
  M2  M1 + the histological grading criteria themselves (necrosis,
      microvascular/endothelial proliferation, mitoses, Ki-67/MIB-1 indices,
      percentages, cellularity/atypia/pleomorphism)

The reports are OCR output (e.g. "Recelpt"), so diagnosis nouns are also
matched by edit distance. Each masked span becomes the token "xmaskx".
"""
from __future__ import annotations

import re
from functools import lru_cache

MASK = " xmaskx "

# tumour-type and grade vocabulary (M1)
DX_WORDS = ["glioblastoma", "glioblastomas", "glioblastomatous", "multiforme", "gliosarcoma",
            "astrocytoma", "astrocytomas", "oligodendroglioma", "oligodendrogliomas",
            "oligoastrocytoma", "oligoastrocytomas", "anaplastic", "glioma", "gliomas"]
FUZZY_TARGETS = ["glioblastoma", "multiforme", "gliosarcoma", "astrocytoma",
                 "oligodendroglioma", "oligoastrocytoma", "anaplastic"]
M1_PATTERNS = [
    r"\bgbm(-?o)?s?\b",
    r"\bmalignan\w*\b", r"\boligodendroglial\b", r"\boligodendrocytic\b",
    r"\b(high|low)[\s\-]*grade\b",
    r"\bwell[\s\-]*differentiated\b",
    r"\bmixed\s+glioma\b",
    # WHO grade statements: "WHO grade IV", "grade: 4", "grade IV/IV", "grade II-III", "(WHO III)"
    r"\b(who\s*)?(grade|gr\.)\s*[:\-]?\s*(iv|i{1,3}|[1-4])\b(\s*(-|to|/|of|out of)\s*(iv|i{1,3}|[1-4])\b)?",
    r"\bwho\s*(iv|i{1,3}|[1-4])\b",
    r"\bgrade\b",
    r"\b9\d{3}\s*/\s*[0-3]\b",                       # ICD-O morphology codes (9440/3 etc.)
    r"\bm-?\s?9\d{3}[\-/]?\d\b",                     # SNOMED morphology codes (M-94403 = glioblastoma)
    r"\b(ii|iii|iv)\b", r"\bwho\s*scale\b",            # stray roman-numeral grades after OCR breaks
    # molecular findings
    r"\bidh[\s\-]?[12]?\b", r"\br132h?\b", r"\b1p\b", r"\b19q\b", r"\b1p\s*/\s*19q\b",
    r"\bco[\s\-]?deleti?on\w*\b", r"\bcodel\w*\b", r"\bdeletion\w*\b", r"\bamplif\w*\b",
    r"\bloss\s+of\s+heterozygosity\b", r"\bloh\b", r"\ballelic\b", r"\bheterozygosity\b",
    r"\begfr\w*\b", r"\bviii\b", r"\bmgmt\b", r"\bmethylat\w*\b", r"\btert\b", r"\batrx\b",
    r"\bcdkn2a\b", r"\bp16\b", r"\bfish\b", r"\bd1s\d+\b", r"\bd19s\d+\b", r"\bchromosom\w*\s*(arm\s*)?(7|10|1|19)\w*\b",
]
# grading criteria (added in M2)
M2_PATTERNS = [
    r"\bnecro\w*\b", r"\b(pseudo)?palisad\w*\b", r"\bmicro[\s\-]?vascular\b",
    r"\b(vascular|endothelial|microvascular)\s+(cell\s+)?(proliferation|hyperplasia)\b",
    r"\bglomeruloid\b", r"\bmitos[ie]s\b", r"\bmitotic\w*\b", r"\bmitotically\b", r"\bmitoses\b",
    r"\bki[\s\-]?67\b", r"\bmib[\s\-]?1\b", r"\blabel(l)?ing\s+index\b", r"\bproliferat\w*\b",
    r"\b\d+(\.\d+)?\s*(%|percent)", r"\b(hyper)?cellular\w*\b", r"\bpleomorph\w*\b", r"\batypi\w*\b",
]
# words that are part of a diagnosis name when they directly precede a masked noun
# ("diffuse astrocytoma", "infiltrating glioma", "small cell glioblastoma", "mixed glioma")
DX_MODIFIERS = r"(infiltrating|diffuse|diffusely|mixed|gemistocytic|fibrillary|protoplasmic|pilocytic|" \
               r"recurrent|residual|secondary|primary|small[\s\-]cell|giant[\s\-]cell|epithelioid|granular[\s\-]cell)"
_MOD = re.compile(r"\b" + DX_MODIFIERS + r"\s+(?=xmaskx)", flags=re.I)
_M1 = re.compile("|".join(M1_PATTERNS), flags=re.I)
_NEG_BEFORE = re.compile(r"\b(no|not|without|absent|lack of|lacks|lacking|negative for|none|nor|or)\s+(?=(\w+\s+){0,2}xmaskx)", flags=re.I)
_NEG_AFTER = re.compile(r"(?<=xmaskx)\s*(,\s*)?(is|are|was|were)?\s*(not\s+(seen|identified|present|noted|found)|absent|negative|inconspicuous|rare|increased|present|identified|seen|noted)\b", flags=re.I)
_M2 = re.compile("|".join(M2_PATTERNS), flags=re.I)
_TOKEN = re.compile(r"[a-z][a-z\-]{5,}")


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=200_000)
def is_dx_word(tok: str) -> bool:
    """True if tok is a tumour-type/grade word, allowing OCR errors."""
    t = tok.strip("-")
    if "-" in t:                              # "oligodendroglioma-like", "glioblastoma-type"
        return any(is_dx_word(part) for part in t.split("-") if len(part) >= 6)
    if t in DX_WORDS:
        return True
    for tgt in FUZZY_TARGETS:
        if abs(len(t) - len(tgt)) > 2:
            continue
        k = 2 if len(tgt) >= 10 else 1
        if t[:3] == tgt[:3] and _edit(t, tgt) <= k:
            return True
    return False


def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    return t


def mask(text: str, level: str) -> str:
    t = normalize(text)
    if level == "M0":
        return t
    t = _M1.sub(MASK, t)                      # multi-word patterns first ("mixed glioma")
    t = _TOKEN.sub(lambda m: MASK if is_dx_word(m.group(0)) else m.group(0), t)
    t = re.sub(r"\s+", " ", t)
    for _ in range(3):                        # chains such as "diffusely infiltrating xmaskx"
        t = _MOD.sub(MASK.strip() + " ", t)
    t = re.sub(r"xmaskx\s*\.?\s*(of|/|out of)\s*\.?\s*(iv|4|four)\b", "xmaskx", t)   # "grade 2 of 4"
    if level == "M2":
        t = _M2.sub(MASK, t)
        t = re.sub(r"\s+", " ", t)
        # negated criteria ("no xmaskx", "xmaskx is not seen") still reveal the criterion
        t = _NEG_BEFORE.sub(MASK.strip() + " ", t)
        t = _NEG_AFTER.sub(" " + MASK.strip(), t)
    t = re.sub(r"\ban\b", "a", t)            # article before a masked noun would reveal its first letter
    return re.sub(r"\s+", " ", t)


def residual_dx_terms(text: str) -> list[str]:
    """Diagnosis words still present after masking (should be empty)."""
    return [m.group(0) for m in _TOKEN.finditer(text) if is_dx_word(m.group(0))]
