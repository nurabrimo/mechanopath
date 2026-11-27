"""Report branch with a fine-tuned biomedical BERT (DCC GPU pipeline).

  python dcc/05_report_bert.py --level M2 --cohort all --scheme institution
  (or, inside a Slurm array, --task-id $SLURM_ARRAY_TASK_ID, which maps to level x cohort x scheme)

Inputs: the masked report texts written by src/01_reports.py (data/processed/reports_masked.csv) and the
shared patient splits. Reports are split into 512-token windows (stride 384, at most 8 per report); every
window carries the patient label during training and the patient probability is the mean of its windows.
Model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext (BiomedBERT, MIT licence), the mask
token "xmaskx" added to the vocabulary. AdamW (lr 2e-5, weight decay 0.01), linear warm-up (10%) and decay,
4 epochs, batch 16, bf16 autocast (fp16 with loss scaling on pre-Ampere GPUs); the epoch with the best AUROC on a 10% validation split of the training
fold is kept. Nothing from the test fold is used for fitting or model selection.

Output: results/bert/report_oof_bert_<level>_<cohort>_<scheme>.csv (same columns as report_oof.csv) and a
per-fold log.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import PROC, RES, SEED, iter_folds, youden_threshold  # noqa: E402

TASKS = [(lv, c, s) for lv in ("M2", "M1") for c in ("all", "x40") for s in ("institution", "patient")]
ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=["M0", "M1", "M2"])
ap.add_argument("--cohort", choices=["all", "x40"])
ap.add_argument("--scheme", choices=["patient", "institution"])
ap.add_argument("--task-id", type=int, default=None)
ap.add_argument("--model", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-windows", type=int, default=8)
ap.add_argument("--repeats", type=int, default=5)
a = ap.parse_args()
if a.task_id is not None:
    a.level, a.cohort, a.scheme = TASKS[a.task_id]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bf16 on Ampere or newer; fp16 with loss scaling on older GPUs (bf16 is only emulated there)
AMP = torch.bfloat16 if dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
# fused SDPA kernels fail with illegal memory access in fp16 on pre-Ampere GPUs (torch 2.6): plain attention there
ATTN = "sdpa" if dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else "eager"
if ATTN == "eager" and dev.type == "cuda":
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
out_dir = RES / "bert"
out_dir.mkdir(parents=True, exist_ok=True)
tag = f"{a.level}_{a.cohort}_{a.scheme}"
print(f"BERT task {tag} on {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}, autocast {AMP}, "
      f"attention {ATTN})", flush=True)

cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
splits = pd.read_csv(PROC / "splits.csv")
texts = pd.read_csv(PROC / "reports_masked.csv").set_index("patient")[f"text_{a.level}"]
slides = pd.read_csv(PROC / "slides.csv").set_index("patient")
tok = AutoTokenizer.from_pretrained(a.model)
if "xmaskx" not in tok.get_vocab():
    tok.add_tokens(["xmaskx"])


def windows(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out, step, L = [], 384, 510
    for s in range(0, max(1, len(ids)), step):
        out.append([tok.cls_token_id] + ids[s:s + L] + [tok.sep_token_id])
        if s + L >= len(ids) or len(out) >= a.max_windows:
            break
    return out


W = {p: windows(t) for p, t in texts.items()}
print(f"{len(W)} reports | windows per report median {int(np.median([len(v) for v in W.values()]))}", flush=True)


def batchify(seqs):
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), tok.pad_token_id)
    att = torch.zeros((len(seqs), m), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids.to(dev), att.to(dev)


@torch.no_grad()
def predict(model, pats):
    model.eval()
    out = []
    for p in pats:
        ids, att = batchify(W[p])
        with torch.autocast(dev.type, dtype=AMP, enabled=dev.type == "cuda"):
            logit = model(input_ids=ids, attention_mask=att).logits[:, 0].float()
        out.append(torch.sigmoid(logit).mean().item())
    return np.array(out)


MICRO = 16 if dev.type != "cuda" or torch.cuda.get_device_properties(0).total_memory > 20e9 else 8


def fit(tr, te, seed):
    torch.manual_seed(seed)
    ytr = cohort.loc[tr, "y_gbm"]
    tr_a, va_a = train_test_split(np.array(tr), test_size=0.1, stratify=ytr, random_state=seed)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, attn_implementation=ATTN)
    model.resize_token_embeddings(len(tok))
    model.to(dev)
    items = [(s, float(cohort.loc[p, "y_gbm"])) for p in tr_a for s in W[p]]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * int(np.ceil(len(items) / 16))
    warm = max(1, int(0.1 * steps))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    lossf = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and AMP == torch.float16)
    best, state, best_ep = -1.0, None, 0
    for ep in range(a.epochs):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(len(items))
        for i in range(0, len(order), 16):                     # effective batch 16 (2 x 8 on GPUs < 20 GB)
            opt.zero_grad(set_to_none=True)
            chunk = [items[j] for j in order[i:i + 16]]
            for c0 in range(0, len(chunk), MICRO):
                b = chunk[c0:c0 + MICRO]
                ids, att = batchify([s for s, _ in b])
                with torch.autocast(dev.type, dtype=AMP, enabled=dev.type == "cuda"):
                    logit = model(input_ids=ids, attention_mask=att).logits[:, 0]
                loss = lossf(logit.float(), torch.tensor([l for _, l in b], device=dev)) * len(b) / len(chunk)
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
        auc = roc_auc_score(cohort.loc[va_a, "y_gbm"], predict(model, va_a))
        if auc > best:
            best, best_ep = auc, ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    p_te, p_tr = predict(model, te), predict(model, tr)
    del model
    torch.cuda.empty_cache()
    return p_te, p_tr, {"val_auroc": best, "best_epoch": best_ep}


x40 = set(slides.index[slides.appmag == 40])
cids = [p for p in texts.index if p in cohort.index and (a.cohort == "all" or p in x40)]
fold_file, log_file = out_dir / f"report_folds_bert_{tag}.csv", out_dir / f"report_fold_log_bert_{tag}.csv"
preds, log = [], []
if fold_file.exists():                                    # resume after pre-emption: keep finished folds
    preds = [pd.read_csv(fold_file)]
    log = pd.read_csv(log_file).to_dict("records") if log_file.exists() else []
done = {(int(r), int(k)) for r, k in pd.concat(preds)[["repeat", "fold"]].drop_duplicates().values} if preds else set()
print(f"{len(cids)} patients | {len(done)} folds already finished", flush=True)


def write_outputs():
    P = pd.concat(preds)
    P.to_csv(fold_file, index=False)
    pd.DataFrame(log).to_csv(log_file, index=False)
    P = P.assign(yhat=(P.p >= P.thr).astype(int))
    agg = P.groupby("patient").agg(p=("p", "mean"), yhat=("yhat", lambda s: int(s.mean() >= 0.5)),
                                   n_repeats=("p", "size")).reset_index()
    agg["model"], agg["scheme"], agg["cohort"] = f"bert_{a.level}", a.scheme, a.cohort
    agg.to_csv(out_dir / f"report_oof_bert_{tag}.csv", index=False)
    return agg


t_start = time.time()
for r, k, tr, te in iter_folds(splits, cids, a.scheme):
    if r >= a.repeats or (r, k) in done:
        continue
    t0 = time.time()
    p_te, p_tr, info = fit(tr, te, seed=SEED + 100 * r + k)
    preds.append(pd.DataFrame({"patient": te, "repeat": r, "fold": k, "p": p_te,
                               "thr": youden_threshold(cohort.loc[tr, "y_gbm"], p_tr)}))
    yte = cohort.loc[te, "y_gbm"]
    log.append({"task": tag, "repeat": r, "fold": k, "n_train": len(tr), "n_test": len(te),
                "test_auroc": roc_auc_score(yte, p_te) if yte.nunique() == 2 else np.nan,
                "seconds": time.time() - t0, **info})
    print(f"{tag} r{r} k{k} test AUROC {log[-1]['test_auroc']:.3f} val {info['val_auroc']:.3f} "
          f"({log[-1]['seconds']:.0f}s, {(time.time() - t_start) / 60:.0f} min total)", flush=True)
    write_outputs()
agg = write_outputs()
yy = cohort.loc[agg.patient, "y_gbm"]
print(f"BERT_DONE {tag}: AUROC {roc_auc_score(yy, agg.p):.3f} over {len(agg)} patients in "
      f"{(time.time() - t_start) / 60:.0f} min", flush=True)
