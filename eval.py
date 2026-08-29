"""Shared eval harness: full-catalog ranking with seen-item masking -> MRR / NDCG@10 / HR@10.

A scorer is any function(history_item_ids, target_item_id) -> np.ndarray of scores over the catalog.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def load_all(data_dir):
    data_dir = Path(data_dir)
    catalog = [json.loads(l) for l in open(data_dir / "catalog.jsonl")]
    users = [json.loads(l) for l in open(data_dir / "events.jsonl")]
    return catalog, users


def rank_metrics(scores, target, seen_mask):
    """scores: [n_items]; returns (mrr, ndcg@10, hr@10) with seen items masked out."""
    scores = np.asarray(scores, dtype=np.float64).copy()
    scores[seen_mask] = -np.inf
    order = np.argsort(-scores, kind="stable")
    rank = int(np.where(order == target)[0][0]) + 1
    mrr = 1.0 / rank
    ndcg = 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0
    hr = 1.0 if rank <= 10 else 0.0
    return mrr, ndcg, hr


def evaluate(score_fn, users, split="test"):
    """score_fn(history_item_ids) -> scores over catalog. Returns mean metrics."""
    idx_key = f"{split}_idx"
    rows = []
    for u in users:
        idx = u[idx_key]
        if idx is None:
            continue
        hist = u["events"][:idx]
        target = u["events"][idx]["i"]
        rows.append((hist, target))
    assert rows, f"no users with a {split} event"

    mrrs, ndcgs, hrs = [], [], []
    for hist, target in rows:
        hist_ids = [e["i"] for e in hist]
        scores = score_fn(hist_ids, hist)
        seen_mask = np.zeros(scores.shape[0], dtype=bool)
        seen_mask[np.asarray(hist_ids, dtype=int)] = True
        m, n, h = rank_metrics(scores, target, seen_mask)
        mrrs.append(m); ndcgs.append(n); hrs.append(h)
    return {"split": split, "n_users": len(rows),
            "MRR": float(np.mean(mrrs)), "NDCG@10": float(np.mean(ndcgs)), "HR@10": float(np.mean(hrs))}


def popularity_scorer(data_dir, n_items):
    """Score by count of positive (rating>=4) targets in train_cuts."""
    counts = Counter()
    users = {u["user_id"]: u for u in map(json.loads, open(Path(data_dir) / "events.jsonl"))}
    for line in open(Path(data_dir) / "train_cuts.jsonl"):
        c = json.loads(line)
        counts[users[c["user_id"]]["events"][c["cut_idx"]]["i"]] += 1
    pop = np.zeros(n_items)
    for item, c in counts.items():
        pop[item] = c
    return lambda hist_ids, hist_events: pop


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["popularity"], default="popularity")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    catalog, users = load_all(args.data_dir)
    n_items = len(catalog)
    if args.scorer == "popularity":
        score_fn = popularity_scorer(args.data_dir, n_items)
    else:
        raise SystemExit("sasrec/genrec scorers are wired in their own modules")
    m = evaluate(score_fn, users, args.split)
    print(json.dumps({"scorer": args.scorer, **m}, indent=2))
