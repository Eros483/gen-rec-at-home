"""ML-1M -> catalog.jsonl (3706 items), events.jsonl (per-user sequences + val/test idx), train_cuts.jsonl (positive-target cut-points).

Split rule (spec 4.1): last rating>=4 event = test, previous = val, earlier positives = train cut-points.
"""
import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
POS_MIN = 4


def download(data_dir: Path) -> Path:
    raw = data_dir / "ml-1m"
    if (raw / "ratings.dat").exists():
        return raw
    zip_path = data_dir / "ml-1m.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(BASE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    raw = download(data_dir)

    ratings = []
    for line in (raw / "ratings.dat").read_text().splitlines():
        u, i, r, t = line.split("::")
        ratings.append((int(u), int(i), int(r), int(t)))

    meta = {}
    for line in (raw / "movies.dat").read_text(encoding="latin-1").splitlines():
        i, title, genres = line.split("::")
        meta[int(i)] = (title, genres.split("|"))

    # contiguous item ids over rated items only (head is a softmax over these)
    ml_ids = sorted({i for _, i, _, _ in ratings})
    id_map = {ml: k for k, ml in enumerate(ml_ids)}

    with open(data_dir / "catalog.jsonl", "w") as f:
        for ml in ml_ids:
            title, genres = meta.get(ml, (str(ml), []))
            m = re.search(r"\((\d{4})\)\s*$", title)
            f.write(json.dumps({
                "item_id": id_map[ml], "ml_id": ml, "title": title,
                "year": int(m.group(1)) if m else None, "genres": genres,
            }) + "\n")

    # per-user chronological events; (ts, item_id) tiebreak for determinism
    users = {}
    for u, i, r, t in sorted(ratings):
        users.setdefault(u, []).append({"i": id_map[i], "r": r, "t": t})

    n_test = n_val = n_cuts = 0
    with open(data_dir / "events.jsonl", "w") as fe, open(data_dir / "train_cuts.jsonl", "w") as ft:
        for u, evs in sorted(users.items()):
            pos = [k for k, e in enumerate(evs) if e["r"] >= POS_MIN]
            test_idx = pos[-1] if pos else None
            val_idx = pos[-2] if len(pos) > 1 else None
            # train targets must precede val (or test when no val) -> no eval target ever leaks into train
            limit = val_idx if val_idx is not None else (test_idx if test_idx is not None else -1)
            cuts = [k for k in pos if k < limit]
            assert all(k < limit for k in cuts)
            fe.write(json.dumps({"user_id": u, "events": evs,
                                 "val_idx": val_idx, "test_idx": test_idx}) + "\n")
            for k in cuts:
                ft.write(json.dumps({"user_id": u, "cut_idx": k}) + "\n")
            n_test += test_idx is not None
            n_val += val_idx is not None
            n_cuts += len(cuts)

    assert len(users) == 6040, len(users)
    assert len(ml_ids) == 3706, len(ml_ids)
    assert len(ratings) == 1000209, len(ratings)
    print(f"users={len(users)} items={len(ml_ids)} ratings={len(ratings)}")
    print(f"users_with_test={n_test} users_with_val={n_val} train_cuts={n_cuts}")


if __name__ == "__main__":
    main()
