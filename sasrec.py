"""SASRec baseline from scratch (spec: d=64, 2 layers, maxlen 200, full softmax).

Left-padding so the query state is always h[:, -1]. Item ids are shifted +1 with 0 = pad;
output projection reuses the item embedding (tied, standard SASRec).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import eval as eval_lib


class SASRec(nn.Module):
    def __init__(self, n_items, d=64, n_layers=2, maxlen=200, dropout=0.2):
        super().__init__()
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d)

    def forward(self, ids):  # ids: [B, L] left-padded, values in 1..n_items, 0 = pad
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device)
        x = self.drop(self.item_emb(ids) + self.pos_emb(pos))
        pad_mask = ids == 0
        causal = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool), diagonal=1)
        mask = causal.unsqueeze(0) | pad_mask[:, None, :]
        mask[:, torch.arange(L, device=ids.device), torch.arange(L, device=ids.device)] = False  # pads attend to self -> no NaN
        mask = mask.unsqueeze(1).expand(B, self.encoder.layers[0].self_attn.num_heads, L, L).reshape(B * self.encoder.layers[0].self_attn.num_heads, L, L)
        h = self.encoder(x, mask=mask)
        return self.ln(h)  # [B, L, d]

    def logits_last(self, ids):
        h = self.forward(ids)[:, -1]  # left-padded -> last position is the query state
        return h @ self.item_emb.weight[1:].T  # [B, n_items]

    def scores_for_histories(self, hists, n_items, device):
        """hists: list of item-id lists (chronological). Returns [N, n_items] numpy."""
        seqs = np.zeros((len(hists), self.maxlen), dtype=np.int64)
        for r, h in enumerate(hists):
            t = np.asarray(h[-self.maxlen:], dtype=np.int64) + 1
            seqs[r, self.maxlen - len(t):] = t
        out = np.empty((len(hists), n_items), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(seqs), 1024):
                ids = torch.from_numpy(seqs[i:i + 1024]).to(device)
                out[i:i + 1024] = self.logits_last(ids).float().cpu().numpy()
        return out


def load_cuts(data_dir, split="train"):
    data_dir = Path(data_dir)
    users = {u["user_id"]: u for u in map(json.loads, open(data_dir / "events.jsonl"))}
    rows = []
    if split == "train":
        for line in open(data_dir / "train_cuts.jsonl"):
            c = json.loads(line)
            u = users[c["user_id"]]
            rows.append((u["events"][:c["cut_idx"]], u["events"][c["cut_idx"]]["i"]))
    else:
        for u in users.values():
            idx = u[f"{split}_idx"]
            if idx is not None:
                rows.append((u["events"][:idx], u["events"][idx]["i"]))
    return rows


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_items = sum(1 for _ in open(Path(args.data_dir) / "catalog.jsonl"))
    model = SASRec(n_items, args.d, args.n_layers, args.maxlen).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_rows = load_cuts(args.data_dir, "train")
    val_rows = load_cuts(args.data_dir, "val")
    print(f"train rows={len(train_rows)} val rows={len(val_rows)} device={device}")

    rng = np.random.default_rng(0)
    best_ndcg = -1.0
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_rows))
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            seqs = np.zeros((len(idx), args.maxlen), dtype=np.int64)
            targets = np.empty(len(idx), dtype=np.int64)
            for r, j in enumerate(idx):
                hist, tgt = train_rows[j]
                t = np.asarray([e["i"] for e in hist[-args.maxlen:]], dtype=np.int64) + 1
                seqs[r, args.maxlen - len(t):] = t
                targets[r] = tgt
            ids = torch.from_numpy(seqs).to(device)
            tgt = torch.from_numpy(targets).to(device)
            loss = torch.nn.functional.cross_entropy(model.logits_last(ids), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        m = eval_lib.evaluate(lambda hist_ids, hist_events: model.scores_for_histories([hist_ids], n_items, device)[0],
                              val_rows_to_users(val_rows), "val")
        print(f"epoch {epoch}: train_loss={tot / nb:.4f} val_MRR={m['MRR']:.4f} val_NDCG@10={m['NDCG@10']:.4f} val_HR@10={m['HR@10']:.4f}")
        if m["NDCG@10"] > best_ndcg:
            best_ndcg = m["NDCG@10"]
            torch.save(model.state_dict(), out / "best.pt")
    return model


def val_rows_to_users(val_rows):
    return [{"user_id": i, "events": hist + [{"i": tgt, "r": 5, "t": 0}], "val_idx": len(hist),
             "test_idx": None} for i, (hist, tgt) in enumerate(val_rows)]


def score_fn_from_ckpt(ckpt, data_dir, device=None):
    n_items = sum(1 for _ in open(Path(data_dir) / "catalog.jsonl"))
    model = SASRec(n_items)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return lambda hist_ids, hist_events: model.scores_for_histories([hist_ids], n_items, device)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--ckpt", default="runs/sasrec/best.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=200)
    ap.add_argument("--out", default="runs/sasrec")
    args = ap.parse_args()

    if args.cmd == "train":
        train(args)
    else:
        catalog, users = eval_lib.load_all(args.data_dir)
        m = eval_lib.evaluate(score_fn_from_ckpt(args.ckpt, args.data_dir), users, args.split)
        print(json.dumps({"scorer": "sasrec", **m}, indent=2))
