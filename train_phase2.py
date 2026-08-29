"""Phase-2 training: ranking CE (alpha) + frozen-LM CE (beta) on verbalized cut-point examples (spec 5)."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from bitsandbytes.optim import PagedAdamW8bit
from transformers import get_cosine_schedule_with_warmup

import eval as eval_lib
from model import GenRecModel, genrec_scorer, load_backbone, load_tokenizer
from verbalize import verbalize


def build_examples(args, tok, catalog_map, users):
    """(ids, target_item) per train cut; per-user most recent --max-per-user cuts (base config)."""
    eos = tok.eos_token_id
    fit = lambda p: len(tok(p, add_special_tokens=False).input_ids)
    cuts_by_user = {}
    for line in open(Path(args.data_dir) / "train_cuts.jsonl"):
        c = json.loads(line)
        cuts_by_user.setdefault(c["user_id"], []).append(c["cut_idx"])  # ascending by construction
    examples = []
    for u, cut_list in sorted(cuts_by_user.items()):
        for k in cut_list[-args.max_per_user:]:
            prompt = verbalize(users[u]["events"][:k], catalog_map, n_events=args.n_events,
                               verbosity=args.verbosity, drop_low_signal=args.drop_low_signal,
                               fit_tokens=fit, max_len=args.max_len)
            ids = (tok(prompt).input_ids + [eos])[-args.max_len:]
            examples.append((ids, users[u]["events"][k]["i"]))
    if args.max_examples:
        examples = examples[:args.max_examples]
    return examples


def collate(batch, pad_id, device):
    L = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    tgt = torch.zeros(len(batch), dtype=torch.long)
    for r, (x, t) in enumerate(batch):
        ids[r, :len(x)] = torch.tensor(x)
        mask[r, :len(x)] = 1
        tgt[r] = t
    return ids.to(device), mask.to(device), tgt.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="runs/phase2_base")
    ap.add_argument("--n-events", type=int, default=10)
    ap.add_argument("--verbosity", choices=["full", "compact"], default="full")
    ap.add_argument("--drop-low-signal", action="store_true")
    ap.add_argument("--max-per-user", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2, help="per-step batch (x --accum = effective 16)")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--val-users", type=int, default=1, help="1=all, 0=skip final val eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="load last ckpt in --out and continue")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    catalog_map = {c["item_id"]: c for c in map(json.loads, open(Path(args.data_dir) / "catalog.jsonl"))}
    users = {u["user_id"]: u for u in map(json.loads, open(Path(args.data_dir) / "events.jsonl"))}

    print("building examples...")
    examples = build_examples(args, tok, catalog_map, users)
    print(f"examples={len(examples)}")
    n_items = len(catalog_map)

    start_step = 0
    if args.resume and (Path(args.out) / "head.pt").exists():
        model = GenRecModel.from_ckpt(args.out, trainable=True)[0].to(device).train()
        start_step = torch.load(Path(args.out) / "head.pt", map_location="cpu",
                                weights_only=True)["step"]
        print(f"resumed from {args.out} at step {start_step}")
    else:
        model = GenRecModel(load_backbone(), n_items).to(device)
    params = [p for p in model.backbone.parameters() if p.requires_grad] + list(model.head.parameters())
    opt = PagedAdamW8bit(params, lr=args.lr)
    steps_per_epoch = (len(examples) + args.batch - 1) // args.batch
    total_steps = ((steps_per_epoch + args.accum - 1) // args.accum) * args.epochs  # optimizer steps, not micro-batches
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
    print(f"total optimizer steps={total_steps} (effective batch={args.batch * args.accum})")

    step, t0, running = start_step, time.time(), {"rank": 0.0, "lm": 0.0, "n": 0}
    for _ in range(start_step):
        sched.step()  # fast-forward cosine schedule on resume
    skip_micro = start_step * args.accum
    micro = 0
    out_dir = Path(args.out)
    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        rng.shuffle(examples)
        for i in range(0, len(examples), args.batch):
            micro += 1
            if micro <= skip_micro:  # resume: replay past micro-batches in same order
                continue
            ids, mask, tgt = collate(examples[i:i + args.batch], tok.pad_token_id, device)
            out = model(ids, mask)
            rank_loss = F.cross_entropy(out["rank_logits"], tgt)
            loss = args.alpha * rank_loss
            if args.beta > 0:
                lm_loss = model.lm_loss(out["h_all"], ids, mask)
                loss = loss + args.beta * lm_loss
            (loss / args.accum).backward()
            running["rank"] += rank_loss.item(); running["lm"] += float(lm_loss) if args.beta > 0 else 0.0
            running["n"] += 1
            last_chunk = i + args.batch >= len(examples)
            if (i // args.batch) % args.accum == args.accum - 1 or last_chunk:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    n = max(running["n"], 1)
                    print(f"epoch {epoch} step {step}/{total_steps} rank={running['rank']/n:.4f} "
                          f"lm={running['lm']/n:.4f} lr={sched.get_last_lr()[0]:.2e} "
                          f"{(time.time()-t0)/step:.2f}s/step", flush=True)
                    running = {"rank": 0.0, "lm": 0.0, "n": 0}
                if step % 500 == 0:
                    model.save(out_dir, step)
                    print(f"ckpt saved at step {step}", flush=True)
        model.save(out_dir, step)
        print(f"epoch {epoch} done, ckpt saved", flush=True)

    model.save(out_dir, step)
    print("training done; final ckpt in", out_dir)

    if args.val_users > 0:
        val_users = list(users.values())
        if args.val_users > 1:
            val_users = val_users[:args.val_users]
        m = eval_lib.evaluate(genrec_scorer(model, tok, catalog_map, args.n_events,
                                            args.verbosity, args.drop_low_signal, args.max_len),
                              val_users, "val")
        print(json.dumps({"scorer": "genrec_val", **m}, indent=2))


if __name__ == "__main__":
    main()
