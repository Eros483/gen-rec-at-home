# GenRec-at-Home — Spec

Reproduce the core claims of Netflix's GenRec (arXiv:2608.10257) on consumer hardware with public data.

## 1. Goal / Non-goal

**Replicate:**
- Verbalized-history LLM ranker with catalog-aware head vs. a traditional sequential baseline (SASRec)
- Context-engineering ablation: MRR vs. #history events (their Figure 5 elbow)
- Phase-2 data scaling: MRR vs. dataset size
- Phase-1 (domain continued pretraining) contribution
- Reward-weighted loss (rating-weight and learned-reward variants)

**Skip (Netflix-only):**
- Internal foundation model, online A/B, retention/business reward models, vLLM fleet serving, multi-content-type catalog

## 2. Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | **Qwen3-4B**, QLoRA 4-bit | Strong base capability (verbalized NL history is the whole game); 2.5GB quantized → fits local 8GB and Kaggle T4 |
| Fine-tune method | QLoRA (r=16, α=32, dropout 0.05, all-linear) + **full-rank head** | Head is a new module — LoRA can't adapt what doesn't exist. Head is 15M params, free to train |
| Phase 1 → 2 handoff | Phase-2 LoRA initializes from Phase-1 adapter weights | Don't merge adapters into 4-bit base (dequantization loss) |
| Dataset | MovieLens-1M (6,040 users / 3,706 movies / 1M ratings) | Full softmax over 3.7K catalog — no negative sampling needed |
| Phase-1 corpus | TMDB / Wikipedia movie plots (~35K films, ~40M tokens) | Catalog *knowledge* is fine to share with eval (Netflix's Phase 1 also sees the catalog). The leak to avoid is user-rating signal — TMDB has none |
| Baseline | SASRec, from scratch (~100 lines, d=64, 2 layers, maxlen 200, full softmax) | RecBole is a heavy dep for one baseline. Plus a 5-line popularity baseline |
| Label | Next item with rating ≥ 4 | Mirrors their "high-value engagement" positives |
| Eval | Leave-one-out per user; full-catalog ranking; MRR, NDCG@10, HR@10 | Standard protocol, matches paper's MRR headline |
| Reward variants | (A) loss weight = rating/5; (B) learned MLP reward model | B is the real experiment, A is a one-line control |

## 3. Hardware plan

| Resource | Use |
|---|---|
| Local 8GB GPU | Data prep, popularity + SASRec baselines, 4B QLoRA smoke tests (short runs), all eval |
| Kaggle 2×T4 (16GB ea., ~30h/week) | Phase-1 continued pretraining, main Phase-2 runs, ablation sweeps |

Throughput estimates (T4, grad checkpointing, bf16 adapters, seq 1024):
- 4B QLoRA: ~8–10K tok/s → base Phase-2 epoch (~45M tok, ~60K examples) ≈ 1.5h

## 4. Data pipeline

### 4.1 Phase 2 examples
- Sort each user's ratings by timestamp.
- Cut-points: base config = up to 10 per user (~60K examples); full = every position (~1M examples, for the max-data run).
- Example = (verbalized context+history, target item id).
- **Split:** last rating≥4 event = test, previous rating≥4 event = val, all earlier = train cut-points. No test cut-point ever appears in training.

### 4.2 Verbalization template (lightly structured text)

```
User profile: tenure 214 ratings since 2000-04.
Context: homepage.

Viewing history (most recent last):
- Watched "Toy Story (1995)" [Animation, Children's, Comedy] — rated 5
- Watched "Heat (1995)" [Action, Crime, Thriller] — rated 4
- ... (N events)
- Started "Speed (1994)" [Action, Romance, Thriller] — rated 2

Task: rank the catalog for this user.
```

Pooling position: hidden state at the final EOS after the "Task:" line.

### 4.3 Context engineering rules (the ablation surface)
- N events ∈ {5, 10, 20, 50, 100}, oldest dropped first, max_len 1024 tokens
- Low-signal events (rating ≤ 2) included/excluded as a variant
- Per-event verbosity: full (title+year+genres+rating) vs. compact (title+rating)

### 4.4 Phase 1 corpus
- TMDB/Wikipedia plot summaries + metadata, formatted as plain documents
- Overlap with ML-1M titles is expected and acceptable (catalog knowledge ≠ label leakage)
- 1 epoch, plain LM loss, QLoRA

## 5. Model

```
prompt → Qwen3-4B (4-bit, frozen) + LoRA(r=16, all-linear)
      → h = hidden state at EOS position            [d = 2560]
      → scores = h · Eᵀ + b      E ∈ R^{3706×d}, full-rank, bf16
      → softmax over full catalog → CE loss on target item
```

Loss: `L = α·L_rank + β·L_lm`, default α=1, β=0.1 (LM loss over prompt tokens keeps verbalization fluency / steering ability; ablate β=0).
Reward variants scale `L_rank` per-example by weight (A: rating/5, B: reward-model output).

Optimizer: paged_adamw_8bit, lr 2e-4, cosine, 3% warmup, 2 epochs (base config), batch 16 via grad accum, grad checkpointing on.

## 6. Reward model

The reward model is a small MLP that learns to score how "valuable" a (user, item) engagement is — a learned proxy for long-term satisfaction, the same role Netflix's separate reward models fill.

**Architecture:**
```
[ user_pooled_h ;  item_embedding e_i ]  →  MLP → predicted_rating
```
- `user_pooled_h`: the LLM's pooled hidden state (same h used by the ranking head)
- `e_i`: the item's row from the catalog embedding table E (the same E the ranking head uses)
- Output: a scalar predicted rating (0–5)

**Training:** MSE (or Huber) on the actual rating, using only the Phase-2 train split. Cheap — ~1h on Kaggle.

**Usage:** once trained, `predicted_rating` becomes the per-example weight in GenRec's ranking loss:
- A 5-star engagement → weight ≈ 1.0 → its ranking loss counts fully
- A 1-star engagement → weight ≈ 0.2 → its ranking loss is down-weighted

**Why bother?** Raw rating is myopic and noisy — a 3-star prestige film might predict long-term retention better than a 5-star popcorn flick, and a learned model can capture that correlation from the engagement features. It mirrors Netflix's setup where reward models estimate long-term satisfaction beyond raw clicks and rebalance across content types. In our MovieLens setting the effect is modest (we only have ratings, not retention signals), but it's a principled variant that tests whether *learned* reweighting beats naive rating weighting.

**Data flow:** `reward_model.py` trains the MLP (one small run) → emits per-example weights → `train_phase2.py` consumes them as loss weights.

## 7. Evaluation

- Full-catalog ranking per test user, masking items already in that user's train history
- Metrics: MRR, NDCG@10, HR@10
- Baselines: popularity, SASRec (same split, same masking)
- Primary comparison: GenRec-4B (base config) vs. SASRec on MRR

## 8. Ablation matrix (priority order)

| # | Ablation | Values | Cost (4B, Kaggle) |
|---|---|---|---|
| 1 | Context length N | 5 / 10 / 20 / 50 / 100 | ~1.5h each at base data |
| 2 | Phase-2 data scale | 1× / 5× | 1.5h / 7h |
| 3 | Phase 1 on/off | TMDB-pretrained adapters vs. scratch LoRA | +2h Phase-1 run |
| 4 | Reward | none / A / B | B needs +1h reward-model training |

Expected outputs: elbow curve (replicates their Fig. 5), monotonic data-scaling curve (their Fig. 4), Phase-1 delta, reward delta.

## 9. Repo layout (flat scripts, no framework)

```
genrec/
  data_prep.py      # ML-1M → cut-point examples → jsonl
  verbalize.py      # example → prompt text (verbosity/N knobs)
  model.py          # QLoRA backbone + full-rank catalog head
  train_phase1.py   # TMDB continued pretraining
  train_phase2.py   # ranking + LM loss, reward hooks
  reward_model.py   # variant B
  eval.py           # full-catalog MRR/NDCG/HR + baselines
  sasrec.py         # from-scratch baseline
```

## 10. Milestones

1. **M1 — data + baselines (local, ~1 day):** pipeline works; popularity + SASRec MRR on the split recorded
2. **M2 — first GenRec run (Kaggle, ~2h):** 4B QLoRA, base config, no Phase 1, no reward → does it beat SASRec MRR?
3. **M3 — core ablations (Kaggle, ~2 weeks of quota):** context sweep + data scaling + Phase 1 on/off
4. **M4 — stretch:** reward variants

**Kill criteria:** if M2 loses to SASRec after sanity checks (head learning, loss weights, pooling position, lr), the honest result is "verbalized-LLM ranking doesn't beat ID-based sequential rec at 1M-rating scale" — also a finding.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Head learns but frozen backbone gives poor `h` | LoRA on all-linear (not just q/v); verify head loss decreases; try last-4-layers LoRA bump |
| 4-bit base + full-rank head optimization instability | Head in bf16, grad clip 1.0, short smoke run first |
| Kaggle session limits (12h max) | Checkpoint every 500 steps; all runs resumable |
| Genres/metadata missing for some ML-1M titles | Fall back to title+year only; note count |
