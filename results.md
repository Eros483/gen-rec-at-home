# Results

## M1 — Baselines (leave-one-out, full-catalog ranking, seen-item masking)

| Scorer | Split | n_users | MRR | NDCG@10 | HR@10 |
|---|---|---|---|---|---|
| popularity | test | 6038 | 0.0052 | 0.0025 | 0.0035 |
| SASRec (d=64, 2 layers, maxlen 200, 10 ep, full softmax) | test | 6038 | **0.3045** | 0.3494 | 0.5351 |

- SASRec val MRR still improving at epoch 10 (0.093 → 0.366); kept best-val checkpoint (`runs/sasrec/best.pt`, epoch 10).
- Harness sanity: SASRec ≫ popularity ✓.

**M2 gate: GenRec-4B base config must beat test MRR 0.3045.**
