# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

See `docs/spec.md` for the GenRec-at-Home experiment specification (data, model, ablations, milestones, risks).

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project-specific: Kaggle (long-running jobs)

Use the Kaggle web UI for everything:

1. **Internet must be on.** Toggle "Internet" ON in the notebook before downloading data (datasets, libraries). Off by default in Kaggle notebooks.
2. **Build notebooks in Kaggle** — editor, Run All, then detach. Open a notebook → set GPU to T4 (Settings → Accelerator) → **"Run All"** → close the browser. Kaggle keeps it running in the background (up to 12h per session, free).

   **Use both GPUs.** Free tier = 2× T4. Configure the accelerator in Settings → Accelerator, then verify both are visible before committing to a long run with a smoke cell:
   ```python
   import torch
   assert torch.cuda.is_available(), "no GPU"
   print("GPU count:", torch.cuda.device_count())
   for i in range(torch.cuda.device_count()):
       print("GPU", i, torch.cuda.get_device_name(i))
   assert torch.cuda.device_count() == 2, "expected 2 GPUs"
   ```
   If both are visible, wrap the model with `torch.nn.DataParallel(model)` (or set `CUDA_VISIBLE_DEVICES`) to span them. Run this cell first — if it fails, fix before wasting a training session.
3. **Directory structure** — keep all scripts, data, and outputs inside the notebook's working directory on Kaggle. Don't rely on absolute paths outside that folder; Kaggle's environment is ephemeral and resets between sessions. The repo (`git`) is the source of truth; Kaggle notebooks mirror it — if they diverge, repo wins.
4. **Checkpoint often.** Kaggle sessions kill at 12h. Inside the training script, checkpoint every N steps (`torch.save` / `hf_hub_save_checkpoint`). Resume from the last checkpoint on restart — don't start over.
5. **Quota.** Free tier = ~30 GPU-hours/week across 2× T4 (16GB each). Budget the week's runs across ablations. If a run hits the 12h session cap, split it into resumable chunks (checkpoint + re-run).
6. **Push before starting a job.** Commit the notebook (.ipynb) and `train_phase2.py` to git so the remote commit is the job's source of truth.

## Project-specific: Git discipline

- **Push after every meaningful unit of work** — a completed script, a passing milestone, a resolved ablation. Don't accumulate hours of uncommitted work.
- Commit with the milestone name + short description: `git commit -m "M2: GenRec-4B base run, beats SASRec"`
- If the session is interrupted (Kaggle kill, crash), push any recovered state immediately before re-running.
- The repo is the single source of truth for scripts and the spec. Kaggle notebooks mirror the repo; if they diverge, repo wins.