# RR 100M Encoder Training Recipe

Status: current decision document for the final Recursive Refiner run. Config/code support is in this worktree; GPU smoke tests still need to run on the training VM.

Implementation note: new factorized-embedding runs use variance-matched
initialization and report `GLUE8_amean` (eight-task validation composite).
Historical results in this document used the earlier initialization and score;
retrain their anchors before making direct comparisons.

Goal: train a roughly 100M parameter encoder efficiently, starting from this Cramming-derived repo, while copying the strongest available encoder pretraining ideas from Cramming, MosaicBERT, ModernBERT, NeoBERT, ELECTRA/DeBERTaV3, RoBERTa, and modern open data work.

## Executive Recommendation

Use this as the main final-run candidate:

- Model: `RecursiveRefinerForMaskedLM`
- Size: about 97M parameters
- Shape: `hidden_size=1024`, `num_attention_heads=16`, `num_hidden_layers=3`, `expansion=8.0`, `embed_factor=4`
- Cycles: `hi_cycles=2`, `lo_cycles=3`
- Effective runtime depth: `2 * (3 + 1) * 3 = 24` shared block applications
- Position/norm: keep repo default `RoPE`, `pre_norm=true`, `RMSNorm`
- Attention kernel: PyTorch SDPA enabled by default via `arch.use_sdpa=true`; disable with `arch.use_sdpa=false` if the VM preflight or smoke fails
- Objective: MLM, no NSP, **masking rate 0.30** for the main run
- Data tonight: `pile-readymade` because it is already prepared and has produced the best local RR results
- Better data path if preparing a new corpus: `fineweb-sample10bt` for the first smoke, `fineweb-sample100bt` for the serious modern web backbone, `fineweb-edu-sample10bt` / `fineweb-edu-sample100bt` for education-filtered comparison or upsampling, and `refinedweb-25m-docs` for a NeoBERT-style RefinedWeb smoke
- Optimizer: start with AdamW for the final serious run unless Muon gets a short stability smoke pass; use `lr=6e-4` to `8e-4`, `betas=(0.9,0.95 or 0.98)`, `weight_decay=0.1`
- Schedule: use `budget-wsd`: warm up, hold LR for most of the wallclock budget, then decay to `lr_end_factor=0.1`
- Precision/system: bf16 if available, `compile_torch=True` after smoke test, large global batch as memory permits
- Evaluation: GLUE full eval, report per-task best validation from logs, 3 seeds only after final shortlist

Primary local command template:

```bash
bash scripts/rr_100m_final.sh final
bash scripts/rr_100m_final.sh eval
bash scripts/rr_100m_final.sh summarize-eval
```

Local PowerShell equivalent:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 final
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 eval
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 summarize-eval
```

Equivalent expanded command:

```bash
python pretrain.py \
  name=rr_100m_h1024_l3_exp8_mlm30 \
  seed=1337 \
  data=pile-readymade \
  train=rr-modern-wsd \
  arch=recursive-refiner-100m \
  budget=72 \
  impl.microbatch_size=128 \
  impl.compile_torch=True \
  "wandb.tags=[rr-final,100m,mlm30]"
```

Run this before the dryrun on the real training VM:

```bash
bash scripts/rr_100m_final.sh preflight
```

Local PowerShell equivalent:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 preflight
```

The preflight constructs the 100M config, checks parameter count, and compares SDPA attention against the manual fallback on small causal/non-causal masked cases. Use `PREFLIGHT_DTYPE=bfloat16` to test the lower-precision kernel explicitly.

After pretraining, run full validation GLUE with log-best epoch selection:

```bash
bash scripts/rr_100m_final.sh eval
bash scripts/rr_100m_final.sh summarize-eval
```

This defaults to `eval=GLUE_sane`, `eval.epoch_selection=best`, `eval.selection_metric=target`, `eval.epochs=5`, `eval.batch_size=16`, `eval.optim.lr=8e-5`, and `eval.checkpoint=latest`. Override with environment variables such as `EVAL_CFG=GLUE`, `EVAL_LR=4e-5`, `EVAL_EPOCHS=10`, or `EVAL_NAME=<run_name>`.

The summarizer writes:

- `tables/best_eval_from_logs.csv`
- `tables/best_eval_from_logs_report.md`

It reads downstream logs directly, selects best epoch per task using target metrics, and handles old MNLI-mm logs that only printed final extra validation.

The command above uses the new repo configs:

- `cramming/config/arch/recursive-refiner-100m.yaml`
- `cramming/config/train/rr-modern-wsd.yaml`
- `cramming/config/data/fineweb-sample10bt.yaml` for a quick FineWeb preprocessing smoke
- `cramming/config/data/fineweb-sample100bt.yaml` for a serious FineWeb backbone
- `cramming/config/data/fineweb-edu-sample10bt.yaml` and `cramming/config/data/fineweb-edu-sample100bt.yaml` for education-filtered data experiments
- `cramming/config/data/refinedweb-25m-docs.yaml` for a RefinedWeb smoke with `text_column=content`
- `scripts/rr_100m_final.sh`
- `scripts/rr_100m_final.ps1`
- `scripts/plot_pretrain_losses.py` for regenerating local pretrain-loss comparisons after VM merges
- `scripts/verify_rr_100m_preflight.py` for VM-side parameter-count and SDPA parity checks
- `scripts/extract_best_eval_from_logs.py` for log-best downstream scoring

If you want the run to be above 100M rather than near 100M, use the wide variant:

```bash
python pretrain.py \
  name=rr_100m_h1280_l2_exp8_mlm30 \
  seed=1337 \
  data=pile-readymade \
  train=rr-me \
  arch=recursive-refiner-large \
  budget=72 \
  impl.microbatch_size=96 \
  impl.compile_torch=True \
  arch.hidden_size=1280 \
  arch.num_attention_heads=20 \
  arch.num_hidden_layers=2 \
  arch.expansion=8.0 \
  arch.hi_cycles=2 \
  arch.lo_cycles=3 \
  arch.embed_factor=4 \
  train.batch_size=4096 \
  train.batch_size_ramp=0.1 \
  train.optim.lr=6e-4 \
  train.optim.betas=[0.9,0.95] \
  train.optim.eps=1e-8 \
  train.optim.weight_decay=0.1 \
  train.warmup_steps=2000 \
  train.cooldown_steps=100000 \
  train.steps=2500000 \
  train.scheduler=budget-wsd \
  train.lr_end_factor=0.1 \
  train.gradient_clipping=1.0 \
  train.objective.mlm_probability=0.30 \
  "wandb.tags=[rr-final,100m,wide,mlm30]"
```

I prefer the `h1024_l3_exp8` variant because it is closer to ModernBERT/NeoBERT's deep-narrow lesson while preserving the local RR `exp8` win. Use `h1280_l2_exp8` if throughput is the overriding constraint or if `L=3` is too slow.

## Why This Recipe

### 1. Local Evidence

The strongest merged-zip run is:

| Run | Params | Pretrain budget | Loss | Corrected GLUE avg | Hmean |
|---|---:|---:|---:|---:|---:|
| `rr_single_high_tiny8x_8xexp` | 1.98M | 48 | 2.884 | 72.40 | 66.95 |
| `rr_single_high_tiny8x` | 1.58M | 24 | 3.238 | 70.74 | 52.91 |
| `albert_shared_h128_g1_inner2_eff16_e32` | 1.50M | 24 | 3.562 | 67.85 | 0.00 |

Important local lesson: larger FFN expansion helped. The `8xexp` run was not just lower loss; it improved CoLA enough that the harmonic mean became meaningful. That makes `expansion=8.0` worth carrying into the 100M candidate.

Local loss-curve evidence is now regenerated from raw pretrain logs rather than only convergence CSVs:

- `tables/all_models_pretrain_loss_vs_step.png`: one curve per model, longest valid run selected
- `tables/all_runs_pretrain_loss_vs_step.png`: repeated attempts included
- `tables/all_models_pretrain_loss_from_logs_summary.csv`: exact source log for each plotted model
- `tables/missing_models_pretrain_loss_summary.csv`: output folders that never emitted `Train loss` rows

Older internal table evidence:

- `rr_me_train_final_tiny8x`: h256, 4 heads, L2, hi2/lo3, `embed_factor=4`, 4.22M params, 8h, GLUE 75.47
- `rr_me_train_final`: h768, 12 heads, L2, `embed_factor=1`, 44.65M params, 36h, GLUE 79.92

That says RR scales with width and has already reached CrammingBERT-ish GLUE at much smaller parameter count, but the current final run should not be judged only by MLM loss. Small tasks like CoLA and RTE are volatile.

### 2. Parameter Sizing

For this repo's RR implementation with vocab 32768 and `embed_factor=4`, approximate parameter counts are:

| Candidate | Params | Effective runtime blocks | Read |
|---|---:|---:|---|
| h1024 L3 exp8 ef4 | 96.87M | 24 | Recommended main run |
| h1280 L2 exp8 ef4 | 102.81M | 16 | Wider, likely faster per token than L3, but less deep |
| h1024 L4 exp6 ef4 | 101.07M | 32 | More depth, likely slower; good if h1024 L3 underfits |
| h1344 L3 exp4 ef4 | 98.34M | 24 | More conservative FFN expansion |
| h1536 L2 exp5 ef4 | 103.03M | 16 | Very wide; less aligned with deep-narrow literature |

The formula used matches the current tiny run exactly:

- Embedding params for `embed_factor=4`: `vocab * hidden/4 + (hidden/4)^2 + (hidden/4 * hidden)`
- One RR block: attention `4D^2 + 4D`, SwiGLU FFN `(3 * expansion * D^2) + (2 * expansion + 1)D`, plus two RMSNorm vectors
- Total: low-rank embedding + `num_hidden_layers * block_params` + `hi_init/lo_init`

### 3. Literature Lessons To Copy

#### Cramming

Cramming is the correct base frame for this repo: it asks how much performance can be achieved from scratch under a hard single-GPU/day-style budget, using MLM and an optimized pretraining pipeline. It is also the reason `pile-readymade` exists here.

Copy:

- time-budgeted training
- aggressive throughput optimization
- short sequence length when the target is GLUE-like classification
- report downstream, not just MLM loss

Do not over-copy:

- 15% MLM masking; later work is stronger here
- old BERT-style absolute position embeddings; RR already has RoPE

Source: https://arxiv.org/abs/2212.14034

#### RoBERTa

RoBERTa's durable lesson is not architecture; it is recipe discipline. BERT was undertrained, NSP was unnecessary, larger/better data and careful hyperparameters matter.

Copy:

- no NSP
- dynamic masking / avoid static corruption
- train longer on more/better data
- compare with careful fine-tuning sweeps

Source: https://arxiv.org/abs/1907.11692

#### ELECTRA and DeBERTaV3

ELECTRA's replaced-token detection is the best sample-efficiency idea for small encoders: the discriminator learns on all tokens rather than only masked positions. DeBERTaV3 confirms ELECTRA-style pretraining can still be extremely strong for modern encoders.

How ELECTRA-Small differs from this final RR plan:

| Axis | ELECTRA-Small | Final RR 100M plan | Practical read |
|---|---|---|---|
| Main objective | Replaced-token detection with a small MLM generator | MLM only, 30% masking | ELECTRA is more sample-efficient because its discriminator has loss on every token; this is the biggest scientific gap. |
| Architecture | BERT-style encoder, hidden 256, small 128-dim token embeddings, about 14M params | Recursive Refiner, hidden 1024, 3 shared blocks, `expansion=8`, `embed_factor=4`, about 97M params | RR is much larger and recurrent/shared; ELECTRA-Small is the objective baseline, not a parameter-matched architecture baseline. |
| Sequence length | 128 for the small efficient setup | 128 for the main GLUE-efficient run | This matches the efficient-short-context lesson. |
| Steps/compute | 1M discriminator steps for full small model; paper reports 6h/12h/1d/2d/4d variants on 1 V100 | time-budgeted Cramming-style run; final launcher defaults to 72h budget on the target VM | Compare by wallclock/tokens and downstream GLUE, not by parameter count alone. |
| Reported GLUE | Dev GLUE 74.1 at 6h, 76.0 at 12h, 77.7 at 1d, 79.0 at 2d, 79.9 at 4d; test avg 78.0 | Best merged tiny RR is 72.40 at 1.98M params; older 4.22M RR anchor is 75.47; final 97M result unknown until run | If 100M RR cannot beat the 6h/12h ELECTRA-Small points, the issue is likely objective/data/training recipe rather than raw capacity. |
| Data/objective risk | Extra generator code and sampling path, but proven efficient | Simpler, already supported in repo | Keep MLM for the immediate final run; implement RTD as the next major ablation after the final RR run. |

Copy later, not for the immediate final run:

- implement RTD as a second training objective for RR
- train a small generator, then fine-tune/evaluate the RR discriminator

Why not immediate:

- this repo currently has only MLM in the collator/training path
- RTD implementation mistakes can easily invalidate the final run

Source: https://arxiv.org/abs/2003.10555<br>
Source: https://arxiv.org/abs/2111.09543

#### MosaicBERT

MosaicBERT is the cleanest "efficient encoder pretraining" recipe. It combines FlashAttention, ALiBi, GLU, unpadding, low precision LayerNorm, 30% MLM, bf16, C4, vocab multiple-of-64, global batch 4096, sequence length 128, and fast streaming. It reaches strong GLUE very quickly on 8x A100.

Copy now:

- MLM 30%
- bf16 where possible
- global batch around 4096 if memory allows
- vocab multiple of 64; current repo vocab 32768 already satisfies this
- short sequence length 128 for efficient GLUE-style pretraining; current `pile-readymade` is exactly seq 128
- unpadding/FlashAttention as a repo improvement

Source: https://arxiv.org/abs/2312.17482<br>
Ready repo: https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert

#### ModernBERT

ModernBERT is the best modern encoder recipe to steal from if you are willing to modify the repo. It uses RoPE, pre-norm, GeGLU, no most biases, alternating local/global attention, unpadding before embeddings, FlashAttention 2/3, torch.compile, 30% MLM, StableAdamW, WSD/trapezoid LR, batch-size warmup, modern BPE tokenizer, 2T tokens, and long-context extension from 1024 to 8192.

Copy now:

- RoPE/pre-norm/GLU style: already mostly present in RR
- `torch.compile`: available in repo implementation
- batch size warmup
- WSD-like schedule: now implemented as `budget-wsd`
- do not chase 8192 context for this final GLUE run; it is expensive and not needed unless retrieval/long-doc is the target

Copy later:

- unpadding before embeddings
- alternating local/global attention for long context
- StableAdamW
- modern BPE tokenizer and sequence packing

Source: https://arxiv.org/abs/2412.13663<br>
Ready repo/framework: https://github.com/AnswerDotAI/ModernBERT and FlexBERT linked from the paper

2026 refresh: the very recent moBERTo work adapts ModernBERT through continued pretraining on a FineWeb2-derived corpus and preserves the same core efficiency stack: RoPE, alternating local/global attention, flash attention, unpadding, 30% MLM, StableAdamW, and long-context post-training. This strengthens the recommendation to keep ModernBERT-style kernels/data/objective as the default reference recipe rather than reverting to older BERT/RoBERTa defaults.

Source: https://arxiv.org/abs/2606.22722

#### NeoBERT

NeoBERT is currently the strongest fully open "from scratch modern encoder" recipe I found. It is 250M params, 28 layers, hidden 768, RoPE, pre-RMSNorm, SwiGLU, RefinedWeb, sequence length 1024 then 4096, 20% MLM, AdamW, cosine decay, 2.1T tokens. It releases code, data, checkpoints, and scripts.

Copy:

- deep/narrow bias: prefer h1024 L3 over a very wide L2 if speed is tolerable
- RMSNorm: RR already uses it
- `betas=(0.9,0.95)`, `eps=1e-8`, `weight_decay=0.1`
- `lr=6e-4` as a stable AdamW peak for a modern encoder
- two-stage context extension only after the short-context model is good
- 20% and 30% MLM both deserve a smoke comparison

Source: https://arxiv.org/abs/2502.19587<br>
Ready repo: https://github.com/chandar-lab/NeoBERT<br>
HF: https://huggingface.co/chandar-lab/NeoBERT

#### FineWeb / RefinedWeb / Dolma

Modern encoder performance is now heavily data-bound. FineWeb is newer and huge: 15T tokens from 96 Common Crawl snapshots, with documented filtering/deduplication; FineWeb-Edu is 1.3T educational tokens. RefinedWeb is what NeoBERT used, and it is simpler to point to because NeoBERT's recipe already validated it for encoders.

Best data choice if you prepare new data:

1. Start with FineWeb or RefinedWeb, not old Wiki+Books only.
2. Keep high-quality web as the backbone.
3. Add/upweight FineWeb-Edu or equivalent educational/scientific text.
4. Add code if retrieval/code tasks matter, following ModernBERT's lesson.
5. Deduplicate aggressively enough to avoid memorization, but do not blindly global-dedup away distribution diversity.
6. Use streaming/sharded format to avoid dataloader bottlenecks.

Source: https://arxiv.org/abs/2406.17557<br>
Source: https://arxiv.org/abs/2306.01116<br>
Source: https://arxiv.org/abs/2402.00159

## Data Decision

### Run Tonight

Use:

```bash
data=pile-readymade
```

Reason:

- already tokenized and streaming
- vocab fixed at 32768, a multiple of 64
- seq length fixed at 128
- it produced the best local RR runs
- avoids turning the final run into a data engineering project

Limitations:

- not the current best public data
- no long-context stage
- likely less fresh/clean than FineWeb/RefinedWeb-style data

### Best Data Path

If preparing data before the real final run is allowed, make a new readymade dataset:

- 50-60% FineWeb high-quality web or RefinedWeb
- 15-25% FineWeb-Edu / educational/scientific text
- 10-15% code, if code/retrieval matters
- 5-10% Wikipedia/books for clean encyclopedic/classic prose
- optional domain slice if this encoder has a known target domain

I added concrete configs for the easiest first step and the larger corpus builds:

```bash
data=fineweb-sample10bt
data=fineweb-sample100bt
data=fineweb-edu-sample10bt
data=fineweb-edu-sample100bt
data=refinedweb-25m-docs
```

Use them like:

```bash
# smallest modern-web smoke
DATA=fineweb-sample10bt bash scripts/rr_100m_final.sh dryrun

# serious modern-web backbone after preprocessing/storage are ready
DATA=fineweb-sample100bt bash scripts/rr_100m_final.sh final

# education-filtered comparison
DATA=fineweb-edu-sample10bt bash scripts/rr_100m_final.sh dryrun
DATA=fineweb-edu-sample100bt bash scripts/rr_100m_final.sh final

# RefinedWeb smoke; scale the config only after this preprocesses cleanly
DATA=refinedweb-25m-docs bash scripts/rr_100m_final.sh dryrun
```

PowerShell equivalent:

```powershell
$env:DATA="fineweb-sample10bt"; powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 dryrun
$env:DATA="fineweb-sample100bt"; powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 final
```

These point to `HuggingFaceFW/fineweb` and `HuggingFaceFW/fineweb-edu` with Hugging Face subset names `sample-10BT` and `sample-100BT`. They are for preprocessing experiments and readymade corpus builds; they should not replace `pile-readymade` in the immediate final command until tokenization throughput and downstream quality are checked.

The RefinedWeb config points to `tiiuae/falcon-refinedweb`, which exposes its raw document text as `content` rather than `text`. I added source-level `text_column` support so the preprocessing path maps it into the repo's expected `text` field before tokenizer training.

Dataset-card evidence checked on 2026-07-08:

- FineWeb exposes `sample-10BT`, `sample-100BT`, and `sample-350BT`, with columns `text`, `id`, `dump`, `url`, `date`, `file_path`, `language`, `language_score`, and `token_count`: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb-Edu exposes the same sample sizes plus educational scores, with extra `score` and `int_score` columns: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Falcon RefinedWeb has one default train split with 968M rows and columns `content`, `url`, `timestamp`, `dump`, `segment`, and `image_urls`: https://huggingface.co/datasets/tiiuae/falcon-refinedweb

Tokenizer:

- keep vocab multiple of 64
- for fastest compatibility with this repo: WordPiece 32768
- for best modern model quality: modern BPE around 50k, but that requires more repo/tokenizer work

Sequence length:

- stage 1: 128 if GLUE/classification efficiency is the target
- stage 1 modern alternative: 512 or 1024 if general embeddings/retrieval are the target
- stage 2: 1024 or 4096 only after short-context model is good

## Objective Decision

Main run:

```bash
train.objective.name=masked-lm
train.objective.mlm_probability=0.30
train.objective.use_80_20_rule=True
```

Why 30%:

- MosaicBERT uses 30% and reports speed/accuracy Pareto improvements.
- ModernBERT follows MosaicBERT and uses 30%.
- Masking-rate literature says 15% is not universal; larger models benefit from higher rates.
- Your 100M RR is no longer a tiny model.

Smoke before final:

- 2h or 4h at `mlm_probability=0.20`
- 2h or 4h at `mlm_probability=0.30`
- If 30% is unstable or downstream worse in quick eval, use 20%.

Do not run 15% as final unless you need strict comparability with old local RR runs.

## Optimizer and LR

### Conservative Final

Use AdamW:

- `lr=6e-4`
- `betas=[0.9,0.95]`
- `eps=1e-8`
- `weight_decay=0.1`
- `gradient_clipping=1.0`
- `warmup_steps=2000`
- `scheduler=budget-wsd`
- `lr_end_factor=0.1`
- `cooldown_steps=100000`

Why:

- NeoBERT uses AdamW with this beta/eps/weight-decay family and peak LR 6e-4.
- ModernBERT uses 8e-4 for base with StableAdamW, but this repo does not currently implement StableAdamW.
- Local RR used 8e-4 successfully at smaller scale, so 6e-4 is a safer first 100M peak.

### Aggressive/Faster If Smoke Passes

Use the existing Muon wrapper:

```bash
train=rr-me-muon \
train.optim.lr=8e-4 \
train.optim.muon_lr=0.008 \
train.optim.weight_decay=0.1 \
train.gradient_clipping=0.5 \
train.batch_size=4096
```

But do not make Muon the final run without a smoke comparison. The literature-backed encoder recipes here are AdamW/StableAdamW, not Muon.

### Schedule

Best literature schedule:

- ModernBERT: WSD/trapezoid with short warmup, long constant LR, final decay.

Current repo approximation:

- `scheduler=budget-wsd`, `warmup_steps=2000`, `cooldown_steps=100000`, `lr_end_factor=0.1`

Already-tested local alternative:

- `scheduler=budget-one-cycle` from `rr-me-onecycle`

I would not use one-cycle for the final 100M run unless the smoke says it beats WSD, because it spends too little time at a stable LR for long training.

## Systems Recipe

Use:

- bf16/mixed precision if your backend supports it
- `impl.compile_torch=True` after dryrun/smoke confirms no compile failure
- `impl.microbatch_size` as high as memory allows, start 128 for h1024 L3 and 96 for h1280 L2
- global `train.batch_size=4096`; if memory or throughput suffers, use 2048
- avoid dataloader bottlenecks; streaming data should be local-cache friendly

Missing high-value repo improvements:

1. Done: configurable PyTorch SDPA path inside RR attention, with manual fallback.
2. Unpadding/packing support before attention.
3. Done: true WSD scheduler.
4. StableAdamW.
5. ELECTRA/RTD objective.
6. Done: ready FineWeb/FineWeb-Edu configs plus a RefinedWeb smoke config with source-level text-column mapping. Still open: custom weighted readymade mixture.
7. Long-context staged data/tokenizer path.

## Smoke Test Plan

Do not burn the final run blind. Run these first:

```bash
# Dryrun
bash scripts/rr_100m_final.sh preflight
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 preflight
bash scripts/rr_100m_final.sh dryrun
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 dryrun

# 2h smoke, 20% MLM
bash scripts/rr_100m_final.sh smoke20
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 smoke20

# 2h smoke, 30% MLM
bash scripts/rr_100m_final.sh smoke30
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 smoke30

# Print downstream eval command before running it
bash scripts/rr_100m_final.sh print-eval
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 print-eval

# Summarize best validation metrics after eval logs exist
bash scripts/rr_100m_final.sh summarize-eval
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 summarize-eval
```

Smoke acceptance:

- no loss explosion
- tokens/sec acceptable relative to h768/h1024 old runs
- at matched early tokens, 30% MLM is not clearly worse than 20%
- eval on a cheap subset, at minimum RTE/CoLA/SST2/MNLI, before committing days

## Evaluation Protocol

Use downstream fine-tune logs, not only YAML/reports:

- CoLA: best `matthews_correlation`
- RTE/MNLI/QNLI/SST2: best `accuracy`
- MRPC/QQP: best `f1`
- STS-B: best `pearson`
- MNLI-mm: now logged per epoch as `accuracy_extra`, so `eval.epoch_selection=best` can select with both MNLI matched and mismatched accuracy when the task config lists both metrics
- Final launcher now includes `eval`, `print-eval`, and `summarize-eval` actions using `GLUE_sane` by default

Report:

- arithmetic mean
- harmonic mean
- per-task scores
- best epochs
- pretrain tokens and kWh
- exact command/config
- at least three seeds only for final shortlisted setup

## Ready Repos To Steal From

1. MosaicBERT / Composer BERT benchmark
   - Best ready-to-run efficient BERT training repo.
   - Has MosaicBERT architecture, C4 streaming, GLUE scripts.
   - Use it as a reference for unpadding, FlashAttention, 30% MLM, data streaming, and GLUE orchestration.
   - https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert

2. ModernBERT / FlexBERT
   - Best modern encoder framework for architecture ideas and long-context efficient inference/training.
   - Use it as reference for unpadding, alternating local/global attention, WSD/StableAdamW, tokenizer/data choices.
   - https://github.com/AnswerDotAI/ModernBERT

3. NeoBERT
   - Best fully open recent from-scratch encoder recipe with code/data/checkpoints/scripts.
   - Use it as reference for 2025-style deep/narrow design, AdamW settings, RefinedWeb data, 1024 to 4096 context extension.
   - https://github.com/chandar-lab/NeoBERT

4. This repo
   - Best place to run RR because the architecture and local reports are here.
   - Use the above repos as implementation references, not replacements, unless you decide to abandon RR.

## What To Improve Before The "No Excuses" Final Run

Highest leverage:

1. Done: add `train=rr-modern-wsd` config: AdamW, `lr=6e-4`, betas 0.9/0.95, eps 1e-8, wd 0.1, clipping 1.0, `budget-wsd`.
2. Done: add `arch=recursive-refiner-100m` config with h1024/L3/exp8/ef4/hi2/lo3.
3. Done: add smoke script for 20% vs 30% MLM and h1024-L3 vs h1280-L2.
4. Done: add per-epoch MNLI-mm logging so log-best aggregation is not mixed.
5. Done: add FineWeb, FineWeb-Edu, and RefinedWeb smoke data configs. Still need a custom mixed readymade builder if this becomes the final data.
6. Done: add configurable PyTorch SDPA to RR attention; run `preflight` on the training VM to benchmark/check against the old manual path.
7. Done: add final GLUE eval launcher action so the pretrain recipe has a paired downstream scoring command.
8. Done: add log-best downstream parser so future reports do not mix last-epoch and best-epoch metrics.

Medium leverage:

9. Add unpadding/packing.
10. Add StableAdamW.
11. Add RTD/ELECTRA objective.

Only after that:

10. Long-context extension.
11. New tokenizer.
12. Multi-stage contrastive/retrieval fine-tuning.

## Local Verification Done

Current repo-side checks completed:

- Added `budget-wsd` to `cramming/backend/optimizers/schedulers.py`.
- Added SDPA-backed attention to `cramming/architectures/recursive_refiner_hf.py`, with manual attention fallback for older PyTorch.
- Added `arch.use_sdpa=true` to `cramming/config/arch/recursive-refiner-100m.yaml`; use `arch.use_sdpa=false` as the immediate fallback.
- Added per-epoch MNLI extra validation logging to `eval.py`.
- Added raw-log pretrain loss plotting in `scripts/plot_pretrain_losses.py`; current merged outputs parse 9 valid pretrain runs from 11 logs and plot 6 model curves.
- Added FineWeb/FineWeb-Edu data configs: `fineweb-sample10bt`, `fineweb-sample100bt`, `fineweb-edu-sample10bt`, and `fineweb-edu-sample100bt`.
- Added source-level `text_column` support in `cramming/data/pretraining_preparation.py` and `data=refinedweb-25m-docs` for the `content` column used by Falcon RefinedWeb.
- Added `scripts/verify_rr_100m_preflight.py` and launcher `preflight`, `eval`, `print-eval`, and `summarize-eval` actions.
- Added `scripts/extract_best_eval_from_logs.py`; current merged outputs parse 24 downstream logs and find 6 complete GLUE runs.
- Compiled `cramming/backend/optimizers/schedulers.py` with `python -m py_compile`.
- Compiled `cramming/architectures/recursive_refiner_hf.py`, `eval.py`, `scripts/plot_pretrain_losses.py`, `scripts/verify_rr_100m_preflight.py`, and `scripts/extract_best_eval_from_logs.py` with `python -m py_compile`.
- Deterministically checked the WSD curve with a stubbed `LambdaLR`: start `0.0`, half-warmup `0.5`, stable `1.0`, mid-decay `0.55`, end floor `0.1`.
- Verified PowerShell launcher print mode emits the expected final command.

Not locally verified:

- Actual Hydra config composition.
- Actual model construction for the 100M config.
- Actual training dryrun.
- Numerical parity between SDPA attention and the old manual attention path.

Reason: the local Windows Python environment is missing `torch`; run these in the real training VM before burning budget:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 dryrun
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 smoke20
powershell -ExecutionPolicy Bypass -File scripts/rr_100m_final.ps1 smoke30
```

## Bottom Line

For the next final RR run, do not chase every shiny thing at once. Run:

- RR h1024/L3/exp8/ef4, hi2/lo3
- 30% MLM unless smoke says 20% wins
- AdamW `6e-4`, beta2 `0.95`, eps `1e-8`, wd `0.1`
- `budget-wsd` with warmup/cooldown
- batch 4096 if possible
- `pile-readymade` if you want a real run now

The biggest scientific upgrade after this is not a larger RR. It is RTD/ELECTRA-style pretraining and better data.
