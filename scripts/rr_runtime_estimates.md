# Runtime Estimates for Recursive Refiner Ablations

## Summary

The short version:

- A full one-pass `all` campaign from `rr_paper_h200_ablations.sh` is about **508 pretraining GPU-hours** with the current H200 budgets.
- Add full GLUE evaluation and the total becomes roughly **563-603 GPU-hours**.
- On one H200, that is about **23.5-25 days** if every run is executed sequentially.
- If the same campaign were run with RTX 5090-style throughput and long-model budgets, it is reasonable to think in the range of **30+ days**.
- A rough H200 speedup of **2-4x** over one RTX 5090 is plausible for this workload, but the only honest number comes from the `utilization` sweep.
- A useful first campaign is **utilization + core + sizes**, about **280 pretraining hours** before eval.

Do not report H200 and RTX 5090 runs as if equal wall-clock hours mean equal compute. Use separate labels:

1. **Wall-clock fixed**: same number of hours on each GPU.
2. **Token matched**: same number of tokens processed.
3. **Convergence/scaling**: larger models get longer budgets because they converge later.

## Hardware Anchors

NVIDIA lists the H200 as having **141GB GPU memory** and **4.8TB/s GPU memory bandwidth**. It is a Hopper data-center GPU with HBM3e memory. Source: <https://www.nvidia.com/en-us/data-center/h200/>

NVIDIA lists the GeForce RTX 5090 as having **32GB GDDR7**, a **512-bit memory interface**, **21,760 CUDA cores**, and **575W total graphics power**. Source: <https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/>

Practical implication:

| GPU | Memory | Memory Type | Notes |
| --- | ---: | --- | --- |
| RTX 5090 | 32GB | GDDR7 | Strong consumer GPU, but small VRAM for large sweeps |
| H200 | 141GB | HBM3e | About 4.4x the memory capacity of RTX 5090, much better suited to large microbatches and larger models |

The H200 should not automatically inherit RTX 5090 wall-clock budgets. If the goal is fair compute, match tokens. If the goal is "what can I do on this hardware," use H200-specific budgets.

## Anchor Run From Logs

Successful run:

```text
rr_me_train_final_tiny8x
RecursiveRefinerForMaskedLM
hidden_size=256
num_attention_heads=4
num_hidden_layers=2
hi_cycles=2
lo_cycles=3
embed_factor=4
parameters=4,222,976
budget=8h
final_step=598,000
step_time=0.0481s
microbatch_size=256
seq_length=128
```

Token estimate:

```text
tokens_per_microbatch_step = microbatch_size * seq_length
                           = 256 * 128
                           = 32,768 tokens

total_tokens_seen ~= 598,000 * 32,768
                  ~= 19.6B tokens

tokens_per_second ~= 32,768 / 0.0481
                  ~= 681k tokens/sec
```

This is the main reason not to blindly say "8 hours on both GPUs." A token-matched H200 run should target roughly **20B tokens** for the tiny8x configuration, not necessarily 8 hours.

## Time Equivalence

If H200 achieves:

| H200 Speedup vs RTX 5090 | 30 RTX 5090 Days Becomes |
| ---: | ---: |
| 2x | 15 H200 days |
| 2.5x | 12 H200 days |
| 3x | 10 H200 days |
| 4x | 7.5 H200 days |

So the statement "almost 30 days on RTX 5090 is about 10 days on H200" is a reasonable **3x-throughput hypothesis**.

But it is not guaranteed. H200 may underperform the hardware ratio if the run is bottlenecked by:

- Python overhead
- dataloader throughput
- tokenizer/data streaming stalls
- torch.compile graph breaks or compilation time
- small models that do not saturate the GPU
- global-batch changes that hurt optimization

That is why `utilization` should be run before the full campaign.

## Script Categories

The H200 script currently has these groups:

```bash
bash scripts/rr_paper_h200_ablations.sh print-pretrain core
bash scripts/rr_paper_h200_ablations.sh print-pretrain baselines
bash scripts/rr_paper_h200_ablations.sh print-pretrain components
bash scripts/rr_paper_h200_ablations.sh print-pretrain sizes
bash scripts/rr_paper_h200_ablations.sh print-pretrain utilization
```

Groups:

| Group | Purpose |
| --- | --- |
| `core` | Smallest defensible paper table around the successful tiny8x run |
| `baselines` | Core plus BERT, CrammedBERT, and ALBERT-style shared-depth baselines |
| `components` | Embedding rank, recurrence schedule, physical depth, FFN, and norm ablations |
| `sizes` | Width scaling sweep |
| `utilization` | Same h256 model with multiple batch/microbatch choices |
| `all` | `baselines + components + sizes + utilization` |

## Current H200 Budget Defaults

From `rr_paper_h200_ablations.sh`:

| Size Class | Microbatch | Global Batch | Budget |
| --- | ---: | ---: | ---: |
| h128 | 2048 | 4096 | 8h |
| h256 | 1536 | 4096 | 12h |
| h512 | 768 | 4096 | 24h |
| h768 | 512 | 8192 | 48h |
| h1024 | 256 | 8192 | 72h |

These are not "fair RTX 5090 budgets." They are H200-oriented first-pass budgets that give larger models more time.

Override examples:

```bash
H256_MBS=1024 BUDGET_SMALL=8 bash scripts/rr_paper_h200_ablations.sh pretrain core
H1024_MBS=384 BUDGET_XL=96 bash scripts/rr_paper_h200_ablations.sh pretrain sizes
TRAIN_BATCH=2048 TRAIN_MBS=512 BUDGET=8 bash scripts/rr_paper_h200_ablations.sh pretrain core
FIXED_RECIPE=1 PREFIX=rr_fixed bash scripts/rr_paper_h200_ablations.sh pretrain core
```

## Estimated H200 Pretraining Time by Category

These estimates assume the current default H200 budgets in the script.

| Group | Runs | Pretraining Hours |
| --- | ---: | ---: |
| `core` | 5 | 56h |
| `baselines` | 9 | 104h |
| `components` | 15 | 180h |
| `sizes` | 5 | 164h |
| `utilization` | 5 | 60h |
| `all` | 34 | 508h |

`all` includes `baselines`, and `baselines` already includes `core`, so `core` is not added twice in the `all` total.

## Category Breakdown

### Core: 56h

| Run | Size | Budget |
| --- | --- | ---: |
| RR tiny8x rerun | h256 | 12h |
| ALBERT shared no-cycle baseline | h256 | 12h |
| HF BERT same hidden/layers | h256 | 12h |
| CrammedBERT same hidden/layers | h256 | 12h |
| HF BERT parameter-ish match | h128 | 8h |

### Baselines: 104h

Includes all `core` runs plus:

| Extra Run | Size | Budget |
| --- | --- | ---: |
| HF BERT effective-depth 16 | h256 | 12h |
| CrammedBERT effective-depth 16 | h256 | 12h |
| ALBERT shared embedding 256 | h256 | 12h |
| ALBERT shared effective-depth 8 | h256 | 12h |

### Components: 180h

All component runs are h256 under current naming, so each is 12h.

| Component Axis | Runs | Hours |
| --- | ---: | ---: |
| Embedding factor sweep | 4 | 48h |
| Recurrence schedule sweep | 6 | 72h |
| Physical block count | 2 | 24h |
| FFN/norm sweep | 3 | 36h |

### Sizes: 164h

| Size | Budget |
| --- | ---: |
| h128 | 8h |
| h256 | 12h |
| h512 | 24h |
| h768 | 48h |
| h1024 | 72h |

### Utilization: 60h

Five h256 utilization runs at 12h each:

| Global Batch | Microbatch | Budget |
| ---: | ---: | ---: |
| 2048 | 256 | 12h |
| 4096 | 512 | 12h |
| 4096 | 1024 | 12h |
| 8192 | 1024 | 12h |
| 8192 | 1536 | 12h |

This group could be shortened. For pure throughput testing, 1-2 hours per setting may be enough. For optimization quality, longer is better.

## Evaluation Time

Your full tiny8x GLUE evaluation took about **2h47m** on RTX 5090. H200 may be faster, but eval can be CPU/dataloader-heavy and not scale like pretraining.

Conservative estimate:

| Scope | Eval Runs | Eval Time |
| --- | ---: | ---: |
| `core` | 5 | 8-14h |
| `baselines` | 9 | 15-25h |
| `components` | 15 | 25-42h |
| `sizes` | 5 | 10-20h |
| `utilization` | 5 | 8-14h |
| `all` | 34 | 55-95h |

Full `all` including eval:

```text
508h pretrain + 55-95h eval = 563-603 GPU-hours
```

On one H200:

```text
563-603 hours ~= 23.5-25.1 days
```

## RTX 5090 Budget Sketch

If running on one RTX 5090, use more conservative size budgets:

| Size | Suggested RTX 5090 Budget |
| --- | ---: |
| h128 | 6-8h |
| h256 tiny8x | 8h |
| h256 full embedding | 12h |
| h512 | 18-24h |
| h768 | 36h |
| h1024 | 48h+ or skip unless necessary |

For RTX 5090, h1024 is likely the least attractive point: it may fit only with small microbatches, move slowly, and consume a lot of wall-clock for a noisy scaling signal. h768 is probably the more useful upper endpoint on consumer hardware.

## H200 Budget Sketch

Use H200 in two ways:

### Token-Matched H200

Use the RTX 5090 token count as the target. For tiny8x, that target is about **20B tokens**.

If H200 reaches:

| H200 Throughput | Time for 20B Tokens |
| ---: | ---: |
| 1.0M tokens/sec | 5.6h |
| 1.5M tokens/sec | 3.7h |
| 2.0M tokens/sec | 2.8h |
| 2.5M tokens/sec | 2.2h |

### Convergence H200

Let larger models run longer:

| Size | H200 Convergence Budget |
| --- | ---: |
| h128 | 2-8h |
| h256 | 4-12h |
| h512 | 12-24h |
| h768 | 24-48h |
| h1024 | 48-96h |

## Recommended Campaigns

### Minimal Sanity Campaign

Purpose: make sure the H200 profile is not broken.

```bash
bash scripts/rr_paper_h200_ablations.sh pretrain utilization
bash scripts/rr_paper_h200_ablations.sh pretrain core
```

Cost:

```text
60h + 56h = 116 pretraining hours
```

This can be shortened by reducing `BUDGET_SMALL` for the utilization pass:

```bash
BUDGET_SMALL=2 bash scripts/rr_paper_h200_ablations.sh pretrain utilization
```

Then utilization costs only 10h.

### First Real Paper Campaign

Purpose: enough data to decide whether the story is real.

```bash
bash scripts/rr_paper_h200_ablations.sh pretrain utilization
bash scripts/rr_paper_h200_ablations.sh pretrain core
bash scripts/rr_paper_h200_ablations.sh pretrain sizes
bash scripts/rr_paper_h200_ablations.sh eval core
bash scripts/rr_paper_h200_ablations.sh eval sizes
```

Cost:

```text
60h + 56h + 164h = 280 pretraining hours
eval ~= 18-34h
total ~= 298-314h
```

On one H200:

```text
12.4-13.1 days
```

### Full One-Seed Campaign

Purpose: broad ablation coverage.

```bash
bash scripts/rr_paper_h200_ablations.sh pretrain all
bash scripts/rr_paper_h200_ablations.sh eval all
```

Cost:

```text
563-603 total GPU-hours
23.5-25.1 days on one H200
```

### Final Multi-Seed Campaign

Do not run all ablations with 3 seeds. Instead:

1. Pick the best RR configuration.
2. Pick the strongest BERT/CrammedBERT/ALBERT baselines.
3. Pick 3-5 sizes for scaling.
4. Run 3 seeds only on this shortlist.

If the shortlist is 8 runs and average budget is 24h:

```text
8 runs * 24h * 3 seeds = 576 pretraining hours
```

That is another **24 days on one H200**, before eval. This is why the first one-seed campaign should prune aggressively.

## Reporting Recommendations

For each result row, record:

- model name
- architecture family
- hidden size
- physical layers
- effective block applications
- parameter count
- GPU
- wall-clock budget
- final step
- estimated tokens
- final pretrain loss
- tokens/sec
- VRAM allocated/reserved
- kWh
- GLUE average
- per-task GLUE metrics

For comparisons, use separate tables:

1. **RTX 5090 anchor runs**
2. **H200 fixed-recipe replication**
3. **H200 token-matched comparison**
4. **H200 convergence/scaling comparison**
5. **Component ablations**

This avoids the most common paper mistake: mixing "better hardware," "more tokens," "larger batch," and "better architecture" into one number.

## Main Risk

The biggest risk is global batch. Increasing `impl.microbatch_size` mostly improves utilization, but increasing `train.batch_size` changes optimization. If larger global batch hurts downstream quality, keep global batch at 2048 or 4096 and use the biggest microbatch that still divides or fits under it.

Rule of thumb:

```text
First maximize tokens/sec with microbatch.
Then only increase global batch if early loss and downstream do not degrade.
```

## Source Notes

- H200 specs from NVIDIA H200 GPU page: <https://www.nvidia.com/en-us/data-center/h200/>
- RTX 5090 specs from NVIDIA GeForce RTX 5090 page: <https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/>
- Tiny8x runtime numbers from local logs and tables in `outputs/rr_me_train_final_tiny8x` and `tables/table_pretrain_reports.csv`.
