# H200 Ablation Plan

## Goal

The original tiny8x run was efficient on an RTX 5090, but it only used about 18.5 GB of VRAM. An H200 has about 141 GB, so the H200 run plan should answer two different questions:

1. Fixed-recipe science: does the architecture comparison still hold when the training recipe is unchanged?
2. Hardware-utilized science: what happens when the H200 is allowed to use larger microbatches, larger global batches, and longer budgets for larger models?

Keep these as separate tables in the paper.

## Script

Use:

```bash
bash scripts/rr_paper_h200_ablations.sh print-pretrain core
bash scripts/rr_paper_h200_ablations.sh pretrain core
bash scripts/rr_paper_h200_ablations.sh eval core
```

The script defaults to:

- `GPU_PROFILE=h200`
- `PREFIX=rr_h200`
- `train=rr-me-onecycle`
- `data=pile-readymade`
- full GLUE evaluation with 4 epochs and `8e-5`

Use `print-pretrain` first. It prints comments with resolved profile, size class, budget, global batch, and microbatch.

## H200 Defaults

Current default profile:

| Size | Microbatch | Global Batch | Budget |
| --- | ---: | ---: | ---: |
| h128 | 2048 | 4096 | 8h |
| h256 | 1536 | 4096 | 12h |
| h512 | 768 | 4096 | 24h |
| h768 | 512 | 8192 | 48h |
| h1024 | 256 | 8192 | 72h |

Override any of these:

```bash
H256_MBS=1024 BUDGET_SMALL=8 bash scripts/rr_paper_h200_ablations.sh pretrain core
H1024_MBS=384 BUDGET_XL=96 bash scripts/rr_paper_h200_ablations.sh pretrain sizes
TRAIN_BATCH=2048 TRAIN_MBS=512 BUDGET=8 bash scripts/rr_paper_h200_ablations.sh pretrain core
```

## Fixed Recipe Mode

For a fair comparison to the RTX 5090 run, use:

```bash
FIXED_RECIPE=1 PREFIX=rr_fixed bash scripts/rr_paper_h200_ablations.sh pretrain core
```

This keeps the small-run recipe close to the original: global batch 2048 and h256 microbatch 256. This will underuse H200 memory, but it is cleaner for architecture ablations.

## Utilization Sweep

Before large runs, run:

```bash
bash scripts/rr_paper_h200_ablations.sh pretrain utilization
```

This tests the same h256 Recursive Refiner with progressively larger microbatch/global-batch settings:

- batch 2048, microbatch 256
- batch 4096, microbatch 512
- batch 4096, microbatch 1024
- batch 8192, microbatch 1024
- batch 8192, microbatch 1536

Pick the setting with the best tokens/sec that does not hurt early loss too much. Then use that as the H200 recipe.

## Budgeting Larger Models

Larger models can converge later, so use two reporting modes:

1. Same wallclock budget: fair hardware-cost comparison.
2. Longer convergence budget: fair quality/scaling comparison.

Recommended first-pass budgets:

- h128: 8h
- h256: 8-12h
- h512: 24h
- h768: 48h
- h1024: 72h

Promote only the best h512/h768/h1024 variants to longer runs. For a final scaling plot, rerun the shortlist with 3 seeds and record params, final loss, tokens, tok/sec, VRAM, kWh, and GLUE.

## What To Watch

Efficient H200 use is not just high VRAM allocation. Look for:

- tokens/sec increases as microbatch grows
- no severe early-loss regression at the same token count
- stable `torch.compile` behavior
- no excessive global-batch degradation
- VRAM headroom left for larger sizes

If tokens/sec stops improving when microbatch increases, keep the smaller microbatch. If loss worsens after increasing global batch, keep global batch fixed and only increase microbatch up to the accumulation limit.
