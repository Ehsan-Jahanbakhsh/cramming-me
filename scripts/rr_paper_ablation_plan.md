# Recursive Refiner Paper Ablation Plan

## Current Anchor

The strongest small run provided by the user is:

- Run: `rr_me_train_final_tiny8x`
- Config: `RecursiveRefinerForMaskedLM`, hidden 256, 4 heads, 2 shared blocks, `hi_cycles=2`, `lo_cycles=3`, `embed_factor=4`
- Effective shared-stack calls: `hi_cycles * (lo_cycles + 1) = 8`
- Effective Transformer-block applications: `8 * num_hidden_layers = 16`
- Parameters: 4,222,976
- Pretrain: 8 hours on `pile-readymade`, loss 2.8375
- Downstream: full GLUE average 0.7547

Nearby existing results:

- `rr_me_train_final_tiny`: hidden 256, `embed_factor=1`, 10.56M params, 12h, GLUE 0.7452
- `rr_me_train_final`: hidden 768, `embed_factor=1`, 44.65M params, 36h, GLUE 0.7992
- `rr_me_train_final_large`: hidden 1024, `embed_factor=4`, Muon run; downstream logs are partial, so do not compare its reported average directly to full GLUE.

## Main Claims To Test

1. Recursive Refiner gets unusually strong downstream quality per parameter at very small scale.
2. The benefit is not just parameter sharing: an ALBERT-style shared Transformer without high/low refinement cycles should be a necessary baseline.
3. Low-rank tied embeddings are a real component, not just a compression footnote.
4. The high/low cycle schedule matters beyond total repeated block applications.
5. The result scales with width under the same recipe.

## Run Order

1. Smoke test:
   `bash scripts/rr_paper_ablations.sh dryrun-pretrain core`

2. Core table:
   `bash scripts/rr_paper_ablations.sh pretrain core`
   `bash scripts/rr_paper_ablations.sh eval core`

3. Baselines:
   `bash scripts/rr_paper_ablations.sh pretrain baselines`
   `bash scripts/rr_paper_ablations.sh eval baselines`

4. Component ablations:
   `bash scripts/rr_paper_ablations.sh pretrain components`
   `bash scripts/rr_paper_ablations.sh eval components`

5. Scaling:
   `bash scripts/rr_paper_ablations.sh pretrain sizes`
   `bash scripts/rr_paper_ablations.sh eval sizes`

For paper numbers, rerun the final shortlist with at least three seeds by setting `SEED=...` and preferably a name prefix such as `PREFIX=rr_paper_s2`.

## Fairness Notes

The script defaults to the successful small-run recipe: `pile-readymade`, `train=rr-me-onecycle`, `budget=8`, `batch_size=2048`, `microbatch_size=256`, MLM probability 0.15, and full GLUE evaluation for 4 epochs at `8e-5`.

The baseline groups include:

- Dimension-matched BERT/CrammedBERT: same hidden width and physical layer count.
- Effective-depth-matched BERT/CrammedBERT: 16 ordinary Transformer layers to match the tiny8x repeated block applications.
- ALBERT-style shared baseline: 16 repeated layers with one shared group and factorized embeddings, but no recursive high/low latent state.
- Approximate parameter-matched full-embedding BERT: hidden 128, 2 layers.

When reporting, separate `dimension matched`, `effective-depth matched`, and `parameter matched`; otherwise the comparison will be easy to misread.
