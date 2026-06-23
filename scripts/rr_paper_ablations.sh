#!/usr/bin/env bash
set -euo pipefail

# Paper ablations for Recursive Refiner.
#
# Usage:
#   bash scripts/rr_paper_ablations.sh print-pretrain core
#   bash scripts/rr_paper_ablations.sh pretrain core
#   bash scripts/rr_paper_ablations.sh print-eval core
#   bash scripts/rr_paper_ablations.sh eval core
#   bash scripts/rr_paper_ablations.sh dryrun-pretrain core
#
# Groups:
#   core        - smallest defensible table around the successful tiny8x model
#   baselines   - BERT/CrammedBERT/ALBERT comparisons at matched dimensions/depths
#   components  - RR component ablations: embedding rank, cycles, depth, FFN, norm
#   sizes       - RR width scaling sweep
#   all         - core + baselines + components + sizes

ACTION="${1:-print-pretrain}"
GROUP="${2:-core}"

PREFIX="${PREFIX:-rr_paper}"
SEED="${SEED:-1975620753}"
BUDGET="${BUDGET:-8}"
TRAIN_MBS="${TRAIN_MBS:-256}"
EVAL_MBS="${EVAL_MBS:-16}"
EVAL_CFG="${EVAL_CFG:-GLUE}"
EVAL_EPOCHS="${EVAL_EPOCHS:-4}"
EVAL_BATCH="${EVAL_BATCH:-16}"
EVAL_LR="${EVAL_LR:-8e-5}"
COMPILE_TORCH="${COMPILE_TORCH:-True}"

PHASE=""
EXECUTE="false"
DRYRUN="False"

case "$ACTION" in
  print-pretrain) PHASE="pretrain" ;;
  print-eval) PHASE="eval" ;;
  pretrain) PHASE="pretrain"; EXECUTE="true" ;;
  eval) PHASE="eval"; EXECUTE="true" ;;
  dryrun-pretrain) PHASE="pretrain"; EXECUTE="true"; DRYRUN="True" ;;
  dryrun-eval) PHASE="eval"; EXECUTE="true"; DRYRUN="True" ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac

run_cmd() {
  if [[ "$EXECUTE" == "true" ]]; then
    "$@"
  else
    printf '%q ' "$@"
    printf '\n'
  fi
}

pretrain_cmd() {
  local name="$1"
  shift
  run_cmd \
    python pretrain.py \
    name="$name" \
    seed="$SEED" \
    data=pile-readymade \
    train=rr-me-onecycle \
    budget="$BUDGET" \
    dryrun="$DRYRUN" \
    impl.microbatch_size="$TRAIN_MBS" \
    impl.compile_torch="$COMPILE_TORCH" \
    "wandb.tags=[rr-paper,pretrain]" \
    "$@"
}

eval_cmd() {
  local name="$1"
  shift
  run_cmd \
    python eval.py \
    name="$name" \
    seed="$SEED" \
    eval="$EVAL_CFG" \
    eval.checkpoint=latest \
    eval.epochs="$EVAL_EPOCHS" \
    eval.batch_size="$EVAL_BATCH" \
    eval.optim.lr="$EVAL_LR" \
    dryrun="$DRYRUN" \
    impl.microbatch_size="$EVAL_MBS" \
    impl.shuffle_in_dataloader=True \
    impl.compile_torch=False \
    "wandb.tags=[rr-paper,eval]" \
    "$@"
}

emit() {
  local name="$1"
  shift
  if [[ "$PHASE" == "pretrain" ]]; then
    pretrain_cmd "$name" "$@"
  else
    eval_cmd "$name"
  fi
}

rr() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local hi="$5"
  local lo="$6"
  local embed_factor="$7"
  local expansion="$8"
  shift 8
  emit "${PREFIX}_rr_${suffix}" \
    arch=recursive-refiner-tiny \
    arch.hidden_size="$hidden" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$layers" \
    arch.hi_cycles="$hi" \
    arch.lo_cycles="$lo" \
    arch.embed_factor="$embed_factor" \
    arch.expansion="$expansion" \
    "$@"
}

hfbert() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local intermediate="$5"
  shift 5
  emit "${PREFIX}_hfbert_${suffix}" \
    arch=hf-bert-tiny \
    arch.hidden_size="$hidden" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$layers" \
    arch.intermediate_size="$intermediate" \
    "$@"
}

crammed() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local layers="$4"
  local intermediate="$5"
  shift 5
  emit "${PREFIX}_crammed_${suffix}" \
    arch=crammed-bert \
    arch.hidden_size="$hidden" \
    arch.num_transformer_layers="$layers" \
    arch.intermed_size="$intermediate" \
    arch.attention.num_attention_heads="$heads" \
    arch.embedding.embedding_dim="$hidden" \
    arch.classification_head.head_dim="$hidden" \
    "$@"
}

albert_shared() {
  local suffix="$1"
  local hidden="$2"
  local heads="$3"
  local effective_layers="$4"
  local intermediate="$5"
  local embedding="$6"
  shift 6
  emit "${PREFIX}_albert_${suffix}" \
    arch=hf-albert-shared \
    arch.hidden_size="$hidden" \
    arch.embedding_size="$embedding" \
    arch.num_attention_heads="$heads" \
    arch.num_hidden_layers="$effective_layers" \
    arch.intermediate_size="$intermediate" \
    arch.num_hidden_groups=1 \
    arch.inner_group_num=1 \
    "$@"
}

group_core() {
  # Core uses full-rank embeddings; embedding-factorization sweeps live in components.
  rr tiny8x_h256_l2_c2x3_ef1 256 4 2 2 3 1 4.0

  # Same dimensions as tiny8x, but no nested RR state/cycles.
  albert_shared h256_eff16_e256 256 4 16 1024 256

  # Same hidden width and physical depth.
  hfbert h256_l2 256 4 2 1024
  crammed h256_l2 256 4 2 1024

  # Smaller full-embedding BERT baseline.
  hfbert h128_l2_param_match 128 2 2 512
}

group_baselines() {
  group_core

  # Match RR's effective Transformer-block applications: 2 layers * 2 hi * (3 lo + 1 hi) = 16.
  hfbert h256_l16_effective_depth 256 4 16 1024 impl.microbatch_size=128
  crammed h256_l16_effective_depth 256 4 16 1024 impl.microbatch_size=128

  # Shared-weight controls at the same effective depth with ALBERT factorized embeddings.
  albert_shared h256_eff8_e64 256 4 8 1024 64
}

group_components() {
  # Low-rank/tied embedding ablation.
  rr h256_l2_c2x3_ef1 256 4 2 2 3 1 4.0
  rr h256_l2_c2x3_ef2 256 4 2 2 3 2 4.0
  rr h256_l2_c2x3_ef4 256 4 2 2 3 4 4.0
  rr h256_l2_c2x3_ef8 256 4 2 2 3 8 4.0

  # Recurrence schedule at roughly fixed physical parameters.
  rr h256_l2_c1x1_ef4 256 4 2 1 1 4 4.0
  rr h256_l2_c1x3_ef4 256 4 2 1 3 4 4.0
  rr h256_l2_c2x1_ef4 256 4 2 2 1 4 4.0
  rr h256_l2_c4x1_ef4 256 4 2 4 1 4 4.0
  rr h256_l2_c1x7_ef4 256 4 2 1 7 4 4.0
  rr h256_l2_c3x2_ef4 256 4 2 3 2 4 4.0

  # Physical block count while preserving the successful 2x3 schedule.
  rr h256_l1_c2x3_ef4 256 4 1 2 3 4 4.0
  rr h256_l4_c2x3_ef4 256 4 4 2 3 4 4.0 impl.microbatch_size=128

  # FFN and normalization choices.
  rr h256_l2_c2x3_ef4_exp2 256 4 2 2 3 4 2.0
  rr h256_l2_c2x3_ef4_exp6 256 4 2 2 3 4 6.0
  rr h256_l2_c2x3_ef4_postnorm 256 4 2 2 3 4 4.0 arch.pre_norm=False
}

group_sizes() {
  rr h128_l2_c2x3_ef4 128 2 2 2 3 4 4.0
  rr h256_l2_c2x3_ef4 256 4 2 2 3 4 4.0
  rr h512_l2_c2x3_ef4 512 8 2 2 3 4 4.0 impl.microbatch_size=128
  rr h768_l2_c2x3_ef4 768 12 2 2 3 4 4.0 impl.microbatch_size=128
  rr h1024_l2_c2x3_ef4 1024 16 2 2 3 4 4.0 impl.microbatch_size=64
}

case "$GROUP" in
  core) group_core ;;
  baselines) group_baselines ;;
  components) group_components ;;
  sizes) group_sizes ;;
  all)
    group_baselines
    group_components
    group_sizes
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac
