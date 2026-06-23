#!/usr/bin/env bash
set -euo pipefail

# H200-aware Recursive Refiner ablations.
#
# This is a hardware-utilization variant of rr_paper_ablations.sh. It keeps the
# same experiment families, but chooses microbatch/global batch/budget by model
# size and GPU profile.
#
# Usage:
#   bash scripts/rr_paper_h200_ablations.sh print-pretrain core
#   bash scripts/rr_paper_h200_ablations.sh pretrain core
#   bash scripts/rr_paper_h200_ablations.sh print-eval core
#   bash scripts/rr_paper_h200_ablations.sh eval core
#
# Useful overrides:
#   GPU_PROFILE=h200 PREFIX=rr_h200 bash scripts/rr_paper_h200_ablations.sh pretrain sizes
#   BUDGET_LARGE=72 H1024_MBS=256 bash scripts/rr_paper_h200_ablations.sh pretrain sizes
#   FIXED_RECIPE=1 bash scripts/rr_paper_h200_ablations.sh pretrain core

ACTION="${1:-print-pretrain}"
GROUP="${2:-core}"

GPU_PROFILE="${GPU_PROFILE:-h200}" # h200, 5090, custom
PREFIX="${PREFIX:-rr_h200}"
SEED="${SEED:-1975620753}"

# Set FIXED_RECIPE=1 for strict recipe comparability with the successful RTX 5090
# tiny8x run. Leave unset/0 for H200 utilization.
FIXED_RECIPE="${FIXED_RECIPE:-0}"

EVAL_MBS="${EVAL_MBS:-32}"
EVAL_CFG="${EVAL_CFG:-GLUE}"
EVAL_EPOCHS="${EVAL_EPOCHS:-4}"
EVAL_BATCH="${EVAL_BATCH:-16}"
EVAL_LR="${EVAL_LR:-8e-5}"
COMPILE_TORCH="${COMPILE_TORCH:-True}"
MIXED_PRECISION_TARGET_DTYPE="${MIXED_PRECISION_TARGET_DTYPE:-float16}"

# Optional global defaults. Per-size defaults are chosen below if these are unset.
TRAIN_BATCH="${TRAIN_BATCH:-}"
TRAIN_MBS="${TRAIN_MBS:-}"
BUDGET="${BUDGET:-}"

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

case "$GPU_PROFILE" in
  5090)
    DEFAULT_BATCH="${DEFAULT_BATCH:-2048}"
    DEFAULT_MBS="${DEFAULT_MBS:-256}"
    H128_MBS="${H128_MBS:-256}"
    H256_MBS="${H256_MBS:-256}"
    H512_MBS="${H512_MBS:-128}"
    H768_MBS="${H768_MBS:-128}"
    H1024_MBS="${H1024_MBS:-64}"
    ;;
  h200)
    DEFAULT_BATCH="${DEFAULT_BATCH:-4096}"
    DEFAULT_MBS="${DEFAULT_MBS:-1024}"
    H128_MBS="${H128_MBS:-2048}"
    H256_MBS="${H256_MBS:-1536}"
    H512_MBS="${H512_MBS:-768}"
    H768_MBS="${H768_MBS:-512}"
    H1024_MBS="${H1024_MBS:-256}"
    ;;
  custom)
    DEFAULT_BATCH="${DEFAULT_BATCH:-2048}"
    DEFAULT_MBS="${DEFAULT_MBS:-256}"
    H128_MBS="${H128_MBS:-$DEFAULT_MBS}"
    H256_MBS="${H256_MBS:-$DEFAULT_MBS}"
    H512_MBS="${H512_MBS:-$DEFAULT_MBS}"
    H768_MBS="${H768_MBS:-$DEFAULT_MBS}"
    H1024_MBS="${H1024_MBS:-$DEFAULT_MBS}"
    ;;
  *)
    echo "Unknown GPU_PROFILE: $GPU_PROFILE" >&2
    exit 2
    ;;
esac

if [[ "$FIXED_RECIPE" == "1" ]]; then
  DEFAULT_BATCH=2048
  DEFAULT_MBS=256
  H128_MBS=256
  H256_MBS=256
  H512_MBS=256
  H768_MBS=128
  H1024_MBS=64
fi

BUDGET_TINY="${BUDGET_TINY:-8}"
BUDGET_SMALL="${BUDGET_SMALL:-12}"
BUDGET_MEDIUM="${BUDGET_MEDIUM:-24}"
BUDGET_LARGE="${BUDGET_LARGE:-48}"
BUDGET_XL="${BUDGET_XL:-72}"

run_cmd() {
  if [[ "$EXECUTE" == "true" ]]; then
    "$@"
  else
    printf '%q ' "$@"
    printf '\n'
  fi
}

comment() {
  if [[ "$EXECUTE" != "true" ]]; then
    printf '# %s\n' "$*"
  fi
}

size_class_for_suffix() {
  local suffix="$1"
  case "$suffix" in
    *h128*) echo "h128" ;;
    *h256*|tiny8x*) echo "h256" ;;
    *h512*) echo "h512" ;;
    *h768*) echo "h768" ;;
    *h1024*) echo "h1024" ;;
    *) echo "default" ;;
  esac
}

budget_for_size() {
  local size="$1"
  if [[ -n "$BUDGET" ]]; then
    echo "$BUDGET"
    return
  fi
  case "$size" in
    h128) echo "$BUDGET_TINY" ;;
    h256|default) echo "$BUDGET_SMALL" ;;
    h512) echo "$BUDGET_MEDIUM" ;;
    h768) echo "$BUDGET_LARGE" ;;
    h1024) echo "$BUDGET_XL" ;;
    *) echo "$BUDGET_SMALL" ;;
  esac
}

mbs_for_size() {
  local size="$1"
  if [[ -n "$TRAIN_MBS" ]]; then
    echo "$TRAIN_MBS"
    return
  fi
  case "$size" in
    h128) echo "$H128_MBS" ;;
    h256|default) echo "$H256_MBS" ;;
    h512) echo "$H512_MBS" ;;
    h768) echo "$H768_MBS" ;;
    h1024) echo "$H1024_MBS" ;;
    *) echo "$DEFAULT_MBS" ;;
  esac
}

batch_for_size() {
  local size="$1"
  local mbs="$2"
  if [[ -n "$TRAIN_BATCH" ]]; then
    echo "$TRAIN_BATCH"
    return
  fi
  case "$size" in
    h768|h1024) echo 8192 ;;
    h512) echo 4096 ;;
    *) echo "$DEFAULT_BATCH" ;;
  esac
}

pretrain_cmd() {
  local name="$1"
  local size="$2"
  shift 2
  local budget
  local mbs
  local batch
  budget="$(budget_for_size "$size")"
  mbs="$(mbs_for_size "$size")"
  batch="$(batch_for_size "$size" "$mbs")"

  if (( mbs > batch )); then
    comment "clamping microbatch from ${mbs} to batch ${batch} for ${name}"
    mbs="$batch"
  fi

  comment "profile=${GPU_PROFILE} fixed_recipe=${FIXED_RECIPE} name=${name} size=${size} budget=${budget}h batch=${batch} microbatch=${mbs}"
  run_cmd \
    python pretrain.py \
    name="$name" \
    seed="$SEED" \
    data=pile-readymade \
    train=rr-me-onecycle \
    budget="$budget" \
    train.batch_size="$batch" \
    dryrun="$DRYRUN" \
    impl.microbatch_size="$mbs" \
    impl.compile_torch="$COMPILE_TORCH" \
    impl.mixed_precision_target_dtype="$MIXED_PRECISION_TARGET_DTYPE" \
    "wandb.tags=[rr-paper,h200,pretrain]" \
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
    impl.mixed_precision_target_dtype="$MIXED_PRECISION_TARGET_DTYPE" \
    "wandb.tags=[rr-paper,h200,eval]" \
    "$@"
}

emit() {
  local name="$1"
  local suffix="$2"
  shift 2
  local size
  size="$(size_class_for_suffix "$suffix")"
  if [[ "$PHASE" == "pretrain" ]]; then
    pretrain_cmd "$name" "$size" "$@"
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
  emit "${PREFIX}_rr_${suffix}" "$suffix" \
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
  emit "${PREFIX}_hfbert_${suffix}" "$suffix" \
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
  emit "${PREFIX}_crammed_${suffix}" "$suffix" \
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
  emit "${PREFIX}_albert_${suffix}" "$suffix" \
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
  rr tiny8x_h256_l2_c2x3_ef4 256 4 2 2 3 4 4.0
  albert_shared h256_eff16_e64 256 4 16 1024 64
  hfbert h256_l2 256 4 2 1024
  crammed h256_l2 256 4 2 1024
  hfbert h128_l2_param_match 128 2 2 512
}

group_baselines() {
  group_core
  hfbert h256_l16_effective_depth 256 4 16 1024
  crammed h256_l16_effective_depth 256 4 16 1024
  albert_shared h256_eff16_e256 256 4 16 1024 256
  albert_shared h256_eff8_e64 256 4 8 1024 64
}

group_components() {
  rr h256_l2_c2x3_ef1 256 4 2 2 3 1 4.0
  rr h256_l2_c2x3_ef2 256 4 2 2 3 2 4.0
  rr h256_l2_c2x3_ef4 256 4 2 2 3 4 4.0
  rr h256_l2_c2x3_ef8 256 4 2 2 3 8 4.0
  rr h256_l2_c1x1_ef4 256 4 2 1 1 4 4.0
  rr h256_l2_c1x3_ef4 256 4 2 1 3 4 4.0
  rr h256_l2_c2x1_ef4 256 4 2 2 1 4 4.0
  rr h256_l2_c4x1_ef4 256 4 2 4 1 4 4.0
  rr h256_l2_c1x7_ef4 256 4 2 1 7 4 4.0
  rr h256_l2_c3x2_ef4 256 4 2 3 2 4 4.0
  rr h256_l1_c2x3_ef4 256 4 1 2 3 4 4.0
  rr h256_l4_c2x3_ef4 256 4 4 2 3 4 4.0
  rr h256_l2_c2x3_ef4_exp2 256 4 2 2 3 4 2.0
  rr h256_l2_c2x3_ef4_exp6 256 4 2 2 3 4 6.0
  rr h256_l2_c2x3_ef4_postnorm 256 4 2 2 3 4 4.0 arch.pre_norm=False
}

group_sizes() {
  rr h128_l2_c2x3_ef4 128 2 2 2 3 4 4.0
  rr h256_l2_c2x3_ef4 256 4 2 2 3 4 4.0
  rr h512_l2_c2x3_ef4 512 8 2 2 3 4 4.0
  rr h768_l2_c2x3_ef4 768 12 2 2 3 4 4.0
  rr h1024_l2_c2x3_ef4 1024 16 2 2 3 4 4.0
}

group_utilization() {
  # Same architecture, progressively larger microbatch/global batch choices.
  TRAIN_BATCH=2048; TRAIN_MBS=256; rr h256_util_b2048_m256 256 4 2 2 3 4 4.0
  TRAIN_BATCH=4096; TRAIN_MBS=512; rr h256_util_b4096_m512 256 4 2 2 3 4 4.0
  TRAIN_BATCH=4096; TRAIN_MBS=1024; rr h256_util_b4096_m1024 256 4 2 2 3 4 4.0
  TRAIN_BATCH=8192; TRAIN_MBS=1024; rr h256_util_b8192_m1024 256 4 2 2 3 4 4.0
  TRAIN_BATCH=8192; TRAIN_MBS=1536; rr h256_util_b8192_m1536 256 4 2 2 3 4 4.0
  TRAIN_BATCH=""; TRAIN_MBS=""
}

case "$GROUP" in
  core) group_core ;;
  baselines) group_baselines ;;
  components) group_components ;;
  sizes) group_sizes ;;
  utilization) group_utilization ;;
  all)
    group_baselines
    group_components
    group_sizes
    group_utilization
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac
