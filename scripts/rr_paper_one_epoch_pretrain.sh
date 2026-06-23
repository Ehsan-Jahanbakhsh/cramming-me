#!/usr/bin/env bash
set -euo pipefail

# One-pass pretraining runs for Recursive Refiner paper ablations.
#
# This script is for token/dataset-pass matching, not wall-clock-budget matching.
# It computes train.steps from the readymade Pile row count:
#
#   train.steps = floor(PILE_ROWS / (impl.microbatch_size * NPROC_PER_NODE))
#
# and switches to a step-based scheduler so the large budget is only a safety
# ceiling, not part of the learning-rate schedule.
#
# Usage:
#   bash scripts/rr_paper_one_epoch_pretrain.sh print all
#   bash scripts/rr_paper_one_epoch_pretrain.sh pretrain core
#   bash scripts/rr_paper_one_epoch_pretrain.sh dryrun core
#
# Useful overrides:
#   GPU_PROFILE=h200 bash scripts/rr_paper_one_epoch_pretrain.sh pretrain all
#   NPROC_PER_NODE=8 GPU_PROFILE=h200 bash scripts/rr_paper_one_epoch_pretrain.sh pretrain sizes
#   PILE_ROWS=85000000 TRAIN_MBS=256 bash scripts/rr_paper_one_epoch_pretrain.sh print core

ACTION="${1:-print}"
GROUP="${2:-all}"

GPU_PROFILE="${GPU_PROFILE:-5090}" # 5090, h200, custom
PREFIX="${PREFIX:-rr_1ep}"
SEED="${SEED:-1975620753}"

PILE_ROWS="${PILE_ROWS:-85000000}"
SEQ_LENGTH="${SEQ_LENGTH:-128}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Keep this high so steps, not budget, stops the run.
SAFETY_BUDGET="${SAFETY_BUDGET:-9999}"

# Step-based scheduler: budget-* schedulers would still depend on wall-clock.
SCHEDULER="${SCHEDULER:-one-cycle}"
BATCH_SIZE_RAMP="${BATCH_SIZE_RAMP:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
COOLDOWN_STEPS="${COOLDOWN_STEPS:-0}"

COMPILE_TORCH="${COMPILE_TORCH:-True}"
MIXED_PRECISION_TARGET_DTYPE="${MIXED_PRECISION_TARGET_DTYPE:-float16}"
TRAIN_BATCH="${TRAIN_BATCH:-}"
TRAIN_MBS="${TRAIN_MBS:-}"

EXECUTE="false"
DRYRUN="False"

case "$ACTION" in
  print) ;;
  pretrain) EXECUTE="true" ;;
  dryrun) EXECUTE="true"; DRYRUN="True" ;;
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

round_batch_to_effective_mbs() {
  local batch="$1"
  local effective_mbs="$2"
  if (( batch < effective_mbs )); then
    echo "$effective_mbs"
  elif (( batch % effective_mbs != 0 )); then
    echo $(( ((batch + effective_mbs - 1) / effective_mbs) * effective_mbs ))
  else
    echo "$batch"
  fi
}

steps_for_one_epoch() {
  local mbs="$1"
  local nproc="$2"
  local effective_mbs=$(( mbs * nproc ))
  local steps=$(( PILE_ROWS / effective_mbs ))
  if (( steps < 1 )); then
    echo 1
  else
    echo "$steps"
  fi
}

token_positions_for_steps() {
  local steps="$1"
  local mbs="$2"
  local nproc="$3"
  echo $(( steps * mbs * nproc * SEQ_LENGTH ))
}

base_launcher() {
  if (( NPROC_PER_NODE > 1 )); then
    echo torchrun --nproc_per_node="$NPROC_PER_NODE" --standalone pretrain.py
  else
    echo python pretrain.py
  fi
}

pretrain_cmd() {
  local name="$1"
  local suffix="$2"
  shift 2
  local size
  local mbs
  local batch
  local effective_mbs
  local steps
  local token_positions

  size="$(size_class_for_suffix "$suffix")"
  mbs="$(mbs_for_size "$size")"
  batch="$(batch_for_size "$size")"
  effective_mbs=$(( mbs * NPROC_PER_NODE ))
  batch="$(round_batch_to_effective_mbs "$batch" "$effective_mbs")"
  steps="$(steps_for_one_epoch "$mbs" "$NPROC_PER_NODE")"
  token_positions="$(token_positions_for_steps "$steps" "$mbs" "$NPROC_PER_NODE")"

  comment "name=${name} profile=${GPU_PROFILE} size=${size} rows=${PILE_ROWS} seq=${SEQ_LENGTH} nproc=${NPROC_PER_NODE}"
  comment "microbatch_per_gpu=${mbs} effective_microbatch=${effective_mbs} global_batch=${batch} steps=${steps} token_positions=${token_positions}"
  comment "scheduler=${SCHEDULER} batch_size_ramp=${BATCH_SIZE_RAMP} safety_budget=${SAFETY_BUDGET}h"

  # shellcheck disable=SC2207
  local launcher=( $(base_launcher) )
  run_cmd \
    "${launcher[@]}" \
    name="$name" \
    seed="$SEED" \
    data=pile-readymade \
    train=rr-me-onecycle \
    budget="$SAFETY_BUDGET" \
    dryrun="$DRYRUN" \
    train.steps="$steps" \
    train.scheduler="$SCHEDULER" \
    train.batch_size="$batch" \
    train.batch_size_ramp="$BATCH_SIZE_RAMP" \
    train.warmup_steps="$WARMUP_STEPS" \
    train.cooldown_steps="$COOLDOWN_STEPS" \
    impl.microbatch_size="$mbs" \
    impl.compile_torch="$COMPILE_TORCH" \
    impl.mixed_precision_target_dtype="$MIXED_PRECISION_TARGET_DTYPE" \
    "wandb.tags=[rr-paper,one-epoch,pretrain]" \
    "$@"
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
  pretrain_cmd "${PREFIX}_rr_${suffix}" "$suffix" \
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
  pretrain_cmd "${PREFIX}_hfbert_${suffix}" "$suffix" \
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
  pretrain_cmd "${PREFIX}_crammed_${suffix}" "$suffix" \
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
  pretrain_cmd "${PREFIX}_albert_${suffix}" "$suffix" \
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

group_sizes_without_h256() {
  rr h128_l2_c2x3_ef4 128 2 2 2 3 4 4.0
  rr h512_l2_c2x3_ef4 512 8 2 2 3 4 4.0
  rr h768_l2_c2x3_ef4 768 12 2 2 3 4 4.0
  rr h1024_l2_c2x3_ef4 1024 16 2 2 3 4 4.0
}

group_utilization() {
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
    group_sizes_without_h256
    group_utilization
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac
